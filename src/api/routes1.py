# src/api/routes.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from io import BytesIO
import os
import re
import pandas as pd

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

from src.ingestion.loader import load_paths
from src.ingestion.splitter import split_documents
from src.embedding.embedder import embed_documents, embed_query
from src.embedding.vector_store import FaissStore
from src.llm.hf_runner import HuggingFaceLLM
from src.llm.prompt_builder import build_prompt

router = APIRouter()

CFG = load_config()
LOG = setup_logger("careerconnect-api")

# Global FAISS store + LLM instance
STORE = FaissStore(
    CFG["vector_store"]["persist_directory"],
    CFG["vector_store"]["index_dim"],
)
STORE.load()
LLM = HuggingFaceLLM(
    model_id=CFG["llm"]["model_id"],
    max_new_tokens=CFG["llm"]["max_tokens"],
)


# ---------- Models ----------

class ChatQuery(BaseModel):
    query: str
    top_k: int = 5


# ---------- Utility ----------

def _ensure_index_loaded():
    """Raise HTTP 500 if index is empty."""
    if STORE.size() == 0:
        raise HTTPException(
            status_code=500,
            detail="Vector index is empty. Run /ingest or the embed step first.",
        )


# ---------- Basic health ----------

@router.get("/health")
def health():
    return {
        "status": "ok",
        "app": "careerconnect-backend",
        "index_size": STORE.size(),
    }


# ---------- Ingestion (build FAISS index) ----------

@router.post("/ingest")
def ingest():
    """
    (Re)build the FAISS index from files under data/raw.

    - CSVs: either 'text' column or concatenation of all columns per row.
    - TXT/MD: read full text.
    - PDF: extract text with pypdf.
    """
    root = Path("data/raw")
    paths = load_paths(root)
    if not paths:
        raise HTTPException(status_code=400, detail="No files found in data/raw")

    LOG.info(f"Starting ingestion over {len(paths)} files...")

    # Streaming build: add batches of chunks to FAISS without holding everything in RAM
    BATCH = int(os.getenv("INGEST_BATCH", "2048"))
    texts_batch, metas_batch = [], []

    STORE.start()  # reset index + open temp meta file

    def _flush():
        nonlocal texts_batch, metas_batch
        if not texts_batch:
            return
        vecs = embed_documents(texts_batch)
        STORE.add_batch(vecs, metas_batch)
        texts_batch, metas_batch = [], []

    for p in paths:
        LOG.info(f"Ingesting {p} ...")
        chunks = []

        suffix = p.suffix.lower()
        try:
            if suffix == ".csv":
                df = pd.read_csv(p, low_memory=False)
                # If there's a 'text' column, use it; otherwise concat all columns
                if "text" in df.columns:
                    rows = df["text"].astype(str).fillna("").tolist()
                else:
                    rows = df.astype(str).fillna("").agg(" ".join, axis=1).tolist()
                chunks = split_documents(rows, CFG["chunk_size"], CFG["chunk_overlap"])

            elif suffix in {".txt", ".md"}:
                text = p.read_text(encoding="utf-8", errors="ignore")
                chunks = split_documents([text], CFG["chunk_size"], CFG["chunk_overlap"])

            elif suffix == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(p))
                text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
                chunks = split_documents([text], CFG["chunk_size"], CFG["chunk_overlap"])

            else:
                LOG.warning(f"Skipping unsupported file type: {p}")
                continue

        except Exception as e:
            LOG.exception(f"Error ingesting file {p}: {e}")
            continue

        for c in chunks:
            texts_batch.append(c)
            metas_batch.append({"source": str(p), "chunk": c})
            if len(texts_batch) >= BATCH:
                _flush()

    _flush()
    STORE.finalize()

    LOG.info(f"Ingestion complete. Index size: {STORE.size()}")

    return {
        "indexed_chunks": STORE.size(),
        "files": [str(p) for p in paths],
    }


# ---------- Raw vector query (no LLM) ----------

@router.post("/query")
def query(payload: ChatQuery):
    _ensure_index_loaded()
    qv = embed_query(payload.query)
    hits = STORE.search(qv, payload.top_k)
    return {"results": hits}


# ---------- RAG chat endpoint ----------

@router.post("/chat")
def chat(payload: ChatQuery):
    _ensure_index_loaded()
    LOG.info(f"Chat query: {payload.query!r}")

    qv = embed_query(payload.query)
    hits = STORE.search(qv, payload.top_k)

    ctx_chunks = [h.get("chunk", "") for h in hits]
    prompt = build_prompt(ctx_chunks, payload.query)

    answer = LLM.generate(prompt)

    citations = [
        {"id": i + 1, "source": h.get("source", "")}
        for i, h in enumerate(hits)
    ]

    return {
        "response": answer,
        "citations": citations,
        "k": payload.top_k,
    }


# ---------- Resume-processing endpoint for Streamlit UI ----------

@router.post("/api/process_resume/")
async def process_resume(file: UploadFile = File(...)):
    """
    Accept a resume PDF, extract text, use RAG+LLM to:
      - extract skills
      - suggest jobs
      - suggest one upskilling course
      - generate a short summary

    Returns JSON matching the Streamlit UI expectations:
      - skills: list[str]
      - jobs: list[{"job_title": ...}]
      - course: {"skill": ..., "name": ..., "url": ...}
      - summary: str
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    contents = await file.read()

    # Extract text from PDF
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(contents))
    full_text = "\n".join((p.extract_text() or "") for p in reader.pages)
    resume_text = full_text.strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

    resume_snippet = resume_text[:2000]

    _ensure_index_loaded()

    # Retrieve context relevant to this resume
    qv = embed_query(f"jobs and courses relevant to this resume: {resume_snippet}")
    hits = STORE.search(qv, top_k=8)
    ctx_chunks = [h.get("chunk", "") for h in hits]

    context_block = "\n\n".join(
        f"[{i+1}] {c}" for i, c in enumerate(ctx_chunks)
    )

    # Prompt: structured answer so we can parse it
    prompt = f"""
You are helping with job and course recommendations using RAG context and a resume.

Context (job postings, skills, salaries, courses):
{context_block}

Resume:
\"\"\"{resume_snippet}\"\"\"

TASK:
1. Extract up to 15 key skills from the resume.
2. Suggest 5 job titles that fit this candidate, based on the context.
3. Suggest ONE upskilling course (or learning resource) that would be most useful, with:
   - skill name
   - course name
   - url (if present in the context; otherwise use NA).
4. Provide a 2-3 sentence career summary for this candidate.

Return the answer in EXACTLY this format:

Skills: skill1, skill2, skill3, ...

Jobs:
- job title 1
- job title 2
- job title 3
- job title 4
- job title 5

Courses:
- skill: <skill>, course: <course name>, url: <url or NA>

Summary: <your summary here>
    """.strip()

    raw_answer = LLM.generate(prompt)
    LOG.info("LLM raw resume answer:\n" + raw_answer)

    # ---- Parse model output into structured fields ----
    skills: list[str] = []
    jobs: list[str] = []
    course_skill = ""
    course_name = ""
    course_url = ""
    summary = ""

    lines = [l.rstrip() for l in raw_answer.splitlines() if l.strip()]

    # Skills line
    for line in lines:
        if line.lower().startswith("skills:"):
            _, rest = line.split(":", 1)
            skills = [s.strip() for s in rest.split(",") if s.strip()]
            break

    in_jobs = False
    in_courses = False
    for idx, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("jobs:"):
            in_jobs = True
            in_courses = False
            continue
        if lower.startswith("courses:"):
            in_jobs = False
            in_courses = True
            continue
        if lower.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
            extra = " ".join(l.strip() for l in lines[idx + 1 :])
            if extra:
                summary = (summary + " " + extra).strip()
            break

        if in_jobs and line.lstrip().startswith("-"):
            jobs.append(line.lstrip("-").strip())

        if in_courses and line.lstrip().startswith("-"):
            text = line.lstrip("-").strip()
            parts = [p.strip() for p in text.split(",") if p.strip()]
            for p in parts:
                if p.lower().startswith("skill:"):
                    course_skill = p.split(":", 1)[1].strip()
                elif p.lower().startswith("course:"):
                    course_name = p.split(":", 1)[1].strip()
                elif p.lower().startswith("url:"):
                    course_url = p.split(":", 1)[1].strip()

    if not summary:
        summary = raw_answer.strip()

    course = {}
    if course_name or course_skill or course_url:
        course = {
            "skill": course_skill or (skills[0] if skills else ""),
            "name": course_name or "Suggested course",
            "url": course_url or "NA",
        }

    job_dicts = [{"job_title": j} for j in jobs]

    return {
        "skills": skills,
        "jobs": job_dicts,
        "course": course,
        "summary": summary,
    }


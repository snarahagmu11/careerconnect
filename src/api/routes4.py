# src/api/routes.py

from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import fitz
import math

from src.embedding.embedder import embed_query
from src.utils.vectorstore import FaissVectorStore
from src.utils.config_loader import load_config
from src.llm.hf_runner import HuggingFaceLLM

router = APIRouter(prefix="/api")


# -------------------------------------------------
# SIMPLE SKILL EXTRACTION
# -------------------------------------------------
def extract_skills(text):
    keywords = [
        "python", "java", "sql", "aws", "ml", "machine learning",
        "deep learning", "docker", "linux", "azure", "tensorflow",
        "pytorch", "nlp", "rag"
    ]
    t = text.lower()
    return sorted({k for k in keywords if k in t})


# -------------------------------------------------
# MAIN ENDPOINT
# -------------------------------------------------
@router.post("/process_resume/")
async def process_resume(file: UploadFile = File(...)):
    cfg = load_config()

    # ---------- 1. Save uploaded resume ----------
    save_path = Path("data/raw/uploaded_resume.pdf")
    save_path.write_bytes(await file.read())

    # ---------- 2. Extract text ----------
    doc = fitz.open(str(save_path))
    resume_text = "\n".join(pg.get_text() for pg in doc)
    doc.close()

    extracted_skills = extract_skills(resume_text)

    # ---------- 3. Load FAISS ----------
    store = FaissVectorStore(
        dim=cfg["vector_store"]["index_dim"],
        index_dir=cfg["vector_store"]["persist_directory"],
    )
    if not store.load() or store.size() == 0:
        return {"error": "FAISS index missing or empty"}

    # --------------------------------------------------------
    # 4. JOB SEARCH  (LinkedIn + Monster + SE Salary dataset)
    # --------------------------------------------------------
    job_query_vec = embed_query(resume_text)
    job_hits = store.search(job_query_vec, top_k=60)

    job_listings = []

    for h in job_hits:
        src = h.get("source_file", "")

        # ---------- URL extraction rules ----------
        if src == "linkedin_jobs_processed.csv":
            url = h.get("url", "N/A")
        elif src == "monster_jobs_processed.csv":
            url = h.get("url", "N/A")
        else:
            url = "N/A"

        # ---------- Salary ----------
        salary = h.get("salary") or "N/A"

        # ---------- Short description ----------
        desc = h.get("description", "")
        if len(desc) > 160:
            desc = desc[:160] + "..."

        # Add row
        if src in {
            "linkedin_jobs_processed.csv",
            "monster_jobs_processed.csv",
            "se_jobs_salaries_processed.csv",
        }:
            job_listings.append({
                "Title": h.get("title", "N/A"),
                "Company": h.get("company", "N/A"),
                "Location": h.get("location", "N/A"),
                "Salary": salary,
                "Url": url,
                "Description": desc,
            })

    job_listings = job_listings[:10]   # keep only top 10

    # --------------------------------------
    # 5. COURSE SEARCH
    # --------------------------------------
    skill_text = " ".join(extracted_skills)
    course_query = f"best online courses for {skill_text}"
    course_vec = embed_query(course_query)
    course_hits = store.search(course_vec, top_k=200)

    course_list = []
    for h in course_hits:
        if h.get("source_file") == "udemy_courses_processed.csv":
            course_list.append({
                "Title": h.get("title", ""),
                "Url": h.get("url", ""),
            })

    # Deduplicate
    seen = set()
    final_courses = []
    for c in course_list:
        if c["Title"] not in seen:
            seen.add(c["Title"])
            final_courses.append(c)

    final_courses = final_courses[:5]

    # --------------------------------------
    # 6. LLM SUMMARY
    # --------------------------------------
    resume_for_llm = resume_text[:3000]

    prompt = f"""
You are an expert career analyst.

Summarize the following resume into clear sections:

1. Skills
2. Certifications
3. Experience (bullet points)
4. Job Fit Summary (2–3 sentences)

Resume:
\"\"\"{resume_for_llm}\"\"\""""

    llm = HuggingFaceLLM(cfg["llm"]["model_id"], cfg["llm"]["max_tokens"])
    summary = llm.generate(prompt)

    # --------------------------------------
    # 7. RETURN FINAL RESPONSE
    # --------------------------------------
    return {
        "skills": extracted_skills,
        "jobs": job_listings,
        "upskilling": final_courses,
        "summary": summary,
    }


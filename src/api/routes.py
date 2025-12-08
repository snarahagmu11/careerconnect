# src/api/routes.py
from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import fitz

from src.embedding.embedder import embed_query
from src.embedding.vector_store import FaissStore
from src.utils.config_loader import load_config
from src.llm.hf_runner import HuggingFaceLLM

router = APIRouter()


# ---------------------------------------------
# Simple Skill Extractor (You may upgrade later)
# ---------------------------------------------
def extract_skills(text):
    skills = [
        "python","java","sql","aws","docker","linux","ml","machine learning",
        "deep learning","pytorch","tensorflow","nlp","rag"
    ]
    return sorted({s for s in skills if s in text.lower()})


# ---------------------------------------------
# MAIN ROUTE
# ---------------------------------------------
@router.post("/api/process_resume/")
async def process_resume(file: UploadFile = File(...)):
    cfg = load_config()

    # Save uploaded file
    save_path = Path("data/raw/uploaded_resume.pdf")
    save_path.write_bytes(await file.read())

    # Extract PDF text
    doc = fitz.open(str(save_path))
    resume_text = "\n".join(pg.get_text() for pg in doc)
    doc.close()

    # Extract skills from resume
    skills = extract_skills(resume_text)

    # Load FAISS index
    store = FaissStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        index_dim=cfg["vector_store"]["index_dim"],
    )
    store.load()

    # ----------------------------------------------------------
    # JOB RETRIEVAL  (unchanged)
    # ----------------------------------------------------------
    qvec = embed_query(resume_text)
    hits = store.search(qvec, top_k=100)

    jobs = []
    for h in hits:
        if h.get("source_file") in (
            "linkedin_jobs_processed.csv",
            "monster_jobs_processed.csv",
        ):
            jobs.append({
                "title": h.get("title"),
                "company": h.get("company"),
                "location": h.get("location"),
                "salary": h.get("salary"),
                "url": h.get("url"),
                "description": h.get("description", "")[:160] + "..."
            })

    # ----------------------------------------------------------
    # UPSKILLING — Fully Improved Logic
    # ----------------------------------------------------------

    # Skill → keyword mapping for relevance scoring
    skill_to_keywords = {
        "python": ["python", "data analysis", "python programming", "machine learning"],
        "ml": ["machine learning", "ml", "data science"],
        "machine learning": ["machine learning", "deep learning", "neural network"],
        "deep learning": ["deep learning", "neural network", "pytorch", "tensorflow"],
        "pytorch": ["pytorch", "deep learning"],
        "tensorflow": ["tensorflow", "deep learning"],
        "nlp": ["nlp", "natural language processing", "transformer", "bert"],
        "sql": ["sql", "database", "data engineering"],
        "docker": ["docker", "devops", "containers"],
        "aws": ["aws", "cloud", "lambda", "cloud computing"],
        "linux": ["linux", "shell"],
        "java": ["java", "spring"],
        "rag": ["rag", "vector search", "embedding", "retrieval augmented"],
    }

    # Build keyword list relevant to detected resume skills
    user_keywords = []
    for s in skills:
        if s.lower() in skill_to_keywords:
            user_keywords.extend(skill_to_keywords[s.lower()])

    # Get Udemy courses from metadata
    udemy_courses = [
        m for m in store.meta
        if "udemy" in m.get("source_file", "").lower()
    ]

    # Score and rank courses
    ranked = []
    for course in udemy_courses:
        text = (course.get("chunk", "") + " " + course.get("title", "")).lower()
        score = sum(1 for kw in user_keywords if kw in text)

        if score > 0:
            ranked.append({
                "title": course.get("title"),
                "url": course.get("url"),
                "score": score
            })

    # Sort by score descending
    ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)

    # Pick top 5 best matches
    final_courses = ranked[:5]

    # ----------------------------------------------------------
    # FIXED SUMMARY PROMPT
    # ----------------------------------------------------------
    prompt = f"""
You are an AI-powered career advisor.

Using the resume text below, generate:

1. A list of the candidate's major technical skills.
2. A brief overview of their experience (2–3 sentences).
3. A longer, clear career summary (5–6 sentences) describing:
   - strengths,
   - technical capabilities,
   - industry fit,
   - ideal roles,
   - and overall professional profile.

Resume:
\"\"\"{resume_text[:3000]}\"\"\" 
"""

    llm = HuggingFaceLLM(cfg["llm"]["model_id"], cfg["llm"]["max_tokens"])
    summary = llm.generate(prompt)

    # ----------------------------------------------------------
    # FINAL JSON RESPONSE
    # ----------------------------------------------------------
    return {
        "skills": skills,
        "jobs": jobs[:10],
        "courses": final_courses,
        "summary": summary,
    }


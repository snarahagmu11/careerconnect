from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
from ..ingestion.indexer import ingest_folder, search
from ..utils.config import CFG
from ..llm.generate import answer

router = APIRouter()

# ---------------- EXISTING ENDPOINTS ---------------- #

class QueryIn(BaseModel):
    query: str
    top_k: int | None = None

@router.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "app": CFG.get("app", {}).get("name", "careerconnect")}

@router.post("/ingest")
def ingest():
    """Index data from the raw folder."""
    n = ingest_folder(Path("data/raw"))
    return {"indexed_chunks": n}

@router.post("/query")
def query(payload: QueryIn):
    """Query documents using embeddings."""
    k = payload.top_k or CFG["app"]["top_k"]
    return {"results": search(payload.query, k)}

class ChatIn(BaseModel):
    question: str
    top_k: int | None = None

@router.post("/chat")
def chat(payload: ChatIn):
    """LLM-based question answering with retrieval."""
    k = payload.top_k or CFG["app"]["top_k"]
    ctx = search(payload.question, k)
    text = answer(payload.question, ctx)
    return {"answer": text, "context_used": ctx}


# ---------------- NEW ENDPOINT: Resume Upload ---------------- #

@router.post("/process_resume/")
async def process_resume(file: UploadFile = File(...)):
    """
    Accepts a PDF resume from Streamlit frontend,
    and returns mock skill and job recommendations.
    """

    # Read the uploaded file (you can later process this)
    pdf_bytes = await file.read()

    # TODO: Implement resume parsing logic later
    # For now, return mock data
    extracted_skills = ["Python", "SQL", "Machine Learning"]
    jobs = [
        {"Job Title": "Data Analyst", "Company": "Google", "Location": "Remote"},
        {"Job Title": "ML Engineer", "Company": "Amazon", "Location": "AP"},
        {"Job Title": "Business Analyst", "Company": "Deloitte", "Location": "New York"},
    ]

    return {
        "skills": extracted_skills,
        "jobs": jobs,
        "course": {
            "skill": "Tableau",
            "name": "Tableau for Beginners",
            "url": "https://udemy.com"
        },
        "summary": "You are well-suited for Data Analyst roles and could strengthen visualization skills (Tableau)."
    }

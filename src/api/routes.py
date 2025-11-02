from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
from ..ingestion.indexer import ingest_folder, search
from ..utils.config import CFG
from ..llm.generate import answer

router = APIRouter()

class QueryIn(BaseModel):
    query: str
    top_k: int | None = None

@router.get("/health")
def health():
    return {"status": "ok", "app": CFG.get("app", {}).get("name", "careerconnect")}

@router.post("/ingest")
def ingest():
    n = ingest_folder(Path("data/raw"))
    return {"indexed_chunks": n}

@router.post("/query")
def query(payload: QueryIn):
    k = payload.top_k or CFG["app"]["top_k"]
    return {"results": search(payload.query, k)}

class ChatIn(BaseModel):
    question: str
    top_k: int | None = None

@router.post("/chat")
def chat(payload: ChatIn):
    k = payload.top_k or CFG["app"]["top_k"]
    ctx = search(payload.question, k)
    text = answer(payload.question, ctx)
    return {"answer": text, "context_used": ctx}


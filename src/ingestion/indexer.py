from pathlib import Path
from .loader import load_paths, read_text
from .splitter import split_text
from ..embedding.model import embed_texts
from ..utils.vectorstore import TinyVectorStore
from ..utils.config import CFG

store = TinyVectorStore()

def ingest_folder(folder: Path) -> int:
    docs = []
    for p in load_paths(folder):
        docs.extend(read_text(p))
    texts, metas = [], []
    for d in docs:
        for ch in split_text(d["text"], CFG["app"]["chunk_size"], CFG["app"]["chunk_overlap"]):
            texts.append(ch)
            metas.append({"source": d["source"], "chunk": ch[:200] + ("..." if len(ch) > 200 else "")})
    if not texts:
        return 0
    vecs = embed_texts(texts)
    store.add(vecs, metas)
    return len(texts)

def search(query_text: str, k: int):
    qvec = embed_texts([query_text])
    return store.search(qvec, top_k=k)


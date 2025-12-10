# src/ingestion/indexer.py
from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..embedding.vector_store import FaissStore as FaissVectorStore
from .loader import load_paths
from .splitter import split_text
from ..embedding.model import embed_texts

RAW_DIR = Path("data/raw")
INDEX_DIR = "data/db"
BATCH = int(os.getenv("INGEST_BATCH", "2048"))  

def _read_texts_for_file(p: Path) -> List[Tuple[str, Dict]]:
    suf = p.suffix.lower()
    chunks = []
    metas = []

    if suf == ".csv":
        df = pd.read_csv(p, low_memory=False)
        if "text" not in df.columns:
            return []

        for _, row in df.iterrows():
            chunk = str(row.get("text", "")).strip()
            if not chunk:
                continue

            meta = {
                "source_file": src,
                "url": df.loc[i].get("url","N/A"),
                "salary": df.loc[i].get("salary","N/A"),
                "title": df.loc[i].get(title,"N/A"),
                "company": df.loc[i].get(comp,"N/A"),
                "location": df.loc[i].get(loc,"N/A"),
                "description": df.loc[i].get(desc,"")
            }

            chunks.append(chunk)
            metas.append(meta)

    elif suf in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        chunks = split_texts([text])
        metas = [{"source": str(p), "chunk_id": i, "chunk": chunks[i]} for i in range(len(chunks))]

    elif suf == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        full_text = "\n".join([(pg.extract_text() or "") for pg in reader.pages]).strip()
        chunks = split_texts([full_text])
        metas = [{"source": str(p), "chunk_id": i, "chunk": chunks[i]} for i in range(len(chunks))]

    return list(zip(chunks, metas))


def _hash_inputs(paths: List[Path]) -> Dict[str, str]:
    return {str(p): sha1_of_path(p) for p in paths}

def ingest_folder_dynamic(folder: Path) -> Dict:
    paths = load_paths(folder)
    current = _hash_inputs(paths)

    store = FaissVectorStore(dim=384, index_dir=INDEX_DIR)
    store.load()
    previous = store.read_fingerprints()

    if previous == current and store.index is not None and store.index.ntotal > 0:
        return {"reused": True, "indexed_chunks": int(store.index.ntotal), "files": list(current.keys())}

    all_chunks: List[str] = []
    all_metas: List[Dict] = []
    for p in paths:
        pairs = _read_texts_for_file(p)
        for ch, mt in pairs:
            all_chunks.append(ch)
            all_metas.append(mt)

    if not all_chunks:
        store.build_from(np.zeros((0, 384), dtype=np.float32), [])
        store.write_fingerprints(current)
        return {"reused": False, "indexed_chunks": 0, "files": list(current.keys())}

    first = True
    total = len(all_chunks)
    for i in tqdm(range(0, total, BATCH), desc="Indexing"):
        batch_chunks = all_chunks[i:i+BATCH]
        batch_metas  = all_metas[i:i+BATCH]
        vecs = embed_texts(batch_chunks)  

        if first:
            store.build_from(vecs, batch_metas)
            first = False
        else:
            store.append(vecs, batch_metas)

    store.write_fingerprints(current)
    return {"reused": False, "indexed_chunks": int(store.index.ntotal), "files": list(current.keys())}


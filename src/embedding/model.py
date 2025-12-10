# src/embedding/model.py
from __future__ import annotations
import os, sqlite3, hashlib, struct, atexit
from pathlib import Path
from typing import List, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # 384d
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv("USE_GPU","1") != "0" else "cpu"

CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", "data/index"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "embed_cache.sqlite3"
_conn = sqlite3.connect(CACHE_DB)
_cur = _conn.cursor()
_cur.execute("CREATE TABLE IF NOT EXISTS cache (sha1 TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
_conn.commit()
atexit.register(_conn.close)

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

def _get_cached(hs: List[str]) -> dict:
    q = "SELECT sha1, dim, vec FROM cache WHERE sha1 IN ({})".format(",".join("?"*len(hs)))
    rows = _cur.execute(q, hs).fetchall() if hs else []
    out = {}
    for k, dim, blob in rows:
        arr = np.frombuffer(blob, dtype=np.float32)
        if dim > 0:
            arr = arr.reshape(dim)
        out[k] = arr
    return out

def _put_cached(mapping: dict, dim: int):
    if not mapping: return
    rows = []
    for k, v in mapping.items():
        rows.append((k, dim, memoryview(np.asarray(v, dtype=np.float32).tobytes())))
    _cur.executemany("INSERT OR REPLACE INTO cache (sha1, dim, vec) VALUES (?, ?, ?)", rows)
    _conn.commit()

_model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
_dim = _model.get_sentence_embedding_dimension()

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embeds texts with caching & batching. Returns (N, dim) float32 normalized.
    """
    if not texts:
        return np.zeros((0, _dim), dtype=np.float32)

    hashes = [_sha1(t or "") for t in texts]
    cached = _get_cached(list(set(hashes)))
    miss_idx = [i for i, h in enumerate(hashes) if h not in cached]
    miss_texts = [texts[i] for i in miss_idx]

    new_map = {}
    if miss_texts:
        for i in tqdm(range(0, len(miss_texts), BATCH_SIZE), desc="Embedding (misses)"):
            batch = miss_texts[i:i+BATCH_SIZE]
            vecs = _model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=False)
            for j, v in enumerate(vecs):
                new_map[_sha1(batch[j])] = v.astype(np.float32)
        _put_cached(new_map, _dim)
        cached.update(new_map)

    mat = np.vstack([cached[h] for h in hashes]).astype(np.float32)
    return mat


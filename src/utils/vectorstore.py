# src/utils/vectorstore.py
"""
FAISS-backed persistent vector store with file-fingerprint tracking.

Files written under data/index/:
  - faiss.index        : FAISS index (IP/cosine with L2-normalized vectors)
  - meta.jsonl         : one JSON per vector (stores at least {"source", "chunk_id", "chunk"})
  - fingerprints.json  : { "path/to/input.csv": "<sha1>", ... } to detect input changes
"""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss

def sha1_of_path(p: Path, block: int = 1 << 20) -> str:
    """Streaming SHA1 of a file."""
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _to_unit(vecs: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    """Cast to float32 and L2-normalize (for cosine via inner-product)."""
    if vecs is None or len(vecs) == 0:
        if dim is None:
            raise ValueError("Empty vectors with unknown dim.")
        return np.zeros((0, dim), dtype="float32")
    vecs = vecs.astype("float32", copy=False)
    faiss.normalize_L2(vecs)
    return vecs

class FaissVectorStore:
    """
    Persistent FAISS index + parallel metadata + input fingerprints.

    Usage:
      store = FaissVectorStore(dim=384, index_dir="data/index")
      store.load()                      # load if present (fast)
      store.build_from(vecs, metas)     # replace & save
      store.append(vecs, metas)         # append & save
      hits = store.search(qvec, top_k)

    Notes:
      - Use cosine similarity via IndexFlatIP with L2-normalized vectors.
      - 'metas' should align 1:1 with vecs (same length).
      - Typical meta fields: {"source": "data/raw/..csv", "chunk_id": int, "chunk": "..."}.
    """

    def __init__(self, dim: int = 384, index_dir: str = "data/index"):
        self.dim = dim
        self.dir = Path(index_dir)
        self.p_index = self.dir / "faiss.index"
        self.p_meta  = self.dir / "meta.jsonl"
        self.p_fp    = self.dir / "fingerprints.json"

        self.index: Optional[faiss.Index] = None
        self.meta: List[Dict] = []
        self._loaded: bool = False

    def load(self) -> bool:
        """Load FAISS + meta from disk if available. Returns True if loaded."""
        if self.p_index.exists() and self.p_meta.exists():
            self.index = faiss.read_index(str(self.p_index))
            with self.p_meta.open("r", encoding="utf-8") as f:
                self.meta = [json.loads(line) for line in f]
            self._loaded = True
            return True
        self.index, self.meta, self._loaded = None, [], False
        return False

    def save(self) -> None:
        """Persist FAISS + meta to disk."""
        ensure_dir(self.dir)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        faiss.write_index(self.index, str(self.p_index))
        with self.p_meta.open("w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def build_from(self, vecs: np.ndarray, metas: List[Dict]) -> None:
        """Replace the entire index with vecs/metas and save."""
        vecs = _to_unit(vecs, dim=self.dim)
        self.index = faiss.IndexFlatIP(self.dim)
        if len(vecs):
            self.index.add(vecs)
        self.meta = metas
        self.save()

    def append(self, vecs: np.ndarray, metas: List[Dict]) -> None:
        """Append vectors/metas and save."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
        vecs = _to_unit(vecs, dim=self.dim)
        self.index.add(vecs)
        self.meta.extend(metas)
        self.save()

    def search(self, qvec: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Return top_k meta dicts with 'score' added."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q = _to_unit(qvec.reshape(1, -1), dim=self.dim)
        D, I = self.index.search(q, min(top_k, self.index.ntotal))
        out: List[Dict] = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            out.append({"score": float(score), **m})
        return out

    def size(self) -> int:
        return 0 if (self.index is None) else int(self.index.ntotal)

    def read_fingerprints(self) -> Dict[str, str]:
        if self.p_fp.exists():
            return json.loads(self.p_fp.read_text())
        return {}

    def write_fingerprints(self, mapping: Dict[str, str]) -> None:
        ensure_dir(self.dir)
        self.p_fp.write_text(json.dumps(mapping, indent=2))


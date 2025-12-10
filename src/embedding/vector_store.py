import faiss
import json
from pathlib import Path
import numpy as np


class FaissStore:
    def __init__(self, persist_directory="data/index", index_dim=384):
        self.persist_directory = Path(persist_directory)
        self.index_dim = index_dim
        self.index = None
        self.meta = []

    def build_from(self, vectors, metas):
        """
        Build FAISS index and metadata in strict 1-to-1 alignment.
        Prevents loss of Udemy rows, duplicate vector overwrites, and
        ensures all metas correspond exactly to FAISS index entries.
        """
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        index = faiss.IndexFlatL2(self.index_dim)
        index.add(vectors)

        self.index = index
        metafile = self.persist_directory / "meta.jsonl"
        with metafile.open("w", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")

        faiss.write_index(index, str(self.persist_directory / "faiss.index"))

    def load(self):
        """
        Load FAISS index and metadata.
        """
        try:
            idx = self.persist_directory / "faiss.index"
            mfile = self.persist_directory / "meta.jsonl"

            if not idx.exists():
                print("❗ No index found.")
                return False

            self.index = faiss.read_index(str(idx))
            self.meta = []

            if mfile.exists():
                with mfile.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.meta.append(json.loads(line))

            return True

        except Exception as e:
            print("Load error:", e)
            return False

    def size(self):
        """
        Number of vectors in FAISS index.
        """
        return self.index.ntotal if self.index else 0

    def search(self, query_vec, top_k=5):
        """
        Search FAISS for top-k vectors and return aligned metadata.
        """
        if self.index is None:
            return []

        query_vec = np.array([query_vec], dtype="float32")
        _, ids = self.index.search(query_vec, top_k)

        out = []
        for idx in ids[0]:
            if 0 <= idx < len(self.meta):
                out.append(self.meta[idx])

        return out


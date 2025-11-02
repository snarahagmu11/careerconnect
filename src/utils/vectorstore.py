from dataclasses import dataclass, field
import numpy as np
from sklearn.neighbors import NearestNeighbors

@dataclass
class TinyVectorStore:
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0,384)))
    metadata: list = field(default_factory=list)
    nn: NearestNeighbors | None = None

    def fit(self):
        if not self.metadata:
            return
        self.nn = NearestNeighbors(
            n_neighbors=min(5, len(self.metadata)), metric="cosine"
        )
        self.nn.fit(self.vectors)

    def add(self, vecs: np.ndarray, metas: list[dict]):
        self.vectors = vecs if self.vectors.size == 0 else np.vstack([self.vectors, vecs])
        self.metadata.extend(metas)
        self.fit()

    def search(self, qvec: np.ndarray, top_k: int = 5):
        if not self.nn:
            return []
        dists, idxs = self.nn.kneighbors(qvec, n_neighbors=min(top_k, len(self.metadata)))
        return [self.metadata[i] | {"score": 1 - float(d)} for i, d in zip(idxs[0], dists[0])]


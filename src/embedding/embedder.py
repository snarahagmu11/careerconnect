import os, torch, numpy as np
from sentence_transformers import SentenceTransformer

MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH = int(os.getenv("EMBED_BATCH_SIZE", "64"))
USE_GPU = os.getenv("USE_GPU", "1") == "1"
DEVICE = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"

_model = SentenceTransformer(MODEL, device=DEVICE)
_dim = _model.get_sentence_embedding_dimension()

def dim():
    return _dim

def embed_documents(texts):
    if not texts:
        return np.zeros((0, _dim), dtype="float32")
    return _model.encode(
        texts,
        batch_size=BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

def embed_query(text):
    return embed_documents([text])[0]


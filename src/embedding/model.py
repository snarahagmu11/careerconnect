from sentence_transformers import SentenceTransformer
from ..utils.config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts: list[str]):
    return get_model().encode(texts, show_progress_bar=False, convert_to_numpy=True)


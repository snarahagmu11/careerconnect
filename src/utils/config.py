from pathlib import Path
import os, yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)

def _load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CFG = _load_yaml(ROOT / "config" / "app.yaml")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# (HF vars are read directly by llm/generate.py; listed here for clarity)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


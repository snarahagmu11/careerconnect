import argparse, os, pandas as pd
from dotenv import load_dotenv; load_dotenv()

from pathlib import Path
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.embedding.embedder import embed_documents
from src.embedding.vector_store import FaissStore


def run_embed(cfg, log):
    raw = Path("data/raw")

    texts = []
    metas = []

    files = [
        "linkedin_jobs_processed.csv",
        "monster_jobs_processed.csv",
        "se_jobs_salaries_processed.csv",
        "udemy_courses_processed.csv",
    ]

    for fname in files:
        f = raw / fname
        if not f.exists():
            log.warning(f"[WARN] Missing file: {fname}")
            continue

        df = pd.read_csv(f).fillna("")
        for idx, row in df.iterrows():

            # Extract text safely
            text = str(row.get("text", "")).strip()

            # Skip invalid/very short text (prevents FAISS mismatch)
            if len(text) < 10:
                continue

            # Add UNIQUE identifier to avoid FAISS dedup removal
            meta = {
                "chunk": text,
                "source_file": fname,
                "uid": f"{fname}:{idx}"
            }

            # Keep all OPTIONAL job/course metadata exactly as before
            for key in ["title", "company", "location", "salary", "description", "url", "is_paid"]:
                if key in row:
                    meta[key] = row[key]

            texts.append(text)
            metas.append(meta)

    # ---- Safety Check: Prevent vector/meta length mismatch ----
    if len(texts) != len(metas):
        raise ValueError(
            f"Text/meta count mismatch! texts={len(texts)}, metas={len(metas)}"
        )

    # ---- Embed ALL texts ----
    vecs = embed_documents(texts)

    if vecs.shape[0] != len(metas):
        raise ValueError(
            f"Embedding mismatch: vectors={vecs.shape[0]}, metas={len(metas)}"
        )

    # ---- Build FAISS Store ----
    store = FaissStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        index_dim=cfg["vector_store"]["index_dim"],
    )
    store.build_from(vecs, metas)

    log.info(f"Indexed {store.size()} items.")


def main():
    cfg = load_config()
    log = setup_logger()

    run_embed(cfg, log)


if __name__ == "__main__":
    main()


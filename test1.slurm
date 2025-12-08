#!/bin/bash
#SBATCH --job-name=careerconnect-full
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2g.20gb:2
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/cc-full-%j.out
#SBATCH --error=logs/cc-full-%j.err

set -euo pipefail

REPO_DIR="$HOME/careerconnect"
VENV_DIR="$HOME/testnlpbot"

echo "[INFO] Host: $(hostname) | JobID: ${SLURM_JOB_ID}"
echo "[INFO] Using repo: $REPO_DIR"
echo "[INFO] Using venv: $VENV_DIR"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

# Optional: load .env if exists
if [ -f .env ]; then
  set -a; source .env; set +a
fi

export PYTHONPATH=.
export EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-64}
export INGEST_BATCH=${INGEST_BATCH:-2048}

# ----------------------------------------------------
# STEP 1 — PREPROCESS DATA
# ----------------------------------------------------
echo "[INFO] Starting preprocessing..."
python scripts/preprocess_data.py
echo "[INFO] Preprocessing complete."
echo "------------------------------------"

# ----------------------------------------------------
# STEP 2 — BUILD FAISS INDEX (Embedding step)
# ----------------------------------------------------
echo "[INFO] Starting embedding/index build..."
python src/main.py --mode embed
echo "[INFO] Embedding complete."
echo "------------------------------------"

# ----------------------------------------------------
# STEP 3 — SANITY CHECK FAISS INDEX
# ----------------------------------------------------
echo "[INFO] Checking FAISS index..."

python - <<'PY'
from src.utils.vectorstore import FaissVectorStore

store = FaissVectorStore(dim=384, index_dir="data/index")
ok = store.load()
print("FAISS Loaded:", ok)
print("Index size:", store.size())

if ok and store.size() > 0:
    m = store.meta[0]
    print("Sample source:", m.get("source"))
    print("Sample chunk preview:", (m.get("chunk") or "")[:200].replace("\n"," "))
else:
    print("❌ ERROR: FAISS index is empty or missing.")
PY

echo "[DONE] Preprocess + Embed job complete."

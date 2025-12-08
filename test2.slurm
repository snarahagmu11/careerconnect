#!/bin/bash
#SBATCH --job-name=careerconnect-full
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2g.20gb:2
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
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

# Optional .env loading
if [ -f .env ]; then
  set -a; source .env; set +a
fi

export PYTHONPATH=.
export EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-64}

echo "[INFO] Step 1: Preprocessing datasets..."
python scripts/preprocess_data.py

echo "[INFO] Step 2: Building FAISS index..."
python src/main.py --mode embed

echo "[INFO] Step 3: FAISS Sanity Check"
python - <<'EOF'
from src.embedding.vector_store import FaissStore

store = FaissStore(
    index_dim=384,
    persist_directory="data/index"
)

ok = store.load()
print("FAISS Loaded:", ok)
print("Index size:", store.size())

if ok and store.size() > 0:
    m = store.meta[0]
    print("Sample source:", m.get("source_file"))
    chunk = m.get("chunk") or ""
    print("Sample chunk preview:", chunk[:200].replace("\n"," "))
else:
    print("❌ ERROR: FAISS index is empty or missing.")
EOF

echo "[DONE] CareerConnect RAG pipeline complete."


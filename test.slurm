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

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PLEASE UPDATE THIS PATH IF YOUR REPO IS ELSEWHERE
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
REPO_DIR="$HOME/careerconnect"
VENV_DIR="$HOME/testnlpbot"

echo "================================================="
echo "[INFO] Host: $(hostname) | Job ID: ${SLURM_JOB_ID}"
echo "[INFO] REPO_DIR = $REPO_DIR"
echo "[INFO] VENV_DIR = $VENV_DIR"
echo "================================================="

# -------------------------------------------------
# Activate environment and enter repo
# -------------------------------------------------
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"

echo "[DEBUG] Slurm PWD = $(pwd)"
echo "[DEBUG] Listing data/raw:"
ls -l data/raw | head

# Optional .env
if [ -f .env ]; then
  set -a; source .env; set +a
fi

export PYTHONPATH=.
export EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-64}

# -------------------------------------------------
# Step 1: Preprocessing
# -------------------------------------------------
echo "================================================="
echo "[INFO] Step 1: Preprocessing datasets..."
echo "================================================="
python scripts/preprocess_data.py

echo "[DEBUG] After preprocessing, data/raw contains:"
ls -l data/raw | head

# -------------------------------------------------
# Step 2: Clear old FAISS index
# -------------------------------------------------
echo "================================================="
echo "[INFO] Step 2: Clearing previous FAISS index..."
echo "================================================="
rm -f data/index/faiss.index data/index/meta.jsonl

# -------------------------------------------------
# Step 3: Build FAISS index
# -------------------------------------------------
echo "================================================="
echo "[INFO] Step 3: Rebuilding FAISS index..."
echo "================================================="
python src/main.py

# -------------------------------------------------
# Step 4: FAISS Sanity Check
# -------------------------------------------------
echo "================================================="
echo "[INFO] Step 4: FAISS Sanity Check"
echo "================================================="
python - <<'EOF'
from src.embedding.vector_store import FaissStore

store = FaissStore(
    index_dim=384,
    persist_directory="data/index"
)

ok = store.load()
print("FAISS Loaded:", ok)
print("Index size:", store.size())

# Print the first item (job usually)
if ok and store.size() > 0:
    m = store.meta[0]
    print("Sample source:", m.get("source_file"))
    print("Sample preview:", (m.get("chunk") or "")[:200].replace("\n"," "))
else:
    print("❌ ERROR: FAISS index is empty or missing.")

# NEW: Check if Udemy rows exist in metadata
udemy = [m for m in store.meta if m.get("source_file") == "udemy_courses_processed.csv"]
print("Udemy rows found in FAISS metadata:", len(udemy))

# Print sample Udemy entry if exists
if udemy:
    print("Sample Udemy title:", udemy[0].get("title"))
    print("Sample Udemy URL:", udemy[0].get("url"))
EOF


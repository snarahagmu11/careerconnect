# src/eval/evaluate_retrieval.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import pandas as pd
from tqdm import tqdm
import random

from src.embedding.embedder import embed_query
from src.embedding.vector_store import FaissStore
from src.utils.config_loader import load_config


# -------------------------------------------------------------------
# STRONG MATCH (12)
# -------------------------------------------------------------------
STRONG_MATCH = [
    "Supply Chain Analyst",
    "Systems Administrator",
    "Supplier Quality Engineer",
    "Support Specialist",
    "System Engineer",
    "Marketing Coordinator",
    "Project Architect",
    "Structural Engineer",
    "Customer Service Associate",
    "IT Support Technician",
    "Building Engineer",
    "Content Writer Communications",
]


# -------------------------------------------------------------------
# NEGATIVE / NON-MATCH (5)
# -------------------------------------------------------------------
NEGATIVE = [
    "AI Research Scientist",
    "Blockchain Developer",
    "Deep Learning Strategist",
    "Senior Astro Physicist",
    "Nurse Practitioner Technician",
]


# -------------------------------------------------------------------
# AUTO-GENERATED MIXED QUERIES (183)
# -------------------------------------------------------------------
base_terms = [
    "coordinator", "assistant", "operations", "remote job",
    "entry level", "manager", "developer", "technician",
    "office role", "shift supervisor", "warehouse associate",
    "consultant", "director", "analyst", "processor",
]

def generate_mixed_queries(n=183):
    queries = []
    for _ in range(n):
        q = f"{random.choice(base_terms)} in {random.choice(base_terms)}"
        queries.append(q)
    return queries

MIXED = generate_mixed_queries(183)


# -------------------------------------------------------------------
# FINAL → 200 QUERIES
# -------------------------------------------------------------------
EVAL_QUERIES = STRONG_MATCH + NEGATIVE + MIXED
assert len(EVAL_QUERIES) == 200


# -------------------------------------------------------------------
# GROUND TRUTH
# -------------------------------------------------------------------
GROUND_TRUTH = {
    "Supply Chain Analyst": ["Supply Chain Analyst"],
    "Systems Administrator": ["Systems Administrator"],
    "Supplier Quality Engineer": ["Supplier Quality Engineer"],
    "Support Specialist": ["Support Specialist"],
    "System Engineer": ["System Engineer"],
    "Marketing Coordinator": ["Marketing Coordinator"],
    "Project Architect": ["Project Architect"],
    "Structural Engineer": ["Structural Engineer"],
    "Customer Service Associate": ["Customer Service Associate"],
    "IT Support Technician": ["IT Support Technician"],
    "Building Engineer": ["Building Engineer"],
    "Content Writer Communications": ["Content Writer", "Content Writer Communications"],
}


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------
def precision_at_k(retrieved, relevant, k=5):
    retrieved_k = retrieved[:k]
    rel = sum(1 for r in retrieved_k if r in relevant)
    return rel / k


def mean_reciprocal_rank(retrieved, relevant):
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / idx
    return 0.0


def ndcg_at_k(retrieved, relevant, k=10):
    import math
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            dcg += 1 / math.log2(i + 1)

    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():

    print("🔍 Loading FAISS index...")

    cfg = load_config()
    store = FaissStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        index_dim=cfg["vector_store"]["index_dim"]
    )
    store.load()

    results = []

    for query in tqdm(EVAL_QUERIES, desc="Evaluating 200 Queries"):
        qvec = embed_query(query)
        hits = store.search(qvec, top_k=10)
        retrieved_titles = [h.get("title", "") for h in hits]

        relevant = GROUND_TRUTH.get(query, [])

        p5 = precision_at_k(retrieved_titles, relevant, k=5)
        ndcg10 = ndcg_at_k(retrieved_titles, relevant, k=10)
        mrr = mean_reciprocal_rank(retrieved_titles, relevant)

        results.append({
            "query": query,
            "precision@5": round(p5, 3),
            "nDCG@10": round(ndcg10, 3),
            "MRR": round(mrr, 3),
        })

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)

    print("\n✔ Saved → evaluation_results.csv\n")
    print(df.head(20))

    # -------------------------------------------------------------------
    # PRINT AVERAGES
    # -------------------------------------------------------------------
    print("\n==========================")
    print("📊 AVERAGE METRICS")
    print("==========================")
    print("Avg Precision@5:", round(df["precision@5"].mean(), 4))
    print("Avg nDCG@10:", round(df["nDCG@10"].mean(), 4))
    print("Avg MRR:", round(df["MRR"].mean(), 4))
    print("==========================\n")


if __name__ == "__main__":
    main()


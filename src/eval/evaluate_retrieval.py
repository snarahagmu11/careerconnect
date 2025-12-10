# src/eval/evaluate_retrieval.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import pandas as pd
from tqdm import tqdm

from src.embedding.embedder import embed_query
from src.embedding.vector_store import FaissStore
from src.utils.config_loader import load_config


EVAL_QUERIES = [

    "Supply Chain Analyst",
    "Systems Administrator",
    "Support Specialist",
    "Supervisor/Manager Part-Time",

    "Supplier Quality Engineer",
    "Supply Chain Coordinator",
    "System Engineer",


    "AI Research Scientist",
    "Nurse Practitioner Technician",
    "Blockchain Developer",
]


GROUND_TRUTH = {
    "Supply Chain Analyst": [
        "Supply Chain Analyst",
        "Supply Chain Analyst II",
        "Supply Chain Analyst III",
        "Supply Chain Analyst IV",
    ],

    "Systems Administrator": [
        "System Administrator",
        "System Administrator I",
        "System Administrator II",
        "System Administrator III",
        "System Administrator Senior",
    ],

    "Support Specialist": [
        "Support Specialist",
        "Support Specialist (Remote)",
        "Support Specialist - Village (Sun-Th,8am-4:30pm)",
        "Support Specialist - Village (Tues-Sat, 8am-4:30pm)",
    ],

    "Supervisor/Manager Part-Time": [
        "Supervisor/Manager Part-Time Gateway Center",
        "Supervisor/Manager Part-Time Kings Plaza",
        "Supervisor/Manager Part-Time Queens Center Mall",
        "Supervisor/Manager-Part Time - Hamilton Place",
    ],

    "Supplier Quality Engineer": [
        "Supplier Quality Engineer",
        "Supplier Quality Engineer I/II",
        "Supplier Quality Engineer II",
        "Supplier Quality Engineer III",
    ],

    "Supply Chain Coordinator": [
        "Supply Chain Coordinator",
        "Supply Chain Coordinator (Customer Service)",
        "Supply Chain Coordinator (Logistics)",
    ],

    "System Engineer": [
        "System Engineer",
        "System Engineer I",
        "System Engineer II",
        "System Engineer III",
    ],

    "AI Research Scientist": [],
    "Nurse Practitioner Technician": [],
    "Blockchain Developer": [],
}


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

    ideal = sum(
        1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0
def main():
    cfg = load_config()
    store = FaissStore(
        persist_directory=cfg["vector_store"]["persist_directory"],
        index_dim=cfg["vector_store"]["index_dim"]
    )
    store.load()

    results = []

    for query in tqdm(EVAL_QUERIES, desc="Evaluating Queries"):
        qvec = embed_query(query)

        hits = store.search(qvec, top_k=10)
        retrieved_titles = [h["title"] for h in hits]

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
    print("\nSaved → evaluation_results.csv")
    print(df)


if __name__ == "__main__":
    main()

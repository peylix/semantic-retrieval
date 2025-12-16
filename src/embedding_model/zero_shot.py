"""
Zero shot test for bge-base model on BEIR-formatted scientific abstracts dataset.
"""

import logging
import pathlib, os

import importlib.util  # this import is to avoid a BEIR bug
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.models import SentenceBERT

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

# Load data from /data/processed/beir_format
project_root = pathlib.Path(__file__).parent.parent.parent.absolute()
data_path = os.path.join(project_root, "data", "processed", "beir_format")

logging.info(f"Loading data from: {data_path}")

# corpus: scientific abstracts to be retrieved
# queries: test queries
# qrels: ground truth mapping of relevant documents
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

logging.info(f"Loading completed: {len(corpus)} documents, {len(queries)} queries")

# Use MiniLM as the baseline (BEIR evaluation)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = DenseRetrievalExactSearch(SentenceBERT(model_name), batch_size=32)
retriever = EvaluateRetrieval(
    model, score_function="cos_sim"
)  # MiniLM with cosine similarity

logging.info("Start retrieval...")
results = retriever.retrieve(corpus, queries)

logging.info("Start evaluation...")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[10, 100])


print("\n------- Baseline Results (Zero-shot) -------")
print(f"Model: {model_name}")
print(f"NDCG@10:  {ndcg['NDCG@10']:.4f}")
print(f"Recall@100: {recall['Recall@100']:.4f}")
print("--------------------------------------------")

# Save results in unified format
import json

results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)

unified_results = {
    "model_name": "Zero-shot MiniLM",
    "model_type": "zero_shot",
    "base_model": model_name,
    "metrics": {
        "NDCG@10": float(ndcg.get("NDCG@10", 0.0)),
        "NDCG@100": float(ndcg.get("NDCG@100", 0.0)),
        "MAP@10": float(_map.get("MAP@10", 0.0)),
        "MAP@100": float(_map.get("MAP@100", 0.0)),
        "Recall@10": float(recall.get("Recall@10", 0.0)),
        "Recall@100": float(recall.get("Recall@100", 0.0)),
        "Precision@10": float(precision.get("P@10", 0.0)),
        "Precision@100": float(precision.get("P@100", 0.0)),
        "MRR": float(_map.get("MRR@10", 0.0)),
    },
}

# Log all available metrics for debugging
logging.info(f"Available NDCG metrics: {list(ndcg.keys())}")
logging.info(f"Available MAP metrics: {list(_map.keys())}")
logging.info(f"Available Recall metrics: {list(recall.keys())}")
logging.info(f"Available Precision metrics: {list(precision.keys())}")

results_path = os.path.join(results_dir, "zero_shot_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(unified_results, f, indent=2, ensure_ascii=False)

logging.info(f"Results saved to: {results_path}")

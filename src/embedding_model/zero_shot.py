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

# Use bge-base as the baseline
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "BAAI/bge-small-en-v1.5"
model = DenseRetrievalExactSearch(SentenceBERT(model_name), batch_size=32)
retriever = EvaluateRetrieval(
    model, score_function="cos_sim"
)  # bge recommends cosine similarity

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
    "model_name": "Zero-shot BGE-small",
    "model_type": "zero_shot",
    "base_model": model_name,
    "metrics": {
        "NDCG@10": ndcg.get("NDCG@10", 0.0),
        "NDCG@100": ndcg.get("NDCG@100", 0.0),
        "MAP@10": _map.get("MAP@10", 0.0),
        "MAP@100": _map.get("MAP@100", 0.0),
        "Recall@10": recall.get("Recall@10", 0.0),
        "Recall@100": recall.get("Recall@100", 0.0),
        "Precision@10": precision.get("P@10", 0.0),
        "Precision@100": precision.get("P@100", 0.0),
        "MRR": _map.get("MRR@10", 0.0),
    },
}

results_path = os.path.join(results_dir, "zero_shot_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(unified_results, f, indent=2, ensure_ascii=False)

logging.info(f"Results saved to: {results_path}")

"""
Zero shot test for bge-base model on BEIR-formatted scientific abstracts dataset.
"""

import logging
import pathlib, os

import importlib.util # this import is to avoid a BEIR bug
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
model_name = "BAAI/bge-base-en-v1.5"
model = DenseRetrievalExactSearch(SentenceBERT(model_name), batch_size=32)
retriever = EvaluateRetrieval(model, score_function="cos_sim") # bge recommends cosine similarity

logging.info("Start retrieval...")
results = retriever.retrieve(corpus, queries)

logging.info("Start evaluation...")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[10, 100])


print("\n------- Baseline Results (Zero-shot) -------")
print(f"Model: {model_name}")
print(f"NDCG@10:  {ndcg['NDCG@10']:.4f}")
print(f"Recall@100: {recall['Recall@100']:.4f}")
print("--------------------------------------------")

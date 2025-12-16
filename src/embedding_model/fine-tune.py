"""
Fine-tune embedding model using MultipleNegativesRankingLoss (in-batch negatives).
"""

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_data import load_qrels, load_queries, load_corpus  # type: ignore

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============== Configuration ==============
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32
EPOCHS = 6
WARMUP_RATIO = 0.1
# LEARNING_RATE = 2e-5
LEARNING_RATE = 3e-5
OUTPUT_DIR = "models/finetuned-mnrl"
# ===========================================


def load_training_data(data_dir: Path) -> list[InputExample]:
    """
    Load training data and convert to InputExample format for sentence-transformers.

    For MultipleNegativesRankingLoss, we only need (query, positive_doc) pairs.
    Negatives are sampled from other examples in the same batch.
    """
    qrels_path = data_dir / "qrels" / "train.tsv"
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"

    logger.info("Loading training data...")
    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    # Create InputExample for each (query, positive_doc) pair
    examples = []
    for query_id, doc_ids in query_to_docs.items():
        if query_id not in queries:
            continue

        query_text = queries[query_id]

        for doc_id in doc_ids:
            if doc_id not in corpus:
                continue

            doc_text = corpus[doc_id]
            # InputExample with texts=[query, positive_doc]
            examples.append(InputExample(texts=[query_text, doc_text]))

    logger.info(f"Created {len(examples)} training examples")
    return examples


def create_evaluator(
    data_dir: Path, split: str = "dev"
) -> InformationRetrievalEvaluator:
    """
    Create an InformationRetrievalEvaluator for validation during training.
    """
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"

    logger.info(f"Loading {split} data for evaluation...")
    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    # Filter queries and corpus to only include those in the split
    eval_queries = {qid: queries[qid] for qid in query_to_docs if qid in queries}

    # Get all doc_ids that are relevant to any query in this split
    relevant_doc_ids = set()
    for doc_ids in query_to_docs.values():
        relevant_doc_ids.update(doc_ids)

    # Use full corpus for evaluation (retrieval should find relevant docs from full corpus)
    eval_corpus = {doc_id: corpus[doc_id] for doc_id in corpus}

    # Convert qrels format: query_id -> set of doc_ids
    eval_qrels: dict[str, set[str]] = {
        qid: set(doc_ids)
        for qid, doc_ids in query_to_docs.items()
        if qid in eval_queries
    }

    logger.info(f"Evaluator: {len(eval_queries)} queries, {len(eval_corpus)} documents")

    return InformationRetrievalEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=eval_qrels,
        name=split,
        ndcg_at_k=[10, 100],
        precision_recall_at_k=[10, 100],
        map_at_k=[100],
        mrr_at_k=[10],
        show_progress_bar=True,
    )


def main():
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed" / "beir_format"
    output_dir = project_root / OUTPUT_DIR

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Load training data
    train_examples = load_training_data(data_dir)
    train_dataloader: DataLoader[InputExample] = DataLoader(
        train_examples,  # type: ignore[arg-type]
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    # Setup loss function - MultipleNegativesRankingLoss uses in-batch negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Create evaluator for dev set
    dev_evaluator = create_evaluator(data_dir, split="dev")

    # Calculate training steps
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    logger.info(f"Training configuration:")
    logger.info(f"  - Model: {MODEL_NAME}")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Epochs: {EPOCHS}")
    logger.info(f"  - Total steps: {total_steps}")
    logger.info(f"  - Warmup steps: {warmup_steps}")
    logger.info(f"  - Learning rate: {LEARNING_RATE}")
    logger.info(f"  - Output directory: {output_dir}")

    # Train the model
    logger.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        output_path=str(output_dir),
        evaluation_steps=len(train_dataloader) // 2,  # Evaluate twice per epoch
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info(f"Training completed! Model saved to: {output_dir}")

    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_evaluator = create_evaluator(data_dir, split="test")
    test_results = test_evaluator(model, output_path=str(output_dir))

    logger.info("\n" + "=" * 50)
    logger.info("Final Test Results:")
    for metric, value in test_results.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()


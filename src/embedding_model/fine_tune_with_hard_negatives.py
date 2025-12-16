"""Fine-tune embedding model with pre-mined hard negatives using CachedMultipleNegativesRankingLoss."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hard_negative_dataset import InputExampleDataset  # type: ignore
from training_data import (
    load_qrels,
    load_queries,
    load_corpus,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 64
MINI_BATCH_SIZE = 16
EPOCHS = 4
WARMUP_RATIO = 0.1
LEARNING_RATE = 3e-5
OUTPUT_DIR = "models/finetuned-mnrl-hardneg"
HARD_NEGATIVES_PATH = "data/processed/beir_format/hard_negatives_train.jsonl"
NEGATIVES_PER_QUERY = 2


def load_hard_negatives(path: Path) -> dict[str, list[str]]:
    hard_negatives: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        import json

        for line in f:
            record = json.loads(line)
            hard_negatives[record["query_id"]] = record.get("hard_negative_doc_ids", [])
    return hard_negatives


def load_training_data_with_hard_negatives(
    data_dir: Path,
    hard_negatives_path: Path,
) -> list[InputExample]:
    qrels_path = data_dir / "qrels" / "train.tsv"
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"

    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)
    hard_negatives = load_hard_negatives(hard_negatives_path)

    examples: list[InputExample] = []
    for query_id, doc_ids in query_to_docs.items():
        if query_id not in queries:
            continue
        query_text = queries[query_id]
        hard_negs = hard_negatives.get(query_id, [])
        if len(hard_negs) < NEGATIVES_PER_QUERY:
            continue

        for doc_id in doc_ids:
            if doc_id not in corpus:
                continue
            doc_text = corpus[doc_id]
            neg_texts = [corpus[neg_id] for neg_id in hard_negs if neg_id in corpus][
                :NEGATIVES_PER_QUERY
            ]
            if len(neg_texts) < NEGATIVES_PER_QUERY:
                continue
            texts = [query_text, doc_text, *neg_texts]
            examples.append(InputExample(texts=texts))

    logger.info("Created %d training examples with hard negatives", len(examples))
    return examples


def create_evaluator(
    data_dir: Path,
    split: str = "dev",
) -> InformationRetrievalEvaluator:
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"

    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    eval_queries = {qid: queries[qid] for qid in query_to_docs if qid in queries}

    eval_corpus = {doc_id: corpus[doc_id] for doc_id in corpus}

    eval_qrels: dict[str, set[str]] = {
        qid: set(doc_ids)
        for qid, doc_ids in query_to_docs.items()
        if qid in eval_queries
    }

    logger.info(
        "Evaluator: %d queries, %d documents (split=%s)",
        len(eval_queries),
        len(eval_corpus),
        split,
    )

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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "processed" / "beir_format"
    output_dir = project_root / OUTPUT_DIR
    hard_negatives_path = project_root / HARD_NEGATIVES_PATH

    if not hard_negatives_path.exists():
        raise FileNotFoundError(
            f"Hard negatives file not found: {hard_negatives_path}. Please run mine_hard_negatives.py first."
        )

    logger.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    train_examples = load_training_data_with_hard_negatives(
        data_dir, hard_negatives_path
    )
    train_dataset = InputExampleDataset(train_examples)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
    )

    train_loss = losses.CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=MINI_BATCH_SIZE,
    )

    dev_evaluator = create_evaluator(data_dir, split="dev")

    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    logger.info("Training configuration:")
    logger.info("  - Model: %s", MODEL_NAME)
    logger.info("  - Batch size: %d", BATCH_SIZE)
    logger.info("  - Mini-batch size: %d", MINI_BATCH_SIZE)
    logger.info("  - Epochs: %d", EPOCHS)
    logger.info("  - Total steps: %d", total_steps)
    logger.info("  - Warmup steps: %d", warmup_steps)
    logger.info("  - Learning rate: %.2e", LEARNING_RATE)
    logger.info("  - Output directory: %s", output_dir)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        output_path=str(output_dir),
        evaluation_steps=max(1, len(train_dataloader) // 2),
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info("Training completed! Model saved to: %s", output_dir)

    logger.info("Running final evaluation on test set...")
    test_evaluator = create_evaluator(data_dir, split="test")
    test_results = test_evaluator(model, output_path=str(output_dir))

    logger.info("\n" + "=" * 50)
    logger.info("Final Test Results:")
    for metric, value in test_results.items():
        logger.info("  %s: %.4f", metric, value)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

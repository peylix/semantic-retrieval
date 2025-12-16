"""Mine hard negatives for semantic retrieval fine-tuning.

This script encodes the full training corpus and queries with a sentence-transformer
model, runs semantic search to obtain the top-k most similar documents per query,
filters out positives, and stores the selected hard negatives to a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from sentence_transformers import SentenceTransformer, util

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_data import load_qrels, load_queries, load_corpus  # type: ignore

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine hard negatives using a SentenceTransformer model"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Base model checkpoint to encode queries and documents.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "beir_format",
        help="Directory containing corpus.jsonl, queries.jsonl, and qrels.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test"],
        help="Qrels split to use for mining hard negatives.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "processed"
        / "beir_format"
        / "hard_negatives_train.jsonl",
        help="Output JSONL file to store mined hard negatives.",
    )
    parser.add_argument(
        "--negatives-per-query",
        type=int,
        default=2,
        help="Number of hard negatives to keep per query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k retrieved documents (before filtering positives).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding corpus and queries.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings before similarity search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize_embeddings: bool,
    desc: str,
) -> torch.Tensor:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=normalize_embeddings,
    )


def mine_hard_negatives(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    logger.info("Loading model: %s", args.model_name_or_path)
    model = SentenceTransformer(args.model_name_or_path, device=device)

    qrels_path = args.data_dir / "qrels" / f"{args.split}.tsv"
    queries_path = args.data_dir / "queries.jsonl"
    corpus_path = args.data_dir / "corpus.jsonl"

    logger.info("Loading data from %s", args.data_dir)
    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    logger.info(
        "Loaded %d queries with positives and %d corpus documents",
        len(query_to_docs),
        len(corpus),
    )

    doc_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id] for doc_id in doc_ids]
    corpus_embeddings = encode_texts(
        model,
        corpus_texts,
        args.batch_size,
        args.normalize,
        desc="Encoding corpus",
    )

    query_ids = [qid for qid in query_to_docs.keys() if qid in queries]
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = encode_texts(
        model,
        query_texts,
        args.batch_size,
        args.normalize,
        desc="Encoding queries",
    )

    effective_top_k = max(args.top_k, args.negatives_per_query * 5)
    logger.info(
        "Running semantic search with top_k=%d (before filtering positives)",
        effective_top_k,
    )

    search_results = util.semantic_search(
        query_embeddings,
        corpus_embeddings,
        top_k=effective_top_k,
        score_function=util.dot_score if args.normalize else util.cos_sim,
    )

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file {args.output_path} already exists. Use --overwrite to replace it."
        )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_queries = 0
    queries_with_negatives = 0

    with args.output_path.open("w", encoding="utf-8") as f:
        for idx, hits in enumerate(search_results):
            query_id = query_ids[idx]
            positives = query_to_docs.get(query_id, set())

            hard_neg_ids: List[str] = []
            for hit in hits:
                corpus_idx = hit.get("corpus_id")
                if corpus_idx is None:
                    continue
                corpus_index_int = int(corpus_idx)
                if corpus_index_int < 0 or corpus_index_int >= len(doc_ids):
                    continue
                doc_id = doc_ids[corpus_index_int]
                if doc_id in positives:
                    continue
                if doc_id in hard_neg_ids:
                    continue
                hard_neg_ids.append(doc_id)
                if len(hard_neg_ids) >= args.negatives_per_query:
                    break

            total_queries += 1
            if len(hard_neg_ids) >= args.negatives_per_query:
                queries_with_negatives += 1

            record = {
                "query_id": query_id,
                "positive_doc_ids": sorted(list(positives)),
                "hard_negative_doc_ids": hard_neg_ids,
                "query_text": queries[query_id],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Finished mining hard negatives: %d/%d queries have >= %d hard negatives",
        queries_with_negatives,
        total_queries,
        args.negatives_per_query,
    )
    logger.info("Saved results to %s", args.output_path)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    args = parse_args()
    mine_hard_negatives(args)


if __name__ == "__main__":
    main()

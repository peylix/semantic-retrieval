"""
Construct training samples from qrels files.
"""

from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, Set, List, Tuple, Optional


def load_qrels(qrels_path: Path) -> Dict[str, Set[str]]:
    """
    Load qrels file and construct a mapping from query_id to set of positive doc_ids.

    Args:
        qrels_path: Path to qrels tsv file (format: query_id\tdoc_id\tscore)

    Returns:
        Dictionary mapping query_id -> {doc_id1, doc_id2, ...}
    """
    query_to_docs = defaultdict(set)

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                query_id = parts[0]
                doc_id = parts[1]
                query_to_docs[query_id].add(doc_id)

    return dict(query_to_docs)


def load_queries(queries_path: Path) -> Dict[str, str]:
    """
    Load queries from queries.jsonl file.

    Args:
        queries_path: Path to queries.jsonl file

    Returns:
        Dictionary mapping query_id -> query_text
    """
    queries = {}

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            query = json.loads(line)
            queries[query["_id"]] = query["text"]

    return queries


def load_corpus(corpus_path: Path) -> Dict[str, str]:
    """
    Load corpus from corpus.jsonl file.

    Args:
        corpus_path: Path to corpus.jsonl file

    Returns:
        Dictionary mapping doc_id -> document_text
    """
    corpus = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc["text"]

    return corpus


def construct_training_samples(
    split: str = "train", data_dir: Optional[Path] = None
) -> List[Tuple[str, str, str]]:
    """
    Construct training samples from qrels, queries, and corpus.

    Args:
        split: Which split to use ("train", "dev", or "test")
        data_dir: Path to the beir_format directory

    Returns:
        List of (query_id, query_text, doc_text) tuples for positive pairs
    """
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data" / "processed" / "beir_format"

    # Load data
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"

    print(f"Loading {split} data...")
    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    # Construct samples
    samples = []
    for query_id, doc_ids in query_to_docs.items():
        if query_id not in queries:
            print(f"Warning: query_id {query_id} not found in queries.jsonl")
            continue

        query_text = queries[query_id]

        for doc_id in doc_ids:
            if doc_id not in corpus:
                print(f"Warning: doc_id {doc_id} not found in corpus.jsonl")
                continue

            doc_text = corpus[doc_id]
            samples.append((query_id, query_text, doc_text))

    print(
        f"Constructed {len(samples)} training samples from {len(query_to_docs)} queries"
    )
    return samples


def get_query_positive_docs(
    split: str = "train", data_dir: Optional[Path] = None
) -> Tuple[Dict[str, Set[str]], Dict[str, str], Dict[str, str]]:
    """
    Get query -> positive docs mapping along with queries and corpus.

    Args:
        split: Which split to use ("train", "dev", or "test")
        data_dir: Path to the beir_format directory

    Returns:
        Tuple of (query_to_docs, queries, corpus) where:
        - query_to_docs: Dict[query_id -> Set[doc_id]]
        - queries: Dict[query_id -> query_text]
        - corpus: Dict[doc_id -> doc_text]
    """
    if data_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / "data" / "processed" / "beir_format"

    # Load data
    qrels_path = data_dir / "qrels" / f"{split}.tsv"
    queries_path = data_dir / "queries.jsonl"
    corpus_path = data_dir / "corpus.jsonl"

    print(f"Loading {split} data...")
    query_to_docs = load_qrels(qrels_path)
    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    print(f"Loaded {len(query_to_docs)} queries with positive documents")
    print(f"Total queries: {len(queries)}")
    print(f"Total documents: {len(corpus)}")

    return query_to_docs, queries, corpus


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Example 1: Get query -> positive docs mapping")
    print("=" * 60)
    query_to_docs, queries, corpus = get_query_positive_docs(split="train")

    # Show first few examples
    for i, (query_id, doc_ids) in enumerate(list(query_to_docs.items())[:3]):
        print(f"\nQuery ID: {query_id}")
        print(f"Query text: {queries[query_id][:100]}...")
        print(f"Positive docs: {doc_ids}")
        for doc_id in list(doc_ids)[:2]:
            print(f"  Doc {doc_id}: {corpus[doc_id][:100]}...")

    print("\n" + "=" * 60)
    print("Example 2: Construct training samples")
    print("=" * 60)
    samples = construct_training_samples(split="train")

    # Show first few samples
    for i, (query_id, query_text, doc_text) in enumerate(samples[:3]):
        print(f"\nSample {i + 1}:")
        print(f"  Query ID: {query_id}")
        print(f"  Query: {query_text[:100]}...")
        print(f"  Doc: {doc_text[:100]}...")

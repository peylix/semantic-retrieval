import bm25s
import Stemmer
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path


def build_bm25_index(corpus: list[str], index_path: Optional[str] = None):
    """
    Build BM25 index from a corpus of documents.

    Args:
        corpus: List of document strings to index
        index_path: Optional path to save the index

    Returns:
        stemmer: The stemmer used for tokenization
        retriever: The BM25 retriever model
    """
    stemmer = Stemmer.Stemmer("english")

    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.method = "bm25l"
    retriever.index(corpus_tokens)

    # Save the index along with the corpus
    if index_path is not None:
        retriever.save(save_dir=index_path, corpus=corpus)

    return stemmer, retriever


def bm25_search(query: str, stemmer, retriever, k: int = 10):
    """
    Search for relevant documents using BM25.

    Args:
        query: Query string
        stemmer: Stemmer for tokenization
        retriever: Pre-built BM25 retriever
        k: Number of top results to return

    Returns:
        results: Document indices of top-k results (shape: (1, k))
        scores: BM25 scores for top-k results
    """
    query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=k)
    return results, scores


def evaluate_bm25_on_scifact(
    corpus_path: str,
    pairs_path: str,
    k: int = 10,
    max_queries: Optional[int] = 1000,
) -> dict:
    """
    Evaluate BM25 on SciFact dataset.

    Strategy:
    - Build corpus from ALL contents in scifact_corpus.csv
    - For each (query, gold_doc_id) in scifact_pairs.csv, check if gold_doc_id is in top-k
    - Compute accuracy@k and MRR

    Args:
        corpus_path: Path to scifact_corpus.csv (columns: doc_id, content)
        pairs_path: Path to scifact_pairs.csv (columns: query_id, query, doc_id, ...)
        k: Number of top results to retrieve
        max_queries: Evaluate on at most this many queries (None = all)

    Returns:
        Dictionary with evaluation metrics (accuracy@k, MRR, etc.)
    """
    # Load corpus and pairs
    corpus_df = pd.read_csv(corpus_path)
    pairs_df = pd.read_csv(pairs_path)

    # Corpus
    doc_ids = corpus_df["doc_id"].tolist()
    corpus = corpus_df["content"].astype(str).tolist()

    # Build BM25 index once for the entire corpus
    print(f"Building BM25 index with {len(corpus)} documents...")
    stemmer, retriever = build_bm25_index(corpus)

    # 限制评估 query 数量（太大会很慢）
    if max_queries is not None and max_queries < len(pairs_df):
        pairs_df = pairs_df.iloc[:max_queries].copy()

    print(f"Evaluating on {len(pairs_df)} queries...")

    hits_at_k = 0
    reciprocal_ranks = []

    for _, row in pairs_df.iterrows():
        query = str(row["query"])
        gold_doc_id = row["doc_id"]

        results, scores = bm25_search(query, stemmer, retriever, k=k)
        retrieved_indices = results[0].tolist()
        retrieved_doc_ids = [doc_ids[i] for i in retrieved_indices]

        if gold_doc_id in retrieved_doc_ids:
            hits_at_k += 1
            rank = retrieved_doc_ids.index(gold_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    accuracy_at_k = hits_at_k / len(pairs_df)
    mrr = float(np.mean(reciprocal_ranks))

    return {
        "accuracy@k": accuracy_at_k,
        "mrr": mrr,
        "k": k,
        "corpus_size": len(corpus),
        "total_queries": len(pairs_df),
        "hits": hits_at_k,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evaluate BM25 (bm25s) on SciFact")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    corpus_path = project_root / "data" / "processed" / "scifact_corpus.csv"
    pairs_path = project_root / "data" / "processed" / "scifact_pairs.csv"

    if corpus_path.exists() and pairs_path.exists():
        results = evaluate_bm25_on_scifact(
            str(corpus_path),
            str(pairs_path),
            k=10,
            max_queries=1000,  # 想全量评估就改成 None
        )

        print(f"\nResults:")
        print(f"  Corpus size: {results['corpus_size']}")
        print(f"  Queries: {results['total_queries']}")
        print(f"  Hits: {results['hits']}/{results['total_queries']}")
        print(f"  Accuracy@{results['k']}: {results['accuracy@k']:.4f}")
        print(f"  MRR: {results['mrr']:.4f}")
    else:
        print("Processed SciFact files not found.")
        print(f"  corpus: {corpus_path}")
        print(f"  pairs : {pairs_path}")

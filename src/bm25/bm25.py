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


def bm25_search(query: str, stemmer, retriever, k: int = 2):
    """
    Search for relevant documents using BM25.

    Args:
        corpus: List of document strings
        query: Query string
        stemmer: Stemmer for tokenization
        retriever: Pre-built BM25 retriever
        k: Number of top results to return

    Returns:
        results: Document indices of top-k results
        scores: BM25 scores for top-k results
    """
    query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)

    # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
    # To return docs instead of IDs, set the `corpus=corpus` parameter.
    results, scores = retriever.retrieve(query_tokens, k=k)

    return results, scores


def evaluate_bm25_on_sts(
    data_path: str, k: int = 10, similarity_threshold: float = 0.5
) -> dict:
    """
    Evaluate BM25 on STSBenchmark dataset.

    Strategy:
    - Build corpus from ALL sentence2s in the dataset (to simulate real retrieval)
    - For each query (sentence1) where score_norm >= threshold, check if correct sentence2 is in top-k
    - This properly evaluates retrieval performance in a realistic setting

    Args:
        data_path: Path to the STS CSV file (with columns: sentence1, sentence2, score, score_norm)
        k: Number of top results to retrieve
        similarity_threshold: Minimum normalized score to consider as "correct" match

    Returns:
        Dictionary with evaluation metrics (accuracy@k, MRR, etc.)
    """
    # Load data
    df = pd.read_csv(data_path)

    # Build corpus from ALL sentence2s (not just similar pairs)
    # This creates a realistic retrieval scenario
    all_sentences = pd.concat([df["sentence1"], df["sentence2"]]).unique().tolist()
    corpus = all_sentences
    # corpus = df["sentence2"].unique().tolist()

    # Create mapping from sentence to indices in corpus
    sentence_to_idx = {sent: idx for idx, sent in enumerate(corpus)}

    # Build BM25 index once for the entire corpus
    print(f"Building BM25 index with {len(corpus)} unique documents...")
    stemmer, retriever = build_bm25_index(corpus)

    # Filter test pairs: only evaluate on pairs with score >= threshold
    test_pairs = df[df["score_norm"] >= similarity_threshold].reset_index(drop=True)

    if len(test_pairs) == 0:
        raise ValueError(f"No pairs found with score_norm >= {similarity_threshold}")

    print(
        f"Evaluating on {len(test_pairs)} query pairs (score_norm >= {similarity_threshold})..."
    )

    # Evaluate
    hits_at_k = 0
    reciprocal_ranks = []

    for idx, row in test_pairs.iterrows():
        query = str(row["sentence1"])
        target_sentence = str(row["sentence2"])

        # Search in the full corpus
        results, scores = bm25_search(query, stemmer, retriever, k=k)

        # Get retrieved document indices
        retrieved_indices = results[0].tolist()

        # Get the actual retrieved sentences
        retrieved_sentences = [corpus[i] for i in retrieved_indices]
        retrieved_sentences = [s for s in retrieved_sentences if s != query]

        # Check if target sentence appears in top-k retrieved sentences
        if target_sentence in retrieved_sentences:
            hits_at_k += 1
            # Calculate reciprocal rank (position of first occurrence)
            rank = retrieved_sentences.index(target_sentence) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    # Calculate metrics
    accuracy_at_k = hits_at_k / len(test_pairs)
    mrr = np.mean(reciprocal_ranks)

    results = {
        "accuracy@k": accuracy_at_k,
        "mrr": mrr,
        "k": k,
        "corpus_size": len(corpus),
        "total_pairs": len(test_pairs),
        "hits": hits_at_k,
        "similarity_threshold": similarity_threshold,
    }

    return results


if __name__ == "__main__":
    # Evaluate on STSBenchmark
    print("\n" + "=" * 60)
    print("Evaluate BM25 on STSBenchmark")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[2]
    test_path = project_root / "data" / "processed" / "sts_test_clean.csv"

    if test_path.exists():
        eval_results = evaluate_bm25_on_sts(
            str(test_path), k=10, similarity_threshold=0.5
        )

        print(f"\nResults on test set:")
        print(f"  Corpus size: {eval_results['corpus_size']} unique sentences")
        print(
            f"  Test pairs (score_norm >= {eval_results['similarity_threshold']}): {eval_results['total_pairs']}"
        )
        print(f"  Hits: {eval_results['hits']}/{eval_results['total_pairs']}")
        print(f"  Accuracy@{eval_results['k']}: {eval_results['accuracy@k']:.4f}")
        print(f"  MRR: {eval_results['mrr']:.4f}")
    else:
        print(f"Test data not found at: {test_path}")

from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import train_test_split


def preprocess_scifact():
    """
    Read data/raw/scifact_raw.csv, apply light cleaning, and produce:
      - scifact_pairs.csv  : each row is a (query, doc) pair
      - scifact_corpus.csv : deduplicated document store (doc_id, content)
      - BEIR formatted data (for dense retrieval):
        - corpus.jsonl : corpus
        - queries.jsonl : queries
        - qrels/train.tsv : query-document relevance labels (70%)
        - qrels/dev.tsv : query-document relevance labels (10%)
        - qrels/test.tsv : query-document relevance labels (20%)
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "scifact_raw.csv"
    df = pd.read_csv(raw_path)

    # Keep only the columns we care about
    cols = ["_id", "title", "text", "query"]
    df = df[cols].copy()

    # Drop rows that lack text or query
    df = df.dropna(subset=["text", "query"])  # type: ignore

    # Rename the id column for later use
    df = df.rename(columns={"_id": "doc_id"})

    # Basic string normalization
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()

    # Combine title and text to form the content field used by BM25 and embeddings
    df["content"] = df["title"] + ". " + df["text"]

    # Assign one query_id per distinct query string
    df = df.reset_index(drop=True)
    df["query_id"] = pd.factorize(df["query"])[0]

    # Step 1: save pairs for training or evaluating dense retrievers
    pairs_path = processed_dir / "scifact_pairs.csv"
    df.to_csv(pairs_path, index=False)

    # Step 2: save a deduplicated corpus (one row per doc_id) for indexing/vector stores
    corpus = (
        df[["doc_id", "content"]]
        .drop_duplicates(subset=["doc_id"])  # type: ignore
        .reset_index(drop=True)
    )
    corpus_path = processed_dir / "scifact_corpus.csv"
    corpus.to_csv(corpus_path, index=False)

    # Step 3: write BEIR formatted files
    print("Generating BEIR formatted data...")

    # Create BEIR data directories
    beir_dir = processed_dir / "beir_format"
    beir_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir = beir_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)

    # Write corpus.jsonl
    corpus_jsonl_path = beir_dir / "corpus.jsonl"
    with open(corpus_jsonl_path, "w", encoding="utf-8") as f:
        for _, row in corpus.iterrows():
            # BEIR corpus format: {"_id": "doc_id", "text": "content", "title": ""}
            doc = {
                "_id": str(row["doc_id"]),
                "text": row["content"],
                "title": "",  # Title has already been merged into content
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Write queries.jsonl
    queries_jsonl_path = beir_dir / "queries.jsonl"
    unique_queries = df[["query_id", "query"]].drop_duplicates(subset=["query_id"])  # type: ignore
    with open(queries_jsonl_path, "w", encoding="utf-8") as f:
        for _, row in unique_queries.iterrows():
            # BEIR queries format: {"_id": "query_id", "text": "query"}
            query = {"_id": str(row["query_id"]), "text": row["query"]}
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    # Generate qrels/train.tsv, qrels/dev.tsv and qrels/test.tsv (query-document relevance labels)
    # Split by query_id to keep all documents for the same query in one split
    # Split ratio: train=70%, dev=10%, test=20%
    unique_query_ids = df["query_id"].unique()

    # First split: separate test set (20%)
    train_dev_query_ids, test_query_ids = train_test_split(
        unique_query_ids, test_size=0.2, random_state=42
    )

    # Second split: separate dev set from train (10% of total = 12.5% of remaining 80%)
    train_query_ids, dev_query_ids = train_test_split(
        train_dev_query_ids, test_size=0.125, random_state=42
    )

    train_df = df[df["query_id"].isin(train_query_ids)]
    dev_df = df[df["query_id"].isin(dev_query_ids)]
    test_df = df[df["query_id"].isin(test_query_ids)]

    # Write train.tsv
    train_qrels_path = qrels_dir / "train.tsv"
    with open(train_qrels_path, "w", encoding="utf-8") as f:
        for _, row in train_df.iterrows():
            # Every query-doc pair is marked as relevant (score=1)
            f.write(f"{row['query_id']}\t{row['doc_id']}\t1\n")

    # Write dev.tsv
    dev_qrels_path = qrels_dir / "dev.tsv"
    with open(dev_qrels_path, "w", encoding="utf-8") as f:
        for _, row in dev_df.iterrows():
            f.write(f"{row['query_id']}\t{row['doc_id']}\t1\n")

    # Write test.tsv
    test_qrels_path = qrels_dir / "test.tsv"
    with open(test_qrels_path, "w", encoding="utf-8") as f:
        for _, row in test_df.iterrows():
            f.write(f"{row['query_id']}\t{row['doc_id']}\t1\n")

    print(f"CSV data saved to: {processed_dir}")
    print(f"   - scifact_pairs.csv: {len(df)} query-doc pairs")
    print(f"   - scifact_corpus.csv: {len(corpus)} documents")
    print(f"BEIR data saved to: {beir_dir}")
    print(f"   - corpus.jsonl: {len(corpus)} documents")
    print(f"   - queries.jsonl: {len(unique_queries)} queries")
    print(
        f"   - qrels/train.tsv: {len(train_df)} relevance labels ({len(train_query_ids)} queries)"
    )
    print(
        f"   - qrels/dev.tsv: {len(dev_df)} relevance labels ({len(dev_query_ids)} queries)"
    )
    print(
        f"   - qrels/test.tsv: {len(test_df)} relevance labels ({len(test_query_ids)} queries)"
    )


if __name__ == "__main__":
    preprocess_scifact()

from __future__ import annotations
from pathlib import Path
from typing import Dict, Set
import json
import pandas as pd
from Word2Vec_Baseline import Word2VecRetriever
from evaluation import traditional_eval


def load_queries_jsonl(path: Path) -> Dict[str, str]:
    q: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id", obj.get("id")))
            text = str(obj.get("text", "")).strip()
            if qid and text:
                q[qid] = text
    return q


def load_qrels_no_header(path: Path) -> Dict[str, Set[str]]:
    """
    qrels/*.tsv (NO header): query_id \t doc_id \t relevance
    -> {query_id: set(relevant_doc_ids)}
    """
    df = pd.read_csv(
        path, sep="\t", header=None, names=["query_id", "doc_id", "relevance"]
    )
    df = df[df["relevance"] > 0]

    gt: Dict[str, Set[str]] = {}
    for qid, group in df.groupby("query_id"):
        gt[str(qid)] = set(group["doc_id"].astype(str).tolist())
    return gt


def main():
    print("[1] script started")
    # Part A: paths
    project_root = Path(__file__).resolve().parents[1]
    beir_dir = project_root / "data" / "processed" / "beir_format"
    qrels_dir = beir_dir / "qrels"

    corpus_path = beir_dir / "corpus.jsonl"
    queries_path = beir_dir / "queries.jsonl"
    train_qrels = qrels_dir / "train.tsv"
    test_qrels = qrels_dir / "test.tsv"

    # Part B: load queries + test qrels
    print("[2] loading queries + test qrels...")
    queries = load_queries_jsonl(queries_path)
    gt_test = load_qrels_no_header(test_qrels)
    print(f"[2] queries loaded: {len(queries)}")
    print(f"[2] test qids with qrels: {len(gt_test)}")

    # Part C: build retriever (train on train.tsv)
    print("[3] building retriever (train word2vec on train qrels docs only)...")
    retriever = Word2VecRetriever(
        corpus_path=corpus_path,
        train_qrels_tsv_path=train_qrels,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,
        workers=4,
        model_path=beir_dir / "w2v_train_only.model",  # cache model
        rebuild_model=False,
    )
    print("[4] retriever ready")

    # Part D: build samples for evaluation.py
    k = 10
    samples = []
    missing_q = 0

    print("[5] retrieving & building samples...")
    for i, (qid, gt_docs) in enumerate(gt_test.items(), start=1):
        qtext = queries.get(str(qid))
        if not qtext:
            missing_q += 1
            continue

        top_docs = retriever.retrieve(qtext, top_k=k)

        samples.append(
            {
                "question": qtext,
                "contexts": top_docs,
                "ground_truth": gt_docs,
            }
        )

        if i % 200 == 0:
            print(f"[5] processed {i}/{len(gt_test)}")

    print(f"[5] done. missing query text: {missing_q}")

    # Part E: evaluate via evaluation.py
    print("[6] running evaluation via evaluation.py...")
    metrics = traditional_eval(samples, k=k)

    print("=== Word2Vec Retrieval Baseline (BEIR train/test) ===")
    for key, val in metrics.items():
        if key == "N":
            print(f"{key}: {int(val)}")
        else:
            print(f"{key}: {val:.4f}")

    # Part F: save results in unified format
    print("[7] saving results to JSON...")
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    unified_results = {
        "model_name": "Word2Vec Baseline",
        "model_type": "word2vec",
        "parameters": {
            "vector_size": 300,
            "window": 5,
            "min_count": 2,
            "sg": 1,
        },
        "metrics": {
            f"Hit@{k}": metrics.get(f"Hit@{k}", 0.0),
            f"Precision@{k}": metrics.get(f"Precision@{k}", 0.0),
            f"Recall@{k}": metrics.get(f"Recall@{k}", 0.0),
            "MRR": metrics.get("MRR", 0.0),
            f"MAP@{k}": metrics.get(f"MAP@{k}", 0.0),
            f"NDCG@{k}": metrics.get(f"NDCG@{k}", 0.0),
            "N": int(metrics.get("N", 0)),
        },
    }

    results_path = results_dir / "word2vec_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(unified_results, f, indent=2, ensure_ascii=False)

    print(f"[7] results saved to: {results_path}")


if __name__ == "__main__":
    main()

# scripts/run_word2vec_eval.py
# ============================================================
# Word2Vec retrieval baseline evaluation (BEIR-format)
# - Train Word2Vec ONLY on docs from qrels/train.tsv (no leakage)
# - Evaluate on queries in qrels/test.tsv
# - Use evaluation.traditional_eval (supports multi-ground-truth)
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set
import json
import pandas as pd

from Word2Vec_Baseline import Word2VecRetriever
from evaluation import traditional_eval


def load_queries_jsonl(path: Path) -> Dict[str, str]:
    """
    queries.jsonl: {"_id": "...", "text": "..."} (or "id")
    -> {query_id: query_text}
    """
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
    df = pd.read_csv(path, sep="\t", header=None, names=["query_id", "doc_id", "relevance"])
    df = df[df["relevance"] > 0]

    gt: Dict[str, Set[str]] = {}
    for qid, group in df.groupby("query_id"):
        gt[str(qid)] = set(group["doc_id"].astype(str).tolist())
    return gt


def main():
    print("[1] script started")

    # -------------------------
    # Part A: paths (your BEIR format)
    # -------------------------
    project_root = Path(r"D:\project\semantic-retrieval")
    beir_dir = project_root / "data" / "processed" / "beir_format"
    qrels_dir = beir_dir / "qrels"

    corpus_path = beir_dir / "corpus.jsonl"
    queries_path = beir_dir / "queries.jsonl"
    train_qrels = qrels_dir / "train.tsv"
    test_qrels  = qrels_dir / "test.tsv"

    # -------------------------
    # Part B: load queries + test qrels
    # -------------------------
    print("[2] loading queries + test qrels...")
    queries = load_queries_jsonl(queries_path)
    gt_test = load_qrels_no_header(test_qrels)
    print(f"[2] queries loaded: {len(queries)}")
    print(f"[2] test qids with qrels: {len(gt_test)}")

    # -------------------------
    # Part C: build retriever (train on train.tsv only)
    # -------------------------
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

    # -------------------------
    # Part D: build samples for evaluation.py
    # -------------------------
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

        samples.append({
            "question": qtext,
            "contexts": top_docs,     # List[str] ranked doc_ids
            "ground_truth": gt_docs,  # Set[str] multi relevant
        })

        if i % 200 == 0:
            print(f"[5] processed {i}/{len(gt_test)}")

    print(f"[5] done. missing query text: {missing_q}")

    # -------------------------
    # Part E: evaluate via evaluation.py
    # -------------------------
    print("[6] running evaluation via evaluation.py...")
    metrics = traditional_eval(samples, k=k)

    print("=== Word2Vec Retrieval Baseline (BEIR train/test) ===")
    for key, val in metrics.items():
        # N 是 float（我之前改成 float 了），打印时做兼容
        if key == "N":
            print(f"{key}: {int(val)}")
        else:
            print(f"{key}: {val:.4f}")


if __name__ == "__main__":
    main()

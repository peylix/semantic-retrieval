# scripts/run_word2vec_eval.py
# ============================================================
# Word2Vec retrieval baseline evaluation (BEIR format, train/test split)
# - Train Word2Vec ONLY on docs referenced by qrels/train.tsv (no leakage)
# - Evaluate on queries in qrels/test.tsv
# - Corpus: corpus.jsonl, Queries: queries.jsonl
# - Qrels: train.tsv/test.tsv (NO header): query_id \t doc_id \t relevance
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, List
import json
import math
import pandas as pd

from Word2Vec_Baseline import Word2VecRetriever
def load_queries(path: Path):
    queries = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj.get("_id", obj.get("id")))
            queries[qid] = obj.get("text", "")
    return queries


def load_qrels(path: Path):
    df = pd.read_csv(path, sep="\t", header=None,
                     names=["query_id", "doc_id", "rel"])
    df = df[df["rel"] > 0]
    gt = {}
    for qid, g in df.groupby("query_id"):
        gt[str(qid)] = set(g["doc_id"].astype(str))
    return gt


def eval_ranking(samples, k=10):
    def recall(gt, res): return len([r for r in res[:k] if r in gt]) / len(gt)
    def precision(gt, res): return len([r for r in res[:k] if r in gt]) / k
    def mrr(gt, res):
        for i, r in enumerate(res, 1):
            if r in gt:
                return 1 / i
        return 0.0
    def ndcg(gt, res):
        dcg = sum((1 if r in gt else 0) / math.log2(i+2)
                  for i, r in enumerate(res[:k]))
        idcg = sum(1 / math.log2(i+2) for i in range(min(len(gt), k)))
        return dcg / idcg if idcg > 0 else 0.0

    return {
        "Recall@10":   sum(recall(s["gt"], s["res"]) for s in samples) / len(samples),
        "Precision@10":sum(precision(s["gt"], s["res"]) for s in samples) / len(samples),
        "MRR":         sum(mrr(s["gt"], s["res"]) for s in samples) / len(samples),
        "NDCG@10":     sum(ndcg(s["gt"], s["res"]) for s in samples) / len(samples),
    }

def load_queries_jsonl(path: Path) -> Dict[str, str]:
    """
    queries.jsonl: {"_id": "...", "text": "..."} (or "id")
    -> {query_id: query_text}
    """
    queries: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("_id", obj.get("id")))
            qtext = str(obj.get("text", "")).strip()
            if qid and qtext:
                queries[qid] = qtext
    return queries


def load_qrels_tsv_no_header(path: Path) -> Dict[str, Set[str]]:
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


def eval_ranking(samples: List[dict], k: int = 10) -> Dict[str, float]:
    """
    Multi-relevant IR metrics:
      Recall@K, Precision@K, MRR, MAP@K, NDCG@K
    Each sample:
      {"contexts": [doc_id...], "ground_truth": set(doc_id...)}
    """
    def precision_at_k(gt: Set[str], results: List[str], k: int) -> float:
        hits = sum(1 for r in results[:k] if r in gt)
        return hits / k if k > 0 else 0.0

    def recall_at_k(gt: Set[str], results: List[str], k: int) -> float:
        if not gt:
            return 0.0
        hits = sum(1 for r in results[:k] if r in gt)
        return hits / len(gt)

    def mrr(gt: Set[str], results: List[str]) -> float:
        for idx, r in enumerate(results, start=1):
            if r in gt:
                return 1.0 / idx
        return 0.0

    def ap_at_k(gt: Set[str], results: List[str], k: int) -> float:
        if not gt:
            return 0.0
        hits = 0
        s = 0.0
        for idx, r in enumerate(results[:k], start=1):
            if r in gt:
                hits += 1
                s += hits / idx
        return s / min(len(gt), k)

    def ndcg_at_k(gt: Set[str], results: List[str], k: int) -> float:
        dcg = 0.0
        for idx, r in enumerate(results[:k], start=1):
            rel = 1.0 if r in gt else 0.0
            dcg += rel / math.log2(idx + 1)

        ideal = [1.0] * min(len(gt), k)
        idcg = 0.0
        for idx, rel in enumerate(ideal, start=1):
            idcg += rel / math.log2(idx + 1)

        return dcg / idcg if idcg > 0 else 0.0

    precisions, recalls, mrrs, aps, ndcgs = [], [], [], [], []
    for s in samples:
        gt = s["ground_truth"]
        results = s["contexts"]
        precisions.append(precision_at_k(gt, results, k))
        recalls.append(recall_at_k(gt, results, k))
        mrrs.append(mrr(gt, results))
        aps.append(ap_at_k(gt, results, k))
        ndcgs.append(ndcg_at_k(gt, results, k))

    n = len(samples) if samples else 1
    return {
        "Recall@K": sum(recalls) / n,
        "Precision@K": sum(precisions) / n,
        "MRR": sum(mrrs) / n,
        "MAP@K": sum(aps) / n,
        "NDCG@K": sum(ndcgs) / n,
    }


def main():
    print("[1] script started")

    # -------------------------
    # Part A: paths (use your fixed BEIR-format directory)
    # -------------------------
    project_root = Path(r"D:\project\semantic-retrieval")
    beir_dir = project_root / "data" / "processed" / "beir_format"
    qrels_dir = beir_dir / "qrels"

    corpus_path = beir_dir / "corpus.jsonl"
    queries_path = beir_dir / "queries.jsonl"
    train_qrels = qrels_dir / "train.tsv"
    test_qrels = qrels_dir / "test.tsv"

    # -------------------------
    # Part B: load queries + test qrels
    # -------------------------
    print("[2] loading queries + test qrels...")
    queries = load_queries_jsonl(queries_path)
    gt_test = load_qrels_tsv_no_header(test_qrels)

    print(f"[2] queries loaded: {len(queries)}")
    print(f"[2] test qids with qrels: {len(gt_test)}")

    # -------------------------
    # Part C: build retriever (train on train.tsv docs only)
    # -------------------------
    print("[3] building retriever (train word2vec on train qrels docs)...")
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
    # Part D: retrieve + build samples
    # -------------------------
    k = 10
    samples: List[dict] = []
    missing_q = 0

    print("[5] retrieving on test queries...")
    for i, (qid, gt_docs) in enumerate(gt_test.items(), start=1):
        qtext = queries.get(str(qid))
        if not qtext:
            missing_q += 1
            continue

        top_docs = retriever.retrieve(qtext, top_k=k)

        samples.append({
            "question_id": str(qid),
            "question": qtext,
            "contexts": top_docs,       # ranked doc_id list
            "ground_truth": gt_docs,    # set of relevant doc_ids
        })

        if i % 200 == 0:
            print(f"[5] processed {i}/{len(gt_test)}")

    print(f"[5] done. missing query text for {missing_q} qids.")

    # -------------------------
    # Part E: evaluate
    # -------------------------
    print("[6] running evaluation...")
    metrics = eval_ranking(samples, k=k)

    print("=== Word2Vec Retrieval Baseline (train=train.tsv, test=test.tsv) ===")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")


if __name__ == "__main__":
    main()

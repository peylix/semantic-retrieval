import math
from typing import List, Dict, Union, Set, Any

GroundTruth = Union[str, List[str], Set[str]]

def traditional_eval(samples: List[Dict[str, Any]], k: int = 10) -> Dict[str, float]:
    """
    Evaluation for retrieval with single or multi-ground-truth.

    samples: [
      {
        "question": str,
        "contexts": List[str],          # ranked top-K doc_ids
        "ground_truth": str | list | set  # gold doc_id(s)
      }
    ]
    """

    def to_set(gt: GroundTruth) -> Set[str]:
        if isinstance(gt, set):
            return set(str(x) for x in gt)
        if isinstance(gt, list):
            return set(str(x) for x in gt)
        # single string / single id
        return {str(gt)}

    def hit_at_k(gt_set: Set[str], results: List[str], k: int) -> float:
        return 1.0 if any(r in gt_set for r in results[:k]) else 0.0

    def precision_at_k(gt_set: Set[str], results: List[str], k: int) -> float:
        if k <= 0:
            return 0.0
        hits = sum(1 for r in results[:k] if r in gt_set)
        return hits / k

    def recall_at_k(gt_set: Set[str], results: List[str], k: int) -> float:
        if not gt_set:
            return 0.0
        hits = sum(1 for r in results[:k] if r in gt_set)
        return hits / len(gt_set)

    def mrr(gt_set: Set[str], results: List[str]) -> float:
        for rank, r in enumerate(results, start=1):
            if r in gt_set:
                return 1.0 / rank
        return 0.0

    def average_precision_at_k(gt_set: Set[str], results: List[str], k: int) -> float:
        """
        AP@K for multi-relevant:
          AP = (1 / min(|GT|, K)) * sum_{i=1..K} Precision@i * rel_i
        """
        if not gt_set:
            return 0.0

        hits = 0
        s = 0.0
        for i, r in enumerate(results[:k], start=1):
            if r in gt_set:
                hits += 1
                s += hits / i
        return s / min(len(gt_set), k)

    def ndcg_at_k(gt_set: Set[str], results: List[str], k: int) -> float:
        # binary relevance
        dcg = 0.0
        for i, r in enumerate(results[:k], start=1):
            rel = 1.0 if r in gt_set else 0.0
            dcg += rel / math.log2(i + 1)

        # ideal DCG: all relevant docs first
        ideal_rels = [1.0] * min(len(gt_set), k)
        idcg = 0.0
        for i, rel in enumerate(ideal_rels, start=1):
            idcg += rel / math.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    hits, precisions, recalls, mrrs, aps, ndcgs = [], [], [], [], [], []

    for s in samples:
        gt_set = to_set(s["ground_truth"])
        results = [str(x) for x in s["contexts"]]

        hits.append(hit_at_k(gt_set, results, k))
        precisions.append(precision_at_k(gt_set, results, k))
        recalls.append(recall_at_k(gt_set, results, k))
        mrrs.append(mrr(gt_set, results))
        aps.append(average_precision_at_k(gt_set, results, k))
        ndcgs.append(ndcg_at_k(gt_set, results, k))

    n = len(samples) if samples else 1
    return {
        f"Hit@{k}": sum(hits) / n,
        f"Precision@{k}": sum(precisions) / n,
        f"Recall@{k}": sum(recalls) / n,
        "MRR": sum(mrrs) / n,
        f"MAP@{k}": sum(aps) / n,
        f"NDCG@{k}": sum(ndcgs) / n,
        "N": float(len(samples)),
    }

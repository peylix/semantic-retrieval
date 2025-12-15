# # Sample form：
# # {
# #   "question": str,
# #   "contexts": List[str],
# #   "answer": str,
# #   "ground_truth": Optional[str],
# # }
# from typing import List, Dict, Optional
# from datasets import Dataset
# import math
# from typing import List, Dict, Optional
#
# def traditional_eval(samples: List[Dict], k: int = 10):
#
#     def precision_at_k(gt, results, k):
#         hits = sum(1 for r in results[:k] if r == gt)
#         return hits / k
#
#     def recall_at_k(gt, results, k):
#         hits = sum(1 for r in results[:k] if r == gt)
#         # ground truth 只有 1 个 → 分母 = 1
#         return hits * 1.0
#
#     def mrr(gt, results):
#         for idx, r in enumerate(results, start=1):
#             if r == gt:
#                 return 1.0 / idx
#         return 0.0
#
#     def average_precision(gt, results, k):
#         for idx, r in enumerate(results[:k], start=1):
#             if r == gt:
#                 return 1.0 / idx  # precision = hits/rank = 1/rank
#         return 0.0
#
#     def ndcg_at_k(gt, results, k):
#         dcg = 0.0
#         for idx, r in enumerate(results[:k], start=1):
#             rel = 1 if r == gt else 0
#             dcg += rel / math.log2(idx + 1)
#
#         # IDCG for a single relevant item is always 1/log2(1+1)
#         idcg = 1.0 / math.log2(1 + 1)
#         return dcg / idcg if idcg > 0 else 0.0
#
#     precisions, recalls, mrrs, aps, ndcgs = [], [], [], [], []
#
#     for s in samples:
#         gt = s["ground_truth"]
#         results = s["contexts"]          # Top-K 排序后的检索结果
#
#         precisions.append(precision_at_k(gt, results, k))
#         recalls.append(recall_at_k(gt, results, k))
#         mrrs.append(mrr(gt, results))
#         aps.append(average_precision(gt, results, k))
#         ndcgs.append(ndcg_at_k(gt, results, k))
#
#     return {
#         "Recall@K": sum(recalls) / len(recalls),
#         "Precision@K": sum(precisions) / len(precisions),
#         "MRR": sum(mrrs) / len(mrrs),
#         "MAP": sum(aps) / len(aps),
#         "NDCG@K": sum(ndcgs) / len(ndcgs),
#     }
# """
#     计算检索任务的排名指标：
#     Recall@K, Precision@K, MRR, MAP, NDCG@K
#
#     参数:
#     -------
#     samples: List[Dict]
#         每条样本格式:
#         {
#             "question": str,
#             "contexts": List[str],   # 排序后的 Top-K 检索结果
#             "ground_truth": str,     # 正确答案文本
#             ...
#         }
#     k: int
#         截断排名长度，默认10。
#
#     返回:
#     -------
#     metrics: Dict[str, float]
#         {
#             "Recall@K": 0.xxx,
#             "Precision@K": 0.xxx,
#             "MRR": 0.xxx,
#             "MAP": 0.xxx,
#             "NDCG@K": 0.xxx
#         }
#     """
#
import math
from typing import List, Dict

def traditional_eval(samples: List[Dict], k: int = 10) -> Dict[str, float]:
    """
    Evaluation for single-ground-truth retrieval.

    Each query has exactly one relevant document.

    samples: [
      {
        "question": str,
        "contexts": List[str],   # ranked top-K doc_ids
        "ground_truth": str      # single gold doc_id
      }
    ]
    """

    def hit_at_k(gt, results, k):
        return 1.0 if gt in results[:k] else 0.0

    def precision_at_k(gt, results, k):
        # single-gold: either 1 hit or 0
        return hit_at_k(gt, results, k) / k

    def recall_at_k(gt, results, k):
        # single-gold recall == hit@k
        return hit_at_k(gt, results, k)

    def mrr(gt, results):
        for rank, r in enumerate(results, start=1):
            if r == gt:
                return 1.0 / rank
        return 0.0

    def average_precision(gt, results, k):
        # single-gold: AP = 1 / rank if found in top-K
        for rank, r in enumerate(results[:k], start=1):
            if r == gt:
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(gt, results, k):
        for rank, r in enumerate(results[:k], start=1):
            if r == gt:
                return 1.0 / math.log2(rank + 1)
        return 0.0

    hits, precisions, recalls, mrrs, aps, ndcgs = [], [], [], [], [], []

    for s in samples:
        gt = s["ground_truth"]
        results = s["contexts"]

        hits.append(hit_at_k(gt, results, k))
        precisions.append(precision_at_k(gt, results, k))
        recalls.append(recall_at_k(gt, results, k))
        mrrs.append(mrr(gt, results))
        aps.append(average_precision(gt, results, k))
        ndcgs.append(ndcg_at_k(gt, results, k))

    n = len(samples)
    return {
        f"Hit@{k}": sum(hits) / n,
        f"Precision@{k}": sum(precisions) / n,
        f"Recall@{k}": sum(recalls) / n,
        "MRR": sum(mrrs) / n,
        f"MAP@{k}": sum(aps) / n,
        f"NDCG@{k}": sum(ndcgs) / n,
        "N": n,
    }
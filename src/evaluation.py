from typing import List, Dict, Optional
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
    ContextRelevance,
    faithfulness,
    answer_relevancy,
    answer_correctness
)
# Sample form：
# {
#   "question": str,
#   "contexts": List[str],
#   "answer": str,
#   "ground_truth": Optional[str],
# }
def Ragas_eval(
    samples: List[Dict],
    system_name: Optional[str] = None,
    add_simple_stats: bool = True
):
    # 1. Tranfor Python list[dict] to HuggingFace Dataset
    dataset = Dataset.from_list(samples)

    # 2. Use RAGAS
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[
            context_precision(),
            context_recall(),
            ContextRelevance(),
            faithfulness(),
            answer_relevancy(),
            answer_correctness()
        ],
    )

    # 3. Transfor to pandas DataFrame
    df = result.to_pandas()
    if system_name is not None:
        df["system"] = system_name

    return df
'''
from evaluation import run_ragas_eval

df_bm25 = run_ragas_eval(samples_bm25, system_name="bm25")
df_emb  = run_ragas_eval(samples_emb, system_name="embedding")

print(df_bm25.head())
print(df_emb.head())
'''


"""
    对一个检索+生成系统进行 RAGAS 评估。

    Parameters
    ----------
    samples : list[dict]
        每条样本必须包含:
        - "question": 用户问题
        - "contexts": 检索到的文档列表 (list[str])
        - "answer": 系统生成的回答
        - "ground_truth": 可选，有标准答案就填，没有可以省略

    system_name : str, optional
        用来标记是哪一个系统，比如 "bm25" 或 "embedding"，
        便于后面对比多个系统。

    Returns
    -------
    df : pandas.DataFrame
        含有 ragas 各项指标的表格。
"""

import math
from typing import List, Dict, Optional

def traditional_eval(samples: List[Dict], k: int = 10):

    def precision_at_k(gt, results, k):
        hits = sum(1 for r in results[:k] if r == gt)
        return hits / k

    def recall_at_k(gt, results, k):
        hits = sum(1 for r in results[:k] if r == gt)
        # ground truth 只有 1 个 → 分母 = 1
        return hits * 1.0

    def mrr(gt, results):
        for idx, r in enumerate(results, start=1):
            if r == gt:
                return 1.0 / idx
        return 0.0

    def average_precision(gt, results, k):
        for idx, r in enumerate(results[:k], start=1):
            if r == gt:
                return 1.0 / idx  # precision = hits/rank = 1/rank
        return 0.0

    def ndcg_at_k(gt, results, k):
        dcg = 0.0
        for idx, r in enumerate(results[:k], start=1):
            rel = 1 if r == gt else 0
            dcg += rel / math.log2(idx + 1)

        # IDCG for a single relevant item is always 1/log2(1+1)
        idcg = 1.0 / math.log2(1 + 1)
        return dcg / idcg if idcg > 0 else 0.0

    precisions, recalls, mrrs, aps, ndcgs = [], [], [], [], []

    for s in samples:
        gt = s["ground_truth"]
        results = s["contexts"]          # Top-K 排序后的检索结果

        precisions.append(precision_at_k(gt, results, k))
        recalls.append(recall_at_k(gt, results, k))
        mrrs.append(mrr(gt, results))
        aps.append(average_precision(gt, results, k))
        ndcgs.append(ndcg_at_k(gt, results, k))

    return {
        "Recall@K": sum(recalls) / len(recalls),
        "Precision@K": sum(precisions) / len(precisions),
        "MRR": sum(mrrs) / len(mrrs),
        "MAP": sum(aps) / len(aps),
        "NDCG@K": sum(ndcgs) / len(ndcgs),
    }
"""
    计算检索任务的排名指标：
    Recall@K, Precision@K, MRR, MAP, NDCG@K
    
    参数:
    -------
    samples: List[Dict]
        每条样本格式:
        {
            "question": str,
            "contexts": List[str],   # 排序后的 Top-K 检索结果
            "ground_truth": str,     # 正确答案文本
            ...
        }
    k: int
        截断排名长度，默认10。
        
    返回:
    -------
    metrics: Dict[str, float]
        {
            "Recall@K": 0.xxx,
            "Precision@K": 0.xxx,
            "MRR": 0.xxx,
            "MAP": 0.xxx,
            "NDCG@K": 0.xxx
        }
    """


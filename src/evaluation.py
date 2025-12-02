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
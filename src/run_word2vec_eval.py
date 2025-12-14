# scripts/run_word2vec_eval.py
# Run Word2Vec retrieval baseline evaluation on SciFact
# - 读取 preprocess 输出的 scifact_pairs.csv / scifact_corpus.csv
# - 对每个 query 取 top-k doc_id
# - 构造 samples 给 traditional_eval 评估

from __future__ import annotations

from pathlib import Path
import pandas as pd

from Word2Vec_Baseline import Word2VecRetriever   # ✅ 改这里
from evaluation import traditional_eval

def main():
    print("[1] script started")
    # Part A: 定位数据路径
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed"

    corpus_csv = processed_dir / "scifact_corpus.csv"
    pairs_csv  = processed_dir / "scifact_pairs.csv"

    # Part B: 初始化 Word2Vec Retriever
    print("[2] building retriever (training word2vec + doc embeddings)...")

    retriever = Word2VecRetriever(
        corpus_csv_path=corpus_csv,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,
        workers=4,
    )
    print("[3] retriever ready")

    # Part C: 构造评测 samples
    print("[4] loading pairs csv")

    df_pairs = pd.read_csv(pairs_csv)

    k = 10
    samples = []
    for _, row in df_pairs.iterrows():
        query = str(row["query"])
        gt_doc_id = str(row["doc_id"])

        top_docs = retriever.retrieve(query, top_k=k)

        samples.append({
            "question": query,
            "contexts": top_docs,       # 这里存 doc_id list（排名顺序）
            "ground_truth": gt_doc_id,  # 单个正确 doc_id
        })

    # Part D: 运行评估
    print("[5] running evaluation")

    metrics = traditional_eval(samples, k=k)
    print("=== Word2Vec Retrieval Baseline ===")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")


if __name__ == "__main__":
    main()

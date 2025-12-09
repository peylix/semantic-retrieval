from pathlib import Path
import pandas as pd


def preprocess_scifact():
    """
    读取 data/raw/scifact_raw.csv，做简单清洗，生成：
      - scifact_pairs.csv  : 每行一个 (query, doc) 对
      - scifact_corpus.csv : 去重后的文档库（doc_id, content）
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "scifact_raw.csv"
    df = pd.read_csv(raw_path)

    # 只保留我们关心的列
    cols = ["_id", "title", "text", "query"]
    df = df[cols].copy()

    # 去掉没有 text 或没有 query 的样本
    df = df.dropna(subset=["text", "query"])

    # 改名，方便后面使用
    df = df.rename(columns={"_id": "doc_id"})

    # 简单清洗一下字符串
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["query"] = df["query"].astype(str).str.strip()

    # 拼接一个 content 字段：后面 BM25 和 embedding 都用它当文档内容
    df["content"] = df["title"] + ". " + df["text"]

    # 给每条 (query, doc) 对一个 query_id
    df = df.reset_index(drop=True)
    df["query_id"] = df.index

    # 1️⃣ 保存 pairs：后面训练 dense retriever / 做评估都用这个
    pairs_path = processed_dir / "scifact_pairs.csv"
    df.to_csv(pairs_path, index=False)

    # 2️⃣ 保存去重后的 corpus：每个 doc_id 一条，用来建索引 / 向量库
    corpus = df[["doc_id", "content"]].drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    corpus_path = processed_dir / "scifact_corpus.csv"
    corpus.to_csv(corpus_path, index=False)


if __name__ == "__main__":
    preprocess_scifact()


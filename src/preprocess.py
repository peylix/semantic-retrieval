from pathlib import Path
import pandas as pd


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    对一个 split 做简单预处理：
    - 保留 sentence1, sentence2, score
    - 去掉缺失行
    - score 转为 float，并加一个归一化列 score_norm (0~1)
    - 去掉前后空格
    """
    # 只保留核心列
    cols = ["sentence1", "sentence2", "score"]
    df = df[cols].dropna().copy()

    # 处理评分
    df["score"] = df["score"].astype(float)
    df["score_norm"] = df["score"] / 5.0

    # 清理文本（去掉首尾空格）
    df["sentence1"] = df["sentence1"].astype(str).str.strip()
    df["sentence2"] = df["sentence2"].astype(str).str.strip()

    return df


def main():
    # 项目根目录
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        raw_path = raw_dir / f"sts_{split}_raw.csv"
        if not raw_path.exists():
            continue

        df_raw = pd.read_csv(raw_path)
        df_clean = preprocess_split(df_raw)

        out_path = processed_dir / f"sts_{split}_clean.csv"
        df_clean.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

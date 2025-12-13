# src/data_loader.py

from pathlib import Path
import pandas as pd
from datasets import load_dataset


def load_scifact_raw():
    """
    从 HuggingFace 加载 BeIR/scifact-generated-queries，
    并把包含 _id, title, text, query 的整张表保存到 data/raw。
    """
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集（默认只有一个 split：train）
    ds = load_dataset("BeIR/scifact-generated-queries")
    df = ds["train"].to_pandas()

    # 保存原始表
    raw_path = raw_dir / "scifact_raw.csv"
    df.to_csv(raw_path, index=False)


if __name__ == "__main__":
    load_scifact_raw()

#
# from pathlib import Path
# import pandas as pd
# import json
# from datasets import load_dataset
#
#
# def load_ms_marco_raw(split: str = "train"):
#     """
#     从 HuggingFace 加载 MS MARCO v2.1 的一个 split，
#     并将原始数据保存到 data/raw/msmarco_<split>_raw.csv
#
#     注意：passages 列用 json.dumps 存成字符串，
#     方便后续用 json.loads 恢复。
#     """
#     project_root = Path(__file__).resolve().parents[1]
#     raw_dir = project_root / "data" / "raw"
#     raw_dir.mkdir(parents=True, exist_ok=True)
#
#     ds = load_dataset("microsoft/ms_marco", "v2.1")
#
#     if split not in ds:
#         raise ValueError(f"Split '{split}' not found in ms_marco v2.1 dataset.")
#
#     df = ds[split].to_pandas()
#
#     # 把 passages 列（原本是 dict）转成 JSON 字符串再存
#     df["passages"] = df["passages"].apply(json.dumps)
#
#     out_path = raw_dir / f"msmarco_{split}_raw.csv"
#     df.to_csv(out_path, index=False)
#
#
# if __name__ == "__main__":
#     load_ms_marco_raw(split="train")

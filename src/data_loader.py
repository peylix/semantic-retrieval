# src/data_loader.py

from pathlib import Path

import pandas as pd
from datasets import load_dataset


def save_stsbenchmark_raw():
    """
    Download STSBenchmark (mteb/stsbenchmark-sts) from HuggingFace
    and save train/validation/test splits as raw CSV files
    under data/raw/.
    """
    # Project root = src/ 的上一级目录
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Raw data directory: {raw_dir}")

    # 1. 加载数据集
    print("[INFO] Loading dataset: mteb/stsbenchmark-sts ...")
    ds = load_dataset("mteb/stsbenchmark-sts")

    # 2. 保存各个 split
    for split_name in ["train", "validation", "test"]:
        if split_name not in ds:
            print(f"[WARN] Split '{split_name}' not found in dataset, skipping.")
            continue

        df = pd.DataFrame(ds[split_name])
        output_path = raw_dir / f"sts_{split_name}_raw.csv"
        df.to_csv(output_path, index=False)
        print(f"[INFO] Saved {split_name} split to: {output_path}")


if __name__ == "__main__":
    save_stsbenchmark_raw()

"""Dataset utilities for hard negative fine-tuning."""

from __future__ import annotations

from typing import Iterator, Sequence

from sentence_transformers import InputExample
from torch.utils.data import Dataset


class InputExampleDataset(Dataset[InputExample]):
    """A thin Dataset wrapper around a list of InputExample objects."""

    def __init__(self, examples: Sequence[InputExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]

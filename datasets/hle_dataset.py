"""Humanity's Last Exam (HLE) — cais/hle benchmark (multiple-choice only)."""

import json
import re
from typing import List
from .base_dataset import BaseDataset, DataSample


class HLEDataset(BaseDataset):
    """HLE benchmark — filters to multipleChoice questions only."""

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "hle"

    def _load(self) -> List[DataSample]:
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("cais/hle", split=self.split)
        except Exception:
            if self.data_path:
                return self._load_from_file(self.data_path)
            return []

        samples = []
        for item in ds:
            if item.get("answer_type") != "multipleChoice":
                continue
            samples.append(DataSample(
                task_id=str(item.get("id", len(samples))),
                task=item.get("question", ""),
                ground_truth=item.get("answer", ""),
                metadata={
                    "answer_type": "multipleChoice",
                    "category": item.get("category", ""),
                    "raw_subject": item.get("raw_subject", ""),
                },
                domain=self.domain,
            ))
        return samples

    def _load_from_file(self, path: str) -> List[DataSample]:
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("id", str(len(samples))),
                    task=item.get("question", ""),
                    ground_truth=item.get("answer", ""),
                    metadata=item,
                    domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction: str, ground_truth: str) -> float:
        pred = re.search(r"\b([A-Z])\b", prediction.upper())
        pred_letter = pred.group(1) if pred else prediction.strip()[:1].upper()
        return 1.0 if pred_letter == ground_truth.strip().upper() else 0.0

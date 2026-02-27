"""Frontier Science benchmark from OpenAI."""

import json
from .base_dataset import BaseDataset, DataSample


class FrontierScienceDataset(BaseDataset):
    """openai/frontierscience"""

    def __init__(self, split="test", limit=None, data_path=None):
        super().__init__(split, limit, data_path)
        self.domain = "gaia"

    def _load(self):
        if self.data_path:
            return self._load_from_file(self.data_path)
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("openai/frontierscience", split="test")
            return [DataSample(
                task_id=str(i),
                task=item["problem"],
                ground_truth=item["answer"],
                metadata={"subject": item.get("subject", ""), "task_group_id": item.get("task_group_id", "")},
                domain=self.domain,
            ) for i, item in enumerate(ds)]
        except Exception as e:
            print(f"[frontierscience] HF load failed: {e}")
            return []

    def _load_from_file(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("id", str(len(samples))),
                    task=item.get("problem", item.get("question", "")),
                    ground_truth=item.get("answer", ""),
                    metadata=item, domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction, ground_truth):
        p, t = prediction.strip().lower(), ground_truth.strip().lower()
        if p == t:
            return 1.0
        if t in p or p in t:
            return 0.5
        return 0.0

import json
from .base_dataset import BaseDataset, DataSample


class GAIADataset(BaseDataset):
    def __init__(self, split="test", limit=None, data_path=None, level=None):
        super().__init__(split, limit, data_path)
        self.domain = "gaia"
        self.level = level

    def _load(self):
        try:
            from .hf_loader import load_hf_dataset
            ds = load_hf_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
            samples = []
            for item in ds:
                if self.level and item.get("Level", 0) != self.level:
                    continue
                samples.append(DataSample(
                    task_id=str(item.get("task_id", len(samples))),
                    task=item.get("Question", ""),
                    ground_truth=item.get("Final answer", ""),
                    metadata={"level": item.get("Level", 0)},
                    domain=self.domain,
                ))
            return samples
        except Exception:
            return self._load_from_file(self.data_path) if self.data_path else []

    def _load_from_file(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                samples.append(DataSample(
                    task_id=item.get("task_id", str(len(samples))),
                    task=item.get("Question", item.get("question", "")),
                    ground_truth=item.get("Final answer", item.get("answer", "")),
                    metadata=item, domain=self.domain,
                ))
        return samples

    def evaluate(self, prediction, ground_truth):
        p, t = prediction.strip().lower(), ground_truth.strip().lower()
        if p == t:
            return 1.0
        return 0.5 if (t in p or p in t) else 0.0

from .base_dataset import BaseDataset, DataSample
from .humaneval_dataset import HumanEvalDataset
from .mmlu_dataset import MMLUProDataset
from .math_dataset import (
    AIME2024Dataset, AIME2025Dataset, AIME2026Dataset,
    BeyondAIMEDataset, HMMT2025Dataset,
)
from .livecodebench_dataset import LiveCodeBenchDataset, LiveCodeBenchTestGenDataset
from .gaia_dataset import GAIADataset
from .frontierscience_dataset import FrontierScienceDataset
from .hle_dataset import HLEDataset
from .tasks import tasks

DATASET_REGISTRY = {
    # Coding
    "humaneval": HumanEvalDataset,
    "livecodebench": LiveCodeBenchDataset,
    "livecodebench_testgen": LiveCodeBenchTestGenDataset,
    # Agentic
    "mmlu_pro": MMLUProDataset,
    "gaia": GAIADataset,
    "frontierscience": FrontierScienceDataset,
    # Math
    "aime_2024": AIME2024Dataset,
    "aime_2025": AIME2025Dataset,
    "aime_2026": AIME2026Dataset,
    "beyond_aime": BeyondAIMEDataset,
    "hmmt_2025": HMMT2025Dataset,
    "hle": HLEDataset,
}


def get_dataset(name: str, **kwargs) -> BaseDataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)

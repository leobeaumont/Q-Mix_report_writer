"""
Report evaluation & reward composition (training_eval plan Stage 2).

`ReportEvaluator` is the object the QMIX controller and runner consume:
grounded per-chunk scoring (audit + claim check against the section's stored
RAG sources) and a terminal whole-report macro judge. `reward` holds the
pure composition math (event/terminal/token terms, length gaussian).
"""

from .judges import ChunkScore, JudgeError, MacroScore, ReportEvaluator
from .reward import (
    compose_event_reward,
    compose_terminal_reward,
    length_gaussian,
    token_penalty,
)

__all__ = [
    "ChunkScore",
    "JudgeError",
    "MacroScore",
    "ReportEvaluator",
    "compose_event_reward",
    "compose_terminal_reward",
    "length_gaussian",
    "token_penalty",
]

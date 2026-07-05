"""
Reward composition v2 (training_eval plan 2.4 — decisions TD1/TD2).

Pure math, config-driven; the single home of every reward term:

    per append event (OD-A: lands on the event step only):
        r_event = quality_weight * chunk_score + length_weight * delta_length_gauss
    at run end (terminal, added to the final step):
        r_terminal = macro_weight * macro_score
    per recorded step (TD2 — flag-gated, token_weight defaults to 0.0):
        penalty = token_weight * tokens / 10_000

`cfg` is the `reward:` config mapping (see configs/default.yaml).
"""

from __future__ import annotations

import math


def length_gaussian(length: float, goal: float, sigma: float) -> float:
    """Gaussian length score in [0, 1], peaked at the length goal."""
    return float(math.exp(-0.5 * ((length - goal) / sigma) ** 2))


def compose_event_reward(chunk_score: float, delta_length_gauss: float, cfg: dict) -> float:
    """Reward for one successful append (TD1: absolute grounded chunk score)."""
    return (
        float(cfg.get("quality_weight", 1.0)) * float(chunk_score)
        + float(cfg.get("length_weight", 0.1)) * float(delta_length_gauss)
    )


def compose_terminal_reward(macro_score: float, cfg: dict) -> float:
    """Terminal reward from the whole-report macro judge (TD1)."""
    return float(cfg.get("macro_weight", 1.0)) * float(macro_score)


def token_penalty(tokens: float, cfg: dict) -> float:
    """Per-step token cost (TD2). token_weight defaults to 0.0 = disabled."""
    return float(cfg.get("token_weight", 0.0)) * (float(tokens) / 10_000.0)

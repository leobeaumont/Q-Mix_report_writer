"""
Phase state singleton — tracks which phase and round the pipeline is in.

Agents and the prompt set read from this to tailor their behaviour to the
current stage of the pipeline without needing it injected through every call.
"""

from __future__ import annotations
from typing import List, Optional


class PhaseState:
    """Singleton that tracks the active pipeline phase and round counter."""

    _instance: Optional["PhaseState"] = None

    @classmethod
    def instance(cls) -> "PhaseState":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        # Import here to avoid circular dependency at module load time
        from handcrafted_graph.phases import PhaseType
        self._current_phase: PhaseType = PhaseType.PLANNING
        self._round_in_phase: int = 0
        self._phase_history: List[PhaseType] = []

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_phase(self):
        return self._current_phase

    @property
    def round_in_phase(self) -> int:
        return self._round_in_phase

    @property
    def phase_history(self):
        return list(self._phase_history)

    # ------------------------------------------------------------------
    # Mutators (called by HandcraftedGraph only)
    # ------------------------------------------------------------------

    def set_phase(self, phase) -> None:
        if self._current_phase != phase:
            self._phase_history.append(self._current_phase)
            self._current_phase = phase
            self._round_in_phase = 0

    def increment_round(self) -> None:
        self._round_in_phase += 1

    def reset(self) -> None:
        from handcrafted_graph.phases import PhaseType
        self._current_phase = PhaseType.PLANNING
        self._round_in_phase = 0
        self._phase_history = []

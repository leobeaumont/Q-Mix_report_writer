"""
Phase definitions for the handcrafted communication graph.

Each phase describes:
  - Which agents execute (required vs. optional) in each round pattern
  - The spatial edge topology for that round (who sends output to whom)
  - How many rounds the phase can run before forcing a transition

Round patterns cycle: if a phase has 2 patterns and max_rounds=6, the patterns
alternate [0, 1, 0, 1, 0, 1] up to the max.

Edge tuples use agent names as keys. "Collector" always refers to the Collector
node regardless of its runtime id.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class PhaseType(Enum):
    PLANNING = "planning"
    RESEARCH = "research"
    DRAFTING = "drafting"
    REVIEW = "review"
    REVISION = "revision"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoundTopology:
    """Communication topology for a single round.

    Attributes:
        required_agents: Always execute regardless of context.
        optional_agents: Execute only when the scheduler deems relevant
                         (e.g., they have prior outputs or pass a skip-check).
        edges: Directed message edges as (sender_name, receiver_name) pairs.
               An edge means the sender's output is forwarded to the receiver's
               spatial_info before the receiver executes. Sending to "Collector"
               triggers the report-append flow.
    """
    required_agents: List[str]
    optional_agents: List[str]
    edges: List[Tuple[str, str]]


@dataclass
class PhaseConfig:
    """Full definition of a pipeline phase.

    Attributes:
        name: Unique phase identifier.
        description: Human-readable description used in logs and prompts.
        round_topologies: Ordered list of RoundTopology patterns. The last
                          pattern repeats when round_idx >= len(patterns).
        max_rounds: Hard upper bound on rounds before forced transition.
        next_phase: Which phase follows; None means end of pipeline.
    """
    name: PhaseType
    description: str
    round_topologies: List[RoundTopology]
    max_rounds: int
    next_phase: Optional[PhaseType] = None


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PLANNING_PHASE = PhaseConfig(
    name=PhaseType.PLANNING,
    description=(
        "Evidence-first outline: Researcher scans the knowledge base for topic coverage "
        "before LeadArchitect commits to any section titles. "
        "The outline is grounded only in topics the corpus actually contains."
    ),
    max_rounds=2,
    next_phase=PhaseType.RESEARCH,
    round_topologies=[
        # Round 0: Researcher runs first (required) and reports corpus coverage to
        # LeadArchitect. Topological sort guarantees Researcher executes before
        # LeadArchitect because of the (Researcher → LeadArchitect) edge.
        RoundTopology(
            required_agents=["Researcher", "LeadArchitect"],
            optional_agents=[],
            edges=[
                ("Researcher", "LeadArchitect"),   # Coverage report informs outline
            ],
        ),
        # Round 1: LeadArchitect refines the outline or confirms the first target.
        # Researcher may re-run if it has prior output (follow-up query).
        RoundTopology(
            required_agents=["LeadArchitect"],
            optional_agents=["Researcher"],
            edges=[
                ("Researcher", "LeadArchitect"),   # Optional follow-up evidence
            ],
        ),
    ],
)


RESEARCH_PHASE = PhaseConfig(
    name=PhaseType.RESEARCH,
    description=(
        "Iterative evidence gathering. LeadArchitect specifies research targets; "
        "Researcher queries the RAG knowledge base and returns evidence atoms; "
        "DataAnalyst synthesizes raw evidence into logical blueprints for writing."
    ),
    max_rounds=6,
    next_phase=PhaseType.DRAFTING,
    round_topologies=[
        # Round A: LeadArchitect directs Researcher with a specific query target.
        RoundTopology(
            required_agents=["LeadArchitect", "Researcher"],
            optional_agents=[],
            edges=[
                ("LeadArchitect", "Researcher"),   # Research directive
            ],
        ),
        # Round B: Researcher feeds evidence to DataAnalyst and reports back to
        # LeadArchitect so State Deficiency signals can update the next query.
        RoundTopology(
            required_agents=["LeadArchitect", "Researcher", "DataAnalyst"],
            optional_agents=[],
            edges=[
                ("Researcher", "LeadArchitect"),   # State Deficiency / evidence summary
                ("LeadArchitect", "DataAnalyst"),  # Structural framing
                ("Researcher", "DataAnalyst"),     # Raw evidence atoms
            ],
        ),
    ],
)


DRAFTING_PHASE = PhaseConfig(
    name=PhaseType.DRAFTING,
    description=(
        "Incremental section writing. LeadArchitect specifies the current section; "
        "DataAnalyst structures the evidence for it; Researcher fills any gaps; "
        "Collector writes and appends the polished section to the report."
    ),
    max_rounds=10,
    next_phase=PhaseType.REVIEW,
    round_topologies=[
        # Round A: LeadArchitect designates the section, Researcher fetches fresh RAG
        # context, DataAnalyst structures content. Researcher is required so DataAnalyst
        # always has a RAG anchor before synthesising — prevents fabricated numbers.
        RoundTopology(
            required_agents=["LeadArchitect", "Researcher", "DataAnalyst"],
            optional_agents=[],
            edges=[
                ("LeadArchitect", "DataAnalyst"),  # Section directive
                ("Researcher", "DataAnalyst"),     # RAG evidence for this section
            ],
        ),
        # Round B: DataAnalyst (+ optional Researcher top-up) feeds Collector to write.
        RoundTopology(
            required_agents=["DataAnalyst", "Collector"],
            optional_agents=["Researcher"],
            edges=[
                ("Researcher", "DataAnalyst"),     # Optional extra citations
                ("DataAnalyst", "Collector"),      # Structured content for writing
            ],
        ),
    ],
)


REVIEW_PHASE = PhaseConfig(
    name=PhaseType.REVIEW,
    description=(
        "Reviewer audits the full report draft for factual accuracy, logical "
        "coherence, and scientific rigour. Feedback is forwarded to LeadArchitect "
        "to determine whether a revision pass is warranted."
    ),
    max_rounds=2,
    next_phase=PhaseType.REVISION,
    round_topologies=[
        # Round 0: Reviewer reads the full report and generates structured feedback.
        RoundTopology(
            required_agents=["Reviewer"],
            optional_agents=[],
            edges=[],  # Reviewer output is stored in temporal memory for next round
        ),
        # Round 1: Reviewer forwards critique to LeadArchitect.
        RoundTopology(
            required_agents=["Reviewer", "LeadArchitect"],
            optional_agents=[],
            edges=[
                ("Reviewer", "LeadArchitect"),     # Structured critique
            ],
        ),
    ],
)


REVISION_PHASE = PhaseConfig(
    name=PhaseType.REVISION,
    description=(
        "LeadArchitect applies reviewer feedback. DataAnalyst prepares corrected "
        "content and flags evidence gaps; Researcher retrieves missing evidence "
        "when gaps are flagged; Collector rewrites the corrected section in-place."
    ),
    max_rounds=4,
    next_phase=None,  # End of pipeline
    round_topologies=[
        # Round A: LeadArchitect issues correction directive; DataAnalyst runs first
        # and may flag evidence gaps; Researcher runs after (topological order enforced
        # by the DataAnalyst→Researcher edge) and can read those gap flags directly.
        RoundTopology(
            required_agents=["LeadArchitect", "DataAnalyst"],
            optional_agents=["Researcher"],
            edges=[
                ("LeadArchitect", "DataAnalyst"),  # Correction directive
                ("DataAnalyst", "Researcher"),     # Gap flags → Researcher reacts
            ],
        ),
        # Round B: Researcher forwards retrieved evidence to DataAnalyst for final
        # synthesis; Collector rewrites the flagged section in-place.
        RoundTopology(
            required_agents=["DataAnalyst", "Collector"],
            optional_agents=["Researcher"],
            edges=[
                ("Researcher", "DataAnalyst"),     # Retrieved evidence (if any gaps)
                ("DataAnalyst", "Collector"),      # Corrected final content
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Ordered pipeline and lookup map
# ---------------------------------------------------------------------------

PHASE_SEQUENCE: List[PhaseConfig] = [
    PLANNING_PHASE,
    RESEARCH_PHASE,
    DRAFTING_PHASE,
    REVIEW_PHASE,
    REVISION_PHASE,
]

PHASE_MAP: dict[PhaseType, PhaseConfig] = {p.name: p for p in PHASE_SEQUENCE}

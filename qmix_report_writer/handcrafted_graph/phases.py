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
    SECTION_REVIEW = "section_review"
    VALIDATION = "validation"


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
                    For section_aware phases this is rounds *per section*
                    (review round + optional revision round = 2).
        next_phase: Which phase follows; None means end of pipeline.
        section_aware: When True, HandcraftedGraph delegates execution to
                       _execute_section_aware_phase which iterates
                       ReportState.sections in order rather than running a
                       fixed number of rounds.
        window_aware: When True, HandcraftedGraph delegates execution to
                      _execute_window_aware_phase which slides a window of
                      window_size chars (with window_overlap_sections overlap)
                      over ReportState.sections, then runs one synthesis round.
        window_size: Maximum characters per window (window_aware phases only).
        window_overlap_sections: Number of sections shared between consecutive
                      windows to preserve cross-boundary context.
    """
    name: PhaseType
    description: str
    round_topologies: List[RoundTopology]
    max_rounds: int
    next_phase: Optional[PhaseType] = None
    section_aware: bool = False
    window_aware: bool = False
    window_size: int = 6000
    window_overlap_sections: int = 1


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
    max_rounds=20,  # estimate only; patched at runtime to 2 × len(planned_sections)
    next_phase=PhaseType.SECTION_REVIEW,
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


SECTION_REVIEW_PHASE = PhaseConfig(
    name=PhaseType.SECTION_REVIEW,
    description=(
        "Per-section review and targeted revision. The Reviewer audits one section "
        "at a time; DataAnalyst and Collector apply corrections in-place. Sections "
        "that pass review are skipped without a revision round."
    ),
    section_aware=True,
    max_rounds=2,  # rounds per section: 1 review + 1 optional revision
    next_phase=PhaseType.VALIDATION,
    round_topologies=[
        # Round A (review): Reviewer reads the single section and produces
        # structured feedback, or outputs [NO_REVISION_NEEDED] if the section
        # is already correct. Output is stored in temporal memory.
        RoundTopology(
            required_agents=["Reviewer"],
            optional_agents=[],
            edges=[],
        ),
        # Round B (revision): the Reviewer does NOT re-execute here. The graph
        # forwards the review round's stored critique to DataAnalyst directly
        # (see _execute_section_aware_phase) — re-running the Reviewer would cost
        # a full re-review and could produce a different critique than the one
        # that triggered this revision. Collector rewrites the section in-place.
        RoundTopology(
            required_agents=["DataAnalyst", "Collector"],
            optional_agents=["Researcher"],
            edges=[
                ("Researcher", "DataAnalyst"),  # Evidence gap fill
                ("DataAnalyst", "Collector"),   # Corrected content
            ],
        ),
    ],
)


VALIDATION_PHASE = PhaseConfig(
    name=PhaseType.VALIDATION,
    description=(
        "Global quality check using a sliding window over the report sections. "
        "Each window pass has the Reviewer audit a subset of adjacent sections for "
        "cross-section issues (transitions, flow, duplications). A final synthesis "
        "round consolidates all window findings into a quality conclusion."
    ),
    window_aware=True,
    window_size=6000,
    window_overlap_sections=1,
    max_rounds=2,  # informational only; actual rounds = n_windows + 1 (synthesis)
    next_phase=None,  # End of pipeline
    round_topologies=[
        # Index 0 (window review): Reviewer audits one window of sections.
        RoundTopology(
            required_agents=["Reviewer"],
            optional_agents=[],
            edges=[],
        ),
        # Index 1 (synthesis): Reviewer + LeadArchitect see all accumulated
        # window notes and produce a consolidated quality conclusion.
        RoundTopology(
            required_agents=["Reviewer", "LeadArchitect"],
            optional_agents=[],
            edges=[
                ("Reviewer", "LeadArchitect"),  # Synthesised quality notes
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
    SECTION_REVIEW_PHASE,
    VALIDATION_PHASE,
]

PHASE_MAP: dict[PhaseType, PhaseConfig] = {p.name: p for p in PHASE_SEQUENCE}

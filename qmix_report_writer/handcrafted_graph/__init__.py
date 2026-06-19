"""
Handcrafted Communication Graph — baseline alternative to the Q-Mixer.

This module implements a deterministic, phase-based multi-agent pipeline where
the communication topology is hand-designed rather than learned. It serves as a
comparison baseline for the QMIX-driven graph.

Pipeline phases (in order):
  1. PLANNING  — LeadArchitect structures the report outline
  2. RESEARCH  — Researcher queries RAG; DataAnalyst synthesizes evidence
  3. DRAFTING  — DataAnalyst + Collector write each section incrementally
  4. REVIEW    — Reviewer audits the full draft
  5. REVISION  — LeadArchitect + Collector apply corrections (conditional)

Agents can skip rounds they have nothing to contribute to, avoiding wasted
inference calls. The scheduler enforces required vs. optional participation per
round.

Entry point:
  from qmix_report_writer.handcrafted_graph.runner import run_handcrafted
  answers, tokens = await run_handcrafted(task="Write a report on X")
"""

from qmix_report_writer.handcrafted_graph.graph import HandcraftedGraph
from qmix_report_writer.handcrafted_graph.runner import run_handcrafted

__all__ = ["HandcraftedGraph", "run_handcrafted"]

"""
Deterministic post-processing filter for the final report text.

Removes sentences that contain pipeline-internal meta-commentary — phrases
like "requery", "data atoms", "RAG Tool", etc. — that should never appear
in a scientific report but can leak through when a small model narrates the
evidence-gathering process instead of writing domain content.

No LLM call is made. The filter is conservative: it only removes sentences
that match unambiguous pipeline-internal patterns. Legitimate scientific prose
is left untouched.

Usage:
    from utils.report_filter import filter_meta_commentary
    clean_report = filter_meta_commentary(raw_report)
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each pattern targets a phrase that is specific to the pipeline's internal
# vocabulary and would never appear in a genuine scientific report.
_META_PATTERNS: List[re.Pattern] = [
    re.compile(r'\brequer(?:y|ied|ying|ies)\b', re.IGNORECASE),        # requery / requeried
    re.compile(r'\bdata atoms?\b', re.IGNORECASE),                      # data atom(s)
    re.compile(r'\bevidence atoms?\b', re.IGNORECASE),                  # evidence atom(s)
    re.compile(r'\batomic data\b', re.IGNORECASE),                      # atomic data (pipeline term)
    re.compile(r'\bState Deficiency\b', re.IGNORECASE),                 # pipeline gap marker
    re.compile(r'\[SECTION_ID:', re.IGNORECASE),                        # section routing tag
    re.compile(r'\bRAG\s+(?:Tool|results?|data|query|queries)\b', re.IGNORECASE),  # RAG Tool / RAG results
    re.compile(r'\byields?\s+no\s+atomic\b', re.IGNORECASE),            # "yields no atomic"
    re.compile(r'\bevidentiary\s+scope\b', re.IGNORECASE),              # pipeline jargon
    re.compile(r'\bspatial_info\b', re.IGNORECASE),                     # internal variable name
    re.compile(r'\btemporal_info\b', re.IGNORECASE),                    # internal variable name
    re.compile(r'\bcalling_agent\b', re.IGNORECASE),                    # internal variable name
    re.compile(r'\bexecution\s+trace\b', re.IGNORECASE),                # internal term
]

# Section-transition sentences: forward references to "the next/following/subsequent section".
# These are inserted by the model when writing one section at a time, but they make no sense
# in the final assembled document where sections flow directly into each other.
_TRANSITION_PATTERNS: List[re.Pattern] = [
    re.compile(r'\b[Tt]he\s+(?:next|following|subsequent)\s+section\b'),
    re.compile(r'\b[Tt]he\s+(?:next|following|subsequent)\s+chapter\b'),
    re.compile(r'\b[Ii]n\s+the\s+(?:next|following|subsequent)\s+section\b'),
    re.compile(r'\b[Tt]he\s+(?:next|following|subsequent)\s+part\b'),
    re.compile(r'\b(?:will|shall)\s+(?:be\s+)?(?:address|discuss|explore|present|detail|cover|examine|describe|introduce|outline)(?:ed|s)?\b.*\bsection\b'),
]

# Sentence boundary: period / ! / ? followed by whitespace and an uppercase letter.
# Intentionally simple — avoids over-splitting on abbreviations like "Fig." while
# still catching the vast majority of sentence boundaries in academic prose.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z\"“])')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_meta_commentary(text: str) -> str:
    """Remove pipeline-internal meta-commentary from a report string.

    Operates line-by-line, then sentence-by-sentence within each line.
    Markdown structure (headings, bullet points) is preserved.
    Empty lines (paragraph separators) are kept intact.

    Returns the cleaned text with trailing whitespace normalised.
    """
    if not text:
        return text

    result: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            result.append(line)
            continue

        cleaned = _filter_line(line)
        if cleaned.strip():
            result.append(cleaned)

    return "\n".join(result).rstrip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_meta_sentence(sentence: str) -> bool:
    """Return True if the sentence matches any pipeline-internal or transition pattern."""
    return (
        any(pattern.search(sentence) for pattern in _META_PATTERNS)
        or any(pattern.search(sentence) for pattern in _TRANSITION_PATTERNS)
    )


def _filter_line(line: str) -> str:
    """Filter meta-commentary from a single line.

    For short lines (Markdown headings, bullet items) the whole line is
    checked as one unit. For longer prose lines, sentence-level filtering
    is applied so only the offending sentences are removed.
    """
    # Fast path: whole line is clean.
    if not _is_meta_sentence(line):
        return line

    # Try to salvage multi-sentence lines by removing only bad sentences.
    sentences = _SENTENCE_BOUNDARY.split(line)
    if len(sentences) == 1:
        # Single-sentence line (or couldn't split) — drop the whole line.
        return ""

    kept = [s for s in sentences if not _is_meta_sentence(s)]
    return " ".join(kept)

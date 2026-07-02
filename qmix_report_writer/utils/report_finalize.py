"""
Report finalization — citation tagging, bibliography, abstract.

Extracted from HandcraftedGraph (upgrade-plan Stage 2.2) so both pipelines
(handcrafted tables and the QMIX controller) share one implementation. None of
this depends on phases: everything operates on a ReportState whose sections
carry the RAG chunks (`sources`) they were written from.

Public entry points:
    apply_citation_tags(report_state, section_idx)   — per-section, after review
    build_bibliography(report_state)                 — once, at assembly time
    generate_abstract(report_state, llm)  (async)    — once, over the final body
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("report_finalize")

# ------------------------------------------------------------------
# Citation tagging — sentence-level n-gram overlap constants
# ------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])(\s+)(?=[A-Z"\'])')
_TOKEN_RE          = re.compile(r'[a-zA-Z0-9]+')
_MIN_TOKEN_LEN        = 5   # minimum token length to be considered meaningful
_MIN_CITATION_OVERLAP = 5   # shared tokens required to assign a citation
_MIN_SENTENCE_TOKENS  = 5   # sentence must have this many tokens to be a candidate

# The model sometimes names a source inline in its prose (e.g. quoting the
# filename and page it drew from) rather than leaving the sentence for the
# citation pass to tag. apply_citation_tags() rewrites these inline references
# into proper [cite:N, p.X] tags alongside the overlap-based tagging. A source
# filename is an arXiv-style "*.pdf". Two surface forms are handled:
#   1. Bracketed:  "[2105.06979.pdf | Page: 26]", "[2605.30554.pdf, Page 2 and 11]",
#                  "[1711.02644.pdf, Page 4]", "[2605.30554.pdf]",
#                  "[Source: 9308022.pdf | Page: 29]" (optional "Source:" prefix)
#   2. Prose:      "Source 2105.06979.pdf (Page 2)", "(2605.26692.pdf)", "Source 0410066.pdf"
_INLINE_FILE = r'[A-Za-z0-9][A-Za-z0-9._-]*\.pdf'
_INLINE_PAGES = r'(?:\|\s*|,\s*)?(?:Pages?|pp?\.)\s*:?\s*([0-9][0-9,\s]*(?:and\s*[0-9]+)?)'
_INLINE_REF_BRACKET_RE = re.compile(
    rf'\[\s*(?:Sources?\s*:?\s*)?({_INLINE_FILE})\s*(?:{_INLINE_PAGES})?\s*\]',
    re.IGNORECASE,
)
_INLINE_REF_SOURCE_RE = re.compile(
    rf'(?:Sources?\s+)({_INLINE_FILE})\s*(?:\(\s*{_INLINE_PAGES}\s*\))?',
    re.IGNORECASE,
)
_INLINE_REF_PAREN_RE = re.compile(rf'\(\s*({_INLINE_FILE})\s*\)')

# Bare bracketed reference markers the model sometimes copies verbatim from a
# retrieved chunk's own reference list, e.g. "Ref. [32]", "[12, 14]". The number
# indexes that source's bibliography, never our [cite:N] scheme, so any such
# marker whose number is not among the [cite:N] tags applied to the same
# sentence is a stray reference dropped by _strip_orphan_citation_markers().
# The leading "\d" requirement means real "[cite:N]" tags (which start with "c")
# are never matched.
_ORPHAN_CITE_RE = re.compile(
    r'(?:Refs?\.?|References?)?\s*\[\s*\d+(?:\s*[,&]\s*\d+)*\s*\]',
    re.IGNORECASE,
)

_STOPWORDS = frozenset({
    # 2–3 letter function words
    "the", "and", "for", "are", "was", "its", "his", "her", "our", "has",
    "had", "not", "but", "all", "can", "may", "one", "two", "any", "few",
    "new", "via", "per", "non", "sub", "pre", "pro", "out", "off", "set",
    "let", "yet", "nor", "use", "due", "far", "low", "top", "end", "key",
    # 4+ letter common words
    "that", "this", "with", "from", "have", "been", "which", "they", "their",
    "also", "both", "such", "more", "when", "where", "than", "then", "into",
    "onto", "upon", "these", "those", "there", "here", "what", "some", "each",
    "over", "after", "under", "about", "through", "between", "along", "while",
    "since", "using", "within", "without", "toward", "above", "below",
    "during", "given", "often", "many", "most", "other", "only", "very",
    "show", "shows", "shown", "note", "noted", "term", "terms", "well",
    "thus", "hence", "which", "where", "when", "how", "now", "then",
})

# Matches new arXiv format (e.g. 2605.30554) and old format (e.g. 0208016 / 9804027).
_ARXIV_NEW_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?\.pdf$", re.IGNORECASE)
_ARXIV_OLD_RE = re.compile(r"^(\d{7})(v\d+)?\.pdf$", re.IGNORECASE)


# ------------------------------------------------------------------
# Citation tagging
# ------------------------------------------------------------------

def _tokenize(text: str) -> frozenset:
    """Return a frozenset of meaningful tokens from text.

    Keeps alphanumeric tokens of length >= _MIN_TOKEN_LEN that are not in
    _STOPWORDS.  Numbers are kept at length >= 3 so that values like "154"
    or "360" contribute to overlap scoring alongside longer word tokens.
    """
    result = set()
    for t in _TOKEN_RE.findall(text.lower()):
        if t in _STOPWORDS:
            continue
        # Numbers: keep at >= 3 chars; words: keep at >= _MIN_TOKEN_LEN
        if t.isdigit():
            if len(t) >= 3:
                result.add(t)
        elif len(t) >= _MIN_TOKEN_LEN:
            result.add(t)
    return frozenset(result)


def apply_citation_tags(report_state, section_idx: int) -> None:
    """Insert citation tags into a section using sentence-level n-gram overlap.

    For each non-heading line in the section, sentences are split on
    punctuation boundaries.  Each sentence is scored against the RAG chunks
    that were retrieved when the section was drafted.  When the shared
    meaningful-token count meets _MIN_CITATION_OVERLAP, a [cite:N, p.X] tag
    is appended after the sentence (or [cite:N] if no page info is stored).

    The updated content is written back via replace_section() so that the
    sources list for the section is preserved unchanged.
    """
    sections = report_state.sections
    if not (0 <= section_idx < len(sections)):
        return

    section = sections[section_idx]
    sources = section.get("sources", [])
    bib_map = report_state.bibliography_map  # mutated in-place on first citation

    # Known source filenames (global, case-insensitive) used to resolve
    # inline references. Includes sources already numbered in earlier sections.
    known_lc: Dict[str, str] = {}
    for doc in report_state.sources:
        s = doc.get("source", "").strip()
        if s:
            known_lc.setdefault(s.lower(), s)
    for s in bib_map:
        known_lc.setdefault(s.lower(), s)

    # Pre-tokenize each source chunk; bib numbers are assigned lazily.
    chunk_refs: List[Tuple[frozenset, str, Optional[str]]] = []
    for doc in sources:
        src = doc.get("source", "").strip()
        if not src:
            continue
        page = doc.get("page")
        page_str = str(page) if page and str(page) != "N/A" else None
        chunk_refs.append((_tokenize(doc.get("content", "")), src, page_str))

    # Nothing to do if we can neither score overlap nor resolve inline refs.
    if not chunk_refs and not known_lc:
        return

    new_lines: List[str] = []
    tagged_count = 0
    inline_count = 0
    orphan_count = 0

    for line in section["content"].split("\n"):
        # Preserve headings and blank lines as-is.
        if not line.strip() or line.lstrip().startswith("#"):
            new_lines.append(line)
            continue

        # Split into (sentence, separator, sentence, separator, …) preserving
        # the inter-sentence whitespace via the capturing group in the regex.
        parts = _SENTENCE_SPLIT_RE.split(line)
        new_parts: List[str] = []

        for j, part in enumerate(parts):
            # Odd-indexed parts are the captured whitespace separators.
            if j % 2 != 0:
                new_parts.append(part)
                continue

            # First rewrite any inline source references the model wrote in
            # this sentence into proper [cite:N, p.X] tags.
            if known_lc:
                part, n_inline = _rewrite_inline_references(
                    part, known_lc, bib_map, report_state
                )
                inline_count += n_inline

            # Then run overlap-based tagging on the (possibly updated) sentence.
            sent_tok = _tokenize(part) if chunk_refs else frozenset()
            if chunk_refs and len(sent_tok) >= _MIN_SENTENCE_TOKENS:
                # Collect matching pages grouped by source name.
                pages_by_src: Dict[str, List[str]] = {}
                for chunk_tok, src_name, page_str in chunk_refs:
                    if len(sent_tok & chunk_tok) >= _MIN_CITATION_OVERLAP:
                        if src_name not in pages_by_src:
                            pages_by_src[src_name] = []
                        if page_str and page_str not in pages_by_src[src_name]:
                            pages_by_src[src_name].append(page_str)

                # Assign bib numbers lazily on first citation, then sort by number.
                for src_name in pages_by_src:
                    if src_name not in bib_map:
                        bib_map[src_name] = len(bib_map) + 1

                tags: List[str] = []
                for src_name, pages in sorted(pages_by_src.items(), key=lambda x: bib_map[x[0]]):
                    bib_num = bib_map[src_name]
                    if not pages:
                        tag = f"[cite:{bib_num}]"
                    elif len(pages) == 1:
                        tag = f"[cite:{bib_num}, p.{pages[0]}]"
                    else:
                        tag = f"[cite:{bib_num}, pp.{','.join(pages)}]"
                    # Skip if this document is already cited in this sentence.
                    if not re.search(rf'\[cite:{bib_num}[,\]]', part):
                        tags.append(tag)
                        report_state.citation_counts[bib_num] = (
                            report_state.citation_counts.get(bib_num, 0) + 1
                        )

                if tags:
                    # Insert tags before the trailing sentence-ending punctuation.
                    trailing_m = re.search(r'([.!?])\s*$', part)
                    if trailing_m:
                        pos = trailing_m.start()
                        part = part[:pos] + " " + " ".join(tags) + part[pos:]
                    else:
                        part = part.rstrip() + " " + " ".join(tags)
                    tagged_count += len(tags)

            # Finally drop stray numeric reference markers (e.g. "Ref. [32]")
            # the model copied from a source chunk that don't correspond to a
            # [cite:N] tag applied to this sentence.
            part, n_orphan = _strip_orphan_citation_markers(part)
            orphan_count += n_orphan
            new_parts.append(part)

        new_lines.append("".join(new_parts))

    if tagged_count > 0 or inline_count > 0 or orphan_count > 0:
        report_state.replace_section(section["id"], "\n".join(new_lines))
        logger.info(
            f"Section {section_idx + 1}: "
            f"{tagged_count} citation tag(s) inserted, "
            f"{inline_count} inline reference(s) rewritten, "
            f"{orphan_count} stray marker(s) removed."
        )
    else:
        logger.info(
            f"Section {section_idx + 1}: "
            f"no citation matches above threshold — section unchanged."
        )


def _rewrite_inline_references(
    text: str,
    known_lc: Dict[str, str],
    bib_map: Dict[str, int],
    report_state,
) -> Tuple[str, int]:
    """Rewrite inline source references in one sentence into citation tags.

    Handles bracketed forms (``[file.pdf | Page: 26]`` and the
    ``[Source: file.pdf | Page: 26]`` variant with an inline ``Source:``
    prefix), ``Source file.pdf (Page 2)`` prose, and bare ``(file.pdf)``
    mentions. Only references whose
    filename resolves to a known source are rewritten; unknown filenames are
    left untouched. Bib numbers are assigned lazily (matching the overlap
    pass) and ``citation_counts`` is updated per new tag. At most one tag is
    kept per source per sentence — the same rule the overlap pass enforces —
    so a source already cited in the sentence (by an existing tag, a prior
    inline reference, or one the overlap pass will add) is never duplicated;
    the redundant inline reference text is dropped instead.

    Returns ``(new_text, n_changes)``.
    """
    # Bib numbers already cited in this sentence. Seeded from any tags
    # already present and grown as we rewrite, so we never emit two tags for
    # the same source. \d+ captures the full number, so "1" never matches
    # inside "[cite:12]".
    present_nums = set(re.findall(r'\[cite:(\d+)', text))
    n = 0

    def _build_tag(num: int, pages: List[str]) -> str:
        if not pages:
            return f"[cite:{num}]"
        if len(pages) == 1:
            return f"[cite:{num}, p.{pages[0]}]"
        return f"[cite:{num}, pp.{','.join(pages)}]"

    def _repl(m: "re.Match") -> str:
        nonlocal n
        src = known_lc.get(m.group(1).strip().lower())
        if src is None:
            return m.group(0)  # unknown filename — leave the prose as written
        pages = re.findall(r'\d+', m.group(2)) if m.re.groups >= 2 and m.group(2) else []
        if src not in bib_map:
            bib_map[src] = len(bib_map) + 1
        num = bib_map[src]
        n += 1
        if str(num) in present_nums:
            return ""  # source already cited in this sentence — drop reference
        present_nums.add(str(num))
        report_state.citation_counts[num] = (
            report_state.citation_counts.get(num, 0) + 1
        )
        return _build_tag(num, pages)

    out = _INLINE_REF_BRACKET_RE.sub(_repl, text)
    out = _INLINE_REF_SOURCE_RE.sub(_repl, out)
    out = _INLINE_REF_PAREN_RE.sub(_repl, out)

    if n:
        # Tidy whitespace / empty parens left by removed references.
        out = re.sub(r'\(\s*\)', '', out)
        out = re.sub(r'[ \t]{2,}', ' ', out)
        out = re.sub(r'\s+([.,;:])', r'\1', out)
    return out, n


def _strip_orphan_citation_markers(text: str) -> Tuple[str, int]:
    """Remove stray bare numeric reference markers from a sentence.

    The model occasionally copies a bracketed reference marker straight out
    of a retrieved chunk (e.g. ``Ref. [32]``), where the number indexes that
    source's own reference list rather than this report's bibliography. The
    report only ever cites sources with ``[cite:N, p.X]`` tags, so a bare
    ``[N]`` marker is never a valid citation — it is always dropped to keep a
    consistent format and avoid duplicated references. Real ``[cite:N, p.X]``
    tags are not matched by the marker pattern, so they are left untouched.

    Returns ``(new_text, n_removed)``.
    """
    n = 0

    def _repl(m: "re.Match") -> str:
        nonlocal n
        n += 1
        return ""

    out = _ORPHAN_CITE_RE.sub(_repl, text)
    if n:
        # Tidy whitespace left where markers were removed.
        out = re.sub(r'[ \t]{2,}', ' ', out)
        out = re.sub(r'\s+([.,;:])', r'\1', out)
    return out, n


# ------------------------------------------------------------------
# Bibliography
# ------------------------------------------------------------------

def build_bibliography(report_state) -> None:
    """Build the final bibliography text using per-entry citation counts.

    bibliography_map is populated incrementally by apply_citation_tags —
    only sources that were actually cited in the text have entries there.
    All other collected sources are listed under Consulted Sources (no tag).

    Must be called after all apply_citation_tags() calls.
    """
    bib_map = report_state.bibliography_map
    counts  = report_state.citation_counts

    if not report_state.sources:
        logger.warning("Bibliography: no sources collected — skipping.")
        return

    # Map each source filename to the bibliographic metadata (title/author/year)
    # carried on its retrieved chunks. First non-empty wins per source.
    meta_by_source: dict = {}
    for doc in report_state.sources:
        src = (doc.get("source") or "").strip()
        if not src or src in meta_by_source:
            continue
        fields = {k: (doc.get(k) or "").strip() for k in ("title", "author", "year")}
        if any(fields.values()):
            meta_by_source[src] = fields

    cited_lines: list = ["## Bibliography\n"]
    for source_name, num in sorted(bib_map.items(), key=lambda x: x[1]):
        c = counts.get(num, 0)
        count_tag = f" *({c} citation{'s' if c != 1 else ''})*"
        cited_lines.append(
            format_bib_entry(num, source_name, meta_by_source.get(source_name)) + count_tag
        )

    seen: set = set()
    consulted_lines: list = []
    for doc in report_state.sources:
        src = (doc.get("source") or "").strip()
        if src and src not in bib_map and src not in seen:
            seen.add(src)
            consulted_lines.append(
                format_consulted_entry(src, meta_by_source.get(src))
            )

    report_state.bibliography = "\n".join(cited_lines)
    if consulted_lines:
        report_state.bibliography += (
            "\n\n### Consulted Sources\n\n" + "\n".join(consulted_lines)
        )

    logger.info(
        f"Bibliography built: "
        f"{len(cited_lines) - 1} cited, {len(consulted_lines)} consulted-only."
    )


def _arxiv_id_of(source_name: str) -> str:
    """Return the arXiv id encoded in the filename, or '' if it is not an arXiv PDF."""
    m = _ARXIV_NEW_RE.match(source_name) or _ARXIV_OLD_RE.match(source_name)
    return m.group(1) if m else ""


def _compose_reference(source_name: str, meta: Optional[dict]) -> str:
    """Build a proper reference string from available metadata.

    Renders "Author. “Title”. Year. (identifier)" using whatever fields are
    present, always keeping the source filename as a locator so [cite:N] tags
    remain traceable. Falls back to a filename-first entry when no
    bibliographic metadata was extracted for the document.
    """
    meta = meta or {}
    title = (meta.get("title") or "").strip()
    author = (meta.get("author") or "").strip()
    year = (meta.get("year") or "").strip()
    arxiv_id = _arxiv_id_of(source_name)

    # Identifier suffix: arXiv id (when present) plus the filename locator.
    ident_bits = []
    if arxiv_id:
        ident_bits.append(f"arXiv:{arxiv_id}")
    ident_bits.append(source_name)
    ident = ", ".join(ident_bits)

    parts = []
    if author:
        parts.append(author if author.endswith(".") else f"{author}.")
    if title:
        # Curly quotes become proper LaTeX quotes after markdown conversion.
        parts.append(f"“{title}”.")
    if year:
        parts.append(f"{year}.")

    if parts:
        return " ".join(parts) + f" ({ident})"

    # No descriptive metadata — keep the legacy filename-first rendering.
    if arxiv_id:
        return f"**{source_name}** *(arXiv:{arxiv_id})*"
    ext = source_name.rsplit(".", 1)[-1].upper() if "." in source_name else ""
    type_tag = f" *({ext})*" if ext else ""
    return f"**{source_name}**{type_tag}"


def format_bib_entry(num: int, source_name: str, meta: Optional[dict] = None) -> str:
    """Return one markdown bibliography line for the given source."""
    return f"[{num}] " + _compose_reference(source_name, meta)


def format_consulted_entry(source_name: str, meta: Optional[dict] = None) -> str:
    """Return one markdown consulted-sources line (no citation number)."""
    return "- " + _compose_reference(source_name, meta)


# ------------------------------------------------------------------
# Abstract
# ------------------------------------------------------------------

async def generate_abstract(report_state, llm) -> str:
    """Write a concise reader-facing abstract for the finished report.

    Runs once at assembly time over the final body content, so it reflects
    the report after all validation/revision is complete. Returns the
    abstract paragraph (no heading), or "" if no body exists or the call
    fails — callers must treat "" as "skip the abstract".
    """
    body = (report_state.content or "").strip()
    if not body:
        return ""

    if llm is None:
        return ""

    system = (
        "You are a scientific editor writing the abstract for a completed "
        "technical report. Summarise the report for a reader deciding whether "
        "to read it.\n\n"
        "RULES:\n"
        "1. Write ONE self-contained paragraph of roughly 150-250 words.\n"
        "2. Cover the report's scope/objective, the approach or evidence it "
        "draws on, its key findings, and its main conclusions — in that order.\n"
        "3. Plain expository prose. Do NOT include a heading, a 'In this report' "
        "preamble, bullet points, or section references of any kind.\n"
        "4. Do NOT include citation tags (e.g. [cite:3]), bibliography numbers, "
        "or figure/section labels. The abstract must read as standalone prose.\n"
        "5. Use only information present in the report body below — do not "
        "introduce claims, numbers, or conclusions that are not in the text."
    )
    user = (
        f"### Report subject\n{report_state.task}\n\n"
        f"### Report body\n{body}\n\n"
        "Write the abstract now. Output ONLY the abstract paragraph, with no "
        "heading and no surrounding commentary."
    )
    message = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        response = await llm.agen(message, calling_agent="LeadArchitect")
    except Exception as exc:
        logger.warning(f"Abstract generation failed: {exc}")
        return ""

    abstract = (response or "").strip()
    if not abstract:
        return ""

    # Strip a leading "Abstract"/"## Abstract" heading if the model added one
    # despite the instruction (we supply the heading ourselves at assembly).
    abstract = re.sub(r"(?i)^\s*#*\s*abstract\s*[:\-]?\s*\n+", "", abstract).strip()
    # Remove any stray citation tags so the abstract stays self-contained.
    abstract = re.sub(r"\s*\[cite:[^\]]*\]", "", abstract).strip()
    return abstract

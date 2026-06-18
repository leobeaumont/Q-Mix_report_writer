"""
Convert a raw markdown report (as produced by the handcrafted pipeline) into a
compilable LaTeX document.

The pipeline emits markdown where:
  * mathematical formulas / symbols / greek letters are *already* in LaTeX,
    delimited by ``$ ... $`` — these must be preserved verbatim;
  * prose is markdown (``**bold**``, ``*italic*``, ``-``/``1.`` lists);
  * in-text citations use ``[cite:N]`` / ``[cite:N, p.X]`` / ``[cite:N, pp.X,Y]``;
  * sections are ``## Heading`` level-2 markdown headings;
  * the final two sections are always ``## Bibliography`` followed by a
    ``### Consulted Sources`` sub-list.

The produced ``.tex`` targets **pdfLaTeX** and is self-contained: all stray
unicode (em-dashes, curly quotes, greek-in-prose, ±, ×, …) is converted to
LaTeX commands so no special engine/font setup is required.

Document structure:
  1. Title page  — report title (the original query), logo, AI-generated
     disclaimer, and the author line.
  2. Table of contents.
  3. The report sections.
  4. ``Bibliography`` rendered as a ``thebibliography`` environment, with every
     ``[cite:N]`` tag converted to a proper ``\\cite``.
  5. ``Consulted Sources`` as an unnumbered list.

Entry point:
    from utils.markdown_to_latex import convert_run_dir
    tex_path = convert_run_dir("output/2026-..._the-nuclear-equation-of-state")
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .const import PROJECT_ROOT
from .report_export import RAW_MARKDOWN_NAME, LATEX_NAME

ASSETS_DIR = PROJECT_ROOT / "assets"

# SCK CEN brand assets copied into each run folder so the .tex is self-contained.
# The title page / running header reference these by name, each guarded by
# \IfFileExists so the document still compiles if an asset is missing.
SCK_LOGO = "SCK_logo.png"          # black "sck cen" wordmark (title page)
QMIX_LOGO = "logo_qmix.png"        # Q-Mix report-writer pipeline emblem
SCK_BACKGROUND = "background_sck.png"  # dot-pattern brand background (title page)
BRAND_ASSETS = [SCK_LOGO, QMIX_LOGO, SCK_BACKGROUND]

# Filenames tried, in order, for an explicit logo override (replaces SCK_LOGO on
# the title page). If none exist the title page falls back to SCK_LOGO.
_DEFAULT_LOGO_CANDIDATES = ["logo.png", "sckcen.png", "sck_cen.png", "SCKCEN.png", "logo.pdf"]

DEFAULT_AUTHOR = r"Q-Mix report writer designed by L\'eo Beaumont at SCK CEN"
DEFAULT_DISCLAIMER = (
    "This report was generated automatically by an AI system. "
    "Its contents may contain inaccuracies or omissions and should be "
    "independently verified before use."
)

# ---------------------------------------------------------------------------
# Inline conversion: markdown + unicode -> LaTeX, preserving math and citations
# ---------------------------------------------------------------------------

# LaTeX special characters that must be escaped when they appear in prose.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESCAPE_RE = re.compile(r"[\\&%$#_{}~^]")

# Unicode punctuation / symbols -> LaTeX. Applied to prose only (never math).
_UNICODE_MAP = {
    "—": "---", "–": "--", "‐": "-", "‑": "-", "−": "$-$",
    "“": "``", "”": "''", "‘": "`", "’": "'", "‚": ",",
    "…": r"\ldots{}",
    "·": r"$\cdot$", "•": r"$\bullet$",
    "×": r"$\times$", "÷": r"$\div$", "±": r"$\pm$", "∓": r"$\mp$",
    "≈": r"$\approx$", "≃": r"$\simeq$", "∼": r"$\sim$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$",
    "→": r"$\rightarrow$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
    "°": r"\textdegree{}", "′": r"$'$", "″": r"$''$",
    "∞": r"$\infty$", "∝": r"$\propto$", "∂": r"$\partial$", "∇": r"$\nabla$",
    "√": r"$\surd$", "∫": r"$\int$", "∑": r"$\sum$", "∏": r"$\prod$",
    "ħ": r"$\hbar$", "ℏ": r"$\hbar$", "Å": r"\AA{}", "µ": r"$\mu$",
    "©": r"\textcopyright{}", "®": r"\textregistered{}", "™": r"\texttrademark{}",
    " ": "~", " ": r"\,", " ": r"\,",
}

# Greek letters appearing in *prose* (math greek is already inside $...$).
_GREEK_MAP = {
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$", "δ": r"$\delta$",
    "ε": r"$\epsilon$", "ζ": r"$\zeta$", "η": r"$\eta$", "θ": r"$\theta$",
    "ι": r"$\iota$", "κ": r"$\kappa$", "λ": r"$\lambda$", "μ": r"$\mu$",
    "ν": r"$\nu$", "ξ": r"$\xi$", "ο": "o", "π": r"$\pi$", "ρ": r"$\rho$",
    "σ": r"$\sigma$", "ς": r"$\varsigma$", "τ": r"$\tau$", "υ": r"$\upsilon$",
    "φ": r"$\phi$", "χ": r"$\chi$", "ψ": r"$\psi$", "ω": r"$\omega$",
    "Γ": r"$\Gamma$", "Δ": r"$\Delta$", "Θ": r"$\Theta$", "Λ": r"$\Lambda$",
    "Ξ": r"$\Xi$", "Π": r"$\Pi$", "Σ": r"$\Sigma$", "Φ": r"$\Phi$",
    "Ψ": r"$\Psi$", "Ω": r"$\Omega$",
}

# Display math the pipeline sometimes emits as $$ ... $$ (often spanning lines).
# Must be handled before inline math so its dollar pairs are consumed cleanly.
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
# Inline math span: $...$ with no embedded $.
_MATH_RE = re.compile(r"\$[^$]+\$")

# A single subscript/superscript "atom": a braced group, a TeX command, or one
# non-space character. Used to detect and merge illegal double scripts.
_SCRIPT_ATOM = r"(?:\{[^{}]*\}|\\[a-zA-Z]+\*?|[^\s_^{}])"
# Two consecutive scripts of the *same* type on one atom (e.g. ``m_0_q`` or
# ``x^a^b``) — TeX rejects these as "Double subscript/superscript".
_DOUBLE_SCRIPT_RE = re.compile(r"(?<!\\)([_^])(" + _SCRIPT_ATOM + r")\1(" + _SCRIPT_ATOM + r")")


def _unbrace(atom: str) -> str:
    """Strip one layer of surrounding braces from a script atom, if present."""
    if len(atom) >= 2 and atom[0] == "{" and atom[-1] == "}":
        return atom[1:-1]
    return atom


def _fix_double_scripts(math: str) -> str:
    """Merge illegal consecutive same-type scripts into a single braced group.

    The pipeline's LLM occasionally emits ``m_0_q`` (two subscripts on one atom),
    which halts TeX with a "Double subscript" error. This collapses such runs to
    ``m_{0 q}`` (and the superscript analogue), iterating so triple+ runs are
    handled too. Math is otherwise preserved verbatim.
    """
    prev = None
    while prev != math:
        prev = math
        math = _DOUBLE_SCRIPT_RE.sub(
            lambda m: f"{m.group(1)}{{{_unbrace(m.group(2))} {_unbrace(m.group(3))}}}",
            math,
        )
    return math
# Inline equations longer than this many characters of source are promoted to a
# display equation (broken onto their own line at full size). \fitmath only
# shrinks them further if they are still wider than a full text line.
_LONG_MATH_THRESHOLD = 80
# Citation tag: [cite:N] optionally followed by ", p.X" or ", pp.X,Y".
_CITE_RE = re.compile(r"\[cite:(\d+)(?:\s*,\s*(pp?\.[^\]]*))?\]")


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain prose."""
    return _ESCAPE_RE.sub(lambda m: _LATEX_ESCAPES[m.group()], text)


def _replace_unicode(text: str) -> str:
    """Replace unicode punctuation, symbols and prose greek with LaTeX."""
    out = []
    for ch in text:
        if ch in _UNICODE_MAP:
            out.append(_UNICODE_MAP[ch])
        elif ch in _GREEK_MAP:
            out.append(_GREEK_MAP[ch])
        else:
            out.append(ch)
    return "".join(out)


def _cite_to_latex(num: str, pages: Optional[str]) -> str:
    """Render one ``[cite:N, p.X]`` tag as a ``\\cite`` command."""
    if pages:
        # "p.3" -> "p.~3", "pp.13,11" -> "pp.~13,11" (non-breaking space).
        note = re.sub(r"^(pp?\.)\s*", r"\1~", pages.strip())
        return f"\\cite[{note}]{{src{num}}}"
    return f"\\cite{{src{num}}}"


def convert_inline(text: str) -> str:
    """Convert a span of markdown prose to LaTeX.

    Preserves ``$...$`` math verbatim, turns ``[cite:N]`` tags into ``\\cite``,
    escapes LaTeX specials, maps unicode to LaTeX, and converts ``**bold**`` /
    ``*italic*`` emphasis.
    """
    placeholders: Dict[str, str] = {}

    # 1. Pull out math spans (kept verbatim) and citation tags (pre-rendered).
    def _stash(value: str, tag: str) -> str:
        token = f"\x00{tag}{len(placeholders)}\x00"
        placeholders[token] = value
        return token

    def _render_display(m: "re.Match") -> str:
        inner = _fix_double_scripts(m.group(1).strip())
        if len(inner) > _LONG_MATH_THRESHOLD:
            return _stash(f"\\fitmath{{{inner}}}", "M")
        return _stash(f"\\[{inner}\\]", "M")

    def _render_math(m: "re.Match") -> str:
        inner = _fix_double_scripts(m.group()[1:-1])  # strip the $ delimiters
        if len(inner) > _LONG_MATH_THRESHOLD:
            return _stash(f"\\fitmath{{{inner}}}", "M")
        return _stash(f"${inner}$", "M")

    # Display math ($$...$$) first, so its dollar pairs are consumed before the
    # inline pass — otherwise stray $ desynchronise all later inline matching.
    text = _DISPLAY_MATH_RE.sub(_render_display, text)
    text = _MATH_RE.sub(_render_math, text)
    text = _CITE_RE.sub(
        lambda m: _stash(_cite_to_latex(m.group(1), m.group(2)), "C"), text
    )

    # 2. Escape specials, then map unicode (inserts LaTeX commands, so after).
    text = _escape_latex(text)
    text = _replace_unicode(text)

    # 3. Markdown emphasis (bold before italic so ** is not eaten by *).
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\textit{\1}", text)

    # 4. Restore stashed math / citations.
    for token, value in placeholders.items():
        text = text.replace(token, value)
    return text


# ---------------------------------------------------------------------------
# Block conversion: paragraphs, headings and lists within a section body
# ---------------------------------------------------------------------------

_LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _heading_command(level: int, text: str) -> str:
    """Map a markdown heading level to a LaTeX sectioning command."""
    cmd = {1: "section", 2: "section", 3: "subsection", 4: "subsubsection"}.get(
        level, "paragraph"
    )
    return f"\\{cmd}{{{convert_inline(text)}}}"


def _consume_list(lines: List[str], start: int) -> Tuple[str, int]:
    """Convert one (possibly nested) markdown list starting at ``lines[start]``.

    Returns the LaTeX for the list and the index of the first line after it.
    """
    base_indent = len(_LIST_ITEM_RE.match(lines[start]).group(1))
    ordered = bool(re.match(r"\d+\.", _LIST_ITEM_RE.match(lines[start]).group(2)))
    env = "enumerate" if ordered else "itemize"

    items: List[str] = []  # each entry is the full LaTeX for one \item
    i = start
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            # A blank line continues the list only if a deeper/equal item follows.
            nxt = i + 1
            m_next = _LIST_ITEM_RE.match(lines[nxt]) if nxt < len(lines) else None
            if m_next and len(m_next.group(1)) >= base_indent:
                i += 1
                continue
            break
        m = _LIST_ITEM_RE.match(line)
        if not m:
            break
        indent = len(m.group(1))
        if indent < base_indent:
            break
        if indent > base_indent:
            # Nested list: attach to the previous item.
            sub, i = _consume_list(lines, i)
            if items:
                items[-1] += "\n" + sub
            continue
        items.append(r"  \item " + convert_inline(m.group(3)))
        i += 1

    body = "\n".join(items)
    return f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}", i


def convert_blocks(text: str) -> str:
    """Convert a section body (paragraphs, lists, sub-headings) to LaTeX."""
    lines = text.split("\n")
    out: List[str] = []
    para: List[str] = []

    def flush_para() -> None:
        if para:
            out.append(convert_inline(" ".join(para).strip()))
            out.append("")
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_para()
            i += 1
            continue
        h = _HEADING_RE.match(stripped)
        if h:
            flush_para()
            out.append(_heading_command(len(h.group(1)), h.group(2)))
            out.append("")
            i += 1
            continue
        if _LIST_ITEM_RE.match(line):
            flush_para()
            block, i = _consume_list(lines, i)
            out.append(block)
            out.append("")
            continue
        para.append(stripped)
        i += 1

    flush_para()
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Markdown document parsing
# ---------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_QUERY_RE = re.compile(r"Query:\s*(.+)")
_GENERATED_RE = re.compile(r"Generated:\s*(.+)")
# A cited bibliography entry is any line opening with ``[N]``; a consulted entry
# any line opening with ``-``. The remainder is the reference body (author /
# title / year / identifier markdown), rendered through convert_inline so the
# parser stays decoupled from the exact reference layout produced upstream.
_BIB_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+(.*\S)\s*$", re.MULTILINE)
_CONSULTED_RE = re.compile(r"^-\s+(.*\S)\s*$", re.MULTILINE)


def _extract_metadata(md_text: str) -> Dict[str, str]:
    """Pull the Query / Generated values from the leading HTML comments."""
    meta: Dict[str, str] = {}
    for body in _COMMENT_RE.findall(md_text):
        q = _QUERY_RE.search(body)
        if q:
            meta["query"] = q.group(1).strip()
        g = _GENERATED_RE.search(body)
        if g:
            meta["generated"] = g.group(1).strip()
    return meta


def _split_body_and_bibliography(md_text: str) -> Tuple[str, str]:
    """Split the document at the ``## Bibliography`` heading.

    Returns (body_markdown, bibliography_region). If there is no bibliography,
    the second element is an empty string.
    """
    text = _COMMENT_RE.sub("", md_text).strip()
    m = re.search(r"(?m)^##\s+Bibliography\s*$", text)
    if not m:
        return text, ""
    return text[: m.start()].rstrip(), text[m.start():].strip()


def _split_sections(body_md: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Split body markdown into (lead_text, [(title, content), ...]) by ``##``."""
    parts = re.split(r"(?m)^##\s+(.+)$", body_md)
    lead = parts[0].strip()
    sections: List[Tuple[str, str]] = []
    for idx in range(1, len(parts), 2):
        title = parts[idx].strip()
        content = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        sections.append((title, content))
    return lead, sections


def _parse_bibliography(bib_region: str) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Parse the bibliography region into cited entries and consulted sources.

    Returns (cited, consulted) where:
      cited     = [(number, reference_markdown), ...]
      consulted = [reference_markdown, ...]
    The reference markdown is whatever the pipeline rendered after the ``[N]`` /
    ``-`` marker; it is converted to LaTeX at render time.
    """
    if not bib_region:
        return [], []

    # Separate the cited list from the "### Consulted Sources" sub-list.
    m = re.search(r"(?m)^###\s+Consulted Sources\s*$", bib_region)
    cited_text = bib_region[: m.start()] if m else bib_region
    consulted_text = bib_region[m.end():] if m else ""

    cited = [(num, body.strip()) for num, body in _BIB_ENTRY_RE.findall(cited_text)]
    consulted = [body.strip() for body in _CONSULTED_RE.findall(consulted_text)]
    return cited, consulted


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

# arXiv ids as they survive convert_inline (``arXiv:2606.14166`` / ``arXiv:0405013``).
_ARXIV_LINK_RE = re.compile(r"arXiv:([0-9A-Za-z][0-9A-Za-z./\-]*)")


def _linkify_arxiv(latex_text: str) -> str:
    """Turn any ``arXiv:ID`` mention into a hyperlink (lavender via urlcolor)."""
    return _ARXIV_LINK_RE.sub(
        lambda m: f"\\href{{https://arxiv.org/abs/{m.group(1)}}}{{arXiv:{m.group(1)}}}",
        latex_text,
    )


def _render_reference_body(body_md: str) -> str:
    """Render one reference's markdown body (author/title/year/id) to LaTeX."""
    return _linkify_arxiv(convert_inline(body_md))


def _render_bibliography(cited: List[Tuple[str, str]]) -> str:
    if not cited:
        return ""
    lines = [r"\begin{thebibliography}{99}"]
    for num, body in cited:
        # Lavender entry label, e.g. [3], to match the brand palette.
        label = f"{{\\textcolor{{scklavender}}{{{num}}}}}"
        lines.append(f"\\bibitem[{label}]{{src{num}}} {_render_reference_body(body)}")
    lines.append(r"\end{thebibliography}")
    return "\n".join(lines)


def _render_consulted(consulted: List[str]) -> str:
    if not consulted:
        return ""
    lines = [
        r"\section*{Consulted Sources}",
        r"\addcontentsline{toc}{section}{Consulted Sources}",
        r"\begin{itemize}",
    ]
    for body in consulted:
        lines.append(r"  \item " + _render_reference_body(body))
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def _pdf_safe(text: str) -> str:
    """Strip math/markdown so a title is usable in PDF metadata."""
    text = _MATH_RE.sub("", text)
    return re.sub(r"[\\{}$*_#]", "", text).strip()


def _title_page(title: str, logo_rel: Optional[str], author: str, disclaimer: str) -> str:
    """Branded SCK CEN title page.

    A flat white page with the SCK wordmark in the top-left corner and the Q-Mix
    pipeline emblem (captioned "Q-Mix report writer") in the top-right. The report
    title sits in dark purple below; the dot-pattern brand graphic is centred as a
    decorative block to fill the page, with the disclaimer, author and date at the
    foot. Every image is wrapped in \\IfFileExists so a missing asset never breaks
    compilation.
    """
    sck_logo = logo_rel or SCK_LOGO
    return "\n".join(
        [
            r"\begin{titlepage}",
            # ── Top logo row: SCK wordmark (left) / Q-Mix emblem (right) ──
            r"\noindent",
            r"\begin{minipage}[t]{0.48\textwidth}",
            r"\raggedright",
            f"\\IfFileExists{{{sck_logo}}}"
            f"{{\\includegraphics[width=0.70\\linewidth]{{{sck_logo}}}}}{{}}",
            r"\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.48\textwidth}",
            r"\raggedleft",
            f"\\IfFileExists{{{QMIX_LOGO}}}"
            f"{{\\includegraphics[width=0.45\\linewidth]{{{QMIX_LOGO}}}\\par\\vspace{{0.2cm}}}}{{}}",
            r"{\large\itshape\color{sckpurple}Q-Mix report writer}",
            r"\end{minipage}",
            # \par closes the logo-row paragraph so the vspace below applies in
            # vertical mode (otherwise it leaks past the title block).
            r"\par",
            # ── Title block (sits lower, just above the report title) ──
            r"\vspace{2.5cm}",
            r"\centering",
            r"{\headingfont\color{scklavender}\Large\scshape Technical Report\par}",
            r"\vspace{0.5cm}",
            r"{\headingfont\color{sckpurple}\huge\bfseries " + convert_inline(title) + r"\par}",
            # ── Decorative brand graphic ──
            r"\vspace{1.5cm}",
            f"\\IfFileExists{{{SCK_BACKGROUND}}}"
            f"{{\\includegraphics[width=0.825\\textwidth]{{{SCK_BACKGROUND}}}\\par}}{{}}",
            # ── Footer block ──
            r"\vfill",
            r"\begin{minipage}{0.85\textwidth}\centering",
            r"{\color{sckpurple!75}\itshape\small " + _escape_latex(disclaimer) + r"\par}",
            r"\end{minipage}\par",
            r"\vspace{0.8cm}",
            r"{\color{sckpurple}\large " + author + r"\par}",
            r"\vspace{0.2cm}",
            r"{\color{scklavender}\today\par}",
            r"\end{titlepage}",
        ]
    )


_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{amsmath,amssymb}

% ---- Fonts (Tectonic / XeTeX) ----------------------------------------------
% Body text: Segoe UI; section headings & running header: Aptos Display.
% \headingfont falls back to the default bold face if a font is not installed,
% so the document still builds.
\usepackage{fontspec}
\providecommand{\headingfont}{}
\IfFontExistsTF{Segoe UI}{\setmainfont{Segoe UI}}{}
\IfFontExistsTF{Aptos Display}{%
  \newfontfamily\aptosheading{Aptos Display}%
  \renewcommand{\headingfont}{\aptosheading}}{}
\usepackage{textcomp}
\usepackage{graphicx}
\usepackage[export]{adjustbox}
\usepackage[margin=2.5cm]{geometry}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[colorlinks=true]{hyperref}

% ---- SCK CEN brand palette -------------------------------------------------
\definecolor{sckpurple}{HTML}{56256B}
\definecolor{scklavender}{HTML}{9F4191}
\definecolor{sckblue}{HTML}{A1DAF7}

\hypersetup{
  linkcolor=sckpurple,
  citecolor=sckpurple,
  urlcolor=scklavender,
}
\renewcommand{\refname}{Bibliography}

% ---- Section headings in brand colours (Aptos Display) ---------------------
\titleformat{\section}
  {\headingfont\Large\bfseries\color{sckpurple}}{\thesection}{1em}{}
  [{\color{scklavender}\titlerule[1pt]}]
\titleformat{\subsection}
  {\headingfont\large\bfseries\color{scklavender}}{\thesubsection}{1em}{}
\titleformat{\subsubsection}
  {\headingfont\normalsize\bfseries\color{sckpurple}}{\thesubsubsection}{1em}{}

% ---- Brand-coloured list bullets -------------------------------------------
\setlist[itemize,1]{label={\color{scklavender}\textbullet}}
\setlist[itemize,2]{label={\color{sckpurple}\textendash}}

% ---- Inline and display maths in dark purple -------------------------------
\everymath\expandafter{\the\everymath \color{sckpurple}}
\everydisplay\expandafter{\the\everydisplay \color{sckpurple}}

% ---- Running header / footer in brand colours (Aptos Display) --------------
% \fancyhead[L] (the document name) is set per-document after the preamble.
\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\headingfont\small\bfseries\color{sckpurple}SCK CEN}
\fancyfoot[C]{\small\color{sckpurple}\thepage}
\renewcommand{\headrule}{{\color{scklavender}\hrule height 0.8pt}}
\renewcommand{\footrulewidth}{0pt}

% Over-long inline equations are promoted to a display equation on their own
% line; adjustbox shrinks them only if they are still wider than the text line.
\newcommand{\fitmath}[1]{\[\adjustbox{max width=\linewidth}{$\displaystyle #1$}\]}
\emergencystretch=2em
"""


def markdown_to_latex(
    md_text: str,
    title: Optional[str] = None,
    logo_rel: Optional[str] = None,
    author: str = DEFAULT_AUTHOR,
    disclaimer: str = DEFAULT_DISCLAIMER,
) -> str:
    """Convert a full raw-markdown report to a complete LaTeX document string."""
    meta = _extract_metadata(md_text)
    if title is None:
        title = meta.get("query", "Report")

    body_md, bib_region = _split_body_and_bibliography(md_text)
    lead, sections = _split_sections(body_md)
    cited, consulted = _parse_bibliography(bib_region)

    parts: List[str] = [_PREAMBLE]
    parts.append(f"\\hypersetup{{pdftitle={{{_pdf_safe(title)}}}}}")
    # Running left header shows the document name (the report title).
    header_name = _escape_latex(_pdf_safe(title))
    parts.append(
        f"\\fancyhead[L]{{\\headingfont\\small\\itshape\\color{{sckpurple}}{header_name}}}"
    )
    parts.append(r"\begin{document}")
    parts.append(_title_page(title, logo_rel, author, disclaimer))
    parts.append(r"\tableofcontents")
    parts.append(r"\newpage")

    if lead:
        parts.append(convert_blocks(lead))

    for sec_title, sec_content in sections:
        parts.append(f"\\section{{{convert_inline(sec_title)}}}")
        parts.append(convert_blocks(sec_content))

    bib = _render_bibliography(cited)
    if bib:
        parts.append(bib)

    consulted_block = _render_consulted(consulted)
    if consulted_block:
        parts.append(consulted_block)

    parts.append(r"\end{document}")
    return "\n\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Run-folder integration
# ---------------------------------------------------------------------------

def _resolve_logo(logo: Optional[str]) -> Optional[Path]:
    """Resolve the logo to an absolute path, or None if unavailable.

    ``logo`` may be an absolute path, a path relative to the project root, or a
    bare filename living in ``assets/``. When None, the default candidates in
    ``assets/`` are tried in order.
    """
    if logo:
        p = Path(logo)
        for cand in (p, PROJECT_ROOT / p, ASSETS_DIR / p.name):
            if cand.is_file():
                return cand
        return None
    for name in _DEFAULT_LOGO_CANDIDATES:
        cand = ASSETS_DIR / name
        if cand.is_file():
            return cand
    return None


def convert_run_dir(
    run_dir,
    logo: Optional[str] = None,
    author: str = DEFAULT_AUTHOR,
    disclaimer: str = DEFAULT_DISCLAIMER,
) -> Path:
    """Convert ``report_raw.md`` in a run folder to ``report.tex``.

    The chosen logo (if any) is copied into the run folder so the resulting
    ``.tex`` is self-contained and can be compiled from that directory.

    Returns the path to the written ``report.tex``.
    """
    run_dir = Path(run_dir)
    md_path = run_dir / RAW_MARKDOWN_NAME
    if not md_path.is_file():
        raise FileNotFoundError(f"No {RAW_MARKDOWN_NAME} found in {run_dir}")

    md_text = md_path.read_text(encoding="utf-8")

    # Copy the SCK CEN brand assets (wordmark, pipeline emblem, background) into
    # the run folder so the .tex compiles standalone. Each is referenced via
    # \IfFileExists, so any that is absent is simply skipped at compile time.
    for asset_name in BRAND_ASSETS:
        src = ASSETS_DIR / asset_name
        if src.is_file():
            shutil.copy(src, run_dir / asset_name)

    # Optional explicit logo override (replaces the SCK wordmark on the title page).
    logo_rel: Optional[str] = None
    logo_src = _resolve_logo(logo)
    if logo_src is not None:
        dest = run_dir / logo_src.name
        if logo_src.resolve() != dest.resolve():
            shutil.copy(logo_src, dest)
        logo_rel = logo_src.name

    tex = markdown_to_latex(
        md_text, logo_rel=logo_rel, author=author, disclaimer=disclaimer
    )
    tex_path = run_dir / LATEX_NAME
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a run folder's report_raw.md into report.tex."
    )
    parser.add_argument("run_dir", help="Path to the report run folder.")
    parser.add_argument(
        "--logo", default=None, help="Logo filename in assets/ (or a path)."
    )
    args = parser.parse_args()

    out = convert_run_dir(args.run_dir, logo=args.logo)
    print(f"Wrote {out}")

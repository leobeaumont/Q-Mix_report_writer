"""
LLM judges v2 (training_eval plan 2.1–2.3 — decisions TD1/TD3, OD-D).

Two scoring surfaces, exposed through `ReportEvaluator`:

  * score_chunk  — GROUNDED micro judging of one appended section: a chunk
    audit that SEES the section's stored RAG sources, plus a claim-level
    check that verdicts every substantive claim against those sources
    (supported / unsupported / contradicted). ~2 judge calls per append.
  * score_report — terminal macro judging of the full report against the
    commissioned task and the planned outline. ONE call per episode.

Both return None on judge failure (the caller skips the reward event —
plan 1.3; a transport/parse hiccup never becomes a score).

Transport (hard-won live findings, plan 1.3 amendments #1–#4):
  * Ollama's NATIVE /api/chat `format` endpoint (grammar-constrained
    decoding, explicit num_ctx) — the OpenAI-compat json_schema path is
    ignored by some Ollama versions and silently truncates long prompts.
  * Even native grammar does NOT enforce `required`: judge calls run as a
    FIELD-COMPLETION loop — partial replies are kept and follow-ups ask
    only for the missing fields.
  * `_conform_to_schema` coerces types client-side (belt and braces).

Judge prompts/schemas live HERE (OD-D): they are reward code, not agent
prompts. Reason-first field order throughout: with structured decoding the
model emits fields in schema order, so the free-text reasoning must precede
the scores it drives.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

from qmix_report_writer.utils.config import get_config, get_llm_config
from qmix_report_writer.utils.globals import CompletionTokens, PromptTokens
from qmix_report_writer.utils.utils import safe_json_parse

logger = logging.getLogger("evaluation.judges")


class JudgeError(Exception):
    """A judge reply stayed unusable after the configured retries."""


# ---------------------------------------------------------------------------
# Score dataclasses (plan 2.1) — the training log consumes the decomposition.
# ---------------------------------------------------------------------------

@dataclass
class ChunkScore:
    """Grounded quality of ONE appended section, in [0, 1]."""
    score: float                            # combined: rubric x flag x grounding
    rubric_mean: float                      # audit rubric mean, [0, 1]
    logical_soundness: int = 0
    verifiability_score: int = 0
    technical_precision: int = 0
    info_density: int = 0
    hallucination_flag: bool = False
    grounding_ratio: Optional[float] = None  # supported/claims; None = no claim check
    n_claims: int = 0
    n_supported: int = 0
    n_unsupported: int = 0
    n_contradicted: int = 0
    notes: str = ""
    # The normalized {claim, verdict} items behind the counts — consumed by
    # the training log and the benchmark's grounding probe diagnostics.
    claims: list = field(default_factory=list)


@dataclass
class MacroScore:
    """Terminal whole-report quality, in [0, 1]."""
    score: float
    subject_coverage: int = 0
    global_flow: int = 0
    structural_score: int = 0
    tone_consistency: int = 0
    redundancy_avoidance: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Prompts & schemas (moved out of the agent prompt registry — OD-D)
# ---------------------------------------------------------------------------

_STRICTNESS = """### Strictness Clause
* **Demanding:** Do not award a 5 unless the text exceeds professional scientific standards. A score of 3 represents "minimum viable quality."
* **Negative Bias:** Look specifically for reasons to deduct points (e.g., hidden circular logic, generic "AI-style" filler, or lack of unique insight).

### Scoring Anchors:
* **5 (Elite):** Peer-review ready; no improvements possible.
* **3 (Average):** Clear, but contains minor redundancies or stylistic inconsistencies.
* **1 (Poor):** Significant logical gaps or heavy repetitive padding."""

CHUNK_AUDIT_PROMPT = f"""### Role
You are a Technical Auditor and Fact-Checker in a multi-stage review pipeline. Your job is to audit ONE chunk of a larger technical report AGAINST the source material stored for it.

### Inputs
You may receive: the commissioned subject (<subject>), the RAG source excerpts stored for this chunk (<sources> — the ONLY admissible evidence), a summary of the report so far (<progress summary>), and the chunk under audit (<current chunk>).

{_STRICTNESS}

### Scoring Rubric (Ground Truth):
* logical_soundness (0-5): 5 = Premises lead perfectly to conclusions; 0 = Logic is broken or "hallucinated."
* verifiability_score (0-5): 5 = Every substantive claim is supported by the provided sources; 0 = Claims contradict the sources or fabricate beyond them. When NO sources are provided, judge instead whether claims are inherently verifiable (fundamental laws, standard results).
* technical_precision (0-5): 5 = Exact terminology and units; 0 = Vague, incorrect, or misleading scientific terms.
* info_density (0-5): 5 = Straight to the point content; 0 = Fluff-heavy or content-free.

### Instructions:
* Judge the chunk against the provided sources — general knowledge does not substitute for them.
* Set hallucination_flag true only for claims that conflict with the sources or invent specifics beyond them.
* Write your local_audit_notes FIRST — they drive the scores — then assign the scores and the hallucination_flag. local_audit_notes must be ONE short paragraph as a single JSON string (2-3 sentences maximum): never a list, never markdown.
* Output your response as a JSON object matching the provided schema.
"""

CHUNK_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "local_audit_notes": {"type": "string", "description": "Very short audit notes on this chunk. Written before the scores."},
        "logical_soundness": {"type": "integer", "minimum": 0, "maximum": 5},
        "verifiability_score": {"type": "integer", "minimum": 0, "maximum": 5},
        "technical_precision": {"type": "integer", "minimum": 0, "maximum": 5},
        "info_density": {"type": "integer", "minimum": 0, "maximum": 5},
        "hallucination_flag": {"type": "boolean"},
    },
    "required": [
        "local_audit_notes", "logical_soundness", "verifiability_score",
        "technical_precision", "info_density", "hallucination_flag",
    ],
    "additionalProperties": False,
}
_CHUNK_AUDIT_KEYS = tuple(k for k in CHUNK_AUDIT_SCHEMA["required"]
                          if k != "local_audit_notes")

CLAIM_CHECK_PROMPT = """### Role
You are a claims-verification engine. Extract the substantive FACTUAL claims from the chunk and give a verdict for each one STRICTLY against the provided sources.

### What counts as a claim — extract ALL of them (up to 10)
* Every quantitative statement: numbers, ranges, units, dates, percentages, equations. These take PRIORITY: when the chunk asserts more claims than fit, quantitative claims are extracted before qualitative ones — they are the checkable ones.
* Every mechanism or causal assertion ("X causes / implies / depends on Y").
* Every named relationship, law, definition, or attribution.
An empty claims list is ONLY acceptable for a chunk that asserts nothing factual at all (pure transitions, headings, editorial framing). If the chunk contains numbers, units, or mechanisms, it contains claims — extract them.

### Verdicts
* supported     — the sources state the claim or directly entail it.
* unsupported   — the sources are SILENT on the claim: they neither confirm nor deny it.
* contradicted  — the sources say OTHERWISE: a different value for the same quantity, an opposite dependence or direction, an incompatible mechanism. Disagreement is contradiction, not absence — when the sources give a different number for the same quantity the verdict is `contradicted`, never `unsupported`.

### Rules
* Extract claims from the <current chunk> ONLY — the <sources> are the evidence to verdict against, never a source of claims.
* ONLY the provided sources count as evidence. A claim that is true general knowledge but absent from the sources is `unsupported` (or `contradicted` if the sources disagree with it).
* For EACH claim, first copy the sources' exact words on it into `evidence` (verbatim quote, digits copied EXACTLY as the source prints them; empty string if the sources are silent), THEN verdict by comparing the claim against your quote.
* When a claim contains a number, compare it DIGIT FOR DIGIT with the quoted evidence — a claim is only `supported` if the value matches exactly; a differing value is `contradicted`.
* Quote each claim briefly (one line), do not rewrite it.

### Output shape
Return a JSON object with ONE key `claims`: an array of OBJECTS, each carrying the keys `claim` (the one-line quote), `evidence` (the sources' exact words, or ""), and `verdict` (one of the three verdicts) — in that order. Never return parallel arrays, never bare strings.
Example: {"claims": [{"claim": "g is about 9.81 m/s^2", "evidence": "free-fall acceleration g is about 9.81 m/s^2", "verdict": "supported"}]}
"""

CLAIM_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "description": "Every substantive factual claim in the chunk, each "
                           "with its evidence quote and verdict. Empty ONLY "
                           "when the chunk asserts nothing factual.",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    # Evidence-before-verdict (live finding 2026-07-08): without
                    # a forced verbatim quote the judge rubber-stamps `supported`
                    # on near-duplicate sources without checking the digits.
                    "evidence": {"type": "string",
                                 "description": "The sources' exact words on "
                                                "this claim; empty if silent."},
                    "verdict": {"type": "string",
                                "enum": ["supported", "unsupported", "contradicted"]},
                },
                "required": ["claim", "evidence", "verdict"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_VERDICTS = ("supported", "unsupported", "contradicted")
_TRAILING_VERDICT_RE = re.compile(
    r"[\s\-—:(\[]*\b(supported|unsupported|contradicted)\b\s*[.!)\]]?\s*$",
    re.IGNORECASE,
)


def _normalize_claims(reply) -> list:
    """The usable {claim, verdict} items of a claim-check reply.

    Live-observed (2026-07-08): Ollama's grammar `format` is NOT enforced on
    the NESTED claim schema — the judge answered with parallel string arrays
    (`{"claims": [...], "verdicts": [...]}`), which the composer silently
    dropped: 0 claims counted, grounding inert, and the empty-claims guard
    blind because the list was non-empty. Normalize every shape seen or
    plausible: proper objects, parallel arrays, strings with a trailing
    verdict. Items whose verdict cannot be determined are dropped — a verdict
    is never guessed.
    """
    if not isinstance(reply, dict):
        return []
    raw = reply.get("claims")
    if not isinstance(raw, list):
        return []
    verdicts = reply.get("verdicts")
    parallel = verdicts if isinstance(verdicts, list) else []
    out = []
    for i, item in enumerate(raw):
        claim = verdict = None
        evidence = ""
        if isinstance(item, dict):
            claim = item.get("claim")
            verdict = item.get("verdict")
            evidence = str(item.get("evidence") or "")
        elif isinstance(item, str):
            claim = item
            if i < len(parallel):
                verdict = parallel[i]
            else:
                match = _TRAILING_VERDICT_RE.search(item)
                if match:
                    verdict = match.group(1)
                    claim = item[:match.start()].strip()
        verdict = re.sub(r"[^a-z]+", "", str(verdict or "").lower())
        if claim and verdict in _VERDICTS:
            out.append({"claim": str(claim), "evidence": evidence,
                        "verdict": verdict})
    return out


def _norm_for_match(text: str) -> str:
    """Case/whitespace/punctuation-insensitive matching alphabet (digits kept).

    NFKD first: PDF-extracted sources carry ligatures (ﬁ/ﬂ) and composed
    accents that the judge's quote renders as plain letters — both sides must
    land on the same alphabet or real quotes get falsely downgraded.
    """
    text = unicodedata.normalize("NFKD", str(text or ""))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _verify_evidence(items: list, sources: list) -> list:
    """Downgrade supported/contradicted verdicts whose evidence is NOT in the
    sources (to `unsupported`).

    Live-observed (2026-07-08): when the chunk and a source carry near-twin
    sentences differing only in digits, the judge COPIES THE CLAIM as its own
    "evidence" and verdicts `supported` without ever looking the value up —
    no prompt wording fixed it. The evidence field is a verbatim source quote
    by contract, so its presence in the sources is machine-checkable: a
    fabricated quote must not count as support (nor as contradiction). The
    original verdict is kept as `verdict_raw` for diagnostics.
    """
    haystack = _norm_for_match(" ".join(
        (src.get("content") or src.get("text") or "") if isinstance(src, dict)
        else str(src)
        for src in sources or []
    ))
    out = []
    for item in items:
        if item["verdict"] in ("supported", "contradicted"):
            evidence = _norm_for_match(item.get("evidence", ""))
            if not evidence or evidence not in haystack:
                item = dict(item, verdict="unsupported",
                            verdict_raw=item["verdict"])
        out.append(item)
    return out


# Empty-claims guard (plan 2.9): grammar-constrained decoding lets the model
# emit a perfectly valid `{"claims": []}`, which silently disables TD3's
# grounding (empty -> factor 1.0). When the chunk visibly carries factual
# content, an extraction with no USABLE claims is treated as under-firing and
# re-asked once. Markers: digits (values, dates, percentages) or inline math.
# Mechanism-only chunks without numbers are not deterministically detectable —
# for those the strengthened prompt is the only line of defense.
_FACTUAL_MARKER_RE = re.compile(r"\d|\$[^$\n]+\$")

_EMPTY_CLAIMS_NUDGE = (
    "You returned an empty claims list, but the chunk contains numbers, units, "
    "or quantitative statements — those ARE factual claims. Re-read the chunk "
    "and extract every quantitative or mechanistic assertion, each with its "
    "verdict against the sources. Return an empty list ONLY if the chunk truly "
    "asserts nothing factual."
)


def _has_factual_markers(text: str) -> bool:
    return bool(_FACTUAL_MARKER_RE.search(text or ""))


MACRO_PROMPT = f"""### Role
You are a Senior Scientific Editor and Content Architect. Your goal is to evaluate the structural integrity and high-level quality of a finished technical report.

### Inputs
You may receive: the commissioned subject (<subject>), the planned outline (<outline> — may be empty), and the full report (<report>). Analyze the report as a whole: fit to the commissioned subject and outline, narrative arc, tone consistency, and overall utility for a learner or researcher.

{_STRICTNESS}

### Scoring Rubric
* subject_coverage (0-5): 5 = The commissioned subject (and outline, when given) is covered in depth; 0 = Off-topic or misinterpretation of the subject.
* global_flow (0-5): 5 = Seamless transitions between concepts; 0 = Subject jumps or disconnected sections.
* structural_score (0-5): 5 = Follows standard scientific/pedagogical hierarchy; 0 = Chaotic or illogical organization.
* tone_consistency (0-5): 5 = Stable "voice" throughout; 0 = Shifts randomly between academic, casual, or marketing speak.
* redundancy_avoidance (0-5): 5 = Every section adds new value; 0 = Significant repetitive padding.

### Instructions
Write your global_reasoning notes FIRST — they drive the scores — then assign the scores. global_reasoning must be ONE short paragraph as a single JSON string (2-3 sentences maximum): never a list, never markdown. Output your final evaluation in the requested JSON format.
"""

MACRO_SCHEMA = {
    "type": "object",
    "properties": {
        "global_reasoning": {"type": "string", "description": "Very short notes justifying the scores. Written before the scores."},
        "subject_coverage": {"type": "integer", "minimum": 0, "maximum": 5},
        "global_flow": {"type": "integer", "minimum": 0, "maximum": 5},
        "structural_score": {"type": "integer", "minimum": 0, "maximum": 5},
        "tone_consistency": {"type": "integer", "minimum": 0, "maximum": 5},
        "redundancy_avoidance": {"type": "integer", "minimum": 0, "maximum": 5},
    },
    "required": [
        "global_reasoning", "subject_coverage", "global_flow",
        "structural_score", "tone_consistency", "redundancy_avoidance",
    ],
    "additionalProperties": False,
}
_MACRO_KEYS = tuple(k for k in MACRO_SCHEMA["required"] if k != "global_reasoning")


# ---------------------------------------------------------------------------
# Transport + call discipline (moved from experiments/eval.py, plan 1.2/1.3)
# ---------------------------------------------------------------------------

def _judge_cfg() -> dict:
    return (get_config().get("reward", {}) or {}).get("judge", {}) or {}


class _NativeJudgeLLM:
    """Judge calls through Ollama's NATIVE /api/chat structured outputs.

    The OpenAI-compat /v1/chat/completions path ignores `response_format:
    json_schema` on the deployed Ollama AND silently truncates prompts to the
    model's default num_ctx (4096) — the macro judge's prompt is 10-15k
    tokens. The native API fixes both: `format` = grammar-constrained
    decoding, `options.num_ctx` = explicit context.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def _base_url(self) -> str:
        providers = get_llm_config().get("providers", {}) or {}
        return (providers.get("ollama", {}) or {}).get(
            "base_url", "http://localhost:11434"
        ).rstrip("/")

    async def agen(self, messages, max_tokens=None, temperature=None,
                   response_schema=None, **_):
        cfg = _judge_cfg()
        payload = {
            "model": self.model_name,
            "messages": [
                m if isinstance(m, dict) else {"role": m.role, "content": m.content}
                for m in messages
            ],
            "stream": False,
            "options": {
                "temperature": 0.0 if temperature is None else float(temperature),
                "num_predict": int(max_tokens or cfg.get("max_tokens", 3072)),
                "num_ctx": int(cfg.get("num_ctx", 32768)),
            },
        }
        if response_schema:
            payload["format"] = response_schema

        timeout = aiohttp.ClientTimeout(total=600, connect=60, sock_read=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self._base_url()}/api/chat",
                                    json=payload) as response:
                data = await response.json()
                if "message" not in data:
                    raise JudgeError(
                        f"native judge call failed: {str(data)[:200]}"
                    )
                # Token accounting parity with achat_ollama.
                PromptTokens.instance().value += data.get("prompt_eval_count", 0)
                CompletionTokens.instance().value += data.get("eval_count", 0)
                return data["message"].get("content") or ""


def _conform_to_schema(parsed, schema):
    """Coerce a judge reply's values to the schema's declared types.

    Ollama does not reliably enforce the response schema, so conformance is
    done client-side: lists→joined strings, numeric strings→ints, ints
    clamped to the schema's min/max. Returns None when a value cannot be
    conformed — the caller then retries (never scores a malformed reply).
    """
    if not isinstance(parsed, dict):
        return None
    props = schema.get("properties", {})
    out = {}
    for key, val in parsed.items():
        spec = props.get(key, {})
        expected = spec.get("type")
        if expected == "string":
            if isinstance(val, list):
                val = " ".join(str(item) for item in val)
            elif not isinstance(val, str):
                val = str(val)
        elif expected == "integer":
            if isinstance(val, bool):
                return None
            if not isinstance(val, int):
                try:
                    val = int(round(float(val)))
                except (TypeError, ValueError):
                    return None
            lo, hi = spec.get("minimum"), spec.get("maximum")
            if lo is not None:
                val = max(lo, val)
            if hi is not None:
                val = min(hi, val)
        elif expected == "boolean":
            if not isinstance(val, bool):
                if isinstance(val, str) and val.strip().lower() in ("true", "false"):
                    val = val.strip().lower() == "true"
                else:
                    return None
        out[key] = val
    return out


def _sub_schema(schema, keys):
    """The schema restricted to `keys` (for field-completion follow-ups)."""
    props = schema.get("properties", {})
    return {
        "type": "object",
        "properties": {k: props[k] for k in keys if k in props},
        "required": [k for k in keys if k in props],
        "additionalProperties": False,
    }


async def _judge_call(llm, messages, schema, required_keys):
    """Judge call as a FIELD-COMPLETION loop: temperature 0, capped tokens.

    Ollama's grammar-constrained decoding does not enforce `required` — the
    model may close the object after the first field (live-observed: a clean
    reasoning-only reply, done_reason=stop). So every attempt KEEPS the
    conformed fields it got and re-asks only for the missing ones (after a
    reasoning-only reply that is a scores-only sub-schema — reason-then-score
    preserved, the reasoning fed back as context). An attempt that makes no
    progress is resampled at temp >= 0.35; re-asking an identical prompt at
    temperature 0 would deterministically reproduce the same reply.
    """
    cfg = _judge_cfg()
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 3072))
    retries = int(cfg.get("retries", 2))

    all_keys = list(schema.get("properties", {}).keys())
    merged = {}
    made_progress = True
    last = ""
    for attempt in range(retries + 1):
        missing = [k for k in all_keys if k not in merged]
        attempt_messages = list(messages)
        if merged:
            attempt_messages.append({
                "role": "user",
                "content": (
                    "Your previous reply was incomplete. It already provided:\n"
                    + json.dumps(merged)
                    + "\nNow output ONLY the missing fields as a JSON object: "
                    + ", ".join(missing)
                ),
            })
        attempt_temperature = (
            temperature if (attempt == 0 or made_progress)
            else max(temperature, 0.35)
        )
        last = await llm.agen(
            attempt_messages,
            max_tokens=max_tokens,
            temperature=attempt_temperature,
            response_schema=_sub_schema(schema, missing) if merged else schema,
        )
        # quiet=True: an unparseable attempt is routine here (truncation at
        # num_predict) — the completion loop retries and JudgeError carries
        # the raw head; the CRITICAL print only alarmed live-run consoles.
        conformed = _conform_to_schema(safe_json_parse(last, quiet=True), schema)
        made_progress = False
        if conformed:
            for key, val in conformed.items():
                if key not in merged:
                    merged[key] = val
                    made_progress = True
        if all(k in merged for k in required_keys):
            return merged
    raise JudgeError(
        f"incomplete judge reply after {retries + 1} attempt(s) "
        f"(got {sorted(merged.keys())}, last raw: {str(last)[:120]}...)"
    )


# ---------------------------------------------------------------------------
# Prompt assembly helpers
# ---------------------------------------------------------------------------

def _render_sources(sources) -> str:
    blocks = []
    for i, src in enumerate(sources or [], 1):
        if isinstance(src, dict):
            name = src.get("source") or f"source_{i}"
            page = src.get("page")
            body = src.get("content") or src.get("text") or ""
            tag = f"{name}, p.{page}" if page is not None else str(name)
        else:
            tag, body = f"source_{i}", str(src)
        blocks.append(f"<source id={i} ({tag})>\n{body}\n</source>")
    return "\n".join(blocks)


def _tagged(tag: str, body: str) -> str:
    return f"<{tag}>\n{body}\n</{tag}>\n"


# ---------------------------------------------------------------------------
# The evaluator (plan 2.1) — the object the controller/runner consume.
# ---------------------------------------------------------------------------

class ReportEvaluator:
    """Grounded chunk scoring + terminal macro scoring (TD1/TD3).

    `llm` is injectable for tests; by default the judge model from
    `reward.judge.model` (falling back to `llm.default_model`) is served
    through the native structured-output transport. Both scoring methods
    return None on judge failure — the caller skips the reward event.
    """

    def __init__(self, llm=None):
        if llm is None:
            model = _judge_cfg().get("model") or get_llm_config().get(
                "default_model", "qwen3:8b"
            )
            llm = _NativeJudgeLLM(model)
        self.llm = llm

    # -- micro: grounded chunk audit + claim check (plan 2.2) ---------------

    async def score_chunk(self, chunk: str, sources: list, task: str,
                          context: Optional[str] = None) -> Optional[ChunkScore]:
        user = ""
        if task:
            user += _tagged("subject", task)
        if sources:
            user += _tagged("sources", _render_sources(sources))
        if context:
            user += _tagged("progress summary", context)
        user += _tagged("current chunk", chunk)

        try:
            audit = await _judge_call(
                self.llm,
                [{"role": "system", "content": CHUNK_AUDIT_PROMPT},
                 {"role": "user", "content": user}],
                CHUNK_AUDIT_SCHEMA,
                _CHUNK_AUDIT_KEYS,
            )
            claims = None
            if sources:
                claim_user = _tagged("sources", _render_sources(sources))
                claim_user += _tagged("current chunk", chunk)
                claim_messages = [
                    {"role": "system", "content": CLAIM_CHECK_PROMPT},
                    {"role": "user", "content": claim_user},
                ]
                claims = await _judge_call(
                    self.llm, claim_messages, CLAIM_CHECK_SCHEMA, ("claims",),
                )
                if not _normalize_claims(claims) and _has_factual_markers(chunk):
                    # No USABLE claims (empty OR all-unusable items) on a
                    # visibly factual chunk — under-extraction, not absence
                    # (plan 2.9): re-ask once with the evidence pointed out.
                    # A failed re-ask keeps the first result — it must not
                    # turn an accepted (if lazy) reply into a skipped event.
                    try:
                        claims = await _judge_call(
                            self.llm,
                            claim_messages + [
                                {"role": "assistant",
                                 "content": json.dumps({"claims": []})},
                                {"role": "user", "content": _EMPTY_CLAIMS_NUDGE},
                            ],
                            CLAIM_CHECK_SCHEMA,
                            ("claims",),
                        )
                    except JudgeError:
                        logger.warning(
                            "empty-claims re-ask failed; keeping the empty "
                            "extraction (no grounding modulation)"
                        )
        except JudgeError as exc:
            logger.warning(f"score_chunk judge failure: {exc}")
            return None

        claim_items = None
        if claims is not None:
            claim_items = _verify_evidence(_normalize_claims(claims), sources)
        return self._compose_chunk_score(audit, claim_items)

    @staticmethod
    def _compose_chunk_score(audit: dict,
                             claim_items: Optional[list]) -> ChunkScore:
        logic = int(audit.get("logical_soundness", 0))
        verif = int(audit.get("verifiability_score", 0))
        tech = int(audit.get("technical_precision", 0))
        density = int(audit.get("info_density", 0))
        flag = bool(audit.get("hallucination_flag", False))
        rubric_mean = (logic + verif + tech + density) / 20.0

        n_sup = n_unsup = n_contra = 0
        claim_items = claim_items or []
        for item in claim_items:
            if item["verdict"] == "supported":
                n_sup += 1
            elif item["verdict"] == "unsupported":
                n_unsup += 1
            elif item["verdict"] == "contradicted":
                n_contra += 1
        n_claims = n_sup + n_unsup + n_contra

        # Grounding modulation: supported = full credit, unsupported = half
        # (absence of evidence), contradicted = none — contradiction hurts
        # more than absence (TD3). No claim check / no claims -> no modulation.
        grounding_ratio = (n_sup / n_claims) if n_claims else None
        factor = ((n_sup + 0.5 * n_unsup) / n_claims) if n_claims else 1.0

        score = rubric_mean * (0.5 if flag else 1.0) * factor
        return ChunkScore(
            score=score,
            rubric_mean=rubric_mean,
            logical_soundness=logic,
            verifiability_score=verif,
            technical_precision=tech,
            info_density=density,
            hallucination_flag=flag,
            grounding_ratio=grounding_ratio,
            n_claims=n_claims,
            n_supported=n_sup,
            n_unsupported=n_unsup,
            n_contradicted=n_contra,
            notes=str(audit.get("local_audit_notes", ""))[:500],
            claims=claim_items,
        )

    # -- macro: terminal whole-report judge (plan 2.3) ----------------------

    async def score_report(self, task: str, outline: List[str],
                           report: str) -> Optional[MacroScore]:
        user = ""
        if task:
            user += _tagged("subject", task)
        if outline:
            user += _tagged("outline", "\n".join(f"- {t}" for t in outline))
        user += _tagged("report", report)

        try:
            macro = await _judge_call(
                self.llm,
                [{"role": "system", "content": MACRO_PROMPT},
                 {"role": "user", "content": user}],
                MACRO_SCHEMA,
                _MACRO_KEYS,
            )
        except JudgeError as exc:
            logger.warning(f"score_report judge failure: {exc}")
            return None

        fields = {k: int(macro.get(k, 0)) for k in _MACRO_KEYS}
        return MacroScore(
            score=sum(fields.values()) / 25.0,
            notes=str(macro.get("global_reasoning", ""))[:800],
            **fields,
        )

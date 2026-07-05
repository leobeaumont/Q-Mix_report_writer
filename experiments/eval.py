import json
from typing import Optional

import aiohttp
import numpy as np

from qmix_report_writer.utils.globals import (
    CompletionTokens, PromptTokens, ReportState, Score,
)
from qmix_report_writer.utils.config import get_config, get_llm_config
from qmix_report_writer.utils.utils import safe_json_parse
from qmix_report_writer.prompt.prompt_set_registry import PromptSetRegistry

_MACRO_KEYS = ("subject_coverage", "global_flow", "structural_score",
               "tone_consistency", "redundancy_avoidance")
_MICRO_KEYS = ("logical_soundness", "verifiability_score", "technical_precision",
               "info_density", "hallucination_flag")


class JudgeError(Exception):
    """A judge reply stayed unusable after the configured retries.

    Raised instead of scoring 0 (training_eval plan 1.3): the caller skips the
    reward event — a transport/parse hiccup must never enter the score history.
    """


def _judge_cfg() -> dict:
    return (get_config().get("reward", {}) or {}).get("judge", {}) or {}


class _NativeJudgeLLM:
    """Judge calls through Ollama's NATIVE /api/chat structured outputs.

    The OpenAI-compat /v1/chat/completions path ignores `response_format:
    json_schema` on the deployed Ollama (live-observed: arrays for strings,
    raw markdown, XML tags) AND silently truncates prompts to the model's
    default num_ctx (4096) — the macro judge's prompt is 10-15k tokens, so
    the system prompt with all scoring instructions was being dropped.
    The native API fixes both: `format` = grammar-constrained decoding
    (malformed JSON is impossible), `options.num_ctx` = explicit context.
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


def _judge_llm():
    """The judge LLM — reward.judge.model, falling back to the pipeline default.

    Deliberately independent of the actors' --llm flag (plan 1.2), and served
    through the native structured-output endpoint (see _NativeJudgeLLM).
    """
    model = _judge_cfg().get("model") or get_llm_config().get(
        "default_model", "qwen3:8b"
    )
    return _NativeJudgeLLM(model)


def _conform_to_schema(parsed, schema):
    """Coerce a judge reply's values to the schema's declared types.

    Ollama does not reliably enforce the response schema (a live run returned
    the notes field as a JSON array), so conformance is done client-side:
    lists→joined strings, numeric strings→ints, ints clamped to the schema's
    min/max. Returns None when a value cannot be conformed — the caller then
    retries (never scores a malformed reply).
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
    reasoning-only reply, done_reason=stop). So instead of discarding partial
    replies, every attempt KEEPS the conformed fields it got and re-asks only
    for the missing ones (after a reasoning-only reply that is a scores-only
    sub-schema — reason-then-score preserved, the reasoning is fed back as
    context). An attempt that makes no progress is resampled at temp >= 0.35;
    re-asking an identical prompt at temperature 0 would deterministically
    reproduce the same reply.
    """
    cfg = _judge_cfg()
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 2048))
    retries = int(cfg.get("retries", 1))

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
        conformed = _conform_to_schema(safe_json_parse(last), schema)
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


def length_score(target, sigma) -> float:
    """Score the length of the production between 0 and 1."""
    length = len(ReportState.instance().content)

    return np.exp(-0.5 * ((length - target) / sigma)**2)


async def report_score(task: Optional[str] = None) -> float:
    """Score the quality of the report between 0 and 1.

    `task` is the commissioned subject; when given, the macro judge scores
    subject_coverage against it (plan 1.1 — was defect A0.1).
    Raises JudgeError when a judge reply stays unparseable (plan 1.3).
    """
    llm = _judge_llm()

    prompt_set = PromptSetRegistry.get("redacting")

    # Macro scoring

    system_prompt = prompt_set.get_description("Macro Scoring")
    user_prompt = ""
    if task:
        user_prompt += "<subject>\n" + task + "\n</subject>\n"
    user_prompt += "<report>\n" + ReportState.instance().content + "\n</report>"
    schema = prompt_set.get_schema("Macro Scoring")

    message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    macro_scores = await _judge_call(llm, message, schema, _MACRO_KEYS)

    coverage_score = macro_scores.get("subject_coverage", 0)
    flow_score = macro_scores.get("global_flow", 0)
    structural_score = macro_scores.get("structural_score", 0)
    tone_score = macro_scores.get("tone_consistency", 0)
    redundancy_score = macro_scores.get("redundancy_avoidance", 0)

    # Truncate notes to avoid token explosions
    global_notes = macro_scores.get("global_reasoning", "[NO GLOBAL ANALYSIS]")[:800]

    # Micro scoring

    score_memory = Score.instance()

    system_prompt = prompt_set.get_description("Micro Scoring")

    user_prompt = "<document summary>\n" + ReportState.instance().progress + "\n<document summary>\n"
    user_prompt += "<global notes>\n" + global_notes + "\n</global notes>\n"
    user_prompt += "<audit history>\n"
    # Only take the last 3 notes to keep the prompt size stable
    history_window = score_memory.micro_notes[-3:]
    for i, notes in enumerate(history_window):
        # str() belt-and-braces: _conform_to_schema guarantees strings for new
        # entries, but the history must never be able to crash the prompt.
        user_prompt += f"<chunk {i} notes>\n" + str(notes) + f"\n</chunk {i} notes>\n"
    user_prompt += "</audit history>\n"

    current_chunk = ReportState.instance().additions[-1] if ReportState.instance().additions else ReportState.instance().content
    user_prompt += "<current chunk>\n" + current_chunk + "\n</current chunk>\n"

    schema = prompt_set.get_schema("Micro Scoring")

    message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    micro_scores = await _judge_call(llm, message, schema, _MICRO_KEYS)

    logic_score = micro_scores.get("logical_soundness", 0)
    verifiability_score = micro_scores.get("verifiability_score", 0)
    technicality_score = micro_scores.get("technical_precision", 0)
    density_score = micro_scores.get("info_density", 0)

    hallucination_flag = micro_scores.get("hallucination_flag", False)

    # Truncate note to avoid token explosion
    micro_analysis = micro_scores.get("local_audit_notes", "[NO LOCAL ANALYSIS]")[:500]

    # Calculations

    macro_score = (coverage_score + flow_score + structural_score + tone_score + redundancy_score) / 25

    current_chunk_score = (logic_score + verifiability_score + technicality_score + density_score) / 20
    if hallucination_flag:
        current_chunk_score /= 2

    score_memory.micro_notes.append(micro_analysis)
    score_memory.micro_scores.append(current_chunk_score)

    micro_score = np.average(score_memory.micro_scores)

    return 0.3 * macro_score + 0.7 * micro_score


if __name__ == "__main__":
    import asyncio

    ReportState.instance().append("Main text", "Summary of the main text")

    asyncio.run(report_score(task="A smoke-test subject"))

    ReportState.instance().append(" with a little more", "Summary of the main text with a little more")

    asyncio.run(report_score(task="A smoke-test subject"))

from typing import Optional

import numpy as np

from qmix_report_writer.utils.globals import ReportState, Score
from qmix_report_writer.utils.config import get_config, get_llm
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


def _judge_llm():
    """The judge LLM — reward.judge.model, falling back to the pipeline default.

    Deliberately independent of the actors' --llm flag (plan 1.2).
    """
    return get_llm(_judge_cfg().get("model") or None)


async def _judge_call(llm, messages, schema, required_keys):
    """One judge call: temperature 0, capped tokens, retry-then-raise parsing."""
    cfg = _judge_cfg()
    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 2048))
    retries = int(cfg.get("retries", 1))

    last = ""
    for _ in range(retries + 1):
        last = await llm.agen(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_schema=schema,
        )
        parsed = safe_json_parse(last)
        if isinstance(parsed, dict) and all(k in parsed for k in required_keys):
            return parsed
    raise JudgeError(
        f"unusable judge reply after {retries + 1} attempt(s): {str(last)[:120]}..."
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
        user_prompt += f"<chunk {i} notes>\n" + notes + f"\n</chunk {i} notes>\n"
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

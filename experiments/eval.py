import numpy as np
import json
import re

from utils.globals import ReportState, Score
from utils.config import get_llm
from utils.utils import safe_json_parse
from prompt.prompt_set_registry import PromptSetRegistry

def length_score(target, sigma) -> float:
    """Score the length of the production between 0 and 1."""
    length = len(ReportState.instance().content)

    return np.exp(-0.5 * ((length - target) / sigma)**2)

async def report_score() -> float:
    """Score the quality of the report between 0 and 1."""
    llm = get_llm()

    prompt_set = PromptSetRegistry.get("redacting")

    # Macro scoring

    system_prompt = prompt_set.get_description("Macro Scoring")
    user_prompt = "<report>\n" + ReportState.instance().content + "\n</report>"
    schema = prompt_set.get_schema("Macro Scoring")

    message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    response = await llm.agen(message, response_schema=schema)
    macro_scores = safe_json_parse(response)

    coverage_score = macro_scores.get("subject_coverage", 0)
    flow_score = macro_scores.get("global_flow", 0)
    structural_score = macro_scores.get("structural_score", 0)
    tone_score = macro_scores.get("tone_consistency", 0)
    redundancy_score = macro_scores.get("redundancy_penalty", 0)

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
    
    response = await llm.agen(message, response_schema=schema)
    micro_scores = safe_json_parse(response)

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

    asyncio.run(report_score())

    ReportState.instance().append(" with a little more", "Summary of the main text with a little more")

    asyncio.run(report_score())

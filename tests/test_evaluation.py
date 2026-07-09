"""
Offline unit tests for qmix_report_writer/evaluation (training_eval plan 2.7).

Covers the semantic contracts the acceptance suite deliberately leaves to
this file (its stubs are schema-fillers and cannot steer verdicts):

  * grounding ordering: same audit, claims all-supported > all-unsupported >
    all-contradicted (contradiction hurts more than absence — TD3);
  * hallucination flag halves the rubric mean;
  * no sources -> single call, no grounding modulation;
  * no factual claims -> no grounding penalty;
  * empty-claims guard (plan 2.9): factual markers in the chunk -> one re-ask
    whose reply drives grounding; marker-free / confirmed-empty / failed
    re-ask -> empty accepted, never a penalty or a skipped event;
  * judge failure -> score_chunk/score_report return None (never raise);
  * macro composition + prompt contents (task + outline).

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_evaluation.py
"""

import asyncio
import json
import sys

sys.path.insert(0, ".")

from qmix_report_writer.evaluation import ChunkScore, MacroScore, ReportEvaluator
from qmix_report_writer.evaluation.judges import _judge_cfg

SOURCES = [{"source": "paper.pdf", "page": 3,
            "content": "The measured mass is 5 GeV at 300 K."}]


class _ScriptedLLM:
    """Returns scripted replies in order; records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def agen(self, messages, max_tokens=None, temperature=None,
                   response_schema=None, **kwargs):
        self.calls.append({"messages": messages, "schema": response_schema,
                           "temperature": temperature})
        if not self.responses:
            raise AssertionError("unexpected extra judge call")
        return self.responses.pop(0)


def _audit_reply(scores=4, flag=False):
    return json.dumps({
        "local_audit_notes": "concise audit",
        "logical_soundness": scores,
        "verifiability_score": scores,
        "technical_precision": scores,
        "info_density": scores,
        "hallucination_flag": flag,
    })


def _claims_reply(verdict, n=4, evidence="The measured mass is 5 GeV at 300 K."):
    # Default evidence = the SOURCES content: supported/contradicted verdicts
    # only count when their evidence quote is actually found in the sources.
    return json.dumps({
        "claims": [{"claim": f"claim {i}", "evidence": evidence,
                    "verdict": verdict} for i in range(n)],
    })


def _score_chunk(responses, sources=SOURCES, chunk="The mass is 5 GeV.",
                 return_llm=False):
    llm = _ScriptedLLM(responses)
    evaluator = ReportEvaluator(llm=llm)
    result = asyncio.run(evaluator.score_chunk(
        chunk=chunk, sources=sources,
        task="Particle masses", context="progress",
    ))
    return (result, llm) if return_llm else result


def test_grounding_verdict_ordering():
    supported = _score_chunk([_audit_reply(), _claims_reply("supported")])
    unsupported = _score_chunk([_audit_reply(), _claims_reply("unsupported")])
    contradicted = _score_chunk([_audit_reply(), _claims_reply("contradicted")])

    assert isinstance(supported, ChunkScore)
    assert supported.score > unsupported.score > contradicted.score, (
        f"expected supported > unsupported > contradicted, got "
        f"{supported.score} / {unsupported.score} / {contradicted.score}"
    )
    # Exact factors: supported=1.0, unsupported=0.5, contradicted=0.0.
    assert abs(supported.score - supported.rubric_mean) < 1e-9
    assert abs(unsupported.score - 0.5 * supported.rubric_mean) < 1e-9
    assert contradicted.score == 0.0
    assert supported.grounding_ratio == 1.0
    assert contradicted.grounding_ratio == 0.0
    assert supported.n_claims == 4 and supported.n_supported == 4
    print("PASS  test_grounding_verdict_ordering")


def test_hallucination_flag_halves():
    clean = _score_chunk([_audit_reply(flag=False), _claims_reply("supported")])
    flagged = _score_chunk([_audit_reply(flag=True), _claims_reply("supported")])
    assert abs(flagged.score - clean.score / 2) < 1e-9
    assert flagged.hallucination_flag is True
    print("PASS  test_hallucination_flag_halves")


def test_no_sources_single_call():
    llm = _ScriptedLLM([_audit_reply()])
    evaluator = ReportEvaluator(llm=llm)
    result = asyncio.run(evaluator.score_chunk(
        chunk="Text.", sources=[], task="T", context=None,
    ))
    assert len(llm.calls) == 1, "no-sources chunk must skip the claim check"
    assert result.grounding_ratio is None
    assert abs(result.score - result.rubric_mean) < 1e-9
    print("PASS  test_no_sources_single_call")


def test_no_claims_no_penalty():
    # Marker-free chunk (no digits/math): an empty extraction is accepted
    # as-is — exactly two calls, no re-ask, no penalty.
    result, llm = _score_chunk(
        [_audit_reply(), json.dumps({"claims": []})],
        chunk="This section introduces the topic and outlines the approach.",
        return_llm=True,
    )
    assert len(llm.calls) == 2, "marker-free chunk must not trigger the re-ask"
    assert result.n_claims == 0 and result.grounding_ratio is None
    assert abs(result.score - result.rubric_mean) < 1e-9, \
        "a chunk with no factual claims must not be grounding-penalized"
    print("PASS  test_no_claims_no_penalty")


def test_empty_claims_reask_on_markers():
    # Chunk WITH factual markers + empty extraction -> one re-ask whose reply
    # drives the grounding factor (plan 2.9: grounding must not silently no-op).
    result, llm = _score_chunk(
        [_audit_reply(), json.dumps({"claims": []}), _claims_reply("supported", n=2)],
        return_llm=True,
    )
    assert len(llm.calls) == 3, "empty claims on a factual chunk must re-ask once"
    assert result.n_claims == 2 and result.grounding_ratio == 1.0
    reask = llm.calls[2]["messages"]
    assert any("empty claims list" in str(m.get("content", "")) for m in reask), \
        "the re-ask must carry the nudge"
    assert any(m.get("role") == "assistant" for m in reask), \
        "the re-ask must include the model's empty reply as context"
    print("PASS  test_empty_claims_reask_on_markers")


def test_empty_claims_reask_still_empty():
    # A confirmed-empty re-ask is accepted: no penalty, no further calls.
    result, llm = _score_chunk(
        [_audit_reply(), json.dumps({"claims": []}), json.dumps({"claims": []})],
        return_llm=True,
    )
    assert len(llm.calls) == 3
    assert result.n_claims == 0 and result.grounding_ratio is None
    assert abs(result.score - result.rubric_mean) < 1e-9
    print("PASS  test_empty_claims_reask_still_empty")


def test_parallel_array_claims_normalized():
    # The EXACT live failure of 2026-07-08 (judge_smoke): grammar format not
    # enforced on the nested schema — parallel string arrays instead of an
    # array of objects. Must be normalized, count as claims, and NOT trigger
    # the re-ask. Evidence-less verdicts are conservatively downgraded to
    # `unsupported` (support/contradiction require a verifiable quote).
    reply = json.dumps({
        "claims": ["g is about 9.81 m/s^2.", "The acceleration is mass-independent."],
        "verdicts": ["supported", "contradicted"],
    })
    result, llm = _score_chunk([_audit_reply(), reply], return_llm=True)
    assert len(llm.calls) == 2, "usable parallel-array claims must not re-ask"
    assert result.n_claims == 2
    assert result.n_supported == 0 and result.n_unsupported == 2
    assert result.grounding_ratio == 0.0
    assert abs(result.score - result.rubric_mean * 0.5) < 1e-9
    assert result.claims == [
        {"claim": "g is about 9.81 m/s^2.", "evidence": "",
         "verdict": "unsupported", "verdict_raw": "supported"},
        {"claim": "The acceleration is mass-independent.", "evidence": "",
         "verdict": "unsupported", "verdict_raw": "contradicted"},
    ], "ChunkScore.claims must carry the verified items with raw verdicts"
    print("PASS  test_parallel_array_claims_normalized")


def test_trailing_verdict_strings_normalized():
    # Bare strings with a trailing verdict ("... — supported") are usable too;
    # evidence-less, so their verdicts downgrade to `unsupported`.
    reply = json.dumps({"claims": [
        "The mass is 5 GeV — supported",
        "The temperature is 300 K (contradicted)",
    ]})
    result = _score_chunk([_audit_reply(), reply])
    assert result.n_claims == 2
    assert result.n_unsupported == 2 and result.n_supported == 0
    print("PASS  test_trailing_verdict_strings_normalized")


def test_fabricated_evidence_downgraded():
    # The 2026-07-08 rubber-stamp: `supported` whose "evidence" does not occur
    # in the sources (the judge copied the claim) must NOT count as support;
    # a real quote (case/whitespace/punctuation-insensitive) must survive.
    reply = json.dumps({"claims": [
        {"claim": "The mass is 5 GeV.",
         "evidence": "The measured   MASS is 5 GeV, at 300 K",  # real, messy
         "verdict": "supported"},
        {"claim": "The spin is 2.",
         "evidence": "The measured spin is 2.",  # fabricated: not in sources
         "verdict": "supported"},
        {"claim": "The charge is 0.",
         "evidence": "",  # no quote at all
         "verdict": "contradicted"},
    ]})
    result = _score_chunk([_audit_reply(), reply])
    assert result.n_claims == 3
    assert result.n_supported == 1, "verbatim-quote support must survive"
    assert result.n_unsupported == 2, "fabricated/absent evidence must downgrade"
    assert result.n_contradicted == 0
    assert result.claims[1]["verdict_raw"] == "supported"
    assert result.claims[2]["verdict_raw"] == "contradicted"
    print("PASS  test_fabricated_evidence_downgraded")


def test_unusable_claims_reask_on_markers():
    # Non-empty but ALL-unusable items (objects without verdicts) on a factual
    # chunk are under-extraction like an empty list: re-ask once.
    unusable = json.dumps({"claims": [{"claim": "the mass is 5 GeV"}]})
    result, llm = _score_chunk(
        [_audit_reply(), unusable, _claims_reply("supported", n=2)],
        return_llm=True,
    )
    assert len(llm.calls) == 3, "all-unusable claims must trigger the re-ask"
    assert result.n_claims == 2 and result.grounding_ratio == 1.0
    print("PASS  test_unusable_claims_reask_on_markers")


def test_empty_claims_reask_failure_keeps_empty():
    # A re-ask that stays garbage must NOT turn an accepted (if lazy) empty
    # extraction into a skipped reward event — fall back to no modulation.
    attempts = int(_judge_cfg().get("retries", 2)) + 1
    result, llm = _score_chunk(
        [_audit_reply(), json.dumps({"claims": []})] + ["not json {{{"] * attempts,
        return_llm=True,
    )
    assert result is not None, "a failed re-ask must not fail the chunk score"
    assert result.n_claims == 0 and abs(result.score - result.rubric_mean) < 1e-9
    print("PASS  test_empty_claims_reask_failure_keeps_empty")


def test_latex_escapes_in_judge_reply():
    # Live failure (5.3 smoke #3): the judge writes raw LaTeX inside the JSON
    # ("$\mu_B$") — an invalid \escape that killed json.loads on EVERY retry,
    # skipping the reward event. safe_json_parse must repair lone backslashes
    # (valid escapes like \n and \\ untouched) so the reply scores normally.
    audit = (
        '{"local_audit_notes": "Correct sign problem at finite $\\mu_B$ and '
        '\\alpha decay;\\nvalid escape kept.", "logical_soundness": 4, '
        '"verifiability_score": 4, "technical_precision": 4, '
        '"info_density": 4, "hallucination_flag": false}'
    )
    assert "\\m" in audit and "\\a" in audit  # genuinely invalid escapes
    result = _score_chunk([audit, _claims_reply("supported", n=2)])
    assert result is not None, "LaTeX in judge notes must not fail the event"
    assert result.rubric_mean == 0.8
    assert "$\\mu_B$" in result.notes
    print("PASS  test_latex_escapes_in_judge_reply")


def test_judge_failure_returns_none():
    attempts = int(_judge_cfg().get("retries", 2)) + 1
    result = _score_chunk(["not json {{{"] * attempts)
    assert result is None, "judge failure must yield None, not raise or score 0"

    evaluator = ReportEvaluator(llm=_ScriptedLLM(["garbage"] * attempts))
    macro = asyncio.run(evaluator.score_report(task="T", outline=[], report="body"))
    assert macro is None
    print("PASS  test_judge_failure_returns_none")


def test_macro_composition_and_prompt():
    reply = json.dumps({
        "global_reasoning": "solid report",
        "subject_coverage": 5, "global_flow": 4, "structural_score": 4,
        "tone_consistency": 3, "redundancy_avoidance": 4,
    })
    llm = _ScriptedLLM([reply])
    evaluator = ReportEvaluator(llm=llm)
    macro = asyncio.run(evaluator.score_report(
        task="Nuclear equation of state",
        outline=["Introduction", "Dense matter"],
        report="## Introduction\n\nBody.",
    ))
    assert isinstance(macro, MacroScore)
    assert abs(macro.score - 20 / 25) < 1e-9
    seen = "\n".join(str(m.get("content", m)) for m in llm.calls[0]["messages"])
    assert "Nuclear equation of state" in seen and "Dense matter" in seen
    assert macro.notes == "solid report"
    print("PASS  test_macro_composition_and_prompt")


def _run_all():
    passed = failed = 0
    cases = [
        ("test_grounding_verdict_ordering", test_grounding_verdict_ordering),
        ("test_hallucination_flag_halves", test_hallucination_flag_halves),
        ("test_no_sources_single_call", test_no_sources_single_call),
        ("test_no_claims_no_penalty", test_no_claims_no_penalty),
        ("test_empty_claims_reask_on_markers", test_empty_claims_reask_on_markers),
        ("test_empty_claims_reask_still_empty", test_empty_claims_reask_still_empty),
        ("test_parallel_array_claims_normalized",
         test_parallel_array_claims_normalized),
        ("test_trailing_verdict_strings_normalized",
         test_trailing_verdict_strings_normalized),
        ("test_fabricated_evidence_downgraded",
         test_fabricated_evidence_downgraded),
        ("test_unusable_claims_reask_on_markers",
         test_unusable_claims_reask_on_markers),
        ("test_empty_claims_reask_failure_keeps_empty",
         test_empty_claims_reask_failure_keeps_empty),
        ("test_latex_escapes_in_judge_reply", test_latex_escapes_in_judge_reply),
        ("test_judge_failure_returns_none", test_judge_failure_returns_none),
        ("test_macro_composition_and_prompt", test_macro_composition_and_prompt),
    ]
    for name, fn in cases:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())

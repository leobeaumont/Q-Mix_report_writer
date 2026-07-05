"""
Offline unit tests for qmix_report_writer/evaluation (training_eval plan 2.7).

Covers the semantic contracts the acceptance suite deliberately leaves to
this file (its stubs are schema-fillers and cannot steer verdicts):

  * grounding ordering: same audit, claims all-supported > all-unsupported >
    all-contradicted (contradiction hurts more than absence — TD3);
  * hallucination flag halves the rubric mean;
  * no sources -> single call, no grounding modulation;
  * no factual claims -> no grounding penalty;
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


def _claims_reply(verdict, n=4):
    return json.dumps({
        "claims": [{"claim": f"claim {i}", "verdict": verdict} for i in range(n)],
    })


def _score_chunk(responses, sources=SOURCES):
    evaluator = ReportEvaluator(llm=_ScriptedLLM(responses))
    return asyncio.run(evaluator.score_chunk(
        chunk="The mass is 5 GeV.", sources=sources,
        task="Particle masses", context="progress",
    ))


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
    result = _score_chunk([_audit_reply(), json.dumps({"claims": []})])
    assert result.n_claims == 0 and result.grounding_ratio is None
    assert abs(result.score - result.rubric_mean) < 1e-9, \
        "a chunk with no factual claims must not be grounding-penalized"
    print("PASS  test_no_claims_no_penalty")


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

"""
~1-minute judge smoke test — grounded chunk audits and one terminal macro call
through the live evaluator (native /api/chat structured outputs). Run before
committing hours to a benchmark or training run.

Includes the plan-2.9 GROUNDING PROBES (TD3 must actually fire):
  (a) chunk + MATCHING source      -> expects >= 1 supported claim;
  (b) chunk + CONTRADICTING source -> expects >= 1 contradicted claim and a
      visible score drop vs (a).
Exit code is nonzero when transport fails OR grounding stays inert.

    .venv\\Scripts\\python.exe experiments\\judge_smoke.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmix_report_writer.evaluation import ReportEvaluator
from qmix_report_writer.evaluation.judges import _judge_cfg

CHUNK = (
    "# Gravity\n\n"
    "Objects near the Earth's surface accelerate downward at approximately "
    "$9.81\\,\\mathrm{m/s^2}$. This acceleration is independent of the "
    "object's mass, as demonstrated by drop experiments in vacuum chambers."
)
MATCHING_SOURCES = [{
    "source": "physics_notes.pdf",
    "page": 1,
    "content": "Near the Earth's surface, free-fall acceleration g is about "
               "9.81 m/s^2 and does not depend on the falling object's mass.",
}]
CONTRADICTING_SOURCES = [{
    "source": "physics_notes.pdf",
    "page": 1,
    "content": "Near the Earth's surface, free-fall acceleration g is about "
               "3.51 m/s^2 and depends strongly on the falling object's mass.",
}]
TASK = "Basic physics of gravitation"


def _fmt_chunk(label, cs):
    return (f"{label}: score {cs.score:.4f}  (rubric {cs.rubric_mean:.2f}, "
            f"grounding {cs.grounding_ratio}, "
            f"claims {cs.n_supported}/{cs.n_unsupported}/{cs.n_contradicted} s/u/c, "
            f"halluc={cs.hallucination_flag})")


def main():
    evaluator = ReportEvaluator()
    cfg = _judge_cfg()
    print(f"judge model : {getattr(evaluator.llm, 'model_name', type(evaluator.llm).__name__)}")
    print(f"options     : temperature={cfg.get('temperature', 0.0)} "
          f"max_tokens={cfg.get('max_tokens', 3072)} "
          f"num_ctx={cfg.get('num_ctx', 32768)} retries={cfg.get('retries', 2)}")

    start = time.time()
    try:
        supported = asyncio.run(evaluator.score_chunk(
            chunk=CHUNK, sources=MATCHING_SOURCES, task=TASK,
            context="One section on gravity.",
        ))
        contradicted = asyncio.run(evaluator.score_chunk(
            chunk=CHUNK, sources=CONTRADICTING_SOURCES, task=TASK,
            context="One section on gravity.",
        ))
        macro = asyncio.run(evaluator.score_report(
            task=TASK, outline=["Gravity"], report=CHUNK,
        ))
    except Exception as exc:
        print(f"\nTRANSPORT FAILURE ({type(exc).__name__}): {exc}")
        print("Could not reach the judge endpoint — is Ollama running?")
        sys.exit(1)
    elapsed = time.time() - start

    if supported is None or contradicted is None or macro is None:
        print(f"\nFAILED after retries (matching={'ok' if supported else 'FAIL'}, "
              f"contradicting={'ok' if contradicted else 'FAIL'}, "
              f"macro={'ok' if macro else 'FAIL'})")
        print("The judge replies are unusable — do NOT launch a long run.")
        sys.exit(1)

    print()
    print(_fmt_chunk("matching src  ", supported))
    print(_fmt_chunk("contradict src", contradicted))
    print(f"macro score   : {macro.score:.4f}  (coverage {macro.subject_coverage}, "
          f"flow {macro.global_flow}, structure {macro.structural_score})")
    print(f"audit notes   : {supported.notes[:160]}")
    print(f"elapsed       : {elapsed:.1f}s for ~5 judge calls")

    # Grounding probes (plan 2.9): inert grounding means TD3 is dead in
    # training — treat it as a failed smoke, not a soft warning.
    problems = []
    if supported.n_supported < 1:
        problems.append(f"probe (a): expected >=1 supported claim with a "
                        f"matching source, got {supported.n_claims} claims "
                        f"({supported.n_supported} supported)")
    if contradicted.n_contradicted < 1:
        problems.append(f"probe (b): expected >=1 contradicted claim with a "
                        f"contradicting source, got {contradicted.n_claims} "
                        f"claims ({contradicted.n_contradicted} contradicted)")
    if contradicted.score >= supported.score:
        problems.append(f"probe (b): contradicted-source score "
                        f"{contradicted.score:.4f} did not drop below the "
                        f"matching-source score {supported.score:.4f}")

    if problems:
        print("\nGROUNDING INERT:")
        for p in problems:
            print(f"  - {p}")
        print("Claim extraction is not engaging (plan 2.9) — fix before training.")
        sys.exit(1)

    print("\nOK — judge transport healthy, grounding engaged "
          f"(drop {supported.score - contradicted.score:+.4f} on contradiction).")


if __name__ == "__main__":
    main()

"""
30-second judge smoke test — one grounded chunk audit (+ claim check) and one
terminal macro call through the live evaluator (native /api/chat structured
outputs). Run before committing hours to a benchmark or training run.

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
SOURCES = [{
    "source": "physics_notes.pdf",
    "page": 1,
    "content": "Near the Earth's surface, free-fall acceleration g is about "
               "9.81 m/s^2 and does not depend on the falling object's mass.",
}]
TASK = "Basic physics of gravitation"


def main():
    evaluator = ReportEvaluator()
    cfg = _judge_cfg()
    print(f"judge model : {getattr(evaluator.llm, 'model_name', type(evaluator.llm).__name__)}")
    print(f"options     : temperature={cfg.get('temperature', 0.0)} "
          f"max_tokens={cfg.get('max_tokens', 3072)} "
          f"num_ctx={cfg.get('num_ctx', 32768)} retries={cfg.get('retries', 2)}")

    start = time.time()
    try:
        chunk = asyncio.run(evaluator.score_chunk(
            chunk=CHUNK, sources=SOURCES, task=TASK, context="One section on gravity.",
        ))
        macro = asyncio.run(evaluator.score_report(
            task=TASK, outline=["Gravity"], report=CHUNK,
        ))
    except Exception as exc:
        print(f"\nTRANSPORT FAILURE ({type(exc).__name__}): {exc}")
        print("Could not reach the judge endpoint — is Ollama running?")
        sys.exit(1)
    elapsed = time.time() - start

    if chunk is None or macro is None:
        print(f"\nFAILED after retries (chunk={'ok' if chunk else 'FAIL'}, "
              f"macro={'ok' if macro else 'FAIL'})")
        print("The judge replies are unusable — do NOT launch a long run.")
        sys.exit(1)

    print(f"\nchunk score   : {chunk.score:.4f}  (rubric {chunk.rubric_mean:.2f}, "
          f"grounding {chunk.grounding_ratio}, "
          f"claims {chunk.n_supported}/{chunk.n_unsupported}/{chunk.n_contradicted} s/u/c, "
          f"halluc={chunk.hallucination_flag})")
    print(f"macro score   : {macro.score:.4f}  (coverage {macro.subject_coverage}, "
          f"flow {macro.global_flow}, structure {macro.structural_score})")
    print(f"audit notes   : {chunk.notes[:160]}")
    print(f"elapsed       : {elapsed:.1f}s for 3 judge calls")
    print("\nOK — judge transport is healthy.")


if __name__ == "__main__":
    main()

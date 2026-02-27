#!/bin/bash
# HMMT Feb 2025 — MathArena/hmmt_feb_2025
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}

echo "[QMIX] HMMT Feb 2025 (MathArena/hmmt_feb_2025)"
python3 -m experiments.run_qmix_eval \
    --dataset hmmt_2025 --split train --llm_name "$LLM" --num_rounds 3 \
    --model_path checkpoints/qmix_math.pt \
    --mode qmix --output_path result/hmmt2025_qmix.json

#!/bin/bash
# AIME 2025 — MathArena/aime_2025
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}

echo "[QMIX] AIME 2025 (MathArena/aime_2025)"
python3 -m experiments.run_qmix_eval \
    --dataset aime_2025 --split train --llm_name "$LLM" --num_rounds 3 \
    --model_path checkpoints/qmix_math.pt \
    --mode qmix --output_path result/aime2025_qmix.json

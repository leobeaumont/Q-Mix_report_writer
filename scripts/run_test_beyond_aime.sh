#!/bin/bash
# Beyond-AIME — ByteDance-Seed/BeyondAIME
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}

echo "[QMIX] Beyond-AIME (ByteDance-Seed/BeyondAIME)"
python3 -m experiments.run_qmix_eval \
    --dataset beyond_aime --split test --llm_name "$LLM" --num_rounds 3 \
    --model_path checkpoints/qmix_math.pt \
    --mode qmix --output_path result/beyond_aime_qmix.json

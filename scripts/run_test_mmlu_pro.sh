#!/bin/bash
# MMLU-Pro — TIGER-Lab/MMLU-Pro (validation split)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}
LIMIT=${DATA_LIMIT:-""}

ARGS="--dataset mmlu_pro --split validation --llm_name $LLM --num_rounds 2"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

echo "[QMIX] MMLU-Pro (validation split)"
python3 -m experiments.run_qmix_eval $ARGS \
    --model_path checkpoints/qmix_agentic.pt \
    --mode qmix --output_path result/mmlu_pro_qmix.json

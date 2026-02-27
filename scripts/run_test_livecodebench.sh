#!/bin/bash
# LiveCodeBench — livecodebench/code_generation (test)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}
LIMIT=${DATA_LIMIT:-""}

ARGS="--dataset livecodebench --llm_name $LLM --num_rounds 2"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

echo "[QMIX] LiveCodeBench (livecodebench/code_generation)"
python3 -m experiments.run_qmix_eval $ARGS \
    --model_path checkpoints/qmix_coding.pt \
    --mode qmix --output_path result/livecodebench_qmix.json

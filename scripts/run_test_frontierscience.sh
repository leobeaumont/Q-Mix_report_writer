#!/bin/bash
# Frontier Science — openai/frontierscience (test)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}
LIMIT=${DATA_LIMIT:-""}

ARGS="--dataset frontierscience --split test --llm_name $LLM --num_rounds 2"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

echo "[QMIX] Frontier Science (openai/frontierscience)"
python3 -m experiments.run_qmix_eval $ARGS \
    --model_path checkpoints/qmix_agentic.pt \
    --mode qmix --output_path result/frontierscience_qmix.json

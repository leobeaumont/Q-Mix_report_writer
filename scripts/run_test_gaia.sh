#!/bin/bash
# GAIA — gaia-benchmark/GAIA (all validation)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"
LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}
LIMIT=${DATA_LIMIT:-""}

ARGS="--dataset gaia --split validation --llm_name $LLM --num_rounds 2"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

echo "[QMIX] GAIA (gaia-benchmark/GAIA, all)"
python3 -m experiments.run_qmix_eval $ARGS \
    --model_path checkpoints/qmix_agentic.pt \
    --mode qmix --output_path result/gaia_qmix.json

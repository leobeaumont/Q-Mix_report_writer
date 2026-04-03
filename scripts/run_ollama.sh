#!/bin/bash
# ============================================================
#  Run pipeline with a local Ollama model (fully private)
#
#  Usage:
#    bash scripts/run_ollama.sh
#    LLM_MODEL=mistral bash scripts/run_ollama.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

LLM=${LLM_MODEL:-"tinyllama"}
EPISODES=${TRAIN_EPISODES:-50}
DEVICE=${DEVICE:-"cpu"}
LIMIT=${DATA_LIMIT:-""}
RESULT_DIR="result_ollama"
CKPT_DIR="checkpoints_ollama"

mkdir -p "$RESULT_DIR" "$CKPT_DIR"

echo "========================================="
echo "  QMIX: FULL PIPELINE (Ollama - local)"
echo "  Model     : $LLM"
echo "  Results   : $RESULT_DIR"
echo "  Checkpoint: $CKPT_DIR/qmix_unified.pt"
echo "========================================="

# ── PHASE 1: TRAINING ───────────────────────────────────────
echo ""
echo ">>> PHASE 1: UNIFIED TRAINING"
echo ""

python -m experiments.run_qmix_train \
    --dataset "livecodebench_testgen,mmlu_pro,aime_2024" \
    --split test \
    --llm_name "$LLM" \
    --num_episodes "$EPISODES" \
    --num_rounds 2 \
    --batch_size 8 \
    --data_limit 15 \
    --token_penalty 0.075 \
    --accuracy_weight 1.50 \
    --device "$DEVICE" \
    --save_path "$CKPT_DIR/qmix_unified.pt" \
    --log_interval 10 \
    || echo "  [WARN] Training exited with non-zero code"

#if [ ! -f "$CKPT_DIR/qmix_unified.pt" ]; then
#    echo "  [ERROR] Checkpoint not found: $CKPT_DIR/qmix_unified.pt"
#    exit 1
#fi

# ── PHASE 2: TESTING ────────────────────────────────────────
echo ""
echo ">>> PHASE 2: TESTING"
echo ""

CKPT="$CKPT_DIR/qmix_unified.pt"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

echo "[1/3] HumanEval..."
python -m experiments.run_qmix_eval \
    --dataset humaneval --llm_name "$LLM" --num_rounds 2 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/humaneval_qmix.json" \
    $LIMIT_ARG

echo "[2/3] MMLU-Pro..."
python -m experiments.run_qmix_eval \
    --dataset mmlu_pro --split validation --llm_name "$LLM" --num_rounds 2 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/mmlu_pro_qmix.json" \
    $LIMIT_ARG

echo "[3/3] AIME 2025..."
python -m experiments.run_qmix_eval \
    --dataset aime_2025 --split train --llm_name "$LLM" --num_rounds 3 \
    --model_path "$CKPT" \
    --mode qmix --output_path "$RESULT_DIR/aime2025_qmix.json"

echo ""
echo "========================================="
echo "  COMPLETE — results in $RESULT_DIR"
echo "========================================="
python summarize_results.py "$RESULT_DIR"
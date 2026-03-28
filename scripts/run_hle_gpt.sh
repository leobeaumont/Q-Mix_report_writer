#!/bin/bash
# ============================================================
#  Humanity's Last Exam (MCQ only, first 250) — GPT-OSS:120B
#  Uses the unified QMIX checkpoint from run_all_gpt.sh
#
#  Usage:
#    bash scripts/run_hle_gpt.sh
# ============================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null && conda activate base 2>/dev/null || true

export BASE_URL="${BASE_URL:-https://api.groq.com/openai/v1}"
export API_KEY="${API_KEY:?Set API_KEY in .env or environment}"
export LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"

LLM="$LLM_MODEL"
CKPT="checkpoints/qmix_unified.pt"
RESULT_DIR="result_hle_gpt"

mkdir -p "$RESULT_DIR"

echo "========================================="
echo "  HLE Benchmark (MCQ, 250) — QMIX"
echo "  Model     : $LLM"
echo "  Checkpoint: $CKPT"
echo "  Results   : $RESULT_DIR"
echo "========================================="

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] Checkpoint not found: $CKPT"
    echo "Run scripts/run_all_gpt.sh first to train."
    exit 1
fi

echo ""
echo "[QMIX] HLE (cais/hle, MCQ only, first 250)..."
python3 -m experiments.run_qmix_eval \
    --dataset hle \
    --split test \
    --llm_name "$LLM" \
    --num_rounds 4 \
    --limit 250 \
    --model_path "$CKPT" \
    --mode qmix \
    --output_path "$RESULT_DIR/hle_qmix.json"

echo ""
echo "========================================="
echo "  HLE COMPLETE (GPT-OSS:120B)"
echo "========================================="

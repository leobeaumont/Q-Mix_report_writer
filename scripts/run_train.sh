#!/bin/bash
# ============================================================
#  QMIX TRAINING — Trains on 3 datasets (15 examples each)
#    1. LiveCodeBench test_generation (15 examples) — Coding
#    2. MMLU-Pro (validation split, 15 examples) — Agentic
#    3. AIME 2024 (first 15 examples) — Math
# ============================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

LLM=${LLM_MODEL:-"openai/gpt-oss-120b"}
EPISODES=${TRAIN_EPISODES:-50}
DEVICE=${DEVICE:-"cpu"}

echo ""
echo "============================================================"
echo "  QMIX MULTI-DATASET TRAINING"
echo "  Training on: LCB-TestGen(15) + MMLU-Pro(15) + AIME2024(15)"
echo "  LLM: $LLM | Episodes per dataset: $EPISODES"
echo "============================================================"
echo ""

# --- Train on LiveCodeBench test_generation (15 examples) ---
echo "[1/3] Training on LiveCodeBench test_generation (15 samples)..."
python3 -m experiments.run_qmix_train \
    --dataset livecodebench_testgen \
    --split test \
    --llm_name "$LLM" \
    --num_episodes "$EPISODES" \
    --num_rounds 2 \
    --batch_size 8 \
    --data_limit 15 \
    --token_penalty 0.1 \
    --accuracy_weight 1.0 \
    --device "$DEVICE" \
    --save_path checkpoints/qmix_coding.pt \
    --log_interval 10

# --- Train on MMLU-Pro (validation split, 15 examples) ---
echo "[2/3] Training on MMLU-Pro (validation, 15 samples)..."
python3 -m experiments.run_qmix_train \
    --dataset mmlu_pro \
    --split validation \
    --llm_name "$LLM" \
    --num_episodes "$EPISODES" \
    --num_rounds 2 \
    --batch_size 8 \
    --data_limit 15 \
    --token_penalty 0.1 \
    --accuracy_weight 1.0 \
    --device "$DEVICE" \
    --save_path checkpoints/qmix_agentic.pt \
    --log_interval 10

# --- Train on AIME 2024 (first 15 examples) ---
echo "[3/3] Training on AIME 2024 (15 samples)..."
python3 -m experiments.run_qmix_train \
    --dataset aime_2024 \
    --split train \
    --llm_name "$LLM" \
    --num_episodes "$EPISODES" \
    --num_rounds 2 \
    --batch_size 8 \
    --data_limit 15 \
    --token_penalty 0.05 \
    --accuracy_weight 1.5 \
    --device "$DEVICE" \
    --save_path checkpoints/qmix_math.pt \
    --log_interval 10

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE"
echo "  Models: qmix_coding.pt, qmix_agentic.pt, qmix_math.pt"
echo "============================================================"

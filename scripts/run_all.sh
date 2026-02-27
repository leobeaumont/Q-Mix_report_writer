#!/bin/bash
# ============================================================
#  Run COMPLETE pipeline: Train (3 datasets) -> Test (9 datasets)
# ============================================================
set -e
cd "$(dirname "$0")"

echo "========================================="
echo "  QMIX: FULL PIPELINE"
echo "========================================="

echo ""
echo ">>> PHASE 1: TRAINING"
echo ""
bash run_train.sh

echo ""
echo ">>> PHASE 2: TESTING (9 benchmarks)"
echo ""

# Coding (2)
bash run_test_livecodebench.sh
bash run_test_humaneval.sh

# Agentic (3)
bash run_test_mmlu_pro.sh
bash run_test_gaia.sh
bash run_test_frontierscience.sh

# Math (4)
bash run_test_aime2025.sh
bash run_test_aime2026.sh
bash run_test_beyond_aime.sh
bash run_test_hmmt2025.sh

echo ""
echo "========================================="
echo "  ALL 9 BENCHMARKS COMPLETE"
echo "  Results in: result/"
echo "========================================="

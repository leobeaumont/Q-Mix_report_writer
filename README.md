# Agent Q-Mix: QMIX for Multi-Agent Communication Topology Optimization

QMIX-based multi-agent reinforcement learning that learns optimal communication topologies for LLM agent collaboration. Maximizes task accuracy while minimizing token usage.

## Architecture

**Networked MMDP** with centralized training, decentralized execution:

1. **GNN Message Passing** — Agents communicate through a learned graph topology
2. **Per-Agent Q-Network** — GNN -> GRU (temporal) -> MLP (Q-values)
3. **QMIX Mixing Network** — Monotonic: dQ_tot/dQ_i >= 0
4. **Reward** = accuracy x w_acc - token_ratio x w_token


## QMIX Actions

| Action | Name | Communication Pattern |
|--------|------|----------------------|
| 1 | `solo_process` | No communication |
| 2 | `broadcast_all` | Send to all neighbors |
| 3 | `selective_query` | Query one neighbor |
| 4 | `aggregate_refine` | Receive from all, refine |
| 5 | `execute_verify` | Tool use, minimal comm |
| 6 | `debate_check` | Adversarial debate pair |


## Quick Start

On Gemini 3.1 Flash Lite Model:

```bash
cp .env.example .env       # Add your API key
pip install -r requirements.txt
# To Train and Test (LCB, HE, MMLU, AIME, B-AIME, HMMT)
bash scripts/run_all_gemini.sh
# To Test For HLE
bash scripts/run_hle_gemini.sh
```


On GPT-oss 120B Model:

```bash
cp .env.example .env       # Add your API key
pip install -r requirements.txt
# To Train and Test (LCB, HE, MMLU, AIME, B-AIME, HMMT)
bash scripts/run_all_gpt.sh
# To Test For HLE
bash scripts/run_hle_gpt.sh
```

## Datasets

### Training (15 examples each)

| Dataset | HuggingFace Path | Split | Total | Used | Domain |
|---------|-----------------|-------|-------|------|--------|
| LiveCodeBench TestGen | `livecodebench/test_generation` | test | 442 | 15 | Coding |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test | 70 | 15 | Agentic |
| AIME 2024 | `Maxwell-Jia/AIME_2024` | train | 30 | 15 | Math |

### Testing (9 benchmarks)

| # | Dataset | HuggingFace Path | Split | Samples | Category | Script |
|---|---------|-----------------|-------|---------|----------|--------|
| 1 | LiveCodeBench | `livecodebench/code_generation` | test | 400 | Coding | `run_test_livecodebench.sh` |
| 2 | HumanEval | `openai_humaneval` | test | 164 | Coding | `run_test_humaneval.sh` |
| 3 | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | validation | 70 | Agentic | `run_test_mmlu_pro.sh` |
| 4 | AIME 2025 | `MathArena/aime_2025` | train | 30 | Math | `run_test_aime2025.sh` |
| 5 | AIME 2026 | `MathArena/aime_2026` | train | 30 | Math | `run_test_aime2026.sh` |
| 6 | Beyond-AIME | `ByteDance-Seed/BeyondAIME` | test | 100 | Math | `run_test_beyond_aime.sh` |
| 7 | HMMT Feb 2025 | `MathArena/hmmt_feb_2025` | train | 30 | Math | `run_test_hmmt2025.sh` |

## Project Structure

```
agent_q_mix/
├── qmix/                   # QMIX core (GNN, Q-networks, mixing, replay, trainer)
├── graph/                   # Multi-agent graph execution engine
├── agents/                  # Agent types (MathSolver, CodeWriter, etc.)
├── llm/                     # LLM API layer (GPT, DeepSeek, Qwen)
├── prompt/                  # Domain-specific prompt sets
├── datasets/                # Benchmark dataset loaders
├── experiments/             # Training and evaluation scripts
├── scripts/                 # Shell scripts (train + 10 test benchmarks)
├── checkpoints/             # Saved QMIX models
└── result/                  # Evaluation results (JSON + JSONL)
```

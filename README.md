# Agent Q-Mix: QMIX for Multi-Agent Communication Topology Optimization

QMIX-based multi-agent reinforcement learning that learns optimal communication topologies for LLM agent collaboration. Maximizes task accuracy while minimizing token usage.

## Architecture

**Networked MMDP** with centralized training, decentralized execution:

1. **GNN Message Passing** — Agents communicate through a learned graph topology
2. **Per-Agent Q-Network** — GNN -> GRU (temporal) -> MLP (Q-values)
3. **QMIX Mixing Network** — Monotonic: dQ_tot/dQ_i >= 0
4. **Reward** = accuracy x w_acc - token_ratio x w_token

## Quick Start

```bash
cp .env.example .env       # Add your API key
pip install -r requirements.txt

# Train on 3 datasets (15 examples each)
bash scripts/run_train.sh

# Test on all 10 benchmarks
bash scripts/run_all.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASE_URL` | LLM API base URL | `https://api.openai.com/v1` |
| `API_KEY` | API key | — |
| `LLM_MODEL` | Override default model | `gpt-4o-mini` |
| `DATA_LIMIT` | Limit eval samples | (full) |
| `TRAIN_EPISODES` | Episodes per training dataset | `100` |

## Datasets

### Training (15 examples each)

| Dataset | HuggingFace Path | Split | Total | Used | Domain |
|---------|-----------------|-------|-------|------|--------|
| LiveCodeBench TestGen | `livecodebench/test_generation` | test | 442 | 15 | Coding |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | validation | 70 | 15 | Agentic |
| AIME 2024 | `Maxwell-Jia/AIME_2024` | train | 30 | 15 | Math |

### Testing (9 benchmarks)

| # | Dataset | HuggingFace Path | Split | Samples | Category | Script |
|---|---------|-----------------|-------|---------|----------|--------|
| 1 | LiveCodeBench | `livecodebench/code_generation` | test | 400 | Coding | `run_test_livecodebench.sh` |
| 2 | HumanEval | `openai_humaneval` | test | 164 | Coding | `run_test_humaneval.sh` |
| 3 | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | validation | 70 | Agentic | `run_test_mmlu_pro.sh` |
| 4 | GAIA | `gaia-benchmark/GAIA` (2023_all) | validation | 165 | Agentic | `run_test_gaia.sh` |
| 5 | Frontier Science | `openai/frontierscience` | test | 160 | Agentic | `run_test_frontierscience.sh` |
| 6 | AIME 2025 | `MathArena/aime_2025` | train | 30 | Math | `run_test_aime2025.sh` |
| 7 | AIME 2026 | `MathArena/aime_2026` | train | 30 | Math | `run_test_aime2026.sh` |
| 8 | Beyond-AIME | `ByteDance-Seed/BeyondAIME` | test | 100 | Math | `run_test_beyond_aime.sh` |
| 9 | HMMT Feb 2025 | `MathArena/hmmt_feb_2025` | train | 30 | Math | `run_test_hmmt2025.sh` |


## Commands

### Full Pipeline

```bash
bash scripts/run_all.sh          # Train + Test everything
bash scripts/run_train.sh        # Train only
bash scripts/run_test_humaneval.sh   # Single benchmark
```


## QMIX Action Space

| Action | Name | Communication Pattern |
|--------|------|----------------------|
| 0 | `solo_process` | No communication |
| 1 | `broadcast_all` | Send to all neighbors |
| 2 | `selective_query` | Query one neighbor |
| 3 | `aggregate_refine` | Receive from all, refine |
| 4 | `execute_verify` | Tool use, minimal comm |
| 5 | `debate_check` | Adversarial debate pair |

## Output

- **JSON** (`result/<dataset>_qmix.json`) — metrics + full results
- **JSONL** (`result/<dataset>_qmix.jsonl`) — one result per line

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

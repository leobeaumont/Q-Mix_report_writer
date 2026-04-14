# Agent Q-Mix: QMIX for Multi-Agent Communication Topology Optimization

QMIX-based multi-agent reinforcement learning that learns optimal communication topologies for LLM agent collaboration. Maximizes task accuracy while minimizing token usage.

## Architecture

**Networked MMDP** with centralized training, decentralized execution:

1. **GNN Message Passing** — Agents communicate through a learned graph topology
2. **Per-Agent Q-Network** — GNN -> GRU (temporal) -> MLP (Q-values)
3. **QMIX Mixing Network** — Monotonic: dQ_tot/dQ_i >= 0
4. **Reward** = $\Delta_{report \, score} \times w_{report \, score} + \Delta_{token \, goal} \times w_{token \, goal}$, where $\Delta_{report \, score}$ is the variation of the report score compared to its last state and $\Delta_{token \, goal}$ is the variation of the token goal score compared to its last state. The token goal score is calculated using a Gaussian curve centered around the token goal.


## QMIX Actions

| Action | Name | Communication Pattern |
|--------|------|----------------------|
| 1 | `solo_process` | No communication |
| 2 | `broadcast_all` | Send to all neighbors |
| 3 | `selective_query` | Query one neighbor |
| 4 | `aggregate_refine` | Receive from all, refine |
| 5 | `execute_verify` | Tool use, minimal comm |
| 6 | `debate_check` | Adversarial debate pair |
| 7 | `append` | Send to a collector node |
| 8 | `terminate` | Output the content of collector node |


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

## Reproducibility

Pre-trained QMIX checkpoints and evaluation results from our runs are included in the repository.

### Checkpoints

| Model | Path |
|---|---|
| GPT-OSS:120B | `checkpoints/qmix_unified.pt` |
| Gemini Flash Lite | `checkpoints_gemini/qmix_unified.pt` |

## Project Structure

```
agent_q_mix/
├── qmix/                    # QMIX core (GNN, Q-networks, mixing, replay, trainer)
├── graph/                   # Multi-agent graph execution engine
├── agents/                  # Agent types (MathSolver, CodeWriter, etc.)
├── llm/                     # LLM API layer (GPT, DeepSeek, Qwen)
├── prompt/                  # Domain-specific prompt sets
├── datasets/                # Benchmark dataset loaders
├── experiments/             # Training and evaluation scripts
├── scripts/                 # Shell scripts for training + evaluation
└── checkpoints/             # Saved QMIX models
```

## Adaptation for report writing

This repository is a specification of [ericjiang18/Agent-Q-Mix](https://github.com/ericjiang18/Agent-Q-Mix) for scientific report writing. This part describes the modifications applied to the code to solve the report writing task.

### New team of agents

The team of agents used for report writing is the following:
1) Lead architect: responsible for the outline, highest messaging and appending priority.
2) Researcher: can use RAG tools to look for information in the database.
3) Data analyst: turns raw data into ideas.
4) Technical writer: responsible for turning ideas into developped text.
5) Reviewer: review and grades writings, used to calculate the reward function, lowest messaging and appending priority.

### From `final answer` to `append loop` system

Agent-Q-Mix uses a `final answer` system, where all the agents interact during their communication rounds. After a fixed set of rounds, all of their output is passed to a `decision` agent to generate the final answer given to the user. This approach is good for problem solving and especially token efficiency, but in our case generating a full report in one pass of the `decision` agent would create poor results.

To solve this problem, an `append loop` approach is used without breaking the structure of the original code. To do that, 2 new actions have been added: `append` and `terminate`. When the `append` action is used by an agent, its output is sent to a `collector` agent. The `collector` is only used to store the current state of the report as it is written and does nothing else. When the `terminate` action is selected by the majority of the agents, the round loop ends and the report stored inside of the `collector` agent is returned. This approach keeps the graph design of the original code. This means that the information flows from the user query to the final report with no interruption. 

Technical changes:
- Creation of `append` and `terminate` actions
- Created a copy of `selective_query` and `debate_check` action for each agent. This way Q-Mix can choose the interlocutor by using the corresponding version of the action. This means Q-Mix has a pool of $6 + 2 N$ actions where $N$ is the number of agents.
- When multiple agents try using the `append` action on the same turn, only the highest ID agent can append (to give priority to `Reviewer` and `Technical Writer` agents).
- Creation of the `collector` node that receives the output of the agents choosing the `append` action.
- The `collector` is a normal node like other agents, but it can't use any action.
- The `collector` node makes sure the text is formatted correctly and no LLM remnants are passed into the report.
- When an agent choose the `append` action, its prompt informs it that its output will be added to the report.
- The verification to avoid graph cycles is still present and gives messaging priority to lower indices agents.
- When an agent action is denied because of an append lock or a cycle, its action is defaulted to `solo_process`.
- Changed Kahn's algorithm in `graph/graph.py: _execute_round` method to accept self edge (for `execute_verify` action) and mutual edge (for `debate` action).
- Modified Q-Mix to only train and select actions for the 5 acting agents, while ignoring the `collector` node.
- Added a dynamic `ReportState` singleton to track the report text, sources and a progress summary. The report state is updated by the `Collector` agents and the summary of progress is given as context to other agents.
- The report state summary is embedded inside of the observation features as context for the Q-Mixer.

### From a `post-process` to `in real time` reward system

Agent-Q-Mix is trained on a set of problem solving database. To train the `Q-Mixer` they use the score based on the answers given by the model to each problem of the datasets. And a penalty is applied for token usage. This forces the model to be precise with few tokens.  

In the `report writer` implementation, since the `Q-Mixer` is not used for problem solving, it can't be scored using answer `accuracy` anymore. On top of that, `report generation` is a longer process than `problem solving`. It means that it is possible and preferable to score the model while it is constructing the report, to help the model understand the quality of each of its decisions. This change is highly compatible with the new `append loop` architecture: every time content is added to the report with the `append` action, it is possible to score the quality of the addition and the token usage. The Q-Mixer reward can be computed mid-process using these scores. More precisely, the `variation of the scores` between the previous and the current state of the report gives a very good insight on how good the last addition was to the overall report. And the better the addition is, the better the reward will be.

Technical changes:
- Changed reward to $\Delta_{report \, score} \times w_{report \, score} + \Delta_{token \, goal} \times w_{token \, goal}$.
- Added a toggleable `JSON` output format to the LLM handlers.
- Added a `Micro Scoring` and `Macro Scoring` LLM judges used during training to help improve the quality of the production. The `Micro Scoring` judge scores each chunk of the document. The `Macro Scoring` judge scores the whole document. 
- `Micro Scoring` judge is responsible for: logic, verifiability, technical precision, information density and hallucination flagging.
- `Macro Scoring` judge is responsible for: subject coverage, flow, structure, tone and avoiding repetition.
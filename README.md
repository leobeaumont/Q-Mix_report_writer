# Agent Q-Mix: QMIX for Multi-Agent Communication Topology Optimization

**REQUIRES: Python >= 3.10**

QMIX-based multi-agent reinforcement learning that learns optimal communication topologies for LLM agent collaboration. Maximizes task accuracy while minimizing token usage.

## Architecture

**Networked MMDP** with centralized training, decentralized execution:

1. **GNN Message Passing** — Agents communicate through a learned graph topology
2. **Per-Agent Q-Network** — GNN -> GRU (temporal) -> MLP (Q-values)
3. **QMIX Mixing Network** — Monotonic: $\frac{\delta Q_{tot}}{\delta Q_i} >= 0$
4. **Reward** = $\Delta_{report \, score} \times w_{report \, score} + \Delta_{token \, goal} \times w_{token \, goal}$, where $\Delta_{report \, score}$ is the variation of the report score compared to its last state and $\Delta_{token \, goal}$ is the variation of the token goal score compared to its last state. The token goal score is calculated using a Gaussian curve centered around the token goal.


## QMIX Actions

| Action | Name | Communication Pattern |
|--------|------|----------------------|
| 0 | `solo_process` | No communication |
| 1 | `broadcast_all` | Send to all neighbors |
| 2 - 5 | `selective_query` | Query one neighbor (one for each agent) |
| 6 | `aggregate_refine` | Receive from all, refine |
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
4) Reviewer: review and grades writings, used to calculate the reward function, lowest messaging and appending priority.

### From `final answer` to `append loop` system

Agent-Q-Mix uses a `final answer` system, where all the agents interact during their communication rounds. After a fixed set of rounds, all of their output is passed to a `decision` agent to generate the final answer given to the user. This approach is good for problem solving and especially token efficiency, but in our case generating a full report in one pass of the `decision` agent would create poor results.

To solve this problem, an `append loop` approach is used without breaking the structure of the original code. To do that, 2 new actions have been added: `append` and `terminate`. When the `append` action is used by an agent, its output is sent to a `collector` agent. The `collector` is only used to store the current state of the report as it is written and does nothing else. When the `terminate` action is selected by the majority of the agents, the round loop ends and the report stored inside of the `collector` agent is returned. This approach keeps the graph design of the original code. This means that the information flows from the user query to the final report with no interruption. 

Technical changes:
- Creation of `append` and `terminate` actions.
- Removed `debate` and `execute_verify` actions.
- Created a copy of `selective_query` action for each agent. This way Q-Mix can choose the interlocutor by using the corresponding version of the action. This means Q-Mix has a pool of $5 + N$ actions where $N$ is the number of agents.
- Multiple agents may choose the `append` action in the same round; all of them get a spatial edge to the `Collector`, which processes every incoming message.
- Creation of the `collector` node that receives the output of the agents choosing the `append` action.
- The `collector` is a normal node like other agents, but it can't use any action.
- The `terminate` action ends the round loop only when a **majority** of the acting agents vote for it (threshold: `>= n_acting / 2` votes). A single agent voting `terminate` is not enough.
- The `Collector` makes **two sequential LLM calls** per append: the first generates the polished prose to add to the report, the second (using a dedicated "Summarizer" role prompt) rewrites the progress summary so all agents have an up-to-date picture of what has been written.
- The `Collector` silently skips execution when its `spatial_info` is empty (no agent sent it a message that round) or when the `DataAnalyst`'s entire message consists only of absence markers (`State Deficiency`, `absent`, `not found`, etc.). This prevents meta-commentary about missing evidence from ever reaching the report.
- When an agent choose the `append` action, its prompt informs it that its output will be added to the report.
- The verification to avoid graph cycles is still present and gives messaging priority to lower indices agents.
- When an agent action is denied because of a cycle, its action is defaulted to `solo_process`.
- Changed Kahn's algorithm in `graph/graph.py: _execute_round` method to accept self edge and mutual edge.
- Modified Q-Mix to only train and select actions for the 5 acting agents, while ignoring the `collector` node.
- Added a dynamic `ReportState` singleton to track the report text, sources and a progress summary. Internally, the report is stored as a structured list of sections, each with a unique `section_id`, a `title` (inferred from the first Markdown heading), and its `content`. This allows targeted in-place replacement via `replace_section(section_id, new_content)` and a formatted section index via `list_sections()`, both used during the review and revision flow. The report state is updated by the `Collector` agent and the summary of progress is given as context to other agents.
- The report state summary is embedded inside of the observation features as context for the Q-Mixer.
- `Lead Architect` now define a `Current Team Objective` to help the team separate the goal of the round from the global report subject. This aims to help the agents decompose the task into small trivial parts.
- Added a deterministic post-processing filter (`utils/report_filter.py`) applied to the final report before it is returned to the caller. The filter works at sentence level and removes any sentence containing pipeline-internal jargon (`State Deficiency`, `data atoms`, `evidence atoms`, `RAG Tool`, `RAG results`, internal variable names, etc.) as well as section-transition sentences that reference "the next/following/subsequent section". No LLM call is made; only unambiguous pipeline-internal patterns are removed so that legitimate scientific prose is left untouched.

### From a `post-process` to `in real time` reward system

Agent-Q-Mix is trained on a set of problem solving database. To train the `Q-Mixer` they use the score based on the answers given by the model to each problem of the datasets. And a penalty is applied for token usage. This forces the model to be precise with few tokens.  

In the `report writer` implementation, since the `Q-Mixer` is not used for problem solving, it can't be scored using answer `accuracy` anymore. On top of that, `report generation` is a longer process than `problem solving`. It means that it is possible and preferable to score the model while it is constructing the report, to help the model understand the quality of each of its decisions. This change is highly compatible with the new `append loop` architecture: every time content is added to the report with the `append` action, it is possible to score the quality of the addition and the token usage. The Q-Mixer reward can be computed mid-process using these scores. More precisely, the `variation of the scores` between the previous and the current state of the report gives a very good insight on how good the last addition was to the overall report. And the better the addition is, the better the reward will be.

Technical changes:
- Changed reward to $\Delta_{report \, score} \times w_{report \, score} + \Delta_{token \, goal} \times w_{token \, goal}$.
- Added an optional `JSON` schema format to the LLM handlers, that forces the LLM's output to a specific format.
- Added a `Micro Scoring` and `Macro Scoring` LLM judges used during training to help improve the quality of the production. The `Micro Scoring` judge scores each chunk of the document. The `Macro Scoring` judge scores the whole document. 
- `Micro Scoring` judge is responsible for: logic, verifiability, technical precision, information density and hallucination flagging.
- `Macro Scoring` judge is responsible for: subject coverage, flow, structure, tone and avoiding repetition.
- The reward is now computed after every `append` action, because the changes in the report cause changes in the report `score` and `token goal` completion.
- All rounds between the previous `append` round and the current `append` round receive an even fraction of the reward. This is done to reward all the rounds that lead to a good `append` to the the report and not only the round where the `append` action occurs. The reward is shared between rounds to motivate the model to be efficient (i.e.: get to a high quality `append` action in as few rounds as possible).
- Added a `Score` and `LengthGoal` singletons to track their current respective value, but also the previous one. This way a `get_delta()` method can be called on them to get the difference between their current and their last value.
- Changed the training pipeline to compute the reward after each `append` action.
- Added a `step buffer` to propagate the reward to all the rounds that lead to the `append` action. The reward is evenly spread between all actions to reward the whole process of creation. Any steps remaining in the buffer at the end of an episode (i.e. after the last `append`) are flushed to the episode with `team_reward = 0`.
- Added a `length scorer` based on a Gausian curve centered at $\mu = 25000$ and $\sigma = 8500$ with a peak of height 1. The goal of 25000 character correspond to ~4500 words which is the average length of scientific documents. The value of $\sigma$ is set to 8500 to start giving a good reward signal at around 5000 characters.
- Added a `report scorer` based on the 2 `LLM judges` described earlier.
    - The `Macro` judge is prompted to analyze the whole report with only a prompt and the report as context.
    - The `Micro` judge is prompted to analyze each chunk of the report individually. When the judge process a chunk he has access to: its prompt, global notes generated by the `Macro` judge, a summary of the report, notes from the `Micro` analysis of all previous chunks and the text of the current chunk.
    - Each judge produces scores for each categories they are asked evaluate, with scores ranging from 0 to 5.
    - The final score of the report is the average of all the `Macro` scores and all the `Micro` scores from each individual chunks.
    - The final score is normalized between 0 and 1. To ensure it has the same scale as the `length score`.
- To compute the `reward`, the evolution of the `length` and `report` scores is used. This ensure that an addition that increases significantly the global score is greatly rewarded. However is mediocre addition is not rewarded much and a bad addition can even be penalized.

### Adding tool usage

The first tool implemented is a `RAG` tool (Retrieval Augmented Generation). It is exclusive to the `Researcher` agent. When used, it lets the agent search for sources documents from a database. The sources used during report generation are tracked, to ensure proper citation and verifiability of the final result.

Technical changes:
- The selected `action` is passed from the graph's main loop (`arun` method) to the agent's execution (`async_execute` method).
- When an agent `execute`, it can trigger tools and modify its prompt depending on the `action` received.
- Implemented `RAG` database inside of `tools/rag`. The `RAG` also uses `oLLama` to ensure the privacy of all the documents. The whole production loop (embedding / storing / querying / generation) is fully local, and doesn't require any network connection.
- The current implementation of the `RAG` uses the `ChromaDB` library.
- Added a `SourceBuffer` singleton to track the sources used to generate each part of the report. The buffer stores the sources given by the `RAG` tool to generate the current report part and it is flushed when the part is added to the `ReportState`. Once flushed, the sources are stored in the `ReportState` with their respective part.
- The `Collector` agent is responsible for flushing the sources from the `SourceBuffer` and storing them inside of the `ReportState` when processing an `append` action.
- When a `Researcher` agent execute, it start by generating a query to the `RAG` tool, using its current context. The query is then used on the `RAG` database to find relevent chunks of information. These chunks are then added to the `Researcher`'s context before generating its notes for the next round of communication.
- To respect the graph architecture of the project, the `RAG`'s selected documents are passed to the `Researcher` agent via a `spatial` edge. With this representation, the tool can be represented by a node just like any other agent.
- Wrote prompt for preparing the query given to the `RAG` tool. The query is prepared with the communication context of the `Researcher` agent and aims to formulate a precise and effective query.
- Added `ChromaDB` and `ollama` libraries to the requirements. `ChromaDB` is used to handle the vectorial database and `ollama` is used to embed the query locally using oLLama (to avoid data leaks).

### Adding animated graph visualization

The advantage of preserving the graph structure of the original code of Agent-Q-Mix is that representing the evolution of the graph creates a very visual and accessible debug tool. The tool is programmed to track the full execution trace of the redaction process. And then plot the communication graph of the agents and its evolution round after round. 

Technical changes:
- Implemented a full `ExecutionTrace` singleton. Every action is tracked with intra-round order and context. This includes for every round:
    - execution order of the agents
    - for each agent:
        - selected action
        - list of the agents that will receive its message
        - prompt
        - response
    - the current state of the report
- When the graph is initiated with `execution_trace = True`, the whole process trace is stored inside the `ExecutionTrace` object. At the end of the execution, the `ExecutionTrace` is stored inside of a `json` file (default name: `execution_trace.json`).
- Creation of a visualizer tool inside of `utils/visualization.py`. The visualizer creates a full interactive animation from the `ExecutionTrace` `json` file. The animation is a simple `html` file (default name: `agent_trace.html`) that can be openned on any browser and contains all the data/logic of the visualization.
- The animation shows the evolution of communications during every step of every round of the process. Displaying the informations of the `ExecutionTrace` in a human friendly interface. The goal of the interface is to debug the redaction process and find the errors in a timely manner.
- On the graph each agent is represented by a `Node`. The communication are represented by `Edges`.
- The `timeline` of the exectuion is represented by a `slide bar`, and the active agent and communications at a specific execution step are highlighted for clarity. The information on the active agent is displayed to the right of the graph, alongside the state of the report at that current step.
- The `play` button lets the user observ the graph evolution by going through steps automatically at a rapid pace.

### Handcrafted graph baseline

The `handcrafted_graph/` module implements a deterministic, phase-ordered pipeline as a structural baseline for the QMIX-trained approach. It reuses the existing `Node`, `AgentRegistry`, and `Collector` infrastructure entirely unchanged, replacing only the action-selection mechanism: instead of the QMIX mixer choosing actions each round, the communication topology is hand-designed and executed in a fixed sequence of phases. This makes it possible to run and evaluate the report-writing pipeline without any RL training.

**Pipeline overview:**

The pipeline is divided into 5 ordered phases. Each phase defines one or more round patterns that cycle until the phase's maximum round count is reached.

| Phase | Max rounds | Purpose |
|-------|-----------|---------|
| PLANNING | 2 | Evidence-first outline. Researcher scans corpus coverage before LeadArchitect commits to any section titles. |
| RESEARCH | 6 | Iterative evidence gathering. LeadArchitect directs Researcher with specific queries; DataAnalyst synthesises raw evidence atoms into structured writing blueprints. |
| DRAFTING | 10 | Section-by-section writing. LeadArchitect designates the section; DataAnalyst structures content; Collector writes and appends polished prose to the report. |
| REVIEW | 2 | Reviewer audits the full draft for factual accuracy, logical coherence, and scientific rigour, then forwards structured critique to LeadArchitect. |
| REVISION | 4 | LeadArchitect applies reviewer feedback. DataAnalyst prepares corrected content; Collector replaces flagged sections in-place (not append). |

**Technical changes:**

- Created `handcrafted_graph/` module with the following components:
    - `graph.py` — `HandcraftedGraph` class. Orchestrates phase-based execution: iterates over phases, builds the spatial edge topology for each round from the phase definition, runs Kahn's topological sort (identical tie-breaking logic as `QMIXGraph`), and returns the finished report from `ReportState`.
    - `phases.py` — `PhaseType` enum, `RoundTopology` dataclass (required agents, optional agents, directed edge list) and `PhaseConfig` dataclass (name, description, round patterns, max rounds, next phase). Defines the complete `PHASE_SEQUENCE` and `PHASE_MAP` constants.
    - `scheduler.py` — `RoundScheduler` class with three `SkipStrategy` modes for optional agents: `ALWAYS_INCLUDE` (safest, highest token cost), `TEMPORAL_HEURISTIC` (include agent only if it has produced output in a prior round), and `LLM_GATECHECK` (single lightweight LLM call ~50 tokens asking the agent "EXECUTE or SKIP?" before its full execution). `LLM_GATECHECK` falls back to `TEMPORAL_HEURISTIC` automatically when no LLM instance is provided.
    - `state.py` — `PhaseState` singleton tracking the current `PhaseType` and the round index within the active phase. Agents and the prompt set read from it to tailor their behaviour without requiring it to be threaded through every call.
    - `prompts/handcrafted_prompt_set.py` — `HandcraftedPromptSet` registered under the key `"handcrafted_redacting"`. Wraps the existing `redacting_prompt_set` role descriptions and constraints, and dynamically injects: (1) a phase context block into the system prompt, and (2) a per-`(phase, role)` objective and current round number into the user-prompt context block. REVIEW and REVISION phases additionally inject the live section ID list from `ReportState.list_sections()` so agents can target corrections precisely.
    - `runner.py` — `run_handcrafted()` async function mirroring the QMIX runner interface. Resets all singletons before each run, constructs `HandcraftedGraph`, applies `filter_meta_commentary()` to the final output, and optionally saves the execution trace to `handcrafted_trace.json`.
- Added `experiments/run_handcrafted.py` as a standalone CLI entry point. Accepts `--task`, `--task-index`, `--llm`, `--skip-strategy`, `--trace`, `--max-tries`, and `--max-time` arguments.
- Spatial edges are cleared and rebuilt from scratch every round according to the current phase topology. Temporal self-edges are re-wired each round: a node that has prior-round output receives its own last output as a temporal predecessor (giving agents memory without an explicit state store).
- Round patterns within a phase cycle: if a phase has N patterns and `max_rounds > N`, pattern `round_idx % N` is selected, so the alternating topology repeats until the phase ends.
- The execution trace format exactly mirrors `QMIXGraph`, with `action=None` for all agents (no QMIX action is selected). This means the existing `StandaloneVisualizer` in `utils/visualization.py` works without any modification on handcrafted traces.
- REVISION Collector performs an in-place section replacement instead of an append. DataAnalyst prefixes its output with a `[SECTION_ID: section_X]` tag so the Collector knows exactly which section to overwrite.
- Strong anti-hallucination guardrails are embedded directly in the phase-role prompts: Researcher signals `"State Deficiency"` when the knowledge base lacks requested data; DataAnalyst propagates the signal and emits `REMOVE:` directives for unverifiable claims; Collector omits those claims entirely — an empty section is preferred over speculative prose.
- `PhaseState` is fully reset between runs via `runner._reset_singletons()`.
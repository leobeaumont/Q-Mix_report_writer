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

The library code is an installable Python package, `qmix_report_writer/`, so it
can be embedded in a host project (`pip install` + `from qmix_report_writer import
run_handcrafted`). Development entry points, tests and runtime data live at the
repo root, outside the package.

```
Q-Mix_report_writer/             # repo root
├── qmix_report_writer/          # ── installable package ──────────────────
│   ├── __init__.py              # public API (re-exports run_handcrafted)
│   ├── handcrafted_graph/       # deterministic phase-based pipeline (+ runner)
│   ├── qmix/                    # QMIX core (GNN, Q-networks, mixing, replay, trainer)
│   ├── graph/                   # multi-agent graph execution engine
│   ├── agents/                  # agent roster (LeadArchitect, Researcher, ...)
│   ├── llm/                     # LLM API layer (Ollama)
│   ├── prompt/                  # prompt sets
│   ├── tools/rag/               # RAG manager (ChromaDB + hybrid retrieval)
│   ├── utils/                   # config/paths, report export, PDF, visualization
│   ├── configs/                 # bundled default.yaml (defaults; host-overridable)
│   └── assets/                  # images used by the PDF/LaTeX template
├── experiments/                 # training and evaluation entry points
├── scripts/                     # ingestion / query utilities
├── datasets/                    # benchmark dataset loaders
├── tests/                       # test suite
├── checkpoints/                 # saved QMIX models
├── pyproject.toml               # package definition + dependencies
├── requirements.txt
└── (git-ignored runtime data: chroma_data/, output/, .tools/, *_trace.json)
```

### Configuration & data paths

`configs/default.yaml` ships inside the package as the built-in defaults. A host
overrides it without editing package files via `configure()`, a YAML override
file (`QMIX_REPORT_CONFIG`), or in-code overrides:

```python
from qmix_report_writer.utils.config import configure
configure(overrides={"paths": {"output_root": "qmix_report_writer_data"}})
```

Runtime locations resolve against two independent, configurable roots (both
default to the current working directory, so standalone use is unchanged):

- **data root** (`paths.data_root` / `QMIX_REPORT_DATA_ROOT`) — *used* resources:
  the Chroma DB (`chroma_path`) and the Tectonic cache (`tools_dir`).
- **output root** (`paths.output_root` / `QMIX_REPORT_OUTPUT_ROOT`) — files the
  pipeline *produces*: report runs (`output_dir`) and execution traces
  (`trace_file`). A host typically sets only this to group generated artifacts in
  one folder, leaving the DB in place.

## Using the package in a host project

A short end-to-end walkthrough for embedding the report writer in another
project. The public entry point is the async `run_handcrafted`.

**1. Install** — pin to a released version tag for reproducibility:

```bash
pip install git+https://github.com/leobeaumont/Q-Mix_report_writer.git@v0.1.0
```

> Use `@main` for the latest stable code, or `@<commit-sha>` to pin an exact
> commit. Pinning a tag is recommended so a given host build always resolves the
> same code.

> Runtime prerequisite: a running [Ollama](https://ollama.com) serving both the
> generation model (see `llm.default_model` in `default.yaml`) and the embedding
> model `nomic-embed-text`. PDF export additionally needs Tectonic on `PATH`
> (otherwise it is auto-downloaded; see *tectonic cache* above), or pass
> `export_pdf=False`.

**2. Wire it up** (optional) — point produced files at a dedicated folder and/or
load your own config. Call `configure()` once, before the first run:

```python
from qmix_report_writer.utils.config import configure

configure(overrides={"paths": {"output_root": "qmix_report_writer_data"}})
# or load a whole override file:  configure(config_path="host_qmix.yaml")
# or set env vars instead of calling configure():
#   QMIX_REPORT_CONFIG=host_qmix.yaml  QMIX_REPORT_OUTPUT_ROOT=qmix_report_writer_data
```

**3. Ingest documents** into the RAG store (once, or whenever the corpus changes):

```python
from qmix_report_writer.tools.rag import RAGManager

rag = RAGManager()                          # uses paths.chroma_path under the data root
rag.add_document_from_path("docs/paper.pdf")  # also supports .txt/.md/.docx
```

**4. Generate a report:**

```python
import asyncio
from qmix_report_writer import run_handcrafted

answers, total_tokens = asyncio.run(run_handcrafted(
    task="Write a technical report on graphene synthesis.",
    export_pdf=True,      # set False to skip LaTeX/PDF and keep only markdown
))
report_markdown = answers[0]
```

Artifacts are written under `<output_root>/output/<timestamp>_<slug>/` (raw
markdown, `.tex`, `.pdf`); the vector DB stays at `<data_root>/chroma_data`. From
inside an existing async context, `await run_handcrafted(...)` directly instead of
`asyncio.run`.

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
- Added cosine distance threshold filtering (ceiling: 0.7). Chunks below the relevance bar are discarded before reranking; if all candidates fail the filter, the method returns an empty list, triggering `[RESEARCH_EXHAUSTED]` rather than injecting noise.
- Added cross-encoder reranking with `BAAI/bge-reranker-base` loaded via `transformers` (bypasses the `sentence_transformers` import chain, which conflicts with the local `datasets/` module). The cross-encoder scores each `(query, chunk)` pair jointly, far more accurately than cosine similarity. Retrieves 15 candidates, returns top 3 after reranking. Model is lazy-loaded on first query.
- Multi-query expansion: the RAG Tool prompt now requests 3 semantically distinct queries per retrieval — one targeting implementation details, one targeting system-level context, one targeting behavioral/process aspects. Candidates from all 3 queries are deduplicated by chunk ID before reranking. The RAG Tool constraint prompt enforces a hard 8-term limit per query line and introduces a `NO_QUERY` fallback sentinel so the LLM never echoes pipeline signals as queries.
- Page number metadata: `add_document_from_path()` tracks a running `current_page` counter across chunks by detecting `[PAGE N]` markers already present in PDF text. Page number is stored in each chunk's ChromaDB metadata; query results now carry real page references instead of `"N/A"`.
- Contextual chunk prefixing: each chunk is embedded with a one-line prefix `[Source: filename | Page: N]` prepended to the text. The stored document text remains the original (no prefix). This anchors semantically vague chunks to their document and page context at embedding time. Because the prefix is embedded together with the chunk, the chunk budget reserves prefix headroom (`effective_chunk_size = min(chunk_size, 512 - 24)`) so prefix + chunk stays within nomic-embed-text's 512-token encoder limit. The server tokenizes with nomic's WordPiece tokenizer (which splits dense math/LaTeX text into more tokens than tiktoken's cl100k), so a chunk under the cl100k budget can still exceed the server's 512-token `n_ubatch` and crash the runner (`Post .../embedding: EOF` → HTTP 500). Rather than pre-shrinking every chunk on a guess, `_embed_batch` self-heals: if a batch is rejected it re-embeds that batch one chunk at a time and progressively truncates only the specific chunk(s) that keep failing, so good chunks keep their full text and an oversized chunk fails neither its batch-mates nor the document.
- Replaced `OllamaEmbeddingFunction.__call__()` (which routes through ChromaDB's internal server causing random-port connection failures) with a direct HTTP call to Ollama's `/api/embed` endpoint via `urllib`, matching the approach in `llm/ollama_chat.py`.
- Added an in-memory BM25 index (`BM25Okapi` from `rank_bm25`) built from all ChromaDB documents at initialisation time and rebuilt after every mutation. BM25 handles exact-term matching that dense vector embeddings dilute — critical for project-specific acronyms, component names, and internal identifiers the embedding model has not seen.
- Added Reciprocal Rank Fusion (`_rrf_merge`, k=60): merges the vector search and BM25 ranked lists into a single ranking by summing `1 / (k + rank + 1)` scores per document. Chunks appearing in both lists accumulate higher scores and surface ahead of chunks found by only one path.

### Adding the PBDS parameter-dependency tool

The second tool is a `PBDS` tool (Parameter-Based Dependency System). It reads an Excel parameter workbook whose cell formulas encode the dependency graph between the system's parameters (applied here to nuclear-reactor components). When the `Researcher` retrieves evidence, the tool detects which parameters the retrieved chunks discuss and surfaces the parameters connected to them — their **sources** (causes) and **effects** (consequences) — together with the formula that links them. This lets the pipeline broaden a report to cover the upstream and downstream of a topic instead of the topic alone. The tool is optional: when no workbook is configured the pipeline runs exactly as before.

Technical changes:
- Implemented the tool under `tools/pbds/`. `pdbs_pareto_core.py` is a trimmed import of the PBDS repository keeping only the graph path: it reads the parameter sheet with `openpyxl` (formulas preserved, `data_only=False`), classifies rows, and builds a `networkx` directed dependency graph in which an edge `source → dependent` means the dependent's cell formula references the source. Each edge stores the connecting Excel formula(s) so the relationship can be reported.
- `PBDSManager` (`pbds_manager.py`) builds the graph once (lazily, cached) and answers k-hop neighbourhood queries: `connected_nodes` / `neighborhood` return, for a given parameter, the nodes reachable within `k` hops split into **sources** (upstream, via `nx.predecessors`) and **effects** (downstream, via `nx.successors`), each carrying its hop distance, the path, and the connecting formula(s).
- `NodeMatcher` (`node_matcher.py`) maps a RAG chunk to exact parameter names in two layers (precision-leaning — a false positive pollutes the report, a false negative only misses an enrichment):
    - **Layer 1 (deterministic, recall-first):** BM25 (`rank_bm25`, reused from the RAG tool) over the parameter *descriptions* as the corpus with the chunk as the query. Descriptions (e.g. "Pellet diameter") are the human bridge; the raw node names (e.g. `Pellet_diameter_F8`) never appear verbatim in prose. IDF down-weights generic shared terms (`core`, `power`) and rewards rare specific ones (`MOX`, `enrichment`); a small stdlib singularizer aligns plurals. Produces a shortlist.
    - **Layer 2 (LLM verifier):** the shortlist + chunk are sent to an LLM whose answer is constrained by a JSON schema to a subset of the shortlisted node ids (multi-pick) or an empty list. This rejects mere keyword overlap and disambiguates near-duplicate descriptions. Model output is parsed leniently (some models wrap the JSON in markdown fences despite the schema).
- Activation is configurable and overloadable exactly like the other paths. `pbds.workbook_path` in `configs/default.yaml` resolves against the data root (`get_pbds_workbook_path`); it is overloadable via `configure(overrides=...)`, a `QMIX_REPORT_CONFIG` override file, or the `QMIX_REPORT_PBDS_WORKBOOK` env var. `get_active_pbds_workbook()` is the existence gate, and `load_pbds_manager()` returns a ready `PBDSManager` when a readable workbook exists or `None` (a silent skip) otherwise — the pipeline is unchanged when the tool is off.
- Added `openpyxl` to the requirements (the only new dependency; `networkx` and `pandas` were already present transitively). `pandas` read-Excel cannot expose cell formulas, so `openpyxl` is required rather than substitutable.
- Wired the tool into the `Researcher` agent only. It activates in `__init__` (guarded by try/except so a missing or invalid workbook disables it silently) and runs after retrieval on the combined retrieved chunks. The connected parameters are rendered as a dedicated, high-priority prompt section (`### Parameter Dependency Analysis`, added via an optional `pbds_block` argument to `Node._build_user_prompt`) that appears only when the tool is active and actually produced results — deliberately not injected as a low-priority inter-agent message.
- The trigger is phase-tailored (no-op outside PLANNING/RESEARCH/DRAFTING): in **RESEARCH** the connected parameters are framed as candidate next research targets (feeding the LeadArchitect's next-query loop); in **PLANNING** only their names/descriptions are shown (no formulas) as outline-coverage hints; in **DRAFTING** only the immediate (`k=1`) relationships and formulas of the section's parameters are shown, to keep causal statements accurate without introducing new topics.
- Anti-recursion via a per-run frontier set (`_pbds_surfaced_nodes`): in PLANNING/RESEARCH every surfaced parameter (a matched node and its neighbours) is recorded, and already-surfaced nodes are skipped on later retrievals. This prevents documents retrieved *for* a connected parameter from re-triggering the tool, so research keeps covering the subject instead of walking the dependency graph. DRAFTING ignores the frontier (it annotates, it does not seed queries).
- PBDS activity is logged in the execution trace under a `"PBDS"` round entry (mirroring the `"RAG"` entry): the `prompt` records the active phase and the matched parameter ids, the `response` records the surfaced block (or `[NO MATCH]`), and the `Researcher ↔ PBDS` message edges and `exec_order` are recorded so the tool's behaviour can be verified from a live run's trace.

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
- **Phase early-exit signals** allow the pipeline to terminate a phase before its round budget is exhausted. The LeadArchitect emits `[DRAFTING_COMPLETE]` when all planned sections have been written or exhausted. The Researcher emits `[RESEARCH_EXHAUSTED]` (code-level, before the synthesis LLM call) when the RAG database returns no documents, or as an LLM output when all retrieved documents are irrelevant to the current directive. The LeadArchitect emits `[REVISION_COMPLETE]` once all Reviewer-flagged corrections have been applied. The graph detects each signal after the round it appears in and breaks the phase loop immediately, avoiding idle rounds consuming the remaining budget.
- **Task-sentinel coordination** advances section assignments across DRAFTING and REVISION without relying on LLM memory. After writing a section the Collector sets `ReportState.task` to `[SECTION_COMPLETE — ASSIGN NEXT SECTION]`; after a failed attempt where the DataAnalyst returned only State Deficiency or absence markers it sets `[SECTION_SKIPPED — ASSIGN NEXT SECTION]`; after a successful in-place correction in REVISION it sets `[CORRECTION_APPLIED — ASSIGN NEXT CORRECTION]`. The LeadArchitect reads the active sentinel as its `Current Team Objective` at the start of each Round A and advances to the next topic accordingly.
- **Phase-boundary memory reset**: all agent temporal memory (`last_memory`) is wiped at the start of every phase. This prevents stale outputs from a finished phase (e.g. DRAFTING-style `<task>` tags) from pattern-biasing agent behaviour in the next phase. All durable cross-phase context — report content, section list, and current directive — is carried by `ReportState`, which is the intended persistent state store.
- **Researcher hold mechanism in REVISION**: before calling the RAG tool the Researcher performs a code-level gap check. In REVISION Round A (where the `DataAnalyst→Researcher` spatial edge is active) it inspects DataAnalyst's output: if no `State Deficiency` entry is present it returns `[HOLD]` immediately without querying RAG. In REVISION Round B (no incoming edge from DataAnalyst) it re-emits its Round A evidence directly to DataAnalyst rather than issuing a redundant second RAG call. The `TEMPORAL_HEURISTIC` scheduler recognises `[HOLD]` and `[RESEARCH_EXHAUSTED]` as non-productive outputs and excludes the Researcher from the following round.
- **PLANNING deficiency persistence**: the Researcher parses `State Deficiency` entries from its PLANNING coverage response and stores them in `ReportState.deficient_topics` via `add_deficiency()`. The LeadArchitect's `_process_inputs` injects this list as a dedicated `Topics absent from knowledge base` block in its `Current report state` context during PLANNING, making unavailable topics an explicit named constraint rather than free text buried in received messages.
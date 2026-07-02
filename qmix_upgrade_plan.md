# QMIX pipeline upgrade — implementation plan & change tracker

**Created:** 2026-07-02 · **Companion:** [qmix_improvement_report.md](qmix_improvement_report.md) (the *why*; this file is the *what, in which order, and what has been done*).

## How to use this file

- Work the stages **top to bottom**; items inside a stage are ordered too. Dependencies are noted where they cross stages.
- Status boxes: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked / waiting on a decision.
- When an item is finished: tick it and fill its **Done note** with 1–3 lines (what was actually done, anything that deviated from the description, commit hash if useful).
- Report references like *(report 2.3)* point to the numbered items in `qmix_improvement_report.md`.
- Open questions live in the **Open decisions** section at the bottom; each has a recommendation so they can be settled with one word.

---

## Decision log (settled 2026-07-02)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Foundation for the upgraded QMIX pipeline | **Handcrafted-first**: QMIX is grafted onto the handcrafted architecture; phases, scaffolding, finalization are reused, not re-implemented. |
| D2 | Integration mechanism | **Controller injection**: `HandcraftedGraph` gains one seam — where it consults `RoundTopology` tables + `RoundScheduler` today, it asks a *RoundController* object for the round plan. Default controller reproduces current behavior exactly; the QMIX controller replaces only that seam. One phase-execution implementation, zero duplicated loops. |
| D3 | Legacy free-form QMIX path | **Delete now**: `graph/graph.py::QMIXGraph`, `experiments/run_qmix_train.py`, `experiments/run_qmix_eval.py`, `scripts/query_qmix.py` are removed (git history keeps them). The `qmix/` networks + trainer + replay buffer stay — they are reused by the new controller. Legacy-only Tier-0 bug fixes are dropped; their *lessons* land in the new code (marked below). |
| D4 | Phase handling in QMIX runs | **Phases are mandatory** (user edit to report 1.3/2.1): the QMIX pipeline drives `PhaseState` exactly like handcrafted; no free-form / phase-neutral mode is built. |
| D5 | `terminate` action | **Dropped** (user edit to report 2.6): episodes end when the phase pipeline completes, the handcrafted way. Action space shrinks accordingly. |
| D6 | Review/validation during training | **Skipped in training episodes** (report 2.5): training runs `PLANNING → RESEARCH → DRAFTING` only, with finalization off; inference runs the full pipeline. Revisit when the evaluator is reworked. |

**Standing constraint (all stages):** the handcrafted pipeline must remain fully usable, with **identical default behavior and public API** (`run_handcrafted`, `HandcraftedGraph(...)` signatures keep working unchanged). Internal refactors of shared files are allowed; behavior drift is not. Enforced by the regression checkpoint items (0.1, 2.5, 3.4).

**Out of scope here:** trainer algorithm internals (`qmix_trainer.py` loss/networks) and evaluator judge internals (`experiments/eval.py` prompts/scoring) — both scheduled for a separate rework. Only their *integration points* (masks in action selection, reward trigger condition, obs dimensions) are touched.

---

## Target architecture (orientation)

```
                        ┌───────────────────────────────┐
 run_handcrafted ─────► │  HandcraftedGraph.arun()      │ ◄───── run_qmix (new)
                        │  phases, sections, sentinels, │
 HandcraftedRound-      │  validation loop, trace       │   QMIXRoundController
 Controller (default,   │                               │   (network + masks +
 = today's tables +     │  seam: controller.round_plan()│    episode recording +
 scheduler)             │  hook: controller.on_round_end│    reward events)
                        └───────────────┬───────────────┘
                                        │ shared, unchanged
                        agents · ReportState · LLM layer · RAG · PBDS
                        + extracted: report_finalize · trace helpers
```

- Training: `run_qmix_train` builds the graph with `QMIXRoundController(train=True)`, `phases=[PLANNING, RESEARCH, DRAFTING]`, finalization off; after each `arun()` the recorded episode goes to the replay buffer and `train_step()` runs.
- Inference: `run_qmix` uses the greedy controller with the **full** phase list and finalization on → same artifacts as handcrafted (markdown, PDF, trace, citations, bibliography, abstract).
- Scripted rounds stay scripted (env rules, not policy choices): DRAFTING write round incl. blueprint-reuse, SECTION_REVIEW review/revision/directive-bypass, VALIDATION windows + synthesis. The controller is consulted exactly where tables+scheduler are consulted today.

---

## Stage 0 — Guardrails (before touching anything)

### 0.1 `[ ]` Capture the handcrafted baseline
- **What:** Run the existing test suite (`tests/test_review_pipeline_fixes.py`, `test_directive_review.py`, `test_remove_duplicate_section.py`, PBDS tests) with `.venv\Scripts\python.exe` and record the result. Do one full smoke run of `run_handcrafted` on a known task with `--trace`, and archive the trace + report output (e.g. copy into `tests/baseline_refs/2026-07-02/`) as the reference for later regression checkpoints.
- **Why:** every later "behavior unchanged" claim needs something concrete to compare against.
- **Done note:** —

---

## Stage 1 — Legacy removal & housekeeping (D3)

### 1.1 `[ ]` Delete the legacy free-form QMIX path
- **What:** Delete `qmix_report_writer/graph/graph.py`, `experiments/run_qmix_train.py`, `experiments/run_qmix_eval.py`, `scripts/query_qmix.py`. Update `qmix_report_writer/graph/__init__.py` to export only `Node` (the `from .graph import QMIXGraph` line goes). Keep `graph/node.py` (used by everything) and keep `experiments/eval.py` (`report_score`/`length_score` are reused by Stage 4.4 — verify it imports cleanly standalone after the deletions).
- **Absorbed lessons** (legacy bugs whose fix now lands in new code instead): roster read from `agent_configs` config key → Stage 4.6 *(report 0.1)*; reward only on *successful* append → Stage 4.4 *(report 0.8)*; `SourceBuffer` reset per episode → Stage 4.6 *(report 0.6)*; self-query masked → Stage 4.1 *(report 0.7)*. Bugs 0.2/0.3/0.4 die with the deleted files.
- **Done note:** —

### 1.2 `[ ]` Config hygiene pass *(report 0.5)*
- **What:** In `configs/default.yaml`: delete the now-meaningless `evaluation:` section and the stale `training.num_rounds`; leave a minimal `qmix:` section with a comment that its values are finalized in Stage 4 (n_actions, dims). Decide layout for new-runner options (e.g. `qmix.training.num_episodes`, `qmix.training.epsilon_*`) so Stage 4.6 reads config instead of hard-coded argparse defaults, with CLI flags as overrides.
- **Done note:** —

---

## Stage 2 — Shared refactors (behavior-neutral; handcrafted must stay green)

> Order matters: 2.1 first, so every later agent-level change is written once, not twice.

### 2.1 `[ ]` Consolidate agents' sync/async duplication *(report non-port #1)*
- **What:** In each agent (`researcher.py` is the big one, plus `collector.py`, `lead_architect.py`, `data_analyst.py`, `reviewer.py`, `technical_writer.py`): collapse `_execute`/`_async_execute` into one async implementation; the sync variant becomes a thin wrapper (same pattern as `OllamaChat.gen` → `agen`). No behavior change.
- **Done note:** —

### 2.2 `[ ]` Extract report finalization into a shared module *(report 1.2)*
- **What:** Move `_apply_citation_tags`, `_rewrite_inline_references`, `_strip_orphan_citation_markers`, `_build_bibliography`, `_format_bib_entry`/`_compose_reference`, `_generate_abstract` (+ their regex constants and `_tokenize`) out of `handcrafted_graph/graph.py` into `utils/report_finalize.py` (functions taking `report_state` and, where needed, an LLM). `HandcraftedGraph` delegates to it — same call sites, same behavior.
- **Also:** add a `finalize: bool = True` parameter to `HandcraftedGraph.arun()` gating abstract + bibliography assembly (citation tagging stays tied to SECTION_REVIEW, which training runs don't execute anyway). Default `True` → handcrafted unchanged; training passes `False` (needed by D6, used in Stage 4.6).
- **Done note:** —

### 2.3 `[ ]` Extract trace helpers *(report 1.5)*
- **What:** Move `_init_trace_round` / `_trace_spatial_edges` into a small shared helper (e.g. `utils/trace.py` or a mixin) so the round-slot schema lives in one place. Add an optional per-agent `action` field populated when a controller supplies actions (Stage 4.2) — `None` for handcrafted runs, exactly like today, so `StandaloneVisualizer` keeps working on both.
- **Done note:** —

### 2.4 `[ ]` (Optional, recommended) Split the validation loop out of `graph.py` *(report non-port #3)*
- **What:** Move `_decompose_validation_directive`, `_validation_windows`, `_build_section_windows`, `_revalidation_sections` into e.g. `handcrafted_graph/validation.py`. Pure code motion; shrinks `graph.py` toward its phase-execution core before Stage 3 modifies it.
- **Done note:** —

### 2.5 `[ ]` Regression checkpoint #1
- **What:** Re-run the Stage 0 protocol (tests + smoke run) and compare against the baseline: same phase sequence in the trace, same section count, bibliography/abstract present, no new warnings. Record findings here.
- **Done note:** —

---

## Stage 3 — The controller seam in `HandcraftedGraph` (D2)

### 3.1 `[ ]` Define the `RoundController` interface
- **What:** New module (suggest `handcrafted_graph/controller.py` — it is part of the graph's contract, not QMIX-specific). Interface (async where it may call LLMs/networks):
  - `round_plan(phase, round_idx, topology, nodes, task_input) -> RoundPlan` where `RoundPlan = (active_agents: set[str], edges: list[(str, str)], actions: dict[str, int] | None)` — `actions` is what gets rendered into prompts and recorded in the trace (`None` = handcrafted mode).
  - `on_round_end(phase, round_idx) -> None` — hook for episode recording / reward events (no-op by default).
  - `on_phase_start(phase)` / `on_run_end()` — lifecycle hooks (no-op by default).
- **Scope note (document in the module docstring):** the controller is consulted **only** where `phase.round_topologies[...]` + `scheduler.get_active_agents(...)` are consulted today — the generic phase loop, the DRAFTING prep round, and the DRAFTING round-B *retry* branch. Scripted rounds (blueprint-reuse write, SECTION_REVIEW, VALIDATION, directive-bypass) never consult it.
- **Done note:** —

### 3.2 `[ ]` `HandcraftedRoundController` (default) + wire the seam
- **What:** Implement the default controller wrapping today's logic (topology tables + `RoundScheduler` with the configured skip strategy). `HandcraftedGraph.__init__` gains `controller: RoundController | None = None` → defaults to `HandcraftedRoundController` built from the existing `skip_strategy` arg (public API unchanged). Replace the direct table/scheduler consultations in `_execute_phase` / `_execute_drafting_phase` with `controller.round_plan(...)`; add the `on_round_end` / `on_phase_start` calls at the existing round boundaries.
- **What (prompt path):** `_execute_round` forwards the plan's per-agent `action` (when not `None`) into `async_execute(..., action=...)` so the prompt context block can render it — handcrafted runs pass nothing and behave as today.
- **Done note:** —

### 3.3 `[ ]` Training-mode run shape verification
- **What:** Verify (with a stub controller and a tiny/dummy LLM) that `HandcraftedGraph(phases=[PLANNING, RESEARCH, DRAFTING]).arun(..., max_validation_attempts=0, finalize=False)` cleanly skips the correction stages: no SECTION_REVIEW/VALIDATION rounds, no directive decomposition on empty reviewer output, no abstract/bibliography, no crash in the validation-loop bookkeeping (progress bars, trace stamps). Fix whatever edge cases surface — this is the exact configuration Stage 4.6 training uses (D6).
- **Done note:** —

### 3.4 `[ ]` Regression checkpoint #2
- **What:** Stage 0 protocol again on the default-controller path. The handcrafted pipeline with no arguments changed must produce an equivalent run (trace round sequence identical to checkpoint #1).
- **Done note:** —

---

## Stage 4 — The QMIX decision layer

### 4.1 `[ ]` Action space v2 + per-phase masks *(report 2.3, 2.4; D5)*
- **What:** Redefine in `qmix/agent_network.py`: `0 no_op` (agent does not execute this round — replaces `solo_process` semantics, *report 2.3*), `1 broadcast_all`, `2–5 selective_query(target 0–3)`, `6 aggregate_refine`, `7 append` → **NUM_ACTIONS = 8** (`terminate` dropped per D5). Note in code that "think privately" (old solo) can be re-added later if an ablation wants it.
- **What:** New `qmix/action_masks.py`: `mask(phase, agent_name, round_ctx) -> bool[8]`. Initial rules: self-targeting `selective_query` always masked *(absorbs report 0.7)*; `append` masked in PLANNING/RESEARCH (and per open decision **OD-1** in DRAFTING); `append` always masked for the Reviewer; `no_op` masked for agents that are hard-required by the phase (e.g. Researcher+LeadArchitect in PLANNING round 0) so a random policy cannot produce a dead round. Keep the rules table in one visible place — they encode the handcrafted topology knowledge as constraints.
- **Done note:** —

### 4.2 `[ ]` `QMIXRoundController`
- **What:** New `qmix/qmix_controller.py` implementing the Stage 3.1 interface: builds observations (v1 = existing 51-dim features + phase one-hot; v2 lands in Stage 5), calls masked action selection, translates actions → `RoundPlan` (`no_op` → agent excluded from `active_agents`; `broadcast`/`selective`/`aggregate` → edges among active agents; `append` → edge to Collector), records `(obs, actions, adj, global_state)` per round, sets `done` on run end. Holds the GRU hidden state across rounds; `train` flag switches ε-greedy vs greedy. Port `get_observation_features` / `get_global_state` / adjacency derivation from the deleted `QMIXGraph` (git) into this module or a small `qmix/observations.py`.
- **Edge-case policy (document):** cycle-denied edges fall back exactly like legacy (edge dropped); an all-`no_op` round advances the round counter with no execution (naturally discouraged by reward-per-token, and prevented in required rounds by masks).
- **Done note:** —

### 4.3 `[ ]` Masked action selection in the trainer (integration-only change)
- **What:** `QMIXTrainer.select_actions` accepts an optional `mask: (n_acting, n_actions)` bool tensor — masked entries set to `-inf` before argmax and excluded from the ε-random draw. Store masks in `EpisodeStep` so the (future, reworked) training step can apply them to target-max computations too. No other trainer changes.
- **Done note:** —

### 4.4 `[ ]` Reward event hook *(absorbs report 0.8; evaluator internals untouched)*
- **What:** In `QMIXRoundController.on_round_end`: detect a **successful** append via `len(ReportState.additions)` delta (not via the chosen action) → update `Score` (`await report_score()`) and `LengthGoal` (`length_score(...)`), compute the delta reward (`compute_reward`), spread it evenly over the steps buffered since the last reward event (port of the old `step_buffer` logic from git, now event-correct); leftover steps flush with reward 0 at `on_run_end`. `NoCorpusCoverageError` → episode ends, buffered steps flush at reward 0 (simple v1; revisit with the reward rework).
- **Done note:** —

### 4.5 `[ ]` Prompt merge *(report Tier 3)*
- **What:** `HandcraftedPromptSet.get_context_block` additionally renders the QMIX action line when `kwargs.get("action")` is not `None` (reuse `_QMIX_ACTION_DESCRIPTIONS` from `redacting_prompt_set.py`, updated for the v2 action list). Handcrafted runs pass no action → byte-identical prompts (verify in checkpoint 4.8).
- **Done note:** —

### 4.6 `[ ]` Runners: `run_qmix_train` + `run_qmix` *(report 1.1; absorbs 0.1, 0.6)*
- **What:** New `qmix/runner.py` mirroring `handcrafted_graph/runner.py`, plus thin CLIs `experiments/run_qmix_train.py` (new file) and `experiments/run_qmix.py`:
  - **Shared per-run plumbing** (reuse/extend the handcrafted `_reset_singletons`, adding `SourceBuffer` — it already has it — and the Researcher's `_reported_chunk_ids` via fresh graph construction): trace saved in `finally`, roster read from `agent_configs` (correct key), config-first defaults (Stage 1.2).
  - **Train mode:** per episode — fresh graph + `QMIXRoundController(train=True)`, `phases=[PLANNING, RESEARCH, DRAFTING]`, `max_validation_attempts=0`, `finalize=False`; after `arun`: push recorded episode, `train_step()`, ε decay, checkpoint save; per-episode stats print (score, reward, tokens, ETA — port the old console format).
  - **Inference mode:** greedy controller, full phase list, `finalize=True`, then the full artifact path: `filter_meta_commentary` → `save_raw_report` → optional LaTeX/PDF (same flags as `run_handcrafted`: `--task`, `--llm`, `--trace`, `--no-pdf`, plus `--model-path`).
- **Done note:** —

### 4.7 `[ ]` Translation-fidelity test (scripted controller)
- **What:** A test controller that emits, through the *QMIX action vocabulary*, the plans the handcrafted tables would produce for PLANNING/RESEARCH/DRAFTING (e.g. Researcher `selective_query(LA)` in planning round 0…). Run both pipelines on the same task with the same seed/model and compare traces: active sets and edges per round must match. This proves the action→plan translation and the seam are faithful before any training happens.
- **Done note:** —

### 4.8 `[ ]` Regression checkpoint #3 + random-policy dry run
- **What:** (a) Handcrafted default path still matches baseline. (b) `run_qmix_train` with an untrained network and small budgets (2 episodes, tiny model): no crashes, masks respected (assert in controller), episodes land in the buffer with sane shapes, reward events fire only on real appends, trace renders in the visualizer with actions shown.
- **Done note:** —

---

## Stage 5 — Observation upgrade *(report 2.7)*

### 5.1 `[ ]` Structured state features
- **What:** Extend the obs builder (from 4.2) with: sections written, planned−written remaining, `len(content)/length_goal`, last append outcome (task sentinel is `SECTION_COMPLETE` vs `SECTION_SKIPPED` vs none), research-exhausted flag, per-agent last-output-is-sentinel flag, phase one-hot (moved from 4.2 if simpler to keep together). Recompute `obs_dim`/`state_dim` in one place used by both controller and trainer construction; old checkpoints are invalid (fine — none are, post-D3).
- **Done note:** —

### 5.2 `[ ]` Embedding features (flag-gated)
- **What:** Replace the byte-hash task/progress features with `nomic-embed-text` embeddings via the existing local Ollama embed path (see `rag_manager`'s direct `/api/embed` call): task embedded once per episode, progress summary re-embedded only after successful appends (cache), truncated/projected to a configurable dim (start 64). Config flag `qmix.obs.use_embeddings` so the training rework can A/B hash vs embeddings.
- **Done note:** —

---

## Stage 6 — Docs & final cleanup

### 6.1 `[ ]` README rewrite *(report 0.9)*
- **What:** Update the QMIX sections: new architecture diagram (controller-in-handcrafted), v2 action table (8 actions, masks), phases-in-env, training/inference run modes; remove references to deleted scripts, the old checkpoints table (checkpoints are void post-D3/D5), and the outdated REVIEW/REVISION phase description (now SECTION_REVIEW/VALIDATION + retry loop). Link the report and this plan.
- **Done note:** —

### 6.2 `[ ]` Roster cleanup — TechnicalWriter *(report non-port #2; see OD-2)*
- **What:** Per OD-2 decision: delete `agents/technical_writer.py` + its `redacting` prompt entries (Collector absorbed the role), or leave registered with a comment. Either way, note that roster changes require revisiting selective-query target count and masks.
- **Done note:** —

### 6.3 `[ ]` Config finalization
- **What:** `qmix:` section reflects reality (n_actions 8, obs flags, training defaults actually read by the runner — closing Stage 1.2's placeholder); prune anything still dead.
- **Done note:** —

### 6.4 `[ ]` `datasets/` pruning *(see OD-3)*
- **What:** Per OD-3: keep `datasets/tasks.py` (training task list — still used); the benchmark loaders (`gaia/hle/mmlu/humaneval/livecodebench/math/...` + `get_dataset`) were only consumed by the deleted legacy eval. Delete or keep per decision.
- **Done note:** —

### 6.5 `[ ]` Memory/report sync
- **What:** Update `qmix_improvement_report.md` status header (plan executed) and the assistant memory files if architecture facts changed during implementation.
- **Done note:** —

---

## Open decisions (answer whenever — blocking only the item referenced)

### OD-1 — Who triggers the write in DRAFTING? (blocks final mask table in 4.1)
The handcrafted write round is scripted: when Round A produced a usable blueprint, the env wires DataAnalyst→Collector itself. Options:
- **(a) Keep the write scripted (recommended for v1):** QMIX controls only prep-round communication; `append` stays masked in DRAFTING too (the action then exists only for future experiments). Lowest variance, every episode produces sections, reward events are regular. The policy learns *who talks to whom before writing*.
- **(b) Policy-triggered append:** DataAnalyst (and Researcher?) may choose `append`; the env keeps a scripted fallback write in round B if nobody appended. More is learnable (when to write), slower early training, reward events irregular.
- **Choice:** **(a)** · **Done note:** —

### OD-2 — TechnicalWriter agent (blocks 6.2)
- **(a) Delete (recommended):** dead code since the Collector absorbed the writing role; smaller roster surface for masks/actions.
- **(b) Keep registered but rosterless:** zero work now, keeps the option of a 6-agent roster experiment.
- **Choice:** **(a)** · **Done note:** —

### OD-3 — Benchmark dataset loaders (blocks 6.4)
- **(a) Delete (recommended):** `datasets/` shrinks to `tasks.py` (+ `base_dataset` if anything imports it); the problem-solving benchmarks belong to the upstream Agent-Q-Mix project, not report writing.
- **(b) Keep:** if you foresee re-running problem-solving benchmarks for a paper baseline.
- **Choice:** **(a)** · **Done note:** —

---

## Progress overview

| Stage | Items | Done |
|-------|-------|------|
| 0 — Guardrails | 1 | 0 |
| 1 — Legacy removal | 2 | 0 |
| 2 — Shared refactors | 5 | 0 |
| 3 — Controller seam | 4 | 0 |
| 4 — QMIX decision layer | 8 | 0 |
| 5 — Observations | 2 | 0 |
| 6 — Docs & cleanup | 5 | 0 |
| **Total** | **27** | **0** |

# QMIX pipeline upgrade — implementation plan & change tracker

**Created:** 2026-07-02 · **Companion:** [qmix_improvement_report.md](qmix_improvement_report.md) (the *why*; this file is the *what, in which order, and what has been done*).

## How to use this file

- Work the stages **top to bottom**; items inside a stage are ordered too. Dependencies are noted where they cross stages.
- Status boxes: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked / waiting on a decision.
- When an item is finished: tick it and fill its **Done note** with 1–3 lines (what was actually done, anything that deviated from the description, commit hash if useful).
- Report references like *(report 2.3)* point to the numbered items in `qmix_improvement_report.md`.
- Open questions live in the **Open decisions** section at the bottom; each has a recommendation so they can be settled with one word.
- **Side-validation tests** (added 2026-07-02, both run standalone with `.venv\Scripts\python.exe`):
  - `tests/test_upgrade_plan_guards.py` — 20 **regression pins**, all green on the pre-change code. They must STAY green after every stage; one turning red means behavior drifted. Where a stage moves code (2.2/2.4), only the test's import is updated, never its assertions.
  - `tests/test_upgrade_plan_acceptance.py` — 12 **acceptance tests** encoding each item's target state. They report `PEND` until the stage lands, then `PASS` (or `FAIL` if the landed API violates the contract). An item with an acceptance test is done when it flips to PASS — note it in the Done note.
  - Items below carry a **Test:** line naming their tests.

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

### 0.1 `[x]` Capture the handcrafted baseline
- **What:** Run the existing test suite (`tests/test_review_pipeline_fixes.py`, `test_directive_review.py`, `test_remove_duplicate_section.py`, PBDS tests) **plus the two new plan suites** (`test_upgrade_plan_guards.py`, `test_upgrade_plan_acceptance.py`) with `.venv\Scripts\python.exe` and record the result. Do one full smoke run of `run_handcrafted` on a known task with `--trace`, and archive the trace + report output (e.g. copy into `tests/baseline_refs/2026-07-02/`) as the reference for later regression checkpoints.
- **Why:** every later "behavior unchanged" claim needs something concrete to compare against.
- **Test:** baseline recorded 2026-07-02 on pre-change code: guards 20/20 PASS, acceptance 12 PEND / 0 FAIL.
- **Done note:** 2026-07-02 — offline battery all green (49 passed across 7 suites; live suites skipped, need Ollama). Instead of a fresh smoke run, archived the last pre-change full-run artifacts (2026-07-01 trace + report) into `tests/baseline_refs/2026-07-02/` with a README recording the battery results. A fresh confirmation run will be requested from the user at checkpoint 2.5.

---

## Stage 1 — Legacy removal & housekeeping (D3)

### 1.1 `[x]` Delete the legacy free-form QMIX path
- **What:** Delete `qmix_report_writer/graph/graph.py`, `experiments/run_qmix_train.py`, `experiments/run_qmix_eval.py`, `scripts/query_qmix.py`. Update `qmix_report_writer/graph/__init__.py` to export only `Node` (the `from .graph import QMIXGraph` line goes). Keep `graph/node.py` (used by everything) and keep `experiments/eval.py` (`report_score`/`length_score` are reused by Stage 4.4 — verify it imports cleanly standalone after the deletions).
- **Absorbed lessons** (legacy bugs whose fix now lands in new code instead): roster read from `agent_configs` config key → Stage 4.6 *(report 0.1)*; reward only on *successful* append → Stage 4.4 *(report 0.8)*; `SourceBuffer` reset per episode → Stage 4.6 *(report 0.6)*; self-query masked → Stage 4.1 *(report 0.7)*. Bugs 0.2/0.3/0.4 die with the deleted files.
- **Test:** acceptance `test_stage1_1_legacy_deleted` (PEND→PASS); guard `test_node_import_and_eval_standalone` must stay green (Node export + `experiments.eval` standalone).
- **Done note:** 2026-07-02 — `git rm` of the four legacy files; `graph/__init__.py` now exports only `Node`. Verified `qmix/__init__.py` untouched by the deletion (imports only networks/trainer/buffer) and `experiments.eval` imports standalone. Acceptance test PASS.

### 1.2 `[x]` Config hygiene pass *(report 0.5)*
- **What:** In `configs/default.yaml`: delete the now-meaningless `evaluation:` section and the stale `training.num_rounds`; leave a minimal `qmix:` section with a comment that its values are finalized in Stage 4 (n_actions, dims). Decide layout for new-runner options (e.g. `qmix.training.num_episodes`, `qmix.training.epsilon_*`) so Stage 4.6 reads config instead of hard-coded argparse defaults, with CLI flags as overrides.
- **Test:** acceptance `test_stage1_2_config_hygiene` (PEND→PASS).
- **Done note:** 2026-07-02 — dropped stale `n_actions: 16` (action count lives in code), deleted top-level `training:`/`evaluation:` sections; runner options now live under `qmix.training.*` (num_episodes, epsilon_start/end, log/save intervals) for Stage 4.6 to consume. `reward:` section kept as-is. Acceptance test PASS.

---

## Stage 2 — Shared refactors (behavior-neutral; handcrafted must stay green)

> Order matters: 2.1 first, so every later agent-level change is written once, not twice.

### 2.1 `[x]` Consolidate agents' sync/async duplication *(report non-port #1)*
- **What:** In each agent (`researcher.py` is the big one, plus `collector.py`, `lead_architect.py`, `data_analyst.py`, `reviewer.py`, `technical_writer.py`): collapse `_execute`/`_async_execute` into one async implementation; the sync variant becomes a thin wrapper (same pattern as `OllamaChat.gen` → `agen`). No behavior change.
- **Test:** guards `test_data_analyst_sync_async_equivalence`, `test_lead_architect_sync_async_equivalence_and_parse` (prove sync/async build identical prompts+outputs — must stay green through the merge), `test_parse_queries`, `test_collector_append_and_skip` (Collector path).
- **Done note:** 2026-07-02 — the sync wrapper now lives ONCE on the `Node` base class (`_execute` is concrete: runs `_async_execute` via asyncio.run / thread-pool when a loop is running, same pattern as `OllamaChat.gen`); all six agents lost their `_execute` duplicates, `Researcher._run_pbds_sync` deleted (subsumed). Guard helper `_captured_messages` made entry-point-agnostic (assertions unchanged). Guards 20/20 + all agent-touching suites green.

### 2.2 `[x]` Extract report finalization into a shared module *(report 1.2)*
- **What:** Move `_apply_citation_tags`, `_rewrite_inline_references`, `_strip_orphan_citation_markers`, `_build_bibliography`, `_format_bib_entry`/`_compose_reference`, `_generate_abstract` (+ their regex constants and `_tokenize`) out of `handcrafted_graph/graph.py` into `utils/report_finalize.py` (functions taking `report_state` and, where needed, an LLM). `HandcraftedGraph` delegates to it — same call sites, same behavior.
- **Also:** add a `finalize: bool = True` parameter to `HandcraftedGraph.arun()` gating abstract + bibliography assembly (citation tagging stays tied to SECTION_REVIEW, which training runs don't execute anyway). Default `True` → handcrafted unchanged; training passes `False` (needed by D6, used in Stage 4.6).
- **Test:** guards `test_citation_overlap_tagging`, `test_inline_reference_rewrite_and_orphans`, `test_bibliography_build` (re-point their imports at the new module in this stage's commit; assertions unchanged); acceptance `test_stage2_2_finalize_module_and_flag` (PEND→PASS; asserts the suggested function names + the `finalize=True` default).
- **Done note:** 2026-07-02 — `utils/report_finalize.py` created (apply_citation_tags / build_bibliography / generate_abstract + private inline-ref/orphan/tokenize helpers, code moved verbatim; log lines lose the `[graph-id]` prefix). `arun()` gained `finalize: bool = True` gating bibliography+abstract (assembly reordered harmlessly: content snapshot before build_bibliography, which never touches content). 3 citation call sites re-pointed; guard-test imports updated; acceptance test PASS.

### 2.3 `[x]` Extract trace helpers *(report 1.5)*
- **What:** Move `_init_trace_round` / `_trace_spatial_edges` into a small shared helper (e.g. `utils/trace.py` or a mixin) so the round-slot schema lives in one place. Add an optional per-agent `action` field populated when a controller supplies actions (Stage 4.2) — `None` for handcrafted runs, exactly like today, so `StandaloneVisualizer` keeps working on both.
- **Test:** guard `test_trace_round_schema` (pins the round-slot keys; additive `action` field keeps it green).
- **Done note:** 2026-07-02 — `utils/trace.py` with `init_trace_round`/`trace_spatial_edges`; graph keeps thin `_init_trace_round`/`_trace_spatial_edges` delegates so all call sites and tests are untouched. The per-agent `action` slot already exists in the schema (always None today); Stage 4.2 will populate it.

### 2.4 `[x]` (Optional, recommended) Split the validation loop out of `graph.py` *(report non-port #3)*
- **What:** Move `_decompose_validation_directive`, `_validation_windows`, `_build_section_windows`, `_revalidation_sections` into e.g. `handcrafted_graph/validation.py`. Pure code motion; shrinks `graph.py` toward its phase-execution core before Stage 3 modifies it.
- **Test:** guard `test_revalidation_sections_and_directive_extract` (re-point import on move); `_build_section_windows` already pinned by `test_no_trailing_singleton_window` in `test_review_pipeline_fixes.py`.
- **Done note:** 2026-07-02 — `handcrafted_graph/validation.py` (build_section_windows / revalidation_sections / validation_windows / decompose_validation_directive, the latter now taking the llm as a parameter). Graph keeps staticmethod ALIASES (`_build_section_windows = staticmethod(...)`) + thin delegates, so no test import needed re-pointing — both old and new suites green unchanged. `graph.py`: 1900 → 1257 lines across Stage 2.

### 2.5 `[x]` Regression checkpoint #1
- **What:** Re-run the Stage 0 protocol (tests + smoke run) and compare against the baseline: same phase sequence in the trace, same section count, bibliography/abstract present, no new warnings. Record findings here.
- **Test:** full guard suite must be 20/20 PASS (with the Stage-2 import re-points applied); acceptance suite must show 0 FAIL.
- **Done note:** 2026-07-02 — offline battery green (49 passed; guards 20/20; acceptance 3 PASS / 9 PEND / 0 FAIL). User smoke run (2026-07-02_164359, same task as baseline) vs baseline trace: 45 rounds both, identical per-round exec patterns, 6 sections both, abstract + bibliography present. Sole difference: PBDS absent from research rounds — environment, not regression (no workbook configured in the launch shell; baseline run had `QMIX_REPORT_PBDS_WORKBOOK` set; PBDS code untouched, its 19 tests green). Checkpoint PASSED.

---

## Stage 3 — The controller seam in `HandcraftedGraph` (D2)

### 3.1 `[x]` Define the `RoundController` interface
- **What:** New module (suggest `handcrafted_graph/controller.py` — it is part of the graph's contract, not QMIX-specific). Interface (async where it may call LLMs/networks):
  - `round_plan(phase, round_idx, topology, nodes, task_input) -> RoundPlan` where `RoundPlan = (active_agents: set[str], edges: list[(str, str)], actions: dict[str, int] | None)` — `actions` is what gets rendered into prompts and recorded in the trace (`None` = handcrafted mode).
  - `on_round_end(phase, round_idx) -> None` — hook for episode recording / reward events (no-op by default).
  - `on_phase_start(phase)` / `on_run_end()` — lifecycle hooks (no-op by default).
- **Scope note (document in the module docstring):** the controller is consulted **only** where `phase.round_topologies[...]` + `scheduler.get_active_agents(...)` are consulted today — the generic phase loop, the DRAFTING prep round, and the DRAFTING round-B *retry* branch. Scripted rounds (blueprint-reuse write, SECTION_REVIEW, VALIDATION, directive-bypass) never consult it.
- **Test:** acceptance `test_stage3_controller_seam` (PEND→PASS; asserts the interface hooks + the `controller=None` default on `HandcraftedGraph.__init__`).
- **Done note:** 2026-07-02 — `handcrafted_graph/controller.py`: `RoundPlan` dataclass (active_agents, edges, actions), `RoundController` base (async `round_plan`, `on_round_end`, `on_run_end`; sync `on_phase_start`), scope documented in the module docstring. Hook scope narrowed vs the sketch: `on_round_end` fires after every PLANNING/RESEARCH/DRAFTING round (incl. the scripted write round — that's where appends happen, which the reward hook needs); correction phases fire no hooks (never run in training, D6). Acceptance test PASS.

### 3.2 `[x]` `HandcraftedRoundController` (default) + wire the seam
- **What:** Implement the default controller wrapping today's logic (topology tables + `RoundScheduler` with the configured skip strategy). `HandcraftedGraph.__init__` gains `controller: RoundController | None = None` → defaults to `HandcraftedRoundController` built from the existing `skip_strategy` arg (public API unchanged). Replace the direct table/scheduler consultations in `_execute_phase` / `_execute_drafting_phase` with `controller.round_plan(...)`; add the `on_round_end` / `on_phase_start` calls at the existing round boundaries.
- **What (prompt path):** `_execute_round` forwards the plan's per-agent `action` (when not `None`) into `async_execute(..., action=...)` so the prompt context block can render it — handcrafted runs pass nothing and behave as today.
- **Test:** guards pinning what the default controller must reproduce: `test_phase_sequence_invariants`, `test_scheduler_temporal_heuristic`, `test_drafting_blueprint_usable`, `test_parse_section_titles` — all must stay green with the seam in place.
- **Done note:** 2026-07-02 — controller consulted at exactly the four planned spots (generic loop, drafting fallback loop, drafting prep, drafting round-B retry); blueprint-reuse write round marked SCRIPTED in code. `HandcraftedRoundController` holds one scheduler for the run (stateless → equivalent to the old per-phase construction). `_build_topology` split into `_build_topology_edges` (plan-driven) + a thin table wrapper (scripted paths unchanged). `_execute_round` gained `actions=` — forwards per-agent action into `async_execute` and records it in the trace slot; handcrafted plans carry `actions=None` so prompts/trace stay byte-identical. `on_run_end` fires at the end of `arun`. Guards 20/20.

### 3.3 `[x]` Training-mode run shape verification
- **What:** Verify (with a stub controller and a tiny/dummy LLM) that `HandcraftedGraph(phases=[PLANNING, RESEARCH, DRAFTING]).arun(..., max_validation_attempts=0, finalize=False)` cleanly skips the correction stages: no SECTION_REVIEW/VALIDATION rounds, no directive decomposition on empty reviewer output, no abstract/bibliography, no crash in the validation-loop bookkeeping (progress bars, trace stamps). Fix whatever edge cases surface — this is the exact configuration Stage 4.6 training uses (D6).
- **Test:** NEW `tests/test_training_mode_shape.py` — full offline `arun` (stub RAG via patch, scripted mock LLMs) in the exact D6 training configuration; asserts outline→2 sections written, Reviewer never called, no SECTION_REVIEW/VALIDATION in phase history, no bibliography/abstract. Add it to every future battery run.
- **Done note:** 2026-07-02 — passed first try, no edge cases surfaced: 12 rounds (2 planning + 6 research + 2×2 drafting), validation loop degrades cleanly to "finalising as-is" with an empty correction-phase list, trace stamping guards hold.

### 3.4 `[x]` Regression checkpoint #2
- **What:** Stage 0 protocol again on the default-controller path. The handcrafted pipeline with no arguments changed must produce an equivalent run (trace round sequence identical to checkpoint #1).
- **Done note:** 2026-07-02 — battery green post-seam (50 passed across 8 suites; guards 20/20; acceptance 4 PASS / 8 PEND / 0 FAIL). User smoke run (2026-07-02_172315): 55 rounds vs baseline 45 — expected LLM variance, not drift: this outline had 8 sections (vs 6) and the run exercised the validation-retry loop (decomposition + re-review markers present). Every round matches a legal phase topology; abstract + bibliography + 101 citation tags present. Checkpoint PASSED. Changes committed by user (e6755f0).

---

## Stage 4 — The QMIX decision layer

### 4.1 `[x]` Action space v2 + per-phase masks *(report 2.3, 2.4; D5)*
- **What:** Redefine in `qmix/agent_network.py`: `0 no_op` (agent does not execute this round — replaces `solo_process` semantics, *report 2.3*), `1 broadcast_all`, `2–5 selective_query(target 0–3)`, `6 aggregate_refine`, `7 append` → **NUM_ACTIONS = 8** (`terminate` dropped per D5). Note in code that "think privately" (old solo) can be re-added later if an ablation wants it.
- **What:** New `qmix/action_masks.py`: `mask(phase, agent_name, round_ctx) -> bool[8]`. Initial rules: self-targeting `selective_query` always masked *(absorbs report 0.7)*; `append` masked in PLANNING/RESEARCH (and per **OD-1 = (a)** also in DRAFTING); `append` always masked for the Reviewer; `no_op` masked for agents that are hard-required by the phase (e.g. Researcher+LeadArchitect in PLANNING round 0) so a random policy cannot produce a dead round. Keep the rules table in one visible place — they encode the handcrafted topology knowledge as constraints.
- **Test:** acceptance `test_stage4_1_action_space_v2` + `test_stage4_1_action_masks` (PEND→PASS; encode 8 actions/no terminate/`no_op` at 0, and the append-mask rules incl. OD-1a); guard `test_action_space_consistency` stays green across the 9→8 change.
- **Done note:** 2026-07-02 — `agent_network.py` action list v2 (NUM_ACTIONS=8, `no_op` at 0, terminate gone, "think privately" re-add noted in code). `qmix/action_masks.py`: `mask()` + `masks_for_round()`, all rules in one table — self-query always masked, append masked in P/R/D (OD-1a; flip `_APPEND_MASKED_PHASES` for OD-1b experiments) and always for Reviewer, `no_op` masked for topology-required agents. Both acceptance tests PASS.

### 4.2 `[x]` `QMIXRoundController`
- **What:** New `qmix/qmix_controller.py` implementing the Stage 3.1 interface: builds observations (v1 = existing 51-dim features + phase one-hot; v2 lands in Stage 5), calls masked action selection, translates actions → `RoundPlan` (`no_op` → agent excluded from `active_agents`; `broadcast`/`selective`/`aggregate` → edges among active agents; `append` → edge to Collector), records `(obs, actions, adj, global_state)` per round, sets `done` on run end. Holds the GRU hidden state across rounds; `train` flag switches ε-greedy vs greedy. Port `get_observation_features` / `get_global_state` / adjacency derivation from the deleted `QMIXGraph` (git) into this module or a small `qmix/observations.py`.
- **Edge-case policy (document):** cycle-denied edges fall back exactly like legacy (edge dropped); an all-`no_op` round advances the round counter with no execution (naturally discouraged by reward-per-token, and prevented in required rounds by masks).
- **Test:** acceptance `test_stage4_2_qmix_controller_module` (existence + hooks; behavior is covered by the 4.7 fidelity test and 4.8 dry run).
- **Done note:** 2026-07-02 — `qmix/observations.py` (legacy v1 features + 5-dim phase one-hot → obs_dim 56; state = N×56+3) and `qmix/qmix_controller.py` (`plan_from_actions` module fn + `QMIXRoundController`: masked selection via trainer, GRU hidden across rounds, per-step recording with mask stored, mask-violation RuntimeError guard, env invariant preserving Collector edges in the drafting retry round). Acceptance PASS.

### 4.3 `[x]` Masked action selection in the trainer (integration-only change)
- **What:** `QMIXTrainer.select_actions` accepts an optional `mask: (n_acting, n_actions)` bool tensor — masked entries set to `-inf` before argmax and excluded from the ε-random draw. Store masks in `EpisodeStep` so the (future, reworked) training step can apply them to target-max computations too. No other trainer changes.
- **Test:** acceptance `test_stage4_3_masked_select_actions` (PEND→PASS; verifies the mask binds at BOTH ε=0 and ε=1); guard `test_trainer_select_actions_shape` stays green (unmasked call unchanged).
- **Done note:** 2026-07-02 — `select_actions(..., mask=None)`: −inf on masked Q before argmax, ε-draw restricted to valid set, fully-masked row falls back defensively. `EpisodeStep` gained `mask` field (stored, not yet consumed by train_step — for the training rework). Acceptance PASS.

### 4.4 `[x]` Reward event hook *(absorbs report 0.8; evaluator internals untouched)*
- **What:** In `QMIXRoundController.on_round_end`: detect a **successful** append via `len(ReportState.additions)` delta (not via the chosen action) → update `Score` (`await report_score()`) and `LengthGoal` (`length_score(...)`), compute the delta reward (`compute_reward`), spread it evenly over the steps buffered since the last reward event (port of the old `step_buffer` logic from git, now event-correct); leftover steps flush with reward 0 at `on_run_end`. `NoCorpusCoverageError` → episode ends, buffered steps flush at reward 0 (simple v1; revisit with the reward rework).
- **Test:** guards pinning the contract this hook builds on: `test_collector_append_and_skip` (additions grow only on real prose — the exact signal the hook watches), `test_score_delta_semantics`, `test_absence_and_sentinel_pins`, and `length_score` sanity inside `test_node_import_and_eval_standalone`. Hook behavior itself: 4.8 dry run.
- **Done note:** 2026-07-02 — in `QMIXRoundController.on_round_end`: trigger = `len(ReportState.additions)` growth (never the action); scorer injected as `score_fn` (package does not import experiments/); length gaussian re-stated env-side from config params; reward via `trainer.compute_reward`, spread evenly over buffered steps; `on_run_end` idempotent 0-flush + done flag (also covers `NoCorpusCoverageError` aborts via the runner's finally). Verified by the 4.8 dry run (2 reward events on 2 real appends).

### 4.5 `[x]` Prompt merge *(report Tier 3)*
- **What:** `HandcraftedPromptSet.get_context_block` additionally renders the QMIX action line when `kwargs.get("action")` is not `None` (reuse `_QMIX_ACTION_DESCRIPTIONS` from `redacting_prompt_set.py`, updated for the v2 action list). Handcrafted runs pass no action → byte-identical prompts (verify in checkpoint 4.8).
- **Test:** acceptance `test_stage4_5_prompt_action_render` (PEND→PASS; action rendered, phase context preserved, no action line without an action); guard `test_context_block_action_rendering` stays green (redacting-set vocabulary + handcrafted no-action path).
- **Done note:** 2026-07-02 — `HandcraftedPromptSet.get_context_block` renders `**QMIX selected action:** <desc>` when an action kwarg arrives (no action → byte-identical block, guard-verified). `_QMIX_ACTION_DESCRIPTIONS` updated for v2 (no_op noted as never-rendering, terminate entry dropped). Acceptance PASS.

### 4.6 `[x]` Runners: `run_qmix_train` + `run_qmix` *(report 1.1; absorbs 0.1, 0.6)*
- **What:** New `qmix/runner.py` mirroring `handcrafted_graph/runner.py`, plus thin CLIs `experiments/run_qmix_train.py` (new file) and `experiments/run_qmix.py`:
  - **Shared per-run plumbing** (reuse/extend the handcrafted `_reset_singletons`, adding `SourceBuffer` — it already has it — and the Researcher's `_reported_chunk_ids` via fresh graph construction): trace saved in `finally`, roster read from `agent_configs` (correct key), config-first defaults (Stage 1.2).
  - **Train mode:** per episode — fresh graph + `QMIXRoundController(train=True)`, `phases=[PLANNING, RESEARCH, DRAFTING]`, `max_validation_attempts=0`, `finalize=False`; after `arun`: push recorded episode, `train_step()`, ε decay, checkpoint save; per-episode stats print (score, reward, tokens, ETA — port the old console format).
  - **Inference mode:** greedy controller, full phase list, `finalize=True`, then the full artifact path: `filter_meta_commentary` → `save_raw_report` → optional LaTeX/PDF (same flags as `run_handcrafted`: `--task`, `--llm`, `--trace`, `--no-pdf`, plus `--model-path`).
- **Test:** acceptance `test_stage4_6_runner_module` (PEND→PASS); guard `test_filter_meta_commentary` pins the inference filter. End-to-end behavior: 4.8 dry run.
- **Done note:** 2026-07-02 — `qmix/runner.py` (`run_qmix` inference: greedy, full phases, finalize on, filter→save→PDF artifact path; `run_qmix_train`: D6 shape, per-episode singleton reset incl. SourceBuffer + ExecutionTrace, `on_run_end` in finally, buffer push + train_step + ε decay + checkpointing; `build_trainer` reads `qmix.*` config with explicit float/int casts — YAML parses `5e-4` as a string). Tasks + scorer are INJECTED by the CLIs (`experiments/run_qmix_train.py` recreated, `experiments/run_qmix.py` new) so the package never imports repo-root `datasets`/`experiments`. QMIX traces go to `qmix_trace.json` (own file — `paths.trace_file` would otherwise force the handcrafted name). Both CLIs import clean.

### 4.7 `[x]` Translation-fidelity test (scripted controller)
- **What:** A test controller that emits, through the *QMIX action vocabulary*, the plans the handcrafted tables would produce for PLANNING/RESEARCH/DRAFTING (e.g. Researcher `selective_query(LA)` in planning round 0…). Run both pipelines on the same task with the same seed/model and compare traces: active sets and edges per round must match. This proves the action→plan translation and the seam are faithful before any training happens.
- **Test/Done note:** 2026-07-02 — NEW `tests/test_controller_fidelity.py` (offline, mocked LLM/RAG so both runs are deterministic): the vocabulary controller reproduces all 12 rounds of the handcrafted run with identical active sets and edges. Notably every handcrafted round pattern IS expressible in the 8-action vocabulary (aggregate_refine covers "active receiver", selective/broadcast cover the send patterns). PASS first try.

### 4.8 `[x]` Regression checkpoint #3 + random-policy dry run
- **What:** (a) Handcrafted default path still matches baseline. (b) `run_qmix_train` with an untrained network and small budgets (2 episodes, tiny model): no crashes, masks respected (assert in controller), episodes land in the buffer with sane shapes, reward events fire only on real appends, trace renders in the visualizer with actions shown.
- **Done note:** 2026-07-02 — offline halves done: (a) full battery green (52 passed / 1 designed live-skip across 11 suites; guards 20/20; acceptance 10 PASS / 2 PEND — only the Stage-6 items pend). (b) NEW `tests/test_qmix_dry_run.py`: full random-policy (ε=1.0) episode through the real controller+seam offline — 10 steps recorded with correct shapes incl. masks, mask compliance on every step, 2 reward events on the 2 real appends, replay push + train_step loss computed. **Live half done 2026-07-03:** user ran `run_qmix_train --num-episodes 2 --trace` on Qwen3-30B — both episodes completed (no crash), nonzero rewards (0.262, 0.178), 15–17 steps each. Trace verified: 14/15 rounds carry recorded actions across the masked vocabulary; mask compliance exact in the LIVE run (no self-query — each agent avoids its own target index; no append anywhere). Two benign notes: (1) `loss=n/a` / `Steps: 0` is expected — `train_step` gates on buffer ≥ batch_size(32), so 2 episodes never train; (2) a `safe_json_parse` "Failed to parse" on one judge reply is EVALUATOR-side (out of scope, slated for rework), handled gracefully (returns {} → chunk scored 0 → reward still computed). Checkpoint PASSED.

---

## Stage 5 — Observation upgrade *(report 2.7)*

### 5.1 `[x]` Structured state features
- **What:** Extend the obs builder (from 4.2) with: sections written, planned−written remaining, `len(content)/length_goal`, last append outcome (task sentinel is `SECTION_COMPLETE` vs `SECTION_SKIPPED` vs none), research-exhausted flag, per-agent last-output-is-sentinel flag, phase one-hot (moved from 4.2 if simpler to keep together). Recompute `obs_dim`/`state_dim` in one place used by both controller and trainer construction; old checkpoints are invalid (fine — none are, post-D3).
- **Test:** NEW `tests/test_observations.py` (offline): dims match the matrix, phase one-hot tracks PhaseState, the 6 structured scalars reflect ReportState, per-agent sentinel flag fires.
- **Done note:** 2026-07-03 — added 6 global scalars (`_structured_global_features`: written/10, remaining/10, capped length-ratio, append-complete, append-skipped, research-exhausted) shared by every agent + a per-agent last-output-is-sentinel flag. Phase one-hot was already present from 4.2 (kept). obs_dim 56 → 63; `get_obs_dim`/`get_state_dim` remain the single source, so the controller and `build_trainer` and every shape-test adapt automatically (dry-run/fidelity re-run green).

### 5.2 `[x]` Embedding features (flag-gated)
- **What:** Replace the byte-hash task/progress features with `nomic-embed-text` embeddings via the existing local Ollama embed path (see `rag_manager`'s direct `/api/embed` call): task embedded once per episode, progress summary re-embedded only after successful appends (cache), truncated/projected to a configurable dim (start 64). Config flag `qmix.obs.use_embeddings` so the training rework can A/B hash vs embeddings.
- **Test:** default `use_embeddings: false` asserted in `test_observations.py` (offline path stays deterministic; the embed branch is exercised only in a live run with the flag on).
- **Done note:** 2026-07-03 — `_embed_features` (direct `/api/embed` for `nomic-embed-text`, per-text cache, truncate/pad to slot width + L2-norm, **fails soft to byte-hash** on any transport error so obs-building never crashes). Config `qmix.obs.{use_embeddings=false, embedding_dim=16}`; `embedding_dim` sets EACH text slot width for both modes, so enabling embeddings at 16 keeps obs_dim identical (true A/B) and raising it grows obs_dim uniformly. Deviation from the sketch: chose `embedding_dim` default 16 (not 64) so the flag flip alone doesn't change obs_dim; the training rework can raise it.

---

## Stage 6 — Docs & final cleanup

### 6.1 `[x]` README rewrite *(report 0.9)*
- **What:** Update the QMIX sections: new architecture diagram (controller-in-handcrafted), v2 action table (8 actions, masks), phases-in-env, training/inference run modes; remove references to deleted scripts, the old checkpoints table (checkpoints are void post-D3/D5), and the outdated REVIEW/REVISION phase description (now SECTION_REVIEW/VALIDATION + retry loop). Link the report and this plan.
- **Done note:** 2026-07-03 — rewrote the top (two-pipelines/controller-seam framing + MMDP learner + evaluator-rework caveat), the action table → v2 (8 actions, no_op, mask rules, terminate removed), Quick Start → the three real CLIs with the Ollama prerequisite (dropped the nonexistent Gemini/GPT `scripts/*.sh` and the void bundled-checkpoints table), the package-structure blurbs for `graph`/`qmix`/`handcrafted_graph`, and the 5-phase table → PLANNING/RESEARCH/DRAFTING/SECTION_REVIEW/VALIDATION with the controller-scope note. Left the deep "technical changes" changelog as build-history provenance (it still uses the old REVIEW/REVISION narrative but reads as history, not current reference).

### 6.2 `[x]` Roster cleanup — TechnicalWriter *(report non-port #2; OD-2 = (a) delete)*
- **What:** Delete `agents/technical_writer.py`, its import in `agents/__init__.py`, and its `redacting` prompt entries (Collector absorbed the role). Note that roster changes require revisiting selective-query target count and masks.
- **Caveat found while writing tests:** the "Technical Writer" prompt entries are the *fallback* for unknown roles — `RedactingPromptSet.get_description/get_constraint` and `HandcraftedPromptSet.get_constraint` all default to `ROLE_*["Technical Writer"]`. Re-point those fallbacks (e.g. to the Collector entries or an explicit generic entry) BEFORE deleting the prompt entries.
- **Test:** acceptance `test_stage6_2_technical_writer_deleted` (PEND→PASS; file gone, agents package imports cleanly, registry has no `TechnicalWriter` key).
- **Done note:** 2026-07-03 — the caveat was real: re-pointed all three fallbacks (`redacting` get_description/get_constraint, handcrafted get_constraint) + `get_answer_prompt` default from "Technical Writer" to "Collector" (the role that absorbed writing) FIRST, then deleted both `Technical Writer` prompt entries, the roster-cycle line, `technical_writer.py`, its `__init__` import, and the visualizer icon entry. Verified: registry = 5 agents, unknown-role fallback returns the Collector template without raising, real roles unaffected. Acceptance PASS.

### 6.3 `[x]` Config finalization
- **What:** `qmix:` section reflects reality (n_actions 8, obs flags, training defaults actually read by the runner — closing Stage 1.2's placeholder); prune anything still dead.
- **Done note:** 2026-07-03 — mostly closed already by 1.2 (dead keys removed) + 4.6/5.2 (added `qmix.training.*`, `qmix.obs.*`). Verified end to end: no `n_actions`/`evaluation`/top-level `training` remnants; every key `build_trainer` and `observations` read is present; `build_trainer(5)` → n_actions 8, obs_dim 63. Action count stays code-defined (NUM_ACTIONS), not config. Left `pyproject.toml` deps (`datasets`, `huggingface-hub`, `jsonlines`) untouched — install-weight only, out of plan scope; flagged as a possible follow-up.

### 6.4 `[x]` `datasets/` pruning *(OD-3 = (a) delete)*
- **What:** Keep `datasets/tasks.py` (training task list — still used); delete the benchmark loaders (`gaia/hle/mmlu/humaneval/livecodebench/math/frontierscience/...`), `hf_loader.py`, the `DATASET_REGISTRY`/`get_dataset` machinery, and rewrite `datasets/__init__.py` to only expose `tasks` (keep `base_dataset.py` only if something still imports it).
- **Test:** acceptance `test_stage6_4_datasets_pruned` (PEND→PASS; loaders gone, `datasets.tasks` still imports, package init clean).
- **Done note:** 2026-07-03 — confirmed only `datasets.tasks` is imported externally (the 3 runners); deleted all 8 benchmark loaders + `base_dataset.py` + `hf_loader.py` (nothing imported them post-legacy-deletion), rewrote `datasets/__init__.py` to expose only `tasks`. Acceptance PASS; all 12 acceptance tests now green.

### 6.5 `[x]` Memory/report sync
- **What:** Update `qmix_improvement_report.md` status header (plan executed) and the assistant memory files if architecture facts changed during implementation.
- **Done note:** 2026-07-03 — plan complete: 27/27 items. Final offline battery green (68 tests / 12 suites; guards 20/20; acceptance 12/12 PASS, 0 pending). Added status banner to `qmix_improvement_report.md`; memory `project_qmix_pipeline.md` already rewritten to the v2 architecture (2026-07-02, refreshed here). Remaining beyond this plan: the training-algorithm + evaluator rework (always out of scope), and optional follow-ups noted in items (pyproject dep prune; embedding_dim raise; OD-1b policy-triggered append experiment).

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
| 0 — Guardrails | 1 | 1 |
| 1 — Legacy removal | 2 | 2 |
| 2 — Shared refactors | 5 | 5 |
| 3 — Controller seam | 4 | 4 |
| 4 — QMIX decision layer | 8 | 8 |
| 5 — Observations | 2 | 2 |
| 6 — Docs & cleanup | 5 | 5 |
| **Total** | **27** | **27 ✅** |

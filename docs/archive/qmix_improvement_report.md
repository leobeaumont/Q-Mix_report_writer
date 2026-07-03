# Porting handcrafted-pipeline improvements into the QMIX pipeline

> **Status (2026-07-03): fully executed.** This report is the *why*; the *what/when/done* tracker is `qmix_upgrade_plan.md` (27/27 items complete). The QMIX pipeline is now the handcrafted phase pipeline with a `QMIXRoundController` in the round-decision seam; the legacy free-form `QMIXGraph` path was deleted. The training algorithm and report evaluator remain out of scope (separate rework).

**Date:** 2026-07-02
**Scope:** Everything the handcrafted pipeline (`qmix_report_writer/handcrafted_graph/`) does better than the QMIX prototype (`qmix_report_writer/graph/` + `qmix/` + `experiments/run_qmix_train.py`), organized as an implementation plan.
**Out of scope (per project direction):** the internals of the training algorithm (`qmix_trainer.py`, replay buffer, network architectures) and the internals of the report evaluator (`experiments/eval.py` judges) — both will be fully reworked. However, the *integration points* between the pipeline and those two components (when the reward fires, what the observations contain) are pipeline-side and are covered here.

---

## 0. The core lesson to carry over

The handcrafted pipeline's quality does not come from the hand-designed topology itself. It comes from a division of labor that the QMIX prototype lacks:

> **Deterministic code owns all scaffolding — target selection, sequencing, state carriage, signal detection, output routing. The LLM only fills content. Code never trusts the LLM to follow protocol: every sentinel is verified, every destructive action needs provenance, every target is resolved by the loop rather than parsed from prose.**

Concrete examples in the handcrafted graph: section targeting comes from `ReportState.review_section_idx` set by the loop (never parsed from LLM output); `[REMOVE_SECTION]` is refused without a matching removal request in the critique; `[NO_REVISION_NEEDED]` is ignored when the same response also contains numbered corrections; drafting iterates `planned_sections` in code instead of trusting the LeadArchitect to remember where it is.

The QMIX pipeline should keep QMIX as the *decision-maker for communication topology* — that is the research question — but inherit this scaffolding as its *environment*. A learned policy inside a sloppy environment will learn to cope with the sloppiness; a learned policy inside the handcrafted environment's scaffolding learns the thing you actually want to optimize.

Everything below follows from that principle. Tiers are ordered by (value ÷ effort) and by dependency.

---

## Tier 0 — Correctness fixes in the existing QMIX path (small, do first)

These are verified bugs, independent of any redesign.

| # | Bug | Where | Fix |
|---|-----|-------|-----|
| 0.1 | Config key typo: reads `agent_config`, key is `agent_configs` — roster silently falls back to the hardcoded default | `experiments/run_qmix_train.py` (`train()`) | Fix the key; ideally share the roster constant with `handcrafted_graph/runner.py::DEFAULT_AGENTS` |
| 0.2 | `run_qmix_eval.py` is dead code: imports `DOMAIN_MAP`/`AGENT_CONFIGS` (no longer defined in `run_qmix_train`) and `experiments.accuracy` (doesn't exist); passes `domain=`/`decision_method=` kwargs `QMIXGraph` doesn't accept | `experiments/run_qmix_eval.py` | Rewrite from scratch mirroring `run_handcrafted` (see Tier 1.1); greedy actions (`epsilon=0`) + the same output/export path |
| 0.3 | Trace crash: `round_template` calls `actions.tolist()`; with `execution_trace=True` and `actions=None` (fixed-topology baseline) this raises | `graph/graph.py::arun` | Guard for `actions is None` |
| 0.4 | Collector identified by hardcoded index (`if node_index != 4`) | `graph/graph.py::_execute_round` | Compare against `self.collector_id` |
| 0.5 | Stale/dead config: `qmix.n_actions: 16` (real action space is 9), `training.*` / `evaluation.*` never read (training script is argparse-only) | `configs/default.yaml`, `run_qmix_train.py` | Either wire the config sections into the script's defaults or delete them; one source of truth |
| 0.6 | `SourceBuffer` never reset between episodes — sources retrieved but not appended in episode N leak into episode N+1's first section (wrong citations) | `run_qmix_train.py::run_episode` | Reset it alongside ReportState/Score/LengthGoal; better: share a `_reset_singletons()` helper with the handcrafted runner |
| 0.7 | Self-query wasted action: `selective_query` target = `action − 2` ∈ {0..3} includes the agent itself → self-edge no-op | `graph/graph.py::apply_qmix_actions` | Mask or remap self-targeting (cheapest: treat as solo). Interacts with action masking in Tier 2.4 |
| 0.8 | Reward fires on *attempted* append, not *successful* append: `(actions == 7).any()` triggers scoring even when the Collector refused to write (absence-marker gate, sentinel output, empty response). Two judge calls are wasted and the Micro running average absorbs a duplicate evaluation of the unchanged last chunk, skewing `Score` deltas | `run_qmix_train.py::run_episode` | Trigger on actual report change: snapshot `len(ReportState.instance().additions)` (or `content` hash) before/after the round and score only when it changed. Keep this even after the evaluator rework — it's the trigger condition, not the judge |
| 0.9 | README drift: action table says 8 actions / "2–5: one per agent" (there are 9 actions and 4 selective targets incl. self); handcrafted section describes the old REVIEW/REVISION phases rather than SECTION_REVIEW/VALIDATION + retry loop | `README.md` | Update once Tier 2 decisions are made, so it's only rewritten once |

---

## Tier 1 — Drop-in ports (infrastructure already shared; just wire it in)

The agents, ReportState, LLM layer, RAG, PBDS, report export utils are all shared code. The handcrafted pipeline uses them through a runner and finalization layer the QMIX path simply never got.

### 1.1 A real `run_qmix` runner (mirror of `handcrafted_graph/runner.py`)
`run_qmix_train.py` prints previews and saves a checkpoint; there is no report artifact. Port from `run_handcrafted`:
- `_reset_singletons()` before each episode (fixes 0.6 as a side effect).
- Trace saved in a `finally` block (`get_trace_path(...)`) so crashed episodes keep their trace — in the handcrafted pipeline this proved to be *the* debugging artifact.
- `filter_meta_commentary()` applied to the final answer (`utils/report_filter.py` is currently handcrafted-only at runtime).
- `save_raw_report()` + optional LaTeX→PDF export per episode (or at least per eval run) so QMIX outputs are inspectable the same way handcrafted outputs are.
- Same CLI surface (`--task`, `--llm`, `--trace`, `--no-pdf`) so the two pipelines stay comparable under one harness.

### 1.2 Report finalization as a shared module
Citation tagging (`_apply_citation_tags`), inline-reference rewriting, orphan-marker stripping, bibliography building (`_build_bibliography`, cited vs consulted, arXiv metadata), and abstract generation (`_generate_abstract`) are all methods on `HandcraftedGraph` today, but none of them depend on phases — they only need `ReportState.sections` (which already carries per-section `sources` in both pipelines, because the Collector stores them on append).

**Action:** extract into e.g. `utils/report_finalize.py` (functions or a `ReportFinalizer` taking `report_state` + an LLM), call from both graphs at the end of `arun`. The QMIX pipeline gets citations, bibliography, and an abstract for free. This also shrinks `handcrafted_graph/graph.py` (~1900 lines) — a maintainability win on its own.

### 1.3 Phase-neutral agent behavior (fix the hidden coupling)
Shared agents call `PhaseState.instance().current_phase`, and `PhaseState` **defaults to PLANNING**. A QMIX run never sets phases, so today:
- the Researcher always uses `top_k_planning = 3` (evidence-starved drafting),
- `_persist_deficiencies` runs on every response (deficiency list grows unbounded, and the LeadArchitect block that consumes it is planning-gated so it half-fires),
- PBDS always renders its PLANNING format and applies the frontier,
- the RESEARCH-only duplicate-chunk filter never fires (same chunks re-retrieved and re-synthesized round after round — a known failure mode the handcrafted pipeline explicitly fixed).

**Action:** whichever Tier 2 direction is chosen, make the ambient state explicit now: the QMIX loop sets `PhaseState` and the behaviors activate correctly by construction.

### 1.4 Episode-level memory & retrieval hygiene
- `_clear_all_memory()` (the handcrafted version that clears `last_memory` *and* live `outputs`/`inputs`) should be available on `QMIXGraph` and called at episode start. Today a fresh graph per episode masks this, but any future graph reuse (or mid-episode boundary, see Tier 2.2) needs it.
- Enable `Researcher._filter_reported_chunks` per episode in QMIX mode (see 1.3) and reset `_reported_chunk_ids` at episode start (currently reset only implicitly by constructing a new graph).

### 1.5 Trace parity
`QMIXGraph`'s trace template lacks the handcrafted extras: `PBDS` slot, `RAG.sources` (with retrieval scores), per-agent `completion_tokens`. The Researcher writes some of these keys defensively (`setdefault`), so nothing crashes, but the visualizer shows less for QMIX runs. Port `_init_trace_round()`/`_trace_spatial_edges()` (they are graph-agnostic) — ideally into a small shared trace helper used by both graphs.

---

## Tier 2 — Structural adoptions (the big wins; these define the training environment)

These change what the QMIX policy is asked to learn. They are the reason to do this port *before* the first serious training run: they shape the MDP.

### 2.1 Bring the report lifecycle (phases) into the QMIX environment — recommended
The single largest quality lever in the handcrafted pipeline is phase specialization: per-(phase, role) objectives, phase-appropriate retrieval depth, phase-gated tool behavior. None of that requires topology to be hand-designed.

**Recommended shape:** the environment (episode loop) drives `PhaseState` exactly like `HandcraftedGraph` does — PLANNING until an outline exists, DRAFTING until planned sections are exhausted, etc. — while **QMIX chooses the communication actions within each phase**, replacing the hand-written `RoundTopology` tables. That is precisely the component worth learning (who talks to whom, when to query vs broadcast vs append), while keeping the proven scaffolding:
- outline extraction + `planned_sections` (code-parsed, `NoCorpusCoverageError` fail-fast on empty corpus),
- task sentinels (`[SECTION_COMPLETE…]` / `[SECTION_SKIPPED…]`) advancing the section target deterministically,
- `HandcraftedPromptSet` phase objectives (see Tier 3),
- phase-tailored RAG `top_k`, chunk dedup, PBDS behavior — all already phase-keyed in shared agents.

Add the phase as a one-hot to the observation vector so the policy can condition on it. Episode termination becomes "phases complete" rather than only the majority-terminate vote; the `terminate` action can be kept within phases as an early-exit vote or dropped (see 2.6).

### 2.2 Section-structured episodes
Even independent of full phases, port the drafting skeleton: parse an outline once (or take it from a fixed PLANNING prologue), then let the env set `ReportState.task = "[NEXT SECTION TO WRITE: <title>]"` and advance on the Collector's task sentinels. Benefits:
- Credit assignment: the reward already fires per append; aligning "one section = one reward event = one buffered step-group" makes the even reward split across buffered steps meaningful instead of arbitrary.
- The memory boundary between sections (`_clear_all_memory`) prevents the cross-section contamination the handcrafted pipeline documented (stale `<task>` tags biasing later sections) — same failure will occur in QMIX episodes, just undiagnosed.
- The temporal-edge policy can copy handcrafted's rule: no temporal self-edges while drafting section content (`_connect_temporal` skip), temporal edges elsewhere.

### 2.3 True no-op semantics for `solo_process` (participation = the skip lever)
In `QMIXGraph._execute_round`, **every agent executes an LLM call every round** regardless of action; `solo_process` output usually goes nowhere. Worse, the Researcher's execution always runs query-formulation + RAG retrieval + synthesis (+ PBDS verifier), so a Researcher on `solo_process` still burns 2–3 LLM calls and a retrieval per round.

**Action:** make action 0 skip execution entirely (node keeps `last_memory`, `has_output=0` in observations). This is the QMIX-side equivalent of the handcrafted `RoundScheduler` skip strategies, but *learned* — which is exactly the token-efficiency lever the reward's token term is supposed to exercise. It also removes pure noise from Q-learning: today action 0 and action 1 cost nearly the same tokens and differ only in edges, so the token-cost gradient between "participate" and "don't" doesn't exist.

### 2.4 Per-phase / per-role action masking
Standard MARL practice, cheap to implement in `select_actions` (mask invalid actions to −inf before argmax, and restrict the ε-random draw to the valid set). Masks encode the handcrafted topology knowledge as *constraints* instead of *fixed choices*:
- PLANNING/RESEARCH: mask `append` (nothing should reach the report yet) — the handcrafted prompt currently enforces this only with words.
- Only the Collector writes prose in the handcrafted design; in QMIX, `append` routes an agent's output to the Collector, which is fine — but consider masking `append` for the Reviewer (its critiques must never become report text; today only prompt discipline prevents it).
- Mask self-targeting selective queries (bug 0.7).
- Optionally mask `terminate` until ≥1 section exists.

Masks also solve the "action space grows with roster" wart cleanly: with masks, invalid target indices are simply never selectable, so roster changes don't corrupt learned action semantics.

### 2.5 Post-draft review as environment post-processing
The entire SECTION_REVIEW + VALIDATION machinery (per-section audit with stored source chunks, directive decomposition, re-validation compliance mode, removal authorization) is code-driven and does not need QMIX decisions. Port it as an optional post-process stage callable on any finished report — i.e., extract `_execute_section_aware_phase` / `_execute_window_aware_phase` dependencies enough that a QMIX run can invoke them after its drafting episode.

**Training-time recommendation:** run review/validation only at *evaluation* time, not inside the training reward loop. Reasons: (a) it multiplies per-episode cost by ~2–3× on a pipeline that is already the bottleneck; (b) the correction pass would launder the policy's mistakes before scoring, diluting the credit signal for the drafting decisions being trained. Score raw drafts during training; ship corrected reports at inference. Revisit when the evaluator is reworked.

### 2.6 Termination semantics
Handcrafted ends when the planned work is done; QMIX ends on majority `terminate` vote or `max_rounds`. It is recommanded to drop `terminate` action and stop the report the same way Handcrafted does it.

### 2.7 Observation upgrades from the now-rich ReportState
(Env-side, so in scope despite the training rework.) Current per-agent features: two 16-dim byte-hashes (task, progress summary — only the first 64 bytes!), one-hot id, 3 scalars. Replace/extend with signals that now exist:
- scalars: `len(sections)`, `len(planned_sections) − written`, `len(content)/length_goal`, last-round append outcome (task sentinel is `SECTION_COMPLETE` vs `SECTION_SKIPPED`), research-exhausted flag, per-agent last-output-is-sentinel flag;
- phase one-hot (with 2.1);
- replace the byte-hashes with real embeddings: `nomic-embed-text` already serves the RAG locally — embed the task once per episode and the progress summary once per append (cache; PCA/truncate to keep obs_dim reasonable).

---

## Tier 3 — Prompt-set port

- The QMIX path uses `RedactingPromptSet` (role descriptions + constraints + a one-line "QMIX selected action" block). The handcrafted `HandcraftedPromptSet` adds the phase context and per-(phase, role) objectives — the most heavily engineered artifacts in the project (sentinel protocols, mutually exclusive decision branches, anti-hallucination and anti-meta-commentary rules refined over many traces).
- With Tier 2.1, reuse `HandcraftedPromptSet` almost as-is: make its `get_context_block` *also* render the QMIX action description when `kwargs["action"]` is present (one `if` + reuse of `_QMIX_ACTION_DESCRIPTIONS`). Inject it in `QMIXGraph._init_nodes` the same way `HandcraftedGraph._inject_prompt_set` does.

---

## Non-port observations (worth fixing while you're in there)

1. **Sync/async duplication:** every agent implements `_execute` and `_async_execute` with byte-identical logic (~200 duplicated lines in `researcher.py` alone; the Collector pair already drifted once). Consolidate (sync delegates to async, or a shared `_prepare/_finalize` pair) *before* porting Tier 2 — otherwise every change lands twice.
2. **TechnicalWriter** is registered but rosterless (the Collector absorbed its role). Delete it or consciously re-add it; if the roster ever changes, revisit `n_actions` (selective-query count), the trace template, and the masks (2.4).
3. **`handcrafted_graph/graph.py` is ~1900 lines.** The Tier 1.2 extraction (finalization) plus a possible `validation.py` split would leave the phase-execution core readable — worth doing as part of the port rather than after.

---

## Recommended implementation order

1. **Tier 0** (0.1–0.8) — one sitting, unblocks trustworthy experiments.
2. **1.1 runner + 1.3 phase-neutral defaults** — QMIX runs become comparable to handcrafted runs under the same harness.
3. **1.2 finalization extraction** (+ 1.5 trace parity) — QMIX outputs become real reports; handcrafted graph shrinks.
4. **2.1 phases-in-env + 2.2 section-structured episodes + Tier 3 prompt merge** — the environment now matches the quality reference.
5. **2.3 no-op solo + 2.4 action masking + 2.6 termination guard** — the action space becomes learnable and token-cost-sensitive.
6. **2.7 observations** — last pipeline-side step before the training/evaluator rework.
7. **2.5 review post-process at inference** — ship-quality outputs from QMIX runs.

After step 7 the two pipelines differ in exactly one component — who chooses the communication topology each round — which is the comparison the project set out to make.

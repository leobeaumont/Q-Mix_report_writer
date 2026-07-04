# Training & evaluator rework — implementation plan & change tracker

**Created:** 2026-07-03 · **Companion:** [docs/training_evaluator_improvement_report.md](docs/training_evaluator_improvement_report.md) (the *why*; this file is the *what, in which order, and what has been done*).

## How to use this file

- Work the stages **top to bottom**; items inside a stage are ordered too. Dependencies are noted where they cross stages.
- Status boxes: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked / waiting on a decision.
- When an item is finished: tick it and fill its **Done note** with 1–3 lines (what was actually done, deviations, commit hash if useful).
- Report references like *(report A1.2)* point to the numbered items in the improvement report.
- **Side-validation tests** (same mechanism as the previous plan, both standalone with `.venv\Scripts\python.exe`):
  - `tests/test_training_eval_guards.py` — **regression pins**, green on pre-change code, must STAY green after every stage (except the two scheduled retirements, see below).
  - `tests/test_training_eval_acceptance.py` — **acceptance tests** encoding each item's target state; `PEND` until the stage lands, then `PASS` (or `FAIL` if the landed API violates the contract).
  - Items below carry a **Test:** line naming their tests. **LIVE** items need Ollama and are run by the user.

## Decision log (settled 2026-07-03)

| # | Decision | Choice |
|---|----------|--------|
| TD1 | Reward composition | **Per-chunk + terminal macro**: event reward = the appended chunk's own grounded micro score (absolute, no running average) + length shaping; macro judge runs ONCE at run end as terminal reward. |
| TD2 | Token-cost term | **Flag-gated, default off** (`reward.token_weight: 0.0`); implemented, disabled. |
| TD3 | Judge grounding | **Sources + claim-level check**: micro judge sees the section's stored RAG sources; a second call verdicts each claim (supported / unsupported / contradicted) → `grounding_ratio`. |
| TD4 | Trainer scope | **Fixes + Double-DQN**; no n-step / PER / architecture changes. |
| OD-A | Event-reward placement | **(a)** On the event step only (the most recent recorded step when the append fires); earlier buffered steps get 0; TD bootstrapping propagates credit. |
| OD-B | GNN aggregation direction | **(a)** Aggregate from in-edges + self-loops (message passing follows "who talks to me"). |
| OD-C | Replay-buffer persistence | **(a)** Persist alongside checkpoints; `--resume` reloads it. |
| OD-D | Judge prompt location | **(a)** Judge prompts/schemas move into the evaluation module; `redacting_prompt_set.py` loses its two scoring entries. |

**Standing constraint (unchanged):** the handcrafted pipeline keeps identical default behavior and public API; the controller-seam environment semantics (reward trigger = real append, plan translation, mask storage) must not drift.

**Relaxed constraint (user, 2026-07-03):** the scorer (`experiments/eval.py`, `Score`/`LengthGoal` machinery) and trainer internals (`qmix_trainer.py`, `replay_buffer.py`) may be **replaced outright** — no compatibility shims, no behavior preservation. Two old guards pin behavior that is deliberately replaced; they are retired/updated in the same commit as the change that invalidates them:
  - `test_upgrade_plan_guards.py::test_score_delta_semantics` — pins `Score`/`LengthGoal` delta semantics → retired in Stage 2.5.
  - `test_upgrade_plan_guards.py::test_node_import_and_eval_standalone` — imports `experiments.eval` → re-pointed in Stage 2.6.
  All other existing suites (guards 20 − 2, acceptance 12, dry-run, fidelity, observations, training-shape, review/PBDS) stay green throughout.

---

## Target architecture (orientation)

```
                     ┌────────────────────────────────────────────┐
 run_qmix_train ───► │ episode loop (qmix/runner.py)              │
 (CLI injects        │  seed · ε · train_steps_per_episode        │
  evaluator)         │  eval episodes · JSONL log · ckpt v2 + buf │
                     └───────┬────────────────────────┬───────────┘
                             │                        │
              QMIXRoundController          QMIXTrainer (pure TD)
              reward events (OD-A):        masked Double-DQN targets,
              append → evaluator.score_chunk   terminal-safe padding,
              run end → evaluator.score_report vectorized train_step
                             │                        ▲
                             ▼                        │ EpisodeBatch (+avail_actions)
              qmix_report_writer/evaluation/          │
              judges.py  (grounded micro + claim      │
                          check + terminal macro)     │
              reward.py  (compose_event_reward,   ReplayBuffer
                          compose_terminal_reward,
                          length_gaussian, weights)
```

- The evaluator is an **object with two async methods** (`score_chunk`, `score_report`) returning dataclasses, injected by the CLI (seam preserved); `trainer.compute_reward` is deleted.
- Score components flow into the JSONL log, so the first real training run is debuggable per term.

---

## Stage 0 — Guardrails & benchmark baseline

### 0.1 `[x]` Plan test suites + battery baseline
- **What:** Create `tests/test_training_eval_guards.py` (pins: reward-event fires only on real append growth; `on_run_end` idempotent + done flag on last step; `plan_from_actions` invariants (no_op exclusion, self-target drop, Collector invariant); `EpisodeStep.mask` recorded with correct shape; length-gaussian values (pins the math through its Stage-2 move); trainer checkpoint save/load round-trip) and `tests/test_training_eval_acceptance.py` (PEND/PASS mechanism copied from `test_upgrade_plan_acceptance.py`; one test per item below that carries an acceptance name). Run the full existing battery + both new suites; record results here as the baseline.
- **Test:** guards all PASS on pre-change code; acceptance all PEND / 0 FAIL.
- **Done note:** 2026-07-03 — both suites created: 6 guards (all PASS pre-change), 22 acceptance (all PEND, each with a precise "what's missing" message, 0 FAIL). Stage-3 target checks are loss-comparison based (immune to the 3.2 pad slot and 3.5 vectorization). Full existing battery green: 68 tests / 12 suites (guards 20/20, old acceptance 12/12) — identical to the previous plan's final state.

### 0.2 `[x]` Scorer benchmark harness *(report A2.2)*
- **What:** `experiments/scorer_benchmark.py`: chunk a document (reuse `test_scorer.py`'s PDF extraction), score it incrementally through a pluggable async **scorer adapter** (legacy `report_score` adapter now; evaluator-v2 adapter in Stage 2), and compute: **ranking accuracy** (fraction of ordered same-paper version pairs where the newer version scores higher, over `tests/test_documents/`), **repeat variance** (score the same doc k times → per-metric σ), **corruption probes** (off-topic section injected / sections shuffled / numbers perturbed → score must drop vs the clean run). Emits one JSON + one markdown summary per run into `tests/scoring_results/`.
- **Test:** acceptance `test_stage0_2_benchmark_harness` (module + metric functions exist; ranking-accuracy/variance math verified on synthetic score fixtures, offline).
- **Done note:** 2026-07-03 — `experiments/scorer_benchmark.py`: three CLI modes (`rank` / `repeat --doc X -k 5` / `corrupt --doc X`), adapter-pluggable (`--adapter legacy|v2`; legacy = frozen test_scorer protocol over `experiments.eval`, imported lazily so the module survives Stage 2.6). Family discovery verified: ACCADA v012→final and Towards v01→v22 grouped and ordered correctly. Ranking accuracy counts ties as half-wins over ALL ordered same-family pairs; corruption probes are deterministic. Singletons reset between scored documents (the leak test_scorer.py had). Acceptance PASS.

### 0.3 `[x]` **LIVE** Baseline benchmark run (current judges)
- **What:** Run the harness on the current `report_score` over all documents in `tests/test_documents/` (+ repeat-variance on one doc, k≈5). Archive results (e.g. `tests/baseline_refs/<date>-scorer/`). This is the before-picture every Part-A change is compared against.
- **Test:** protocol item — results recorded here.
- **Done note:** 2026-07-04 — user ran `rank` + `repeat` (results: `tests/scoring_results/benchmark_{rank,repeat}_legacy_20260704_*`). **Baseline: ranking accuracy 0.35** (20 pairs — BELOW coin-flip; ACCADA_final, the best version, scores 0.013 vs ~0.14 for its drafts; Towards v22 0.139 < v01 0.154), **repeat σ = 0.0127** on a ~0.145 signal (~9 % relative noise). Bimodal doc scores (~0.14 vs ~0.02) are the A0.3 signature: parse failures recording zeros into the running micro average. Forensics: the May-2026 runs that scored 0.69–0.86 used qwen3:8b and pre-"Strictness Clause" prompts (clause added 2026-05-11, judge model → Qwen3-30B mid-June), so the collapse is a real property of TODAY's scorer, not a harness artifact. Runtime (~hours) confirmed expected: macro judge re-reads the whole growing report per chunk (O(n²) tokens — killed by TD1). Harness amended to store per-chunk scores in rank results for future diagnosis.

---

## Stage 1 — Judge de-noising (report A-Tier 0, applied to the current scorer)

> These fixes transfer into the Stage-2 module (prompts/schemas move, discipline stays). Doing them first separates "noise reduction" from "redesign" in the benchmark attribution.

### 1.1 `[x]` Reason-first schemas + prompt/schema alignment *(report A0.1 partial, A0.5)*
- **What:** In the scoring prompts/schemas: reasoning field FIRST in both schemas; rubric names aligned with schema keys; `redundancy_penalty` renamed (e.g. `redundancy_avoidance`); "AI training data" framing removed; the **task added to the macro judge's user prompt** (from `ReportState.instance().task` is wrong — pass the episode task explicitly) and `subject_coverage` scored against it.
- **Test:** acceptance `test_stage1_1_schema_reason_first` (reasoning key first in `properties` order; no `redundancy_penalty` key; macro user prompt contains a task block when task passed).
- **Done note:** 2026-07-04 — both schemas reordered reasoning-first with reason-first noted in the prompts' Instructions; rubric bullets now use the literal schema keys; `redundancy_avoidance` renamed; "AI training data" framing dropped; `report_score(task=None)` renders an optional `<subject>` block before the report. Strictness Clause deliberately kept as-is (Stage 1 = de-noising, not re-calibration — attribution stays clean for the 1.4 re-run). Acceptance PASS.

### 1.2 `[x]` Judge call discipline *(report A0.4, A1.4)*
- **What:** All judge calls pass `temperature=0` explicitly (via `agen(..., temperature=0.0)`); new config `reward.judge.model` (default = `llm.default_model`) + `reward.judge.max_tokens` read by the scorer; judge LLM resolved once, independent of the actors' `--llm` flag (now explicit and documented).
- **Test:** acceptance `test_stage1_2_judge_discipline` (capture-stub LLM: temperature 0 received; judge model taken from `reward.judge.model` when set).
- **Done note:** 2026-07-04 — `reward.judge.{model, temperature, max_tokens, retries}` added to config; `experiments/eval.py` gained `_judge_cfg`/`_judge_llm`/`_judge_call` — every judge call passes temperature (default 0.0) and max_tokens (2048) explicitly. Acceptance PASS.

### 1.3 `[x]` Parse-failure policy *(report A0.3)*
- **What:** On judge JSON parse failure: retry once; on second failure the scoring call reports **failure** (exception or `None`) instead of zeros — the controller then *skips the reward event* (steps stay buffered for the next event; warning logged + counted). No score of 0 is ever recorded from a transport/parse problem.
- **Test:** acceptance `test_stage1_3_parse_failure_skips` (stub judge returning garbage twice → no event, buffer intact; garbage once then valid → event fires).
- **Done note:** 2026-07-04 — `_judge_call` validates all required schema keys per reply, retries (`reward.judge.retries`, default 1), then raises `JudgeError`; `QMIXRoundController._reward_event` wraps the scorer call — any judge exception increments `self.judge_failures`, logs, and returns with the buffer intact (Score/LengthGoal untouched). The benchmark adapter also forwards `task` when given (corpus PDFs keep `None` — a family stem is not a subject). Acceptance PASS; controller dry-run + full battery green.
- **Amendment (2026-07-04, after a live 1.4 crash):** the first live run crashed on a judge reply whose `local_audit_notes` was a JSON *array* — proof that Ollama does not reliably enforce the response schema. `_judge_call` now runs `_conform_to_schema` on every reply: list→joined string, numeric-string→int, ints clamped to schema min/max, string-booleans parsed; non-conformable values → retry → `JudgeError`. History assembly additionally `str()`-guards. Acceptance test extended with conformance probes (coercion on first call, exactly one retry before raising); live crash scenario reproduced offline and confirmed fixed.
- **Amendment #2 (2026-07-05, after the second live 1.4 attempt):** two more hardening fixes. (1) **Retries now resample** — attempt 0 at the configured temperature (0), later attempts at ≥0.35: a temp-0 re-ask deterministically reproduced the same malformed reply, making the retry useless (live-observed). `retries` default → 2, `max_tokens` → 3072 (reason-first replies truncated mid-notes lose their scores). Prompts now demand the notes be ONE short single-string paragraph (never a list/markdown) — the live replies included arrays, raw markdown, and XML-ish tags, confirming `/v1/chat/completions` `json_schema` is ignored on the user's Ollama. (2) **Benchmark adapter resilience** — a failed chunk logs, records `None`, and continues (all-failed aborts loudly); `judge_failures` counted per doc in results. Acceptance probes updated (config-driven retry count, resample floor, adapter survival). *Stage-2 note:* the evaluator should consider Ollama's NATIVE structured outputs (`format` on `/api/chat`) for real constrained decoding instead of the unenforced OpenAI-compat path.
- **Partial 1.4 evidence (crashed run, 2026-07-05):** completed docs already show the de-noising working — ACCADA v012 0.4725 / v019 0.5365 / v026 0.5242 (baseline ~0.14, and v012<v019 now correctly ordered); repeat runs 0.577/0.548/0.595/0.592 → σ≈0.018 on a ~0.58 signal (~3 % relative noise vs ~9 % at baseline).

### 1.4 `[ ]` **LIVE** Benchmark re-run #1
- **What:** Re-run 0.3's protocol on the de-noised judges. Expectation: repeat-variance collapses (temp 0), ranking accuracy ≥ baseline. Record deltas here.
- **Done note:** —

---

## Stage 2 — Evaluation module & reward v2 (TD1/TD2/TD3, OD-A, OD-D)

### 2.1 `[ ]` Package module `qmix_report_writer/evaluation/` *(report A2.1)*
- **What:** New package: `judges.py` (prompts + schemas move here per OD-D — `redacting_prompt_set.py` loses "Macro Scoring"/"Micro Scoring"; registry fallback behavior unaffected), `reward.py` (composition + the single `length_gaussian` home — closes A0.7), dataclasses `ChunkScore` (rubric scores, `grounding_ratio`, claim verdict counts, `contradicted` flag, notes, combined `score` ∈ [0,1]) and `MacroScore` (rubric scores + combined `score`). `class ReportEvaluator(llm=None)` reads `reward.judge.*` config; async `score_chunk(chunk, sources, task, context) -> ChunkScore | None` and `score_report(task, outline, report) -> MacroScore | None` (`None` = judge failure per 1.3).
- **Test:** acceptance `test_stage2_1_evaluation_module` (imports, dataclass fields, prompt entries gone from the prompt set, method signatures).
- **Done note:** —

### 2.2 `[ ]` Grounded micro judge + claim check *(report A1.2, TD3)*
- **What:** `score_chunk` = two judge calls: (1) chunk audit — prompt contains task, the section's source chunks, progress summary, chunk; verifiability rubric rewritten as "supported by the provided sources"; (2) claim check — extract factual claims, verdict each against the sources (`supported/unsupported/contradicted`) → `grounding_ratio`. Combined `score`: rubric mean modulated by grounding (contradiction hurts more than absence; exact formula documented in code + config-weighted if simple). Empty-source sections (possible: Collector can append with no new sources) get the audit call only, `grounding_ratio=None`, score = rubric mean — never a spurious penalty.
- **Test:** acceptance `test_stage2_2_grounded_micro` (structural: two calls, sources + task in the prompts, ChunkScore contract, no-sources path clean); the verdict-ordering semantics (contradicted < unsupported < supported) land in a dedicated `tests/test_evaluation.py` written with this stage.
- **Done note:** —

### 2.3 `[ ]` Terminal macro judge *(report A1.3)*
- **What:** `score_report(task, outline, report)`: one call on the full report with the task and `planned_sections`; rubric = coverage-vs-task/outline, flow, structure, redundancy (renamed key). Used at run end only.
- **Test:** acceptance `test_stage2_3_terminal_macro` (prompt carries task + outline; score composed from stub JSON).
- **Done note:** —

### 2.4 `[ ]` Reward composition v2 *(report A1.1, TD1/TD2)*
- **What:** `reward.py`: `compose_event_reward(chunk_score, delta_length_gauss, cfg)` = `quality_weight·chunk.score + length_weight·Δgauss`; `compose_terminal_reward(macro_score, cfg)` = `macro_weight·macro.score`; per-step token penalty `token_weight·(tokens/10_000)` subtracted at flush time (flag-gated: default weight 0.0). Config `reward:` v2: `quality_weight, macro_weight, length_weight, token_weight, length_goal, length_sigma, judge.{model, temperature, max_tokens, retries, samples}` — every key read by code *(report A2.3)*.
- **Test:** acceptance `test_stage2_4_reward_composition` (pure math incl. token term on/off; config keys read; defaults match the table).
- **Done note:** —

### 2.5 `[ ]` Controller rewiring *(report A1.1/A1.3 wiring; absorbs B0.4; OD-A)*
- **What:** `QMIXRoundController(trainer, agent_names, train, epsilon, evaluator=None, ...)` — `score_fn` replaced by the evaluator (breaking, sanctioned). `_reward_event`: score the *just-appended* section (`sections[-1]` content + sources) → event reward assigned to the **most recent recorded step only** (OD-A); earlier buffered steps flush 0 at that moment; judge failure → event skipped, buffer intact (1.3). `on_run_end`: flush leftovers at 0 (minus token term), then if training and report non-empty → `score_report` → terminal reward **added to the final step's** `team_reward`; still idempotent. `trainer.compute_reward` + `length_weight`/`report_quality_weight` deleted from the trainer. `Score`/`LengthGoal` singletons retired from the reward path (kept only as plain last-value holders for the episode print, or deleted if nothing else reads them — decide in-commit). **Retire `test_score_delta_semantics` in this commit.**
- **Test:** acceptance `test_stage2_5_controller_rewired` (offline scripted episode through the real controller with a stub evaluator: event reward on the event step only, zeros elsewhere, terminal macro on last step, skip-on-failure, token term applied when weight > 0); guards `test_reward_event_on_append_only`, `test_on_run_end_idempotent` stay green.
- **Done note:** —

### 2.6 `[ ]` Runner/CLI wiring + `experiments/eval.py` retirement
- **What:** `run_qmix_train(tasks, evaluator=None, ...)` (builds a default `ReportEvaluator` when None); `experiments/run_qmix_train.py` updated (no `report_score` import); `experiments/eval.py` **deleted** — the benchmark harness's legacy adapter (0.2) carries its own frozen copy of the old prompts if still needed for comparison runs, or the harness keeps only the v2 adapter once 2.8 is recorded. **Re-point `test_node_import_and_eval_standalone` in this commit** (drop the `experiments.eval` import; keep the Node pin; length-gaussian pin moves to the evaluation module import).
- **Test:** acceptance `test_stage2_6_runner_wiring` (runner signature; CLI imports clean; `experiments/eval.py` gone).
- **Done note:** —

### 2.7 `[ ]` Offline test refresh
- **What:** Update `tests/test_qmix_dry_run.py` to the evaluator interface (stub evaluator instead of stub score_fn); assert the new reward placement (event step + terminal) inside the dry run. Full battery green.
- **Test:** the dry-run suite itself + full battery.
- **Done note:** —

### 2.8 `[ ]` **LIVE** Benchmark re-run #2 (evaluator v2)
- **What:** Run the harness with the v2 adapter (grounded scoring needs source chunks — for benchmark PDFs, source-less mode exercises the audit path; add one RAG-backed run on a generated report for the grounded path). Compare against 0.3/1.4: ranking accuracy must not regress, corruption probes must pass (off-topic ↓ `subject_coverage`, perturbed numbers ↓ grounding). Record here.
- **Done note:** —

---

## Stage 3 — Trainer correctness & Double-DQN (TD4, OD-B)

### 3.1 `[ ]` Masks through the buffer *(report B0.1)*
- **What:** `Episode.to_tensors()` stacks `EpisodeStep.mask` (steps with `mask=None` → all-valid); `EpisodeBatch` gains `avail_actions: (B, T, N−1, n_actions)` bool; padded slots all-valid (prevents −inf-everywhere rows). `train_step`: masked Q (−inf on invalid) for **both** the target max and the Double-DQN argmax (3.3).
- **Test:** acceptance `test_stage3_1_masked_targets` (synthetic batch where an invalid action has the highest target Q → target must ignore it); guard `test_episode_mask_recorded` stays green.
- **Done note:** —

### 3.2 `[ ]` Terminal-transition fix *(report B0.2)*
- **What:** `ReplayBuffer.sample` pads to `max_len + 1` (PyMARL-style filled slot: obs/adj/state zero, avail all-valid, reward 0, done carried as 1 after episode end, timestep mask 0) so `targets[:, :-1]` covers every REAL transition including the terminal one of max-length episodes.
- **Test:** acceptance `test_stage3_2_terminal_trained` (two-episode batch, equal max length, terminal reward ≠ 0 → appears in the TD targets / loss gradient).
- **Done note:** —

### 3.3 `[ ]` Double-DQN targets *(report B1.1)*
- **What:** `a* = argmax_a Q_online(s', a)` (masked), `y = r + γ(1−done)·Q_target_tot(s', a*)` — the target-side max over `target_acting_q` is replaced by a gather at the online argmax before mixing.
- **Test:** acceptance `test_stage3_3_double_dqn` (online and target nets forced to disagree → target uses online's argmax evaluated by target net, not target's own max).
- **Done note:** —

### 3.4 `[ ]` Dropout removal *(report B0.3)*
- **What:** Remove `Dropout` from `GNNMessagePassing` (keep LayerNorm); `select_actions` additionally wraps in `eval()`/restore (belt and braces for any future stochastic layer). Old checkpoints are invalid — none exist worth keeping (relaxed constraint).
- **Test:** acceptance `test_stage3_4_no_dropout` (no Dropout modules in the network; two greedy `select_actions` calls on identical inputs return identical actions).
- **Done note:** —

### 3.5 `[ ]` Vectorized `train_step` *(report B1.2)*
- **What:** `GNNLayer`/`GNNMessagePassing` accept batched inputs (`(B,N,obs)`, `(B,N,N)`) via `matmul`; `train_step` drops the per-sample Python loop (one forward per timestep for the whole batch, online + target). Keep the GRU time loop.
- **Test:** acceptance `test_stage3_5_vectorized_equivalence` (batched forward ≡ per-sample loop numerically, atol 1e-5; train_step runs); guard `test_trainer_checkpoint_roundtrip` stays green.
- **Done note:** —

### 3.6 `[ ]` GNN aggregation direction *(report B1.3, OD-B)*
- **What:** Aggregate over **in-edges + self-loops**: effective adjacency `Â = Aᵀ + I` (row-normalized as today). `build_adj` semantics unchanged (`A[i,j]=1` ⇔ i sends to j); the transpose lives in the GNN and is documented as the convention in both modules.
- **Test:** acceptance `test_stage3_6_in_edge_aggregation` (single directed edge i→j on distinguishable features → j's embedding changes, i's only via self-loop).
- **Done note:** —

### 3.7 `[ ]` Cosmetics *(report B0.5)*
- **What:** Fix stale buffer comments; drop unused `HyperNetwork.n_agents` and dead `EpisodeStep.rewards` per-agent array; correct the reward-event log line (buffered-step count, not episode length); remove the trainer's `__main__` toy block or update it to real dims.
- **Test:** battery green (no dedicated test).
- **Done note:** —

---

## Stage 4 — Training-loop engineering

### 4.1 `[ ]` Replay ratio & warmup *(report B2.1)*
- **What:** `qmix.training.train_steps_per_episode` (default 4) and `qmix.training.min_buffer_episodes` (default 8): after each episode, if buffer ≥ min, run k train_steps (batch = min(batch_size, len(buffer)), sampled with replacement below `batch_size`). Revisit `target_update_interval` default against the new gradient-step rate (note the chosen value here).
- **Test:** acceptance `test_stage4_1_replay_ratio` (stub trainer counts train_step calls; warmup respected).
- **Done note:** —

### 4.2 `[ ]` Eval protocol & best-checkpoint *(report B2.2)*
- **What:** Every `qmix.training.eval_interval` episodes (default 25): one **greedy** episode (ε=0, D6 shape, scored by the evaluator, NOT pushed to the buffer); its score decomposition logged. `best checkpoint` = best eval score (fallback: moving average of last 5 training rewards until the first eval) — replaces the single-episode `best_reward` criterion.
- **Test:** acceptance `test_stage4_2_eval_protocol` (stubbed runner: eval fires at the interval, greedy, not pushed; best-ckpt follows eval score).
- **Done note:** —

### 4.3 `[ ]` JSONL run log *(report B2.3)*
- **What:** Append one record per episode to `<output_root>/qmix_train_log.jsonl`: episode idx, task, ε, steps, tokens, per-term reward decomposition (chunk scores, grounding ratios, macro breakdown, length/token terms), judge failures/retries, loss stats (all k steps), wall time, eval results when present; header record with config echo + seed.
- **Test:** acceptance `test_stage4_3_jsonl_log` (records written, required keys present, decomposition fields populated from stub evaluator dataclasses).
- **Done note:** —

### 4.4 `[ ]` Seeding *(report B2.4)*
- **What:** `qmix.training.seed` (default unset = current behavior) → seeds `torch`, `numpy`, `random` at run start; logged in the JSONL header.
- **Test:** acceptance `test_stage4_4_seeding` (same seed → identical network init; logged).
- **Done note:** —

### 4.5 `[ ]` Checkpoint v2 + buffer persistence *(report B2.5, OD-C)*
- **What:** Checkpoints add `epsilon`, `episode_idx`, `best_eval`, config echo. `qmix.training.persist_buffer` (default true): buffer episodes saved next to the checkpoint (`<save_path>.buffer.pt`); `--resume` restores buffer + ε + counters so resume actually resumes.
- **Test:** acceptance `test_stage4_5_checkpoint_v2` (save→load round-trip restores ε/counters/buffer length); guard `test_trainer_checkpoint_roundtrip` updated additively in this commit if keys changed.
- **Done note:** —

---

## Stage 5 — Docs, config, sync

### 5.1 `[ ]` README update
- **What:** Rewrite the training/evaluation sections: reward v2 (per-chunk + terminal macro, grounding, token flag), judge config, new training-loop knobs, benchmark harness usage; remove the old scorer description.
- **Done note:** —

### 5.2 `[ ]` Config finalization
- **What:** `reward:` + `qmix.training.*` reflect exactly what the code reads; no dead keys (previous project's hygiene standard); defaults documented inline.
- **Test:** acceptance `test_stage5_2_config_clean` (every documented key read; no stale keys).
- **Done note:** —

### 5.3 `[ ]` **LIVE** End-to-end smoke (2 episodes)
- **What:** `run_qmix_train --num-episodes 2 --trace` on the live model (checkpoint-4.8 protocol): no crashes, reward events fire with the new decomposition, JSONL written, terminal macro present, judge failures handled if any occur. Record findings here.
- **Done note:** —

### 5.4 `[ ]` Report/memory sync + archive
- **What:** Update the improvement report's status header (plan executed); move report + this plan to `docs/archive/` alongside the previous pair; sync assistant memory.
- **Done note:** —

---

## Open decisions

None — TD1–TD4 and OD-A–D are all settled (see decision log). New decisions discovered during implementation get added here with a recommendation.

---

## Progress overview

| Stage | Items | Done |
|-------|-------|------|
| 0 — Guardrails & baseline | 3 | 3 |
| 1 — Judge de-noising | 4 | 3 |
| 2 — Evaluation module & reward v2 | 8 | 0 |
| 3 — Trainer correctness | 7 | 0 |
| 4 — Loop engineering | 5 | 0 |
| 5 — Docs & sync | 4 | 0 |
| **Total** | **31** | **0** |

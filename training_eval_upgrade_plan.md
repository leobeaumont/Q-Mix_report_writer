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
- **Amendment #3 (2026-07-05, after the third live attempt showed frequent skips):** failures were long `global_reasoning` essays exhausting the token budget before the scores (reason-first makes truncation fatal) — and the probable root cause of BOTH the rambling and the earlier markdown/XML replies is Ollama's **silent 4096-token default context**: the macro prompt is 10-15k tokens, so the system prompt with all instructions was being silently dropped. Fix pulled forward from the Stage-2 note: judges now call Ollama's **native `/api/chat` structured outputs** (`_NativeJudgeLLM` in eval.py) — grammar-constrained decoding (malformed JSON impossible), explicit `reward.judge.num_ctx: 32768`, no frequency_penalty on judge calls, token accounting kept. `_judge_call`/conformance/retry machinery unchanged on top. Payload shape verified offline (endpoint, format, options); acceptance re-pointed to the `_judge_llm` seam. Stage 2's evaluator inherits this transport. NEW `experiments/judge_smoke.py` (30-second macro+micro health check, to run before any long live run) — **verified LIVE 2026-07-05**: composite 0.668, chunk 0.80, coherent notes, 8.8 s/pair, zero retries.
- **Amendment #4 (2026-07-05, fourth live attempt still skipped chunks):** live diagnosis on the exact failing input settled it — the reply was **complete, valid, `done_reason=stop`, 100/3072 tokens, containing ONLY `global_reasoning`**: Ollama's grammar-constrained decoding does **not enforce `required`**, and reason-first ordering makes "close the object after the notes" natural. Fix: `_judge_call` is now a **field-completion loop** — every attempt keeps the conformed fields it received and re-asks ONLY for the missing ones (`_sub_schema`; after a reasoning-only reply that is a scores-only sub-schema, with the reasoning fed back as context — reason-then-score preserved). No-progress attempts resample at ≥0.35; productive follow-ups stay at temp 0. **Live-verified on the previously-failing ACCADA_v012 chunks 1–3: scores [0.45, 0.275, 0.55], zero judge failures.** Acceptance extended with a completion-loop probe (partial kept, follow-up sub-schema, reasoning context).

### 1.4 `[x]` **LIVE** Benchmark re-run #1
- **What:** Re-run 0.3's protocol on the de-noised judges. Expectation: repeat-variance collapses (temp 0), ranking accuracy ≥ baseline. Record deltas here.
- **Done note:** 2026-07-05 (results: `benchmark_rank_legacy_20260705_023308`, `benchmark_repeat_legacy_20260705_025056`). **Ranking accuracy 0.35 → 0.800** (16/20 pairs); **repeat σ 0.0127 → 0.0000** (five identical 0.7161 runs — fully deterministic reward). Judge failures: 1 chunk of ~123 (0.8 %), skipped gracefully (the reply failed to parse identically across resamples — rare pathological input, acceptable under the skip policy). Remaining inversions are exactly the pattern the report predicted Stage 2 must fix: the Towards middle band (v06 0.661 > v18 0.625 > v20 0.622) and one ACCADA pair (v019 > v026, the doc with the failed chunk) — ungrounded style-judging can't separate mid-revisions. **STAGE 1 COMPLETE.**

---

## Stage 2 — Evaluation module & reward v2 (TD1/TD2/TD3, OD-A, OD-D)

### 2.1 `[x]` Package module `qmix_report_writer/evaluation/` *(report A2.1)*
- **What:** New package: `judges.py` (prompts + schemas move here per OD-D — `redacting_prompt_set.py` loses "Macro Scoring"/"Micro Scoring"; registry fallback behavior unaffected), `reward.py` (composition + the single `length_gaussian` home — closes A0.7), dataclasses `ChunkScore` (rubric scores, `grounding_ratio`, claim verdict counts, `contradicted` flag, notes, combined `score` ∈ [0,1]) and `MacroScore` (rubric scores + combined `score`). `class ReportEvaluator(llm=None)` reads `reward.judge.*` config; async `score_chunk(chunk, sources, task, context) -> ChunkScore | None` and `score_report(task, outline, report) -> MacroScore | None` (`None` = judge failure per 1.3).
- **Test:** acceptance `test_stage2_1_evaluation_module` (imports, dataclass fields, prompt entries gone from the prompt set, method signatures).
- **Done note:** 2026-07-05 — `qmix_report_writer/evaluation/` created: `judges.py` (ReportEvaluator, prompts+schemas per OD-D, the native transport + field-completion `_judge_call` moved here), `reward.py`, `__init__.py` exporting ChunkScore/MacroScore/ReportEvaluator + the reward fns. `redacting_prompt_set.py` lost both "…Scoring" ROLE_DESCRIPTION + JSON_SCHEMA entries (JSON_SCHEMA now `{}`, `get_schema` fallback → `{}`). Acceptance PASS.

### 2.2 `[x]` Grounded micro judge + claim check *(report A1.2, TD3)*
- **What:** `score_chunk` = two judge calls: (1) chunk audit — prompt contains task, the section's source chunks, progress summary, chunk; verifiability rubric rewritten as "supported by the provided sources"; (2) claim check — extract factual claims, verdict each against the sources (`supported/unsupported/contradicted`) → `grounding_ratio`. Combined `score`: rubric mean modulated by grounding (contradiction hurts more than absence; exact formula documented in code + config-weighted if simple). Empty-source sections (possible: Collector can append with no new sources) get the audit call only, `grounding_ratio=None`, score = rubric mean — never a spurious penalty.
- **Test:** acceptance `test_stage2_2_grounded_micro` (structural: two calls, sources + task in the prompts, ChunkScore contract, no-sources path clean); the verdict-ordering semantics (contradicted < unsupported < supported) land in a dedicated `tests/test_evaluation.py` written with this stage.
- **Done note:** 2026-07-05 — `score_chunk` = audit call + (when sources) claim-check call. Grounding factor = `(n_supported + 0.5·n_unsupported) / n_claims` (supported full, unsupported half, contradicted zero — contradiction hurts more than absence); `score = rubric_mean · (0.5 if hallucination_flag) · factor`. No sources → audit only, `grounding_ratio=None`, no modulation; no claims → factor 1.0 (no penalty). `tests/test_evaluation.py` (6 tests) pins the ordering, the flag-halving, and both no-penalty paths. Acceptance PASS.

### 2.3 `[x]` Terminal macro judge *(report A1.3)*
- **What:** `score_report(task, outline, report)`: one call on the full report with the task and `planned_sections`; rubric = coverage-vs-task/outline, flow, structure, redundancy (renamed key). Used at run end only.
- **Test:** acceptance `test_stage2_3_terminal_macro` (prompt carries task + outline; score composed from stub JSON).
- **Done note:** 2026-07-05 — `score_report` renders `<subject>`/`<outline>`/`<report>`, one macro call, `MacroScore.score = sum(5 rubric fields)/25`. Acceptance PASS; `test_evaluation.py::test_macro_composition_and_prompt` pins composition + task/outline in the prompt.

### 2.4 `[x]` Reward composition v2 *(report A1.1, TD1/TD2)*
- **What:** `reward.py`: `compose_event_reward(chunk_score, delta_length_gauss, cfg)` = `quality_weight·chunk.score + length_weight·Δgauss`; `compose_terminal_reward(macro_score, cfg)` = `macro_weight·macro.score`; per-step token penalty `token_weight·(tokens/10_000)` subtracted at flush time (flag-gated: default weight 0.0). Config `reward:` v2: `quality_weight, macro_weight, length_weight, token_weight, length_goal, length_sigma, judge.{model, temperature, max_tokens, retries, samples}` — every key read by code *(report A2.3)*.
- **Test:** acceptance `test_stage2_4_reward_composition` (pure math incl. token term on/off; config keys read; defaults match the table).
- **Done note:** 2026-07-05 — `reward.py`: `length_gaussian`, `compose_event_reward`, `compose_terminal_reward`, `token_penalty`. Config `reward:` v2 = quality_weight 1.0 / macro_weight 1.0 / length_weight 0.1 / token_weight 0.0 (TD2 off) + length_goal/sigma + the judge block from Stage 1. Judge `samples` key NOT added (the field-completion loop + temp-0 made median-of-k unnecessary; noted as a future knob if needed). Acceptance PASS.

### 2.5 `[x]` Controller rewiring *(report A1.1/A1.3 wiring; absorbs B0.4; OD-A)*
- **What:** `QMIXRoundController(trainer, agent_names, train, epsilon, evaluator=None, ...)` — `score_fn` replaced by the evaluator (breaking, sanctioned). `_reward_event`: score the *just-appended* section (`sections[-1]` content + sources) → event reward assigned to the **most recent recorded step only** (OD-A); earlier buffered steps flush 0 at that moment; judge failure → event skipped, buffer intact (1.3). `on_run_end`: flush leftovers at 0 (minus token term), then if training and report non-empty → `score_report` → terminal reward **added to the final step's** `team_reward`; still idempotent. `trainer.compute_reward` + `length_weight`/`report_quality_weight` deleted from the trainer. `Score`/`LengthGoal` singletons retired from the reward path (kept only as plain last-value holders for the episode print, or deleted if nothing else reads them — decide in-commit). **Retire `test_score_delta_semantics` in this commit.**
- **Test:** acceptance `test_stage2_5_controller_rewired` (offline scripted episode through the real controller with a stub evaluator: event reward on the event step only, zeros elsewhere, terminal macro on last step, skip-on-failure, token term applied when weight > 0); guards `test_reward_event_on_append_only`, `test_on_run_end_idempotent` stay green.
- **Done note:** 2026-07-05 — `QMIXRoundController(..., evaluator=None)` replaces `score_fn`. `_reward_event`: scores `sections[-1]` (content + stored sources), reward on the EVENT STEP only (`*earlier, event_step = buffer`), earlier steps flush 0; judge None → skip + `judge_failures++`, buffer kept. `on_run_end`: leftovers at 0, then `score_report(task, planned_sections, content)` → `compose_terminal_reward` added to the final step. `token_penalty` subtracted per step at finalize (off by default). Length gaussian baseline captured at construction; `_task` captured in `round_plan`. `trainer.compute_reward` + length/quality weights DELETED from the trainer (B0.4). `Score`/`LengthGoal` left in globals (still reset by the shared handcrafted `_reset_singletons`; no longer read in the reward path — harmless, kept to preserve handcrafted no-drift). `test_score_delta_semantics` retired (guards 20→19). Acceptance PASS; both named guards green.

### 2.6 `[x]` Runner/CLI wiring + `experiments/eval.py` retirement
- **What:** `run_qmix_train(tasks, evaluator=None, ...)` (builds a default `ReportEvaluator` when None); `experiments/run_qmix_train.py` updated (no `report_score` import); `experiments/eval.py` **deleted** — the benchmark harness's legacy adapter (0.2) carries its own frozen copy of the old prompts if still needed for comparison runs, or the harness keeps only the v2 adapter once 2.8 is recorded. **Re-point `test_node_import_and_eval_standalone` in this commit** (drop the `experiments.eval` import; keep the Node pin; length-gaussian pin moves to the evaluation module import).
- **Test:** acceptance `test_stage2_6_runner_wiring` (runner signature; CLI imports clean; `experiments/eval.py` gone).
- **Done note:** 2026-07-05 — `run_qmix_train(tasks, evaluator=None, …)`; runner builds `ReportEvaluator()` when None; episode print now shows chunk/macro/judge_fails from the controller (no more `Score.instance()`). CLI drops the `report_score` import. `experiments/eval.py` + `tests/test_scorer.py` `git rm`'d. Benchmark harness's `legacy` adapter removed → default `v2` adapter (grounded evaluator; corpus PDFs are source-less so it exercises the audit path + terminal macro, composite `0.3·macro+0.7·mean(chunk)` for comparability). `judge_smoke.py` rewritten for the evaluator (grounded chunk with a source + macro). Re-pointed BOTH old suites: guard `test_node_import_and_eval_standalone` → evaluation package + `length_gaussian`; acceptance `test_stage1_1_legacy_deleted` → asserts eval.py gone + `ReportEvaluator` importable. Acceptance PASS; old acceptance 12/12, old guards 19/19.

### 2.7 `[x]` Offline test refresh
- **What:** Update `tests/test_qmix_dry_run.py` to the evaluator interface (stub evaluator instead of stub score_fn); assert the new reward placement (event step + terminal) inside the dry run. Full battery green.
- **Test:** the dry-run suite itself + full battery.
- **Done note:** 2026-07-05 — dry run uses a stub evaluator (per-chunk `score_chunk` + terminal `score_report`); asserts exactly 2 chunk calls / 1 macro call, reward on exactly the 2 event steps, final step > 0.7 (carries the terminal macro), done flag. NEW `tests/test_evaluation.py` (6 semantic tests). Full offline battery green: new guards 6, new acceptance 10 PASS/12 PEND, evaluation 6, dry-run/fidelity/training-shape/observations, old guards 19, old acceptance 12, review/PBDS suites.

### 2.8 `[x]` **LIVE** Benchmark re-run #2 (evaluator v2)
- **What:** Run the harness with the v2 adapter (grounded scoring needs source chunks — for benchmark PDFs, source-less mode exercises the audit path; add one RAG-backed run on a generated report for the grounded path). Compare against 0.3/1.4: ranking accuracy must not regress, corruption probes must pass (off-topic ↓ `subject_coverage`, perturbed numbers ↓ grounding). Record here.
- **Done note:** 2026-07-05 (results: `benchmark_rank_v2_20260705_114224`, `benchmark_corrupt_v2_20260705_114747`). Transport rock-solid: **0 judge failures** across ~123 chunks. `numbers` corruption probe strong and correct (0.524 → 0.370, −0.153) — the audit rubric genuinely penalizes factually-wrong content. **Ranking accuracy 0.40** (below Stage 1's 0.80) and the `off-topic`/`shuffled` probes flat — but this is **not an apples-to-apples regression**: (a) the corpus PDFs carry NO task and NO sources, so `rank`/`corrupt` never exercise grounding (the one thing Stage 2 added) and the judges are subject-blind; (b) Stage 1's 0.80 came from a different protocol (legacy running-average scorer). Per user (2026-07-05): the task/source-absence effects are **benchmark artifacts, not evaluator bugs** — real generated documents always have both. Acceptance bar ("ranking must not regress") is therefore treated as **not applicable to this source-less corpus**; the genuine findings are spun into 2.9–2.10 below. The ONE real evaluator problem surfaced (grounding stayed inert even WITH a source in the smoke test) → **2.9**.

---

## Stage 2 follow-ups — real evaluator findings from the 2.8 live run

> Added 2026-07-05 after the 2.8 results. Scope filter (user): problems caused by the test PDFs lacking a task/sources are NOT evaluator bugs (production documents always carry both) — only genuine evaluator behaviour is tracked here.

### 2.9 `[x]` **LIVE** Claim extraction under-fires → grounding stays inert *(the real 2.8 finding; TD3 core)*
- **Problem:** In `judge_smoke.py` the claim check returned **0 claims** (`grounding None`, `claims 0/0/0`) for a chunk with a clearly factual sentence ("free-fall acceleration ≈ 9.81 m/s², independent of mass") **even though a matching source was provided**. When `score_chunk` gets an empty `claims` list, `grounding_ratio=None` and the grounding factor is 1.0 — so the modulation never happens. In *training* every appended section carries Collector sources, so if extraction habitually returns empty, **TD3's grounding mechanism is effectively dead** and chunk scores reduce to the ungrounded audit rubric.
- **Likely mechanism:** grammar-constrained decoding lets the model emit a valid `{"claims": []}`; the field-completion loop can't help (the `claims` key IS present, just empty); the prompt's "the most load-bearing ones, at most 8" invites conservatism.
- **Possible fixes (pick at implementation):**
  - Strengthen `CLAIM_CHECK_PROMPT`: require every quantitative/mechanistic assertion be extracted; explicitly forbid an empty list when the chunk contains numbers, units, or named relationships.
  - Add a client-side guard: if `claims == []` but the chunk carries factual markers (digits/units/`$…$`), re-ask once with a stronger nudge before accepting empty.
  - Consider grounding off the audit's `verifiability_score` as a fallback when no claims are extractable (so grounding never silently no-ops), or fold source-grounding into the verifiability rubric directly.
  - Set a `min_claims` expectation in the schema description; possibly raise the claim-check `num_predict`.
- **Test:** extend `judge_smoke.py`/a dedicated **LIVE** grounding probe: (a) chunk + matching source → ≥1 `supported`; (b) chunk + a source with the numbers perturbed → ≥1 `contradicted` and a visible score drop. Offline: a `test_evaluation.py` case asserting a non-empty claims reply drives `grounding_ratio` and the factor as expected (already covered structurally; add the "digits present ⇒ re-ask on empty" guard test if that fix is chosen).
- **Done note:** 2026-07-08 — implementation landed, **LIVE verification pending**. Fixes chosen: (1) `CLAIM_CHECK_PROMPT` rewritten — "extract ALL claims (up to 10)" with an explicit what-counts list (quantitative / mechanism / named relationship) and "empty ONLY when nothing factual is asserted; numbers or units ⇒ claims exist"; claims schema gains the same expectation as a `description`. (2) Client-side empty-claims guard in `score_chunk`: `claims == []` on a chunk with factual markers (digits or inline `$math$`) → ONE re-ask through `_judge_call` with the empty reply + nudge appended; a still-empty or failed re-ask keeps the empty result (never a penalty, never a skipped event — the first reply was valid). The `verifiability_score`-fallback option was deliberately NOT taken (changes the no-claims-no-penalty semantics pinned in `test_evaluation.py`; revisit only if the live probes still under-fire). Offline: `test_evaluation.py` 6→9 tests (re-ask fires on markers + reply drives grounding; marker-free empty accepted with no re-ask; confirmed-empty and failed re-ask keep no-penalty), all green; full battery green. `judge_smoke.py` extended into the LIVE probe: matching source (expects ≥1 supported) + contradicting source (expects ≥1 contradicted AND a score drop), exits nonzero on inert grounding.
- **Amendment (2026-07-08, after the first live smoke stayed inert):** raw-reply diagnosis (scratchpad spy on `llm.agen`) found the ACTUAL mechanism — not conservatism but **shape**: the judge returned valid JSON with **parallel string arrays** (`{"claims": [...], "verdicts": [...]}`) instead of the schema's array of objects. So Ollama's grammar `format` is NOT enforced on NESTED schemas (extra key + wrong item types got through); `_compose_chunk_score` silently dropped the non-dict items (0 claims counted) and the re-ask guard was blind because the list was non-empty. Three fixes: (1) **`_normalize_claims`** conformance layer — accepts proper objects, parallel arrays, and bare strings with a trailing verdict; verdicts cleaned (`"Supported."`→`supported`); never guesses a verdict; used by both the composer and the guard (guard now keys on *usable* claims, catching non-empty-but-unusable replies too). (2) **Output-shape block in `CLAIM_CHECK_PROMPT`** with a one-line example — live effect: the model emits the correct object shape first-try (normalization stays as belt-and-braces). (3) **Verdict rubric sharpened** — first green-shape run verdicted a numeric disagreement (source 3.51 vs chunk 9.81) as `unsupported`; rubric now states "a different value for the same quantity is `contradicted`, never `unsupported`". `test_evaluation.py` 9→12 (parallel arrays normalized without re-ask; trailing-verdict strings; all-unusable items trigger the re-ask).
- **LIVE verification (2026-07-08): smoke GREEN** — after the full hardening (incl. the 2.10-round evidence machinery): matching source 2/2 supported, grounding 1.0, score 1.0000; contradicting source 2/2 **contradicted**, grounding 0.0, halluc flagged, score 0.0000 (**drop +1.0000**, maximal). Claim extraction engages; TD3 grounding operational in the paraphrase regime training runs in.

### 2.10 `[~]` Re-measure discrimination once grounding engages (benchmark fidelity for grounding)
- **Problem:** Same-revision discrimination was poor in 2.8 (ACCADA all within 0.425–0.447; Towards *descended* v01→v20 with the 3-chunk v01 scoring highest). This is **confounded**: the differences between paper revisions are largely factual/evidential, exactly the axis grounding scores — and grounding was inert (2.9) and source-less here. It cannot be judged a real evaluator defect until grounding actually fires. Also, the benchmark currently cannot test the grounded path at all (no sources in the corpus).
- **Possible fixes:**
  - Add a **grounding probe** to `scorer_benchmark.py`: synthesize sources for a document (e.g. take each chunk's own sentences as "supporting" sources, and a numbers-perturbed copy as "contradicting"), so the claim-check + grounding factor are exercised and measurable offline-of-corpus.
  - Optionally derive a per-PDF `subject` from the title/first chunk so the audit/macro are not subject-blind — improves comparability, though task-absence itself is a benchmark artifact (not tracked as a bug).
  - Re-run ranking with grounding engaged (post-2.9) and record whether discrimination recovers; only then decide if a real evaluator problem remains.
- **Test:** the new grounding probe (LIVE) + its offline metric-shape checks in the acceptance suite.
- **Done note:** 2026-07-08 — harness support landed. `scorer_benchmark.py` gains: (1) `ground` mode (`ground --doc X [--max-chunks 8] [--probe-chars 1500]`) — per numeric chunk, one pass against a SUPPORTING source and one against a `perturb_numbers` copy; pure `grounding_probe_summary` reports `claims_fired_fraction`, `mean_grounding_supporting`, `contradiction_detected_fraction`, `mean_contradicted_claims`, `mean_score_drop`, `numeric_claim_fraction`. (2) `derive_subject` (first 20 words ≈ the title) + `--derive-subject` on rank/corrupt — un-blinds `subject_coverage` (the flat off-topic probe of 2.8). (3) `--self-sources` on rank (reordered per-chunk sources) so the claim-check path runs during ranking. Adapter gained `sources_per_chunk` (default None = old behavior). NEW acceptance `test_stage2_10_grounding_probe` (23rd test) PASS; battery green (15 suites).
- **Amendment (2026-07-08) — four probe-driven evaluator-hardening rounds** (each diagnosed on raw judge replies via a scratchpad `llm.agen` spy; all live on the user's Qwen3-30B):
  1. *Probe run 1* (verbatim self-source, 4000-char pieces): claims fired everywhere but perturbed passes verdicted **10/10 supported**. Raw replies showed the extracted claims were mostly QUALITATIVE (component lists, process statements) — verdicts genuinely correct, the corpus digits are DOIs/years/citations. Fix: quantitative claims get PRIORITY in extraction; digit-for-digit comparison rule; `ChunkScore.claims` field (normalized items, feeds the 4.3 JSONL log later); probe gained `n_numeric_claims`/`numeric_claim_fraction` diagnostics.
  2. *Probe run 2*: numeric claims now extracted ("600 MeV accelerator", "100 MeV linac by 2026") but STILL "supported" against digit-mangled sources — the judge **fabricates evidence by copying the claim** when chunk and source are near-verbatim twins (its "evidence" for 600 MeV quoted "600 MeV"; the source said "199 MeV"; it even extracted the source's 199-value as a chunk claim → cross-text leakage). Fix: `evidence` field (verbatim source quote, claim→evidence→verdict order) + "claims come from the chunk ONLY" rule + probe pieces trimmed to `--probe-chars 1500` (≈ a training section — measure the evaluator at reward scale).
  3. *Probe run 3* (sentence-reversed sources — a verbatim self-source is an unrealistic regime; training sources are paraphrases, never byte-copies): evidence STILL copied from the claim. Conclusion: prompt wording cannot make this model digit-compare near-twin sentence pairs — behavioral floor.
  4. *Fix that landed:* **client-side evidence verification** (`_verify_evidence` + NFKD `_norm_for_match`): `supported`/`contradicted` verdicts whose evidence quote does NOT occur in the sources (case/whitespace/punctuation/ligature-insensitive, digits preserved) are downgraded to `unsupported` (original kept as `verdict_raw`). Deterministic, offline-tested (`test_evaluation.py` 12→13 incl. `test_fabricated_evidence_downgraded`). *Probe run 4 (final):* supporting passes keep 58 supported vs 10 downgraded (mean grounding 0.834); perturbed passes get **25 fabricated supports caught** (grounding 1.0→0.5/0.2 on the numeric chunks); the 39 kept are digit-free qualitative claims the perturbed source legitimately still contains. Results: `benchmark_ground_v2_20260708_{103349,104628,110423,111326}`.
- **Residual (accepted, documented):** on digit-twin sentence pairs the judge emits `supported` (now → `unsupported`, factor 0.5) rather than `contradicted` (factor 0) — `contradiction_detected_fraction` stays 0.0 on this corpus probe. In the PARAPHRASE regime (smoke; = training conditions, where sections restate RAG excerpts) contradiction detection is perfect (2/2, drop +1.0). Pushing further would overfit judge prompts to an adversarial artifact; the audit call independently catches blatant numeric corruption regardless (halluc flag, perturbed-pass scores 0.09–0.35). Revisit only if training logs show grounded scores not separating factually-wrong sections.
- **REMAINING (user):** the long discrimination re-read — `rank --derive-subject` (optionally `--self-sources`) vs 2.8's 0.40; record here, then close 2.10.

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
| 1 — Judge de-noising | 4 | 4 |
| 2 — Evaluation module & reward v2 | 10 | 8 (+2 in progress: 2.9/2.10 live runs pending) |
| 3 — Trainer correctness | 7 | 0 |
| 4 — Loop engineering | 5 | 0 |
| 5 — Docs & sync | 4 | 0 |
| **Total** | **33** | **15** |

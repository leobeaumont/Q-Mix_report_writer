# Reworking the QMIX training algorithm and the report evaluator

> **Status: VALIDATED (user, 2026-07-03) — in implementation.** All decisions settled (TD1–TD4 in the log below; OD-A–D all resolved to their recommended options). The *what/when/done* tracker is [`training_eval_upgrade_plan.md`](../training_eval_upgrade_plan.md) at the repo root. Constraint relaxation granted by the user: the scorer and trainer internals may be replaced outright (no behavior-preservation shims); the handcrafted pipeline and controller-seam environment semantics keep the strict no-drift rule.

**Date:** 2026-07-03
**Scope:** The two components the previous rework explicitly deferred: the **report evaluator** (`experiments/eval.py` judges + the `Score`/`LengthGoal` reward path) and the **training algorithm** (`qmix/qmix_trainer.py`, `qmix/replay_buffer.py`, network details, and the training loop in `qmix/runner.py`).
**Out of scope (settled by the previous project):** the pipeline/environment itself — phases, controller seam, action space v2, masks, observation builder, prompt sets. Only their *integration points* are touched: the `score_fn` seam becomes a richer evaluator interface, `compute_reward` moves out of the trainer, the stored `EpisodeStep.mask` finally gets consumed, and the reward-event wiring inside `QMIXRoundController` changes shape. The handcrafted pipeline must remain byte-identical in default behavior (same standing constraint as last time).

---

## 0. The core lesson to carry over

The previous rework made the *environment* trustworthy: deterministic scaffolding, masked actions, reward fired only on real appends. This rework makes the *learning signal* trustworthy, under the same principle applied one level up:

> **The policy will learn exactly what the reward measures — including its blind spots. A judge that never sees the task cannot reward topicality; a judge that never sees the sources cannot punish hallucination; a signal that shrinks as 1/n cannot teach the second half of an episode; a target max over impossible actions bootstraps value from moves that can never be played.**

Every defect below is an instance of one of those four sentences. Evidence for the "shrinking signal" is already in the repo: the incremental scoring benchmark (`tests/scoring_results/`, real papers at successive revision stages) discriminates coarse quality (Towards v01 0.694 → v22 0.857) but the middle band is flat and inverted (v06 0.785, v18 0.789, v20 0.778) — the deltas the reward is built from live inside that noise floor.

The rework has two workstreams: **A — evaluator & reward signal**, **B — trainer & training loop**. A defines the target; B makes the optimizer actually chase it. Ordering inside each workstream is by (value ÷ effort) and dependency, like last time.

---

## Decision log (settled with user, 2026-07-03)

| # | Decision | Choice |
|---|----------|--------|
| TD1 | Reward composition | **Per-chunk + terminal macro**: the event reward is the appended chunk's own (grounded) micro score — absolute, no running average — plus length shaping; the macro judge runs ONCE per episode at run end as a terminal reward. The telescoping Δ-of-composite semantics is dropped deliberately: constant per-event signal strength beats the shaping property, and macro cost falls from O(n²) to O(n) tokens per episode. |
| TD2 | Token-cost term | **Flag-gated, default off**: a per-step penalty ∝ tokens spent is implemented and configurable (`reward.token_weight`), but ships disabled. First real training runs optimize pure quality; the efficiency A/B (the lever `no_op` was designed for) comes later by flipping one config key. |
| TD3 | Judge grounding | **Sources + claim-level check**: the micro judge receives the section's stored RAG source chunks (`ReportState.sections[i]["sources"]`), and a second judge call extracts the chunk's factual claims and verdicts each against those sources (supported / unsupported / contradicted). Verifiability and hallucination become measured quantities, at ~2 judge calls per append. |
| TD4 | Trainer scope | **Fixes + Double-DQN**: all verified-defect fixes and loop engineering, plus Double-DQN targets (argmax with the online net, evaluate with the target net). No n-step / PER / architecture changes — stays faithful to the QMIX paper. |

---

# Part A — Evaluator & reward signal

## A-Tier 0 — Verified defects in the current judges (independent of any redesign)

| # | Defect | Where | Evidence / consequence |
|---|--------|-------|------------------------|
| A0.1 | **The macro judge is never told the task.** Its `subject_coverage` rubric ("0 = off-topic or misinterpretation of subject") is scored with no statement of what the subject is — the judge infers it from the report itself, so a fluent report on the wrong topic scores high. It also never sees the outline (`planned_sections`). | `experiments/eval.py` (user prompt = report only), `redacting_prompt_set.py` "Macro Scoring" | Topicality — the one thing the task defines — is unrewarded. |
| A0.2 | **The micro judge is blind to the evidence.** Its "Verifiability: 5 = claims are cited" rubric runs without the section's RAG sources, and training-shape chunks *cannot* carry citation tags (tagging happens in SECTION_REVIEW, which training skips per D6). `hallucination_flag` (which halves the chunk score) is therefore a style guess, not a groundedness check. | `experiments/eval.py` micro prompt assembly; `redacting_prompt_set.py` "Micro Scoring" | The anti-hallucination signal — the most important quality dimension for a RAG system — is cosmetic. |
| A0.3 | **A JSON parse failure poisons the episode.** `safe_json_parse` → `{}` → every `.get(...)` defaults to 0 → the chunk is recorded as 0.0 in `Score.micro_scores` *forever* (running average), and the macro score is 0 for that event → a large negative reward spike followed by a rebound, both fake. Observed live in the 4.8 checkpoint run. | `experiments/eval.py` | One transport hiccup rewrites the reward history of the whole episode. |
| A0.4 | **Judges sample at temperature 0.7.** `report_score()` uses `get_llm()` which applies `llm.temperature: 0.7` / `top_p: 0.8`; `agen()` accepts a per-call temperature but the evaluator never passes one. The judge model is also implicitly the pipeline default model — the training CLI's `--llm` flag changes the actors but not the judges (undocumented). Related: a schema response that hits the `max_tokens` ceiling is returned as truncated JSON (`ollama_chat.py` deliberately skips trimming for schema responses) → feeds A0.3. | `experiments/eval.py`, `utils/config.py::get_llm`, `configs/default.yaml` | The reward is a high-temperature sample; Δscores are dominated by sampling noise. |
| A0.5 | **Schema/prompt mismatches.** The reasoning field (`global_reasoning` / `local_audit_notes`) is *last* in the schema — with strict structured decoding the scores are generated *before* the reasoning, violating the prompt's own "perform your reasoning before assigning scores". Rubric names don't match schema keys ("Narrative Flow" → `global_flow`, "Global Redundancy" → `redundancy_penalty` — named as a penalty but 5 = best). Leftover framing: "document intended for AI training data". | `redacting_prompt_set.py` (prompts + `JSON_SCHEMA`) | Free score-quality loss; reason-then-score is the cheapest known judge improvement. |
| A0.6 | **The first reward event returns the full score as its "delta".** `Score.get_delta()` with `previous_score=None` returns `current_score` (~0.5–0.8) while every later event returns a difference (~±0.05) — the first append earns an order of magnitude more reward than any other. (Defensible under telescoping semantics; moot under TD1, but must not survive into the new code.) | `utils/globals.py::Score.get_delta` | Systematic credit bias toward the first section. |
| A0.7 | **`length_score` logic is duplicated** in `experiments/eval.py` and `QMIXRoundController._length_score` (env-side re-statement, flagged in plan 4.4). | both files | Divergence risk; consolidate in the new reward module. |
| A0.8 | **The running micro average is unweighted and append-only** — a 200-char stub weighs as much as a 4000-char section, and (outside training shape) `remove_section` doesn't remove its score. | `experiments/eval.py`, `utils/globals.py` | Dies with the running average under TD1; listed for completeness. |

## A-Tier 1 — Reward v2 (implements TD1/TD2/TD3)

### A1.1 Per-event reward = grounded chunk score + length shaping
On each real append (trigger unchanged — `len(additions)` growth in `on_round_end`):

```
r_event = w_quality · chunk_score + w_length · Δlength_gauss     (w_token · tokens term: flag-gated, default off — TD2)
```

- `chunk_score` is the appended chunk's own score from the grounded micro judge (A1.2) — absolute, in [0,1], *not* a delta of a running average. Every section's reward has the same scale, first to last; A0.6 and the 1/n decay disappear by construction.
- `Δlength_gauss` keeps the existing gaussian-around-goal shaping (one implementation, in the new reward module — closes A0.7).
- The `Score` singleton's `micro_scores`/`micro_notes` running-average machinery is retired from the reward path (kept only if the judges still want short audit-history context).

### A1.2 Grounded micro judging (TD3) — two calls per event
1. **Chunk audit** (existing call, upgraded): prompt now contains the *task*, the *section's stored source chunks*, the progress summary, and the chunk. Rubric rewritten so verifiability means "supported by the provided sources", not "has citation marks". Reasoning field first (fixes A0.5).
2. **Claim check** (new call): extract the chunk's factual claims and give each a verdict against the sources — `supported / unsupported / contradicted`. Produces `grounding_ratio` = supported/total and a hard flag on any `contradicted`. `chunk_score` combines rubric score and grounding ratio (exact combination = plan-time detail; contradiction should hurt more than absence).

Cost: ~2 judge calls per append (vs 2 today — the macro call moves to run end, so per-event cost is *unchanged* while the signal becomes grounded).

### A1.3 Terminal macro reward
At `on_run_end`, one macro call on the full report — now given the **task** and the **outline** (`planned_sections`) — scoring coverage-vs-task, flow, structure, redundancy. Its score lands as a terminal reward component (`w_macro`) on the final step, where TD bootstrapping propagates it backward. This is also where episode-level failure modes (empty report, aborted episode) naturally score ~0.

### A1.4 Judge reliability kit
- **Temperature 0** (explicit per-call override) for all judge calls; `top_p`/`top_k` neutralized.
- **Dedicated judge config**: `reward.judge_model` (default = pipeline default model) + `reward.judge_max_tokens` — makes the current implicit behavior explicit and lets a stronger/cheaper judge be swapped in (A0.4).
- **Parse-failure policy**: one retry; on persistent failure the event is *skipped* (steps stay buffered for the next event; a warning is logged and counted) — a failure never records a 0 (A0.3).
- **Reason-first schemas**, aligned rubric/key names, `redundancy_penalty` renamed (A0.5).
- Optional `reward.judge_samples: k` (median of k, default 1) as a config-only noise knob for later.

## A-Tier 2 — Evaluator infrastructure

### A2.1 Package module with a real interface
`experiments/eval.py` (repo-root, injected as a bare `score_fn`) becomes package module(s), e.g. `qmix_report_writer/evaluation/` — `judges.py` (prompts live with the other prompt sets), `reward.py` (composition, weights, length gaussian, token term). The controller stops calling `trainer.compute_reward` (B0.4) and instead takes an **evaluator object** with two async methods (`score_chunk(...)`, `score_report(...)`) plus a pure `compose_reward(...)`. The CLI injection seam survives — it now injects the evaluator instead of a closure; `experiments/eval.py` remains as a thin shim or is deleted.

Score components (rubric scores, grounding ratio, verdict counts, macro breakdown) are returned as small dataclasses, not floats — the training log (B2.3) records the decomposition.

### A2.2 Scorer benchmark — measure the judges before and after
Formalize `tests/test_scorer.py` + `tests/test_documents/` (versioned real papers: ACCADA v012→final, Towards v01→v22) into a reproducible harness:
- **Ranking accuracy**: over ordered version pairs of the same paper, the fraction where the newer version scores higher (current judges: known inversions, e.g. Towards v18 > v20).
- **Repeat variance**: score the same document k times; report per-metric σ (the reward's noise floor — should collapse with temp 0).
- **Corruption probes**: score a document, then score it with an off-topic section injected / sections shuffled / numbers perturbed against sources — the score must drop, the relevant sub-metric most.
Run once on the CURRENT judges before any change (baseline), archive results; re-run after A-Tier 1 and require: no ranking regressions, lower variance, corruption probes pass. Live-optional (needs Ollama), like the previous plan's smoke runs.

### A2.3 Config: `reward:` section v2
All weights (`quality`, `macro`, `length`, `token`), judge settings (model, temperature, samples, retries), and the length-gaussian parameters in one place; every key actually read by the code (the previous project's config-hygiene standard).

---

# Part B — Trainer & training loop

## B-Tier 0 — Verified defects

| # | Defect | Where | Fix direction |
|---|--------|-------|---------------|
| B0.1 | **Stored action masks never reach `train_step`.** `EpisodeStep.mask` exists (Stage 4.3 left it "stored, not consumed") but `Episode.to_tensors()` drops it and `EpisodeBatch` has no field for it — so the target `max` ranges over actions that were invalid at that step (masked self-queries, masked `append`, masked `no_op`), bootstrapping value from moves that can never be played. | `replay_buffer.py`, `qmix_trainer.py` | Plumb masks through `to_tensors`/`EpisodeBatch`; set masked Q to −inf before the target max (and before the Double-DQN argmax, B1.1). |
| B0.2 | **The longest episode in each batch loses its terminal transition.** Padding is to `max_len`, and the loss uses `targets = rewards[:, :-1] + γ(1−done[:, :-1])·Q[:, 1:]` — for an episode of exactly `max_len` steps, index `T−1` (the step whose reward carries the last section's share, and under A1.3 the terminal macro reward) is cut off by `[:-1]` and never gets a TD target. Shorter episodes are fine (zero-padding gives them a slot). Episode lengths cluster (~15–17), so several episodes per batch are affected, always at the step that matters most. | `qmix_trainer.py::train_step`, `replay_buffer.py::sample` | Pad to `max_len + 1` (PyMARL-style "filled" slot) so every real transition, terminal included, has a target row. |
| B0.3 | **Dropout is live during action selection and inference.** The GNN has `Dropout(0.1)`; the online network is never put in `.eval()` — `select_actions` runs under `no_grad` but in train mode, so acting Q-values are stochastic beyond ε-greedy (also true in the greedy inference runner). | `gnn.py`, `qmix_trainer.py::select_actions`, `runner.py` | Remove dropout from the GNN (value networks in RL rarely want it) — simpler than mode juggling; keep LayerNorm. |
| B0.4 | **Reward composition lives on the trainer.** `compute_reward` + `length_weight`/`report_quality_weight` are env/reward semantics sitting in the TD-learning class; the controller calls into the trainer to compute a number the trainer never uses itself. | `qmix_trainer.py`, `qmix_controller.py` | Moves to the evaluation/reward module (A2.1); the trainer keeps pure TD machinery. |
| B0.5 | **Cosmetics/drift:** stale comments in `replay_buffer.py` ("6 agents", "5 actions"); `HyperNetwork.n_agents` unused; the reward-event log line says "spread over `episode.length` recorded step(s)" but the spread is over `len(_step_buffer)` — misleading during debugging; per-agent `EpisodeStep.rewards` array is dead weight (team reward only). | various | Clean up in passing. |

## B-Tier 1 — Algorithm (TD4 scope)

### B1.1 Double-DQN targets
`a* = argmax_a Q_online(s', a)` (masked), `y = r + γ·Q_target(s', a*)`. A few lines once B0.1/B0.2 land; the standard cure for the maximization bias that plain `max` over a noisy Q inflicts — particularly relevant here because the reward itself is judge-noisy.

### B1.2 Vectorized `train_step`
The current inner loops run B×T (~544) single-graph forward passes per gradient step, twice (online + target). `GNNLayer` only needs `torch.mm → torch.matmul` (batched `(B,N,N) @ (B,N,D)`) and the GRU/MLP already broadcast — one forward per timestep over the whole batch, ~32× fewer calls. No behavior change (guard-tested numerically); makes B2.1's higher replay ratio affordable.

### B1.3 Adjacency direction audit (open decision OD-B)
`build_adj` sets `adj[i,j]=1` for i→j (i sends to j); `GNNLayer` aggregates `mm(adj, messages)` — i.e., each agent aggregates messages from the nodes it *sends to*, not the ones it *hears from*. Information flow in the GNN is thus opposite to the communication semantics. Recommendation: aggregate over in-edges (use `adjᵀ`) and add self-loops, documented as the convention. No trained checkpoints exist, so the change is free; the fidelity/dry-run suites pin the shapes.

### B1.4 Reward scale sanity (note, no mechanism)
Under TD1 the per-event reward is ~0.3–0.8 (chunk score) instead of ~±0.05 (deltas) — a healthier scale for TD learning with γ=0.99 over ~15-step episodes. No normalizer needed now; `grad_clip` stays. Revisit only if the first real runs show instability.

## B-Tier 2 — Training-loop engineering

### B2.1 Replay ratio & warmup
One `train_step` per episode with a 32-episode warmup means: zero learning for ~32 × minutes-long episodes, then 1 gradient step per expensive episode forever. Add `qmix.training.train_steps_per_episode` (default ~4) and `qmix.training.min_buffer_episodes` (default ~8, decoupled from `batch_size` — sample with replacement below batch_size or cap the batch at buffer size). `target_update_interval` is already in gradient steps; its default (200) should be revisited against the new step rate at plan time.

### B2.2 Evaluation protocol & checkpoint selection
Every `eval_interval` episodes, run one **greedy** episode (ε=0, same D6 shape, scored by the same evaluator) on a held-out task; log its score decomposition. "Best checkpoint" becomes best eval score (or moving average of the last m evals) — today it's the single highest ε-noisy *training* episode reward, which is closer to selecting for luck. Config: `qmix.training.eval_interval`, `eval_task`.

### B2.3 Structured run log
Per episode, append one JSONL record (episode idx, task, ε, steps, tokens, per-term reward decomposition from the evaluator dataclasses, judge failures/retries, loss stats, wall time, eval-episode results) to the output root. The console print stays; the JSONL is what the first real training run will be debugged from — the trace proved to be *the* artifact last time, this is its training-curve equivalent. TensorBoard: not now (one more dep, no hardware yet); the JSONL is trivially plottable.

### B2.4 Seeding & reproducibility
`qmix.training.seed` → `torch`/`numpy`/`random`, logged into the JSONL header. (The env stays stochastic through the LLM — the point is reproducible *network init and sampling*, not bitwise runs.)

### B2.5 Checkpoint v2 + buffer persistence (open decision OD-C)
Checkpoints gain `epsilon`, episode index, best-eval metric, and config echo. Episodes cost minutes of LLM time; a crash at episode 30 currently loses everything. Recommendation: persist the replay buffer alongside the checkpoint (episodes are small numpy arrays — tens of KB each) and reload it on `--resume`, making resume actually resume.

---

## Non-rework observations (documented, not fixed here)

1. **GRU hidden-state gap on scripted rounds:** `round_plan` (where the hidden state advances) is only consulted on controller-driven rounds; scripted write rounds happen between two policy decisions without a hidden-state update. The policy sees their effect only through the next observation. Acceptable — noted so nobody mistakes it for a bug later.
2. **Observation adjacency lags one round** (`build_adj` reads the previous round's edges at plan time). Same status: known, accepted.
3. **`qmix.obs.use_embeddings` / `embedding_dim`** stay orthogonal to this rework (flagged as a follow-up in the previous plan; unchanged here).
4. **`pyproject` dep prune** (datasets/hf-hub/jsonlines) — still the previous plan's optional follow-up, untouched.

---

## Open decisions (recommendation first; settle before or during plan writing)

### OD-A — Where does the event reward land? (blocks the controller wiring in A1.1)
- **(a) On the event step only (recommended):** the write-round step gets `r_event`, intermediate steps get 0; TD bootstrapping propagates credit backward — that is what the γ=0.99 recurrent machinery is *for*. Simple, standard, and the terminal macro (A1.3) already relies on it.
- **(b) Keep the even spread** over steps buffered since the last event (status quo): denser signal, but blurs which prep round earned the append and double-counts under bootstrapping.

Choice: (a)

### OD-B — GNN aggregation direction (blocks B1.3)
- **(a) Aggregate from in-edges + self-loops (recommended):** message passing follows "who talks to me".
- **(b) Keep current out-edge aggregation:** zero code change, semantics stay inverted but consistent.

Choice: (a)

### OD-C — Replay-buffer persistence on resume (blocks B2.5)
- **(a) Persist alongside checkpoints (recommended):** resume keeps the expensive experience.
- **(b) Checkpoint-only:** simpler files, resume restarts data collection from empty.

Choice: (a)

### OD-D — Where do the judge prompts live? (blocks A2.1)
- **(a) In the evaluation module (recommended):** judges are reward code, not agent prompts; `redacting_prompt_set.py` loses its two scoring entries.
- **(b) Stay in the prompt registry:** one prompt home, but the registry keeps a consumer that isn't an agent.

Choice: (a)

---

## Recommended implementation order

1. **Guardrails:** full offline battery green (68 tests / 12 suites baseline); **scorer-benchmark baseline run on the CURRENT judges** (A2.2 harness, live) archived — the before/after evidence for everything in Part A.
2. **A-Tier 0 quick fixes** (temp 0, parse-failure policy, reason-first schemas, task into macro prompt) — small, independently testable, immediately de-noises the baseline.
3. **A2.1 evaluation module + A1.1–A1.3 reward v2** (TD1/TD2/TD3) with the controller/runner rewired to the evaluator interface; B0.4 lands here.
4. **Benchmark re-run + corruption probes** — evaluator rework validated against #1's baseline.
5. **B-Tier 0 + B1.1–B1.2** (masks-in-targets, terminal transition, dropout, Double-DQN, vectorized train_step) — offline unit-tested against hand-computed TD targets.
6. **B-Tier 2 loop engineering** (replay ratio, eval episodes, JSONL log, seeding, checkpoint v2).
7. **Docs/config/memory sync** + final battery + a 2-episode live smoke (same protocol as checkpoint 4.8) to confirm the new reward path end-to-end.

After step 7 the system is, for the first time, actually trainable as designed: a grounded, low-noise, constant-scale reward; a TD learner that respects masks, terminals, and its own exploration; and a training loop that can be observed, evaluated, resumed, and reproduced. The first real training run (hardware permitting) starts from there.

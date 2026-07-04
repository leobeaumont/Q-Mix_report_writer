"""
Acceptance tests for training_eval_upgrade_plan.md — each test encodes the
TARGET state of one plan item, written before the work is done.

  PEND  — the future module/API/behavior does not exist yet (stage not started).
  PASS  — the stage landed and matches the planned contract.
  FAIL  — the API exists but violates the contract (a real problem).

The exit code counts only FAILs. An item is "done" when its test flips from
PEND to PASS (note it in the plan's Done note). Where the plan leaves a
name/signature to the implementer, the test asserts the suggested name —
if you choose differently during implementation, update the test in the same
commit.

Several tests adapt to the mid-plan location of an API (e.g. the judge
schemas live in the prompt registry until Stage 2 moves them into the
evaluation module) — they always test the CURRENT home against the TARGET
contract.

Run standalone from the repo root:
    .venv\\Scripts\\python.exe tests\\test_training_eval_acceptance.py
"""

import asyncio
import dataclasses
import importlib
import inspect
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
from qmix_report_writer.utils.config import get_config
from qmix_report_writer.utils.globals import ReportState, Score, LengthGoal, SourceBuffer


class Pending(Exception):
    """Raised when the target API of a plan item does not exist yet."""


def _module_or_pending(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise Pending(f"module '{name}' not present yet ({exc})")


def _attr_or_pending(module, attr: str):
    if not hasattr(module, attr):
        raise Pending(f"'{module.__name__}.{attr}' not present yet")
    return getattr(module, attr)


def _reset_state():
    ReportState.instance().reset()
    SourceBuffer.instance().reset()
    try:
        Score.instance().reset()
        LengthGoal.instance().reset()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stub-LLM machinery (schema-aware: fills any JSON schema with valid values)
# ---------------------------------------------------------------------------

def _fill_from_schema(schema):
    t = schema.get("type")
    if t == "object":
        return {k: _fill_from_schema(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_fill_from_schema(schema.get("items", {"type": "string"}))]
    if "enum" in schema:
        return schema["enum"][0]
    if t == "integer":
        return 3
    if t == "number":
        return 3.0
    if t == "boolean":
        return False
    return "stub note"


class _CaptureLLM:
    """LLM stand-in: records every call, answers with a schema-valid JSON."""

    def __init__(self, responses=None):
        self.calls = []
        self._responses = list(responses) if responses else None

    async def agen(self, messages, max_tokens=None, temperature=None,
                   response_schema=None, num_comps=1, calling_agent=None):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "schema": response_schema,
        })
        if self._responses is not None and self._responses:
            return self._responses.pop(0)
        return json.dumps(_fill_from_schema(response_schema or {"type": "object"}))

    def all_text(self):
        return "\n".join(
            str(m.get("content", m)) for c in self.calls for m in c["messages"]
        )


# ---------------------------------------------------------------------------
# Stage 0.2 — scorer benchmark harness
# ---------------------------------------------------------------------------

def test_stage0_2_benchmark_harness():
    bench = _module_or_pending("experiments.scorer_benchmark")

    ranking_accuracy = _attr_or_pending(bench, "ranking_accuracy")
    repeat_variance = _attr_or_pending(bench, "repeat_variance")
    # (older_score, newer_score) pairs; newer should win.
    assert abs(ranking_accuracy([(0.5, 0.7), (0.6, 0.4)]) - 0.5) < 1e-9
    assert abs(ranking_accuracy([(0.1, 0.2), (0.2, 0.3)]) - 1.0) < 1e-9
    assert repeat_variance([1.0, 1.0, 1.0]) == 0.0
    assert repeat_variance([0.0, 1.0]) > 0.0

    doc = "# Title\n\nAlpha section about nuclei with mass 5 GeV.\n\n# Second\n\nBeta text."
    for fn_name in ("inject_offtopic_section", "shuffle_sections", "perturb_numbers"):
        corrupted = _attr_or_pending(bench, fn_name)(doc)
        assert isinstance(corrupted, str) and corrupted != doc, f"{fn_name} must alter the document"

    # Adapter resilience: a judge failure skips the chunk's measurement
    # instead of killing an hours-long run; all-failed aborts loudly.
    import experiments.eval as eval_mod
    from qmix_report_writer.utils.globals import Score as _Score

    calls = {"n": 0}

    async def _flaky_score(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise eval_mod.JudgeError("boom")
        _Score.instance().micro_scores.append(0.5)
        return 0.6

    async def _dead_score(**kwargs):
        raise eval_mod.JudgeError("down")

    original = eval_mod.report_score
    try:
        eval_mod.report_score = _flaky_score
        try:
            out = asyncio.run(bench.legacy_scorer_adapter(None, ["a", "b", "c"]))
        except eval_mod.JudgeError:
            raise Pending("benchmark adapter does not survive judge failures yet")
        assert out.get("judge_failures") == 1
        assert out["chunk_scores"][1] is None and out["chunk_scores"][0] == 0.5
        assert abs(out["final_score"] - 0.6) < 1e-9

        eval_mod.report_score = _dead_score
        try:
            asyncio.run(bench.legacy_scorer_adapter(None, ["a", "b"]))
            raise AssertionError("all-failed run must abort, not report a fake score")
        except RuntimeError:
            pass
    finally:
        eval_mod.report_score = original
        _reset_state()


# ---------------------------------------------------------------------------
# Stage 1.1 — reason-first schemas, aligned keys, task in the macro prompt
# ---------------------------------------------------------------------------

def _current_scoring_schemas():
    """(macro_schema, micro_schema) from wherever they currently live."""
    try:
        judges = importlib.import_module("qmix_report_writer.evaluation.judges")
        for macro_name in ("MACRO_SCHEMA", "MACRO_SCORING_SCHEMA"):
            if hasattr(judges, macro_name):
                macro = getattr(judges, macro_name)
                break
        else:
            raise Pending("evaluation.judges exists but exposes no MACRO_SCHEMA")
        for micro_name in ("MICRO_SCHEMA", "CHUNK_AUDIT_SCHEMA"):
            if hasattr(judges, micro_name):
                return macro, getattr(judges, micro_name)
        raise Pending("evaluation.judges exists but exposes no MICRO_SCHEMA")
    except ImportError:
        from qmix_report_writer.prompt.redacting_prompt_set import JSON_SCHEMA
        return JSON_SCHEMA["Macro Scoring"], JSON_SCHEMA["Micro Scoring"]


def test_stage1_1_schema_reason_first():
    macro, micro = _current_scoring_schemas()

    for label, schema in (("macro", macro), ("micro", micro)):
        props = schema["properties"]
        first_key = next(iter(props))
        if props[first_key].get("type") != "string":
            raise Pending(f"{label} schema: reasoning field is not first yet "
                          f"(first property: '{first_key}')")

    if "redundancy_penalty" in macro["properties"]:
        raise Pending("macro schema still uses the inverted 'redundancy_penalty' key")

    # The macro judge must receive the task. Stage-1 form: report_score(task=...);
    # Stage-2 form: ReportEvaluator.score_report(task, outline, report).
    try:
        judges = importlib.import_module("qmix_report_writer.evaluation.judges")
        evaluator_cls = _attr_or_pending(judges, "ReportEvaluator")
        assert "task" in inspect.signature(evaluator_cls.score_report).parameters
    except ImportError:
        import experiments.eval as eval_mod
        if "task" not in inspect.signature(eval_mod.report_score).parameters:
            raise Pending("report_score does not accept the episode task yet")


# ---------------------------------------------------------------------------
# Stage 1.2 — judge call discipline (temp 0, explicit judge config)
# ---------------------------------------------------------------------------

def test_stage1_2_judge_discipline():
    reward_cfg = get_config().get("reward", {}) or {}
    judge_cfg = reward_cfg.get("judge")
    if not isinstance(judge_cfg, dict):
        raise Pending("config has no reward.judge section yet")
    assert "model" in judge_cfg, "reward.judge.model key missing"
    assert float(judge_cfg.get("temperature", 1.0)) == 0.0, \
        "reward.judge.temperature must default to 0.0"

    stub = _CaptureLLM()
    _reset_state()
    try:
        judges = importlib.import_module("qmix_report_writer.evaluation.judges")
        evaluator = _attr_or_pending(judges, "ReportEvaluator")(llm=stub)
        asyncio.run(evaluator.score_report(task="T", outline=["A"], report="## A\n\nBody."))
    except ImportError:
        import experiments.eval as eval_mod
        original = eval_mod.get_llm
        eval_mod.get_llm = lambda *a, **k: stub
        try:
            ReportState.instance().append("## A\n\nBody.", "progress")
            kwargs = {}
            if "task" in inspect.signature(eval_mod.report_score).parameters:
                kwargs["task"] = "T"
            asyncio.run(eval_mod.report_score(**kwargs))
        finally:
            eval_mod.get_llm = original
            _reset_state()

    assert stub.calls, "no judge call captured"
    for call in stub.calls:
        if call["temperature"] != 0.0:
            raise Pending(f"judge call passed temperature={call['temperature']} "
                          f"instead of 0.0")


# ---------------------------------------------------------------------------
# Stage 1.3 — parse/judge failure skips the reward event (never records 0)
# ---------------------------------------------------------------------------

def test_stage1_3_parse_failure_skips():
    from qmix_report_writer.qmix.qmix_controller import QMIXRoundController
    from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
    from qmix_report_writer.qmix.replay_buffer import EpisodeStep

    _reset_state()
    trainer = SimpleNamespace(n_actions=NUM_ACTIONS,
                              compute_reward=lambda *a, **k: 0.5)

    params = inspect.signature(QMIXRoundController.__init__).parameters
    failing_calls = {"n": 0}

    if "evaluator" in params:
        async def _fail_chunk(*a, **k):
            failing_calls["n"] += 1
            return None  # judge failure after retry

        async def _ok_chunk(*a, **k):
            return SimpleNamespace(score=0.7, grounding_ratio=1.0)

        evaluator = SimpleNamespace(score_chunk=_fail_chunk,
                                    score_report=_ok_chunk)
        ctrl = QMIXRoundController(trainer, ["A", "B", "Collector"],
                                   train=True, evaluator=evaluator)
        set_ok = lambda: setattr(evaluator, "score_chunk", _ok_chunk)
    elif "score_fn" in params:
        state = {"fail": True}

        async def score_fn():
            failing_calls["n"] += 1
            if state["fail"]:
                raise ValueError("judge JSON unparseable")
            return 0.7

        ctrl = QMIXRoundController(trainer, ["A", "B", "Collector"],
                                   train=True, score_fn=score_fn)
        set_ok = lambda: state.update(fail=False)
    else:
        raise Pending("controller has neither score_fn nor evaluator parameter")

    def _mk_step():
        fields = {f.name for f in dataclasses.fields(EpisodeStep)}
        base = dict(observations=np.zeros((3, 4), dtype=np.float32),
                    actions=np.zeros(2, dtype=np.int64),
                    rewards=np.zeros(2), team_reward=0.0,
                    adj_matrix=np.zeros((3, 3), dtype=np.float32),
                    global_state=np.zeros(5, dtype=np.float32), done=False)
        return EpisodeStep(**{k: v for k, v in base.items() if k in fields})

    async def _drive():
        ctrl._step_buffer.append(_mk_step())
        ReportState.instance().append("## S1\n\nText.", "p")
        # Failing judge: must NOT raise, must NOT flush, must NOT record a 0.
        try:
            await ctrl.on_round_end("DRAFTING", 1)
        except Exception as exc:
            raise Pending(f"controller still propagates judge failures ({exc})")
        if not ctrl._step_buffer:
            raise Pending("failed judge event still consumed the step buffer")
        assert ctrl.episode.length == 0

        # Working judge on the next append: the event fires normally.
        set_ok()
        ReportState.instance().append("## S2\n\nText.", "p")
        await ctrl.on_round_end("DRAFTING", 2)
        assert ctrl.episode.length >= 1, "recovered judge did not fire the event"

    asyncio.run(_drive())
    _reset_state()

    # --- Type conformance at the judge-call level (Ollama does not reliably
    # enforce the response schema: a live run returned notes as a JSON array
    # and crashed the audit-history prompt) ---------------------------------
    try:
        judges_mod = importlib.import_module("qmix_report_writer.evaluation.judges")
        judge_call = getattr(judges_mod, "_judge_call", None)
        judge_error = getattr(judges_mod, "JudgeError", None)
    except ImportError:
        judge_call = judge_error = None
    if judge_call is None:
        import experiments.eval as eval_mod
        judge_call = getattr(eval_mod, "_judge_call", None)
        judge_error = getattr(eval_mod, "JudgeError", None)
    if judge_call is None:
        raise Pending("no _judge_call helper with type conformance yet")

    _, micro_schema = _current_scoring_schemas()
    required = ("logical_soundness", "verifiability_score",
                "technical_precision", "info_density", "hallucination_flag")

    # Coercible reply: list notes, string int, out-of-range int, string bool —
    # must be conformed on the FIRST call (no retry burned).
    messy = json.dumps({
        "local_audit_notes": ["note a", "note b"],
        "logical_soundness": "4",
        "verifiability_score": 7,
        "technical_precision": 3,
        "info_density": 3,
        "hallucination_flag": "false",
    })
    stub = _CaptureLLM(responses=[messy])
    result = asyncio.run(judge_call(stub, [], micro_schema, required))
    assert len(stub.calls) == 1, "coercible reply should not burn a retry"
    assert result["local_audit_notes"] == "note a note b"
    assert result["logical_soundness"] == 4
    assert result["verifiability_score"] == 5, "score must be clamped to the schema max"
    assert result["hallucination_flag"] is False

    # Unusable replies: retry (resampled!), then raise — never score a
    # malformed reply. At temperature 0 an identical re-ask would just
    # reproduce the same bad reply, so retries must use a nonzero floor.
    cfg_retries = int((get_config().get("reward", {}).get("judge", {}) or {})
                      .get("retries", 1))
    stub = _CaptureLLM(responses=["not json {{{"] * (cfg_retries + 1))
    try:
        asyncio.run(judge_call(stub, [], micro_schema, required))
        raise AssertionError("garbage judge replies must raise, not score")
    except AssertionError:
        raise
    except Exception as exc:
        assert judge_error is not None and isinstance(exc, judge_error), \
            f"expected JudgeError, got {type(exc).__name__}: {exc}"
    assert len(stub.calls) == cfg_retries + 1, \
        f"expected {cfg_retries + 1} attempts, saw {len(stub.calls)}"
    assert stub.calls[0]["temperature"] == 0.0
    if not all(c["temperature"] >= 0.3 for c in stub.calls[1:]):
        raise Pending("retries are not resampled (still temperature 0)")


# ---------------------------------------------------------------------------
# Stage 2.1 — evaluation package
# ---------------------------------------------------------------------------

def test_stage2_1_evaluation_module():
    pkg = _module_or_pending("qmix_report_writer.evaluation")
    judges = _module_or_pending("qmix_report_writer.evaluation.judges")
    reward = _module_or_pending("qmix_report_writer.evaluation.reward")

    evaluator_cls = _attr_or_pending(judges, "ReportEvaluator")
    assert inspect.iscoroutinefunction(evaluator_cls.score_chunk)
    assert inspect.iscoroutinefunction(evaluator_cls.score_report)

    chunk_score = _attr_or_pending(pkg, "ChunkScore")
    macro_score = _attr_or_pending(pkg, "MacroScore")
    chunk_fields = {f.name for f in dataclasses.fields(chunk_score)}
    assert {"score", "grounding_ratio"} <= chunk_fields
    assert "score" in {f.name for f in dataclasses.fields(macro_score)}

    for fn in ("length_gaussian", "compose_event_reward", "compose_terminal_reward"):
        _attr_or_pending(reward, fn)

    # OD-D: the scoring entries left the agent prompt registry.
    from qmix_report_writer.prompt.redacting_prompt_set import JSON_SCHEMA, ROLE_DESCRIPTION
    assert "Macro Scoring" not in JSON_SCHEMA and "Micro Scoring" not in JSON_SCHEMA
    assert "Macro Scoring" not in ROLE_DESCRIPTION and "Micro Scoring" not in ROLE_DESCRIPTION


# ---------------------------------------------------------------------------
# Stage 2.2 — grounded micro judge + claim check (structural contract;
# verdict-ordering semantics are covered by the stage's own unit tests)
# ---------------------------------------------------------------------------

def test_stage2_2_grounded_micro():
    judges = _module_or_pending("qmix_report_writer.evaluation.judges")
    evaluator_cls = _attr_or_pending(judges, "ReportEvaluator")

    stub = _CaptureLLM()
    evaluator = evaluator_cls(llm=stub)
    sources = [{"content": "The measured mass is 5 GeV.", "source": "paper.pdf"}]
    result = asyncio.run(evaluator.score_chunk(
        chunk="The mass is 5 GeV.", sources=sources,
        task="Nuclear masses", context="progress summary",
    ))

    assert len(stub.calls) >= 2, "grounded scoring must make audit + claim-check calls"
    seen = stub.all_text()
    assert "The measured mass is 5 GeV." in seen, "sources not in any judge prompt"
    assert "Nuclear masses" in seen, "task not in any judge prompt"
    assert result is not None and 0.0 <= result.score <= 1.0
    assert result.grounding_ratio is None or 0.0 <= result.grounding_ratio <= 1.0

    # Source-less sections: audit only, no spurious grounding penalty.
    stub2 = _CaptureLLM()
    evaluator2 = evaluator_cls(llm=stub2)
    result2 = asyncio.run(evaluator2.score_chunk(
        chunk="Text.", sources=[], task="T", context="c",
    ))
    assert len(stub2.calls) == 1, "no-sources chunk should skip the claim check"
    assert result2.grounding_ratio is None


# ---------------------------------------------------------------------------
# Stage 2.3 — terminal macro judge
# ---------------------------------------------------------------------------

def test_stage2_3_terminal_macro():
    judges = _module_or_pending("qmix_report_writer.evaluation.judges")
    evaluator_cls = _attr_or_pending(judges, "ReportEvaluator")

    stub = _CaptureLLM()
    evaluator = evaluator_cls(llm=stub)
    result = asyncio.run(evaluator.score_report(
        task="Nuclear equation of state",
        outline=["Introduction", "Dense matter"],
        report="## Introduction\n\nBody.\n\n## Dense matter\n\nBody.",
    ))
    assert len(stub.calls) == 1
    seen = stub.all_text()
    assert "Nuclear equation of state" in seen, "task missing from macro prompt"
    assert "Dense matter" in seen, "outline missing from macro prompt"
    assert result is not None and 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Stage 2.4 — reward composition v2 + config
# ---------------------------------------------------------------------------

def test_stage2_4_reward_composition():
    reward = _module_or_pending("qmix_report_writer.evaluation.reward")

    length_gaussian = _attr_or_pending(reward, "length_gaussian")
    compose_event = _attr_or_pending(reward, "compose_event_reward")
    compose_terminal = _attr_or_pending(reward, "compose_terminal_reward")
    token_penalty = _attr_or_pending(reward, "token_penalty")

    assert abs(length_gaussian(25000, 25000, 8500) - 1.0) < 1e-9
    assert abs(compose_event(0.8, 0.05, {"quality_weight": 1.0, "length_weight": 0.1})
               - 0.805) < 1e-9
    assert abs(compose_terminal(0.6, {"macro_weight": 2.0}) - 1.2) < 1e-9
    assert abs(token_penalty(20000, {"token_weight": 0.5}) - 1.0) < 1e-9
    assert token_penalty(20000, {"token_weight": 0.0}) == 0.0

    cfg = get_config().get("reward", {}) or {}
    for key in ("quality_weight", "macro_weight", "length_weight", "token_weight"):
        if key not in cfg:
            raise Pending(f"config reward.{key} not present yet")
    assert float(cfg["token_weight"]) == 0.0, "TD2: token term must default OFF"


# ---------------------------------------------------------------------------
# Stage 2.5 — controller rewired (OD-A placement, terminal macro, no
#             trainer.compute_reward)
# ---------------------------------------------------------------------------

def test_stage2_5_controller_rewired():
    from qmix_report_writer.qmix.qmix_controller import QMIXRoundController
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
    from qmix_report_writer.qmix.agent_network import NUM_ACTIONS
    from qmix_report_writer.qmix.replay_buffer import EpisodeStep

    params = inspect.signature(QMIXRoundController.__init__).parameters
    if "evaluator" not in params:
        raise Pending("controller does not take an evaluator yet")
    reward_mod = _module_or_pending("qmix_report_writer.evaluation.reward")
    cfg = get_config().get("reward", {}) or {}
    if "quality_weight" not in cfg:
        raise Pending("config reward v2 not present yet")

    assert not hasattr(QMIXTrainer, "compute_reward"), \
        "trainer.compute_reward must be deleted (moved to evaluation.reward)"

    _reset_state()

    async def _chunk(*a, **k):
        return SimpleNamespace(score=0.8, grounding_ratio=1.0)

    async def _macro(*a, **k):
        return SimpleNamespace(score=0.6)

    evaluator = SimpleNamespace(score_chunk=_chunk, score_report=_macro)
    trainer = SimpleNamespace(n_actions=NUM_ACTIONS)
    # NOTE (plan 2.5): the controller initializes its previous length gaussian
    # at construction time — deviations must update this test in-commit.
    ctrl = QMIXRoundController(trainer, ["A", "B", "Collector"],
                               train=True, evaluator=evaluator)

    fields = {f.name for f in dataclasses.fields(EpisodeStep)}

    def _mk_step():
        base = dict(observations=np.zeros((3, 4), dtype=np.float32),
                    actions=np.zeros(2, dtype=np.int64),
                    rewards=np.zeros(2), team_reward=0.0,
                    adj_matrix=np.zeros((3, 3), dtype=np.float32),
                    global_state=np.zeros(5, dtype=np.float32), done=False)
        return EpisodeStep(**{k: v for k, v in base.items() if k in fields})

    goal = int(cfg.get("length_goal", 25000))
    sigma = int(cfg.get("length_sigma", 8500))
    g0 = reward_mod.length_gaussian(0, goal, sigma)

    async def _drive():
        ctrl._step_buffer.extend([_mk_step(), _mk_step()])
        ReportState.instance().append("x" * 1000, "p",
                                      [{"content": "c", "source": "s"}])
        await ctrl.on_round_end("DRAFTING", 1)

    asyncio.run(_drive())

    assert ctrl.episode.length == 2, "event must flush the buffered steps"
    g1 = reward_mod.length_gaussian(len(ReportState.instance().content), goal, sigma)
    expected_event = reward_mod.compose_event_reward(0.8, g1 - g0, cfg)
    assert ctrl.episode.steps[0].team_reward == 0.0, \
        "OD-A: pre-event steps must carry 0 reward"
    assert abs(ctrl.episode.steps[1].team_reward - expected_event) < 1e-6, \
        f"event step reward {ctrl.episode.steps[1].team_reward} != {expected_event}"

    asyncio.run(ctrl.on_run_end())
    expected_final = expected_event + reward_mod.compose_terminal_reward(0.6, cfg)
    assert abs(ctrl.episode.steps[-1].team_reward - expected_final) < 1e-6, \
        "terminal macro reward must be added to the final step"
    assert ctrl.episode.steps[-1].done is True
    _reset_state()


# ---------------------------------------------------------------------------
# Stage 2.6 — runner/CLI wiring, experiments/eval.py retired
# ---------------------------------------------------------------------------

def test_stage2_6_runner_wiring():
    from qmix_report_writer.qmix import runner
    params = inspect.signature(runner.run_qmix_train).parameters
    if "evaluator" not in params:
        raise Pending("run_qmix_train still takes score_fn, not evaluator")
    assert "score_fn" not in params, "score_fn seam should be replaced, not duplicated"

    if os.path.exists("experiments/eval.py"):
        raise Pending("experiments/eval.py still present")

    with open("experiments/run_qmix_train.py", encoding="utf-8") as f:
        src = f.read()
    assert "experiments.eval" not in src and "from experiments import eval" not in src


# ---------------------------------------------------------------------------
# Stage 3 helpers — deterministic stub networks
# ---------------------------------------------------------------------------

class _StubAgentNet(nn.Module):
    """Deterministic per-action Q template times a real parameter (grad flows)."""

    def __init__(self, n_actions, action_values, hidden_dim=4):
        super().__init__()
        assert len(action_values) == n_actions
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("template", torch.tensor(action_values, dtype=torch.float32))
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, obs, adj, hidden):
        shape = obs.shape[:-1] + (self.template.shape[0],)
        q = self.template.expand(shape).clone() * self.scale
        return q, hidden


class _CaptureMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, agent_qs, state):
        self.seen.append(agent_qs.detach().clone())
        return agent_qs.sum(dim=-1)


def _mk_step_for(n_agents, n_actions, obs_dim=4, state_dim=None, action=1,
                 reward=0.0, done=False, invalid_idx=0):
    from qmix_report_writer.qmix.replay_buffer import EpisodeStep
    state_dim = state_dim or (obs_dim * n_agents + 3)
    mask = np.ones((n_agents - 1, n_actions), dtype=bool)
    mask[:, invalid_idx] = False
    fields = {f.name for f in dataclasses.fields(EpisodeStep)}
    base = dict(
        observations=np.zeros((n_agents, obs_dim), dtype=np.float32),
        actions=np.full(n_agents - 1, action, dtype=np.int64),
        rewards=np.zeros(n_agents - 1),
        team_reward=reward,
        adj_matrix=np.eye(n_agents, dtype=np.float32),
        global_state=np.zeros(state_dim, dtype=np.float32),
        done=done,
        mask=mask,
    )
    return EpisodeStep(**{k: v for k, v in base.items() if k in fields})


def _stubbed_trainer(online_vals, target_vals, n_agents=3, T=3,
                     terminal_reward=0.0, batch_size=2):
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
    from qmix_report_writer.qmix.replay_buffer import Episode

    trainer = QMIXTrainer(
        n_agents=n_agents, obs_dim=4, state_dim=4 * n_agents + 3,
        batch_size=batch_size, gnn_hidden_dim=4, gnn_layers=1,
        rnn_hidden_dim=4, mixing_hidden_dim=4, target_update_interval=10_000,
    )
    trainer.agent_network = _StubAgentNet(trainer.n_actions, online_vals)
    trainer.target_agent_network = _StubAgentNet(trainer.n_actions, target_vals)
    trainer.mixing_network = _CaptureMixer()
    trainer.target_mixing_network = _CaptureMixer()
    trainer.params = list(trainer.agent_network.parameters())
    trainer.optimizer = torch.optim.Adam(trainer.params, lr=1e-5)

    for _ in range(batch_size):
        ep = Episode()
        for t in range(T):
            ep.add_step(_mk_step_for(
                n_agents, trainer.n_actions,
                reward=(terminal_reward if t == T - 1 else 0.0),
                done=(t == T - 1),
            ))
        trainer.replay_buffer.push(ep)
    return trainer


def _avail_or_pending():
    from qmix_report_writer.qmix.replay_buffer import EpisodeBatch
    if "avail_actions" not in getattr(EpisodeBatch, "__dataclass_fields__", {}):
        raise Pending("EpisodeBatch has no avail_actions field yet (3.1)")


# ---------------------------------------------------------------------------
# Stage 3.1 — action masks applied to the TD targets
# ---------------------------------------------------------------------------

def test_stage3_1_masked_targets():
    """If the stored masks bind, the target value of an INVALID action is
    irrelevant — two runs differing only in that value must give identical
    losses. (Loss-based, so the 3.2 pad slot and 3.5 vectorization can't
    pollute the check: padded targets are multiplied by (1-done)=0.)"""
    _avail_or_pending()
    online = [1.0] * NUM_ACTIONS
    target_a = [9.0] * NUM_ACTIONS
    target_b = [9.0] * NUM_ACTIONS
    target_a[0] = 1000.0   # action 0 is invalid in every stored mask
    target_b[0] = 7777.0

    loss_a = _stubbed_trainer(online, target_a).train_step()["loss"]
    loss_b = _stubbed_trainer(online, target_b).train_step()["loss"]
    assert abs(loss_a - loss_b) < 1e-9, (
        f"changing an INVALID action's target Q changed the loss "
        f"({loss_a} vs {loss_b}): masks are not applied to the target max"
    )


# ---------------------------------------------------------------------------
# Stage 3.2 — terminal transitions get a TD target (max-length episodes)
# ---------------------------------------------------------------------------

def test_stage3_2_terminal_trained():
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
    from qmix_report_writer.qmix.replay_buffer import Episode

    # Shape probe: sampling must pad one slot past the longest episode.
    trainer = QMIXTrainer(n_agents=3, obs_dim=4, state_dim=15, batch_size=1,
                          gnn_hidden_dim=4, gnn_layers=1, rnn_hidden_dim=4,
                          mixing_hidden_dim=4)
    ep = Episode()
    for t in range(3):
        ep.add_step(_mk_step_for(3, trainer.n_actions, done=(t == 2)))
    trainer.replay_buffer.push(ep)
    batch = trainer.replay_buffer.sample(1)
    if batch.obs.shape[1] == 3:
        raise Pending("no terminal pad slot yet: sample() pads to max_len")
    assert batch.obs.shape[1] == 4, f"expected max_len+1 slots, got {batch.obs.shape[1]}"

    # Substance probe: the terminal reward must influence the loss.
    n_act = trainer.n_actions
    t0 = _stubbed_trainer([1.0] * n_act, [1.0] * n_act, terminal_reward=0.0)
    t5 = _stubbed_trainer([1.0] * n_act, [1.0] * n_act, terminal_reward=5.0)
    loss0 = t0.train_step()["loss"]
    loss5 = t5.train_step()["loss"]
    assert abs(loss0 - loss5) > 1e-6, \
        "terminal reward does not reach the TD loss"


# ---------------------------------------------------------------------------
# Stage 3.3 — Double-DQN targets (masked online argmax, target evaluation)
# ---------------------------------------------------------------------------

def test_stage3_3_double_dqn():
    """Double-DQN evaluates the TARGET net at the ONLINE net's (masked) argmax.
    Online argmax = action 1 (q=2.0 among valid). Two runs differ ONLY in the
    target value at action 1 (7.0 vs 7.5) while the target's own valid max
    stays action 2 (q=9.0) in both:
      - plain (masked) target max → uses 9.0 both times → identical losses;
      - Double-DQN → uses 7.0 vs 7.5 → losses differ."""
    _avail_or_pending()
    online = [1.0] * NUM_ACTIONS
    online[0] = 50.0    # invalid — masked argmax must skip it
    online[1] = 2.0     # the valid online argmax
    target_a = [1.0] * NUM_ACTIONS
    target_a[0] = 1000.0
    target_a[1] = 7.0
    target_a[2] = 9.0   # the target's own valid max
    target_b = list(target_a)
    target_b[1] = 7.5

    loss_a = _stubbed_trainer(online, target_a).train_step()["loss"]
    loss_b = _stubbed_trainer(online, target_b).train_step()["loss"]
    if abs(loss_a - loss_b) < 1e-9:
        raise Pending("target still uses its own max — Double-DQN not landed")


# ---------------------------------------------------------------------------
# Stage 3.4 — no dropout in the value networks; deterministic greedy actions
# ---------------------------------------------------------------------------

def test_stage3_4_no_dropout():
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer

    trainer = QMIXTrainer(n_agents=3, obs_dim=4, state_dim=15,
                          gnn_hidden_dim=4, gnn_layers=1, rnn_hidden_dim=4,
                          mixing_hidden_dim=4)
    for module in trainer.agent_network.modules():
        if isinstance(module, nn.Dropout) and module.p > 0:
            raise Pending(f"agent network still contains Dropout(p={module.p})")

    obs = torch.randn(3, 4)
    adj = torch.eye(3)
    hidden = trainer.agent_network.init_hidden(3)
    a1, _ = trainer.select_actions(obs, adj, hidden.clone(), epsilon=0.0)
    a2, _ = trainer.select_actions(obs, adj, hidden.clone(), epsilon=0.0)
    assert torch.equal(a1, a2), "greedy action selection is not deterministic"


# ---------------------------------------------------------------------------
# Stage 3.5 — vectorized train_step (batched GNN)
# ---------------------------------------------------------------------------

def test_stage3_5_vectorized_equivalence():
    from qmix_report_writer.qmix.gnn import GNNMessagePassing
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer

    gnn = GNNMessagePassing(obs_dim=4, hidden_dim=8, num_layers=1)
    gnn.eval()
    obs = torch.randn(2, 3, 4)
    adj = torch.ones(2, 3, 3)
    try:
        batched = gnn(obs, adj)
    except Exception as exc:
        raise Pending(f"GNN does not accept batched inputs yet ({exc})")
    for b in range(2):
        single = gnn(obs[b], adj[b])
        assert torch.allclose(batched[b], single, atol=1e-5), \
            "batched GNN forward diverges from per-sample forward"

    src = inspect.getsource(QMIXTrainer.train_step)
    if "for b in range(B)" in src:
        raise Pending("train_step still loops over the batch per sample")


# ---------------------------------------------------------------------------
# Stage 3.6 — in-edge aggregation + self-loops (OD-B)
# ---------------------------------------------------------------------------

def test_stage3_6_in_edge_aggregation():
    from qmix_report_writer.qmix.gnn import GNNMessagePassing

    torch.manual_seed(0)
    gnn = GNNMessagePassing(obs_dim=4, hidden_dim=8, num_layers=2)
    gnn.eval()
    obs = torch.randn(3, 4)
    adj = torch.zeros(3, 3)
    adj[0, 1] = 1.0  # node 0 SENDS to node 1

    base = gnn(obs, adj)
    obs_d0 = obs.clone(); obs_d0[0] += 1.0
    out_d0 = gnn(obs_d0, adj)
    obs_d1 = obs.clone(); obs_d1[1] += 1.0
    out_d1 = gnn(obs_d1, adj)

    receiver_hears_sender = not torch.allclose(base[1], out_d0[1], atol=1e-6)
    sender_ignores_receiver = torch.allclose(base[0], out_d1[0], atol=1e-6)

    if receiver_hears_sender and sender_ignores_receiver:
        return  # in-edge aggregation landed
    if (not receiver_hears_sender) and (not sender_ignores_receiver):
        raise Pending("GNN still aggregates over out-edges (current behavior)")
    raise AssertionError(
        f"ambiguous aggregation semantics: receiver_hears_sender="
        f"{receiver_hears_sender}, sender_ignores_receiver={sender_ignores_receiver}"
    )


# ---------------------------------------------------------------------------
# Stage 4 — training-loop engineering
# ---------------------------------------------------------------------------

def _training_cfg():
    return (get_config().get("qmix", {}) or {}).get("training", {}) or {}


def test_stage4_1_replay_ratio():
    tcfg = _training_cfg()
    for key in ("train_steps_per_episode", "min_buffer_episodes"):
        if key not in tcfg:
            raise Pending(f"config qmix.training.{key} not present yet")
    from qmix_report_writer.qmix import runner
    src = inspect.getsource(runner)
    assert "train_steps_per_episode" in src and "min_buffer_episodes" in src


def test_stage4_2_eval_protocol():
    tcfg = _training_cfg()
    if "eval_interval" not in tcfg:
        raise Pending("config qmix.training.eval_interval not present yet")
    from qmix_report_writer.qmix import runner
    _attr_or_pending(runner, "_run_eval_episode")


def test_stage4_3_jsonl_log():
    from qmix_report_writer.qmix import runner
    filename = _attr_or_pending(runner, "TRAIN_LOG_FILENAME")
    assert filename.endswith(".jsonl")
    append_fn = _attr_or_pending(runner, "_append_train_log")

    path = os.path.join(tempfile.mkdtemp(), filename)
    append_fn({"episode": 1, "reward": 0.5}, path)
    append_fn({"episode": 2, "reward": 0.7}, path)
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2 and records[1]["episode"] == 2


def test_stage4_4_seeding():
    from qmix_report_writer.qmix import runner
    seed_fn = _attr_or_pending(runner, "_seed_everything")
    seed_fn(123)
    a = torch.rand(3), np.random.rand(3)
    seed_fn(123)
    b = torch.rand(3), np.random.rand(3)
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_stage4_5_checkpoint_v2():
    from qmix_report_writer.qmix.qmix_trainer import QMIXTrainer
    from qmix_report_writer.qmix.replay_buffer import ReplayBuffer, Episode

    if not hasattr(ReplayBuffer, "save"):
        raise Pending("ReplayBuffer has no save/load yet")
    buf = ReplayBuffer(capacity=10)
    ep = Episode()
    ep.add_step(_mk_step_for(3, NUM_ACTIONS, done=True))
    buf.push(ep)
    tmp = os.path.join(tempfile.mkdtemp(), "buffer.pt")
    buf.save(tmp)
    buf2 = ReplayBuffer(capacity=10)
    buf2.load(tmp)
    assert len(buf2) == 1

    save_params = inspect.signature(QMIXTrainer.save).parameters
    if "epsilon" not in save_params:
        raise Pending("trainer.save does not persist epsilon/episode metadata yet")
    trainer = QMIXTrainer(n_agents=3, obs_dim=4, state_dim=15,
                          gnn_hidden_dim=4, gnn_layers=1, rnn_hidden_dim=4,
                          mixing_hidden_dim=4)
    path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
    trainer.save(path, epsilon=0.3, episode_idx=5)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt.get("epsilon") == 0.3 and ckpt.get("episode_idx") == 5


# ---------------------------------------------------------------------------
# Stage 5.2 — config finalization
# ---------------------------------------------------------------------------

def test_stage5_2_config_clean():
    reward_cfg = get_config().get("reward", {}) or {}
    required = ("quality_weight", "macro_weight", "length_weight",
                "token_weight", "length_goal", "length_sigma", "judge")
    missing = [k for k in required if k not in reward_cfg]
    if missing:
        raise Pending(f"reward config v2 incomplete (missing {missing})")
    assert "report_quality_weight" not in reward_cfg, \
        "stale reward.report_quality_weight key must be removed"

    tcfg = _training_cfg()
    missing = [k for k in ("train_steps_per_episode", "min_buffer_episodes",
                           "eval_interval", "persist_buffer") if k not in tcfg]
    if missing:
        raise Pending(f"qmix.training config incomplete (missing {missing})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_all():
    passed = pending = failed = 0
    cases = [
        ("test_stage0_2_benchmark_harness", test_stage0_2_benchmark_harness),
        ("test_stage1_1_schema_reason_first", test_stage1_1_schema_reason_first),
        ("test_stage1_2_judge_discipline", test_stage1_2_judge_discipline),
        ("test_stage1_3_parse_failure_skips", test_stage1_3_parse_failure_skips),
        ("test_stage2_1_evaluation_module", test_stage2_1_evaluation_module),
        ("test_stage2_2_grounded_micro", test_stage2_2_grounded_micro),
        ("test_stage2_3_terminal_macro", test_stage2_3_terminal_macro),
        ("test_stage2_4_reward_composition", test_stage2_4_reward_composition),
        ("test_stage2_5_controller_rewired", test_stage2_5_controller_rewired),
        ("test_stage2_6_runner_wiring", test_stage2_6_runner_wiring),
        ("test_stage3_1_masked_targets", test_stage3_1_masked_targets),
        ("test_stage3_2_terminal_trained", test_stage3_2_terminal_trained),
        ("test_stage3_3_double_dqn", test_stage3_3_double_dqn),
        ("test_stage3_4_no_dropout", test_stage3_4_no_dropout),
        ("test_stage3_5_vectorized_equivalence", test_stage3_5_vectorized_equivalence),
        ("test_stage3_6_in_edge_aggregation", test_stage3_6_in_edge_aggregation),
        ("test_stage4_1_replay_ratio", test_stage4_1_replay_ratio),
        ("test_stage4_2_eval_protocol", test_stage4_2_eval_protocol),
        ("test_stage4_3_jsonl_log", test_stage4_3_jsonl_log),
        ("test_stage4_4_seeding", test_stage4_4_seeding),
        ("test_stage4_5_checkpoint_v2", test_stage4_5_checkpoint_v2),
        ("test_stage5_2_config_clean", test_stage5_2_config_clean),
    ]
    for name, fn in cases:
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                asyncio.run(result)
            print(f"PASS  {name}")
            passed += 1
        except Pending as why:
            print(f"PEND  {name} — {why}")
            pending += 1
        except Exception as exc:
            print(f"FAIL  {name}: {exc}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {pending} pending, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(_run_all())

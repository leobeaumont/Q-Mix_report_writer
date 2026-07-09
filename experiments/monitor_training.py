"""
Training progress monitor for run_qmix_train (reads qmix_train_log.jsonl).

WHAT TO WATCH (in priority order):
  1. EVAL reward trend  -- the single most important signal. Every eval_interval
     episodes a GREEDY (eps=0) episode is scored and NOT used for training, so it
     measures the policy's actual quality. It should TREND UP over the run.
     Flat or declining across the last few evals = the policy is not improving.
  2. aborts             -- with the outline-contract fix these should stay ~0.
     Any abort is a 2-step, reward-0 episode; a cluster of them means either a
     genuinely thin subject or the contract slipping under the exploring policy.
  3. loss               -- should stay finite and bounded (sawtooth is normal at
     each target-net sync). NaN/Inf or a blow-up = optimizer instability.
  4. judge_failures     -- a skipped reward event (LLM judge reply unparseable).
     A few are fine (skip policy handles them); a rising rate corrupts the
     reward signal -- check the judge model / num_ctx / VRAM.
  5. total_reward       -- noisy (eps-exploration), but its rolling mean should
     drift up as eps decays. A collapse toward 0 is a red flag.

Usage (run in a second terminal on the GPU box, from the project root):
    python experiments/monitor_training.py                 # one-shot summary
    python experiments/monitor_training.py --watch 30      # refresh every 30 s
    python experiments/monitor_training.py --log path/to/qmix_train_log.jsonl
    python experiments/monitor_training.py --all           # every run in the file
"""

import os
import sys
import json
import time
import argparse

# Windows consoles default to cp1252 and crash on non-ASCII; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _default_log_path() -> str:
    try:
        from qmix_report_writer.utils.config import get_output_root
        return str(get_output_root() / "qmix_train_log.jsonl")
    except Exception:
        return "qmix_train_log.jsonl"


def _load(path):
    """Return (header, episodes) for the LAST run in the file (records after the
    final header). Malformed/partial trailing lines are skipped."""
    headers, runs, cur = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a half-written trailing line during a live run
            if rec.get("type") == "header":
                if cur or headers:
                    runs.append(cur)
                headers.append(rec)
                cur = []
            elif rec.get("type") == "episode":
                cur.append(rec)
        runs.append(cur)
    return headers, runs


def _spark(values, width=48):
    """Tiny ASCII sparkline for a list of numbers."""
    bars = " .:-=+*#%@"
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    step = max(1, len(vals) // width)
    sampled = vals[::step][-width:]
    return "".join(bars[min(len(bars) - 1, int((v - lo) / rng * (len(bars) - 1)))]
                    for v in sampled)


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"


def summarize(header, eps, run_idx=None, n_runs=None):
    if header:
        cfg = header.get("config", {}).get("qmix", {}).get("training", {})
        tag = f"  (run {run_idx}/{n_runs})" if run_idx else ""
        print("=" * 78)
        print(f"RUN started {header.get('timestamp','?')}{tag} | "
              f"llm={header.get('llm','?')} | seed={header.get('seed')}")
        print(f"  target episodes={header.get('num_episodes','?')} "
              f"start={header.get('start_episode',0)} | "
              f"eps_decay_eps={cfg.get('epsilon_decay_episodes')} "
              f"train_steps/ep={cfg.get('train_steps_per_episode')} "
              f"warmup={cfg.get('min_buffer_episodes')} "
              f"eval_every={cfg.get('eval_interval')}")
        print("=" * 78)
    if not eps:
        print("  (no episodes logged yet)")
        return

    n = len(eps)
    aborts = [e for e in eps if e.get("aborted")]
    total_time = sum(e.get("wall_time_s", 0) for e in eps)
    last = eps[-1]

    # ---- EVAL trend (the key learning signal) --------------------------------
    evals = [(e["episode"], e["eval"]) for e in eps
             if e.get("eval") and e["eval"].get("total_reward") is not None]
    print(f"\nEVAL reward (greedy, eps=0 -- WATCH THIS TREND):")
    if evals:
        for epn, ev in evals[-12:]:
            flag = " ABORTED" if ev.get("aborted") else ""
            print(f"    ep {epn:>4}:  reward={_fmt(ev['total_reward'])}  "
                  f"steps={ev.get('steps')}  judge_fails={ev.get('judge_failures')}{flag}")
        er = [ev["total_reward"] for _, ev in evals]
        print(f"    trend: {_spark(er)}   "
              f"first={_fmt(er[0])} -> last={_fmt(er[-1])}  "
              f"best={_fmt(max(er))}")
    else:
        print("    (no eval episode yet -- first one at "
              f"episode {header.get('config',{}).get('qmix',{}).get('training',{}).get('eval_interval','?') if header else '?'})")

    # ---- Rolling window ------------------------------------------------------
    w = eps[-10:]
    def _mean(key, src=w):
        xs = [e.get(key) for e in src if isinstance(e.get(key), (int, float))]
        return sum(xs) / len(xs) if xs else None
    losses = [l for e in w for l in (e.get("losses") or [])]
    mean_loss = sum(losses) / len(losses) if losses else None
    train_rewards = [e.get("total_reward") for e in eps
                     if isinstance(e.get("total_reward"), (int, float))]

    print(f"\nPROGRESS: {n} episodes logged | wall {total_time/3600:.1f} h | "
          f"eps now={_fmt(last.get('epsilon'))} | "
          f"buffer={last.get('buffer_episodes')} eps | "
          f"train_step={last.get('training_step')}")
    print(f"LAST 10 EPISODES (rolling):")
    print(f"    train reward mean={_fmt(_mean('total_reward'))}  "
          f"steps mean={_fmt(_mean('steps'),1)}  "
          f"tokens mean={_fmt(_mean('tokens'),0)}")
    print(f"    loss mean={_fmt(mean_loss,4)}  "
          f"judge_fails (sum)={sum(e.get('judge_failures',0) for e in w)}  "
          f"aborts={sum(1 for e in w if e.get('aborted'))}")
    print(f"    train-reward trend (all): {_spark(train_rewards)}")

    # ---- Problem detector ----------------------------------------------------
    warns = []
    if aborts:
        subj = {}
        for e in aborts:
            subj[e.get("task","?")] = subj.get(e.get("task","?"), 0) + 1
        top = "; ".join(f"{t[:40]} (x{c})" for t, c in
                        sorted(subj.items(), key=lambda x: -x[1])[:3])
        warns.append(f"{len(aborts)} ABORT(S) -- outline contract slipping or thin "
                     f"subject: {top}")
    if any((l != l) or abs(l) == float("inf") for l in losses):  # NaN/Inf
        warns.append("NON-FINITE LOSS (NaN/Inf) -- optimizer diverged; lower lr.")
    elif len(losses) > 40 and mean_loss and mean_loss > 10 * (
            sum(losses[:20]) / 20):
        warns.append("LOSS BLOW-UP -- last-window loss >>10x early; check lr/grad_clip.")
    jf_rate = sum(e.get("judge_failures", 0) for e in w) / max(len(w), 1)
    if jf_rate > 1.5:
        warns.append(f"HIGH JUDGE-FAILURE RATE (~{jf_rate:.1f}/episode) -- reward "
                     f"signal degrading; check judge model / num_ctx / VRAM.")
    if len(evals) >= 3:
        e3 = [ev["total_reward"] for _, ev in evals[-3:]]
        if e3[-1] <= e3[0] and max(e3) - min(e3) < 0.15:
            warns.append("EVAL REWARD FLAT/DECLINING over last 3 evals -- policy "
                         "may not be learning; consider more exploration or reward review.")
    if train_rewards and len(train_rewards) >= 10:
        recent = sum(train_rewards[-5:]) / 5
        if recent < 0.3:
            warns.append(f"TRAIN REWARD COLLAPSE (last-5 mean={recent:.2f}) -- most "
                         f"episodes near 0; likely aborts or empty reports.")

    print("\nPROBLEM CHECK:")
    if warns:
        for wn in warns:
            print(f"  [!] {wn}")
    else:
        print("  [ok] no problems detected -- aborts 0, loss finite, judges healthy, "
              "eval trend not declining.")
    # ETA
    if header and header.get("num_episodes") and n:
        done = last.get("episode", n)
        remain = header["num_episodes"] - done
        per = total_time / max(n, 1)
        print(f"\n  ~{per/60:.1f} min/episode | {remain} episodes left | "
              f"ETA ~{remain*per/3600:.1f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="path to qmix_train_log.jsonl")
    ap.add_argument("--watch", type=int, default=0, metavar="SECS",
                    help="refresh every N seconds")
    ap.add_argument("--all", action="store_true", help="summarize every run in the file")
    args = ap.parse_args()
    path = args.log or _default_log_path()

    def render():
        if not os.path.exists(path):
            print(f"No log at {path} yet -- waiting for the first episode...")
            return
        headers, runs = _load(path)
        if args.watch:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"[{time.strftime('%H:%M:%S')}]  {path}\n")
        if args.all:
            for i, (h, r) in enumerate(zip(headers, runs), 1):
                summarize(h, r, i, len(headers))
        else:
            h = headers[-1] if headers else None
            r = runs[-1] if runs else []
            summarize(h, r)

    if args.watch:
        try:
            while True:
                render()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n(stopped)")
    else:
        render()


if __name__ == "__main__":
    main()

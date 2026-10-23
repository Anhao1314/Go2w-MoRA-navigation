#!/usr/bin/env python3
"""Unified A/B trainer with convergence early-stop for the sequence task.

One entry point for both groups of the comparison, so they share everything
except the action interface:

- ``--mode single``  (group A): the agent acts in 12-dim joint space
- ``--mode layered`` (group B): the agent acts in 2-dim command space; a
  frozen S0 policy (``--s0-checkpoint``) executes the commands

Training runs in chunks of ``--eval-every`` iterations.  After each chunk the
deterministic completion rate is measured on the training environments
(2048 envs, capped rollout -- see scripts/converge_stop.py for why training
log statistics cannot be used).  After ``--stop-after`` consecutive
measurements at or above the threshold, training stops and the convergence
iteration is recorded.  Every measurement is appended to
``convergence_log.jsonl`` in the run directory.

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/train_ab.py --mode layered \
        --s0-checkpoint logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02/model_1499.pt \
        --num_envs 2048 --max_iterations 1500 --seed 0

Runs only inside the Isaac Lab environment (``.venv311``).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.converge_stop import deterministic_completion_rate  # noqa: E402
from scripts.run_evidence import finalize, snapshot_start  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-Sequence-v0")
    parser.add_argument("--mode", choices=("single", "layered"), default="single")
    parser.add_argument("--s0-checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument(
        "--stop-after", type=int, default=2, help="consecutive passing evaluations before stopping"
    )
    parser.add_argument(
        "--stop-threshold", type=float, default=0.95, help="completion rate that counts as passing"
    )
    parser.add_argument("--eval-max-steps", type=int, default=900)
    args = parser.parse_args(argv)

    if args.mode == "layered" and args.s0_checkpoint is None:
        print("[ab] --mode layered requires --s0-checkpoint")
        return 1
    if args.s0_checkpoint is not None and not args.s0_checkpoint.is_file():
        print(f"[ab] S0 checkpoint not found: {args.s0_checkpoint}")
        return 1

    run_id = f"{args.mode}-{args.task}-seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    pending = snapshot_start(run_id, args.task, args.seed, sys.argv[1:])
    print(f"[evidence] start record: {pending}", flush=True)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import locomotion.isaac  # noqa: F401
    import locomotion.sequence  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    inner = env.unwrapped

    if args.mode == "layered":
        from locomotion.sequence.layered_env import LayeredSequenceEnv, load_s0_actor

        s0_actor = load_s0_actor(args.s0_checkpoint, device=device)
        env = LayeredSequenceEnv(env, s0_actor, device=device)
        print(f"[ab] layered mode, S0 from {args.s0_checkpoint}", flush=True)
    else:
        print("[ab] single mode (12-dim joint actions)", flush=True)

    vec_env = RslRlVecEnvWrapper(env)
    if args.mode == "layered":
        vec_env.num_actions = int(gym.spaces.flatdim(env.action_space))
    print(f"[ab] num_actions: {vec_env.num_actions}", flush=True)

    runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    log_root = PROJECT_ROOT / "logs" / ("rsl_rl_layered" if args.mode == "layered" else "rsl_rl")
    log_dir = log_root / time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ab] log dir: {log_dir}", flush=True)

    runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=str(log_dir), device=device)
    convergence_path = log_dir / "convergence_log.jsonl"

    started = time.time()
    passing = 0
    converged_at: int | None = None
    chunk = max(1, args.eval_every)
    done_iterations = 0

    while done_iterations < args.max_iterations:
        block = min(chunk, args.max_iterations - done_iterations)
        runner.learn(
            num_learning_iterations=block,
            init_at_random_ep_len=(done_iterations == 0),
        )
        done_iterations += block

        policy = runner.get_inference_policy(device=device)
        rate, steps = deterministic_completion_rate(
            vec_env, policy, inner, max_steps=args.eval_max_steps
        )
        passing = passing + 1 if rate >= args.stop_threshold else 0
        record = {
            "iteration": done_iterations,
            "completion_rate": round(rate, 4),
            "eval_steps": steps,
            "passing_streak": passing,
            "wall_time_s": round(time.time() - started, 1),
        }
        with convergence_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[ab] iter {done_iterations:>5}: completion {rate * 100:5.1f}% "
            f"(streak {passing}/{args.stop_after})",
            flush=True,
        )
        if passing >= args.stop_after:
            converged_at = done_iterations
            print(f"[ab] converged at iteration {converged_at}; stopping early", flush=True)
            break

    if converged_at is None:
        print(f"[ab] no early stop; ran all {args.max_iterations} iterations", flush=True)

    vec_env.close()

    summary = {
        "mode": args.mode,
        "converged_at": converged_at,
        "iterations_run": done_iterations,
        "stop_threshold": args.stop_threshold,
        "eval_every": args.eval_every,
        "wall_time_s": round(time.time() - started, 1),
    }
    (log_dir / "convergence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[ab] summary: {json.dumps(summary)}", flush=True)

    try:
        config = finalize(pending, log_dir)
        print(f"[evidence] finalized: {config}", flush=True)
        report = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_report.py"),
                "--run-dir",
                str(log_dir),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if report.returncode == 0:
            print(f"[evidence] report: {(report.stdout or '').strip()}", flush=True)
        else:
            print(
                f"[evidence] report skipped: {(report.stderr or report.stdout or '').strip()[:200]}",
                flush=True,
            )
    except SystemExit as exc:
        print(f"[evidence] finalize skipped: {exc}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

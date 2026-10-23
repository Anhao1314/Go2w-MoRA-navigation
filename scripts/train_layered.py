#!/usr/bin/env python3
"""Train the layered agent (group B): S1 commands, frozen S0 executes.

Difference from ``train_locomotion.py``: the environment is wrapped so the
agent acts in 2-dim command space via :class:`LayeredSequenceEnv`.  Everything
else -- evidence bundle, automatic report, RSL-RL PPO -- matches group A, so
the two runs are directly comparable.

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/train_layered.py \
        --task Go2Loco-Sequence-v0 \
        --s0-checkpoint logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02/model_1499.pt \
        --num_envs 2048 --max_iterations 1500 --seed 0

Exit codes: 0 = trained, 1 = failure.

Runs only inside the Isaac Lab environment (``.venv311``).
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_evidence import finalize, snapshot_start  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-Sequence-v0")
    parser.add_argument("--s0-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.s0_checkpoint.is_file():
        print(f"[layered] S0 checkpoint not found: {args.s0_checkpoint}")
        return 1

    run_id = f"layered-{args.task}-seed{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
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

    from locomotion.sequence.layered_env import LayeredSequenceEnv, load_s0_actor

    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    base_env = gym.make(args.task, cfg=env_cfg)

    print(f"[layered] loading S0 from {args.s0_checkpoint}", flush=True)
    s0_actor = load_s0_actor(args.s0_checkpoint, device=device)
    layered_env = LayeredSequenceEnv(base_env, s0_actor, device=device)

    vec_env = RslRlVecEnvWrapper(layered_env)
    # The official wrapper resolves the action dimension through
    # ``unwrapped.action_manager`` (12 joints).  The layered agent acts in
    # 2-dim command space, so the dimension is overridden here; everything else
    # (TensorDict observations, dones, episode buffer) comes from the official
    # adapter.
    vec_env.num_actions = int(gym.spaces.flatdim(layered_env.action_space))
    print(f"[layered] reported num_actions: {vec_env.num_actions}", flush=True)

    runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    train_cfg = runner_cfg.to_dict()

    log_root = PROJECT_ROOT / "logs" / "rsl_rl_layered"
    log_dir = log_root / time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[layered] action space: {layered_env.action_space}", flush=True)
    print(f"[layered] log dir: {log_dir}", flush=True)
    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=str(log_dir), device=device)
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )

    vec_env.close()

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

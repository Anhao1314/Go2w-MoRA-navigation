#!/usr/bin/env python3
"""Evaluate a trained Go2 policy on the frozen protocol seed set.

Runs one episode per seed in ``locomotion.eval.seeds.EVAL_SEEDS`` (E={0..19}),
on a single environment, and computes the frozen metrics with
``locomotion/eval/metrics.py`` -- never a second, locally invented definition.

Reported per episode: ``vx_rmse``, ``omega_z_rmse``, fall flag, episode length.
Summary: mean/std across seeds plus ``fall_rate``.  The protocol forbids
reporting only the best seed, so the summary is always the full set.

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/eval_policy.py \
        --run-dir logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02 \
        --episodes 20

Exit codes: 0 = evaluated, 1 = failed.

This script runs only inside the Isaac Lab environment (``.venv311``), which CI
does not install, so unresolved isaaclab imports are expected there.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, cast

import numpy as np  # noqa: E402

from locomotion.eval.metrics import fall_rate, vx_rmse, wz_rmse  # noqa: E402
from locomotion.eval.seeds import EVAL_SEEDS  # noqa: E402


def latest_run_dir(root: pathlib.Path) -> pathlib.Path:
    candidates = [p for p in root.glob("*") if (p / "params" / "agent.yaml").is_file()]
    if not candidates:
        raise SystemExit(f"no completed run under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def latest_checkpoint(run_dir: pathlib.Path) -> pathlib.Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise SystemExit(f"no checkpoint in {run_dir}")
    return checkpoints[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-S1-Flat-Blind-v0")
    parser.add_argument("--run-dir", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    episodes = EVAL_SEEDS[: args.episodes]

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import locomotion.isaac  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    run_dir = args.run_dir or latest_run_dir(PROJECT_ROOT / "logs" / "rsl_rl" / "go2_loco_s1_flat")
    checkpoint = args.checkpoint or latest_checkpoint(run_dir)

    results = []
    for seed in episodes:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, use_fabric=True)
        env_cfg.seed = int(seed)
        env = gym.make(args.task, cfg=env_cfg)
        # Isaac Lab's manager-based env exposes attributes gym.Env does not type
        inner = cast(Any, env.unwrapped)
        vec_env = RslRlVecEnvWrapper(env)

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device="cuda:0")

        obs, _ = vec_env.reset()
        dt = float(inner.step_dt)
        commanded = []
        measured = []
        fell = False
        steps = 0
        with torch.inference_mode():
            while steps < 1000:
                cmd = inner.command_manager.get_command("base_velocity")[0].cpu().numpy()
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                vel = inner.scene["robot"].data.root_lin_vel_b[0].cpu().numpy()
                ang = inner.scene["robot"].data.root_ang_vel_b[0].cpu().numpy()
                commanded.append([float(cmd[0]), float(cmd[2])])
                measured.append([float(vel[0]), float(ang[2])])
                steps += 1
                if bool(dones[0]):
                    fell = bool(inner.termination_manager.get_term("base_contact")[0])
                    break

        vec_env.close()  # closes the underlying manager-based env too

        t = np.arange(len(commanded)) * dt
        cmd = np.asarray(commanded)
        mea = np.asarray(measured)
        entry = {
            "seed": int(seed),
            "steps": steps,
            "vx_rmse": vx_rmse(t, cmd[:, 0], mea[:, 0]) if len(t) else None,
            "wz_rmse": wz_rmse(t, cmd[:, 1], mea[:, 1]) if len(t) else None,
            "mean_cmd_vx": float(cmd[:, 0].mean()) if len(cmd) else None,
            "mean_vx": float(mea[:, 0].mean()) if len(mea) else None,
            "fall": fell,
        }
        results.append(entry)
        print(
            f"[eval] seed={seed} steps={steps} vx_rmse={entry['vx_rmse']:.4f} "
            f"wz_rmse={entry['wz_rmse']:.4f} fall={fell}",
            flush=True,
        )

    vx = np.array([r["vx_rmse"] for r in results if r["vx_rmse"] is not None])
    wz = np.array([r["wz_rmse"] for r in results if r["wz_rmse"] is not None])
    falls = [bool(r["fall"]) for r in results]

    summary = {
        "episodes": len(results),
        "seeds": [r["seed"] for r in results],
        "vx_rmse_mean": float(vx.mean()) if vx.size else None,
        "vx_rmse_std": float(vx.std(ddof=0)) if vx.size else None,
        "wz_rmse_mean": float(wz.mean()) if wz.size else None,
        "wz_rmse_std": float(wz.std(ddof=0)) if wz.size else None,
        "fall_rate": fall_rate(falls),
        "checkpoint": str(checkpoint),
        "task": args.task,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    output = args.output or (run_dir / "evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "episodes": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[eval] summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"[eval] wrote {output}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

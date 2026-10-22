#!/usr/bin/env python3
"""Evaluate a trained Go2 policy on the frozen protocol seed set.

Runs one episode per seed in ``locomotion.eval.seeds.EVAL_SEEDS`` (E={0..19})
and computes the frozen metrics with ``locomotion/eval/metrics.py`` -- never a
second, locally invented definition.

Two execution modes:

- default (parallel): 20 environments step together, ``cfg.seed = 0``; each
  environment derives its own command stream from its ``env_id``, so the batch
  is reproducible but is not a literal replay of "one environment with seed i".
- ``--serial``: one environment per seed, rebuilt per episode. Faithful to the
  literal reading, roughly 20x slower because every episode recreates the
  scene.

Both produce the same schema; ``method`` in the JSON records which was used.

Reported per episode: ``vx_rmse``, ``omega_z_rmse``, fall flag, episode length.
Summary: mean/std across seeds plus ``fall_rate``.  The protocol forbids
reporting only the best seed, so the summary is always the full set.

Usage::

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/eval_policy.py \
        --run-dir logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02

Exit codes: 0 = evaluated, 1 = failed.

Runs only inside the Isaac Lab environment (``.venv311``); CI does not install
isaaclab, so unresolved imports are expected there.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, cast

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from locomotion.eval.metrics import fall_rate, vx_rmse, wz_rmse  # noqa: E402
from locomotion.eval.seeds import EVAL_SEEDS  # noqa: E402

EPISODE_STEPS = 1000  # 20 s at 50 Hz, matching the training episode length
TRANSIENT_S = 0.5


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


def _as_bool(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=bool)


def _as_float(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def evaluate_parallel(
    args: argparse.Namespace,
    checkpoint: pathlib.Path,
    seeds: list[int],
    modules: dict[str, Any],
) -> list[dict[str, Any]]:
    """Step all seeds at once; each env derives its own command stream."""
    gym = modules["gym"]
    torch = modules["torch"]
    parse_env_cfg = modules["parse_env_cfg"]
    RslRlVecEnvWrapper = modules["RslRlVecEnvWrapper"]
    OnPolicyRunner = modules["OnPolicyRunner"]
    load_cfg_from_registry = modules["load_cfg_from_registry"]

    num_envs = len(seeds)
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=num_envs, use_fabric=True)
    env_cfg.seed = 0
    env = gym.make(args.task, cfg=env_cfg)
    inner = cast(Any, env.unwrapped)
    vec_env = RslRlVecEnvWrapper(env)

    runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = vec_env.reset()
    dt = float(inner.step_dt)
    steps = EPISODE_STEPS

    cmd_hist = np.zeros((steps, num_envs, 2))
    vel_hist = np.zeros((steps, num_envs, 2))
    done_step = np.full(num_envs, steps, dtype=int)
    fell = np.zeros(num_envs, dtype=bool)
    active = np.ones(num_envs, dtype=bool)

    with torch.inference_mode():
        for step in range(steps):
            command = _as_float(inner.command_manager.get_command("base_velocity"))[:, [0, 2]]
            cmd_hist[step, active] = command[active]
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            lin = _as_float(inner.scene["robot"].data.root_lin_vel_b)[:, 0]
            ang = _as_float(inner.scene["robot"].data.root_ang_vel_b)[:, 2]
            vel_hist[step, active, 0] = lin[active]
            vel_hist[step, active, 1] = ang[active]
            newly_done = _as_bool(dones) & active
            if newly_done.any():
                # timeout fires exactly at the last step; anything earlier is a fall
                fell[newly_done] = step < steps - 1
                done_step[newly_done] = step + 1
                active[newly_done] = False
            if not active.any():
                break

    vec_env.close()

    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        end = int(done_step[index])
        row: dict[str, Any] = {
            "seed": seed,
            "steps": end,
            "fall": bool(fell[index]),
            "vx_rmse": None,
            "wz_rmse": None,
            "mean_cmd_vx": None,
            "mean_vx": None,
        }
        if end * dt > TRANSIENT_S + 0.1:
            time_axis = np.arange(end) * dt
            row["vx_rmse"] = vx_rmse(
                time_axis, cmd_hist[:end, index, 0], vel_hist[:end, index, 0]
            )
            row["wz_rmse"] = wz_rmse(
                time_axis, cmd_hist[:end, index, 1], vel_hist[:end, index, 1]
            )
            row["mean_cmd_vx"] = float(cmd_hist[:end, index, 0].mean())
            row["mean_vx"] = float(vel_hist[:end, index, 0].mean())
        rows.append(row)
        print(
            f"[eval] seed={seed} steps={end} vx_rmse={row['vx_rmse']} "
            f"wz_rmse={row['wz_rmse']} fall={row['fall']}",
            flush=True,
        )
    return rows


def evaluate_serial(
    args: argparse.Namespace,
    checkpoint: pathlib.Path,
    seeds: list[int],
    modules: dict[str, Any],
) -> list[dict[str, Any]]:
    """One environment per seed, rebuilt each episode (faithful, slow)."""
    gym = modules["gym"]
    torch = modules["torch"]
    parse_env_cfg = modules["parse_env_cfg"]
    RslRlVecEnvWrapper = modules["RslRlVecEnvWrapper"]
    OnPolicyRunner = modules["OnPolicyRunner"]
    load_cfg_from_registry = modules["load_cfg_from_registry"]

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, use_fabric=True)
        env_cfg.seed = int(seed)
        env = gym.make(args.task, cfg=env_cfg)
        inner = cast(Any, env.unwrapped)
        vec_env = RslRlVecEnvWrapper(env)
        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device="cuda:0")

        obs, _ = vec_env.reset()
        dt = float(inner.step_dt)
        commands: list[list[float]] = []
        measured: list[list[float]] = []
        fell = False
        steps = 0
        with torch.inference_mode():
            while steps < EPISODE_STEPS:
                command = _as_float(inner.command_manager.get_command("base_velocity"))[0]
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                lin = float(_as_float(inner.scene["robot"].data.root_lin_vel_b)[0, 0])
                ang = float(_as_float(inner.scene["robot"].data.root_ang_vel_b)[0, 2])
                commands.append([float(command[0]), float(command[2])])
                measured.append([lin, ang])
                steps += 1
                if bool(_as_bool(dones)[0]):
                    fell = steps < EPISODE_STEPS
                    break
        vec_env.close()

        time_axis = np.arange(len(commands)) * dt
        cmd_array = np.asarray(commands)
        mea_array = np.asarray(measured)
        row: dict[str, Any] = {
            "seed": seed,
            "steps": steps,
            "fall": bool(fell),
            "vx_rmse": None,
            "wz_rmse": None,
            "mean_cmd_vx": None,
            "mean_vx": None,
        }
        if len(time_axis) * dt > TRANSIENT_S + 0.1:
            row["vx_rmse"] = vx_rmse(time_axis, cmd_array[:, 0], mea_array[:, 0])
            row["wz_rmse"] = wz_rmse(time_axis, cmd_array[:, 1], mea_array[:, 1])
            row["mean_cmd_vx"] = float(cmd_array[:, 0].mean())
            row["mean_vx"] = float(mea_array[:, 0].mean())
        rows.append(row)
        print(
            f"[eval] seed={seed} steps={steps} vx_rmse={row['vx_rmse']} "
            f"wz_rmse={row['wz_rmse']} fall={row['fall']}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-S1-Flat-Blind-v0")
    parser.add_argument("--run-dir", type=pathlib.Path, default=None)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None)
    parser.add_argument("--episodes", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument(
        "--serial",
        action="store_true",
        help="one environment per seed, rebuilt per episode (faithful but ~20x slower)",
    )
    args = parser.parse_args(argv)

    seeds = list(EVAL_SEEDS[: args.episodes])

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import locomotion.isaac  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    modules = {
        "gym": gym,
        "torch": torch,
        "OnPolicyRunner": OnPolicyRunner,
        "RslRlVecEnvWrapper": RslRlVecEnvWrapper,
        "parse_env_cfg": parse_env_cfg,
        "load_cfg_from_registry": load_cfg_from_registry,
    }

    run_dir = args.run_dir or latest_run_dir(PROJECT_ROOT / "logs" / "rsl_rl" / "go2_loco_s1_flat")
    checkpoint = args.checkpoint or latest_checkpoint(run_dir)

    started = time.time()
    if args.serial:
        rows = evaluate_serial(args, checkpoint, seeds, modules)
        method = "serial_per_seed"
    else:
        rows = evaluate_parallel(args, checkpoint, seeds, modules)
        method = "parallel_20_envs_seed0"
    elapsed = time.time() - started

    vx = np.array([row["vx_rmse"] for row in rows if row["vx_rmse"] is not None])
    wz = np.array([row["wz_rmse"] for row in rows if row["wz_rmse"] is not None])
    falls = [bool(row["fall"]) for row in rows]

    summary = {
        "episodes": len(rows),
        "seeds": [row["seed"] for row in rows],
        "vx_rmse_mean": float(vx.mean()) if vx.size else None,
        "vx_rmse_std": float(vx.std(ddof=0)) if vx.size else None,
        "wz_rmse_mean": float(wz.mean()) if wz.size else None,
        "wz_rmse_std": float(wz.std(ddof=0)) if wz.size else None,
        "fall_rate": fall_rate(falls),
        "checkpoint": str(checkpoint),
        "task": args.task,
        "method": method,
        "wall_time_s": round(elapsed, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    output = args.output or (run_dir / "evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[eval] summary: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"[eval] wrote {output}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

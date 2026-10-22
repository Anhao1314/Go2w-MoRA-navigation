#!/usr/bin/env python3
"""Record MP4 videos of a trained Go2 policy in the Isaac Lab scene.

Unlike ``render_training_video.py`` (which replays log metrics on CPU), this
script needs the simulator and a camera, so run it when the GPU is free.

It records a short clip for one or more checkpoints, so the learning
progression can be shown side by side:

    python scripts/render_policy_video.py --run-dir logs/rsl_rl/go2_loco_s1_flat/<stamp> \
        --checkpoints model_100.pt,model_500.pt,model_1000.pt,model_1499.pt \
        --seconds 6

Output: ``media/training/<run>_<checkpoint>.mp4``.

Notes:
- the velocity task has no camera by default, so one is added to the scene
  here (tiled camera, RGB) purely for recording;
- recording is slower than training because rendering enters the loop;
- each clip starts from a fresh reset, uses the seeded command sequence of the
  episode, and is labelled with the checkpoint name.

Exit codes: 0 = recorded, 1 = failure.

Runs only inside the Isaac Lab environment (``.venv311``); CI does not install
isaaclab, so unresolved imports are expected there.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUT = PROJECT_ROOT / "media" / "training"


def latest_run_dir(root: pathlib.Path) -> pathlib.Path:
    candidates = [p for p in root.glob("*") if (p / "params" / "agent.yaml").is_file()]
    if not candidates:
        raise SystemExit(f"no run under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-S1-Flat-Blind-v0")
    parser.add_argument("--run-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--checkpoints",
        default="model_100.pt,model_500.pt,model_1000.pt,model_1499.pt",
        help="comma-separated checkpoint names inside the run directory",
    )
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    run_dir = args.run_dir or latest_run_dir(PROJECT_ROOT / "logs" / "rsl_rl" / "go2_loco_s1_flat")
    requested = [name.strip() for name in args.checkpoints.split(",") if name.strip()]
    checkpoints = [run_dir / name for name in requested if (run_dir / name).is_file()]
    if not checkpoints:
        available = sorted(p.name for p in run_dir.glob("model_*.pt"))
        raise SystemExit(f"none of the requested checkpoints exist in {run_dir}; have: {available}")

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    from typing import Any, cast

    import gymnasium as gym
    import torch

    import locomotion.isaac  # noqa: F401
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recorded: list[pathlib.Path] = []

    for checkpoint in checkpoints:
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
        env_cfg.seed = args.seed
        # recording-only camera; the task itself stays blind
        env_cfg.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(2.6, -2.6, 1.7), rot=(0.84, 0.22, 0.22, 0.44), convention="ros"
            ),
            data_types=["rgb"],
            width=960,
            height=540,
            update_period=0.0,
        )
        env = gym.make(args.task, cfg=env_cfg)
        prefix = f"{run_dir.name}_{checkpoint.stem}"
        # RecordVideo wraps the gym env; RSL-RL wraps the recording env. This
        # keeps the 5-tuple gym API inside and the 4-tuple RSL-RL API outside.
        recording_env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.out_dir),
            name_prefix=prefix,
            episode_trigger=lambda episode: episode == 0,
            disable_logger=True,
        )
        vec_env = RslRlVecEnvWrapper(recording_env)
        runner_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(vec_env, runner_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device="cuda:0")

        inner = cast(Any, env.unwrapped)
        steps = max(1, int(args.seconds / float(inner.step_dt)))
        obs, _ = vec_env.reset()

        with torch.inference_mode():
            for _ in range(steps):
                actions = policy(obs)
                obs, _, dones, _ = vec_env.step(actions)
                if bool(dones[0]):
                    break
        recording_env.close()

        produced = sorted(args.out_dir.glob(f"{prefix}*.mp4"), key=lambda p: p.stat().st_mtime)
        if produced:
            recorded.append(produced[-1])
            print(f"[video] recorded {checkpoint.name} -> {produced[-1]}", flush=True)
        else:
            print(f"[video] WARNING: no file produced for {checkpoint.name}", flush=True)

    simulation_app.close()
    for path in recorded:
        if not path.exists():
            print(f"[video] WARNING: expected file missing: {path}")
    print(f"[video] done: {len(recorded)} clip(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

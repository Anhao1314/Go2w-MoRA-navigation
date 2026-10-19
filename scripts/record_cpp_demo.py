#!/usr/bin/env python3
"""Record a multi-segment demo driven by the C++ (or scripted) System 0.

The renderer uses the environment's own MjModel instance so runtime scenario
mutations (walls/terrain) are reflected in the video.

Usage:
    python scripts/record_cpp_demo.py --s0 cpp --seed 1000
    python scripts/record_cpp_demo.py --s0 scripted --out media/scripted_demo
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import mujoco
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_demos import common  # noqa: E402
from scripts.train_multi_segment_curriculum import make_stage_env  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0", choices=("cpp", "scripted"), default="cpp")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default="media/cpp_stack_p0_demo")
    parser.add_argument("--camera", default="overview")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-steps", type=int, default=4000)
    args = parser.parse_args()

    env = make_stage_env(2, False, s0=args.s0, pre_brake=True)
    renderer = mujoco.Renderer(env.base_env.model, args.height, args.width)
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=args.seed)
        action = np.array([1.0, 0.0], dtype=np.float32)
        steps = 0
        while True:
            obs, _reward, term, trunc, info = env.step(action)
            steps += 1
            if steps % 2 == 0:
                renderer.update_scene(env.base_env.data, camera=args.camera)
                frames.append(renderer.render().copy())
            if term or trunc or steps >= args.max_steps:
                break
        summary = {
            "s0": args.s0,
            "seed": args.seed,
            "steps": steps,
            "sim_time_s": round(float(env.base_env.data.time), 4),
            "x": round(float(info.get("x", 0.0)), 6),
            "goal": bool(info.get("goal")),
            "passed_subgoal": bool(info.get("passed_subgoal")),
            "frames": len(frames),
        }
    finally:
        renderer.close()
        env.close()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    common.save_mp4(frames, out.with_suffix(".mp4"), args.fps)
    common.save_gif(frames, out.with_suffix(".gif"), args.fps)
    print(json.dumps({**summary, "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

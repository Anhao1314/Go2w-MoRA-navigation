"""Junction 岔路口脚本控制器演示轨迹生成（A 左 / B 右，各 10 条）。"""

from __future__ import annotations

import math
import pathlib
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv  # noqa: E402
from rl.high_level_env_wrapper import junction_teacher_bias  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "demo_trajectories"
MAX_STEPS = 2000


def run_episode(seed: int, target: str) -> dict:
    env = Go2wEnv(
        task="traverse_curve", domain_randomize=False,
        junction=True, reward_version="junction", scope=2, target_goal=target,
    )
    obs, info = env.reset(seed=seed)
    rec: dict[str, Any] = {
        "obs": [], "actions": [], "positions": [], "yaws": [],
        "branches": [], "speeds": [], "times": [],
    }
    goal = False
    term = trunc = False
    for _ in range(MAX_STEPS):
        rec["obs"].append(np.asarray(obs, dtype=np.float32))
        x = float(env.data.body("base_link").xpos[0])
        y = float(env.data.body("base_link").xpos[1])
        yaw = float(env._yaw_of())
        cfg = env.junction_config
        if (
            env.get_branch_selected() is None
            and cfg["decision_x_start"] <= x <= cfg["decision_x_end"]
        ):
            env.set_branch_selected("left" if target == "A" else "right")
        bias = junction_teacher_bias(x, y, yaw, env)
        fwd = 0.12
        action = np.array(
            [fwd - bias, fwd + bias, fwd - bias, fwd + bias, 0.0, 0.0],
            dtype=np.float32,
        )
        rec["actions"].append(action)
        rec["positions"].append((x, y))
        rec["yaws"].append(yaw)
        rec["branches"].append(env.get_branch_selected())
        rec["speeds"].append(1.0)
        rec["times"].append(float(env.data.time))
        obs, _r, term, trunc, info = env.step(action)
        if info.get("goal"):
            goal = True
            break
        if term or trunc:
            break
    rec.update({
        "goal": goal,
        "final_x": float(env.data.body("base_link").xpos[0]),
        "final_y": float(env.data.body("base_link").xpos[1]),
        "branch": env.get_branch_selected(),
    })
    env.close()
    return rec


def main() -> None:
    count = 0
    for target, seeds in (("A", range(0, 10)), ("B", range(10, 20))):
        for seed in seeds:
            rec = run_episode(seed, target)
            meta = {"seed": seed, "target": target, "goal": rec["goal"],
                    "branch": rec["branch"],
                    "final_x": round(rec["final_x"], 3),
                    "final_y": round(rec["final_y"], 3)}
            path = OUT_DIR / f"junction_demo_seed{seed:02d}.npz"
            np.savez_compressed(
                path,
                obs=np.array(rec["obs"], dtype=np.float32),
                actions=np.array(rec["actions"], dtype=np.float32),
                positions=np.array(rec["positions"], dtype=np.float32),
                yaws=np.array(rec["yaws"], dtype=np.float32),
                branches=np.array(
                    [1.0 if b == "left" else 0.0 for b in rec["branches"]],
                    dtype=np.float32,
                ),
                speeds=np.array(rec["speeds"], dtype=np.float32),
                times=np.array(rec["times"], dtype=np.float32),
                meta=np.array([str(meta)], dtype=object),
            )
            count += int(rec["goal"])
            print(f"seed{seed:02d} target={target} goal={rec['goal']} "
                  f"steps={len(rec['obs'])}")
    print(f"成功 {count}/20")


if __name__ == "__main__":
    main()

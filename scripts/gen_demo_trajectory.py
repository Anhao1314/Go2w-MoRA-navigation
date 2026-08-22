"""生成第一条 BC 演示轨迹：脚本控制器跑通 traverse_curve 弯道。

控制器结构（初始四轮模式，若跑不通再启用双轮逻辑）：
  1. 每步取 x/y/yaw，计算航向误差 herr 与横向偏差 dev；
  2. 差速转向：bias = K_YAW*herr - K_DEV*dev，限幅 ±BIAS_CLIP；
  3. 前进：基础力矩 F，progress_rate 超速时减半；
  4. 动作 [F-bias, F+bias, F-bias, F+bias, 0, 0]（左轮加速=右转）；
  5. 失败时按原因调参重试（最多 10 次）。

输出：data/demo_trajectories/curve_demo_seed0.npz（成功）或 failed/ 目录（失败）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv, curve_center_y, heading_error, path_tangent_angle

OUT_DIR = PROJECT_ROOT / "data" / "demo_trajectories"
MAX_ATTEMPTS = 10
MAX_STEPS = 1500


def default_params() -> dict:
    return {
        "k_ang": 2.5,        # 目标航向误差转向增益（纯跟踪）
        "k_dev": 0.5,        # 横向偏差转向增益
        "bias_clip": 1.0,    # 差速 bias 限幅（归一化动作，允许内侧轮反转急转）
        "forward": 0.12,     # 基础前进力矩（归一化，×15Nm），四轮全驱
        "rear_only": False,  # 四轮全驱（实测比只驱后轮更能过弯）
        "lookahead": 0.7,    # 纯跟踪前视距离（米，0.7 在干净环境下可跑通全程）
        "speed_cap": 5.0,    # 速度上限 m/s（演示不限速，BC 数据后续可过滤）
    }


def compute_teacher_bias(
    x: float, y: float, yaw: float, params: dict | None = None
) -> float:
    """纯跟踪基础转向 bias（前视点 + 横向偏差），供高层控制器叠加调整量。"""
    p = params or default_params()
    xt = x + float(p["lookahead"])
    yt = float(curve_center_y(xt))
    desired_yaw = float(np.arctan2(yt - y, xt - x))
    ang_err = float(((desired_yaw - yaw + np.pi) % (2 * np.pi)) - np.pi)
    dev = float(y - float(curve_center_y(x)))
    bias = float(
        np.clip(
            -float(p["k_ang"]) * ang_err + float(p["k_dev"]) * dev,
            -float(p["bias_clip"]),
            float(p["bias_clip"]),
        )
    )
    return bias


def compute_teacher_action(
    x: float, y: float, yaw: float, params: dict | None = None
) -> np.ndarray:
    """纯跟踪教师动作（与演示轨迹控制逻辑完全一致，无噪声）。

    动作轮序 [FR, FL, RR, RL]：左轮(FL/RL)加 bias → 右转。
    """
    p = params or default_params()
    bias = compute_teacher_bias(x, y, yaw, p)
    fwd = float(p["forward"])
    front = fwd if not p.get("rear_only", True) else 0.0
    action = np.array(
        [front - bias, front + bias, fwd - bias, fwd + bias, 0.0, 0.0],
        dtype=np.float32,
    )
    return np.clip(action, -1.0, 1.0)


def run_episode(env: Go2wEnv, params: dict, seed: int,
                rng: np.random.Generator) -> dict:
    """跑一次 episode，返回轨迹数据与结果。"""
    obs, info = env.reset(seed=seed)
    rec: dict[str, Any] = {
        "obs": [], "actions": [], "rewards": [], "dones": [],
        "positions": [], "yaws": [], "heading_errors": [],
        "lateral_deviations": [], "times": [],
    }
    prev_x = float(info["x"])
    prev_y = float(info["y"])
    for _ in range(MAX_STEPS):
        # BC 对齐：先记录决策前 obs，再执行动作
        rec["obs"].append(obs.astype(np.float32))
        x = float(env.data.body("base_link").xpos[0])
        y = float(env.data.body("base_link").xpos[1])
        yaw = float(env._yaw_of())
        herr = float(heading_error(x, yaw, is_curve=True))
        dev = float(y - float(curve_center_y(x)))

        # 前进速度（世界 x 位移/策略步 → m/s）
        vx = (x - prev_x) / 0.01
        F = params["forward"]
        if vx > params["speed_cap"]:
            F *= 0.5

        # 纯跟踪教师动作（与 DAgger 共用同一实现）
        teacher_params = dict(params)
        teacher_params["forward"] = F
        action = compute_teacher_action(x, y, yaw, teacher_params)
        # 每个 seed 用独立噪声制造轨迹多样性（BC 需要多种状态-动作样本）
        noise = rng.normal(0.0, params.get("noise", 0.03), size=6).astype(np.float32)
        action = np.clip(action + noise, -1.0, 1.0)

        obs, reward, term, trunc, info = env.step(action)
        rec["actions"].append(action)
        rec["rewards"].append(float(reward))
        rec["dones"].append(bool(term or trunc))
        rec["positions"].append((x, y))
        rec["yaws"].append(yaw)
        rec["heading_errors"].append(herr)
        rec["lateral_deviations"].append(dev)
        rec["times"].append(float(env.data.time))
        prev_x, prev_y = x, y

        if term or trunc:
            break

    rec["goal"] = bool(info.get("goal", False))
    rec["steps"] = len(rec["actions"])
    rec["final_x"] = float(env.data.body("base_link").xpos[0])
    rec["final_y"] = float(env.data.body("base_link").xpos[1])
    return rec


def diagnose(rec: dict, params: dict) -> str:
    """判断失败原因，用于调参。"""
    if rec["goal"]:
        return "success"
    devs = rec["lateral_deviations"]
    xs = [p[0] for p in rec["positions"]]
    if any(abs(d) > 0.40 and 0.8 <= xs[i] <= 4.5 for i, d in enumerate(devs)):
        return "wall"        # 撞墙/越界
    if xs and max(xs) - xs[0] < 0.1:
        return "stuck"       # 卡住不动
    return "fell"            # 摔倒/其他终止


def adjust_params(params: dict, reason: str) -> dict:
    p = dict(params)
    if reason == "fell":
        p["forward"] = round(max(0.05, p["forward"] - 0.05), 2)
    elif reason == "wall":
        p["k_ang"] = min(6.0, p["k_ang"] + 1.0)
        p["k_dev"] = min(3.0, p["k_dev"] + 0.5)
    elif reason == "stuck":
        p["forward"] = round(min(0.8, p["forward"] + 0.1), 2)
    else:
        p["k_ang"] = min(6.0, p["k_ang"] + 1.0)
        p["bias_clip"] = min(0.6, p["bias_clip"] + 0.1)
    return p


def save_npz(path: pathlib.Path, rec: dict, params: dict, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "steps": rec["steps"],
        "duration_s": round(rec["times"][-1], 3) if rec["times"] else 0.0,
        "goal": rec["goal"],
        "final_x": round(rec["final_x"], 3),
        "final_y": round(rec["final_y"], 3),
        "params": params,
        "seed": seed,
    }
    np.savez_compressed(
        path,
        obs=np.array(rec["obs"], dtype=np.float32),
        actions=np.array(rec["actions"], dtype=np.float32),
        rewards=np.array(rec["rewards"], dtype=np.float32),
        dones=np.array(rec["dones"], dtype=bool),
        positions=np.array(rec["positions"], dtype=np.float32),
        yaws=np.array(rec["yaws"], dtype=np.float32),
        heading_errors=np.array(rec["heading_errors"], dtype=np.float32),
        lateral_deviations=np.array(rec["lateral_deviations"], dtype=np.float32),
        times=np.array(rec["times"], dtype=np.float32),
        meta=np.array([str(meta)], dtype=object),
    )


def summarize(rec: dict) -> dict:
    xs = [p[0] for p in rec["positions"]]
    vels = [abs((b - a) / 0.01) for a, b in zip(xs[:-1], xs[1:])] if len(xs) > 1 else [0.0]
    devs = rec["lateral_deviations"]
    hers = rec["heading_errors"]
    acts = rec["actions"]
    smooth = (
        float(np.mean(np.abs(np.diff(acts, axis=0))))
        if len(acts) > 1 else 0.0
    )
    return {
        "steps": rec["steps"],
        "duration_s": round(rec["times"][-1], 3),
        "mean_speed": round(float(np.mean(vels)), 3),
        "max_speed": round(float(np.max(vels)), 3),
        "mean_dev": round(float(np.mean(np.abs(devs))), 3),
        "max_dev": round(float(np.max(np.abs(devs))), 3),
        "mean_herr": round(float(np.mean(np.abs(hers))), 3),
        "max_herr": round(float(np.max(np.abs(hers))), 3),
        "in_corridor": bool(max(np.abs(devs)) < 0.40),
        "action_smoothness": round(smooth, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 BC 演示轨迹（脚本控制器）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.03,
                        help="动作高斯噪声标准差（制造轨迹多样性）")
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    args = parser.parse_args()

    env = Go2wEnv(task="traverse_curve", domain_randomize=False)
    params = default_params()
    params["noise"] = args.noise
    rng = np.random.default_rng(args.seed)
    attempts: list[dict] = []
    success_path = OUT_DIR / f"curve_demo_seed{args.seed:02d}.npz"

    for i in range(1, args.attempts + 1):
        rec = run_episode(env, params, args.seed, rng)
        reason = diagnose(rec, params)
        attempts.append({"attempt": i, "reason": reason, "params": dict(params),
                         "steps": rec["steps"], "final_x": rec["final_x"]})
        print(f"[{i}/{args.attempts}] reason={reason} steps={rec['steps']} "
              f"x={rec['final_x']:.2f} params={params}")
        if reason == "success":
            save_npz(success_path, rec, params, args.seed)
            print("成功轨迹已保存:", success_path)
            print("质量评估:", summarize(rec))
            break
        save_npz(OUT_DIR / "failed" / f"curve_attempt{i:02d}.npz", rec, params, args.seed)
        params = adjust_params(params, reason)
    else:
        print(f"10 次尝试均失败。失败原因统计: "
              f"{ {r['reason'] for r in attempts} }")
        print("已保存全部失败轨迹到", OUT_DIR / "failed")

    env.close()


if __name__ == "__main__":
    main()

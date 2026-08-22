"""评估 RL 策略与手写控制器，输出验收指标、对比表与演示视频。

用法示例：
    python rl/eval.py --task balance --model rl/runs/balance/seed00/best_model.zip \
        --vec-norm rl/runs/balance/seed00/best_vec_normalize.pkl --compare
    python rl/eval.py --task balance --model ... --disturbance --record media/rl_balance
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_demos import common
from rl.go2w_env import (
    Go2wEnv,
    MODEL_PATH,
    MODEL_SCENARIO_PATH,
    PITCH_REF,
    SCENARIOS,
    TASK_SCENARIO,
    TRAVERSE_TASKS,
    WHEEL_RADIUS,
    init_four_wheel_pose,
    init_two_wheel_pose,
    pitch_of,
)
from scripts.demo_go2w import Go2wBalanceController

POLICY_DT = 0.01


def base_reward(entry: dict, prev_action: np.ndarray) -> float:
    a = np.asarray(entry["action"], dtype=float)
    return (
        1.0
        - 2.0 * entry["pitch_err"] ** 2
        - 0.2 * entry["pitch_rate"] ** 2
        - 0.01 * float(np.dot(a, a))
        - 0.02 * float(np.dot(a - prev_action, a - prev_action))
    )


def summarize(log: list[dict], task: str) -> dict:
    nan = any(e["nan"] for e in log)
    total_reward = float(sum(e["reward"] for e in log))
    base_rews = [base_reward(log[0], np.zeros_like(np.asarray(log[0]["action"])))]
    for prev, cur in zip(log[:-1], log[1:]):
        base_rews.append(base_reward(cur, np.asarray(prev["action"])))
    mean_base = float(np.mean(base_rews)) if log else float("nan")

    if task == "balance":
        # Balance v2：四轮站姿 + 撞击干扰，验收 |pitch|≤0.35、base_z≥0.30
        devs = [abs(e["pitch"] - e["pitch_ref"]) for e in log]
        clears = [e["base_z"] for e in log]
        max_dev = float(max(devs)) if log else float("inf")
        min_clear = float(min(clears)) if log else float("-inf")
        hold = (
            sum(
                1
                for e in log
                if abs(e["pitch"] - e["pitch_ref"]) <= 0.35 and e["base_z"] >= 0.30
            )
            * POLICY_DT
        )
        success = (not nan) and max_dev <= 0.35 and min_clear >= 0.30
        return {
            "task": task,
            "max_dev": max_dev,
            "min_clear": min_clear,
            "dual_hold": hold,
            "recovered": None,
            "settle_seconds": None,
            "success": success,
            "total_reward": total_reward,
            "mean_base_reward": mean_base,
            "nan": nan,
            "distance": None,
            "time_to_goal": None,
            "falls": 0,
            "backward_dist": None,
            "backward_frac": None,
        }

    if task in TRAVERSE_TASKS:
        success = bool(log) and log[-1]["goal"] and not nan
        distance = float(max((e.get("progress", e["x"]) for e in log), default=0.0))
        xs = [e.get("progress", e["x"]) for e in log]
        backward_dist = float(max(0.0, max(xs, default=0.0) - xs[-1])) if xs else 0.0
        backward_frac = (
            float(
                sum(
                    1
                    for prev, cur in zip(xs[:-1], xs[1:])
                    if cur < prev
                )
                / max(1, len(xs) - 1)
            )
            if xs
            else 0.0
        )
        time_to_goal = next((e["t"] for e in log if e["goal"]), None)
        max_dev = float(max((abs(e["pitch_err"]) for e in log), default=0.0))
        two = [e for e in log if e["stage"] == 2]
        min_clear_two = (
            float(min(e["front_clearance"] for e in two)) if two else None
        )
        dual_hold = (
            sum(
                1
                for e in two
                if abs(e["pitch_err"]) <= 0.1 and e["front_clearance"] >= 0.10
            )
            * POLICY_DT
        )
        return {
            "task": task,
            "max_dev": max_dev,
            "min_clear": min_clear_two,
            "dual_hold": dual_hold,
            "recovered": None,
            "settle_seconds": None,
            "success": success,
            "total_reward": total_reward,
            "mean_base_reward": mean_base,
            "nan": nan,
            "distance": distance,
            "time_to_goal": time_to_goal,
            "falls": (
                0
                if success
                else (1 if log[-1]["term"] and not log[-1]["trunc"] else 0)
            ),
            "backward_dist": backward_dist,
            "backward_frac": backward_frac,
        }

    two = [e for e in log if e["stage"] == 2]
    max_dev_two = max((abs(e["pitch_err"]) for e in two), default=float("inf"))
    min_clear_two = min((e["front_clearance"] for e in two), default=float("-inf"))
    dual_hold = (
        sum(
            1
            for e in two
            if abs(e["pitch_err"]) <= 0.1 and e["front_clearance"] >= 0.10
        )
        * POLICY_DT
    )
    settled = [e for e in log if e["phi"] >= 2.0]
    recovered = len(settled) > 0
    settle_seconds = (
        sum(1 for e in settled if abs(e["pitch"]) <= 0.05) * POLICY_DT
        if recovered
        else 0.0
    )
    success = (
        (not nan)
        and recovered
        and settle_seconds >= 2.0
        and dual_hold >= 5.0
        and max_dev_two <= 0.1
        and min_clear_two >= 0.10
    )
    return {
        "task": task,
        "max_dev": max_dev_two,
        "min_clear": min_clear_two,
        "dual_hold": dual_hold,
        "recovered": recovered,
        "settle_seconds": settle_seconds,
        "success": success,
        "total_reward": total_reward,
        "mean_base_reward": mean_base,
        "nan": nan,
        "distance": None,
        "time_to_goal": None,
        "falls": 0,
        "backward_dist": None,
        "backward_frac": None,
    }


def aggregate(summaries: list[dict]) -> dict:
    def mean_or_none(values):
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    min_clears = [s["min_clear"] for s in summaries if s["min_clear"] is not None]
    return {
        "task": summaries[0]["task"],
        "max_dev": float(np.max([s["max_dev"] for s in summaries])),
        "min_clear": float(np.min(min_clears)) if min_clears else None,
        "dual_hold": float(np.mean([s["dual_hold"] for s in summaries])),
        "recovered": bool(summaries) and all(s["recovered"] is True for s in summaries),
        "settle_seconds": mean_or_none([s["settle_seconds"] for s in summaries]),
        "success": all(s["success"] for s in summaries),
        "success_rate": float(np.mean([s["success"] for s in summaries])),
        "total_reward": float(np.mean([s["total_reward"] for s in summaries])),
        "mean_base_reward": float(np.mean([s["mean_base_reward"] for s in summaries])),
        "nan": any(s["nan"] for s in summaries),
        "distance": mean_or_none([s["distance"] for s in summaries]),
        "time_to_goal": mean_or_none([s["time_to_goal"] for s in summaries]),
        "falls": float(np.mean([s["falls"] for s in summaries])),
        "backward_dist": mean_or_none([s["backward_dist"] for s in summaries]),
        "backward_frac": mean_or_none([s["backward_frac"] for s in summaries]),
    }


def load_policy(model_path: str, vec_norm_path: str, task: str,
                terrain: str = "hfield"):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = Go2wEnv(task=task, domain_randomize=False, terrain=terrain)
    vec_env = DummyVecEnv([lambda: env])
    vec_norm = VecNormalize(vec_env, training=False, norm_obs=True,
                            norm_reward=False, clip_obs=10.0)
    vec_norm = VecNormalize.load(str(vec_norm_path), vec_env)
    vec_norm.training = False
    model = PPO.load(str(model_path), device="cpu")
    return model, env, vec_norm


def run_rl_episode(
    model: PPO,
    env: Go2wEnv,
    vec_norm,
    record: bool = False,
    renderer: mujoco.Renderer | None = None,
    camera: str = "track",
) -> tuple[list[dict], list[np.ndarray]]:
    obs, _ = env.reset()
    obs = vec_norm.normalize_obs(obs)
    log: list[dict] = []
    frames: list[np.ndarray] = []
    step_i = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = vec_norm.normalize_obs(obs)
        entry = {
            "t": env.data.time,
            "pitch": info["pitch"],
            "pitch_ref": float(info.get("pitch_ref", 0.0)),
            "pitch_err": info["pitch_err"],
            "pitch_rate": info["pitch_rate"],
            "front_clearance": info["front_clearance"],
            "base_z": float(info.get("base_z", 0.0)),
            "progress": float(info.get("progress", info.get("x", 0.0))),
            "stage": info["stage"],
            "phi": info["phi"],
            "nan": info["nan"],
            "scenario": info.get("scenario"),
            "x": float(info.get("x", 0.0)),
            "y": float(info.get("y", 0.0)),
            "vx": float(info.get("vx", 0.0)),
            "goal": bool(info.get("goal", False)),
            "action": np.asarray(action, dtype=float),
            "reward": float(reward),
            "term": bool(terminated),
            "trunc": bool(truncated),
        }
        log.append(entry)
        if record and renderer is not None and step_i % 2 == 0:
            renderer.update_scene(env.data, camera=camera)
            frames.append(renderer.render().copy())
        step_i += 1
        if terminated or truncated:
            break
    return log, frames


def controller_action(data: mujoco.MjData) -> np.ndarray:
    """把现有 Go2wBalanceController 的平衡力矩换算成 env 动作 [-1,1]。"""
    front = (
        min(data.body("FL_wheel_link").xpos[2], data.body("FR_wheel_link").xpos[2])
        - WHEEL_RADIUS
    )
    if front > 0.02:
        p = pitch_of(data)
        pd = data.qvel[4]
        w = 0.5 * (data.qvel[21] + data.qvel[17])
        tau = -100.0 * (p - PITCH_REF) - 25.0 * pd + 8.0 * (0.0 - w * WHEEL_RADIUS)
        tau = max(-30.0, min(30.0, tau)) / 2.0
    else:
        tau = 8.0 * (0.0 - 0.5 * (data.qvel[21] + data.qvel[17])) / 2.0
    return np.clip(np.array([tau, tau]) / 15.0, -1.0, 1.0)


def run_controller_episode(
    task: str,
    duration: float,
    record: bool = False,
    renderer: mujoco.Renderer | None = None,
    camera: str = "track",
) -> tuple[list[dict], list[np.ndarray]]:
    model = common.load_model(MODEL_PATH)
    data = mujoco.MjData(model)
    controller = Go2wBalanceController(model)
    if task == "balance":
        init_two_wheel_pose(model, data)
        data.time = 3.0  # 跳过控制器前 1s 的四轮逻辑，直接进入双轮保持
    else:
        init_four_wheel_pose(model, data)
    mujoco.mj_forward(model, data)

    log: list[dict] = []
    frames: list[np.ndarray] = []
    step_i = 0
    prev_action = np.zeros(2)
    n_phys = int(duration / model.opt.timestep)
    for i in range(n_phys):
        controller.step(data, data.time)
        mujoco.mj_step(model, data)
        if i % 5 == 0:
            t = data.time
            pitch = pitch_of(data)
            clearance = (
                min(data.body("FL_wheel_link").xpos[2], data.body("FR_wheel_link").xpos[2])
                - WHEEL_RADIUS
            )
            if task == "balance":
                stage, phi = 2, 1.0
            elif t < 1.0:
                stage, phi = 0, 0.0
            elif t < 2.5:
                stage, phi = 1, (t - 1.0) / 1.5
            else:
                stage, phi = 2, 1.0
            action = controller_action(data)
            entry = {
                "t": t,
                "pitch": pitch,
                "pitch_err": pitch - (PITCH_REF if stage in (1, 2) else 0.0),
                "pitch_rate": float(data.qvel[4]),
                "front_clearance": clearance,
                "stage": stage,
                "phi": phi,
                "nan": bool(np.isnan(data.qpos).any() or np.isnan(data.qvel).any()),
                "action": action,
                "reward": 0.0,
            }
            entry["reward"] = base_reward(entry, prev_action)
            prev_action = action
            log.append(entry)
            if record and renderer is not None and step_i % 2 == 0:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
            step_i += 1
    return log, frames


def save_video(frames: list[np.ndarray], prefix: pathlib.Path, fps: int = 50) -> None:
    common.save_mp4(frames, prefix.with_suffix(".mp4"), fps)
    common.save_gif(frames, prefix.with_suffix(".gif"), fps)
    print(f"已生成: {prefix.with_suffix('.mp4')} / {prefix.with_suffix('.gif')}")


def print_report(label: str, summary: dict) -> None:
    print(f"\n== {label} ==")
    if summary["task"] in TRAVERSE_TASKS:
        print(f"  成功率: {summary['success_rate'] * 100:.0f}%")
        print(f"  平均穿越距离: {summary['distance']:.2f} m")
        time_str = f"{summary['time_to_goal']:.2f} s" if summary["time_to_goal"] is not None else "未到达"
        print(f"  平均到达时间: {time_str}")
        print(f"  最大俯仰偏差: {summary['max_dev']:.4f} rad")
        print(f"  平均摔倒次数: {summary['falls']:.2f}")
        if summary["backward_dist"] is not None:
            print(
                f"  平均后退距离: {summary['backward_dist']:.2f} m"
                f"   后退占比: {summary['backward_frac'] * 100:.1f}%"
            )
    elif summary["task"] == "balance":
        print(f"  最大俯仰偏差: {summary['max_dev']:.4f} rad")
        print(f"  最小基座高度: {summary['min_clear']:.4f} m")
        print(f"  抗撞保持(偏差≤0.35 & 高度≥0.30): {summary['dual_hold']:.2f} s")
    else:
        print(f"  最大俯仰偏差(双轮期): {summary['max_dev']:.4f} rad")
        print(f"  最小前轮离地: {summary['min_clear']:.4f} m")
        print(f"  双轮保持(偏差≤0.1 & 离地≥0.1): {summary['dual_hold']:.2f} s")
    if summary["task"] == "full_chain":
        print(f"  四轮恢复成功: {summary['recovered']}")
        print(f"  恢复后稳定( |pitch|≤0.05 )时长: {summary['settle_seconds']:.2f} s")
    print(f"  验收通过: {summary['success']}")
    print(f"  总奖励: {summary['total_reward']:.2f}  平均基础奖励: {summary['mean_base_reward']:.4f}")
    if summary["nan"]:
        print("  警告: 出现 NaN")


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2w RL / 手写控制器评估")
    parser.add_argument(
        "--task",
        choices=["balance", "full_chain", "full_chain_simple", *TRAVERSE_TASKS],
        default="balance",
    )
    parser.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all",
                        help="traverse 任务评估场景覆盖，默认取任务自身场景")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--vec-norm", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--compare", action="store_true",
                        help="同协议运行手写控制器并输出对比")
    parser.add_argument("--disturbance", action="store_true",
                        help="运行 40N/0.2s 扰动恢复测试（RL）")
    parser.add_argument("--terrain", choices=["hfield", "boxes"], default="hfield",
                        help="traverse 斜坡任务地形：hfield=运行时高度场（默认），boxes=旧盒状斜坡")
    parser.add_argument("--camera", default="track",
                        help="录制视频使用的相机（默认 track 跟随；overview 为全局俯视）")
    parser.add_argument("--record", type=str, default=None,
                        help="输出视频前缀，如 media/rl_balance")
    parser.add_argument("--report", type=str, default=None,
                        help="把对比表写入 CSV")
    args = parser.parse_args()

    duration = args.duration or (10.0 if args.task == "balance" else 15.0)
    renderer = (
        mujoco.Renderer(
            common.load_model(
                MODEL_SCENARIO_PATH if args.task in TRAVERSE_TASKS else MODEL_PATH
            ),
            480,
            640,
        )
        if args.record
        else None
    )

    rows: list[tuple[str, dict]] = []
    if args.model:
        if not args.vec_norm:
            raise SystemExit("评估 RL 需要 --vec-norm 指向 best_vec_normalize.pkl")
        model, env, vec_norm = load_policy(
            args.model, args.vec_norm, args.task, terrain=args.terrain
        )
        if args.task in TRAVERSE_TASKS:
            scenarios = (
                [args.scenario]
                if args.scenario != "all"
                else [TASK_SCENARIO[args.task]]
            )
            for sc in scenarios:
                sc_env = Go2wEnv(
                    task=args.task, domain_randomize=False, scenario=sc,
                    terrain=args.terrain,
                )
                summaries = []
                for _ in range(args.episodes):
                    log, _ = run_rl_episode(model, sc_env, vec_norm)
                    summaries.append(summarize(log, args.task))
                agg = aggregate(summaries)
                print_report(f"RL PPO {args.task}/{sc} ({args.episodes} episodes)", agg)
                rows.append((f"RL PPO ({sc})", agg))
                if args.record:
                    log, frames = run_rl_episode(
                        model, sc_env, vec_norm, record=True, renderer=renderer,
                        camera=args.camera,
                    )
                    save_video(frames, pathlib.Path(f"{args.record}_{sc}"))
        else:
            summaries = []
            for ep in range(args.episodes):
                log, _ = run_rl_episode(model, env, vec_norm)
                summaries.append(summarize(log, args.task))
            agg = aggregate(summaries)
            print_report(f"RL PPO ({args.task}, {args.episodes} episodes)", agg)
            rows.append(("RL PPO", agg))

        if args.disturbance:
            dist_env = Go2wEnv(
                task=args.task,
                domain_randomize=False,
                disturbance=True,
                scenario=(
                    TASK_SCENARIO[args.task] if args.task in TRAVERSE_TASKS else None
                ),
                terrain=args.terrain,
            )
            log, _ = run_rl_episode(model, dist_env, vec_norm)
            after = [e for e in log if 5.2 <= e["t"] <= 6.2]
            max_dev_after = max((abs(e["pitch_err"]) for e in after), default=float("inf"))
            recovered = max_dev_after <= 0.1
            print(f"\n== 扰动测试（40N/0.2s，5.0s 施加）==")
            print(f"  5.2~6.2s 最大俯仰偏差: {max_dev_after:.4f} rad")
            print(f"  1s 内恢复: {recovered}")

        if args.record and args.task not in TRAVERSE_TASKS:
            log, frames = run_rl_episode(
                model, env, vec_norm, record=True, renderer=renderer,
                camera=args.camera,
            )
            save_video(frames, pathlib.Path(args.record))

    if (args.compare or not args.model) and args.task == "full_chain":
        summaries = []
        for _ in range(args.episodes):
            log, _ = run_controller_episode(args.task, duration)
            summaries.append(summarize(log, args.task))
        agg = aggregate(summaries)
        print_report(f"手写控制器 ({args.task}, {args.episodes} episodes)", agg)
        rows.append(("手写控制器", agg))

    if args.report:
        import csv
        with open(args.report, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "task", "max_dev", "min_clear", "dual_hold",
                             "recovered", "settle_seconds", "success", "success_rate",
                             "distance", "time_to_goal", "falls",
                             "total_reward", "mean_base_reward", "nan"])
            for label, s in rows:
                writer.writerow([label, s["task"], s["max_dev"], s["min_clear"],
                                 s["dual_hold"], s["recovered"], s["settle_seconds"],
                                 s["success"], s["success_rate"], s["distance"],
                                 s["time_to_goal"], s["falls"], s["total_reward"],
                                 s["mean_base_reward"], s["nan"]])
        print(f"对比表已写入: {args.report}")


if __name__ == "__main__":
    main()

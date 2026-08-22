"""多段路径（MoRA 阶段2：A→B 两段导航 + 精确停止）脚本控制器验证。

控制器：纯跟踪前视点 + 差速转向（与单段演示一致），
A/B 点预减速并在容差内零力矩保持，等待环境精确停止判定自动切换/完成。

用法：
    python scripts/gen_demo_multi_segment.py                 # 主实验 20 seeds
    python scripts/gen_demo_multi_segment.py --skip-stop     # 测试A：A点不停留
    python scripts/gen_demo_multi_segment.py --start-at-a    # 测试B：段2单独
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv, MULTI_SEGMENT_DEFAULT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "demo_trajectories"
FAILED_DIR = OUT_DIR / "failed_multi_segment"
MAX_STEPS = 4000


def default_params() -> dict:
    return {
        "k_ang": 2.5,
        "k_dev": 0.5,
        "bias_clip": 1.0,
        "forward": 0.12,
        "lookahead": 0.7,
        "noise": 0.0,
        "skip_stop": False,
        "start_at_a": False,
        "stage2": False,
    }


def compute_mseg_teacher_bias(
    x: float,
    y: float,
    yaw: float,
    seg_idx: int,
    params: dict | None = None,
    segments: list | None = None,
) -> float:
    """多段路径教师转向 bias（纯跟踪，与控制器一致）。"""
    p = params or default_params()
    segs = segments or MULTI_SEGMENT_DEFAULT
    seg = segs[seg_idx]
    x0, length, amp = seg["x0"], seg["x1"] - seg["x0"], seg["amp"]

    def center(px: float) -> float:
        if seg_idx == 0:
            return float(amp * math.sin(2.0 * math.pi * (px - x0) / length))
        ya = float(
            MULTI_SEGMENT_DEFAULT[0]["amp"]
            * math.sin(
                2.0
                * math.pi
                * (MULTI_SEGMENT_DEFAULT[0]["x1"] - MULTI_SEGMENT_DEFAULT[0]["x0"])
                / (MULTI_SEGMENT_DEFAULT[0]["x1"] - MULTI_SEGMENT_DEFAULT[0]["x0"])
            )
        )
        return ya + float(amp * math.sin(2.0 * math.pi * (px - x0) / length))

    xt = min(x + p["lookahead"], seg["x1"])
    yt = center(xt)
    desired_yaw = float(np.arctan2(yt - y, xt - x))
    ang_err = float(((desired_yaw - yaw + math.pi) % (2 * math.pi)) - math.pi)
    dev = float(y - center(x))
    return float(
        np.clip(
            -p["k_ang"] * ang_err + p["k_dev"] * dev,
            -p["bias_clip"],
            p["bias_clip"],
        )
    )


def make_env() -> Go2wEnv:
    return Go2wEnv(task="traverse_curve", domain_randomize=False, multi_segment=True)


def compute_action(
    env: Go2wEnv,
    params: dict,
) -> tuple[np.ndarray, float]:
    x = float(env.data.body("base_link").xpos[0])
    y = float(env.data.body("base_link").xpos[1])
    yaw = float(env._yaw_of())
    seg_idx = env._current_segment()
    seg = env.segments[seg_idx]
    gx, gy = env._segment_goal(seg_idx)
    dist = float(math.hypot(x - gx, y - gy))

    xt = min(x + params["lookahead"], seg["x1"])
    bias = compute_mseg_teacher_bias(
        x, y, yaw, seg_idx, params, segments=env.segments
    )

    speed_scale = 1.0
    if not params.get("skip_stop", False):
        if dist < 0.3:
            speed_scale = 0.4
        elif dist < 0.8:
            speed_scale = 0.3 + (dist - 0.3) / 0.5 * 0.7

    if (
        not params.get("skip_stop", False)
        and dist <= seg["goal_tolerance"]
    ):
        vx, _ = env._body_vel()
        if abs(vx) > seg["stop_speed_threshold"]:
            bias = 0.0
            fwd = -0.02 * abs(vx)
        else:
            bias = 0.0
            fwd = 0.0
        action = np.array(
            [fwd, fwd, fwd, fwd, 0.0, 0.0],
            dtype=np.float32,
        )
        if params.get("noise", 0.0) > 0:
            rng = np.random.default_rng()
            action = np.clip(
                action + rng.normal(0.0, params["noise"], size=6).astype(np.float32),
                -1.0,
                1.0,
            )
        return action, speed_scale

    fwd = float(params["forward"]) * speed_scale
    action = np.array(
        [fwd - bias, fwd + bias, fwd - bias, fwd + bias, 0.0, 0.0],
        dtype=np.float32,
    )
    if params.get("noise", 0.0) > 0:
        rng = np.random.default_rng()
        action = np.clip(
            action + rng.normal(0.0, params["noise"], size=6).astype(np.float32),
            -1.0,
            1.0,
        )
    return action, speed_scale


def run_episode(
    env: Go2wEnv,
    params: dict,
    seed: int,
) -> dict:
    obs, info = env.reset(seed=seed)
    if params.get("stage2", False):
        obs, info = env.reset(
            seed=seed, options={"start_at_subgoal": "A", "scope": 1}
        )
        rng = np.random.default_rng(seed)
        ax, ay = env._segment_goal(0)
        env.data.qpos[0] = ax + float(rng.uniform(-0.05, 0.05))
        env.data.qpos[1] = ay + float(rng.uniform(-0.05, 0.05))
        yaw0 = env._segment_tangent_angle(float(env.data.qpos[0]), 1)
        env.data.qpos[3:7] = (
            math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0),
        )
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        env._prev_x = float(env.data.body("base_link").xpos[0])
        env._prev_progress = env._prev_x
        env._max_x = env._prev_x
        obs, info = env._get_obs(), env._info()
    elif params.get("start_at_a", False):
        ax, ay = env._segment_goal(0)
        env.data.qpos[0] = ax
        env.data.qpos[1] = ay
        yaw0 = float(env._segment_tangent_angle(ax, 1))
        env.data.qpos[3:7] = (
            math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0),
        )
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        env._segment_idx = 1
        env._subgoal_reached_a = True
        env._prev_x = ax
        env._prev_progress = ax
        env._max_x = ax
        obs, info = env._get_obs(), env._info()

    rec: dict[str, Any] = {
        "obs": [], "actions": [], "positions": [], "yaws": [],
        "deviations": [], "heading_errors": [], "segments": [],
        "speed_scales": [], "times": [],
    }
    max_dev = 0.0
    steps = 0
    goal = False
    term = False
    trunc = False
    for _ in range(MAX_STEPS):
        rec["obs"].append(np.asarray(obs, dtype=np.float32))
        x = float(env.data.body("base_link").xpos[0])
        y = float(env.data.body("base_link").xpos[1])
        yaw = float(env._yaw_of())
        seg_idx = env._current_segment()
        dev = float(env._segment_deviation(x, y, seg_idx))
        tan = float(env._segment_tangent_angle(x, seg_idx))
        herr = float(((yaw - tan + math.pi) % (2 * math.pi)) - math.pi)
        action, speed_scale = compute_action(env, params)
        rec["actions"].append(action)
        rec["positions"].append((x, y))
        rec["yaws"].append(yaw)
        rec["deviations"].append(dev)
        rec["heading_errors"].append(herr)
        rec["segments"].append(seg_idx)
        rec["speed_scales"].append(speed_scale)
        rec["times"].append(float(env.data.time))
        max_dev = max(max_dev, abs(dev))
        obs, _r, term, trunc, info = env.step(action)
        steps += 1
        if params.get("skip_stop", False) and env._current_segment() == 0:
            gx, gy = env._segment_goal(0)
            xa = float(env.data.body("base_link").xpos[0])
            ya = float(env.data.body("base_link").xpos[1])
            if math.hypot(xa - gx, ya - gy) <= env.segments[0]["goal_tolerance"]:
                env._switch_to_next_segment()
        if info.get("goal"):
            goal = True
            break
        if term or trunc:
            break

    rec.update(
        {
            "steps": steps,
            "goal": goal,
            "term": term,
            "trunc": trunc,
            "max_dev": max_dev,
            "final_x": float(env.data.body("base_link").xpos[0]),
            "final_y": float(env.data.body("base_link").xpos[1]),
            "seg1_ok": bool(env._subgoal_reached_a),
            "seg2_ok": goal,
            "a_stop_ok": bool(env._subgoal_reached_a),
            "b_stop_ok": goal,
            "avg_speed": float(
                np.mean(np.abs(np.diff([p[0] for p in rec["positions"]])))
                / 0.01
                if len(rec["positions"]) > 1
                else 0.0
            ),
            "segment_idx_final": env._current_segment(),
        }
    )
    return rec


def save_npz(path: pathlib.Path, rec: dict, params: dict, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": seed,
        "goal": rec["goal"],
        "steps": rec["steps"],
        "seg1_ok": rec["seg1_ok"],
        "a_stop_ok": rec["a_stop_ok"],
        "seg2_ok": rec["seg2_ok"],
        "max_dev": round(rec["max_dev"], 4),
        "params": params,
    }
    np.savez_compressed(
        path,
        obs=np.array(rec["obs"], dtype=np.float32),
        actions=np.array(rec["actions"], dtype=np.float32),
        positions=np.array(rec["positions"], dtype=np.float32),
        yaws=np.array(rec["yaws"], dtype=np.float32),
        deviations=np.array(rec["deviations"], dtype=np.float32),
        heading_errors=np.array(rec["heading_errors"], dtype=np.float32),
        segments=np.array(rec["segments"], dtype=np.int32),
        speed_scales=np.array(rec["speed_scales"], dtype=np.float32),
        times=np.array(rec["times"], dtype=np.float32),
        meta=np.array([str(meta)], dtype=object),
    )


def adjust_params(params: dict, rec: dict) -> dict:
    p = dict(params)
    if rec["term"] and not rec["goal"] and rec["max_dev"] > 0.4:
        p["k_ang"] = min(3.5, p["k_ang"] + 0.5)
    elif rec["term"] and not rec["goal"]:
        p["forward"] = max(0.05, p["forward"] - 0.04)
    else:
        p["forward"] = min(0.2, p["forward"] + 0.03)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="多段路径脚本控制器验证")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--skip-stop", action="store_true", help="测试A：A点不停留")
    parser.add_argument("--start-at-a", action="store_true", help="测试B：从A点出发段2")
    parser.add_argument("--stage2", action="store_true",
                        help="生成段2（A→B）演示轨迹（scope=1）")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--tag", type=str, default="",
                        help="输出文件标签，避免不同实验互相覆盖")
    parser.add_argument("--plot", action="store_true", help="生成俯视轨迹图")
    parser.add_argument("--plot-file", type=str, default="",
                        help="用于绘图的 npz 轨迹文件")
    parser.add_argument("--plot-out", type=str, default="",
                        help="PNG 输出路径")
    args = parser.parse_args()

    if args.plot:
        plot_file = pathlib.Path(args.plot_file or OUT_DIR / "curve_multi_segment_demo_main_seed00.npz")
        plot_out = pathlib.Path(args.plot_out or OUT_DIR / "multi_segment_trajectory.png")
        plot_trajectory(plot_file, plot_out)
        return

    params = default_params()
    params["noise"] = args.noise
    params["skip_stop"] = args.skip_stop
    params["start_at_a"] = args.start_at_a
    params["stage2"] = args.stage2
    tag = f"_{args.tag}" if args.tag else ""

    rows: list[dict] = []
    for seed in range(args.seeds):
        attempt_params = dict(params)
        best: dict | None = None
        for attempt in range(1, args.attempts + 1):
            env = make_env()
            rec = run_episode(env, attempt_params, seed)
            env.close()
            if rec["goal"] or attempt == args.attempts:
                best = rec
                best["params"] = dict(attempt_params)
                best["attempt"] = attempt
                if not rec["goal"]:
                    attempt_params = adjust_params(attempt_params, rec)
                break
            attempt_params = adjust_params(attempt_params, rec)
        assert best is not None
        stem = (
            f"{'multi_segment_stage2_demo' if params['stage2'] else 'curve_multi_segment_demo'}{tag}_seed{seed:02d}"
            if best["goal"]
            else f"fail{tag}_seed{seed:02d}"
        )
        save_npz(OUT_DIR / f"{stem}.npz" if best["goal"] else FAILED_DIR / f"{stem}.npz", best, best["params"], seed)
        rows.append(
            {
                "seed": seed,
                "success": best["goal"],
                "steps": best["steps"],
                "time_s": round(float(best["times"][-1]) if best["times"] else 0.0, 2),
                "seg1_ok": best["seg1_ok"],
                "a_stop_ok": best["a_stop_ok"],
                "seg2_ok": best["seg2_ok"],
                "max_dev": round(best["max_dev"], 3),
                "avg_speed": round(best["avg_speed"], 3),
            }
        )
        print(
            f"seed{seed:02d} {'OK ' if best['goal'] else 'FAIL'} "
            f"steps={best['steps']} t={rows[-1]['time_s']}s "
            f"seg1={int(best['seg1_ok'])} A_stop={int(best['a_stop_ok'])} "
            f"seg2={int(best['seg2_ok'])} max_dev={rows[-1]['max_dev']}"
        )

    success = sum(r["success"] for r in rows)
    print(f"\n成功率: {success}/{len(rows)} = {success/len(rows):.0%}")
    print(f"A点停止达标: {sum(r['a_stop_ok'] for r in rows)}/{len(rows)}")
    print(f"平均完成时间: {np.mean([r['time_s'] for r in rows if r['success']] or [0]):.2f}s")
    out_csv = OUT_DIR / (
        f"{'multi_segment_stage2_validation' if params['stage2'] else 'multi_segment_validation'}{tag}.csv"
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("明细:", out_csv)


def plot_trajectory(npz_path: pathlib.Path, out_path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = make_env()
    d = np.load(npz_path, allow_pickle=True)
    pos = np.asarray(d["positions"], dtype=float)
    xs = np.linspace(0.8, 10.8, 600)
    ys0 = np.array([env._segment_center_y(x, 0) for x in xs])
    ys1 = np.array([env._segment_center_y(x, 1) for x in xs])
    fig, ax = plt.subplots(figsize=(10, 4))
    mask0 = xs <= 4.3
    mask1 = xs >= 4.3
    ax.plot(xs[mask0], ys0[mask0], "b-", label="Segment 1 centerline")
    ax.plot(xs[mask1], ys1[mask1], "r-", label="Segment 2 centerline")
    for mask, ys, cw, color in (
        (mask0, ys0, 0.50, "b"),
        (mask1, ys1, 0.40, "r"),
    ):
        ax.fill_between(xs[mask], ys[mask] - cw, ys[mask] + cw, color=color, alpha=0.12)
    ax.plot(pos[:, 0], pos[:, 1], "k-", lw=1.2, label="Actual trajectory")
    ax.plot([4.3], [0.0], "go", ms=10, label="A")
    ax.plot([10.8], [0.0], "r*", ms=14, label="B")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Multi-segment trajectory (A→B)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print("已生成:", out_path)
    env.close()


if __name__ == "__main__":
    main()

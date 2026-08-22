"""traverse_curve DAgger 教师路线训练（方案C）。

流程：
  1. 用 6 条脚本演示轨迹（固定 seed03 做验证）训练初始 BC；
  2. 每轮让当前 BC 随机策略跑 20 个 episode，收集“犯错状态”
     （摔倒前 K 步 / 横向偏差 >0.20m / 航向误差 >0.30rad）；
  3. 用脚本教师控制器为犯错状态生成动作，加入训练集后整体重训 BC；
  4. 迭代 ≤5 轮，直到 20-episode 平均距离 ≥2.0m 或收敛；
  5. 输出每轮模型/数据集/报告；不自动启动 PPO 微调。

用法：python scripts/dagger_train.py [--max-iter 5] [--episodes 20]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import pickle
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv, curve_center_y, heading_error  # noqa: E402
from scripts import bc_pretrain  # noqa: E402
from scripts.gen_demo_trajectory import (  # noqa: E402
    compute_teacher_action,
    default_params,
)

OUT = PROJECT_ROOT / "data" / "demo_trajectories"
NORM_PATH = OUT / "curve_vecnorm.pkl"
RUN_DIR = PROJECT_ROOT / "rl" / "runs" / "traverse_curve_dagger" / "seed00"

DEMO_SEEDS = list(range(6))
VAL_SEED = 3
MAX_STEPS = 1500
EPOCHS_INIT = 50
EPOCHS_ITER = 30
LR = 1e-3
BATCH_SIZE = 512
K_FALL = 5
DEV_THRESHOLD = 0.20
HERR_THRESHOLD = 0.30
MAX_ERROR_STATES_PER_EP = 50
TARGET_DISTANCE = 2.0
MIN_GAIN = 0.1


def load_demos(out_dir: pathlib.Path = OUT) -> dict[str, np.ndarray]:
    """加载 6 条演示轨迹，返回原始 obs/actions/轨迹 id。"""
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    traj_ids: list[int] = []
    for seed in DEMO_SEEDS:
        path = out_dir / f"curve_demo_seed{seed:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=True)
        obs_all.append(d["obs"])
        act_all.append(d["actions"])
        traj_ids.extend([seed] * len(d["obs"]))
    return {
        "obs": np.concatenate(obs_all).astype(np.float32),
        "actions": np.concatenate(act_all).astype(np.float32),
        "traj_ids": np.asarray(traj_ids, dtype=int),
    }


def train_dagger_bc(
    norm: dict,
    demo_obs: np.ndarray,
    demo_acts: np.ndarray,
    demo_train_mask: np.ndarray,
    demo_val_mask: np.ndarray,
    corr_obs: np.ndarray,
    corr_acts: np.ndarray,
    epochs: int,
) -> tuple[Any, list[dict]]:
    """用“演示训练集 + 全部纠正数据”重训 BC，验证集固定为 seed03 轨迹。"""
    if len(corr_obs):
        train_obs = np.concatenate([demo_obs[demo_train_mask], corr_obs])
        train_acts = np.concatenate([demo_acts[demo_train_mask], corr_acts])
    else:
        train_obs = demo_obs[demo_train_mask]
        train_acts = demo_acts[demo_train_mask]
    val_obs = demo_obs[demo_val_mask]
    val_acts = demo_acts[demo_val_mask]

    obs_all = np.concatenate([train_obs, val_obs]).astype(np.float32)
    acts_all = np.concatenate([train_acts, val_acts]).astype(np.float32)
    obs_n = bc_pretrain.normalize(obs_all, norm)
    n_train = len(train_obs)
    train_mask = np.zeros(len(obs_all), dtype=bool)
    train_mask[:n_train] = True
    val_mask = np.zeros(len(obs_all), dtype=bool)
    val_mask[n_train:] = True
    return bc_pretrain.train_bc(
        obs_n,
        acts_all,
        train_mask,
        val_mask,
        epochs=epochs,
        lr=LR,
        batch_size=BATCH_SIZE,
    )


def collect_error_indices(
    devs: list[float] | np.ndarray,
    hers: list[float] | np.ndarray,
    k_fall: int = K_FALL,
    dev_thresh: float = DEV_THRESHOLD,
    herr_thresh: float = HERR_THRESHOLD,
    max_states: int = MAX_ERROR_STATES_PER_EP,
    fell: bool = True,
) -> list[int]:
    """返回需要教师纠正的状态下标：摔倒前 K 步 + 超偏差/超航向误差，去重限幅。"""
    n = len(devs)
    idx: list[int] = []
    if fell and n:
        idx.extend(range(max(0, n - k_fall), n))
    for i in range(n):
        if abs(float(devs[i])) > dev_thresh or abs(float(hers[i])) > herr_thresh:
            idx.append(i)
    seen: set[int] = set()
    out: list[int] = []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out[:max_states]


def collect_episode_states(
    model: Any,
    norm: dict,
    episodes: int,
) -> tuple[list[tuple[np.ndarray, float, float, float]], list[dict]]:
    """随机策略跑 N 个 episode，收集（决策前 obs, x, y, yaw）犯错状态。"""
    error_states: list[tuple[np.ndarray, float, float, float]] = []
    per_episode: list[dict] = []
    params = default_params()
    for ep in range(episodes):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        obs, info = env.reset(seed=1000 + ep)
        obs_list: list[np.ndarray] = []
        x_list: list[float] = []
        y_list: list[float] = []
        yaw_list: list[float] = []
        dev_list: list[float] = []
        herr_list: list[float] = []
        max_prog = 0.0
        goal = False
        term = False
        trunc = False
        for _ in range(MAX_STEPS):
            obs_list.append(np.asarray(obs, dtype=np.float32))
            x = float(env.data.body("base_link").xpos[0])
            y = float(env.data.body("base_link").xpos[1])
            yaw = float(env._yaw_of())
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
            dev_list.append(float(y - float(curve_center_y(x))))
            herr_list.append(float(heading_error(x, yaw, is_curve=True)))
            obs_n = bc_pretrain.normalize(obs, norm)
            action, _ = model.predict(obs_n, deterministic=False)
            obs, _r, term, trunc, info = env.step(action)
            max_prog = max(max_prog, float(info.get("progress", info.get("x", 0.0))))
            if info.get("goal"):
                goal = True
            if term or trunc:
                break
        indices = collect_error_indices(
            dev_list,
            herr_list,
            fell=bool(term and not trunc),
        )
        for i in indices:
            error_states.append((obs_list[i], x_list[i], y_list[i], yaw_list[i]))
        per_episode.append(
            {
                "episode": ep,
                "steps": len(obs_list),
                "max_progress": float(max_prog),
                "goal": bool(goal),
                "error_states": len(indices),
                "fell": bool(term and not trunc),
            }
        )
        env.close()
    return error_states, per_episode


def evaluate_bc(model: Any, norm: dict, episodes: int) -> dict:
    return bc_pretrain.evaluate(model, norm, episodes=episodes, deterministic=True)


def decide_stop(
    history: list[dict],
    max_iter: int,
    target: float = TARGET_DISTANCE,
    min_gain: float = MIN_GAIN,
) -> tuple[bool, str]:
    """停止条件：达标 / 连续两轮增益 <0.1m / 达到轮数上限。"""
    if len(history) > 1 and history[-1]["mean_distance"] >= target:
        return True, "success"
    if len(history) >= 3:
        g1 = history[-1]["mean_distance"] - history[-2]["mean_distance"]
        g2 = history[-2]["mean_distance"] - history[-3]["mean_distance"]
        if g1 < min_gain and g2 < min_gain:
            return True, "converged"
    if len(history) - 1 >= max_iter:
        return True, "max_iter"
    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="DAgger 教师路线训练")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epochs-init", type=int, default=EPOCHS_INIT)
    parser.add_argument("--epochs-iter", type=int, default=EPOCHS_ITER)
    args = parser.parse_args()

    with open(NORM_PATH, "rb") as f:
        norm = pickle.load(f)
    demos = load_demos()
    traj = demos["traj_ids"]
    demo_train_mask = traj != VAL_SEED
    demo_val_mask = traj == VAL_SEED
    print(
        f"演示数据: {len(demos['obs'])} 步, "
        f"训练轨迹 {int(demo_train_mask.sum())} 步, "
        f"验证轨迹 seed{VAL_SEED} {int(demo_val_mask.sum())} 步"
    )

    corr_obs_list: list[np.ndarray] = []
    corr_act_list: list[np.ndarray] = []
    history: list[dict] = []
    seen_states: set[bytes] = set()
    params = default_params()

    # 初始 BC
    model, log = train_dagger_bc(
        norm, demos["obs"], demos["actions"],
        demo_train_mask, demo_val_mask,
        np.zeros((0, 61), dtype=np.float32), np.zeros((0, 6), dtype=np.float32),
        epochs=args.epochs_init,
    )
    result = evaluate_bc(model, norm, args.episodes)
    history.append(
        {
            "iteration": 0,
            "dataset_size": int(demo_train_mask.sum()),
            "val_loss": log[-1]["val_loss"],
            **result,
        }
    )
    print(f"[iter0] dataset={history[-1]['dataset_size']} "
          f"mean_dist={result['mean_distance']:.3f} ep_len={result['mean_ep_len']:.0f}")

    for k in range(1, args.max_iter + 1):
        error_states, per_ep = collect_episode_states(
            model, norm, episodes=args.episodes
        )
        added = 0
        for obs_raw, x, y, yaw in error_states:
            obs_n = bc_pretrain.normalize(np.asarray(obs_raw, dtype=np.float32), norm)
            key = obs_n.round(3).tobytes()
            if key in seen_states:
                continue
            seen_states.add(key)
            corr_obs_list.append(obs_n.astype(np.float32))
            corr_act_list.append(compute_teacher_action(x, y, yaw, params))
            added += 1
        print(
            f"[iter{k}] 犯错状态 {len(error_states)}，新增去重后 {added}，"
            f"累计纠正 {len(corr_obs_list)}"
        )
        corr_obs = (
            np.stack(corr_obs_list) if corr_obs_list
            else np.zeros((0, 61), dtype=np.float32)
        )
        corr_acts = (
            np.stack(corr_act_list) if corr_act_list
            else np.zeros((0, 6), dtype=np.float32)
        )
        model, log = train_dagger_bc(
            norm, demos["obs"], demos["actions"],
            demo_train_mask, demo_val_mask,
            corr_obs, corr_acts,
            epochs=args.epochs_iter,
        )
        result = evaluate_bc(model, norm, args.episodes)
        history.append(
            {
                "iteration": k,
                "dataset_size": int(demo_train_mask.sum()) + len(corr_obs_list),
                "correction_size": len(corr_obs_list),
                "val_loss": log[-1]["val_loss"],
                **result,
            }
        )
        model_path = OUT / f"bc_iter{k}.zip"
        model.save(str(model_path))
        np.savez_compressed(
            OUT / f"dagger_dataset_iter{k}.npz",
            obs_norm=corr_obs,
            actions=corr_acts,
        )
        print(
            f"[iter{k}] dataset={history[-1]['dataset_size']} "
            f"val_loss={log[-1]['val_loss']:.6f} "
            f"mean_dist={result['mean_distance']:.3f} "
            f"success={result['success_rate']:.0%} ep_len={result['mean_ep_len']:.0f}"
        )
        stop, reason = decide_stop(history, args.max_iter)
        if stop:
            print(f"停止：{reason}")
            break

    # 报告 + 量化端 eval_log（runs.csv/labels.csv 由 go2w-quant 采集）
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "eval_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "mean_reward", "std_reward", "mean_ep_len"])
        for h in history:
            w.writerow(
                [
                    h["iteration"] * 100_000,
                    round(h["mean_distance"], 4),
                    round(h["max_distance"] - h["mean_distance"], 4),
                    round(h["mean_ep_len"], 2),
                ]
            )
    (RUN_DIR / ".completed").write_text("", encoding="utf-8")

    report = [
        "# traverse_curve DAgger 训练报告",
        "",
        f"生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 初始 BC epoch：{args.epochs_init}；每轮 BC epoch：{args.epochs_iter}",
        f"- 每轮采样：{args.episodes} episodes；教师参数：{params}",
        f"- 犯错条件：摔倒前 {K_FALL} 步 / |dev|>{DEV_THRESHOLD} / |herr|>{HERR_THRESHOLD}",
        "",
        "| 迭代 | 数据集大小 | 纠正数 | val_loss | 平均距离 | 最大距离 | 成功率 | ep_len |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in history:
        report.append(
            f"| {h['iteration']} | {h['dataset_size']} | {h.get('correction_size', 0)} "
            f"| {h['val_loss']:.6f} | {h['mean_distance']:.3f} | {h['max_distance']:.3f} "
            f"| {h['success_rate']:.0%} | {h['mean_ep_len']:.0f} |"
        )
    final = history[-1]
    verdict = (
        "成功（平均距离 ≥2.0m，可进入 PPO 微调）"
        if final["mean_distance"] >= TARGET_DISTANCE
        else "未达标（需分析；不建议直接 PPO 微调）"
    )
    report += [
        "",
        f"## 结论：{verdict}",
        "",
        f"最终模型：`data/demo_trajectories/bc_iter{final['iteration']}.zip`",
    ]
    (OUT / "dagger_training_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print("报告:", OUT / "dagger_training_report.md")


if __name__ == "__main__":
    main()

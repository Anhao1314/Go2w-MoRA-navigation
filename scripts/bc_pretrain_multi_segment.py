"""多段路径（A→B）BC 预热：用 20 条脚本轨迹监督训练高层 2 维策略。

轨迹存 61 维底层 obs + 6 维低层动作；这里动态重建 67 维 MoRA 观测，
并从低层动作反推高层动作 [speed_scale, turn_adjust]。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import pickle
import sys
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
from rl.go2w_env import Go2wEnv  # noqa: E402
from rl.high_level_env_wrapper import HighLevelEnvWrapper, compute_mora_obs  # noqa: E402
from scripts.gen_demo_multi_segment import (  # noqa: E402
    compute_mseg_teacher_bias,
    default_params,
)

OUT_DIR = PROJECT_ROOT / "data" / "demo_trajectories"
DEMO_PREFIX = OUT_DIR / "curve_multi_segment_demo_main_seed"
TRAIN_SEEDS = list(range(16))
VAL_SEEDS = list(range(16, 20))
EPS = 1e-8


def make_high_env() -> HighLevelEnvWrapper:
    base = Go2wEnv(
        task="traverse_curve",
        domain_randomize=False,
        multi_segment=True,
        reward_version="mseg",
    )
    return HighLevelEnvWrapper(
        base,
        append_mora=True,
        use_goal_condition=False,
        goal_min=1.4,
        goal_max=1.4,
    )


def load_demos(scope: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (obs67, high_actions, seed_ids)。"""
    prefix = (
        OUT_DIR / "multi_segment_stage2_demo_seed"
        if scope == 1
        else DEMO_PREFIX
    )
    env = make_high_env().base_env
    params = default_params()
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    seed_ids: list[int] = []
    for seed in TRAIN_SEEDS + VAL_SEEDS:
        p = pathlib.Path(f"{prefix}{seed:02d}.npz")
        if not p.exists():
            raise FileNotFoundError(p)
        d = np.load(p, allow_pickle=True)
        pos = np.asarray(d["positions"], dtype=float)
        yaws = np.asarray(d["yaws"], dtype=float)
        segs = np.asarray(d["segments"], dtype=int)
        speed = np.asarray(d["speed_scales"], dtype=float)
        actions = np.asarray(d["actions"], dtype=float)
        base_obs = np.asarray(d["obs"], dtype=np.float32)
        for i in range(len(pos)):
            x, y = float(pos[i, 0]), float(pos[i, 1])
            yaw = float(yaws[i])
            seg_idx = int(segs[i])
            env.data.qpos[0] = x
            env.data.qpos[1] = y
            env.data.qpos[3:7] = (
                math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0),
            )
            env._segment_idx = seg_idx
            env._passed_subgoal = seg_idx >= 1
            env._subgoal_reached_a = seg_idx >= 1
            mujoco.mj_forward(env.model, env.data)
            mora = compute_mora_obs(env)
            obs67 = np.concatenate([base_obs[i], mora]).astype(np.float32)
            obs_all.append(obs67)
            controller_bias = float((actions[i, 1] - actions[i, 0]) / 2.0)
            teacher_bias = compute_mseg_teacher_bias(
                x, y, yaw, seg_idx, params, segments=env.segments
            )
            turn_adjust = float(np.clip(controller_bias - teacher_bias, -0.5, 0.5))
            speed_label = float(np.clip(speed[i], 0.9, 1.5))
            act_all.append(np.array([speed_label, turn_adjust], dtype=np.float32))
            seed_ids.append(seed)
        print(f"seed{seed:02d}: {len(pos)} 步")
    return (
        np.stack(obs_all),
        np.stack(act_all),
        np.asarray(seed_ids, dtype=int),
    )


def train_bc(
    obs: np.ndarray,
    acts: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience_limit: int = 10,
) -> tuple[PPO, list[dict]]:
    env = make_vec_env(make_high_env, n_envs=1)
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0, verbose=0,
    )
    policy = model.policy
    params = (
        list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)
    mse = torch.nn.MSELoss()
    x_tr = torch.from_numpy(obs[train_mask])
    y_tr = torch.from_numpy(acts[train_mask])
    x_va = torch.from_numpy(obs[val_mask])
    y_va = torch.from_numpy(acts[val_mask])
    n = len(x_tr)
    log: list[dict] = []
    best_val = float("inf")
    patience = 0
    for epoch in range(1, epochs + 1):
        policy.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        total = 0.0
        cnt = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = policy.action_net(policy.mlp_extractor.policy_net(x_tr[idx]))
            loss = mse(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
            cnt += len(idx)
        train_loss = total / max(1, cnt)
        policy.eval()
        with torch.no_grad():
            pred_v = policy.action_net(policy.mlp_extractor.policy_net(x_va))
            val_loss = float(mse(pred_v, y_va).item())
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6)})
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"早停 epoch={epoch}")
                break
    return model, log


def evaluate_seg1(
    model: PPO, norm: dict, episodes: int = 20, scope: int = 0
) -> dict:
    successes = 0
    full = 0
    for ep in range(episodes):
        env = make_high_env()
        if scope == 1:
            obs, info = env.reset(
                seed=ep, options={"start_at_subgoal": "A", "scope": 1}
            )
        else:
            obs, info = env.reset(seed=ep)
        seg1_ok = False
        goal = False
        while True:
            obs_n = (obs - norm["mean"]) / np.sqrt(norm["var"] + EPS)
            obs_n = np.clip(obs_n, -10.0, 10.0).astype(np.float32)
            action, _ = model.predict(obs_n, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            if info.get("passed_subgoal"):
                seg1_ok = True
            if info.get("goal"):
                goal = True
            if term or trunc:
                break
        successes += int(seg1_ok)
        full += int(goal)
        env.close()
    return {
        "episodes": episodes,
        "seg1_success": successes,
        "seg1_rate": successes / episodes,
        "full_success": full,
        "full_rate": full / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多段路径 BC 预热")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scope", type=int, default=0,
                        help="0=主任务(段1)，1=段2(A→B) BC")
    parser.add_argument("--patience", type=int, default=10,
                        help="验证早停耐心")
    args = parser.parse_args()

    obs, acts, seeds = load_demos(scope=args.scope)
    train_mask = np.isin(seeds, TRAIN_SEEDS)
    val_mask = np.isin(seeds, VAL_SEEDS)
    mean = obs.mean(axis=0).astype(np.float32)
    var = obs.var(axis=0).astype(np.float32)
    count = float(len(obs))
    norm = {"mean": mean, "var": var, "count": count}
    suffix = "_stage2" if args.scope == 1 else ""
    with open(OUT_DIR / f"multi_segment{suffix}_bc_vecnorm.pkl", "wb") as f:
        pickle.dump(norm, f)
    print(f"数据集 {len(obs)} 步，训练 {int(train_mask.sum())} / 验证 {int(val_mask.sum())}")

    model, log = train_bc(obs, acts, train_mask, val_mask,
                          epochs=args.epochs, lr=args.lr,
                          batch_size=args.batch_size,
                          patience_limit=args.patience)
    model_path = OUT_DIR / f"multi_segment{suffix}_bc_pretrain.zip"
    model.save(str(model_path))
    with open(OUT_DIR / f"multi_segment{suffix}_bc_training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(log)
    result = evaluate_seg1(model, norm, scope=args.scope)
    print("BC 验证:", result)
    report = OUT_DIR / f"multi_segment{suffix}_bc_report.md"
    report.write_text(
        "\n".join(
            [
                "# 多段路径 BC 预热报告",
                "",
                f"- 训练步数：{len(obs)}；训练/验证 seed：{len(TRAIN_SEEDS)}/{len(VAL_SEEDS)}",
                f"- 最终 train/val loss：{log[-1]['train_loss']:.6f}/{log[-1]['val_loss']:.6f}",
                f"- 段1成功率（A 精确停止）：{result['seg1_rate']:.0%}"
                f"（{result['seg1_success']}/{result['episodes']}）",
                f"- 全程成功率：{result['full_rate']:.0%}"
                f"（{result['full_success']}/{result['episodes']}）",
                "",
                "## 判定",
                "",
                "✅ 段1≥80%，可进入课程学习"
                if result["seg1_rate"] >= 0.8
                else "❌ 段1<80%，需加 epoch/DAgger",
            ]
        ),
        encoding="utf-8",
    )
    print("已保存:", model_path, "norm, log, report")


if __name__ == "__main__":
    main()

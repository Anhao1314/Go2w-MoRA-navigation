"""Junction 岔路口 BC 预热：20 条轨迹 → 68 维观测 + 高层动作。"""

from __future__ import annotations

import csv
import math
import pathlib
import pickle
import sys
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
from rl.go2w_env import Go2wEnv  # noqa: E402
from rl.high_level_env_wrapper import (  # noqa: E402
    HighLevelEnvWrapper,
    compute_junction_obs,
    junction_teacher_bias,
)

OUT = PROJECT_ROOT / "data" / "demo_trajectories"
EPS = 1e-8


def make_env() -> HighLevelEnvWrapper:
    base = Go2wEnv(
        task="traverse_curve", domain_randomize=False,
        junction=True, reward_version="junction", scope=2,
    )
    return HighLevelEnvWrapper(
        base, append_junction=True, use_goal_condition=False
    )


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = make_env().base_env
    obs_all, act_all, seeds = [], [], []
    for seed in range(20):
        p = OUT / f"junction_demo_seed{seed:02d}.npz"
        d = np.load(p, allow_pickle=True)
        pos = np.asarray(d["positions"], dtype=float)
        yaws = np.asarray(d["yaws"], dtype=float)
        branches = np.asarray(d["branches"], dtype=float)
        speeds = np.asarray(d["speeds"], dtype=float)
        acts = np.asarray(d["actions"], dtype=float)
        base_obs = np.asarray(d["obs"], dtype=np.float32)
        env.target_goal = "A" if seed < 10 else "B"
        for i in range(len(pos)):
            x, y = pos[i]
            yaw = float(yaws[i])
            branch = "left" if float(branches[i]) > 0.5 else "right"
            env.data.qpos[0] = x
            env.data.qpos[1] = y
            env.data.qpos[3:7] = (
                math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2),
            )
            env.branch_selected = branch
            mujoco.mj_forward(env.model, env.data)
            obs = np.concatenate(
                [base_obs[i], compute_junction_obs(env)]
            ).astype(np.float32)
            obs_all.append(obs)
            controller_bias = float((acts[i, 1] - acts[i, 0]) / 2.0)
            teacher = junction_teacher_bias(x, y, yaw, env)
            turn = float(np.clip(controller_bias - teacher, -0.5, 0.5))
            act_all.append(
                np.array([float(np.clip(speeds[i], 0.9, 1.5)), turn],
                         dtype=np.float32)
            )
            seeds.append(seed)
    return np.stack(obs_all), np.stack(act_all), np.asarray(seeds)


def train(obs, acts, train_mask, val_mask, norm):
    env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [256, 256]},
                seed=0, verbose=0)
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + \
        list(policy.action_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = torch.nn.MSELoss()
    def _norm(arr):
        return (arr - norm["mean"]) / np.sqrt(norm["var"] + EPS)
    x_tr = torch.from_numpy(_norm(obs[train_mask]))
    y_tr = torch.from_numpy(acts[train_mask])
    x_va = torch.from_numpy(_norm(obs[val_mask]))
    y_va = torch.from_numpy(acts[val_mask])
    best_val = float("inf")
    patience = 0
    log = []
    for epoch in range(1, 301):
        policy.train()
        perm = torch.randperm(len(x_tr), generator=torch.Generator().manual_seed(epoch))
        for i in range(0, len(x_tr), 512):
            idx = perm[i:i+512]
            pred = policy.action_net(policy.mlp_extractor.policy_net(x_tr[idx]))
            loss = mse(pred, y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        policy.eval()
        with torch.no_grad():
            pred_v = policy.action_net(policy.mlp_extractor.policy_net(x_va))
            val_loss = float(mse(pred_v, y_va).item())
        log.append({"epoch": epoch, "val_loss": round(val_loss, 6)})
        if epoch % 50 == 0 or epoch == 1:
            print(epoch, val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss; patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    return model, log


def evaluate(model, norm):
    suc = 0
    for seed in range(20):
        env = make_env()
        obs, info = env.reset(
            seed=seed,
            options={"target_goal": "A" if seed < 10 else "B", "scope": 2},
        )
        goal = False
        while True:
            on = (obs - norm["mean"]) / np.sqrt(norm["var"] + EPS)
            a, _ = model.predict(np.clip(on, -10, 10), deterministic=True)
            obs, _r, term, trunc, info = env.step(a)
            if info.get("goal"):
                goal = True
            if term or trunc:
                break
        suc += int(goal)
        env.close()
    return suc


def main():
    obs, acts, seeds = load()
    train_mask = ~((seeds == 8) | (seeds == 9) | (seeds == 18) | (seeds == 19))
    val_mask = ~train_mask
    mean = obs.mean(axis=0).astype(np.float32)
    var = obs.var(axis=0).astype(np.float32)
    norm = {"mean": mean, "var": var, "count": float(len(obs))}
    with open(OUT / "junction_bc_vecnorm.pkl", "wb") as f:
        pickle.dump(norm, f)
    model, log = train(obs, acts, train_mask, val_mask, norm)
    model.save(str(OUT / "junction_bc_pretrain.zip"))
    with open(OUT / "junction_bc_training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_loss"])
        w.writeheader(); w.writerows(log)
    suc = evaluate(model, norm)
    print(f"BC 验证成功 {suc}/20")
    (OUT / "junction_bc_report.md").write_text(
        f"# Junction BC 预热\n\n- 训练 {int(train_mask.sum())} / 验证 {int(val_mask.sum())} 步\n"
        f"- 完整任务成功率：{suc}/20\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

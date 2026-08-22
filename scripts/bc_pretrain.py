"""BC 监督学习预热（SB3 框架内，零格式转换）+ 20 episode 环境验证。

流程：
  1. 加载 curve_vecnorm.pkl，归一化 6 条演示轨迹，按轨迹 5/1 划分训练/验证；
  2. 建 PPO(MlpPolicy, net_arch=[256,256])，只监督更新 actor 参数
     （mlp_extractor.policy_net + action_net），MSE 损失，不更新 value_net；
  3. model.save(bc_pretrain.zip)，记录训练曲线；
  4. 在 curve 环境跑 20 个 episode（每 episode 全新环境，防 warmstart 假成功），
     输出成功率/平均距离/最大距离，写入 bc_evaluation.md。
"""

from __future__ import annotations

import csv
import json
import pathlib
import pickle
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv

OUT = PROJECT_ROOT / "data" / "demo_trajectories"
NORM_PATH = OUT / "curve_vecnorm.pkl"
EPS = 1e-8
CLIP = 10.0

# BC 训练超参
LR = 1e-3
BATCH_SIZE = 512
EPOCHS = 400
VAL_PATIENCE = 30
VAL_MIN_DELTA = 1e-5
EVAL_EPISODES = 20


def normalize(obs: np.ndarray, norm: dict) -> np.ndarray:
    """与 PPO VecNormalize(norm_obs=True, clip_obs=10) 一致的手动归一化。"""
    o = (obs - norm["mean"]) / np.sqrt(norm["var"] + EPS)
    return np.clip(o, -CLIP, CLIP).astype(np.float32)


def build_dataset(norm: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 6 条轨迹，归一化，按轨迹划分 90/10（5 训练 / 1 验证）。"""
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    traj_ids: list[int] = []
    for seed in range(6):
        path = OUT / f"curve_demo_seed{seed:02d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=True)
        obs_all.append(d["obs"])
        act_all.append(d["actions"])
        traj_ids.extend([seed] * len(d["obs"]))

    obs = np.concatenate(obs_all).astype(np.float32)
    acts = np.concatenate(act_all).astype(np.float32)
    obs_n = normalize(obs, norm)

    # 按轨迹划分：5 条训练 / 1 条验证（保持轨迹完整）
    seeds = np.arange(6)
    rng = np.random.default_rng(0)
    rng.shuffle(seeds)
    val_seed = int(seeds[0])
    traj = np.asarray(traj_ids)
    val_mask = traj == val_seed
    train_mask = ~val_mask
    np.savez_compressed(
        OUT / "bc_dataset.npz",
        obs_norm=obs_n, actions=acts, train_mask=train_mask,
        val_mask=val_mask, trajectory_ids=traj,
    )
    print(f"数据集: 总 {len(obs_n)} 步, 训练 {int(train_mask.sum())}, "
          f"验证 {int(val_mask.sum())}（验证轨迹 seed{val_seed}）")
    return obs_n, acts, train_mask, val_mask


def train_bc(
    obs: np.ndarray,
    acts: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
    val_patience: int = VAL_PATIENCE,
) -> tuple[PPO, list[dict]]:
    """在 SB3 PPO 框架内做 actor 监督训练。"""
    env = make_vec_env(
        lambda: Go2wEnv(task="traverse_curve", domain_randomize=False), n_envs=1
    )
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs={"net_arch": [256, 256]},
        seed=0, verbose=0,
    )
    policy = model.policy
    # 只更新 actor 参数：policy_net(256-256) + action_net(256->6)，不动 value_net/log_std
    actor_params = (
        list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )
    optimizer = torch.optim.Adam(actor_params, lr=lr)
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
        model.policy.train()   # 注意：不能用 model.train()（那是 SB3 的 PPO 训练方法）
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(epoch))
        total = 0.0
        nbatch = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = policy.mlp_extractor.policy_net(x_tr[idx])
            pred = policy.action_net(pred)
            loss = mse(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(idx)
            nbatch += len(idx)
        train_loss = total / max(1, nbatch)

        model.policy.eval()
        with torch.no_grad():
            pred_v = policy.action_net(policy.mlp_extractor.policy_net(x_va))
            val_loss = float(mse(pred_v, y_va).item())
        log.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6)})
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val - VAL_MIN_DELTA:
            best_val = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= val_patience:
                print(f"早停于 epoch={epoch}（val_loss 连续 {patience} 轮未改善）")
                break
    return model, log


def evaluate(
    model: PPO,
    norm: dict,
    episodes: int = EVAL_EPISODES,
    deterministic: bool = True,
) -> dict:
    """20 个 episode，每 episode 全新环境（防 warmstart 假成功）。"""
    successes = 0
    distances: list[float] = []
    ep_lens: list[int] = []
    falls = 0
    for ep in range(episodes):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        obs, info = env.reset(seed=ep)
        max_prog = 0.0
        steps = 0
        while True:
            obs_n = normalize(obs, norm)
            action, _ = model.predict(obs_n, deterministic=deterministic)
            obs, _r, term, trunc, info = env.step(action)
            steps += 1
            max_prog = max(max_prog, float(info.get("progress", info.get("x", 0.0))))
            if term or trunc:
                break
        ep_lens.append(steps)
        distances.append(max_prog)
        if info.get("goal"):
            successes += 1
        elif term and not trunc:
            falls += 1
        env.close()
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / EVAL_EPISODES,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "mean_ep_len": float(np.mean(ep_lens)),
        "falls": falls,
    }


def main() -> None:
    with open(NORM_PATH, "rb") as f:
        norm = pickle.load(f)
    obs, acts, tr, va = build_dataset(norm)
    model, log = train_bc(obs, acts, tr, va)

    model_path = OUT / "bc_pretrain.zip"
    model.save(str(model_path))
    with open(OUT / "bc_training_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(log)

    result = evaluate(model, norm)
    print("验证结果:", result)
    verdict = ("成功（平均距离 >1.0m，可进入 PPO 微调）" if result["mean_distance"] > 1.0
               else "部分成功（0.5~1.0m，可尝试 PPO 但需更多训练）"
               if result["mean_distance"] >= 0.5
               else "失败（<0.5m，排查 norm/数据/训练）")
    print("判定:", verdict)

    config = {
        "demo": {"trajectories": 6, "steps": int(len(obs)), "params": {
            "lookahead": 0.7, "k_ang": 2.5, "k_dev": 0.5, "bias_clip": 1.0,
            "forward": 0.12, "noise_per_seed": {f"seed{i:02d}": v for i, v in
            enumerate([0.03, 0.03, 0.01, 0.01, 0.01, 0.005])}}},
        "vecnorm": {"method": "random_policy_10000_steps",
                    "mean_range": [round(float(norm["mean"].min()), 3),
                                   round(float(norm["mean"].max()), 3)],
                    "var_range": [round(float(norm["var"].min()), 4),
                                  round(float(norm["var"].max()), 4)]},
        "bc_train": {"optimizer": "Adam", "lr": LR, "batch_size": BATCH_SIZE,
                     "epochs_ran": len(log), "val_patience": VAL_PATIENCE,
                     "final_train_loss": log[-1]["train_loss"],
                     "final_val_loss": log[-1]["val_loss"],
                     "network": "Linear(61,256)-Tanh-Linear(256,256)-Tanh-Linear(256,6)",
                     "loss": "MSE", "updated_params": "policy_net+action_net"},
        "eval": result, "verdict": verdict,
        "files": {"vecnorm": "curve_vecnorm.pkl", "dataset": "bc_dataset.npz",
                  "model": "bc_pretrain.zip", "log": "bc_training_log.csv",
                  "eval": "bc_evaluation.md"},
    }
    with open(OUT / "bc_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(OUT / "bc_evaluation.md", "w", encoding="utf-8") as f:
        f.write(f"# BC 预热环境验证\n\n")
        f.write(f"- episodes: {result['episodes']}\n")
        f.write(f"- 成功率: {result['success_rate']:.0%}（{result['successes']}/{result['episodes']}）\n")
        f.write(f"- 平均距离: {result['mean_distance']:.3f} m\n")
        f.write(f"- 最大距离: {result['max_distance']:.3f} m\n")
        f.write(f"- 平均 ep_len: {result['mean_ep_len']:.1f}\n")
        f.write(f"- 摔倒次数: {result['falls']}\n")
        f.write(f"- 判定: {verdict}\n")
    print("已保存: bc_pretrain.zip / bc_dataset.npz / bc_training_log.csv / bc_config.json / bc_evaluation.md")


if __name__ == "__main__":
    main()

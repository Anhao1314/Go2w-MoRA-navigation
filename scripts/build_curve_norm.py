"""建立 curve 专用观测归一化统计（随机策略采集）。

BC 数据是原始 obs；PPO 训练时 obs 会经过 VecNormalize。这里用随机策略在
traverse_curve 环境采集 N 步原始 obs，计算 mean/var，保存为
data/demo_trajectories/curve_vecnorm.pkl（dict 格式，供 BC 训练/验证共用）。
"""

from __future__ import annotations

import pathlib
import pickle
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv

OUT_PATH = PROJECT_ROOT / "data" / "demo_trajectories" / "curve_vecnorm.pkl"
N_STEPS = 10000


def main() -> None:
    env = Go2wEnv(task="traverse_curve", domain_randomize=False)
    obs, _ = env.reset(seed=0)
    collected: list[np.ndarray] = []
    while len(collected) < N_STEPS:
        # 记录决策前 obs（与演示轨迹一致）
        collected.append(obs.astype(np.float32))
        action = env.action_space.sample()
        obs, _rew, term, trunc, _info = env.step(action)
        if term or trunc:
            obs, _ = env.reset(seed=0)
    env.close()

    arr = np.stack(collected)
    mean = arr.mean(axis=0).astype(np.float32)
    var = arr.var(axis=0).astype(np.float32)
    norm = {"mean": mean, "var": var, "count": len(arr), "n_steps": N_STEPS}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(norm, f)

    print("obs 样本数:", len(arr), "| 维度:", arr.shape[1])
    print("mean 范围: [%.3f, %.3f]" % (mean.min(), mean.max()))
    print("var  范围: [%.4f, %.4f]" % (var.min(), var.max()))
    print("NaN 检查: obs=%s mean=%s var=%s" % (
        bool(np.isnan(arr).any()), bool(np.isnan(mean).any()), bool(np.isnan(var).any())))
    print("已保存:", OUT_PATH)


if __name__ == "__main__":
    main()

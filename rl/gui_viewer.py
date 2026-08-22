"""在 MuJoCo 查看器中可视化已训练好的 RL 策略。

用法：
    python rl/gui_viewer.py --task balance \
        --model rl/runs/balance/seed00/best_model.zip \
        --vec-norm rl/runs/balance/seed00/best_vec_normalize.pkl

    python rl/gui_viewer.py --task traverse \
        --model rl/runs/traverse/seed00/best_model.zip \
        --vec-norm rl/runs/traverse/seed00/best_vec_normalize.pkl \
        --scenario narrow

窗口内按 R 可复位当前 episode。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv, SCENARIOS, TASK_SCENARIO, TRAVERSE_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description="RL 策略 MuJoCo 可视化")
    parser.add_argument(
        "--task",
        choices=["balance", "full_chain", "full_chain_simple", *TRAVERSE_TASKS],
        required=True,
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--vec-norm", type=str, required=True)
    parser.add_argument("--scenario", choices=[*SCENARIOS, "random"], default="random",
                        help="traverse 任务场景覆盖，默认取任务自身场景")
    parser.add_argument("--terrain", choices=["hfield", "boxes"], default="hfield",
                        help="traverse 斜坡任务地形：hfield=运行时高度场（默认），boxes=旧盒状斜坡")
    parser.add_argument("--stochastic", action="store_true",
                        help="使用随机动作采样而非确定性动作")
    args = parser.parse_args()

    default_sc = TASK_SCENARIO.get(args.task)
    env = Go2wEnv(
        task=args.task,
        domain_randomize=False,
        scenario=(
            default_sc if args.scenario == "random" else args.scenario
        ),
        terrain=args.terrain,
    )
    vec_env = DummyVecEnv([lambda: env])
    vec_norm = VecNormalize.load(str(args.vec_norm), vec_env)
    vec_norm.training = False
    model = PPO.load(str(args.model), device="cpu")

    reset_requested = False

    def on_key(key: int) -> None:
        nonlocal reset_requested
        if key == ord("R"):
            reset_requested = True

    obs, _ = env.reset()
    obs = vec_norm.normalize_obs(obs)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            if reset_requested:
                obs, _ = env.reset()
                obs = vec_norm.normalize_obs(obs)
                reset_requested = False
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, _, terminated, truncated, _ = env.step(action)
            obs = vec_norm.normalize_obs(obs)
            if terminated or truncated:
                obs, _ = env.reset()
                obs = vec_norm.normalize_obs(obs)
            viewer.sync()


if __name__ == "__main__":
    main()

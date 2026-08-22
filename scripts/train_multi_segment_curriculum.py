"""多段路径（A→B）三阶段课程学习：段1→A、段2→B、完整 A→B。

阶段1用 BC 预热模型初始化，阶段2/3 用上一阶段 best 初始化。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import pathlib
import pickle
import sys
import time
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
from mujoco_demos import common  # noqa: E402
from rl.go2w_env import MODEL_SCENARIO_PATH, Go2wEnv  # noqa: E402
from rl.eval import save_video  # noqa: E402
from rl.high_level_env_wrapper import HighLevelEnvWrapper  # noqa: E402
from rl.train import (  # noqa: E402
    NormFreezeCallback,
    make_lr_schedule,
    select_best_window,
)

STAGES = [
    {"name": "stage1", "scope": 0, "start_a": False, "steps": 500_000,
     "threshold": 0.8},
    {"name": "stage2", "scope": 1, "start_a": True, "steps": 500_000,
     "threshold": 0.8},
    {"name": "stage3", "scope": 2, "start_a": False, "steps": 1_000_000,
     "threshold": 0.6},
]
RUN_ROOT = PROJECT_ROOT / "rl" / "runs" / "traverse_curve_multi_segment" / "seed00_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "traverse_curve_multi_segment" / "seed00_v2"
EVAL_FREQ = 50_000
EVAL_EPISODES = 5


class MsegWrapper(HighLevelEnvWrapper):
    """在 reset 时注入 scope / start_at_subgoal。"""

    def __init__(self, base: Go2wEnv, scope: int, start_a: bool):
        super().__init__(
            base,
            append_mora=True,
            use_goal_condition=False,
            goal_min=1.4,
            goal_max=1.4,
        )
        self._scope = scope
        self._start_a = start_a

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        opts = dict(options or {})
        opts["scope"] = self._scope
        if self._start_a:
            opts["start_at_subgoal"] = "A"
        return super().reset(seed=seed, options=opts)


def make_stage_env(scope: int, start_a: bool) -> MsegWrapper:
    base = Go2wEnv(
        task="traverse_curve",
        domain_randomize=False,
        multi_segment=True,
        reward_version="mseg",
        scope=scope,
    )
    return MsegWrapper(base, scope=scope, start_a=start_a)


class MsegEvalCallback(BaseCallback):
    """阶段评估：成功率为 goal 比例，best 用平均距离（ep_len>500 才保存）。"""

    def __init__(
        self,
        save_dir: pathlib.Path,
        n_envs: int,
        scope: int,
        start_a: bool,
        threshold: float = 0.6,
        window: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = max(1, EVAL_FREQ // n_envs)
        self.save_dir = save_dir
        self.scope = scope
        self.start_a = start_a
        self.threshold = float(threshold)
        self.window = max(1, window)
        self.best_window_mean = -np.inf
        self._metric_history: list[tuple[float, float]] = []
        self._recent: list[dict] = []
        self.csv_path = save_dir / "eval_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "mean_reward", "mean_ep_len", "mean_x",
                 "success_rate", "passed_rate"]
            )
        dummy = make_stage_env(scope, start_a)
        self.eval_norm = VecNormalize(
            DummyVecEnv([lambda: dummy]),
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        train_vec = self.model.get_vec_normalize_env()
        if train_vec is not None:
            self.eval_norm.obs_rms = copy.deepcopy(train_vec.obs_rms)
        stats = self._run_eval()
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    round(stats["mean_reward"], 4),
                    round(stats["mean_ep_len"], 2),
                    round(stats["mean_x"], 4),
                    round(stats["success_rate"], 4),
                    round(stats["passed_rate"], 4),
                ]
            )
        if self.verbose:
            print(
                f"[mseg-eval] t={self.num_timesteps} "
                f"success={stats['success_rate']:.0%} x={stats['mean_x']:.2f} "
                f"ep_len={stats['mean_ep_len']:.0f} passed={stats['passed_rate']:.0%}"
            )
        self._metric_history.append((float(self.num_timesteps), float(stats["mean_x"])))
        window_mean, window_ts = select_best_window(
            self._metric_history, self.window, 0.05
        )
        if (
            window_mean is not None
            and window_mean > self.best_window_mean
            and stats["mean_ep_len"] > 500.0
        ):
            self.best_window_mean = window_mean
            self.model.save(str(self.save_dir / "best_model.zip"))
            if train_vec is not None:
                train_vec.save(str(self.save_dir / "best_vec_normalize.pkl"))
            if self.verbose:
                print(f"[mseg-best] x={window_mean:.3f} @ {window_ts:.0f}")
        self._recent.append(stats)
        self._recent = self._recent[-5:]
        if len(self._recent) >= 5 and all(
            r["success_rate"] >= self.threshold for r in self._recent[-5:]
        ):
            (self.save_dir / "stage_ok.txt").write_text("success", encoding="utf-8")
            return False
        return True

    def _run_eval(self) -> dict:
        rewards: list[float] = []
        lengths: list[int] = []
        xs: list[float] = []
        succ = 0
        passed = 0
        for ep in range(EVAL_EPISODES):
            env = make_stage_env(self.scope, self.start_a)
            obs, info = env.reset(seed=ep)
            ep_rew = 0.0
            steps = 0
            max_x = 0.0
            while True:
                obs_n = self.eval_norm.normalize_obs(obs)
                action, _ = self.model.predict(obs_n, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_rew += float(r)
                steps += 1
                max_x = max(max_x, float(info.get("progress", info.get("x", 0.0))))
                if info.get("passed_subgoal"):
                    passed += 1
                if info.get("goal"):
                    succ += 1
                if term or trunc:
                    break
            rewards.append(ep_rew)
            lengths.append(steps)
            xs.append(max_x)
            env.close()
        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_ep_len": float(np.mean(lengths)),
            "mean_x": float(np.mean(xs)),
            "success_rate": succ / EVAL_EPISODES,
            "passed_rate": passed / EVAL_EPISODES,
        }


def load_norm_dict(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def final_acceptance(model: PPO, vec: VecNormalize, episodes: int = 20) -> dict:
    succ = 0
    passed = 0
    xs: list[float] = []
    lens: list[int] = []
    times: list[float] = []
    falls = 0
    for ep in range(episodes):
        env = make_stage_env(scope=2, start_a=False)
        obs, info = env.reset(seed=ep)
        steps = 0
        max_x = 0.0
        ep_passed = False
        while True:
            obs_n = vec.normalize_obs(obs)
            action, _ = model.predict(obs_n, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            steps += 1
            max_x = max(max_x, float(info.get("progress", info.get("x", 0.0))))
            if info.get("passed_subgoal"):
                ep_passed = True
            if info.get("goal"):
                succ += 1
            if term or trunc:
                break
        xs.append(max_x)
        lens.append(steps)
        times.append(float(env.base_env.data.time))
        if ep_passed:
            passed += 1
        if term and not trunc and not info.get("goal"):
            falls += 1
        env.close()
    return {
        "success_rate": succ / episodes,
        "a_stop_rate": passed / episodes,
        "mean_x": float(np.mean(xs)),
        "mean_ep_len": float(np.mean(lens)),
        "mean_time": float(np.mean(times)),
        "falls": falls,
    }


def record_demo(model: PPO, vec: VecNormalize, prefix: pathlib.Path) -> None:
    env = make_stage_env(scope=2, start_a=False)
    renderer = mujoco.Renderer(common.load_model(MODEL_SCENARIO_PATH), 480, 640)
    obs, _ = env.reset(seed=0)
    frames: list[np.ndarray] = []
    step = 0
    while True:
        obs_n = vec.normalize_obs(obs)
        action, _ = model.predict(obs_n, deterministic=True)
        if step % 2 == 0:
            renderer.update_scene(env.base_env.data, camera="overview")
            frames.append(renderer.render().copy())
        obs, _r, term, trunc, info = env.step(action)
        step += 1
        if term or trunc or step >= 4000:
            break
    env.close()
    save_video(frames, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="多段路径三阶段课程学习")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", type=str, default=str(RUN_ROOT))
    parser.add_argument("--bc-model", type=str,
                        default=str(PROJECT_ROOT / "data" / "demo_trajectories" / "multi_segment_bc_pretrain.zip"))
    parser.add_argument("--bc-norm", type=str,
                        default=str(PROJECT_ROOT / "data" / "demo_trajectories" / "multi_segment_bc_vecnorm.pkl"))
    parser.add_argument("--stage2-bc-model", type=str,
                        default=str(PROJECT_ROOT / "data" / "demo_trajectories" / "multi_segment_stage2_bc_pretrain.zip"))
    parser.add_argument("--stage2-bc-norm", type=str,
                        default=str(PROJECT_ROOT / "data" / "demo_trajectories" / "multi_segment_stage2_bc_vecnorm.pkl"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_root = pathlib.Path(args.run_root)
    n_envs = 2 if args.smoke else args.envs
    stage_steps_scale = 10_000 if args.smoke else None
    prev_best = None
    prev_norm = None

    for stage in STAGES:
        stage_dir = run_root / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        steps: int = 10_000 if args.smoke else int(stage["steps"])
        lr = make_lr_schedule(
            type("A", (), {
                "learning_rate": 1.5e-4,
                "lr_warmup_steps": 100_000 if not args.smoke else 1_000,
                "lr_warmup_start": 5e-5,
                "lr_warmup_end": 1.5e-4,
            })(),
            steps,
        )
        train_env = make_vec_env(
            lambda: make_stage_env(stage["scope"], stage["start_a"]),
            n_envs=n_envs,
            seed=args.seed,
            vec_env_cls=DummyVecEnv,
        )
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
        )
        if stage["name"] == "stage1":
            nd = load_norm_dict(pathlib.Path(args.bc_norm))
            obs_rms: Any = train_env.obs_rms
            obs_rms.mean = np.asarray(nd["mean"], dtype=np.float32)
            obs_rms.var = np.asarray(nd["var"], dtype=np.float32)
            obs_rms.count = float(nd.get("count", 1e4))
            model = PPO.load(args.bc_model, env=train_env, device="cpu",
                             learning_rate=lr)
        elif stage["name"] == "stage2":
            nd = load_norm_dict(pathlib.Path(args.stage2_bc_norm))
            obs_rms: Any = train_env.obs_rms
            obs_rms.mean = np.asarray(nd["mean"], dtype=np.float32)
            obs_rms.var = np.asarray(nd["var"], dtype=np.float32)
            obs_rms.count = float(nd.get("count", 1e4))
            model = PPO.load(args.stage2_bc_model, env=train_env, device="cpu",
                             learning_rate=lr)
        else:
            assert prev_best is not None and prev_norm is not None
            train_env = VecNormalize.load(str(prev_norm), train_env)
            train_env.training = True
            train_env.norm_reward = False
            model = PPO.load(str(prev_best), env=train_env, device="cpu",
                             learning_rate=lr)
        model.ent_coef = 0.005
        model.tensorboard_log = str(stage_dir / "tensorboard")
        cb = MsegEvalCallback(
            save_dir=stage_dir,
            n_envs=n_envs,
            scope=stage["scope"],
            start_a=stage["start_a"],
            threshold=stage["threshold"],
            verbose=1,
        )
        callbacks: list[BaseCallback] = [cb]
        if not args.smoke:
            callbacks.append(NormFreezeCallback(500_000, verbose=1))
        print(f"\n===== {stage['name']} scope={stage['scope']} steps={steps} =====")
        model.learn(
            total_timesteps=steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
            tb_log_name=f"multi_segment_{stage['name']}",
        )
        model.save(str(stage_dir / "final_model.zip"))
        train_env.save(str(stage_dir / "final_vec_normalize.pkl"))
        (stage_dir / ".completed").touch()
        best = stage_dir / "best_model.zip"
        prev_best = best if best.exists() else stage_dir / "final_model.zip"
        prev_norm = (
            stage_dir / "best_vec_normalize.pkl"
            if best.exists()
            else stage_dir / "final_vec_normalize.pkl"
        )

    # 最终验收
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dummy = make_stage_env(scope=2, start_a=False)
    vec = VecNormalize.load(str(prev_norm), DummyVecEnv([lambda: dummy]))
    vec.training = False
    model = PPO.load(str(prev_best), device="cpu")
    acc = final_acceptance(model, vec)
    print("最终验收:", acc)
    (REPORT_DIR / "metrics.csv").write_text(
        "label,task,success_rate,a_stop_rate,mean_x,mean_ep_len,mean_time,falls\n"
        f"RL multi-segment,traverse_curve_multi_segment,"
        f"{acc['success_rate']},{acc['a_stop_rate']},{acc['mean_x']},"
        f"{acc['mean_ep_len']},{acc['mean_time']},{acc['falls']}\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "task": "traverse_curve_multi_segment",
                "seed": 0,
                "verdict": "pass" if acc["success_rate"] >= 0.6 else "fail",
                "success_rate": acc["success_rate"],
                "distance": acc["mean_x"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "report.md").write_text(
        "\n".join(
            [
                "# 多段路径 A→B RL 训练报告（MoRA 三项能力验证）",
                "",
                f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"- 完整成功率：{acc['success_rate']:.0%}",
                f"- A 点精确停止达标率：{acc['a_stop_rate']:.0%}",
                f"- 平均距离：{acc['mean_x']:.2f}m",
                f"- 平均 ep_len：{acc['mean_ep_len']:.0f}，平均时间：{acc['mean_time']:.1f}s",
                f"- 摔倒次数：{acc['falls']}",
                "",
                "各阶段曲线见 rl/runs/traverse_curve_multi_segment/seed00_v2/stage*/eval_log.csv",
            ]
        ),
        encoding="utf-8",
    )
    record_demo(
        model,
        vec,
        PROJECT_ROOT / "media" / "rl_traverse_curve_multi_segment_seed00_v2",
    )
    print("报告:", REPORT_DIR / "report.md")


if __name__ == "__main__":
    main()

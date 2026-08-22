"""MoRA-inspired 分层控制第一步：高层 RL 导航策略训练。

System 0 = LowLevelController（脚本控制器 + 高层 2 维指令），
System 1 = PPO 高层策略（观测 61 维不变，动作 2 维 [speed_scale, turn_adjust]）。

用法：
    python scripts/train_high_level_curve.py                     # 2M 步
    python scripts/train_high_level_curve.py --smoke             # 10k 步快速验证
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys
import time
from typing import Any

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv  # noqa: E402
from rl.go2w_env import MODEL_SCENARIO_PATH  # noqa: E402
from rl.eval import save_video  # noqa: E402
from rl.high_level_env_wrapper import HighLevelEnvWrapper  # noqa: E402
from rl.train import select_best_window  # noqa: E402
from mujoco_demos import common  # noqa: E402

DEFAULT_RUN_DIR = PROJECT_ROOT / "rl" / "runs" / "traverse_curve_high_level" / "seed00"
EVAL_EPISODES = 20
EVAL_FREQ = 50_000
ENV_KWARGS: dict[str, Any] = {
    "goal_min": 0.5,
    "goal_max": 1.4,
    "use_goal_condition": True,
    "her_buffer": None,
}
EVAL_FIXED_GOAL = 1.4
GENERALIZATION_GOALS = (0.65, 0.95, 1.25, 1.40, 0.30)


def make_high_level_env() -> HighLevelEnvWrapper:
    base = Go2wEnv(
        task="traverse_curve",
        domain_randomize=False,
        reward_version="v5",
        fall_penalty=-10.0,
        goal_bonus=200.0,
    )
    return HighLevelEnvWrapper(base, **ENV_KWARGS)


class HerAuxCallback(BaseCallback):
    """轻量 HER：用 future relabeling 样本对 actor 做加权 MSE 辅助更新。"""

    def __init__(
        self,
        her_buffer: list,
        update_freq: int,
        batch_size: int,
        save_dir: pathlib.Path,
        buffer_size: int = 20_000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.her_buffer = her_buffer
        self.update_freq = max(1, int(update_freq))
        self.batch_size = max(1, int(batch_size))
        self.buffer_size = max(1, int(buffer_size))
        self.her_log_path = save_dir / "her_log.csv"
        with open(self.her_log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "buffer_size", "loss", "positive_ratio"]
            )
        self._optimizer: Any = None

    def _on_step(self) -> bool:
        if self.num_timesteps <= 0 or self.num_timesteps % self.update_freq != 0:
            return True
        if len(self.her_buffer) > self.buffer_size:
            del self.her_buffer[: len(self.her_buffer) - self.buffer_size]
        if len(self.her_buffer) < self.batch_size:
            return True
        idx = np.random.default_rng(self.num_timesteps).choice(
            len(self.her_buffer), size=self.batch_size, replace=False
        )
        obs = np.stack([self.her_buffer[i][0] for i in idx]).astype(np.float32)
        acts = np.stack([self.her_buffer[i][1] for i in idx]).astype(np.float32)
        weights = np.asarray([self.her_buffer[i][2] for i in idx], dtype=np.float32)
        train_vec = self.model.get_vec_normalize_env()
        if train_vec is not None:
            obs = train_vec.normalize_obs(obs)
        policy: Any = self.model.policy
        if self._optimizer is None:
            params = (
                list(policy.mlp_extractor.policy_net.parameters())
                + list(policy.action_net.parameters())
            )
            self._optimizer = torch.optim.Adam(params, lr=3e-4)
        policy.train()
        x = torch.from_numpy(obs)
        y = torch.from_numpy(acts)
        w = torch.from_numpy(weights)
        pred = policy.action_net(policy.mlp_extractor.policy_net(x))
        loss = torch.mean(w * torch.mean((pred - y) ** 2, dim=1))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        policy.eval()
        with open(self.her_log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    len(self.her_buffer),
                    round(float(loss.item()), 6),
                    round(float((weights > 0).mean()), 4),
                ]
            )
        return True


class HighLevelEvalCallback(BaseCallback):
    """高层训练评估：距离优先选 best + 提前停止。"""

    def __init__(
        self,
        eval_freq_timesteps: int,
        n_eval_episodes: int,
        save_dir: pathlib.Path,
        n_envs: int,
        total_steps: int,
        goal_min: float = 0.5,
        goal_max: float = 1.4,
        eval_fixed_goal: float = 1.4,
        window_size: int = 3,
        std_ratio: float = 0.05,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = max(1, eval_freq_timesteps // n_envs)
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.total_steps = int(total_steps)
        self.goal_min = float(goal_min)
        self.goal_max = float(goal_max)
        self.eval_fixed_goal = float(eval_fixed_goal)
        self.window_size = max(1, window_size)
        self.std_ratio = float(std_ratio)
        self.best_window_mean = -np.inf
        self._metric_history: list[tuple[float, float]] = []
        self._recent: list[dict] = []
        self.csv_path = save_dir / "eval_log.csv"
        self.high_csv_path = save_dir / "high_action_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "mean_reward", "std_reward", "mean_ep_len",
                 "fixed_success_rate", "fixed_mean_x",
                 "random_success_rate", "random_mean_x"]
            )
        with open(self.high_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "speed_mean", "speed_std", "turn_mean", "turn_std"]
            )
        dummy = make_high_level_env()
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
                    round(stats["std_reward"], 4),
                    round(stats["mean_ep_len"], 2),
                    round(stats["fixed_success_rate"], 4),
                    round(stats["fixed_mean_x"], 4),
                    round(stats["random_success_rate"], 4),
                    round(stats["random_mean_x"], 4),
                ]
            )
        with open(self.high_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    round(stats["speed_mean"], 4),
                    round(stats["speed_std"], 4),
                    round(stats["turn_mean"], 4),
                    round(stats["turn_std"], 4),
                ]
            )
        if self.verbose:
            print(
                f"[hl-eval] t={self.num_timesteps} "
                f"rew={stats['mean_reward']:.2f} fixed={stats['fixed_mean_x']:.3f}"
                f"({stats['fixed_success_rate']:.0%}) "
                f"rand={stats['random_mean_x']:.3f}({stats['random_success_rate']:.0%}) "
                f"ep_len={stats['mean_ep_len']:.0f} "
                f"speed={stats['speed_mean']:.2f}±{stats['speed_std']:.2f} "
                f"turn={stats['turn_mean']:.2f}±{stats['turn_std']:.2f}"
            )

        best_x = (
            stats["random_mean_x"]
            if stats["random_mean_x"] is not None
            else stats["fixed_mean_x"]
        )
        self._metric_history.append((float(self.num_timesteps), float(best_x)))
        window_mean, window_ts = select_best_window(
            self._metric_history, self.window_size, self.std_ratio
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
                print(f"[hl-best] dist={window_mean:.3f} @ {window_ts:.0f}")

        self._recent.append(stats)
        self._recent = self._recent[-5:]
        stop, reason = self._early_stop(stats)
        if stop:
            (self.save_dir / "early_stop.txt").write_text(reason, encoding="utf-8")
            if self.verbose:
                print(f"[hl-stop] {reason}")
            return False
        return True

    def _run_eval(self) -> dict:
        fixed = self._run_goal_eval(fixed_goal=self.eval_fixed_goal)
        random = self._run_goal_eval(fixed_goal=None)
        return {
            "mean_reward": fixed["mean_reward"],
            "std_reward": fixed["std_reward"],
            "mean_ep_len": fixed["mean_ep_len"],
            "fixed_mean_x": fixed["mean_x"],
            "fixed_success_rate": fixed["success_rate"],
            "random_mean_x": random["mean_x"],
            "random_success_rate": random["success_rate"],
            "speed_mean": fixed["speed_mean"],
            "speed_std": fixed["speed_std"],
            "turn_mean": fixed["turn_mean"],
            "turn_std": fixed["turn_std"],
        }

    def _run_goal_eval(self, fixed_goal: float | None) -> dict:
        rewards: list[float] = []
        lengths: list[int] = []
        distances: list[float] = []
        successes = 0
        falls = 0
        speed_all: list[float] = []
        turn_all: list[float] = []
        for ep in range(self.n_eval_episodes):
            env = make_high_level_env()
            options = {"goal_arc": fixed_goal} if fixed_goal is not None else None
            obs, info = env.reset(seed=ep, options=options)
            ep_rew = 0.0
            steps = 0
            max_prog = 0.0
            fell = False
            while True:
                obs_n = self.eval_norm.normalize_obs(obs)
                action, _ = self.model.predict(obs_n, deterministic=True)
                speed_all.append(float(action[0]))
                turn_all.append(float(action[1]))
                obs, reward, term, trunc, info = env.step(action)
                ep_rew += float(reward)
                steps += 1
                max_prog = max(max_prog, float(info.get("progress", info.get("x", 0.0))))
                if info.get("goal"):
                    successes += 1
                if term and not trunc and not info.get("goal"):
                    fell = True
                if term or trunc:
                    break
            rewards.append(ep_rew)
            lengths.append(steps)
            distances.append(max_prog)
            if fell:
                falls += 1
            env.close()
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_ep_len": float(np.mean(lengths)),
            "mean_x": float(np.mean(distances)),
            "max_x": float(np.max(distances)),
            "success_rate": successes / max(1, self.n_eval_episodes),
            "falls": falls,
            "speed_mean": float(np.mean(speed_all)) if speed_all else 1.0,
            "speed_std": float(np.std(speed_all)) if speed_all else 0.0,
            "turn_mean": float(np.mean(turn_all)) if turn_all else 0.0,
            "turn_std": float(np.std(turn_all)) if turn_all else 0.0,
        }

    def _early_stop(self, stats: dict) -> tuple[bool, str]:
        recent = self._recent[-5:]
        if (
            len(recent) >= 5
            and all(r["fixed_success_rate"] == 0.0 for r in recent)
            and all(r["fixed_mean_x"] < 0.1 for r in recent)
        ):
            return True, "early_low_distance"
        if len(recent) >= 5 and all(
            r["fixed_success_rate"] >= 0.8 and r["random_success_rate"] >= 0.6
            for r in recent
        ):
            return True, "success"
        return False, ""


def run_acceptance(
    model: PPO,
    vec: VecNormalize,
    episodes: int,
    fixed_goal: float | None = None,
) -> dict:
    rewards: list[float] = []
    lengths: list[int] = []
    distances: list[float] = []
    successes = 0
    falls = 0
    max_pitch = 0.0
    speed_all: list[float] = []
    turn_all: list[float] = []
    for ep in range(episodes):
        env = make_high_level_env()
        options = {"goal_arc": fixed_goal} if fixed_goal is not None else None
        obs, info = env.reset(seed=ep, options=options)
        ep_rew = 0.0
        steps = 0
        max_prog = 0.0
        fell = False
        while True:
            obs_n = vec.normalize_obs(obs)
            action, _ = model.predict(obs_n, deterministic=True)
            speed_all.append(float(action[0]))
            turn_all.append(float(action[1]))
            obs, reward, term, trunc, info = env.step(action)
            ep_rew += float(reward)
            steps += 1
            max_prog = max(max_prog, float(info.get("progress", info.get("x", 0.0))))
            max_pitch = max(max_pitch, abs(float(info.get("pitch_err", 0.0))))
            if info.get("goal"):
                successes += 1
            if term and not trunc and not info.get("goal"):
                fell = True
            if term or trunc:
                break
        rewards.append(ep_rew)
        lengths.append(steps)
        distances.append(max_prog)
        if fell:
            falls += 1
        env.close()
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
        "mean_ep_len": float(np.mean(lengths)),
        "falls": falls,
        "mean_total_reward": float(np.mean(rewards)),
        "max_pitch": max_pitch,
        "speed_mean": float(np.mean(speed_all)) if speed_all else 1.0,
        "speed_std": float(np.std(speed_all)) if speed_all else 0.0,
        "turn_mean": float(np.mean(turn_all)) if turn_all else 0.0,
        "turn_std": float(np.std(turn_all)) if turn_all else 0.0,
    }


def run_generalization(
    model: PPO,
    vec: VecNormalize,
    goals: tuple[float, ...],
    episodes: int = 10,
) -> list[dict]:
    rows: list[dict] = []
    for g in goals:
        acc = run_acceptance(model, vec, episodes=episodes, fixed_goal=g)
        rows.append(
            {
                "goal_arc": g,
                "success_rate": acc["success_rate"],
                "mean_distance": acc["mean_distance"],
                "mean_ep_len": acc["mean_ep_len"],
                "falls": acc["falls"],
            }
        )
    return rows


def write_report(
    run_dir: pathlib.Path,
    fixed_acceptance: dict,
    random_acceptance: dict,
    generalization: list[dict],
    early_stop: str,
    eval_rows: list[dict],
    report_dir: pathlib.Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    table = "\n".join(
        f"| {r['timesteps']} | {r['mean_reward']:.2f} | {r['mean_ep_len']:.0f} "
        f"| {r['fixed_mean_x']:.3f} | {r['fixed_success_rate']:.0%} "
        f"| {r['random_mean_x']:.3f} | {r['random_success_rate']:.0%} |"
        for r in eval_rows
    )
    gen_table = "\n".join(
        f"| {r['goal_arc']:.2f} | {r['success_rate']:.0%} | "
        f"{r['mean_distance']:.3f} | {r['mean_ep_len']:.0f} | {r['falls']} |"
        for r in generalization
    )
    use_goal = bool(ENV_KWARGS.get("use_goal_condition", True))
    obs_dim = 63 if use_goal else 61
    text = "\n".join(
        [
            "# traverse_curve 分层控制（MoRA B+：目标多样化 + HER）训练报告",
            "",
            f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 架构",
            "",
            "- System 0：`LowLevelController`（前视跟踪 + 差速转向，forward=0.12）",
            "- System 1：PPO 高层策略，动作 2 维 [speed_scale, turn_adjust]",
            f"- 观测：{obs_dim} 维（{'61+remaining_arc+target_heading' if use_goal else '61，无目标条件'}）",
            f"- 目标多样化：U({ENV_KWARGS.get('goal_min')}, {ENV_KWARGS.get('goal_max')})；"
            f"HER：{'开' if ENV_KWARGS.get('her_buffer') is not None else '关'}",
            "- 奖励：v5（摔倒 -10、到达 +200）",
            "",
            "## 训练配置",
            "",
            f"- 总步数目标：2M；实际提前停止：{early_stop or '未触发'}",
            "- envs=4，lr=3e-4，ent_coef=0.01，best=随机目标距离稳定窗口（ep_len>500）",
            f"- run 目录：`{run_dir}`",
            "",
            "## 训练曲线（每 50k 评估：固定目标 vs 随机目标）",
            "",
            "| timesteps | reward | ep_len | fixed_x | fixed_succ | rand_x | rand_succ |",
            "|---|---|---|---|---|---|---|",
            table,
            "",
            "## 正式验收（20 episodes，全新环境）",
            "",
            f"- 固定目标 {EVAL_FIXED_GOAL:.2f}m：成功率 {fixed_acceptance['success_rate']:.0%}"
            f"（{fixed_acceptance['successes']}/{fixed_acceptance['episodes']}），"
            f"平均距离 {fixed_acceptance['mean_distance']:.3f}m，ep_len {fixed_acceptance['mean_ep_len']:.0f}，"
            f"摔倒 {fixed_acceptance['falls']}，奖励 {fixed_acceptance['mean_total_reward']:.2f}",
            f"- 随机目标 U({ENV_KWARGS.get('goal_min')}, {ENV_KWARGS.get('goal_max')})："
            f"成功率 {random_acceptance['success_rate']:.0%}，"
            f"平均距离 {random_acceptance['mean_distance']:.3f}m，"
            f"ep_len {random_acceptance['mean_ep_len']:.0f}，摔倒 {random_acceptance['falls']}",
            f"- 高层动作（固定目标）：speed={fixed_acceptance['speed_mean']:.2f}±{fixed_acceptance['speed_std']:.2f}，"
            f"turn={fixed_acceptance['turn_mean']:.2f}±{fixed_acceptance['turn_std']:.2f}",
            "",
            "## 轻量泛化验证（10 episodes/目标）",
            "",
            "| goal_arc | 成功率 | 平均距离 | ep_len | 摔倒 |",
            "|---|---|---|---|---|",
            gen_table,
            "",
            "## 基线对比",
            "",
            "| 方法 | 固定目标成功率 | 平均距离 |",
            "|---|---|---|",
            "| A步骤 seed01（固定1.4，无目标条件） | 100% | 1.355m |",
            "| B步骤 seed02（固定1.4，目标条件） | 100% | 1.353m |",
            f"| B+ 本 run（目标多样化{'，目标条件' if use_goal else '，无目标条件'}） | "
            f"{fixed_acceptance['success_rate']:.0%} | {fixed_acceptance['mean_distance']:.3f}m |",
            "",
            "## 结论",
            "",
            (
                "✅ 目标多样化 + HER 训练成功，随机目标具备泛化能力"
                if random_acceptance["success_rate"] >= 0.6
                else "❌ 随机目标未达标，需分析目标范围/HER/训练步数"
            ),
            "",
        ]
    )
    (report_dir / "report.md").write_text(text, encoding="utf-8")
    metrics = {
        "label": "RL high-level (MoRA B+)",
        "task": "traverse_curve",
        "max_dev": round(fixed_acceptance["max_pitch"], 6),
        "min_clear": "",
        "dual_hold": 0.0,
        "recovered": False,
        "settle_seconds": "",
        "success": bool(random_acceptance["success_rate"] >= 0.6),
        "success_rate": round(random_acceptance["success_rate"], 4),
        "distance": round(random_acceptance["mean_distance"], 4),
        "time_to_goal": "",
        "falls": round(fixed_acceptance["falls"], 2),
        "total_reward": round(fixed_acceptance["mean_total_reward"], 4),
        "mean_base_reward": 0.0,
        "nan": False,
    }
    with open(report_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "label", "task", "max_dev", "min_clear", "dual_hold", "recovered",
                "settle_seconds", "success", "success_rate", "distance",
                "time_to_goal", "falls", "total_reward", "mean_base_reward", "nan",
            ],
        )
        w.writeheader()
        w.writerow(metrics)
    verdict = "pass" if metrics["success"] else "fail"
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": "traverse_curve_high_level",
                "seed": 0,
                "verdict": verdict,
                "success_rate": metrics["success_rate"],
                "distance": metrics["distance"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def read_eval_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "timesteps": int(rec["timesteps"]),
                        "mean_reward": float(rec["mean_reward"]),
                        "mean_ep_len": float(rec["mean_ep_len"]),
                        "fixed_mean_x": float(
                            rec.get("fixed_mean_x", rec.get("mean_x", 0.0))
                        ),
                        "fixed_success_rate": float(
                            rec.get("fixed_success_rate", rec.get("success_rate", 0.0))
                        ),
                        "random_mean_x": float(
                            rec.get("random_mean_x", rec.get("mean_x", 0.0))
                        ),
                        "random_success_rate": float(
                            rec.get("random_success_rate", rec.get("success_rate", 0.0))
                        ),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def finalize(run_dir: pathlib.Path) -> dict:
    """用 best/final 模型跑 20-episode 验收并生成报告。"""
    if run_dir.name.startswith("_") or run_dir.parent.name == "_smoke":
        # smoke 目录不写正式报告/量化产物
        best = run_dir / "best_model.zip"
        model_path = best if best.exists() else run_dir / "final_model.zip"
        norm_path = (
            run_dir / "best_vec_normalize.pkl"
            if best.exists()
            else run_dir / "final_vec_normalize.pkl"
        )
        dummy = make_high_level_env()
        vec = VecNormalize.load(str(norm_path), DummyVecEnv([lambda: dummy]))
        vec.training = False
        model = PPO.load(str(model_path), device="cpu")
        acceptance = run_acceptance(model, vec, episodes=EVAL_EPISODES)
        print("smoke 验收:", acceptance)
        return acceptance
    early_stop = ""
    stop_path = run_dir / "early_stop.txt"
    if stop_path.exists():
        early_stop = stop_path.read_text(encoding="utf-8").strip()
    best = run_dir / "best_model.zip"
    model_path = best if best.exists() else run_dir / "final_model.zip"
    norm_path = (
        run_dir / "best_vec_normalize.pkl"
        if best.exists()
        else run_dir / "final_vec_normalize.pkl"
    )
    dummy = make_high_level_env()
    vec = VecNormalize.load(
        str(norm_path), DummyVecEnv([lambda: dummy])
    )
    vec.training = False
    model = PPO.load(str(model_path), device="cpu")
    fixed_acceptance = run_acceptance(
        model, vec, episodes=EVAL_EPISODES, fixed_goal=EVAL_FIXED_GOAL
    )
    random_acceptance = run_acceptance(model, vec, episodes=EVAL_EPISODES)
    generalization = run_generalization(
        model, vec, GENERALIZATION_GOALS, episodes=10
    )
    with open(run_dir / "generalization.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "goal_arc", "success_rate", "mean_distance",
                "mean_ep_len", "falls",
            ],
        )
        w.writeheader()
        w.writerows(generalization)
    report_dir = PROJECT_ROOT / "reports" / run_dir.parent.name / run_dir.name
    write_report(
        run_dir,
        fixed_acceptance,
        random_acceptance,
        generalization,
        early_stop,
        read_eval_rows(run_dir / "eval_log.csv"),
        report_dir,
    )
    print("固定目标验收:", fixed_acceptance)
    print("随机目标验收:", random_acceptance)
    print("泛化:", generalization)
    print("报告:", report_dir / "report.md")
    return random_acceptance


def record_demo(
    run_dir: pathlib.Path,
    prefix: pathlib.Path,
    camera: str = "overview",
) -> None:
    """用 best/final 模型录制一个全局俯视演示视频（mp4 + gif）。"""
    best = run_dir / "best_model.zip"
    model_path = best if best.exists() else run_dir / "final_model.zip"
    norm_path = (
        run_dir / "best_vec_normalize.pkl"
        if best.exists()
        else run_dir / "final_vec_normalize.pkl"
    )
    dummy = make_high_level_env()
    vec = VecNormalize.load(str(norm_path), DummyVecEnv([lambda: dummy]))
    vec.training = False
    model = PPO.load(str(model_path), device="cpu")
    env = make_high_level_env()
    renderer = mujoco.Renderer(common.load_model(MODEL_SCENARIO_PATH), 480, 640)
    obs, info = env.reset(seed=0)
    frames: list[np.ndarray] = []
    step_i = 0
    while True:
        obs_n = vec.normalize_obs(obs)
        action, _ = model.predict(obs_n, deterministic=True)
        if step_i % 2 == 0:
            renderer.update_scene(env.base_env.data, camera=camera)
            frames.append(renderer.render().copy())
        obs, _r, term, trunc, info = env.step(action)
        step_i += 1
        if term or trunc or step_i >= 1500:
            break
    env.close()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="高层 RL 导航策略训练（MoRA 第一步）")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--smoke", action="store_true",
                        help="10k 步、2 envs 快速验证")
    parser.add_argument("--report-only", action="store_true",
                        help="跳过训练，只用已有模型生成验收报告")
    parser.add_argument("--record", type=str, default=None,
                        help="演示视频输出前缀，如 media/rl_traverse_curve_high_level_seed01")
    parser.add_argument("--camera", type=str, default="overview",
                        help="录制相机（overview=全局俯视，track=跟随）")
    parser.add_argument("--record-only", action="store_true",
                        help="跳过训练/验收，只用已有模型录制演示视频")
    parser.add_argument("--goal-min", type=float, default=0.5,
                        help="随机目标弧长最小值")
    parser.add_argument("--goal-max", type=float, default=1.4,
                        help="随机目标弧长最大值")
    parser.add_argument("--use-her", action="store_true",
                        help="启用轻量 HER 回调（future relabeling + 辅助更新）")
    parser.add_argument("--no-goal-condition", action="store_true",
                        help="关闭目标条件输入（观测回到 61 维，消融用）")
    parser.add_argument("--eval-fixed-goal", type=float, default=1.4,
                        help="固定目标评估的 goal_arc")
    parser.add_argument("--her-update-freq", type=int, default=10_000,
                        help="HER 辅助更新频率（环境步数）")
    parser.add_argument("--her-buffer-size", type=int, default=20_000,
                        help="HER buffer 容量上限")
    parser.add_argument("--her-batch-size", type=int, default=256,
                        help="HER 辅助更新 batch size")
    args = parser.parse_args()

    global ENV_KWARGS, EVAL_FIXED_GOAL
    her_buffer: list = []
    ENV_KWARGS.update(
        {
            "goal_min": args.goal_min,
            "goal_max": args.goal_max,
            "use_goal_condition": not args.no_goal_condition,
            "her_buffer": her_buffer if args.use_her else None,
        }
    )
    EVAL_FIXED_GOAL = args.eval_fixed_goal

    n_envs = 2 if args.smoke else args.envs
    total_steps = 10_000 if args.smoke else args.total_steps
    eval_freq = 5_000 if args.smoke else args.eval_freq
    run_dir = pathlib.Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        finalize(run_dir)
        return
    if args.record_only:
        if not args.record:
            raise SystemExit("--record-only 需要 --record 指定输出前缀")
        record_demo(run_dir, pathlib.Path(args.record), args.camera)
        return

    train_env = make_vec_env(
        make_high_level_env,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv if args.use_her else None,
    )
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
    )
    callback = HighLevelEvalCallback(
        eval_freq_timesteps=eval_freq,
        n_eval_episodes=args.eval_episodes,
        save_dir=run_dir,
        n_envs=n_envs,
        total_steps=total_steps,
        goal_min=args.goal_min,
        goal_max=args.goal_max,
        eval_fixed_goal=args.eval_fixed_goal,
        verbose=1,
    )
    callbacks: list[BaseCallback] = [callback]
    if args.use_her:
        callbacks.append(
            HerAuxCallback(
                her_buffer=her_buffer,
                update_freq=args.her_update_freq,
                batch_size=args.her_batch_size,
                save_dir=run_dir,
                buffer_size=args.her_buffer_size,
                verbose=1,
            )
        )
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList(callbacks),
        tb_log_name="traverse_curve_high_level",
    )
    model.save(str(run_dir / "final_model.zip"))
    train_env.save(str(run_dir / "final_vec_normalize.pkl"))
    (run_dir / ".completed").touch()

    finalize(run_dir)
    if not (run_dir.name.startswith("_") or run_dir.parent.name == "_smoke"):
        record_prefix = (
            pathlib.Path(args.record)
            if args.record
            else PROJECT_ROOT / "media" / f"rl_{run_dir.parent.name}_{run_dir.name}"
        )
        record_demo(run_dir, record_prefix, args.camera)


if __name__ == "__main__":
    main()

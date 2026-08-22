"""SB3 PPO 训练入口（Phase 1 balance / Phase 2 full_chain）。

用法示例：
    python rl/train.py --task balance --total-steps 8000000 --seed 0
    python rl/train.py --task full_chain --total-steps 8000000 --seed 0 \
        --init-from rl/runs/balance/seed00/best_model.zip
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnvWrapper,
    VecNormalize,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import TRAVERSE_TASKS  # noqa: E402

from rl.go2w_env import Go2wEnv
from rl.go2w_env import CURVE_HALF_WIDTH, TRAVERSE_HEADING_PENALTY


def touch_sync_trigger() -> None:
    """训练事件触发自动同步（只写触发文件，不直接操作 git）。"""
    try:
        (PROJECT_ROOT / "rl" / ".autosync.trigger").touch()
    except OSError:
        pass


def select_best_window(
    points: list[tuple[float, float]],
    size: int = 3,
    std_ratio: float = 0.05,
) -> tuple[float | None, float | None]:
    """在评估历史里选“连续 size 个点、奖励方差小且均值最高”的窗口。

    points: [(timesteps, metric), ...]；返回 (窗口均值, 窗口末端步数)，
    没有任何窗口达标时返回 (None, None)。绝对方差下限 0.1，避免均值为 0 时误判。
    """
    if size <= 0 or std_ratio <= 0.0 or len(points) < size:
        return None, None
    best_mean: float | None = None
    best_ts: float | None = None
    for i in range(len(points) - size + 1):
        window = [v for _ts, v in points[i : i + size]]
        mean = float(np.mean(window))
        std = float(np.std(window))
        floor = max(abs(mean) * std_ratio, 0.1)
        if std > floor:
            continue
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_ts = float(points[i + size - 1][0])
    return best_mean, best_ts


def train_config_payload(args: Any) -> dict:
    """把本轮超参与关键环境常量落盘，保证可复现。"""
    traverse = args.task in TRAVERSE_TASKS
    return {
        "task": args.task,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "envs": args.envs,
        "fresh": bool(args.fresh),
        "init_from": args.init_from,
        "resume_from": args.resume_from,
        "resume_norm": args.resume_norm,
        "learning_rate": args.learning_rate,
        "ent_coef": getattr(args, "ent_coef", 0.0),
        "init_norm": args.init_norm,
        "corridor_width": args.corridor_width,
        "curve_amplitude": args.curve_amplitude,
        "reward_version": args.reward_version,
        "fall_penalty": getattr(args, "fall_penalty", -5.0),
        "goal_bonus": getattr(args, "goal_bonus", 100.0),
        "terrain": args.terrain,
        "curriculum_steps": args.curriculum_steps,
        "heading_penalty": args.heading_penalty,
        "best_window_size": args.best_window_size,
        "best_window_std_ratio": args.best_window_std_ratio,
        "best_metric": getattr(args, "best_metric", "reward"),
        "lr_warmup_steps": getattr(args, "lr_warmup_steps", 0),
        "lr_warmup_start": getattr(args, "lr_warmup_start", 5e-5),
        "lr_warmup_end": getattr(args, "lr_warmup_end", 1.5e-4),
        "norm_freeze_steps": getattr(args, "norm_freeze_steps", 0),
        "exploration_boost": bool(getattr(args, "exploration_boost", False)),
        "boost_window": getattr(args, "boost_window", 5),
        "boost_reward_threshold": getattr(args, "boost_reward_threshold", -5.0),
        "boost_ent_coef": getattr(args, "boost_ent_coef", 0.01),
        "boost_lr": getattr(args, "boost_lr", 5e-4),
        "boost_noise_std": getattr(args, "boost_noise_std", 0.1),
        "boost_cycles": getattr(args, "boost_cycles", 10),
        "curve_half_width": CURVE_HALF_WIDTH,
        "default_heading_penalty": TRAVERSE_HEADING_PENALTY,
        "obs_dim": 61 if traverse else 44,
        "action_dim": 6 if traverse else 2,
    }


def pick_best_metric(
    task: str,
    best_metric: str,
    score: float,
    mean_x: float,
    mean_r: float,
) -> float:
    """best 选择指标：distance 模式用平均前进距离，否则用任务分/奖励。"""
    if task in TRAVERSE_TASKS:
        return mean_x if best_metric == "distance" else score
    return mean_r


def make_lr_schedule(args: Any, total_steps: int) -> Any:
    """学习率调度：warmup_steps 前用 warmup_start，之后用 warmup_end。"""
    warmup = int(getattr(args, "lr_warmup_steps", 0) or 0)
    if warmup <= 0:
        return args.learning_rate
    start = float(getattr(args, "lr_warmup_start", 5e-5))
    end = float(getattr(args, "lr_warmup_end", 1.5e-4))
    total = max(1, int(total_steps))

    def schedule(progress_remaining: float) -> float:
        steps_done = (1.0 - float(progress_remaining)) * total
        return start if steps_done < warmup else end

    return schedule


def norm_should_train(num_timesteps: int, freeze_steps: int) -> bool:
    """norm 冻结策略：freeze_steps 之前不更新 obs_rms，之后放开。"""
    return int(num_timesteps) >= max(0, int(freeze_steps))


def build_vec_env(
    task: str,
    n_envs: int,
    seed: int,
    domain_randomize: bool,
    vec_env_cls: type[DummyVecEnv] | type[SubprocVecEnv] | None = SubprocVecEnv,
    disturbance: bool = False,
    terrain: str = "hfield",
    heading_penalty: float = TRAVERSE_HEADING_PENALTY,
    corridor_width: float = 0.40,
    curve_amplitude: float = 0.35,
    reward_version: str = "v4",
    fall_penalty: float = -5.0,
    goal_bonus: float = 100.0,
):
    def make_env() -> Go2wEnv:
        return Go2wEnv(
            task=task,
            domain_randomize=domain_randomize,
            disturbance=disturbance,
            terrain=terrain,
            heading_penalty=heading_penalty,
            corridor_width=corridor_width,
            curve_amplitude=curve_amplitude,
            reward_version=reward_version,
            fall_penalty=fall_penalty,
            goal_bonus=goal_bonus,
        )

    return make_vec_env(
        make_env,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
    )


class ActionNoiseVecEnv(VecEnvWrapper):
    """训练期动作噪声注入：boost 激活时在 step_async 前给 action 加高斯噪声。"""

    def __init__(self, venv, noise_state: dict):
        super().__init__(venv)
        self._noise_state = noise_state

    def step_async(self, actions) -> None:
        if self._noise_state.get("active"):
            std = float(self._noise_state.get("std", 0.1))
            arr = np.asarray(actions, dtype=np.float32)
            noise = np.random.normal(0.0, std, size=arr.shape).astype(np.float32)
            actions = np.clip(arr + noise, -1.0, 1.0)
        self.venv.step_async(actions)

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        return self.venv.step_wait()


class EvalAndSaveCallback(BaseCallback):
    """周期性评估，保存最优模型与 VecNormalize 统计，并输出 CSV 曲线。"""

    def __init__(
        self,
        eval_env: VecNormalize,
        eval_freq_timesteps: int,
        n_eval_episodes: int,
        save_dir: pathlib.Path,
        n_envs: int,
        task: str,
        window_size: int = 3,
        window_std_ratio: float = 0.05,
        best_metric: str = "reward",
        boost_state: dict | None = None,
        boost_window: int = 5,
        boost_reward_threshold: float = -5.0,
        boost_ent_coef: float = 0.01,
        boost_lr: float = 5e-4,
        boost_noise_std: float = 0.1,
        boost_cycles: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.task = task
        self.eval_env = eval_env
        self.eval_freq = max(1, eval_freq_timesteps // n_envs)
        self.n_eval_episodes = n_eval_episodes
        self.save_dir = save_dir
        self.window_size = max(1, window_size)
        self.window_std_ratio = float(window_std_ratio)
        self.best_metric = best_metric
        self.best_mean_reward = -np.inf
        self.best_task_score = -np.inf
        self.best_window_mean = -np.inf
        self._metric_history: list[tuple[float, float]] = []
        self._recent_evals: list[tuple[float, float, float, float | None]] = []
        self._boost_remaining = 0
        self._base_ent_coef: float | None = None
        self._base_lr_schedule = None
        self.boost_state = boost_state
        self.boost_window = max(1, int(boost_window))
        self.boost_reward_threshold = float(boost_reward_threshold)
        self.boost_ent_coef = float(boost_ent_coef)
        self.boost_lr = float(boost_lr)
        self.boost_noise_std = float(boost_noise_std)
        self.boost_cycles = max(1, int(boost_cycles))
        self.boost_log_path = save_dir / "boost_log.csv"
        if boost_state is not None:
            with open(self.boost_log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["timesteps", "event", "mean_reward", "mean_ep_len",
                     "ent_coef", "learning_rate", "noise_std"]
                )
        self.csv_path = save_dir / "eval_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "mean_reward", "std_reward", "mean_ep_len"]
            )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        train_vec_norm = self.model.get_vec_normalize_env()
        if train_vec_norm is not None and hasattr(train_vec_norm, "obs_rms"):
            self.eval_env.obs_rms = copy.deepcopy(train_vec_norm.obs_rms)

        rewards, lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards))
        mean_len = float(np.mean(lengths))
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [self.num_timesteps, round(mean_r, 4), round(std_r, 4), round(mean_len, 2)]
            )
        if self.verbose:
            print(
                f"[eval] timesteps={self.num_timesteps} "
                f"mean_reward={mean_r:.3f}±{std_r:.3f} mean_ep_len={mean_len:.1f}"
            )

        if self.task == "balance":
            min_ok_len = 200.0
        elif self.task == "full_chain_simple":
            min_ok_len = 150.0
        elif self.task in TRAVERSE_TASKS:
            # traverse 通过 task score 做质量门槛，ep_len 只排除“立刻终止”的极端情况，
            # 避免正常训练早期（ep_len 常为 150~300）长期不保存 checkpoint 导致中断后无法续训。
            min_ok_len = 20.0
        else:  # full_chain
            min_ok_len = 300.0
        if mean_len < min_ok_len:
            print(
                f"[warn] 评估异常：ep_len={mean_len:.1f} < {min_ok_len:.0f}，"
                f"跳过本次 best 保存，避免弱模型覆盖好权重"
            )
            return True

        if self.task in TRAVERSE_TASKS:
            score, success_rate, mean_x = self._traverse_task_score()
            if self.verbose:
                print(
                    f"[task] score={score:.2f} success={success_rate:.0%} mean_x={mean_x:.2f}"
                )
        else:
            score = mean_r
            mean_x = None
        metric = pick_best_metric(
            self.task, self.best_metric, score, mean_x if mean_x is not None else 0.0, mean_r
        )
        if self.boost_state is not None:
            self._update_boost(mean_r, std_r, mean_len, mean_x)
        self._metric_history.append((float(self.num_timesteps), float(metric)))
        window_mean, window_ts = select_best_window(
            self._metric_history, self.window_size, self.window_std_ratio
        )
        can_save = self.best_metric != "distance" or mean_len > 500.0
        if (
            window_mean is not None
            and window_mean > self.best_window_mean
            and can_save
        ):
            self.best_window_mean = window_mean
            if self.verbose:
                print(
                    f"[best-window] mean={window_mean:.3f} @ {window_ts:.0f} "
                    f"窗口数={len(self._metric_history)}"
                )
            self.model.save(str(self.save_dir / "best_model.zip"))
            if train_vec_norm is not None:
                train_vec_norm.save(str(self.save_dir / "best_vec_normalize.pkl"))
        touch_sync_trigger()
        return True

    def _update_boost(
        self,
        mean_r: float,
        std_r: float,
        mean_len: float,
        mean_x: float | None = None,
    ) -> None:
        self._recent_evals.append((mean_r, std_r, mean_len, mean_x))
        self._recent_evals = self._recent_evals[-self.boost_window:]
        standing_stalled = (
            len(self._recent_evals) >= self.boost_window
            and all(295.0 <= r[2] <= 305.0 for r in self._recent_evals)
            and all(abs(r[1]) < 0.5 for r in self._recent_evals)
            and max(r[0] for r in self._recent_evals) < self.boost_reward_threshold
        )
        fall_stalled = (
            len(self._recent_evals) >= self.boost_window
            and all(r[2] < 100.0 for r in self._recent_evals)
            and float(np.std([r[2] for r in self._recent_evals])) < 20.0
            and all(
                r[3] is not None and r[3] < 0.1
                for r in self._recent_evals
            )
        )
        stalled = standing_stalled or fall_stalled
        mode = "standing" if standing_stalled else "falling"
        if self._boost_remaining > 0:
            if not stalled:
                self._restore_boost(mean_r, mean_len)
                return
            self._boost_remaining -= 1
            if self._boost_remaining <= 0:
                self._restore_boost(mean_r, mean_len)
            return
        if stalled:
            self._start_boost(mean_r, mean_len, mode)

    def _start_boost(self, mean_r: float, mean_len: float, mode: str) -> None:
        state = self.boost_state
        if state is None:
            return
        model: Any = self.model
        self._base_ent_coef = float(model.ent_coef)
        self._base_lr_schedule = model.lr_schedule
        model.ent_coef = self.boost_ent_coef
        model.lr_schedule = lambda _: self.boost_lr
        self._boost_remaining = self.boost_cycles
        state["active"] = True
        state["std"] = self.boost_noise_std
        self._log_boost(f"boost_start_{mode}", mean_r, mean_len)
        if self.verbose:
            print(
                f"[boost] 检测到{mode}局部最优，启动探索增强 "
                f"{self.boost_cycles} 个 eval 周期："
                f"ent_coef={self.boost_ent_coef} lr={self.boost_lr} "
                f"noise_std={self.boost_noise_std}"
            )

    def _restore_boost(self, mean_r: float, mean_len: float) -> None:
        state = self.boost_state
        if state is None:
            return
        model: Any = self.model
        if self._base_ent_coef is not None:
            model.ent_coef = self._base_ent_coef
        if self._base_lr_schedule is not None:
            model.lr_schedule = self._base_lr_schedule
        self._base_ent_coef = None
        self._base_lr_schedule = None
        self._boost_remaining = 0
        state["active"] = False
        self._log_boost("boost_end", mean_r, mean_len)
        if self.verbose:
            print("[boost] 探索增强结束，参数已恢复")

    def _log_boost(self, event: str, mean_r: float, mean_len: float) -> None:
        state = self.boost_state
        if state is None:
            return
        model: Any = self.model
        with open(self.boost_log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    event,
                    round(mean_r, 4),
                    round(mean_len, 2),
                    model.ent_coef,
                    self.boost_lr if self._boost_remaining > 0 else model.learning_rate,
                    state.get("std", 0.0) if state.get("active") else 0.0,
                ]
            )

    def _traverse_task_score(self) -> tuple[float, float, float]:
        """traverse 任务分：成功率×1000 + 平均最终进度。"""
        obs: Any = self.eval_env.reset()
        successes = 0
        final_xs: list[float] = []
        for _ in range(self.n_eval_episodes):
            done = False
            infos: Any = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.eval_env.step(action)
                done = bool(np.any(dones))
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            final_xs.append(float(info0.get("progress", info0.get("x", 0.0))))
            if info0.get("goal"):
                successes += 1
            obs = self.eval_env.reset()
        success_rate = successes / self.n_eval_episodes
        mean_x = float(np.mean(final_xs))
        return success_rate * 1000.0 + mean_x, success_rate, mean_x


class CurriculumCallback(BaseCallback):
    """课程学习：达到 warmup 步数后开启域随机化（穿透 VecNormalize 到 SubprocVecEnv）。"""

    def __init__(self, warmup_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.warmup_steps = warmup_steps
        self._switched = False

    def _on_step(self) -> bool:
        if not self._switched and self.num_timesteps >= self.warmup_steps:
            venv: Any = self.model.get_env()
            while hasattr(venv, "venv"):
                venv = venv.venv
            if hasattr(venv, "env_method"):
                venv.env_method("set_domain_randomize", True)
            self._switched = True
            if self.verbose:
                print(f"[curriculum] {self.num_timesteps} 步，开启域随机化")
        return True


class NormFreezeCallback(BaseCallback):
    """前 freeze_steps 步冻结 VecNormalize obs_rms，之后解冻更新。"""

    def __init__(self, freeze_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.freeze_steps = max(0, int(freeze_steps))
        self._unfrozen = False

    def _on_step(self) -> bool:
        vec = self.model.get_vec_normalize_env()
        if vec is None:
            return True
        train = norm_should_train(int(self.num_timesteps), self.freeze_steps)
        if not train and vec.training:
            vec.training = False
            if self.verbose:
                print(f"[norm-freeze] {self.num_timesteps} 步冻结 obs_rms")
        elif train and not self._unfrozen:
            vec.training = True
            self._unfrozen = True
            if self.verbose:
                print(f"[norm-freeze] {self.num_timesteps} 步解冻 obs_rms")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2w SB3 PPO 训练")
    parser.add_argument(
        "--task",
        choices=["balance", "full_chain", "full_chain_simple", *TRAVERSE_TASKS],
        default="balance",
    )
    parser.add_argument("--total-steps", type=int, default=8_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--init-from", type=str, default=None,
                        help="Phase 2 从 Phase 1 最优权重初始化")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="从已保存模型断点续训（自动只补剩余步数）")
    parser.add_argument("--resume-norm", type=str, default=None,
                        help="续训时加载的 VecNormalize 统计文件")
    parser.add_argument("--init-norm", type=str, default=None,
                        help="BC 预热：用外部 norm pkl（{mean,var,count}）初始化 obs 统计")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO 学习率（BC 微调常减半）")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="PPO 熵系数（探索增强用，默认 0.0）")
    parser.add_argument("--fresh", action="store_true",
                        help="从零训练：忽略 resume/init，并清理完成标记")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="评估间隔（按总环境步数）")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="快速冒烟：2k 步、2 env，验证管线")
    parser.add_argument("--curriculum-steps", type=int, default=0,
                        help="课程学习：前 N 步关闭域随机化，之后开启（balance 用）")
    parser.add_argument("--terrain", choices=["hfield", "boxes"], default="hfield",
                        help="traverse 斜坡任务地形：hfield=运行时高度场（默认），boxes=旧盒状斜坡")
    parser.add_argument("--heading-penalty", type=float, default=TRAVERSE_HEADING_PENALTY,
                        help="traverse 航向误差惩罚系数（curve 调参实验用，默认 1.5）")
    parser.add_argument("--corridor-width", type=float, default=0.40,
                        help="curve 走廊半宽（课程学习用，默认 0.40）")
    parser.add_argument("--curve-amplitude", type=float, default=0.35,
                        help="curve 弯道振幅（课程学习用，0=直道，默认 0.35）")
    parser.add_argument("--reward-version", choices=["v4", "simple", "v5"], default="v4",
                        help="奖励版本：v4=当前默认，simple=课程简洁奖励")
    parser.add_argument("--fall-penalty", type=float, default=-5.0,
                        help="v5 摔倒/早停惩罚（负值，默认 -5）")
    parser.add_argument("--goal-bonus", type=float, default=100.0,
                        help="v5 到达奖励（默认 100）")
    parser.add_argument("--best-window-size", type=int, default=3,
                        help="稳定窗口选 best：连续 N 个评估点")
    parser.add_argument("--best-window-std-ratio", type=float, default=0.05,
                        help="窗口奖励标准差阈值（相对均值比例，下限 0.1）")
    parser.add_argument("--best-metric", choices=["reward", "distance"], default="reward",
                        help="best 选择指标：reward=奖励/任务分，distance=平均前进距离")
    parser.add_argument("--lr-warmup-steps", type=int, default=0,
                        help="学习率预热步数（前 N 步用 warmup_start，之后 warmup_end）")
    parser.add_argument("--lr-warmup-start", type=float, default=5e-5,
                        help="学习率预热起始值")
    parser.add_argument("--lr-warmup-end", type=float, default=1.5e-4,
                        help="学习率预热结束值")
    parser.add_argument("--norm-freeze-steps", type=int, default=0,
                        help="前 N 步冻结 VecNormalize obs_rms，之后解冻")
    parser.add_argument("--exploration-boost", action="store_true",
                        help="启用不动最优检测 + 探索增强（ent_coef/学习率/动作噪声）")
    parser.add_argument("--boost-window", type=int, default=5,
                        help="局部最优检测窗口（连续 N 个 eval 点）")
    parser.add_argument("--boost-reward-threshold", type=float, default=-5.0,
                        help="局部最优检测的奖励上限阈值")
    parser.add_argument("--boost-ent-coef", type=float, default=0.01,
                        help="探索增强时的 PPO 熵系数")
    parser.add_argument("--boost-lr", type=float, default=5e-4,
                        help="探索增强时的学习率")
    parser.add_argument("--boost-noise-std", type=float, default=0.1,
                        help="探索增强时的动作高斯噪声标准差")
    parser.add_argument("--boost-cycles", type=int, default=10,
                        help="探索增强持续的 eval 周期数")
    args = parser.parse_args()

    n_envs = 2 if args.smoke else args.envs
    total_steps = 2_000 if args.smoke else args.total_steps
    eval_freq = 1_000 if args.smoke else args.eval_freq
    lr = make_lr_schedule(args, total_steps)
    remaining = total_steps
    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir)
    elif args.smoke:
        run_dir = pathlib.Path(f"rl/runs/_smoke/{args.task}/seed{args.seed:02d}")
    else:
        run_dir = pathlib.Path(f"rl/runs/{args.task}/seed{args.seed:02d}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_config.json").write_text(
        json.dumps(train_config_payload(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.fresh:
        (run_dir / ".completed").unlink(missing_ok=True)
    tb_dir = run_dir / "tensorboard"

    use_curriculum = args.curriculum_steps > 0 and args.task == "balance"
    raw_train_env = build_vec_env(
        args.task, n_envs, args.seed, domain_randomize=not use_curriculum,
        terrain=args.terrain, heading_penalty=args.heading_penalty,
        corridor_width=args.corridor_width, curve_amplitude=args.curve_amplitude,
        reward_version=args.reward_version,
        fall_penalty=args.fall_penalty,
        goal_bonus=args.goal_bonus,
    )
    boost_state = (
        {"active": False, "std": 0.0}
        if args.exploration_boost and not args.smoke
        else None
    )
    if boost_state is not None:
        raw_train_env = ActionNoiseVecEnv(raw_train_env, boost_state)
    if args.resume_norm and not args.fresh:
        train_env = VecNormalize.load(str(args.resume_norm), raw_train_env)
        train_env.training = True
        train_env.norm_reward = False
    else:
        train_env = VecNormalize(
            raw_train_env, norm_obs=True, norm_reward=False, clip_obs=10.0
        )
    if args.init_norm and not args.fresh:
        import pickle
        with open(args.init_norm, "rb") as f:
            nd = pickle.load(f)
        obs_rms: Any = train_env.obs_rms
        obs_rms.mean = np.asarray(nd["mean"], dtype=np.float32)
        obs_rms.var = np.asarray(nd["var"], dtype=np.float32)
        obs_rms.count = float(nd.get("count", 10000))
        print(f"已用 {args.init_norm} 初始化 obs_rms "
              f"(mean∈[{obs_rms.mean.min():.3f},{obs_rms.mean.max():.3f}])")

    eval_env = build_vec_env(
        args.task, 1, args.seed + 10_000, domain_randomize=False,
        vec_env_cls=DummyVecEnv, terrain=args.terrain,
        heading_penalty=args.heading_penalty,
        corridor_width=args.corridor_width, curve_amplitude=args.curve_amplitude,
        reward_version=args.reward_version,
        fall_penalty=args.fall_penalty,
        goal_bonus=args.goal_bonus,
    )
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True,
                            norm_reward=False, clip_obs=10.0)

    callback = EvalAndSaveCallback(
        eval_env=eval_env,
        eval_freq_timesteps=eval_freq,
        n_eval_episodes=args.eval_episodes,
        save_dir=run_dir,
        n_envs=n_envs,
        task=args.task,
        window_size=args.best_window_size,
        window_std_ratio=args.best_window_std_ratio,
        best_metric=args.best_metric,
        boost_state=boost_state,
        boost_window=args.boost_window,
        boost_reward_threshold=args.boost_reward_threshold,
        boost_ent_coef=args.boost_ent_coef,
        boost_lr=args.boost_lr,
        boost_noise_std=args.boost_noise_std,
        boost_cycles=args.boost_cycles,
        verbose=1,
    )

    if args.fresh:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
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
            tensorboard_log=str(tb_dir),
        )
        print(f"从零训练 task={args.task} seed={args.seed}")
    elif args.resume_from:
        if args.init_from:
            raise SystemExit("--resume-from 与 --init-from 不能同时使用")
        model = PPO.load(args.resume_from, env=train_env, device="cpu",
                         learning_rate=lr)
        model.tensorboard_log = str(tb_dir)
        steps_done = int(model.num_timesteps)
        remaining = max(0, total_steps - steps_done)
        print(f"断点已训练 {steps_done} 步，继续 {remaining} 步")
        if remaining == 0:
            (run_dir / ".completed").touch()
            model.save(str(run_dir / "final_model.zip"))
            train_env.save(str(run_dir / "final_vec_normalize.pkl"))
            print(f"已达标，无需继续。输出目录: {run_dir}")
            return
    elif args.init_from:
        model = PPO.load(args.init_from, env=train_env, device="cpu",
                         learning_rate=lr)
        model.tensorboard_log = str(tb_dir)
        print(f"已从 {args.init_from} 初始化，继续训练 {args.task}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=lr,
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
            tensorboard_log=str(tb_dir),
        )

    print(f"开始训练 task={args.task} seed={args.seed} steps={total_steps} envs={n_envs}")
    callbacks: list[BaseCallback] = [callback]
    if args.norm_freeze_steps > 0:
        callbacks.append(NormFreezeCallback(args.norm_freeze_steps, verbose=1))
    if use_curriculum:
        callbacks.append(CurriculumCallback(args.curriculum_steps, verbose=1))
    model.learn(
        total_timesteps=remaining if (args.resume_from and not args.fresh) else total_steps,
        callback=CallbackList(callbacks),
        reset_num_timesteps=args.fresh or not (args.init_from or args.resume_from),
        tb_log_name=f"{args.task}_seed{args.seed}",
    )
    touch_sync_trigger()

    (run_dir / ".completed").touch()
    model.save(str(run_dir / "final_model.zip"))
    train_env.save(str(run_dir / "final_vec_normalize.pkl"))
    print(f"完成。输出目录: {run_dir}")


if __name__ == "__main__":
    main()

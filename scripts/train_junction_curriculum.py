"""Junction 岔路口三阶段课程学习：公共段→岔路口、岔路口→分支终点、完整任务。

阶段1 用 BC 预热模型初始化，阶段2/3 用上一阶段 best 初始化。
System 2 规则决策（A→左 / B→右）由 high_level_env_wrapper 在 step 内完成。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
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
from rl.high_level_env_wrapper import (  # noqa: E402
    HighLevelEnvWrapper,
    compute_junction_obs,
)
from rl.experiment_io import (  # noqa: E402
    PROTOCOL_VERSION, acceptance_seeds, fingerprint, start_run, write_json_new,
)
from rl.normalization import normalize_once  # noqa: E402
from rl.train import (  # noqa: E402
    NormFreezeCallback,
    make_lr_schedule,
    select_best_window,
)

STAGES = [
    {"name": "stage1", "scope": 0, "start_junction": False, "steps": 300_000,
     "threshold": 0.8},
    {"name": "stage2", "scope": 1, "start_junction": True, "steps": 500_000,
     "threshold": 0.8},
    {"name": "stage3", "scope": 2, "start_junction": False, "steps": 1_000_000,
     "threshold": 0.6},
]
RUN_ROOT = PROJECT_ROOT / "rl" / "runs" / "traverse_curve_junction" / "seed00"
REPORT_DIR = PROJECT_ROOT / "reports" / "traverse_curve_junction" / "seed00"
EVAL_FREQ = 50_000
EVAL_EPISODES = 5
LIVE_WRITE_INTERVAL = 5_000


def stage_done(recent: list[dict], threshold: float) -> bool:
    """课程阶段切换判定：连续 5 个 eval 点成功率达标。"""
    return len(recent) >= 5 and all(
        r["success_rate"] >= threshold for r in recent[-5:]
    )


class LiveStatusCallback(BaseCallback):
    """周期性把实时训练进度写入 stage_dir/live_status.json，供面板读取。"""

    def __init__(
        self,
        stage_dir: pathlib.Path,
        stage_name: str,
        stage_target: int,
        stage_start: int,
        total_target: int,
        interval: int = LIVE_WRITE_INTERVAL,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.path = stage_dir / "live_status.json"
        self.stage = stage_name
        self.stage_target = int(stage_target)
        self.stage_start = int(stage_start)
        self.total_target = int(total_target)
        self.interval = int(interval)
        self._last = -1

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.interval:
            return True
        self._last = self.num_timesteps
        data = {
            "stage": self.stage,
            "timesteps": int(self.num_timesteps),
            "stage_timesteps": int(
                min(self.stage_target, self.num_timesteps - self.stage_start)
            ),
            "stage_target": self.stage_target,
            "total_target": self.total_target,
            "updated_at": time.time(),
        }
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf:
            vals = [r for r in buf if r is not None]
            if vals:
                data["mean_reward"] = float(np.mean([r["r"] for r in vals]))
                data["mean_ep_len"] = float(np.mean([r["l"] for r in vals]))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)
        return True


class JunctionWrapper(HighLevelEnvWrapper):
    """在 reset 时注入 scope / start_at_junction，并同步目标与分支。"""

    def __init__(self, base: Go2wEnv, scope: int, start_junction: bool):
        super().__init__(
            base,
            append_junction=True,
            use_goal_condition=False,
            goal_min=5.5,
            goal_max=5.5,
        )
        self._scope = scope
        self._start_junction = start_junction

    def _reset_options(self, options: dict | None) -> dict:
        opts = dict(options or {})
        opts["scope"] = self._scope
        if self._start_junction:
            opts["start_at_junction"] = True
        if self._scope == 2 and "target_goal" not in opts:
            opts["target_goal"] = "A" if self.np_random.random() < 0.5 else "B"
        return opts

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        if self._start_junction:
            # 阶段2：分支已由底层随机选择，同步目标使奖励一致，并避免起点白拿岔路口奖励
            self.base_env._junction_reached_bonus_given = True
            branch = self.base_env.get_branch_selected()
            if branch is not None:
                self.base_env.target_goal = "A" if branch == "left" else "B"
            obs = np.concatenate(
                [
                    np.asarray(obs[:61], dtype=np.float32),
                    compute_junction_obs(self.base_env),
                ]
            ).astype(np.float32)
        return obs, info


def make_stage_env(scope: int, start_junction: bool) -> JunctionWrapper:
    base = Go2wEnv(
        task="traverse_curve",
        domain_randomize=False,
        junction=True,
        reward_version="junction",
        scope=scope,
    )
    return JunctionWrapper(base, scope=scope, start_junction=start_junction)


class JunctionEvalCallback(BaseCallback):
    """阶段评估：success 为 goal 比例，correct 为分支决策正确率。"""

    def __init__(
        self,
        save_dir: pathlib.Path,
        n_envs: int,
        scope: int,
        start_junction: bool,
        threshold: float = 0.6,
        window: int = 3,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_freq = max(1, EVAL_FREQ // n_envs)
        self.save_dir = save_dir
        self.scope = scope
        self.start_junction = start_junction
        self.threshold = float(threshold)
        self.window = max(1, window)
        self.best_window_mean = -np.inf
        self._metric_history: list[tuple[float, float]] = []
        self._recent: list[dict] = []
        self.csv_path = save_dir / "eval_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "mean_reward", "mean_ep_len", "mean_x",
                 "success_rate", "correct_rate"]
            )
        dummy = make_stage_env(scope, start_junction)
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
                    round(stats["correct_rate"], 4),
                ]
            )
        if self.verbose:
            print(
                f"[junction-eval] t={self.num_timesteps} "
                f"success={stats['success_rate']:.0%} x={stats['mean_x']:.2f} "
                f"ep_len={stats['mean_ep_len']:.0f} "
                f"correct={stats['correct_rate']:.0%}"
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
                print(f"[junction-best] x={window_mean:.3f} @ {window_ts:.0f}")
        self._recent.append(stats)
        self._recent = self._recent[-5:]
        if stage_done(self._recent, self.threshold):
            (self.save_dir / "stage_ok.txt").write_text("success", encoding="utf-8")
            return False
        return True

    def _run_eval(self) -> dict:
        rewards: list[float] = []
        lengths: list[int] = []
        xs: list[float] = []
        succ = 0
        correct = 0
        for ep in range(EVAL_EPISODES):
            env = make_stage_env(self.scope, self.start_junction)
            try:
                options = {}
                if self.scope == 2:
                    options["target_goal"] = "A" if ep % 2 == 0 else "B"
                obs, info = env.reset(seed=ep, options=options)
                ep_rew = 0.0
                steps = 0
                max_x = 0.0
                ep_success = False
                while True:
                    obs_n = self.eval_norm.normalize_obs(obs)
                    action, _ = self.model.predict(obs_n, deterministic=True)
                    obs, r, term, trunc, info = env.step(action)
                    ep_rew += float(r)
                    steps += 1
                    max_x = max(max_x, float(info.get("progress", info.get("x", 0.0))))
                    if info.get("goal"):
                        ep_success = True
                    if term or trunc:
                        break
                branch = env.base_env.get_branch_selected()
                tgt = env.base_env.get_target_goal()
                if branch is not None and (
                    (branch == "left") == (tgt == "A")
                ):
                    correct += 1
                succ += int(ep_success)
                rewards.append(ep_rew)
                lengths.append(steps)
                xs.append(max_x)
            finally:
                env.close()
        return {
            "mean_reward": float(np.mean(rewards)),
            "mean_ep_len": float(np.mean(lengths)),
            "mean_x": float(np.mean(xs)),
            "success_rate": succ / EVAL_EPISODES,
            "correct_rate": correct / EVAL_EPISODES,
        }


def load_norm_dict(path: pathlib.Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def final_acceptance(
    model: Any, vec: Any, episodes: int = 40, seed_start: int = 1000
) -> dict:
    succ = 0
    correct = 0
    xs: list[float] = []
    lens: list[int] = []
    times: list[float] = []
    failed_terminations = 0
    records: list[dict] = []
    for ep, eval_seed in enumerate(acceptance_seeds(episodes, seed_start)):
        env = make_stage_env(scope=2, start_junction=False)
        try:
            target = "A" if ep < episodes // 2 else "B"
            obs, info = env.reset(seed=eval_seed, options={"target_goal": target})
            steps = 0
            ep_success = False
            max_x = 0.0
            while True:
                obs_n = vec.normalize_obs(obs)
                action, _ = model.predict(obs_n, deterministic=True)
                obs, _r, term, trunc, info = env.step(action)
                steps += 1
                max_x = max(max_x, float(info.get("progress", info.get("x", 0.0))))
                if info.get("goal"):
                    ep_success = True
                if term or trunc:
                    break
            branch = env.base_env.get_branch_selected()
            tgt = env.base_env.get_target_goal()
            if branch is not None and ((branch == "left") == (tgt == "A")):
                correct += 1
            succ += int(ep_success)
            xs.append(max_x)
            lens.append(steps)
            times.append(float(env.base_env.data.time))
            if term and not trunc and not info.get("goal"):
                failed_terminations += 1
            records.append({
                "seed": eval_seed, "success": ep_success,
                "max_progress": max_x, "steps": steps,
                "time_seconds": float(env.base_env.data.time),
                "terminated": bool(term), "truncated": bool(trunc),
                "target": target, "selected_branch": branch,
            })
        finally:
            env.close()
    return {
        "success_rate": succ / episodes,
        "correct_rate": correct / episodes,
        "mean_x": float(np.mean(xs)),
        "mean_ep_len": float(np.mean(lens)),
        "mean_time": float(np.mean(times)),
        "failed_terminations": failed_terminations,
        "episodes": episodes,
        "episode_records": records,
    }


def record_demo(model: PPO, vec: VecNormalize, prefix: pathlib.Path) -> None:
    env = make_stage_env(scope=2, start_junction=False)
    renderer = mujoco.Renderer(common.load_model(MODEL_SCENARIO_PATH), 480, 640)
    obs, _ = env.reset(seed=0, options={"target_goal": "A"})
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
    renderer.close()
    env.close()
    save_video(frames, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Junction 岔路口三阶段课程学习")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-root", type=str, default=None,
                        help="New output directory; existing directories are rejected")
    parser.add_argument("--record", action="store_true", help="Record a run-local demo")
    parser.add_argument("--bc-model", type=str,
                        default=str(PROJECT_ROOT / "data" / "demo_trajectories" / "junction_bc_pretrain.zip"))
    parser.add_argument("--bc-norm", type=str,
                        default=str(PROJECT_ROOT / "data" / "demo_trajectories" / "junction_bc_vecnorm.pkl"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n_envs = 2 if args.smoke else args.envs
    if n_envs < 1 or args.seed < 0:
        parser.error("envs must be positive and seed non-negative")
    eval_episodes = 4 if args.smoke else 40
    config = {
        "task": "traverse_curve_junction", "seed": args.seed,
        "envs": n_envs, "smoke": args.smoke, "record": args.record,
        "observation_dim": 68, "action_dim": 2,
        "domain_randomize": False, "stages": STAGES,
        "smoke_steps_per_stage": 10000 if args.smoke else None,
        "validation_seeds": list(range(EVAL_EPISODES)),
        "acceptance_seeds": acceptance_seeds(eval_episodes),
        "normalization_layers": 1, "ent_coef": 0.005,
        "learning_rate": {"start": 5e-5, "end": 1.5e-4,
                          "warmup_steps": 1000 if args.smoke else 100000},
        "distribution": "fixed_scene_repetition_not_generalization",
    }
    run_root = start_run(
        PROJECT_ROOT, "traverse_curve_junction", config,
        {"bc_model": args.bc_model, "bc_norm": args.bc_norm,},
        requested=args.run_root,
    )
    report_dir = run_root / ("smoke_evaluation" if args.smoke else "evaluation")
    prev_best = None
    prev_norm = None
    global_ts = 0
    total_target = sum(
        (10_000 if args.smoke else int(s["steps"])) for s in STAGES
    )

    for stage in STAGES:
        stage_dir = run_root / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        steps: int = 10_000 if args.smoke else int(stage["steps"])
        stage_start_ts = global_ts
        lr = make_lr_schedule(
            type("A", (), {
                "learning_rate": 1.5e-4,
                "lr_warmup_steps": 100_000 if not args.smoke else 1_000,
                "lr_warmup_start": 5e-5,
                "lr_warmup_end": 1.5e-4,
            })(),
            steps,
        )
        raw_env = make_vec_env(
            lambda: make_stage_env(stage["scope"], stage["start_junction"]),
            n_envs=n_envs,
            seed=args.seed,
            vec_env_cls=DummyVecEnv,
        )
        restore_norm = prev_norm if stage["name"] != "stage1" else None
        train_env = normalize_once(raw_env, checkpoint=restore_norm)
        if stage["name"] == "stage1":
            nd = load_norm_dict(pathlib.Path(args.bc_norm))
            obs_rms: Any = train_env.obs_rms
            obs_rms.mean = np.asarray(nd["mean"], dtype=np.float32)
            obs_rms.var = np.asarray(nd["var"], dtype=np.float32)
            obs_rms.count = float(nd.get("count", 1e4))
            model = PPO.load(args.bc_model, env=train_env, device="cpu",
                             learning_rate=lr, seed=args.seed)
        else:
            assert prev_best is not None and prev_norm is not None
            model = PPO.load(str(prev_best), env=train_env, device="cpu",
                             learning_rate=lr, seed=args.seed)
        model.ent_coef = 0.005
        model.tensorboard_log = str(stage_dir / "tensorboard")
        cb = JunctionEvalCallback(
            save_dir=stage_dir,
            n_envs=n_envs,
            scope=stage["scope"],
            start_junction=stage["start_junction"],
            threshold=stage["threshold"],
            verbose=1,
        )
        callbacks: list[BaseCallback] = [cb]
        callbacks.append(
            LiveStatusCallback(
                stage_dir=stage_dir,
                stage_name=stage["name"],
                stage_target=steps,
                stage_start=stage_start_ts,
                total_target=total_target,
                interval=1_000 if args.smoke else LIVE_WRITE_INTERVAL,
            )
        )
        if not args.smoke:
            callbacks.append(NormFreezeCallback(500_000, verbose=1))
        print(f"\n===== {stage['name']} scope={stage['scope']} steps={steps} =====")
        model.learn(
            total_timesteps=steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
            tb_log_name=f"junction_{stage['name']}",
        )
        model.save(str(stage_dir / "final_model.zip"))
        train_env.save(str(stage_dir / "final_vec_normalize.pkl"))
        (stage_dir / ".completed").touch()
        global_ts = int(model.num_timesteps)
        best = stage_dir / "best_model.zip"
        prev_best = best if best.exists() else stage_dir / "final_model.zip"
        prev_norm = (
            stage_dir / "best_vec_normalize.pkl"
            if best.exists()
            else stage_dir / "final_vec_normalize.pkl"
        )
        cb.eval_norm.close()
        train_env.close()

    # 最终验收
    assert prev_best is not None and prev_norm is not None
    report_dir.mkdir(parents=True, exist_ok=False)
    dummy = make_stage_env(scope=2, start_junction=False)
    vec = normalize_once(DummyVecEnv([lambda: dummy]), checkpoint=prev_norm, training=False)
    model = PPO.load(str(prev_best), device="cpu")
    acc = final_acceptance(model, vec, episodes=eval_episodes)
    write_json_new(report_dir / "episodes.json", acc["episode_records"])
    print("最终验收:", acc)
    (report_dir / "metrics.csv").write_text(
        "label,task,success_rate,correct_rate,mean_x,mean_ep_len,mean_time,failed_terminations\n"
        f"RL junction,traverse_curve_junction,"
        f"{acc['success_rate']},{acc['correct_rate']},{acc['mean_x']},"
        f"{acc['mean_ep_len']},{acc['mean_time']},{acc['failed_terminations']}\n",
        encoding="utf-8",
    )
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": "traverse_curve_junction",
                "seed": args.seed,
                "schema_version": PROTOCOL_VERSION,
                "episodes": eval_episodes,
                "acceptance_seeds": config["acceptance_seeds"],
                "checkpoint": fingerprint(prev_best),
                "normalization": fingerprint(prev_norm),
                "verdict": "smoke_only" if args.smoke else (
                    "pass" if acc["success_rate"] >= 0.6 else "fail"
                ),
                "success_rate": acc["success_rate"],
                "distance": acc["mean_x"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(
        "\n".join(
            [
                "# 岔路口（A→左 / B→右）RL 训练报告（MoRA System 2 + System 1 验证）",
                "",
                f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"- 完整成功率（{eval_episodes} episodes，A/B 均分）：{acc['success_rate']:.0%}",
                f"- 分支决策正确率：{acc['correct_rate']:.0%}",
                f"- 平均距离：{acc['mean_x']:.2f}m",
                f"- 平均 ep_len：{acc['mean_ep_len']:.0f}，平均时间：{acc['mean_time']:.1f}s",
                f"- 非成功终止次数（不等于摔倒）：{acc['failed_terminations']}",
                "- 固定场景重复评估，不能作为泛化成功率；smoke 不构成性能验收。",
                "",
                f"各阶段曲线见 {run_root}/stage*/eval_log.csv",
            ]
        ),
        encoding="utf-8",
    )
    try:
        if args.record and not args.smoke:
            record_demo(model, vec, report_dir / "demo")
    finally:
        vec.close()
    print("报告:", report_dir / "report.md")


if __name__ == "__main__":
    main()

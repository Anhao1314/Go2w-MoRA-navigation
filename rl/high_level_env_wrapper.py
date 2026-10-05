"""高层环境包装器：把 6 维底层动作压缩为 2 维导航决策。

动作空间：Box([0.9, -0.5], [1.5, 0.5]) = [speed_scale, turn_adjust]。
观测：底层 61 维；目标条件 63 维；多段 67 维；岔路口 68 维。
岔路口仍使用规则分支与教师转向，turn_adjust 不生效；未改变历史动作职责。
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.go2w_env import (
    GOAL_ARC_MAX,
    GOAL_ARC_MIN,
    path_tangent_angle,
    x_for_arc,
)
from rl.low_level_controller import (
    LowLevelController,
    MultiSegmentLowLevelController,
    default_controller,
)


def compute_mora_obs(env: Any) -> np.ndarray:
    """多段任务 6 维 MoRA 高层观测（供包装器与 BC 脚本共用）。"""
    x = float(env.data.body("base_link").xpos[0])
    y = float(env.data.body("base_link").xpos[1])
    yaw = float(env._yaw_of())
    seg_idx = env._current_segment()
    gx, gy = env._segment_goal(seg_idx)
    dist = float(np.hypot(x - gx, y - gy))
    ang = float(np.arctan2(gy - y, gx - x))
    docking = float(1.0 - np.clip(dist / 0.8, 0.0, 1.0))
    passed = 1.0 if bool(getattr(env, "_passed_subgoal", False)) else 0.0
    return np.array(
        [float(seg_idx), dist, math.sin(ang), math.cos(ang), docking, passed],
        dtype=np.float32,
    )


def compute_junction_obs(env: Any) -> np.ndarray:
    """Junction 任务 7 维 MoRA 高层观测。"""
    target = env.get_target_goal()
    branch = env.get_branch_selected()
    t_onehot = [1.0, 0.0] if target == "A" else [0.0, 1.0]
    b_onehot = (
        [1.0, 0.0]
        if branch == "left"
        else [0.0, 1.0]
        if branch == "right"
        else [0.0, 0.0]
    )
    return np.array(
        [
            *t_onehot,
            *b_onehot,
            env.get_distance_to_junction(),
            env.get_distance_to_goal(),
            env.get_system2_decision_flag(),
        ],
        dtype=np.float32,
    )


def junction_teacher_bias(x: float, y: float, yaw: float, env: Any) -> float:
    """Junction 纯跟踪教师 bias（System 0）。"""
    branch = env.get_branch_selected()
    cfg = env.junction_config
    xt = (
        min(x + 0.7, cfg["goal_x"])
        if branch
        else min(x + 0.7, cfg["decision_x_end"])
    )
    yt = float(env._junction_center_y(xt, branch))
    desired = float(np.arctan2(yt - y, xt - x))
    ang_err = float(((desired - yaw + math.pi) % (2 * math.pi)) - math.pi)
    dev = float(env._junction_deviation(x, y, branch))
    return float(np.clip(-2.5 * ang_err + 0.5 * dev, -1.0, 1.0))


def relabel_episode(
    transitions: list[dict],
    n_sampled_goal: int = 4,
    use_goal_condition: bool = True,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Future-goal relabeling 辅助样本；不是标准 off-policy HER 实现。"""
    rng = rng or np.random.default_rng(0)
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    n = len(transitions)
    if n < 2:
        return out
    for i, tr in enumerate(transitions):
        future = list(range(i + 1, n))
        if not future:
            continue
        goals = [
            float(transitions[j]["achieved_arc"])
            for j in rng.choice(future, size=min(n_sampled_goal, len(future)), replace=False)
        ]
        base_obs = np.asarray(tr["base_obs"], dtype=np.float32)
        action = np.asarray(tr["action"], dtype=np.float32)
        achieved_i = float(tr["achieved_arc"])
        for g in goals:
            new_goal = float(np.clip(g, GOAL_ARC_MIN, GOAL_ARC_MAX))
            reached = float(achieved_i) >= new_goal - 0.05
            reward = 1.0 if reached else 0.0
            if use_goal_condition:
                remaining = max(0.0, new_goal - achieved_i)
                target_heading = float(
                    path_tangent_angle(
                        x_for_arc(new_goal), is_curve=True
                    )
                )
                obs = np.concatenate(
                    [base_obs, np.array([remaining, target_heading], dtype=np.float32)]
                ).astype(np.float32)
            else:
                obs = base_obs
            out.append((obs, action, reward))
    return out


class HighLevelEnvWrapper(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        base_env: Any,
        controller: LowLevelController | None = None,
        goal_min: float = 0.5,
        goal_max: float = 1.4,
        use_goal_condition: bool = True,
        append_mora: bool = False,
        append_junction: bool = False,
        her_buffer: list | None = None,
        n_sampled_goal: int = 4,
    ):
        super().__init__()
        self.base_env = base_env
        if controller is None and getattr(base_env, "multi_segment", False):
            self.controller = MultiSegmentLowLevelController(base_env.segments)
        else:
            self.controller = controller or default_controller()
        self.goal_min = float(goal_min)
        self.goal_max = float(goal_max)
        self.use_goal_condition = bool(use_goal_condition)
        self.append_mora = bool(append_mora)
        self.append_junction = bool(append_junction)
        self.her_buffer = her_buffer
        self.n_sampled_goal = int(n_sampled_goal)
        self._current_goal = float(base_env.get_goal_arc())
        self._transitions: list[dict] = []
        self._last_base_obs: np.ndarray | None = None

        self._goal_arc = float(base_env._goal_arc_len())
        base_low = np.asarray(base_env.observation_space.low, dtype=np.float32)
        base_high = np.asarray(base_env.observation_space.high, dtype=np.float32)
        if append_junction and getattr(base_env, "junction", False):
            jun_low = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            jun_high = np.array([1, 1, 1, 1, 10, 10, 1], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=np.concatenate([base_low, jun_low]),
                high=np.concatenate([base_high, jun_high]),
                dtype=np.float32,
            )
        elif append_mora and getattr(base_env, "multi_segment", False):
            mora_low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
            mora_high = np.array([1.0, 12.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=np.concatenate([base_low, mora_low]),
                high=np.concatenate([base_high, mora_high]),
                dtype=np.float32,
            )
        elif use_goal_condition:
            goal_low = np.array([0.0, -np.pi], dtype=np.float32)
            goal_high = np.array([self.goal_max, np.pi], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=np.concatenate([base_low, goal_low]),
                high=np.concatenate([base_high, goal_high]),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=base_low, high=base_high, dtype=np.float32
            )
        self.action_space = spaces.Box(
            low=np.array([0.9, -0.5], dtype=np.float32),
            high=np.array([1.5, 0.5], dtype=np.float32),
            dtype=np.float32,
        )
        self.metadata = getattr(base_env, "metadata", {})
        self.render_mode = getattr(base_env, "render_mode", None)
        self.task = base_env.task
        self.scenario = getattr(base_env, "scenario", None)

    def _compute_goal(self) -> np.ndarray:
        remaining = float(self.base_env.get_remaining_arc())
        target_heading = float(self.base_env.get_target_heading())
        return np.array([remaining, target_heading], dtype=np.float32)

    def _maybe_mask_goal(self, obs: np.ndarray) -> np.ndarray:
        """无目标条件消融：把底层 61 维观测里的 remaining（nav[8]）置零，
        避免目标信息经原始观测泄漏。"""
        arr = np.asarray(obs, dtype=np.float32).copy()
        if not self.use_goal_condition and arr.shape[0] > 60:
            arr[60] = 0.0
        return arr

    def _reset_options(self, options: dict | None) -> dict:
        """Subclass sampling hook, called only after this wrapper is seeded."""
        return dict(options or {})

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        base_opts = self._reset_options(options)
        if "goal_arc" in base_opts:
            goal = float(base_opts["goal_arc"])
        else:
            goal = float(self.np_random.uniform(self.goal_min, self.goal_max))
            base_opts["goal_arc"] = goal
        obs, info = self.base_env.reset(seed=seed, options=base_opts)
        self._current_goal = goal
        self._transitions = []
        obs = self._maybe_mask_goal(obs)
        self._last_base_obs = np.asarray(obs, dtype=np.float32)
        if self.append_junction and getattr(self.base_env, "junction", False):
            jun = compute_junction_obs(self.base_env)
            return np.concatenate([obs, jun]).astype(np.float32), info
        if self.append_mora and getattr(self.base_env, "multi_segment", False):
            mora = compute_mora_obs(self.base_env)
            return np.concatenate([obs, mora]).astype(np.float32), info
        if self.use_goal_condition:
            goal_vec = self._compute_goal()
            return np.concatenate([obs, goal_vec]).astype(np.float32), info
        return np.asarray(obs, dtype=np.float32), info

    def step(
        self, action: np.ndarray
    ) -> tuple[Any, float, bool, bool, dict]:
        speed_scale = float(action[0])
        turn_adjust = float(action[1])
        x, y, yaw = self.controller.get_state_from_env(self.base_env)
        if self.append_junction and getattr(self.base_env, "junction", False):
            cfg = self.base_env.junction_config
            if (
                self.base_env.get_branch_selected() is None
                and cfg["decision_x_start"] <= x <= cfg["decision_x_end"]
            ):
                self.base_env.set_branch_selected(
                    "left" if self.base_env.get_target_goal() == "A" else "right"
                )
            bias = junction_teacher_bias(x, y, yaw, self.base_env)
            fwd = float(0.12 * speed_scale)
            low_action = np.array(
                [fwd - bias, fwd + bias, fwd - bias, fwd + bias, 0.0, 0.0],
                dtype=np.float32,
            )
            obs, reward, term, trunc, info = self.base_env.step(low_action)
            self._last_base_obs = np.asarray(obs, dtype=np.float32)
            jun = compute_junction_obs(self.base_env)
            return (
                np.concatenate([obs, jun]).astype(np.float32),
                reward,
                term,
                trunc,
                info,
            )
        gx, gy = self.base_env._segment_goal()
        dist = float(math.hypot(x - gx, y - gy))
        vx, _ = self.base_env._body_vel()
        low_action = self.controller.compute_action(
            x,
            y,
            yaw,
            {
                "speed_scale": speed_scale,
                "turn_adjust": turn_adjust,
                "segment_idx": self.base_env._current_segment(),
                "dist_to_subgoal": dist,
                "speed": abs(vx),
            },
        )
        base_obs_before = (
            self._last_base_obs
            if self._last_base_obs is not None
            else np.zeros(self.base_env.observation_space.shape, dtype=np.float32)
        )
        obs, reward, term, trunc, info = self.base_env.step(low_action)
        obs = self._maybe_mask_goal(obs)
        self._last_base_obs = np.asarray(obs, dtype=np.float32)
        achieved = float(self.base_env._progress_metric(
            float(self.base_env.data.body("base_link").xpos[0])
        ))
        self._transitions.append(
            {
                "base_obs": base_obs_before,
                "action": np.asarray(action, dtype=np.float32),
                "achieved_arc": achieved,
                "reward": float(reward),
            }
        )
        if (term or trunc) and self.her_buffer is not None:
            relabeled = relabel_episode(
                self._transitions,
                n_sampled_goal=self.n_sampled_goal,
                use_goal_condition=self.use_goal_condition,
                rng=self.np_random,
            )
            self.her_buffer.extend(relabeled)
            self._transitions = []
        if self.append_junction and getattr(self.base_env, "junction", False):
            jun = compute_junction_obs(self.base_env)
            obs = np.concatenate([obs, jun]).astype(np.float32)
        elif self.append_mora and getattr(self.base_env, "multi_segment", False):
            mora = compute_mora_obs(self.base_env)
            obs = np.concatenate([obs, mora]).astype(np.float32)
        elif self.use_goal_condition:
            goal_vec = self._compute_goal()
            obs = np.concatenate([obs, goal_vec]).astype(np.float32)
        return obs, reward, term, trunc, info

    def render(self, *args: Any, **kwargs: Any):
        return self.base_env.render(*args, **kwargs)

    def close(self):
        return self.base_env.close()

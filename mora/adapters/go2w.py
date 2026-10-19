"""Go2W task adapter: the first MoRA-Train experiment bed.

The adapter deliberately reuses the existing training factories so that the
MoRA runtime sees exactly the same observation, goal, reward, and termination
semantics as the P0/Go2W experiments.  It is a binding layer, not a new
environment implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mora.types import Action, Goal, Observation

SUPPORTED_TASKS = ("multi-segment", "junction", "high-level-curve")


class Go2wTaskAdapter:
    """Expose a Go2W task through the :class:`mora.task.TaskAdapter` contract."""

    def __init__(
        self,
        task: str = "multi-segment",
        *,
        scope: int = 2,
        start_a: bool = False,
        start_junction: bool = False,
        s0: str = "scripted",
        pre_brake: bool = False,
    ) -> None:
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported Go2W task: {task}")
        if task != "multi-segment" and s0 != "scripted":
            raise ValueError("C++ S0 is currently wired only for multi-segment")
        self.task_name = task
        self._last_info: dict[str, Any] = {}
        self._success = False
        self._env: Any

        if task == "multi-segment":
            from scripts.train_multi_segment_curriculum import make_stage_env

            self._env = make_stage_env(
                scope=scope,
                start_a=start_a,
                s0=s0,
                pre_brake=pre_brake,
            )
        elif task == "junction":
            from scripts.train_junction_curriculum import make_stage_env

            self._env = make_stage_env(
                scope=scope, start_junction=start_junction
            )
        else:
            from scripts.train_high_level_curve import make_high_level_env

            self._env = make_high_level_env()

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def env(self) -> Any:
        """Underlying Gymnasium wrapper, for training/evaluation integration."""

        return self._env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        observation, info = self._env.reset(seed=seed, options=options)
        self._last_info = dict(info)
        self._success = False
        return np.asarray(observation), dict(info)

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._env.step(
            np.asarray(action, dtype=np.float32)
        )
        self._last_info = dict(info)
        if bool(info.get("goal", False)):
            self._success = True
        return (
            np.asarray(observation),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )

    def goal(self) -> Goal:
        base = self._env.base_env
        if self.task_name == "multi-segment":
            index = int(base._current_segment())
            gx, gy = base._segment_goal(index)
            total = len(base.segments)
            return Goal(
                kind="subgoal",
                label=f"segment-{index}",
                index=index,
                position=(float(gx), float(gy)),
                metadata={"final": index >= total - 1, "total": total},
            )
        if self.task_name == "junction":
            target = str(base.get_target_goal())
            return Goal(
                kind="junction_target",
                label=target,
                metadata={"branch": base.get_branch_selected()},
            )
        arc = float(base.get_goal_arc())
        return Goal(
            kind="arc",
            label=f"arc:{arc:.3f}",
            metadata={"arc": arc},
        )

    def progress(self) -> float:
        if not self._last_info:
            return 0.0
        return float(self._last_info.get("progress", 0.0))

    def success(self) -> bool:
        if self.task_name == "junction":
            base = self._env.base_env
            target = str(base.get_target_goal())
            expected = "left" if target == "A" else "right"
            return bool(self._last_info.get("goal", False)) and (
                base.get_branch_selected() == expected
            )
        return bool(self._last_info.get("goal", False)) or self._success

    def metadata(self) -> dict[str, Any]:
        base = self._env.base_env
        return {
            "task": self.task_name,
            "wrapper": type(self._env).__name__,
            "multi_segment": bool(getattr(base, "multi_segment", False)),
            "junction": bool(getattr(base, "junction", False)),
            "scope": int(getattr(base, "scope", -1)),
            "observation_shape": list(self.observation_space.shape),
            "action_shape": list(self.action_space.shape),
        }

    def replan(self) -> Goal:
        """Current adapters have no alternate subgoal to choose yet."""
        return self.goal()

    def close(self) -> None:
        self._env.close()

"""Tests for the MoRA-Train agent runtime with a fake task."""

from __future__ import annotations

import unittest
from typing import Any

import gymnasium as gym
import numpy as np

from mora.policy import ZeroPolicy
from mora.progress import HeuristicProgressEstimator
from mora.runtime import AgentRuntime
from mora.types import Action, Goal, Observation


class FakeTask:
    """Deterministic task used to test the runtime loop."""

    def __init__(self, success_after: int = 3, progress_value: float = 1.0) -> None:
        self.success_after = success_after
        self.progress_value = progress_value
        self._step = 0
        self._last_info: dict[str, Any] = {}
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        self._step = 0
        self._last_info = {"progress": 0.0, "goal": False}
        return np.zeros(4, dtype=np.float32), dict(self._last_info)

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._step += 1
        done = self._step >= self.success_after
        self._last_info = {
            "progress": self.progress_value * self._step,
            "goal": done,
            "step": self._step,
        }
        return (
            np.zeros(4, dtype=np.float32),
            1.0,
            done,
            False,
            dict(self._last_info),
        )

    def goal(self) -> Goal:
        return Goal(kind="test", label=f"step-{self._step}")

    def progress(self) -> float:
        return float(self._last_info.get("progress", 0.0))

    def success(self) -> bool:
        return bool(self._last_info.get("goal", False))

    def metadata(self) -> dict[str, Any]:
        return {"task": "fake"}

    def replan(self) -> Goal:
        return self.goal()

    def close(self) -> None:
        return None


class RuntimeTests(unittest.TestCase):
    def test_episode_terminates_and_records_memory(self) -> None:
        task = FakeTask(success_after=3)
        runtime = AgentRuntime(task, ZeroPolicy(action_dim=2), max_steps=10)
        result = runtime.run_episode(seed=0)
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.success)
        self.assertTrue(result.terminated)
        self.assertFalse(result.truncated)
        self.assertEqual(result.total_reward, 3.0)
        self.assertEqual(len(runtime.memory.episodic), 1)
        self.assertEqual(runtime.memory.task.get("planner"), "RulePlanner")
        runtime.close()

    def test_explicit_action_reaches_task(self) -> None:
        task = FakeTask(success_after=2)
        runtime = AgentRuntime(task, ZeroPolicy(action_dim=2))
        runtime.reset(seed=0)
        result = runtime.step(np.array([0.25, -0.5], dtype=np.float32))
        self.assertFalse(result.terminated)
        self.assertEqual(result.goal.kind, "test")
        self.assertEqual(result.progress.value, 1.0)

    def test_max_steps_records_truncated_episode(self) -> None:
        task = FakeTask(success_after=100)
        runtime = AgentRuntime(task, ZeroPolicy(action_dim=2), max_steps=5)
        result = runtime.run_episode(seed=0, max_steps=5)
        self.assertEqual(result.steps, 5)
        self.assertFalse(result.success)
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertEqual(len(runtime.memory.episodic), 1)

    def test_stall_requests_replan(self) -> None:
        task = FakeTask(success_after=100, progress_value=0.0)
        runtime = AgentRuntime(
            task,
            ZeroPolicy(action_dim=2),
            progress=HeuristicProgressEstimator(
                stall_epsilon=0.0,
                stall_patience=1,
                replan_patience=2,
            ),
            max_steps=5,
        )
        result = runtime.run_episode(seed=0, max_steps=5)
        self.assertGreaterEqual(result.interventions, 1)
        self.assertTrue(runtime.memory.task.get("replan_requested"))

    def test_step_before_reset_fails(self) -> None:
        runtime = AgentRuntime(
            FakeTask(), ZeroPolicy(action_dim=2)
        )
        with self.assertRaises(RuntimeError):
            runtime.step()


if __name__ == "__main__":
    unittest.main()

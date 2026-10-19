"""Agent runtime: the S2 → S1 → memory/progress loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mora.memory import MultiGranularityMemory
from mora.planner import Planner, RulePlanner
from mora.policy import Policy
from mora.progress import HeuristicProgressEstimator, ProgressEstimator
from mora.task import TaskAdapter
from mora.types import (
    EpisodeSummary,
    Goal,
    ProgressEstimate,
    StepResult,
    Transition,
)


@dataclass(frozen=True)
class EpisodeResult:
    """Summary returned by :meth:`AgentRuntime.run_episode`."""

    goal: Goal
    steps: int
    total_reward: float
    success: bool
    max_progress: float
    terminated: bool
    truncated: bool
    interventions: int


class AgentRuntime:
    """Compose a task adapter, S2 planner, S1 policy, memory, and progress."""

    def __init__(
        self,
        task: TaskAdapter,
        policy: Policy,
        *,
        memory: MultiGranularityMemory | None = None,
        progress: ProgressEstimator | None = None,
        planner: Planner | None = None,
        max_steps: int = 4000,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.task = task
        self.policy = policy
        self.memory = memory or MultiGranularityMemory()
        self.progress = progress or HeuristicProgressEstimator()
        self.planner = planner or RulePlanner()
        self.max_steps = int(max_steps)
        self._last_observation: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._last_goal: Goal | None = None
        self._last_result: StepResult | None = None
        self._steps = 0
        self._total_reward = 0.0
        self._max_progress = 0.0
        self._interventions = 0
        self._replan_active = False
        self._finished = False
        self._episode_recorded = False

    @property
    def observation_space(self) -> Any:
        return self.task.observation_space

    @property
    def action_space(self) -> Any:
        return self.task.action_space

    @property
    def last_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    @property
    def last_result(self) -> StepResult | None:
        return self._last_result

    @property
    def steps(self) -> int:
        return self._steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.task.reset(seed=seed, options=options)
        self.policy.reset()
        self.memory.reset_episode()
        self.progress.reset()
        self.planner.reset(self.task, self.memory)
        self._last_observation = np.asarray(observation)
        self._last_info = dict(info)
        self._last_goal = self.task.goal()
        self._last_result = None
        self._steps = 0
        self._total_reward = 0.0
        self._max_progress = 0.0
        self._interventions = 0
        self._replan_active = False
        self._finished = False
        self._episode_recorded = False
        return self._last_observation, dict(info)

    def step(self, action: np.ndarray | None = None) -> StepResult:
        if self._last_observation is None:
            raise RuntimeError("reset() must be called before step()")
        if self._finished:
            raise RuntimeError("episode already finished; call reset()")

        goal = self.planner.decide(
            self.task, self.memory, self.progress.estimate
        )
        if action is None:
            action = self.policy.act(
                self._last_observation, goal, self.memory.snapshot()
            )
        action_array = np.asarray(action, dtype=np.float32)

        observation, reward, terminated, truncated, info = self.task.step(
            action_array
        )
        progress_value = info.get("progress")
        if progress_value is None:
            progress_value = self.task.progress()
        progress_estimate = self.progress.update(float(progress_value))
        transition = Transition(
            observation=np.asarray(observation),
            action=action_array,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            progress=float(progress_value),
        )
        self.memory.observe(transition)

        self._steps += 1
        self._total_reward += float(reward)
        self._max_progress = max(self._max_progress, float(progress_value))
        if progress_estimate.should_replan and not self._replan_active:
            self._interventions += 1
        self._replan_active = progress_estimate.should_replan

        result = StepResult(
            observation=np.asarray(observation),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
            goal=goal,
            progress=progress_estimate,
            intervention=progress_estimate.should_replan,
        )
        self._last_observation = result.observation
        self._last_info = dict(info)
        self._last_goal = goal
        self._last_result = result

        if result.terminated or result.truncated:
            self._finish_episode(
                terminated=result.terminated, truncated=result.truncated
            )
        return result

    def run_episode(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        max_steps: int | None = None,
    ) -> EpisodeResult:
        limit = self.max_steps if max_steps is None else int(max_steps)
        if limit < 1:
            raise ValueError("max_steps must be positive")
        self.reset(seed=seed, options=options)
        result: StepResult | None = None
        while self._steps < limit:
            result = self.step()
            if result.terminated or result.truncated:
                break
        if result is None:
            result = self._last_result
        exhausted = (
            result is not None
            and not result.terminated
            and not result.truncated
            and self._steps >= limit
        )
        terminated = bool(result.terminated) if result else False
        truncated = bool(result.truncated) if result else True
        if exhausted:
            truncated = True
        if not self._episode_recorded:
            self._finish_episode(terminated=terminated, truncated=truncated)
        assert self._last_goal is not None
        return EpisodeResult(
            goal=self._last_goal,
            steps=self._steps,
            total_reward=self._total_reward,
            success=self.task.success(),
            max_progress=self._max_progress,
            terminated=terminated,
            truncated=truncated,
            interventions=self._interventions,
        )

    def _finish_episode(self, *, terminated: bool, truncated: bool) -> None:
        if self._episode_recorded:
            return
        self._episode_recorded = True
        self._finished = True
        assert self._last_goal is not None
        self.memory.record_episode(
            EpisodeSummary(
                goal=self._last_goal,
                steps=self._steps,
                total_reward=self._total_reward,
                success=self.task.success(),
                max_progress=self._max_progress,
                terminated=bool(terminated),
                truncated=bool(truncated),
                interventions=self._interventions,
            )
        )

    def close(self) -> None:
        self.task.close()

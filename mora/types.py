"""Shared value objects for the MoRA-Train runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

Observation = np.ndarray
Action = np.ndarray


@dataclass(frozen=True)
class Goal:
    """A structured goal passed from S2 to S1.

    ``kind`` is intentionally open-ended: adapters may use ``subgoal``,
    ``junction_target``, ``arc``, or a task-specific label.  The runtime only
    requires a stable label and optional position/index for logging.
    """

    kind: str
    label: str
    index: int | None = None
    position: tuple[float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressEstimate:
    """Output of the progress-aware closed loop."""

    value: float
    delta: float
    trend: float
    stalled_steps: int
    should_replan: bool
    confidence: float = 1.0


@dataclass(frozen=True)
class Transition:
    """One high-level agent transition."""

    observation: Observation
    action: Action
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    progress: float | None = None


@dataclass(frozen=True)
class EpisodeSummary:
    """Episode-level memory record; not a training label."""

    goal: Goal
    steps: int
    total_reward: float
    success: bool
    max_progress: float
    terminated: bool
    truncated: bool
    interventions: int


@dataclass(frozen=True)
class MemorySnapshot:
    """Read-only view passed to the policy on each step."""

    recent: tuple[Transition, ...]
    episodes: tuple[EpisodeSummary, ...]
    task_state: Mapping[str, Any]


@dataclass(frozen=True)
class StepResult:
    """Runtime output for one environment step."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    goal: Goal
    progress: ProgressEstimate
    intervention: bool

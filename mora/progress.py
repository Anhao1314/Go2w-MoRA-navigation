"""Progress-aware closed loop.

The estimator converts a task-provided progress scalar into a small decision
signal: is progress increasing, has it stalled, and should S1 ask S2 to
replan?  The first implementation is deterministic and inspectable; a learned
progress head can replace it behind the same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mora.types import ProgressEstimate


@runtime_checkable
class ProgressEstimator(Protocol):
    """Interface consumed by :class:`mora.runtime.AgentRuntime`."""

    def reset(self) -> None:
        """Start a new episode."""
        ...

    def update(self, value: float | None) -> ProgressEstimate:
        """Consume the latest task progress scalar."""
        ...

    @property
    def estimate(self) -> ProgressEstimate:
        """Return the most recent estimate."""
        ...


class NullProgressEstimator:
    """Estimator for tasks without a meaningful progress scalar."""

    def __init__(self) -> None:
        self._estimate = ProgressEstimate(
            value=0.0,
            delta=0.0,
            trend=0.0,
            stalled_steps=0,
            should_replan=False,
            confidence=0.0,
        )

    def reset(self) -> None:
        self._estimate = ProgressEstimate(
            value=0.0,
            delta=0.0,
            trend=0.0,
            stalled_steps=0,
            should_replan=False,
            confidence=0.0,
        )

    def update(self, value: float | None) -> ProgressEstimate:
        return self._estimate

    @property
    def estimate(self) -> ProgressEstimate:
        return self._estimate


class HeuristicProgressEstimator:
    """EMA trend + stall counter + replanning signal.

    Parameters are intentionally conservative and configurable.  The estimator
    does not claim a calibrated failure probability; ``confidence`` only
    reflects whether a progress value was observed.
    """

    def __init__(
        self,
        *,
        stall_epsilon: float = 1e-3,
        stall_patience: int = 5,
        replan_patience: int = 3,
        trend_alpha: float = 0.3,
        value_scale: float = 1.0,
    ) -> None:
        if stall_epsilon < 0.0:
            raise ValueError("stall_epsilon must be non-negative")
        if stall_patience < 1 or replan_patience < 1:
            raise ValueError("patience values must be positive")
        if not 0.0 < trend_alpha <= 1.0:
            raise ValueError("trend_alpha must be in (0, 1]")
        if value_scale <= 0.0:
            raise ValueError("value_scale must be positive")
        self.stall_epsilon = float(stall_epsilon)
        self.stall_patience = int(stall_patience)
        self.replan_patience = int(replan_patience)
        self.trend_alpha = float(trend_alpha)
        self.value_scale = float(value_scale)
        self.reset()

    def reset(self) -> None:
        self._last_value: float | None = None
        self._trend = 0.0
        self._stalled_steps = 0
        self._estimate = ProgressEstimate(
            value=0.0,
            delta=0.0,
            trend=0.0,
            stalled_steps=0,
            should_replan=False,
            confidence=0.0,
        )

    def update(self, value: float | None) -> ProgressEstimate:
        if value is None:
            self._estimate = ProgressEstimate(
                value=self._estimate.value,
                delta=0.0,
                trend=self._trend,
                stalled_steps=self._stalled_steps,
                should_replan=False,
                confidence=0.0,
            )
            return self._estimate

        scaled = float(value) / self.value_scale
        if self._last_value is None:
            delta = 0.0
        else:
            delta = scaled - self._last_value
        self._trend = (
            self.trend_alpha * delta
            + (1.0 - self.trend_alpha) * self._trend
        )
        if delta <= self.stall_epsilon:
            self._stalled_steps += 1
        else:
            self._stalled_steps = 0
        self._last_value = scaled
        self._estimate = ProgressEstimate(
            value=scaled,
            delta=delta,
            trend=self._trend,
            stalled_steps=self._stalled_steps,
            should_replan=self._stalled_steps >= self.replan_patience,
            confidence=1.0,
        )
        return self._estimate

    @property
    def estimate(self) -> ProgressEstimate:
        return self._estimate

"""S1 policy interfaces and lightweight adapters."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from mora.types import Action, Goal, MemorySnapshot, Observation

PolicyFn = Callable[[Observation, Goal, MemorySnapshot], Action]


@runtime_checkable
class Policy(Protocol):
    """S1 execution policy."""

    def reset(self) -> None:
        """Reset recurrent/policy state at episode start."""
        ...

    def act(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySnapshot,
    ) -> Action:
        """Return a high-level action."""
        ...


class CallablePolicy:
    """Wrap a function as a policy (useful for scripted baselines and tests)."""

    def __init__(self, fn: PolicyFn) -> None:
        self._fn = fn

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySnapshot,
    ) -> Action:
        return np.asarray(self._fn(observation, goal, memory), dtype=np.float32)


class ZeroPolicy:
    """Deterministic zero action baseline."""

    def __init__(self, action_dim: int = 2) -> None:
        if action_dim < 1:
            raise ValueError("action_dim must be positive")
        self.action_dim = int(action_dim)

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySnapshot,
    ) -> Action:
        return np.zeros(self.action_dim, dtype=np.float32)


class SB3Policy:
    """Duck-typed Stable-Baselines3 policy adapter.

    The model is not imported here; any object with ``predict`` works.  The
    current Go2W SB3 policies consume the environment observation, which already
    contains the adapter's goal/progress features.  ``goal`` and ``memory`` are
    accepted so a future memory-conditioned policy can use them without a
    runtime API change.
    """

    def __init__(self, model: Any, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = bool(deterministic)

    def reset(self) -> None:
        return None

    def act(
        self,
        observation: Observation,
        goal: Goal,
        memory: MemorySnapshot,
    ) -> Action:
        action, _state = self.model.predict(
            np.asarray(observation), deterministic=self.deterministic
        )
        return np.asarray(action, dtype=np.float32)

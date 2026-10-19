"""Task adapter contract.

A task adapter is the only environment-specific part of MoRA-Train.  It hides
Gymnasium/MuJoCo details and exposes the information the agent runtime needs:
observations, goals, progress, success, and a task-level replan hook.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mora.types import Action, Goal, Observation


@runtime_checkable
class TaskAdapter(Protocol):
    """Minimal interface for an embodied task."""

    @property
    def observation_space(self) -> Any:
        """Gymnasium-compatible observation space."""
        ...

    @property
    def action_space(self) -> Any:
        """Gymnasium-compatible action space."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a new episode."""
        ...

    def step(
        self, action: Action
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Apply one high-level action."""
        ...

    def goal(self) -> Goal:
        """Return the current structured goal."""
        ...

    def progress(self) -> float:
        """Return the task's current progress scalar."""
        ...

    def success(self) -> bool:
        """Return whether the task has been completed."""
        ...

    def metadata(self) -> dict[str, Any]:
        """Return serializable task metadata for event logs."""
        ...

    def replan(self) -> Goal:
        """Return a goal after a progress stall; default adapters may no-op."""
        ...

    def close(self) -> None:
        """Release environment resources."""
        ...

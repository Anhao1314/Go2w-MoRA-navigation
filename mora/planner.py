"""S2 planner interfaces and the first rule-based implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mora.memory import MultiGranularityMemory
from mora.task import TaskAdapter
from mora.types import Goal, ProgressEstimate


@runtime_checkable
class Planner(Protocol):
    """S2-level decision interface."""

    def reset(self, task: TaskAdapter, memory: MultiGranularityMemory) -> None:
        """Start a new episode or task."""
        ...

    def decide(
        self,
        task: TaskAdapter,
        memory: MultiGranularityMemory,
        progress: ProgressEstimate,
    ) -> Goal:
        """Return the structured goal for the next S1 step."""
        ...


class RulePlanner:
    """Deterministic S2 baseline.

    The task adapter owns the task-specific rule (current segment, junction
    target, or goal arc).  On a progress stall the planner calls the adapter's
    ``replan`` hook; adapters that do not support replanning simply return the
    current goal.
    """

    def reset(self, task: TaskAdapter, memory: MultiGranularityMemory) -> None:
        memory.task.set("planner", type(self).__name__)

    def decide(
        self,
        task: TaskAdapter,
        memory: MultiGranularityMemory,
        progress: ProgressEstimate,
    ) -> Goal:
        if progress.should_replan:
            memory.task.set("replan_requested", True)
            return task.replan()
        return task.goal()

"""MoRA-Train: a hierarchical training runtime for embodied agents.

The package is intentionally small and dependency-light.  It defines the
system-level interfaces that sit above a concrete environment adapter:

- :mod:`mora.task` — the task adapter contract
- :mod:`mora.runtime` — S2 planner / S1 policy / memory / progress loop
- :mod:`mora.memory` — short-term, episodic, and task memory
- :mod:`mora.progress` — progress estimation and replanning signal
- :mod:`mora.observability` — v0.2-compatible training events

The Go2W adapter lives in :mod:`mora.adapters.go2w`.  It is the first
experiment bed, not the definition of the system.
"""

from mora.memory import EpisodicMemory, MultiGranularityMemory, ShortTermMemory, TaskMemory
from mora.progress import HeuristicProgressEstimator, NullProgressEstimator, ProgressEstimator
from mora.runtime import AgentRuntime, EpisodeResult
from mora.task import TaskAdapter
from mora.types import (
    EpisodeSummary,
    Goal,
    MemorySnapshot,
    ProgressEstimate,
    StepResult,
    Transition,
)

__all__ = [
    "AgentRuntime",
    "EpisodeResult",
    "EpisodeSummary",
    "EpisodicMemory",
    "Goal",
    "HeuristicProgressEstimator",
    "MemorySnapshot",
    "MultiGranularityMemory",
    "NullProgressEstimator",
    "ProgressEstimate",
    "ProgressEstimator",
    "ShortTermMemory",
    "StepResult",
    "TaskAdapter",
    "TaskMemory",
    "Transition",
]

"""Frozen evaluation seed set for the M1/M2 protocol.

Single source of truth for the episode seed set ``E = {0..19}`` frozen by
``docs/EXPERIMENT_DESIGN.md`` §3.1 *before* any M1 training run.  Every
condition (RL seeds T1–T3, baselines C0/C1, S4 Sim2Sim) must evaluate on
exactly this tuple paired seed-by-seed.

Changing this tuple after the first M1 training run is a protocol violation;
if a change is ever justified, record it in the protocol changelog and keep
the old tuple archived in the affected run directories.
"""

from __future__ import annotations

EVAL_SEEDS: tuple[int, ...] = tuple(range(20))
"""Frozen episode seeds E = {0..19}, 20 episodes per condition."""

NUM_EVAL_EPISODES: int = len(EVAL_SEEDS)

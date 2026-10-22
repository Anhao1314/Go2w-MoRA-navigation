#!/usr/bin/env python3
"""Launch Isaac Lab RSL-RL training for the ``Go2Loco-*`` tasks (M1/M2).

Isaac Lab's own ``train.py`` registers only the official ``Isaac-*`` tasks, so
this wrapper registers the tasks of :mod:`locomotion.isaac` first and then hands
control to that script unchanged.  Keeping the registration *before* the
launcher also documents the import-order contract of ``locomotion/isaac``:
importing it needs no Kit runtime (string-only Gymnasium registrations), while
the env configs it points at need ``pxr``, which exists only after
``AppLauncher`` boots Isaac Sim.

Run with the repository's ``.venv311`` interpreter from the repo root::

    .venv311/Scripts/python.exe scripts/train_locomotion.py \
        --task Go2Loco-S1-Flat-Blind-v0 --headless --num_envs 2048 \
        --seed 0 --max_iterations 1500

Isaac Lab is located via ``ISAACLAB_ROOT``; when unset the usual checkouts are
probed (sibling ``../IsaacLab`` and ``D:/work/IsaacLab``).
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Register Go2Loco-* before Isaac Sim starts (Kit-free by contract).
import locomotion.isaac  # noqa: E402,F401

TRAIN_RELPATH = pathlib.Path("scripts") / "reinforcement_learning" / "rsl_rl" / "train.py"


def find_isaaclab_root() -> pathlib.Path:
    """Resolve the Isaac Lab checkout holding the RSL-RL launcher."""
    candidates: list[pathlib.Path] = []
    env_root = os.environ.get("ISAACLAB_ROOT")
    if env_root:
        candidates.append(pathlib.Path(env_root))
    candidates += [PROJECT_ROOT.parent / "IsaacLab", pathlib.Path("D:/work/IsaacLab")]
    for candidate in candidates:
        if (candidate / TRAIN_RELPATH).is_file():
            return candidate
    raise SystemExit(
        "Isaac Lab launcher not found. Set ISAACLAB_ROOT to your Isaac Lab checkout "
        f"(expected {TRAIN_RELPATH} inside it). Tried: {[str(c) for c in candidates]}"
    )


def main() -> None:
    isaaclab_root = find_isaaclab_root()
    train_script = isaaclab_root / TRAIN_RELPATH
    # Isaac Lab's train.py imports its sibling ``cli_args`` module.
    sys.path.insert(0, str(train_script.parent))
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()

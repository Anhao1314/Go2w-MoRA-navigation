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
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_evidence import finalize, snapshot_start  # noqa: E402

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


def _arg_value(argv: list[str], name: str) -> str | None:
    """Return the value of ``--name value`` or ``--name=value`` from argv."""
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def main() -> None:
    isaaclab_root = find_isaaclab_root()
    train_script = isaaclab_root / TRAIN_RELPATH
    # Isaac Lab's train.py imports its sibling ``cli_args`` module.
    sys.path.insert(0, str(train_script.parent))

    argv = sys.argv[1:]
    task = _arg_value(argv, "--task") or "unknown-task"
    seed_text = _arg_value(argv, "--seed")
    seed = int(seed_text) if seed_text and seed_text.lstrip("-").isdigit() else None
    run_id = f"{task}-seed{seed if seed is not None else 'na'}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Protocol §7: the evidence record is created at start, never rewritten.
    pending = snapshot_start(run_id, task, seed, argv)
    print(f"[evidence] start record: {pending}", flush=True)

    # Isaac Sim terminates its process from C++ during shutdown, so the run
    # cannot be finalized in-process: launch the launcher as a child and write
    # the evidence bundle from this parent once it exits.
    env = dict(os.environ)
    python_path = [str(PROJECT_ROOT), str(train_script.parent)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    # Gymnasium registrations are per-process: the child must register the
    # tasks itself before calling the upstream launcher.
    child_launcher = pathlib.Path(__file__).with_name("_isaac_launcher.py")
    completed = subprocess.run(
        [sys.executable, str(child_launcher), str(train_script), *argv],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    try:
        config = finalize(pending, None)
        print(f"[evidence] finalized: {config}", flush=True)
    except SystemExit as exc:
        print(f"[evidence] finalize skipped: {exc}", flush=True)

    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

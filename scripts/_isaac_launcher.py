#!/usr/bin/env python3
"""Internal launcher: run Isaac Lab's ``train.py`` with ``Go2Loco-*`` registered.

Gymnasium registrations live in the process, not on disk, and Isaac Sim must
run in the process that registers nothing heavy -- so the registration has to
happen inside the very process that will call ``gym.make``.  ``train_locomotion.py``
therefore spawns this module, which registers first and then executes the
upstream launcher unchanged.

Not intended for direct use.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe before Kit starts: string-only Gymnasium registrations.
import locomotion.isaac  # noqa: E402,F401
import locomotion.sequence  # noqa: E402,F401


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: _isaac_launcher.py <train.py> [train.py args...]")
    train_script = pathlib.Path(sys.argv[1]).resolve()
    if not train_script.is_file():
        raise SystemExit(f"train script not found: {train_script}")
    # upstream train.py imports its sibling ``cli_args`` module
    sys.path.insert(0, str(train_script.parent))
    sys.argv = [str(train_script), *sys.argv[2:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()

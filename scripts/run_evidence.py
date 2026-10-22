#!/usr/bin/env python3
"""Per-run evidence bundles for the M1/M2 protocol (EXPERIMENT_DESIGN.md §7).

Two phases, matching the evidence rules in ``docs/EXPERIMENT_DATA.md``:

``start``
    Called by the launcher *before* training.  Writes
    ``logs/rsl_rl/_pending/<run-id>.json`` with the git SHA, dirty flag,
    dependency versions, command line, task and seed.  The record is created
    at start and never rewritten afterwards.

``finalize``
    Called after training.  Copies the pending record into the run directory
    as ``run_config.json`` and writes ``sha256_manifest.json`` over every
    file in the bundle.

The split exists because Isaac Lab owns the run directory (it creates it with
a timestamp): the start record must not depend on knowing that path yet.

Usage::

    python scripts/run_evidence.py start --run-id seed00-20261023 \
        --task Go2Loco-S1-Flat-Blind-v0 --seed 0 -- --num_envs 2048
    python scripts/run_evidence.py finalize --run-dir logs/rsl_rl/go2_loco_s1_flat/<stamp>

``finalize`` without ``--run-dir`` picks the most recent run directory that
contains ``params/agent.yaml`` and was modified after the pending record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PENDING_DIR = PROJECT_ROOT / "logs" / "rsl_rl" / "_pending"
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"

SCHEMA_VERSION = "locomotion-run-evidence-v1"


def find_git() -> str | None:
    import shutil

    found = shutil.which("git")
    if found:
        return found
    for candidate in (
        pathlib.Path(r"C:\Program Files\Git\cmd\git.exe"),
        pathlib.Path(r"C:\Program Files\Git\bin\git.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def git_state() -> dict[str, object]:
    """Record the commit and whether the working tree was dirty at start."""
    git = find_git()
    if git is None:
        return {"available": False}
    def run(*args: str) -> str:
        result = subprocess.run(
            [git, *args], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return result.stdout.strip()
    dirty = run("status", "--porcelain")
    return {
        "available": True,
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": len(dirty.splitlines()) if dirty else 0,
    }


def dependency_versions() -> dict[str, str | None]:
    """Versions that determine whether a run can be reproduced."""
    wanted = (
        "isaaclab",
        "isaaclab-tasks",
        "isaaclab-rl",
        "isaacsim",
        "rsl-rl-lib",
        "torch",
        "gymnasium",
        "numpy",
        "mujoco",
    )
    versions: dict[str, str | None] = {}
    for name in wanted:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version.split()[0]
    return versions


def snapshot_start(run_id: str, task: str, seed: int | None, argv: list[str]) -> pathlib.Path:
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "start",
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": task,
        "seed": seed,
        "argv": argv,
        "cwd": str(PROJECT_ROOT),
        "git": git_state(),
        "dependencies": dependency_versions(),
    }
    path = PENDING_DIR / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(run_dir: pathlib.Path) -> dict[str, object]:
    files: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative == "sha256_manifest.json":
            continue  # a manifest cannot contain itself
        files[relative] = sha256_of(path)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "finalize",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": run_dir.as_posix(),
        "file_count": len(files),
        "files": files,
    }


def find_pending(run_id: str | None) -> pathlib.Path:
    if run_id:
        candidate = PENDING_DIR / f"{run_id}.json"
        if not candidate.is_file():
            raise SystemExit(f"no pending record: {candidate}")
        return candidate
    candidates = sorted(PENDING_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"no pending records in {PENDING_DIR}")
    return candidates[-1]


def find_recent_run_dir(since_epoch: float) -> pathlib.Path:
    """Most recent run directory with an Isaac Lab agent config, modified since start."""
    candidates = [
        agent.parent.parent
        for agent in LOG_ROOT.rglob("params/agent.yaml")
        if agent.parent.parent.stat().st_mtime >= since_epoch
    ]
    if not candidates:
        raise SystemExit(f"no run directory with params/agent.yaml under {LOG_ROOT}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def finalize(pending_path: pathlib.Path, run_dir: pathlib.Path | None) -> pathlib.Path:
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    if run_dir is None:
        # pending was written just before the launcher started
        started = pending_path.stat().st_mtime - 5.0
        run_dir = find_recent_run_dir(started)
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "run_config.json"
    if config_path.exists():
        raise SystemExit(f"refusing to overwrite existing {config_path}")
    record = dict(pending)
    record["phase"] = "finalized"
    record["finalized_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record["run_dir"] = run_dir.as_posix()
    config_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = build_manifest(run_dir)
    (run_dir / "sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="write the pending start record")
    start.add_argument("--run-id", required=True)
    start.add_argument("--task", required=True)
    start.add_argument("--seed", type=int, default=None)
    start.add_argument("extra", nargs=argparse.REMAINDER, help="launcher arguments")

    fin = sub.add_parser("finalize", help="write run_config.json and the SHA-256 manifest")
    fin.add_argument("--pending", type=pathlib.Path, default=None)
    fin.add_argument("--run-id", default=None)
    fin.add_argument("--run-dir", type=pathlib.Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "start":
        path = snapshot_start(args.run_id, args.task, args.seed, args.extra)
        print(f"[evidence] start record: {path}")
        return 0

    pending = args.pending or find_pending(args.run_id)
    config = finalize(pending, args.run_dir)
    print(f"[evidence] run_config.json: {config}")
    print(f"[evidence] manifest: {config.parent / 'sha256_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run-local evidence files. No training result is inferred from file existence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

PROTOCOL_VERSION = "navigation-integrity-v1"
VALIDATION_SEEDS = tuple(range(5))


def acceptance_seeds(episodes: int, seed_start: int = 1000) -> list[int]:
    """Disjoint episode seeds do not imply independent scenes or training runs."""
    if episodes < 1 or seed_start < 0:
        raise ValueError("episodes must be positive and seed_start non-negative")
    seeds = list(range(seed_start, seed_start + episodes))
    if set(seeds).intersection(VALIDATION_SEEDS):
        raise ValueError("acceptance seeds overlap model-selection seeds 0..4")
    return seeds


def fingerprint(path: str | pathlib.Path) -> dict[str, Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def runtime_metadata(root: pathlib.Path) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "mujoco", "gymnasium", "stable-baselines3"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision: str | None = None
    dirty: bool | None = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "source_commit": revision, "source_dirty": dirty}


def write_json_new(path: pathlib.Path, data: Any) -> None:
    """Exclusive creation; reject non-finite JSON and never replace old evidence."""
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def start_run(
    project_root: pathlib.Path, task: str, config: Mapping[str, Any],
    inputs: Mapping[str, str | pathlib.Path], requested: str | None = None,
) -> pathlib.Path:
    """Preflight inputs before creating an exclusive, isolated output directory."""
    artifacts = {key: fingerprint(value) for key, value in inputs.items()}
    metadata = runtime_metadata(project_root)
    if requested is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = project_root / "rl" / "runs" / task / f"{timestamp}_{uuid4().hex[:8]}"
    else:
        path = pathlib.Path(requested).expanduser().resolve()
    manifest = {"schema_version": PROTOCOL_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "config": dict(config), "runtime": metadata, "inputs": artifacts}
    # Validate serializability before creating anything.
    json.dumps(manifest, allow_nan=False)
    path.mkdir(parents=True, exist_ok=False)
    write_json_new(path / "run_config.json", manifest)
    return path

#!/usr/bin/env python3
"""Collect a Go2W run into a paper/report-ready evidence manifest.

The collector is read-only with respect to the run directory.  It writes one
new manifest (and optionally a compact bundle) with hashes and validation
results.  It does not invent timestamps, retrain, or rewrite existing files.

Example:

    python tools/win_train.py fetch mseg-events-full-s0b
    python scripts/collect_run_evidence.py \
        --run-dir rl/runs/_fetched/mseg-events-full-s0b \
        --bundle-dir reports/evidence/mseg-events-full-s0b
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "go2w-evidence-v1"
EVENT_FIELDS = {
    "schema_version", "event_id", "run_id", "kind",
    "event_time", "available_at", "payload",
}
EVENT_KINDS = {"start", "sample", "finish", "label"}
FINAL_DIRS = ("evaluation", "smoke_evaluation")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def relative_files(root: pathlib.Path, exclude: set[pathlib.Path]) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for path in sorted(root.rglob("*")):
        if path in exclude or not path.is_file() or path.is_symlink():
            continue
        files.append(path)
    return files


def validate_events(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty event stream")
    header = json.loads(lines[0])
    if not isinstance(header, dict) or set(header) != {"schema_version", "origin"}:
        raise ValueError("invalid event stream header")
    if header["schema_version"] != 2 or header["origin"] not in (
        "observed", "controlled", "synthetic"
    ):
        raise ValueError("unsupported event stream header")
    events: list[dict[str, Any]] = []
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            raise ValueError(f"blank event line {number}")
        event = json.loads(line)
        if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
            raise ValueError(f"event line {number}: invalid fields")
        if event["schema_version"] != 2:
            raise ValueError(f"event line {number}: unsupported schema")
        if event["kind"] not in EVENT_KINDS:
            raise ValueError(f"event line {number}: invalid kind")
        event_time = float(event["event_time"])
        available_at = float(event["available_at"])
        if not math.isfinite(event_time) or not math.isfinite(available_at):
            raise ValueError(f"event line {number}: non-finite time")
        if event_time < 0 or available_at < event_time:
            raise ValueError(f"event line {number}: invalid time order")
        events.append(event)

    starts = [e for e in events if e["kind"] == "start"]
    finishes = [e for e in events if e["kind"] == "finish"]
    if len(starts) != 1:
        raise ValueError("event stream requires exactly one start")
    if len(finishes) > 1:
        raise ValueError("event stream has multiple finishes")
    start = starts[0]
    finish = finishes[0] if finishes else None
    if finish is not None and (
        finish["event_time"] < start["event_time"]
        or finish["available_at"] < start["available_at"]
    ):
        raise ValueError("finish precedes start")
    samples = sorted(
        (e for e in events if e["kind"] == "sample"),
        key=lambda e: (e["event_time"], e["event_id"]),
    )
    steps = [int(e["payload"]["step"]) for e in samples]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("sample steps must strictly increase in event time")
    if finish is not None:
        for event in samples:
            if event["event_time"] > finish["event_time"]:
                raise ValueError("sample occurs after finish")
    for event in events:
        if event["kind"] == "label":
            if finish is None:
                raise ValueError("label without finish")
            if (
                event["event_time"] < finish["event_time"]
                or event["available_at"] < finish["available_at"]
            ):
                raise ValueError("label precedes finish")
    summary = {
        "kind_counts": {
            kind: sum(1 for e in events if e["kind"] == kind)
            for kind in sorted(EVENT_KINDS)
        },
        "start": start["payload"],
        "finish": finish["payload"] if finish else None,
        "label": next(
            (e["payload"] for e in events if e["kind"] == "label"), None
        ),
        "sample_count": len(samples),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "event_time_min": min(float(e["event_time"]) for e in events),
        "event_time_max": max(float(e["event_time"]) for e in events),
    }
    return summary, events


def stage_summary(run_dir: pathlib.Path) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage_dir in sorted(run_dir.glob("stage*")):
        if not stage_dir.is_dir():
            continue
        record: dict[str, Any] = {
            "completed": (stage_dir / ".completed").exists(),
            "eval_log": None,
            "checkpoints": [],
            "tensorboard": [],
        }
        eval_log = stage_dir / "eval_log.csv"
        if eval_log.is_file():
            with eval_log.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            record["eval_log"] = {
                "rows": len(rows),
                "last": rows[-1] if rows else None,
            }
        for path in sorted(stage_dir.glob("*.zip")) + sorted(stage_dir.glob("*.pkl")):
            record["checkpoints"].append(hash_file(path))
        record["tensorboard"] = [
            hash_file(path)
            for path in sorted((stage_dir / "tensorboard").rglob("*"))
            if path.is_file()
        ] if (stage_dir / "tensorboard").is_dir() else []
        stages[stage_dir.name] = record
    return stages


def final_summary(run_dir: pathlib.Path) -> dict[str, Any] | None:
    for name in FINAL_DIRS:
        path = run_dir / name / "summary.json"
        if path.is_file():
            data = read_json(path)
            return {"directory": name, "summary": data, "artifact": hash_file(path)}
    return None


def load_provenance(run_dir: pathlib.Path, job_dir: pathlib.Path | None) -> dict[str, Any]:
    candidates = [run_dir / "_job"]
    if job_dir is not None:
        candidates.insert(0, job_dir)
    spec = None
    meta = None
    for candidate in candidates:
        spec_path = candidate / "spec.json"
        meta_path = candidate / "meta.json"
        if spec is None and spec_path.is_file():
            spec = read_json(spec_path)
        if meta is None and meta_path.is_file():
            meta = read_json(meta_path)
    return {
        "spec": spec,
        "meta": meta,
        "source": spec.get("source") if isinstance(spec, dict) else None,
        "command": spec.get("command") if isinstance(spec, dict) else None,
        "created_at": spec.get("created_at") if isinstance(spec, dict) else None,
    }


def copy_bundle(run_dir: pathlib.Path, bundle_dir: pathlib.Path,
                manifest_path: pathlib.Path) -> dict[str, Any]:
    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)
    bundle_dir.mkdir(parents=True)
    members: list[str] = ["run_config.json", "events.jsonl"]
    for stage_dir in sorted(run_dir.glob("stage*")):
        eval_log = stage_dir / "eval_log.csv"
        if eval_log.is_file():
            members.append(str(eval_log.relative_to(run_dir)))
    for name in FINAL_DIRS:
        final_dir = run_dir / name
        for filename in ("summary.json", "episodes.json", "metrics.csv", "report.md"):
            path = final_dir / filename
            if path.is_file():
                members.append(str(path.relative_to(run_dir)))
    for filename in ("spec.json", "meta.json", "train.log"):
        path = run_dir / "_job" / filename
        if path.is_file():
            members.append(str(path.relative_to(run_dir)))
    try:
        members.append(str(manifest_path.relative_to(run_dir)))
    except ValueError:
        pass
    copied: list[dict[str, Any]] = []
    for member in members:
        source = run_dir / member
        if not source.is_file():
            continue
        destination = bundle_dir / member
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(hash_file(destination))
    bundle_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "source_run": str(run_dir),
        "members": copied,
    }
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    with bundle_manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(bundle_manifest, ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n")
    return {"path": str(bundle_dir), "members": copied,
            "manifest": hash_file(bundle_manifest_path)}


def collect(run_dir: pathlib.Path, job_dir: pathlib.Path | None,
            out: pathlib.Path, bundle_dir: pathlib.Path | None,
            allow_incomplete: bool) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    config_path = run_dir / "run_config.json"
    events_path = run_dir / "events.jsonl"
    for required in (config_path, events_path):
        if not required.is_file():
            raise FileNotFoundError(f"required artifact missing: {required}")
    if out.exists():
        raise FileExistsError(out)

    config = read_json(config_path)
    event_summary, events = validate_events(events_path)
    final = final_summary(run_dir)
    provenance = load_provenance(run_dir, job_dir)
    stages = stage_summary(run_dir)

    if not allow_incomplete:
        if event_summary["finish"] is None:
            raise ValueError("final evidence requires a finish event")
        if event_summary["label"] is None:
            raise ValueError("final evidence requires a label event")
        if final is None:
            raise ValueError("final evidence requires evaluation/summary.json")

    artifacts = [hash_file(path) for path in relative_files(run_dir, {out})]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "provenance": provenance,
        "config": config,
        "events": event_summary,
        "final": final,
        "stages": stages,
        "artifacts": artifacts,
        "checks": {
            "required_artifacts_present": True,
            "events_valid": True,
            "events_complete": event_summary["finish"] is not None
            and event_summary["label"] is not None,
            "final_summary_present": final is not None,
            "provenance_source_present": provenance["source"] is not None,
            "stage_count": len(stages),
            "completed_stages": sum(
                1 for stage in stages.values() if stage["completed"]
            ),
        },
        "limitations": [
            "The manifest hashes existing artifacts; it does not retrain or replay them.",
            "Eval CSV rows are offline evidence without per-row availability timestamps; "
            "use events.jsonl for strict as-of analysis.",
            "The bundle is a compact copy; checkpoints remain in the full run directory.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n")
    if bundle_dir is not None:
        manifest["bundle"] = copy_bundle(run_dir, bundle_dir, out)
        with out.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect Go2W run evidence")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-dir", default=None,
                        help="Job directory with spec.json/meta.json; defaults to <run-dir>/_job")
    parser.add_argument("--out", default=None,
                        help="New manifest path; defaults to <run-dir>/evidence_manifest.json")
    parser.add_argument("--bundle-dir", default=None,
                        help="New compact bundle directory; never overwritten")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Allow an in-progress run without finish/label/summary")
    args = parser.parse_args(argv)

    run_dir = pathlib.Path(args.run_dir).expanduser()
    job_dir = pathlib.Path(args.job_dir).expanduser() if args.job_dir else None
    out = (pathlib.Path(args.out).expanduser() if args.out
           else run_dir / "evidence_manifest.json")
    bundle_dir = pathlib.Path(args.bundle_dir).expanduser() if args.bundle_dir else None
    manifest = collect(run_dir, job_dir, out, bundle_dir, args.allow_incomplete)
    print(json.dumps({
        "manifest": str(out),
        "run_id": manifest["run_id"],
        "events": manifest["events"]["kind_counts"],
        "checks": manifest["checks"],
        "bundle": manifest.get("bundle", {}).get("path"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

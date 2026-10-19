"""Tests for the Go2W run evidence collector."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from scripts.collect_run_evidence import collect, main


def _write(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _event(event_id: str, kind: str, event_time: float,
           payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "event_id": event_id,
        "run_id": "run-1",
        "kind": kind,
        "event_time": event_time,
        "available_at": event_time,
        "payload": payload,
    }


def _make_run(root: pathlib.Path, monotonic: bool = True,
              complete: bool = True) -> pathlib.Path:
    run = root / "run-1"
    _write(run / "run_config.json", json.dumps({
        "schema_version": "navigation-integrity-v1",
        "config": {"task": "fake", "seed": 0},
        "runtime": {"python": "3.12", "source_commit": None},
        "inputs": {},
    }))
    events = [
        {"schema_version": 2, "origin": "observed"},
        _event("e-start", "start", 100, {
            "task": "fake", "seed": "0", "planned_steps": 20,
        }),
        _event("e-s1", "sample", 110, {"step": 10, "reward": 1.0}),
        _event("e-s2", "sample", 115, {
            "step": 20 if monotonic else 5, "reward": 0.5,
        }),
    ]
    if complete:
        events += [
            _event("e-finish", "finish", 120, {"status": "completed"}),
            _event("e-label", "label", 121, {
                "verdict": "pass", "source": "evaluation",
            }),
        ]
    _write(run / "events.jsonl",
           "\n".join(json.dumps(event) for event in events) + "\n")
    _write(run / "stage1" / "eval_log.csv",
           "timesteps,success_rate\n10000,1.0\n")
    (run / "stage1" / "best_model.zip").write_bytes(b"checkpoint")
    if complete:
        _write(run / "evaluation" / "summary.json", json.dumps({
            "verdict": "pass", "success_rate": 1.0,
        }))
    _write(run / "_job" / "spec.json", json.dumps({
        "job_id": "run-1",
        "source": {"commit": "abc123", "dirty": False},
        "command": ["python", "train.py"],
        "created_at": "2026-10-19T00:00:00Z",
    }))
    _write(run / "_job" / "meta.json", json.dumps({"status": "finished"}))
    return run


class CollectRunEvidenceTests(unittest.TestCase):
    def test_complete_run_manifest_and_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _make_run(root)
            out = run / "evidence_manifest.json"
            bundle = root / "bundle"
            manifest = collect(run, None, out, bundle, allow_incomplete=False)
            self.assertTrue(out.is_file())
            self.assertTrue((bundle / "bundle_manifest.json").is_file())
            self.assertEqual(manifest["events"]["kind_counts"], {
                "finish": 1, "label": 1, "sample": 2, "start": 1,
            })
            self.assertEqual(manifest["provenance"]["source"]["commit"], "abc123")
            self.assertTrue(manifest["checks"]["events_valid"])
            self.assertTrue(manifest["checks"]["events_complete"])
            self.assertTrue(manifest["checks"]["final_summary_present"])
            self.assertTrue(manifest["checks"]["provenance_source_present"])
            self.assertTrue(any(
                item["path"].endswith("events.jsonl")
                for item in manifest["artifacts"]
            ))

    def test_nonmonotonic_steps_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _make_run(root, monotonic=False)
            with self.assertRaises(ValueError):
                collect(run, None, run / "manifest.json", None,
                        allow_incomplete=True)

    def test_incomplete_requires_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _make_run(root, complete=False)
            with self.assertRaises(ValueError):
                collect(run, None, run / "manifest.json", None,
                        allow_incomplete=False)
            manifest = collect(run, None, run / "manifest.json", None,
                               allow_incomplete=True)
            self.assertFalse(manifest["checks"]["events_complete"])

    def test_main_prints_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = _make_run(root)
            self.assertEqual(main(["--run-dir", str(run)]), 0)
            self.assertTrue((run / "evidence_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()

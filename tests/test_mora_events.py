"""Tests for the MoRA-Train v0.2-style event writer and SB3 callback."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from mora.observability.events import EventWriter
from mora.observability.sb3 import Sb3EventCallback


class EventWriterTests(unittest.TestCase):
    def test_event_lifecycle_and_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "events.jsonl"
            writer = EventWriter(
                path,
                "run-1",
                task="fake",
                seed=0,
                planned_steps=10,
            )
            writer.start()
            writer.sample({"step": 1, "reward": np.float32(0.5)})
            writer.finish({"status": "completed"})
            writer.label({"verdict": "pass", "source": "evaluation"})
            writer.close()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["schema_version"], 2)
            events = [json.loads(line) for line in lines[1:]]
            self.assertEqual(
                [event["kind"] for event in events],
                ["start", "sample", "finish", "label"],
            )
            for event in events:
                self.assertLessEqual(event["event_time"], event["available_at"])
            self.assertEqual(events[1]["payload"]["reward"], 0.5)
            self.assertEqual(events[0]["payload"]["task"], "fake")
            self.assertEqual(events[0]["payload"]["planned_steps"], 10)

    def test_duplicate_start_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = EventWriter(pathlib.Path(tmp) / "events.jsonl", "run-2")
            writer.start()
            with self.assertRaises(ValueError):
                writer.start()
            writer.close()

    def test_sample_requires_active_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = EventWriter(pathlib.Path(tmp) / "events.jsonl", "run-3")
            with self.assertRaises(ValueError):
                writer.sample({"step": 1})
            writer.close()

    def test_label_requires_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = EventWriter(pathlib.Path(tmp) / "events.jsonl", "run-4")
            writer.start()
            with self.assertRaises(ValueError):
                writer.label({"verdict": "pass"})
            writer.close()

    def test_available_at_cannot_precede_event_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = EventWriter(pathlib.Path(tmp) / "events.jsonl", "run-5")
            with self.assertRaises(ValueError):
                writer.start(event_time=10.0, available_at=5.0)
            writer.close()

    def test_nan_payload_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = EventWriter(pathlib.Path(tmp) / "events.jsonl", "run-6")
            writer.start()
            with self.assertRaises(ValueError):
                writer.sample({"step": 1, "reward": float("nan")})
            writer.close()

    def test_exclusive_file_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "events.jsonl"
            first = EventWriter(path, "run-7")
            first.close()
            with self.assertRaises(FileExistsError):
                EventWriter(path, "run-8")


class Sb3CallbackTests(unittest.TestCase):
    def test_callback_writes_start_sample_finish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "events.jsonl"
            writer = EventWriter(
                path,
                "run-cb",
                task="fake",
                seed=0,
                planned_steps=20,
            )
            callback = Sb3EventCallback(writer, sample_interval=10)
            callback._on_training_start()
            callback.num_timesteps = 10
            callback.locals = {
                "infos": [{"progress": 0.5, "goal": True, "segment": 1}],
                "rewards": [1.0],
            }
            self.assertTrue(callback._on_step())
            callback.num_timesteps = 20
            callback._on_training_end()

            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()[1:]
            ]
            self.assertEqual([event["kind"] for event in events], [
                "start", "sample", "finish"
            ])
            sample = events[1]["payload"]
            self.assertEqual(sample["step"], 10)
            self.assertEqual(sample["reward"], 1.0)
            self.assertEqual(sample["progress"], 0.5)
            self.assertEqual(sample["success_rate"], 1.0)
            self.assertEqual(sample["segment"], 1.0)
            self.assertEqual(events[2]["payload"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()

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

    def test_step_offset_keeps_global_steps_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "events.jsonl"
            writer = EventWriter(
                path,
                "run-offset",
                task="fake",
                seed=0,
                planned_steps=20,
            )
            callback = Sb3EventCallback(
                writer,
                sample_interval=5,
                manage_lifecycle=False,
                step_offset=10,
            )
            writer.start()
            callback.num_timesteps = 5
            callback.locals = {"infos": [], "rewards": []}
            self.assertTrue(callback._on_step())
            callback.set_step_offset(20)
            callback.num_timesteps = 5
            self.assertTrue(callback._on_step())
            writer.finish({"status": "completed"})
            writer.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()[1:]
            ]
            samples = [e for e in events if e["kind"] == "sample"]
            self.assertEqual([e["payload"]["step"] for e in samples], [15, 25])

    def test_training_start_resets_stage_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "events.jsonl"
            writer = EventWriter(
                path,
                "run-stage",
                task="fake",
                seed=0,
                planned_steps=20000,
            )
            callback = Sb3EventCallback(
                writer,
                sample_interval=5,
                manage_lifecycle=False,
                step_offset=10000,
            )
            writer.start()
            callback.num_timesteps = 10000
            callback._on_training_start()
            callback.num_timesteps = 15000
            callback.locals = {"infos": [], "rewards": []}
            self.assertTrue(callback._on_step())
            writer.finish({"status": "completed"})
            writer.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()[1:]
            ]
            samples = [e for e in events if e["kind"] == "sample"]
            self.assertEqual([e["payload"]["step"] for e in samples], [15000])

    def test_callback_can_leave_lifecycle_to_caller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "events.jsonl"
            writer = EventWriter(
                path,
                "run-manual",
                task="fake",
                seed=0,
                planned_steps=10,
            )
            callback = Sb3EventCallback(
                writer, sample_interval=1, manage_lifecycle=False
            )
            writer.start()
            callback.num_timesteps = 1
            callback.locals = {
                "infos": [{"progress": 0.2, "goal": False}],
                "rewards": [0.5],
            }
            self.assertTrue(callback._on_step())
            callback._on_training_end()
            self.assertFalse(writer.finished)
            writer.finish({"status": "completed"})
            writer.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()[1:]
            ]
            self.assertEqual(
                [event["kind"] for event in events],
                ["start", "sample", "finish"],
            )


if __name__ == "__main__":
    unittest.main()

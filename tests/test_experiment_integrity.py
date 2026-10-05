"""Evidence path and seed protocol regressions; no simulation dependencies."""

import hashlib
import json
import pathlib
import tempfile
import unittest

from rl.experiment_io import acceptance_seeds, fingerprint, start_run, write_json_new


class ExperimentIOTests(unittest.TestCase):
    def test_default_acceptance_seeds_are_disjoint(self):
        self.assertEqual(acceptance_seeds(3), [1000, 1001, 1002])

    def test_invalid_or_overlapping_seeds_are_rejected(self):
        for episodes, start in [(0, 1000), (-1, 1000), (2, -1), (1, 4), (8, 0)]:
            with self.subTest(episodes=episodes, start=start), self.assertRaises(ValueError):
                acceptance_seeds(episodes, start)

    def test_fingerprint_matches_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "model.zip"
            path.write_bytes(b"fixture bytes, not a trained policy")
            record = fingerprint(path)
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["bytes"], path.stat().st_size)

    def test_missing_artifact_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            output = root / "new-run"
            with self.assertRaises(FileNotFoundError):
                start_run(root, "fixture", {}, {"model": root / "absent"}, str(output))
            self.assertFalse(output.exists())

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            output = start_run(root, "fixture", {"seed": 7}, {}, str(root / "run"))
            before = (output / "run_config.json").read_bytes()
            with self.assertRaises(FileExistsError):
                start_run(root, "fixture", {"seed": 9}, {}, str(output))
            self.assertEqual(before, (output / "run_config.json").read_bytes())

    def test_metadata_records_configuration_not_an_invented_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            output = start_run(root, "fixture", {"seed": 7, "smoke": True}, {})
            record = json.loads((output / "run_config.json").read_text())
            self.assertEqual(record["config"]["seed"], 7)
            self.assertTrue(record["config"]["smoke"])
            self.assertNotIn("success_rate", record)
            self.assertIsNone(record["runtime"]["source_commit"])

    def test_json_is_exclusive_and_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "summary.json"
            with self.assertRaises(ValueError):
                write_json_new(path, {"metric": float("nan")})
            self.assertFalse(path.exists())
            write_json_new(path, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                write_json_new(path, {"status": "replaced"})
            self.assertEqual(json.loads(path.read_text())["status"], "complete")

    def test_invalid_config_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            output = root / "invalid"
            with self.assertRaises(ValueError):
                start_run(root, "fixture", {"seed": float("inf")}, {}, str(output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

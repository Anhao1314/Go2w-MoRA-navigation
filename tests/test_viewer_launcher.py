"""MuJoCo 查看器启动器测试：校验、命令构造、状态与启停。"""

import os
import pathlib
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rl import viewer_launcher as vl  # noqa: E402


class ValidatorTest(unittest.TestCase):
    def test_validators(self):
        self.assertTrue(vl.valid_task("balance"))
        self.assertFalse(vl.valid_task("evil"))
        self.assertTrue(vl.valid_seed("seed01"))
        self.assertFalse(vl.valid_seed("01"))
        self.assertFalse(vl.valid_seed("seed01/../x"))
        self.assertTrue(vl.valid_scenario("slope"))
        self.assertFalse(vl.valid_scenario("../../etc"))


class FindModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name) / "runs"
        run = self.root / "balance" / "seed01"
        run.mkdir(parents=True)
        (run / "best_model.zip").write_bytes(b"m")
        (run / "best_vec_normalize.pkl").write_bytes(b"v")

    def tearDown(self):
        self.tmp.cleanup()

    def test_find_model(self):
        found = vl.find_model("balance", "seed01", runs_root=self.root)
        assert found is not None
        model, vec = found
        self.assertEqual(model.name, "best_model.zip")
        self.assertEqual(vec.name, "best_vec_normalize.pkl")

    def test_missing_model(self):
        self.assertIsNone(vl.find_model("balance", "seed02", runs_root=self.root))
        self.assertIsNone(vl.find_model("traverse_slope", "seed01", runs_root=self.root))

    def test_invalid_input(self):
        self.assertIsNone(vl.find_model("bad", "seed01", runs_root=self.root))
        self.assertIsNone(vl.find_model("balance", "01", runs_root=self.root))


class BuildCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name) / "runs"
        for task in vl.TASKS:
            run = self.root / task / "seed00"
            run.mkdir(parents=True)
            (run / "best_model.zip").write_bytes(b"m")
            (run / "best_vec_normalize.pkl").write_bytes(b"v")

    def tearDown(self):
        self.tmp.cleanup()

    def test_traverse_has_scenario(self):
        cmd = vl.build_command("traverse_slope", "seed00", "slope", runs_root=self.root)
        self.assertIn("--task", cmd)
        self.assertIn("traverse_slope", cmd)
        self.assertIn("--scenario", cmd)
        self.assertIn("slope", cmd)
        self.assertIn("best_model.zip", " ".join(cmd))

    def test_balance_no_scenario(self):
        cmd = vl.build_command("balance", "seed00", "slope", runs_root=self.root)
        self.assertNotIn("--scenario", cmd)

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            vl.build_command("balance", "seed99", runs_root=self.root)


class StateAndLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.state_path = self.base / "viewers.json"
        self.log_dir = self.base / "logs"
        self.runs_root = self.base / "runs"
        run = self.runs_root / "balance" / "seed00"
        run.mkdir(parents=True)
        (run / "best_model.zip").write_bytes(b"m")
        (run / "best_vec_normalize.pkl").write_bytes(b"v")
        self.fake_python = self.base / "fake_viewer.py"
        self.fake_python.write_text(
            "import time\n"
            "import sys\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        self.orig_script = vl.VIEWER_SCRIPT
        vl.VIEWER_SCRIPT = self.fake_python

    def tearDown(self):
        vl.VIEWER_SCRIPT = self.orig_script
        self.tmp.cleanup()

    def test_state_roundtrip(self):
        vl.write_state({"balance/seed00": {"pid": 1, "task": "balance"}}, self.state_path)
        state = vl.read_state(self.state_path)
        self.assertEqual(state["balance/seed00"]["pid"], 1)
        self.assertEqual(vl.read_state(self.base / "nope.json"), {})

    def test_is_alive(self):
        self.assertTrue(vl.is_alive(os.getpid()))
        self.assertFalse(vl.is_alive(99999999))

    def test_status_alive_flag(self):
        vl.write_state({"x": {"pid": os.getpid(), "started_at": time.time()}}, self.state_path)
        items = vl.viewer_status(state_path=self.state_path)
        self.assertTrue(items[0]["alive"])

    def test_launch_invalid(self):
        result = vl.launch_viewer("bad", "seed00", state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertTrue(result.get("invalid"))

    def test_launch_missing_model(self):
        result = vl.launch_viewer("balance", "seed99", state_path=self.state_path)
        self.assertFalse(result["ok"])
        self.assertIn("best_model", result["error"])

    def test_launch_no_display(self):
        result = vl.launch_viewer(
            "balance", "seed00", state_path=self.state_path,
            runs_root=self.runs_root, display="",
        )
        self.assertFalse(result["ok"])
        self.assertIn("DISPLAY", result["error"])

    def test_launch_and_stop(self):
        result = vl.launch_viewer(
            "balance", "seed00",
            state_path=self.state_path,
            runs_root=self.runs_root,
            log_dir=self.log_dir,
            display=":0",
        )
        self.assertTrue(result["ok"], result)
        pid = result["pid"]
        self.assertTrue(vl.is_alive(pid))
        info = vl.viewer_info("balance", "seed00", state_path=self.state_path)
        self.assertTrue(info["running"])
        self.assertEqual(info["pid"], pid)

        # 重复启动应拒绝
        dup = vl.launch_viewer(
            "balance", "seed00",
            state_path=self.state_path,
            runs_root=self.runs_root,
            log_dir=self.log_dir,
            display=":0",
        )
        self.assertFalse(dup["ok"])
        self.assertIn("已在运行", dup["error"])

        stop = vl.stop_viewer("balance", "seed00", state_path=self.state_path)
        self.assertTrue(stop["ok"])
        info = vl.viewer_info("balance", "seed00", state_path=self.state_path)
        self.assertFalse(info["running"])
        # 再次停止应报未运行
        again = vl.stop_viewer("balance", "seed00", state_path=self.state_path)
        self.assertFalse(again["ok"])


if __name__ == "__main__":
    unittest.main()

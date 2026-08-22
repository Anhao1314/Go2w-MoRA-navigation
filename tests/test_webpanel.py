"""Web 面板纯函数测试：渲染、告警去重、历史、路径安全、TB 字段透传。"""

import json
import tempfile
import time
import unittest
from pathlib import Path

from rl.monitor import EvalRow, SeedState, snapshot_dict
from rl.report import make_bar_svg, make_lines_svg
from rl.task_info import TASK_INFO, task_info
from rl.webpanel import (
    append_snapshot_history,
    build_curriculum_status,
    cloud_sync_health,
    dedupe_alerts,
    list_demos,
    read_high_level_runs,
    read_curriculum_stages,
    read_resource_history,
    read_snapshot_history,
    render_markdown,
    safe_media_file,
    safe_report_file,
    seed_demos,
    tail_file,
    valid_task_seed,
)


def make_seed() -> SeedState:
    row = EvalRow(1_000_000, 100.0, 0.0, 1000.0)
    return SeedState(
        task="balance",
        seed="01",
        eval_row=row,
        eval_reward=100.0,
        tb_reward=None,
        tb_ep_len=None,
        timesteps=1_000_000,
        completed=False,
        alive=True,
        stale=False,
        nan=False,
        speed=500.0,
        eta=3600.0,
        reward_history=[100.0],
        source_mtime=time.time(),
        sample_time=time.time(),
        tb_std=0.01,
        tb_value_loss=0.1,
        tb_kl=0.02,
        tb_curve=[[100.0, 99.0], [200.0, 101.0]],
    )


class MarkdownRenderTest(unittest.TestCase):
    def test_headings_and_bold(self):
        html_text = render_markdown("# 标题\n\n**加粗**")
        self.assertIn("<h1>标题</h1>", html_text)
        self.assertIn("<b>加粗</b>", html_text)

    def test_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        html_text = render_markdown(md)
        self.assertIn("<table>", html_text)
        self.assertIn("<th>A</th>", html_text)
        self.assertIn("<td>1</td>", html_text)

    def test_list_and_image(self):
        md = "- 甲\n- 乙\n\n![curve](curve.svg)"
        html_text = render_markdown(md)
        self.assertIn("<ul>", html_text)
        self.assertIn("<li>甲</li>", html_text)
        self.assertIn("<img src=\"curve.svg\"", html_text)


class AlertDedupeTest(unittest.TestCase):
    def test_dedupe_and_timestamps(self):
        seen: dict = {}
        now = time.time()
        alerts = [{"kind": "stale", "key": "balance/seed00", "message": "m1"}]
        current, history = dedupe_alerts(alerts, seen, now)
        self.assertEqual(len(current), 1)
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(current[0]["first_seen"], now)
        self.assertAlmostEqual(current[0]["last_seen"], now)

        current, history = dedupe_alerts(alerts, seen, now + 60)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen["stale:balance/seed00"]["last_seen"], now + 60)


class PathSafetyTest(unittest.TestCase):
    def test_traversal_rejected(self):
        self.assertIsNone(safe_report_file("../secret.txt"))
        self.assertIsNone(safe_report_file("../../etc/passwd"))
        self.assertIsNone(safe_report_file("balance/seed00/missing.md"))
        self.assertIsNone(safe_media_file("../secret.txt"))
        self.assertIsNone(safe_media_file("rl/runs/balance/seed00/best_model.zip"))


class DemoVideoTest(unittest.TestCase):
    def test_list_demos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rl_balance_seed00.mp4").write_bytes(b"mp4")
            (root / "rl_balance_seed00.gif").write_bytes(b"gif")
            (root / "rl_traverse_slope_seed01.mp4").write_bytes(b"mp4")
            (root / "unrelated.txt").write_text("x", encoding="utf-8")
            items = list_demos(root)
        self.assertEqual(len(items), 2)
        by_key = {(d["task"], d["seed"]): d for d in items}
        self.assertEqual(
            by_key[("balance", "seed00")]["mp4"], "/media/rl_balance_seed00.mp4"
        )
        self.assertEqual(
            by_key[("balance", "seed00")]["gif"], "/media/rl_balance_seed00.gif"
        )
        self.assertIn("mtime", by_key[("traverse_slope", "seed01")])

    def test_seed_demos_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "rl_balance_seed00.mp4").write_bytes(b"mp4")
            demos = seed_demos("balance", "seed00", root)
            missing = seed_demos("balance", "seed01", root)
        self.assertEqual(demos["mp4"], "/media/rl_balance_seed00.mp4")
        self.assertIsNone(missing["mp4"])
        self.assertIsNone(missing["gif"])


class HistoryAndActionsTest(unittest.TestCase):
    def test_history_append_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.jsonl"
            for step, reward in ((100, 10), (200, 20), (300, 30)):
                snap = {
                    "seeds": [{
                        "key": "balance/seed01",
                        "timesteps": step,
                        "eval_reward": reward,
                        "tb_reward": None,
                    }]
                }
                append_snapshot_history(snap, path, cap=2)
            points = read_snapshot_history("balance/seed01", path)
            self.assertEqual(points, [(200.0, 20.0), (300.0, 30.0)])

    def test_tail_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.txt"
            path.write_text("\n".join(f"line{i}" for i in range(120)), encoding="utf-8")
            self.assertEqual(tail_file(path, 3), "line117\nline118\nline119")

    def test_valid_task_seed(self):
        self.assertTrue(valid_task_seed("balance", "seed01"))
        self.assertFalse(valid_task_seed("balance", "seed1"))
        self.assertFalse(valid_task_seed("bad", "seed01"))

    def test_bar_svg(self):
        svg = make_bar_svg(["flat", "slope"], [0.8, 0.6], "场景成功率")
        self.assertIn("<svg", svg)
        self.assertIn("80%", svg)

    def test_snapshot_tb_fields(self):
        states = [make_seed()]
        snap = snapshot_dict(
            states,
            [],
            {"cpu": 50.0, "mem": 60.0, "load": [1.0, 2.0, 3.0]},
            [],
        )
        self.assertEqual(snap["seeds"][0]["tb_std"], 0.01)
        self.assertEqual(snap["seeds"][0]["tb_value_loss"], 0.1)
        self.assertEqual(snap["seeds"][0]["tb_kl"], 0.02)
        self.assertEqual(len(snap["seeds"][0]["tb_curve"]), 2)


class CloudSyncHealthTest(unittest.TestCase):
    def test_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud_sync.log"
            path.write_text("...\n完成。本地 monitor 已可看到最新进度。\n", encoding="utf-8")
            health = cloud_sync_health(path)
        self.assertTrue(health["ok"])
        self.assertFalse(health["fail"])
        self.assertFalse(health["stale"])

    def test_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cloud_sync.log"
            path.write_text("rsync: connection refused\n", encoding="utf-8")
            health = cloud_sync_health(path)
        self.assertTrue(health["fail"])
        self.assertFalse(health["ok"])

    def test_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            health = cloud_sync_health(Path(tmp) / "nope.log")
        self.assertFalse(health["exists"])
        self.assertTrue(health["stale"])


class CurriculumStatusTest(unittest.TestCase):
    def test_read_curriculum_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage1 = root / "stage1_straight"
            stage1.mkdir()
            (stage1 / "eval_log.csv").write_text(
                "timesteps,mean_reward,std_reward,mean_ep_len\n"
                "500000,-20.5,0.1,800.0\n",
                encoding="utf-8",
            )
            (stage1 / "best_model.zip").write_bytes(b"model")
            root.joinpath("stage2_big_curve").mkdir()
            stages = read_curriculum_stages(root)
            self.assertEqual(stages[0]["timesteps"], 500000.0)
            self.assertEqual(stages[0]["reward"], -20.5)
            self.assertEqual(stages[0]["ep_len"], 800.0)
            self.assertTrue(stages[0]["best_model"])
            self.assertTrue(stages[1]["exists"])
            self.assertEqual(stages[1]["timesteps"], 0.0)
            self.assertIsNone(stages[3]["reward"])

    def test_build_status_running(self):
        data = [
            {"name": "stage1_straight", "exists": True, "timesteps": 1_000_000,
             "reward": -33.0, "ep_len": 1500.0, "best_model": True},
            {"name": "stage2_big_curve", "exists": True, "timesteps": 100_000,
             "reward": -25.0, "ep_len": 90.0, "best_model": False},
            {"name": "stage3_mid_curve", "exists": False, "timesteps": 0.0,
             "reward": None, "ep_len": None, "best_model": False},
            {"name": "stage4_target", "exists": False, "timesteps": 0.0,
             "reward": None, "ep_len": None, "best_model": False},
        ]
        status = build_curriculum_status(
            data, running=True, current_stage="stage2_big_curve"
        )
        self.assertEqual([s["status"] for s in status["stages"]],
                         ["completed", "running", "pending", "pending"])
        self.assertFalse(status["completed"])
        self.assertAlmostEqual(status["total_steps"], 1_100_000.0)

    def test_build_status_completed(self):
        data = [
            {"name": "stage1_straight", "exists": True, "timesteps": 1_000_000,
             "reward": 10.0, "ep_len": 500.0, "best_model": True},
            {"name": "stage2_big_curve", "exists": True, "timesteps": 1_000_000,
             "reward": 12.0, "ep_len": 600.0, "best_model": True},
            {"name": "stage3_mid_curve", "exists": True, "timesteps": 1_000_000,
             "reward": 8.0, "ep_len": 700.0, "best_model": True},
            {"name": "stage4_target", "exists": True, "timesteps": 1_000_000,
             "reward": 15.0, "ep_len": 800.0, "best_model": True},
        ]
        status = build_curriculum_status(
            data, running=False, current_stage=None,
            config={"final_eval": {"success_rate": 0.6}},
            report=True,
        )
        self.assertTrue(status["completed"])
        self.assertTrue(status["report"])
        self.assertEqual([s["status"] for s in status["stages"]],
                         ["completed"] * 4)

    def test_build_status_stopped(self):
        data = [
            {"name": "stage1_straight", "exists": True, "timesteps": 800_000,
             "reward": -30.0, "ep_len": 1500.0, "best_model": True},
            {"name": "stage2_big_curve", "exists": False, "timesteps": 0.0,
             "reward": None, "ep_len": None, "best_model": False},
        ]
        status = build_curriculum_status(
            data, running=False, current_stage=None,
            stage_steps=1_000_000, total_target=2_000_000,
        )
        self.assertEqual(status["stages"][0]["status"], "stopped")
        self.assertEqual(status["stages"][1]["status"], "pending")


class HighLevelPanelTest(unittest.TestCase):
    def test_read_high_level_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed00"
            seed.mkdir()
            (seed / "eval_log.csv").write_text(
                "timesteps,mean_reward,std_reward,mean_ep_len,"
                "fixed_success_rate,fixed_mean_x,random_success_rate,random_mean_x\n"
                "200000,282.4,0.0,694.0,1.0,1.3541,0.8,0.861\n",
                encoding="utf-8",
            )
            (seed / "her_log.csv").write_text(
                "timesteps,buffer_size,loss,positive_ratio\n"
                "240000,20000,0.0507,0.1445\n",
                encoding="utf-8",
            )
            (seed / ".completed").write_text("", encoding="utf-8")
            (seed / "best_model.zip").write_bytes(b"m")
            runs = read_high_level_runs([root])
            self.assertEqual(len(runs), 1)
            r = runs[0]
            self.assertEqual(r["timesteps"], 200000.0)
            self.assertEqual(r["fixed_success_rate"], 1.0)
            self.assertEqual(r["random_success_rate"], 0.8)
            self.assertAlmostEqual(r["random_mean_x"], 0.861)
            self.assertTrue(r["completed"])
            self.assertTrue(r["best_model"])
            self.assertEqual(r["her"]["buffer_size"], 20000)


class TaskInfoTest(unittest.TestCase):
    def test_all_tasks_covered(self):
        from rl.monitor import TASKS

        self.assertEqual(
            set(TASK_INFO),
            set(TASKS),
            "TASK_INFO 必须覆盖面板的全部训练任务",
        )

    def test_required_fields_nonempty(self):
        for key, info in TASK_INFO.items():
            self.assertIn("name", info)
            self.assertIn("desc", info)
            self.assertIn("requirements", info)
            self.assertIn("goal", info)
            self.assertIn("params", info)
            for field in ("name", "desc", "goal", "params"):
                self.assertTrue(str(info[field]).strip(), f"{key}.{field} 不应为空")
            self.assertGreater(len(info["requirements"]), 0, f"{key} 至少一条要求")

    def test_task_info_accessor(self):
        self.assertIs(task_info("balance"), TASK_INFO["balance"])
        self.assertIsNone(task_info("unknown_task"))


class ResourceHistoryAndSvgTest(unittest.TestCase):
    def test_read_resource_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.jsonl"
            now = time.time()
            path.write_text(
                json.dumps({"time": now - 100, "resources": {"mem_available_mb": 4096.0}}) + "\n"
                + json.dumps({"time": now, "resources": {"mem_available_mb": 2048.0}}) + "\n",
                encoding="utf-8",
            )
            rows = read_resource_history(path)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[-1][1]["mem_available_mb"], 2048.0)

    def test_lines_svg(self):
        svg = make_lines_svg([("可用内存", [(0.0, 4.0), (1.0, 3.0)], "#4fc3f7")], "资源")
        self.assertIn("<svg", svg)
        self.assertIn("<polyline", svg)
        self.assertIn("可用内存", svg)


class TokenAuthTest(unittest.TestCase):
    def test_authorized_sources(self):
        import rl.webpanel as wp

        class FakeHandler:
            def __init__(self, path, headers):
                self.path = path
                self.headers = headers

        old = wp.TOKEN
        wp.TOKEN = "secret123"
        try:
            self.assertTrue(wp._authorized(FakeHandler("/?token=secret123", {})))
            self.assertTrue(wp._authorized(
                FakeHandler("/", {"Authorization": "Bearer secret123"})))
            self.assertTrue(wp._authorized(
                FakeHandler("/", {"Cookie": "go2w_token=secret123"})))
            self.assertFalse(wp._authorized(FakeHandler("/", {})))
        finally:
            wp.TOKEN = old

    def test_no_token_no_auth(self):
        import rl.webpanel as wp

        class FakeHandler:
            def __init__(self):
                self.path = "/"
                self.headers = {}

        old = wp.TOKEN
        wp.TOKEN = ""
        try:
            self.assertTrue(wp._authorized(FakeHandler()))
        finally:
            wp.TOKEN = old


if __name__ == "__main__":
    unittest.main()

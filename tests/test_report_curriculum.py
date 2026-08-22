"""scripts/report_curriculum.py 纯函数测试：解析、汇总、报告渲染。"""

import tempfile
import unittest
from pathlib import Path

from scripts.report_curriculum import (
    build_report_text,
    metrics_row,
    read_eval_csv,
    stage_summaries,
)


class ReportCurriculumTest(unittest.TestCase):
    def test_read_eval_csv_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_eval_csv(Path(tmp) / "nope.csv"), [])

    def test_read_eval_csv_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_log.csv"
            path.write_text(
                "timesteps,mean_reward,std_reward,mean_ep_len\n"
                "100000,-1.0,0.1,50.0\n"
                "200000,-2.0,0.2,60.0\n",
                encoding="utf-8",
            )
            rows = read_eval_csv(path)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[1]["mean_reward"], -2.0)
        self.assertAlmostEqual(rows[1]["mean_ep_len"], 60.0)

    def test_stage_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "stage1_straight"
            d.mkdir()
            (d / "eval_log.csv").write_text(
                "timesteps,mean_reward,std_reward,mean_ep_len\n"
                "1000000,-30.0,0.5,1500.0\n",
                encoding="utf-8",
            )
            (d / "best_model.zip").write_bytes(b"m")
            (d / "final_model.zip").write_bytes(b"m")
            (d / ".completed").write_text("", encoding="utf-8")
            stages = stage_summaries(
                root,
                [{"name": "stage1_straight", "corridor": 1.0, "amp": 0.0,
                  "threshold": 0.8}],
            )
        self.assertEqual(stages[0]["eval_points"], 1)
        self.assertAlmostEqual(stages[0]["last_reward"], -30.0)
        self.assertTrue(stages[0]["best_model"])
        self.assertTrue(stages[0]["completed"])

    def test_build_report_text_fail(self):
        stages = [
            {
                "name": "stage4_target", "corridor": 0.4, "amp": 0.35,
                "threshold": 0.6, "eval_points": 20, "last_reward": -16.7,
                "last_ep_len": 301.0, "completed": True,
            }
        ]
        agg = {
            "success_rate": 0.0, "distance": 0.08, "time_to_goal": None,
            "falls": 1.0, "total_reward": -47.46, "mean_base_reward": 0.93,
            "max_dev": 0.405, "nan": False,
        }
        text = build_report_text(stages, agg, "2026-08-21 10:00:00")
        self.assertIn("❌ 未通过", text)
        self.assertIn("0% ≥ 60%", text)
        self.assertIn("站桩", text)
        self.assertIn("stage4_target", text)

    def test_metrics_row_columns(self):
        agg = {
            "task": "traverse_curve", "max_dev": 0.4, "min_clear": None,
            "dual_hold": 0.0, "recovered": False, "settle_seconds": None,
            "success": False, "success_rate": 0.0, "distance": 0.08,
            "time_to_goal": None, "falls": 1.0, "total_reward": -47.0,
            "mean_base_reward": 0.9, "nan": False,
        }
        row = metrics_row(agg)
        self.assertEqual(row["label"], "RL PPO (curve)")
        self.assertEqual(row["success_rate"], 0.0)
        self.assertEqual(row["distance"], 0.08)


if __name__ == "__main__":
    unittest.main()

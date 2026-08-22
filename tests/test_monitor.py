"""监控脚本单元测试：解析、趋势、告警、节流、聚合（纯函数，不依赖真实训练）。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from rl.monitor import (
    EvalRow,
    SeedState,
    aggregate_tasks,
    evaluate_alerts,
    external_seeds,
    parse_eval_csv,
    resource_alerts,
    resource_snapshot,
    sparkline,
    throttle_alerts,
)


def make_state(
    task: str = "balance",
    seed: str = "00",
    timesteps: float = 4_000_000,
    eval_reward: float | None = 999.0,
    tb_reward: float | None = None,
    completed: bool = False,
    alive: bool = True,
    stale: bool = False,
    nan: bool = False,
    speed: float | None = 500.0,
) -> SeedState:
    row = EvalRow(timesteps, eval_reward or 0.0, 0.0, 1000.0)
    return SeedState(
        task=task,
        seed=seed,
        eval_row=row if eval_reward is not None else None,
        eval_reward=eval_reward,
        tb_reward=tb_reward,
        tb_ep_len=None,
        timesteps=timesteps,
        completed=completed,
        alive=alive,
        stale=stale,
        nan=nan,
        speed=speed,
        eta=None,
        reward_history=[eval_reward] if eval_reward is not None else [],
        source_mtime=time.time(),
        sample_time=time.time(),
    )


class MonitorTest(unittest.TestCase):
    def test_parse_eval_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval_log.csv"
            path.write_text(
                "timesteps,mean_reward,std_reward,mean_ep_len\n"
                "100000,10.5,0.2,1000.0\n"
                "nan,nan,nan,nan\n"
                "150000,12.0,0.3,1000.0\n",
                encoding="utf-8",
            )
            rows, nan = parse_eval_csv(path)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[-1].mean_reward, 12.0)
        self.assertTrue(nan)

    def test_sparkline(self):
        self.assertEqual(sparkline([5.0, 5.0, 5.0]), "▄▄▄")
        self.assertEqual(sparkline([0.0, 10.0])[-1], "█")
        self.assertEqual(sparkline([]), "—")

    def test_alerts(self):
        states = [
            make_state(seed="00", alive=False, eval_reward=800.0, stale=True, nan=True),
            make_state(seed="01", eval_reward=100.0),
        ]
        alerts = evaluate_alerts(
            states,
            prev_alive={"balance/seed00": True, "balance/seed01": True},
            prev_reward={"balance/seed00": 999.0, "balance/seed01": 200.0},
            min_reward=150.0,
            now=time.time(),
        )
        kinds = {k for k, _, _ in alerts}
        self.assertIn("process_died", kinds)
        self.assertIn("stale", kinds)
        self.assertIn("nan", kinds)
        self.assertIn("reward_drop", kinds)
        self.assertIn("low_reward", kinds)

    def test_throttle(self):
        now = time.time()
        alerts = [("stale", "balance/seed00", "msg")]
        last_sent: dict[str, float] = {}
        first = throttle_alerts(alerts, last_sent, now)
        second = throttle_alerts(alerts, last_sent, now + 60)
        third = throttle_alerts(alerts, last_sent, now + 601)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0)
        self.assertEqual(len(third), 1)

    def test_aggregate_tasks(self):
        states = [
            make_state(task="balance", seed="00", timesteps=8_000_000, completed=True),
            make_state(task="balance", seed="01", timesteps=4_000_000, eval_reward=998.0),
        ]
        rows = aggregate_tasks(states, seeds_total=3, target=8_000_000)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["done"], 1)
        self.assertEqual(rows[0]["total"], 3)
        self.assertAlmostEqual(rows[0]["best"], 999.0)
        self.assertEqual(rows[0]["total_steps"], 12_000_000)

    def test_resource_alerts(self):
        alerts = resource_alerts({"mem_available_mb": 1024.0, "swap_percent": 80.0})
        kinds = {k for k, _, _ in alerts}
        self.assertIn("low_mem", kinds)
        self.assertIn("high_swap", kinds)
        self.assertFalse(resource_alerts({"mem_available_mb": 4096.0, "swap_percent": 10.0}))

    def test_resource_snapshot_fields(self):
        res = resource_snapshot()
        for key in ("mem_available_mb", "mem_total_mb", "swap_used_mb",
                    "swap_total_mb", "swap_percent"):
            self.assertIn(key, res)
        self.assertGreaterEqual(res["mem_available_mb"], 0.0)

    def test_external_seeds_owner_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = root / "_guard"
            guard.mkdir(parents=True)
            (guard / "state.json").write_text(
                json.dumps(
                    {
                        "runs": {
                            "balance/seed00": {
                                "status": "external",
                                "owner": "cloud",
                            },
                            "traverse_slope/seed00": {
                                "status": "external",
                                "owner": "local",
                            },
                            "traverse_slope/seed01": {"status": "external"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            import rl.monitor as mon

            with mock.patch.object(mon, "RUNS_DIR", root):
                ext = mon.external_seeds()
        self.assertIn(("balance", "00"), ext)
        self.assertNotIn(("traverse_slope", "00"), ext)
        self.assertIn(("traverse_slope", "01"), ext)  # 未标 owner 视为云端

    def test_remote_alive_by_freshness(self):
        import rl.monitor as mon
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dir = root / "balance" / "seed00"
            seed_dir.mkdir(parents=True)
            (seed_dir / "eval_log.csv").write_text(
                "timesteps,mean_reward,std_reward,mean_ep_len\n"
                "100000,1.0,0.0,100.0\n",
                encoding="utf-8",
            )
            with mock.patch.object(mon, "RUNS_DIR", root):
                st = mon.collect_seed_state(
                    "balance", "00", 8_000_000, 10.0, time.time(),
                    running=set(), external=set(), remote=True,
                )
        self.assertTrue(st.alive)
        self.assertEqual(st.source, "local")


if __name__ == "__main__":
    unittest.main()

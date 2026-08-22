"""训练逻辑单元测试：稳定窗口选 best、超参落盘、balance 离线回放。"""

import pathlib
import tempfile
import unittest

import numpy as np

from rl.train import (
    EvalAndSaveCallback,
    make_lr_schedule,
    norm_should_train,
    pick_best_metric,
    select_best_window,
    train_config_payload,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


class SelectBestWindowTest(unittest.TestCase):
    def test_window_insufficient(self):
        self.assertEqual(select_best_window([(0.0, 1.0)], size=3), (None, None))

    def test_spike_window_rejected(self):
        points = [(1.0, 100.0), (2.0, 100.0), (3.0, -50.0)]
        mean, ts = select_best_window(points, size=3, std_ratio=0.05)
        self.assertIsNone(mean)
        self.assertIsNone(ts)

    def test_stable_window_selected(self):
        points = [(1.0, 90.0), (2.0, 91.0), (3.0, 92.0), (4.0, 40.0)]
        mean, ts = select_best_window(points, size=3, std_ratio=0.05)
        assert mean is not None
        assert ts is not None
        self.assertAlmostEqual(mean, 91.0, places=6)
        self.assertEqual(ts, 3.0)

    def test_highest_stable_window_wins(self):
        points = [(1.0, 80.0), (2.0, 81.0), (3.0, 82.0),
                  (4.0, 99.0), (5.0, 99.5), (6.0, 100.0)]
        mean, ts = select_best_window(points, size=3, std_ratio=0.05)
        assert mean is not None
        assert ts is not None
        self.assertAlmostEqual(mean, 99.5, places=6)
        self.assertEqual(ts, 6.0)

    def test_floor_tolerance(self):
        # 均值很小但波动极小时仍可入选（绝对方差下限 0.1）
        points = [(1.0, 0.05), (2.0, 0.06), (3.0, 0.07)]
        mean, _ = select_best_window(points, size=3, std_ratio=0.05)
        assert mean is not None


class BalanceOfflineReplayTest(unittest.TestCase):
    @unittest.skipUnless(
        (REPO / "rl/runs/balance/seed00/eval_log.csv").exists(),
        "需要真实 balance/seed00 评估日志",
    )
    def test_window_avoids_crash_spike(self):
        path = REPO / "rl/runs/balance/seed00/eval_log.csv"
        points = []
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                points.append((float(parts[0]), float(parts[1])))
        mean, ts = select_best_window(points, size=3, std_ratio=0.05)
        assert mean is not None
        assert ts is not None
        self.assertGreater(mean, 90.0)          # 选的是稳定高分段
        self.assertLess(ts, 3_000_000.0)        # 在崩溃前（末端 -18.8）


class TrainConfigPayloadTest(unittest.TestCase):
    def test_payload_fields(self):
        args = type(
            "Args",
            (),
            {
                "task": "traverse_curve",
                "seed": 0,
                "total_steps": 4_000_000,
                "envs": 4,
                "fresh": True,
                "init_from": None,
                "resume_from": None,
                "resume_norm": None,
                "terrain": "hfield",
                "curriculum_steps": 0,
                "heading_penalty": 2.5,
                "best_window_size": 3,
                "best_window_std_ratio": 0.05,
                "learning_rate": 3e-4,
                "init_norm": None,
                "corridor_width": 0.40,
                "curve_amplitude": 0.35,
                "reward_version": "v4",
            },
        )()
        payload = train_config_payload(args)
        for key in ("task", "total_steps", "heading_penalty", "best_window_size",
                    "curve_half_width", "obs_dim", "action_dim"):
            self.assertIn(key, payload)
        self.assertEqual(payload["heading_penalty"], 2.5)
        self.assertEqual(payload["obs_dim"], 61)
        self.assertEqual(payload["best_metric"], "reward")
        self.assertEqual(payload["fall_penalty"], -5.0)
        self.assertEqual(payload["norm_freeze_steps"], 0)


class LrAndNormScheduleTest(unittest.TestCase):
    def test_make_lr_schedule_warmup(self):
        args = type(
            "Args",
            (),
            {
                "learning_rate": 1.5e-4,
                "lr_warmup_steps": 100_000,
                "lr_warmup_start": 5e-5,
                "lr_warmup_end": 1.5e-4,
            },
        )()
        sched = make_lr_schedule(args, 2_000_000)
        self.assertAlmostEqual(sched(1.0), 5e-5)      # 0 步
        self.assertAlmostEqual(sched(0.96), 5e-5)     # 80k 步
        self.assertAlmostEqual(sched(0.9), 1.5e-4)    # 200k 步

    def test_make_lr_schedule_disabled(self):
        args = type("Args", (), {"learning_rate": 3e-4, "lr_warmup_steps": 0})()
        self.assertEqual(make_lr_schedule(args, 1_000_000), 3e-4)

    def test_norm_should_train(self):
        self.assertFalse(norm_should_train(499_999, 500_000))
        self.assertTrue(norm_should_train(500_000, 500_000))
        self.assertTrue(norm_should_train(100, 0))

    def test_pick_best_metric(self):
        self.assertEqual(pick_best_metric("traverse_curve", "distance", 999.0, 0.9, -5.0), 0.9)
        self.assertEqual(pick_best_metric("traverse_curve", "reward", 999.0, 0.9, -5.0), 999.0)
        self.assertEqual(pick_best_metric("balance", "distance", 1.0, 0.0, 2.0), 2.0)


class ExplorationBoostTest(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.ent_coef = 0.0
            self.learning_rate = 3e-4
            self.lr_schedule = lambda _: 3e-4

    def _make_cb(self, tmp: str) -> tuple[EvalAndSaveCallback, dict, "ExplorationBoostTest.FakeModel"]:
        save_dir = pathlib.Path(tmp)
        boost_state = {"active": False, "std": 0.0}
        cb = EvalAndSaveCallback(
            eval_env=None,  # type: ignore[arg-type]
            eval_freq_timesteps=50_000,
            n_eval_episodes=5,
            save_dir=save_dir,
            n_envs=4,
            task="traverse_curve",
            boost_state=boost_state,
            boost_window=3,
            boost_reward_threshold=-5.0,
            boost_ent_coef=0.01,
            boost_lr=5e-4,
            boost_noise_std=0.1,
            boost_cycles=2,
        )
        model = self.FakeModel()
        cb.model = model  # type: ignore[assignment]
        return cb, boost_state, model

    def test_detect_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb, state, model = self._make_cb(tmp)
            for _ in range(3):
                cb._update_boost(-10.0, 0.1, 301.0)
            self.assertTrue(state["active"])
            self.assertEqual(model.ent_coef, 0.01)
            self.assertEqual(model.lr_schedule(0.5), 5e-4)
            self.assertEqual(cb._boost_remaining, 2)
            # 持续 2 个 eval 周期后自动恢复
            cb._update_boost(-10.0, 0.1, 301.0)
            cb._update_boost(-10.0, 0.1, 301.0)
            self.assertFalse(state["active"])
            self.assertEqual(model.ent_coef, 0.0)
            self.assertEqual(model.lr_schedule(0.5), 3e-4)
            self.assertEqual(cb._boost_remaining, 0)

    def test_early_restore_when_moving(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb, state, model = self._make_cb(tmp)
            for _ in range(3):
                cb._update_boost(-10.0, 0.1, 301.0)
            self.assertTrue(state["active"])
            # 一旦 ep_len 脱离 301，提前恢复正常参数
            cb._update_boost(-3.0, 0.8, 120.0)
            self.assertFalse(state["active"])
            self.assertEqual(model.ent_coef, 0.0)
            self.assertEqual(cb._boost_remaining, 0)

    def test_rapid_fall_stagnation_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb, state, model = self._make_cb(tmp)
            for _ in range(3):
                cb._update_boost(-15.0, 0.1, 50.0, mean_x=0.0)
            self.assertTrue(state["active"])
            self.assertEqual(model.ent_coef, 0.01)
            # 一旦出现前进距离，提前恢复
            cb._update_boost(-5.0, 0.5, 80.0, mean_x=0.6)
            self.assertFalse(state["active"])
            self.assertEqual(model.ent_coef, 0.0)


if __name__ == "__main__":
    unittest.main()

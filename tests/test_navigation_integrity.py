"""Regression checks for seeded resets, normalization and episode-level metrics."""

import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from rl.high_level_env_wrapper import HighLevelEnvWrapper
from rl.normalization import normalize_once
from scripts import evaluate_navigation as benchmark
from scripts import train_junction_curriculum as junction
from scripts import train_multi_segment_curriculum as mseg


class FixtureEnv(gym.Env):
    """Synthetic flags are intentional test inputs, never benchmark evidence."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(61,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    metadata = {}
    task = "traverse_curve"
    scenario = "curve"
    multi_segment = False
    junction = True

    def __init__(self, success=True):
        super().__init__()
        self.base_env = self
        self.data = SimpleNamespace(time=0.0)
        self.goal_arc = 1.4
        self.target = "A"
        self.success = success
        self.steps = 0

    def _goal_arc_len(self):
        return 1.4

    def get_goal_arc(self):
        return self.goal_arc

    def get_remaining_arc(self):
        return self.goal_arc

    def get_target_heading(self):
        return 0.0

    def get_target_goal(self):
        return self.target

    def get_branch_selected(self):
        return "left" if self.target == "A" else "right"

    def get_distance_to_junction(self):
        return 1.0

    def get_distance_to_goal(self):
        return 1.0

    def get_system2_decision_flag(self):
        return 1.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.data.time = 0.0
        opts = options or {}
        self.goal_arc = opts.get("goal_arc", 1.4)
        self.target = opts.get("target_goal", "A")
        return np.ones(61, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.data.time = self.steps * 0.01
        return np.ones(61, dtype=np.float32), 1.0, self.steps == 3, False, {
            "goal": self.success, "passed_subgoal": self.success,
            "progress": float(self.steps),
        }


class ResetSeedTests(unittest.TestCase):
    def test_same_seed_repeats_goal(self):
        env = HighLevelEnvWrapper(FixtureEnv())
        try:
            first, _ = env.reset(seed=17)
            env.reset(seed=99)
            again, _ = env.reset(seed=17)
            np.testing.assert_array_equal(first, again)
        finally:
            env.close()

    def test_separate_instances_use_same_seed(self):
        first = HighLevelEnvWrapper(FixtureEnv())
        second = HighLevelEnvWrapper(FixtureEnv())
        try:
            a, _ = first.reset(seed=42)
            b, _ = second.reset(seed=42)
            np.testing.assert_array_equal(a, b)
        finally:
            first.close()
            second.close()

    def test_none_seed_continues_stream(self):
        env = HighLevelEnvWrapper(FixtureEnv())
        rng = np.random.default_rng(123)
        try:
            env.reset(seed=123)
            self.assertEqual(env.base_env.get_goal_arc(), rng.uniform(0.5, 1.4))
            env.reset()
            self.assertEqual(env.base_env.get_goal_arc(), rng.uniform(0.5, 1.4))
        finally:
            env.close()

    def test_explicit_options_are_not_mutated(self):
        env = HighLevelEnvWrapper(FixtureEnv())
        opts = {"goal_arc": 1.1}
        try:
            env.reset(seed=7, options=opts)
            self.assertEqual(opts, {"goal_arc": 1.1})
            self.assertEqual(env.base_env.get_goal_arc(), 1.1)
        finally:
            env.close()

    def test_junction_samples_target_after_seeding(self):
        env = junction.JunctionWrapper(FixtureEnv(), 2, False)
        try:
            for seed in range(20):
                env.reset(seed=seed)
                expected = "A" if np.random.default_rng(seed).random() < 0.5 else "B"
                self.assertEqual(env.base_env.get_target_goal(), expected)
        finally:
            env.close()


class NormalizationTests(unittest.TestCase):
    def test_rejects_direct_double_normalization(self):
        vec = normalize_once(DummyVecEnv([FixtureEnv]))
        try:
            with self.assertRaises(ValueError):
                normalize_once(vec)
        finally:
            vec.close()

    def test_rejects_nested_normalizer(self):
        vec = VecMonitor(normalize_once(DummyVecEnv([FixtureEnv])))
        try:
            with self.assertRaises(ValueError):
                normalize_once(vec)
        finally:
            vec.close()

    def test_restored_observations_are_normalized_once_and_frozen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "normalization.pkl"
            original = normalize_once(DummyVecEnv([FixtureEnv]), training=False)
            original.obs_rms.mean[:] = 2.0
            original.obs_rms.var[:] = 4.0
            original.save(str(path))
            original.close()
            restored = normalize_once(DummyVecEnv([FixtureEnv]), path, training=False)
            try:
                self.assertNotIsInstance(restored.venv, VecNormalize)
                before = restored.obs_rms.count
                np.testing.assert_allclose(restored.reset(), -0.5, atol=1e-6)
                restored.step(np.zeros((1, 2), dtype=np.float32))
                self.assertEqual(before, restored.obs_rms.count)
                self.assertFalse(restored.norm_reward)
            finally:
                restored.close()

    def test_missing_normalization_fails(self):
        raw = DummyVecEnv([FixtureEnv])
        try:
            with self.assertRaises(FileNotFoundError):
                normalize_once(raw, "/definitely-missing-go2w-normalization.pkl")
        finally:
            raw.close()


class EvaluationMetricTests(unittest.TestCase):
    def test_mseg_persistent_flags_count_once_per_episode(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            mseg, "make_stage_env", side_effect=lambda *a, **kw: FixtureEnv()
        ):
            callback = mseg.MsegEvalCallback(pathlib.Path(folder), 1, 2, False)
            callback.model = benchmark.ControllerPolicy()
            try:
                result = callback._run_eval()
                self.assertEqual(result["passed_rate"], 1.0)
                self.assertEqual(result["success_rate"], 1.0)
            finally:
                callback.eval_norm.close()

    def test_junction_persistent_success_counts_once(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
            junction, "make_stage_env", side_effect=lambda *a, **kw: FixtureEnv()
        ):
            callback = junction.JunctionEvalCallback(pathlib.Path(folder), 1, 2, False)
            callback.model = benchmark.ControllerPolicy()
            try:
                self.assertEqual(callback._run_eval()["success_rate"], 1.0)
            finally:
                callback.eval_norm.close()

    def test_final_acceptance_records_disjoint_seeds_and_outcomes(self):
        with patch.object(mseg, "make_stage_env", side_effect=lambda *a, **kw: FixtureEnv()):
            result = mseg.final_acceptance(
                benchmark.ControllerPolicy(), benchmark.IdentityNormalization(), episodes=2,
            )
            self.assertEqual(result["success_rate"], 1.0)
            self.assertEqual([r["seed"] for r in result["episode_records"]], [1000, 1001])
            self.assertEqual(result["failed_terminations"], 0)

    def test_failure_is_not_labeled_as_a_fall(self):
        with patch.object(mseg, "make_stage_env", side_effect=lambda *a, **kw: FixtureEnv(False)):
            result = mseg.final_acceptance(
                benchmark.ControllerPolicy(), benchmark.IdentityNormalization(), episodes=2,
            )
            self.assertEqual(result["failed_terminations"], 2)
            self.assertNotIn("falls", result)

    def test_junction_final_targets_are_balanced(self):
        with patch.object(junction, "make_stage_env", side_effect=lambda *a, **kw: FixtureEnv()):
            result = junction.final_acceptance(
                benchmark.ControllerPolicy(), benchmark.IdentityNormalization(), episodes=4,
            )
            self.assertEqual([r["target"] for r in result["episode_records"]], ["A", "A", "B", "B"])
            self.assertEqual(result["success_rate"], 1.0)

    def test_cli_rejects_unbalanced_or_missing_checkpoint(self):
        cases = [
            ["--task", "junction", "--method", "controller", "--episodes", "3"],
            ["--task", "multi-segment", "--method", "ppo"],
            ["--task", "multi-segment", "--method", "controller", "--seed-start", "0"],
        ]
        for args in cases:
            with self.subTest(args=args), patch("sys.argv", ["evaluate_navigation", *args]):
                with self.assertRaises(SystemExit) as caught:
                    benchmark.main()
                self.assertEqual(caught.exception.code, 2)


class RealControllerSmokeTests(unittest.TestCase):
    def test_controller_cli_writes_real_episode_evidence(self):
        # Real MuJoCo, not FixtureEnv. Only two episodes per task: integration smoke,
        # not a performance or generalization benchmark. Zero success is valid here.
        with tempfile.TemporaryDirectory() as folder:
            for task in ("multi-segment", "junction"):
                output = pathlib.Path(folder) / task
                argv = ["evaluate_navigation", "--task", task, "--method", "controller",
                        "--episodes", "2", "--output", str(output)]
                with self.subTest(task=task), patch("sys.argv", argv):
                    benchmark.main()
                    summary = json.loads((output / "summary.json").read_text())
                    records = json.loads((output / "episodes.json").read_text())
                    self.assertEqual(summary["status"], "complete")
                    self.assertEqual(len(records), 2)
                    self.assertEqual([r["seed"] for r in records], [1000, 1001])
                    self.assertGreaterEqual(summary["metrics"]["success_rate"], 0.0)
                    self.assertLessEqual(summary["metrics"]["success_rate"], 1.0)
                    self.assertTrue(all(r["steps"] > 0 for r in records))


if __name__ == "__main__":
    unittest.main()

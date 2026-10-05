"""Evaluate controller-only or a trusted PPO checkpoint under one fixed protocol.

This command does not train, select a model, or establish scene generalization.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.experiment_io import (  # noqa: E402
    PROTOCOL_VERSION, acceptance_seeds, start_run, write_json_new,
)
from rl.normalization import normalize_once  # noqa: E402
from scripts import train_junction_curriculum as junction  # noqa: E402
from scripts import train_multi_segment_curriculum as multi_segment  # noqa: E402


class ControllerPolicy:
    """Hold the high-level command at [1, 0]; rules and scripted docking remain."""

    def predict(self, observation: Any, deterministic: bool = True):
        return np.array([1.0, 0.0], dtype=np.float32), None


class IdentityNormalization:
    def normalize_obs(self, observation: Any):
        return np.asarray(observation, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("multi-segment", "junction"), required=True)
    parser.add_argument("--method", choices=("controller", "ppo"), required=True)
    parser.add_argument("--checkpoint", help="Trusted PPO model zip; never auto-downloaded")
    parser.add_argument("--normalization", help="Matching trusted VecNormalize pickle")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--output", help="New output directory; must not already exist")
    args = parser.parse_args()
    try:
        seeds = acceptance_seeds(args.episodes, args.seed_start)
    except ValueError as exc:
        parser.error(str(exc))
    if args.task == "junction" and args.episodes % 2:
        parser.error("junction requires an even episode count for balanced A/B targets")
    inputs: dict[str, str] = {}
    if args.method == "ppo":
        if not args.checkpoint or not args.normalization:
            parser.error("ppo requires both --checkpoint and --normalization")
        inputs = {"checkpoint": args.checkpoint, "normalization": args.normalization}
    elif args.checkpoint or args.normalization:
        parser.error("controller must not receive learned-policy artifacts")
    config = {
        "task": args.task, "method": args.method, "episodes": args.episodes,
        "episode_seeds": seeds, "deterministic_policy": True,
        "domain_randomize": False, "scope": 2,
        "distribution": "fixed_scene_repetition_not_generalization",
        "controller_command": [1.0, 0.0] if args.method == "controller" else None,
        "junction_steering": "scripted_teacher_turn_adjust_unused",
    }
    output = start_run(PROJECT_ROOT, "navigation_evaluation", config, inputs, args.output)
    suite = junction if args.task == "junction" else multi_segment
    vec = None
    try:
        if args.method == "ppo":
            # Both factories accept (scope, start_at_intermediate_goal).
            raw = DummyVecEnv([lambda: suite.make_stage_env(2, False)])
            try:
                vec = normalize_once(raw, checkpoint=args.normalization, training=False)
            except Exception:
                raw.close()
                raise
            policy = PPO.load(args.checkpoint, env=vec, device="cpu")
            normalizer = vec
        else:
            policy = ControllerPolicy()
            normalizer = IdentityNormalization()
        result = suite.final_acceptance(
            policy, normalizer, episodes=args.episodes, seed_start=args.seed_start,
        )
        records = result.pop("episode_records")
        write_json_new(output / "episodes.json", records)
        write_json_new(output / "summary.json", {
            "schema_version": PROTOCOL_VERSION, "status": "complete",
            "config": config, "metrics": result,
            "limitations": [
                "Fixed scenes; disjoint episode seeds are not independent training seeds.",
                "No scene/domain-shift generalization claim.",
                "Non-success terminations include more than falls.",
                "max_progress follows the environment metric, not traveled path length.",
            ],
        })
        print(json.dumps({"output": str(output), "metrics": result}, indent=2))
    finally:
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    main()

# Unitree Go2W Hierarchical RL Navigation (MoRA-inspired)

Hierarchical navigation for the Unitree Go2W in MuJoCo: reduce the learning problem to high-level PPO commands over a scripted controller, and evaluate curve tracking, multi-stage docking and junction routing. **Simulation-only.**

`PPO` `Behavior Cloning` `DAgger` `Curriculum Learning` `Ablation Studies`

**Stack:** Python · PyTorch · MuJoCo · Gymnasium · Stable-Baselines3

[中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Experiments](docs/EXPERIMENTS.md) · [Reproduction guide](docs/USAGE.md) · [Validation record](docs/PORTFOLIO_VALIDATION.md) · [MIT](LICENSE)

## Experiment integrity update (2026-10-06)

The new `navigation-integrity-v1` path fixes wrapper seeding, double normalization
between curriculum stages, per-episode metric counting and accidental historical
report overwrites. Run configurations record source revision, installed versions,
input hashes and evaluation seeds. **Archived results below have not been rerun by
this update.** See [changes, commands and remaining issues](docs/INTEGRITY_UPGRADE.md).

After installing the dependencies, evaluate the controller-only baseline without
training or downloading a policy:

```bash
python scripts/evaluate_navigation.py --task multi-segment --method controller --episodes 20
python scripts/evaluate_navigation.py --task junction --method controller --episodes 40
```

Each command creates a new run directory containing `run_config.json`,
`episodes.json` and `summary.json`. Use `--method ppo --checkpoint ...
--normalization ...` only with a matching trusted checkpoint pair. Missing files
are errors, not a reason to substitute BC weights or invented results. The tasks
still use fixed scenes: disjoint episode seeds **do not establish generalization**.

## Demo

Archived trained policies running in MuJoCo. The quick-start teacher trajectories are a different artifact, and no real-robot footage is claimed.

| Curve navigation | Multi-stage A→B | Junction A/B routing |
| --- | --- | --- |
| ![Curve policy](media/rl_traverse_curve_high_level_seed01.gif) | ![Multi-stage policy](media/rl_traverse_curve_multi_segment_seed00_v2.gif) | ![Junction policy](media/rl_traverse_curve_junction_seed00.gif) |

## Key Results (archived)

These describe the complete rule/controller/policy system, **not PPO alone**; episode-seed repetition is not equivalent to independent scenes or training seeds.

| Task | Recorded result | Evidence |
| --- | --- | --- |
| Curve navigation | **20/20 success**, mean distance 1.355 m, **0 falls** | [seed01 report](reports/traverse_curve_high_level/seed01/report.md) |
| 10 m-class multi-stage A→B | **20/20 success**, A-stop 100%, mean distance 10.73 m, 15.2 s, **0 falls** | [v2 report](reports/traverse_curve_multi_segment/seed00_v2/report.md), [protocol](docs/USAGE.md) |
| Junction A/B routing | **40/40 success**, correct branch 100%, mean time 7.0 s, **0 falls** | [junction report](reports/traverse_curve_junction/seed00/report.md) |
| Quality check (2026-09-12) | **196 passed / 1 skipped** (197 discovered); Pyright 0 errors | [scope and environment](docs/PORTFOLIO_VALIDATION.md) |

Final PPO weights and full training logs are **not included** in this snapshot; reproduction requires retraining. See [Final Results](#final-results) and [Limitations](#limitations) for the evidence boundaries.

## Why this design (Problem)

Directly learning six low-level actions from a 61-dimensional observation produced stationary or short-range policies in the recorded curve experiments. The research question is whether explicit task state and a smaller navigation action space can support the repository's longer, staged tasks.

## Architecture

![System 2 rule-based branch selection feeds System 1 high-level PPO, which commands the System 0 scripted controller that drives the MuJoCo Go2W](docs/images/hierarchy.svg)

- [`Go2wEnv`](rl/go2w_env.py): simulation, rewards, progress, docking and termination.
- [`HighLevelEnvWrapper`](rl/high_level_env_wrapper.py): goal/progress observations, two-dimensional policy interface and rule-based branch locking.
- [`LowLevelController`](rl/low_level_controller.py): known-path tracking and scripted braking; junction steering uses the teacher branch geometry and currently **ignores `turn_adjust`**.

MoRA-inspired denotes an independent mapping of a layered idea, **not an official reproduction**. System 2 is a rule module; no VLM/VLA or learned semantic planner is implemented. Layer responsibilities and observation/action contracts are detailed in [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Algorithms

| Method | Implemented role | Entry point |
| --- | --- | --- |
| PPO | Low-level baselines and high-level navigation learning | [baseline](rl/train.py), [high-level curve](scripts/train_high_level_curve.py) |
| BC | Scripted-teacher supervision; multi-stage/junction policy warm-start | [curve baseline](scripts/bc_pretrain.py), [multi-stage](scripts/bc_pretrain_multi_segment.py), [junction](scripts/bc_pretrain_junction.py) |
| DAgger | Iterative teacher corrections for the low-level curve policy; recorded as unsuccessful | [training](scripts/dagger_train.py) |
| Curriculum | Staged training, BC initialization and complete-task evaluation | [multi-stage](scripts/train_multi_segment_curriculum.py), [junction](scripts/train_junction_curriculum.py) |

## Experimental Findings

| Failure → diagnosis | Iteration / observation | Interpretation boundary |
| --- | --- | --- |
| Low-level PPO stagnates or travels ≤0.7 m | Separate scripted tracking from two-dimensional PPO commands | Architecture, state and exploration change together; no isolated causal proof |
| High-level seed00 parks at speed lower bound 0.5 | Raise normal-navigation lower bound to 0.9; seed01 succeeds | Needs matched multi-seed controls |
| First multi-stage run overshoots B; complete success 0% | Strengthen docking reward, penalize overshoot, terminate at B and add stage-specific BC; v2 succeeds | Contributions are bundled, not individually attributed |
| Junction curriculum starts stage2 at an artificial stationary/yaw-zero state; 0% | Complete-start stage3 succeeds | State-distribution mismatch is a hypothesis requiring a matched-start control |

Curriculum progression figures: [multi-stage curriculum](docs/images/multi_segment_curriculum.png) · [junction curriculum](docs/images/junction_curriculum.png). See [negative results and evidence caveats](docs/EXPERIMENTS.md), including discrepancies in the historical B+ HER narrative.

## Ablation Studies

- **BC vs random initialization:** both reach 0% success after 0.5 M steps. Training seeds and learning rates differ; this is an exploratory comparison, not a controlled estimate of BC's independent benefit. [Report](data/demo_trajectories/bc_ablation_report.md) · [curve ablation figure](docs/images/curve_high_level_ablation.png).
- **DAgger:** five correction rounds; best recorded distance 0.915 m, still 0% success. Correction quality and near-fall labels remain concerns. [Report](data/demo_trajectories/dagger_training_report.md).
- **Goal conditioning:** B+ random-goal evaluation records 90% and two falls; no-goal comparison records 100% under different goal conditions. This does not establish that goals are unnecessary in harder tasks. [B+](reports/traverse_curve_high_level_bplus/seed00/report.md), [no-goal](reports/traverse_curve_high_level_bplus_no_goal/seed00/report.md).

## Final Results

The Key Results table summarizes the archived successful runs. Episode-seed repetition is not equivalent to independent scenes or independent training seeds. Multi-stage/junction domain randomization is disabled, evaluation seeds overlap with stage evaluation, and distances use each report's metric definition. No real-robot or open-world claim is made.

## Reproduction

### Quick Start

Use Python 3.10 or 3.12 on **Linux / WSL2's Linux filesystem**. Third-party model files include case-colliding filenames; common macOS/Windows filesystems cannot faithfully check out both.

```bash
git clone https://github.com/Anhao1314/go2w-MoRA-navigation.git
cd go2w-MoRA-navigation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python - <<'PYTHON'
from rl.go2w_env import Go2wEnv
env = Go2wEnv(task="traverse_curve", domain_randomize=False)
obs, _ = env.reset(seed=0)
print(obs.shape, env.action_space.shape)  # (61,), (6,)
env.close()
PYTHON
```

### Training and evaluation

The command below uses the current training entry point. The archived seed01 report describes 61-dimensional observations; current goal-conditioning defaults may add fields. Compare code/configuration with the report before attempting a matched historical experiment; do not treat the current command as a byte-for-byte replay.

```bash
python scripts/train_high_level_curve.py \
  --total-steps 2000000 --envs 4 --seed 1 \
  --learning-rate 3e-4 --ent-coef 0.01 \
  --run-dir rl/runs/traverse_curve_high_level/seed01
```

After training has created the matching model and normalization files:

```bash
python scripts/train_high_level_curve.py \
  --report-only --run-dir rl/runs/traverse_curve_high_level/seed01
```

For updated curriculum output locations and recording, [the integrity guide](docs/INTEGRITY_UPGRADE.md) takes precedence over historical paths. For multi-stage/junction teacher generation, BC warm-start, staged training, smoke runs and video recording, follow the [full guide](docs/USAGE.md). Teacher generation and reporting may overwrite same-named samples/reports: use an isolated clone or back up historical artifacts. Dependencies partly use version ranges; save the exact environment and run configuration for each experiment. Historical results are reference observations, not retraining guarantees.

## Project Structure

| Path | Purpose |
| --- | --- |
| `rl/` | Environments, controllers, training, evaluation and local tools |
| `scripts/` | Teacher data, BC, DAgger, curricula and recording |
| `reports/` | Selected archived evaluations |
| `data/demo_trajectories/` | Selected teacher data, BC models, normalization and research reports |
| `tests/` | Environment, controller and tooling checks |
| `media/`, `models/go2w/` | Recorded demos and third-party simulation assets |
| `docs/` | Architecture, experiments, usage and attribution |

## Tests & Quality

```bash
python -m unittest discover -s tests -p 'test_*.py'
pyright rl scripts mujoco_demos
```

[CI](.github/workflows/ci.yml) installs dependencies, runs the suite and checks these source directories on Linux. The workflow definition is not a claim that CI has passed; inspect the actual Actions run. For headless rendering, CI uses EGL and installs Mesa. Current counts belong to the actual run, not a permanent badge.

## Limitations

Known path geometry and simulation pose are required. Branch decisions and parts of docking are scripted. Successful PPO weights are absent; BC artifacts are not substitutes. Scene diversity, independent held-out tests, multi-training-seed variability, real sensors and sim-to-real remain unverified.

## Roadmap

- Publish final PPO checkpoints with normalization, configuration and artifact hashes.
- Add matched controller-only baselines and independent held-out scene/seed evaluation.
- Test curriculum start-state matching and isolate docking/BC contributions.
- Evaluate perception and real-robot interfaces before making deployment claims.

## Contributors

- **Anhao1314** — project owner; original system, experiments, robotics/RL implementation and archived evidence.
- **ChatGPT (OpenAI)** — AI collaborator on repository audit, the [experiment-integrity refactor](https://github.com/Anhao1314/go2w-MoRA-navigation/pull/1), reproducible evaluation/test design and documentation. This credit does not imply a separate GitHub identity or independent authorship.

## License

Original code: [MIT](LICENSE). Go2W model: [model license](models/go2w/LICENSE). See [third-party notices](docs/THIRD_PARTY_NOTICES.md).

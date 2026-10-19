# MoRA-Train: A Hierarchical Training System for Embodied Agents

### First adapter: Unitree Go2W wheeled quadruped long-horizon navigation

*An open, reproducible experiment bed for hierarchical embodied training.*

- **Document type:** System engineering and research record (not peer-reviewed)
- **Project:** MoRA-Train (a MoRA-inspired hierarchical embodied training system)
- **Role of this repository:** First task adapter / experiment bed + C++ S0 control core
- **Training side:** Windows (all Go2W training and evaluation)
- **Quantification side:** RL-Sentinel (macOS, time-correct training reliability analysis)
- **Scope:** Simulation only; no real-robot deployment; not a reproduction of official MoRA

[中文版](README.md) · [Architecture and interfaces](docs/ARCHITECTURE.md) · [Experiments](docs/EXPERIMENTS.md) · [Reproduction](docs/USAGE.md) · [Integrity upgrade](docs/INTEGRITY_UPGRADE.md)

## 0. Project positioning

MoRA-Train is not primarily about making one robot dog complete one path. It asks:

> How can we train an embodied agent composed of S2 planning, S1 goal-conditioned execution (multi-granularity memory + progress-aware closed loop), and S0 verifiable control, and train, evaluate, and quantify it across tasks rather than tuning for a single platform?

Go2W is the **first task adapter and experiment bed** for MoRA-Train. It tests:

- whether the hierarchy composes: S2 rule decisions → S1 learned policy → S0 C++ control;
- whether S0 is numerically trustworthy: FK/IK/Jacobian, centroidal dynamics, and MPC against MuJoCo;
- whether S1 can complete long-horizon multi-segment, junction, and goal tasks;
- whether training can be quantified time-correctly by RL-Sentinel.

| Layer | Responsibility | Current state in this repository |
| --- | --- | --- |
| S2 | Task decomposition, branch/skill selection | Rule-based inside the Go2W adapter |
| S1 | Goal-conditioned policy | PPO / BC / DAgger / curriculum learning |
| Memory | Short-term / episodic / task memory | `mora/` skeleton implemented; neural Memory Token pending |
| Progress | Progress estimation, stall detection, replanning | `mora/` heuristic implemented; learned progress head pending |
| S0 | Motion control, MPC, safety state machine | P0 delivered and numerically validated |
| Observability | Event stream, as-of early-stop analysis | `EventWriter` + SB3 callback implemented; RL-Sentinel ingester pending |

See [MoRA-Train system layer](docs/MORA_TRAIN.md) for the interfaces and current capabilities. Results in this repository therefore establish feasibility of the **first adapter**, not the whole MoRA-Train system, and not a reproduction of official MoRA.

## Abstract

**Background.** MoRA-Train needs a reproducible embodied task to validate its hierarchical training interfaces. The Go2W wheeled quadruped provides long-horizon navigation, junction decisions, and low-level control validation in known-path simulation. **Objective.** Deliver the first adapter and test whether a rule-based S2 + learned S1 + verifiable S0 composition can complete the specified tasks. **Methods.** Combine rule-based goal/branch selection, PPO high-level commands, and scripted tracking / C++ S0 execution; record PPO, BC, DAgger, curriculum learning, and exploratory comparisons. **Results.** Under their respective archived fixed simulation protocols, curve, A→B, and junction tasks recorded **20/20**, **20/20**, and **40/40** successes; C++ S0 and scripted S0 match step by step on multi-segment, with **10/10** acceptance. **Conclusion.** These results support engineering feasibility of the first adapter, but do not isolate PPO/BC benefits, nor validate MoRA-Train's memory/progress modules, unseen-scene generalization, or real-robot reliability.

**Keywords:** MoRA-inspired; hierarchical embodied agents; robot navigation; reinforcement learning; PPO; MuJoCo; training reliability

## 1. First adapter: research questions

These RQs are at the **Go2W adapter level**, not the full set of MoRA-Train research questions; system-level memory, progress-loop, and cross-task questions are addressed in later iterations.

Curved and multi-stage navigation couples stability, path following, speed regulation, and arrival/stop constraints. Learning all of these directly as low-level actions may produce stationary or ineffective policies. The initial recorded formulation used **61-dimensional observations and six-dimensional actions**, with unsuccessful curve-navigation attempts ([archived experiments](docs/EXPERIMENTS.md)).

This work does **not** introduce a new PPO algorithm. It investigates an **engineering hypothesis**: delegate known geometric tracking and braking to a deterministic controller, restrict the learned interface to a few high-level navigation commands, and provide explicit goal and task-progress context.

The study asks:

- **RQ1:** Can a complete rule + PPO + controller hierarchy carry out the specified curve, intermediate-stop, and junction tasks?
- **RQ2:** What do stationary policies, target overshoot, and failed curriculum stages suggest about action interfaces, rewards, and state distributions?
- **RQ3:** Do the existing comparisons independently identify the contributions of PPO, BC, DAgger, and scripted control?

**Scope and naming:** “MoRA-inspired” means an independent engineering interpretation of layered control, **not an official MoRA model or training reproduction**. System 2 currently consists of deterministic rules, not learned VLM/VLA planning or open-world semantic navigation. No real Go2W deployment was evaluated.

## 2. Go2W adapter methods

The methods below are specific to the Go2W adapter, not the complete design of the general MoRA-Train system; the system-level task API, memory, and progress modules are abstracted in later iterations.

### 2.1 Simulation environment and tasks

The [`Go2wEnv` environment](rl/go2w_env.py) integrates MuJoCo simulation with Gymnasium observations, actions, rewards, termination, and task progress. The primary tasks are curved-path traversal, multi-segment A→B navigation with an intermediate stop at A, and A/B junction routing under a known target.

The physics step is **0.002 s (500 Hz)** and the policy is updated once every five physics steps (**100 Hz**). These are current simulator settings, **not** measurements of the physical robot's SDK frequency.

The underlying traversal action has six dimensions: **four wheel torque-related commands plus two grouped leg-pose commands**. Observations include orientation, angular/wheel/leg state, prior actions, and navigation features. Source: [environment implementation](rl/go2w_env.py).

### 2.2 Hierarchical system and information flow

```text
Known goal / task state
   ↓
System 2: rule-based goal and branch selection (A→left, B→right)
   ↓
System 1: PPO MLP → [speed_scale, turn_adjust]
   ↓
System 0: lookahead path tracking, differential steering,
          scripted docking and leg PD control
   ↓
MuJoCo: execute → observe state, reward, termination
   └──────────────────→ next decision
```

| Layer | Current implementation | Responsibility | Source |
| --- | --- | --- | --- |
| System 2 | Deterministic target-to-branch mapping | Lock left/right branch within a decision region | [High-level wrapper](rl/high_level_env_wrapper.py) |
| System 1 | Stable-Baselines3 PPO, MLP `[256, 256]` | Produce two-dimensional navigation commands | [High-level PPO](scripts/train_high_level_curve.py) |
| System 0 | Known-path lookahead, differential steering, scripted braking/PD | Convert navigation commands into executable controls | [Controller](rl/low_level_controller.py), [environment](rl/go2w_env.py) |

**Observation interface.** The base traversal observation is **61D**, extended to **63D** with goal context, **67D** for multi-stage tasks, or **68D** for junction state. Added fields express goal distance/direction, task phase, or branch state.

**Action interface.** The high-level action `[speed_scale, turn_adjust]` has current bounds **[0.9, 1.5]** and **[-0.5, 0.5]**. A low-level controller maps this to six-dimensional execution controls. Normal navigation's minimum speed multiplier is distinct from scripted braking or zero action.

**Crucial implementation boundary.** Junction branch choice is **rule-based**, and steering uses scripted teacher geometry. In the current junction path, the policy's `turn_adjust` is **not used**; the learned high-level output effectively adjusts speed only. Parts of multi-stage docking are scripted. Learned PPO should not be credited with all routing, steering, or stopping behavior.

### 2.3 Policy learning and feedback design

| Approach | Role in this repository | Implementation |
| --- | --- | --- |
| PPO [1] | Initial low-level baseline and subsequent high-level learning | [Low-level training](rl/train.py), [high-level curve](scripts/train_high_level_curve.py) |
| BC | Supervised learning from scripted-teacher trajectories; warm starts for selected curriculum stages | [BC](scripts/bc_pretrain.py), [multi-stage BC](scripts/bc_pretrain_multi_segment.py) |
| DAgger [2] | Collect teacher corrections for error states visited by the current learner, then retrain | [DAgger](scripts/dagger_train.py) |
| Curriculum | Train reach-A, start-at-A to B, then full A→B tasks in stages | [Multi-stage curriculum](scripts/train_multi_segment_curriculum.py), [junction curriculum](scripts/train_junction_curriculum.py) |

Rewards vary by task and code version. They include progress, heading and lateral deviation, stability, intermediate goals, and failure terms. **Different archived experiments do not necessarily use the same reward function.** See [`_get_reward`](rl/go2w_env.py) and the original run configurations.

Intermediate **A-stop** acceptance requires roughly 0.15 m tolerance, speed below 0.2 m/s, and a 0.3 s hold. Reaching **B** currently ends the episode within endpoint tolerance and is **not an independent demonstration of the same sustained hold condition**.

### 2.4 Evaluation evidence

Archived metrics include task success, reported progress/distance, elapsed time, falls, A-stop compliance, and junction branch correctness. Results below come from committed reports under `reports/` and `data/demo_trajectories/`; **this documentation change did not rerun any training**.

Varying evaluation episode seeds, independently retraining with new random seeds, and testing novel scenes are distinct levels of validation. Most positive archived claims establish only the first.

<a id="final-results"></a>
## 3. Go2W adapter archived results

| Experiment | Recorded result | Interpretation | Evidence |
| --- | --- | --- | --- |
| Early low-level PPO | Failed to meet objective | Stationary / short travel (≤ ~0.7 m in recorded experiments) | [Overview](docs/EXPERIMENTS.md) |
| Hierarchical curve `seed00` | **0/20** | Policy selected the 0.5 speed lower bound and stalled | [Failure report](reports/traverse_curve_high_level/seed00/report.md) |
| Hierarchical curve `seed01` | **20/20** | Mean reported progress 1.355 m, 0 falls | [Curve report](reports/traverse_curve_high_level/seed01/report.md) |
| Multi-stage A→B initial | **0%** | Overshot goal; full task acceptance failed | [Initial report](reports/traverse_curve_multi_segment/seed00/report.md) |
| Multi-stage A→B `v2` | **20/20** | A-stop 100%, mean 15.2 s, 0 falls | [v2 report](reports/traverse_curve_multi_segment/seed00_v2/report.md) |
| A/B junction | **40/40** | Rule-selected correct branch 100%, mean 7.0 s, 0 falls | [Junction report](reports/traverse_curve_junction/seed00/report.md) |
| Random-goal B+ | **18/20** | 2 falls in that goal-sampling protocol | [B+ report](reports/traverse_curve_high_level_bplus/seed00/report.md) |

**Measurement warning:** Some archived “distance” metrics are progress-based rather than uniformly measured physical path lengths; a documented curve arc-length calculation needs a separate versioned correction. Cross-version distance comparisons should therefore be avoided ([integrity note](docs/INTEGRITY_UPGRADE.md)).

Supplementary simulation recordings: [curve](media/rl_traverse_curve_high_level_seed01.mp4) · [A→B](media/rl_traverse_curve_multi_segment_seed00_v2.mp4) · [junction](media/rl_traverse_curve_junction_seed00.mp4). These are **MuJoCo videos, not real-robot demonstrations**.

## 4. Negative results, controls, and discussion

### 4.1 What did BC and DAgger contribute?

The archived [BC versus random-initialization comparison](data/demo_trajectories/bc_ablation_report.md) gave **0% task success for both methods at 0.5M steps**. Training seeds and learning rates differed, so this is exploratory evidence, **not a controlled estimate** of BC's independent effect or a conclusion that BC is generally ineffective.

In [DAgger experiments](data/demo_trajectories/dagger_training_report.md), five correction rounds did not achieve the target task; the best reported progress was ~**0.915 m**, still with **0% success**. Inadequate near-fall teacher recovery actions and distribution shift remain plausible, unverified explanations.

### 4.2 What does hierarchical success establish?

Early single-level PPO failed on the archived curve task while later hierarchical `seed01` succeeded. This establishes a successful **complete configuration**, not a single-variable causal effect: architecture, action constraints, and learning conditions were not fully matched across independent training seeds.

The first multi-stage version failed; v2 recorded 20/20 after **multiple simultaneous changes** to docking reward, overshoot penalties, termination, and stage-specific BC. Their isolated contributions remain unmeasured.

### 4.3 A necessary strong baseline

An earlier [scripted-only multi-stage controller validation](data/demo_trajectories/multi_segment_validation_report.md) also achieved **20/20** under its own protocol. The final system necessarily benefits from established geometric tracking and stopping logic; attributing all final success to PPO would be incorrect. However, these are not a properly paired comparison, so the independent benefit (or lack thereof) from PPO remains unresolved.

### 4.4 Historical HER terminology

A B+ historical report mentions HER in prose although its configuration states **HER was disabled**. Separately, the implemented future-goal relabeling plus auxiliary actor MSE is **not standard off-policy HER**. The project does not claim validated standard HER gains based on those records ([experimental ledger](docs/EXPERIMENTS.md); [integrity note](docs/INTEGRITY_UPGRADE.md)).

## 5. Threats to validity and limitations

<a id="limitations"></a>

1. **Internal validity:** Reward terms, policy constraints, architecture, and curriculum initialization were sometimes changed together; matched single-factor ablations are incomplete.
2. **Statistical validity:** 20/20 or 40/40 applies only to its archived evaluation batch; episode seeds are not independent training seeds, and some historical model-selection/final evaluation seeds overlap.
3. **External validity:** Path geometry and privileged simulation pose are available; domain randomization was off in multi-stage/junction evaluations. Held-out terrains, real sensors, and sim-to-real have not been demonstrated.
4. **Measurement validity:** Progress metrics may not represent actual traveled path length; A docking and B arrival have different criteria.
5. **Reproducibility:** Final successful PPO checkpoints, matching normalization statistics, and complete training logs are not contained in the public snapshot. Committed code, selected data, videos, and reports are **not a byte-for-byte trained-policy replay** without retraining.
6. **Capability boundary:** Route choices, teacher steering, and portions of braking are scripted; no learned System 2, open-world vision navigation, or certified real-robot safety is established.
7. **System-level boundary:** This repository is the first adapter of MoRA-Train; multi-granularity memory, progress-aware closed loops, and cross-task adapters are not implemented yet, so these results are not evidence for the whole training system.

## 6. Reproduction and experiment integrity

Python 3.10 or 3.12 is recommended on **Linux or WSL2's Linux filesystem**. Third-party model assets have case-colliding filenames; common case-insensitive Windows/macOS filesystems cannot faithfully represent all of them. See the [full setup guide](docs/USAGE.md).

```bash
git clone https://github.com/Anhao1314/Go2w-MoRA-navigation.git
cd Go2w-MoRA-navigation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python - <<'PY'
from rl.go2w_env import Go2wEnv
env = Go2wEnv(task="traverse_curve", domain_randomize=False)
obs, _ = env.reset(seed=0)
print(obs.shape, env.action_space.shape)  # (61,), (6,)
env.close()
PY
```

**Current controller-only runs (no policy weights or training required):**

```bash
python scripts/evaluate_navigation.py --task multi-segment --method controller --episodes 20
python scripts/evaluate_navigation.py --task junction --method controller --episodes 40
```

Each run creates its own `run_config.json`, `episodes.json`, and `summary.json`. The configuration records source revision, dependency versions, input hashes, and seeds. An evaluation's `complete` flag means execution finished, **not that task success is established**. PPO evaluation requires a trusted, matched checkpoint and `VecNormalize` file; BC weights are not a valid substitute.

```bash
python -m unittest discover -s tests -p 'test_*.py'
pyright rl scripts mujoco_demos
```

The archived [2026-09-12 validation](docs/PORTFOLIO_VALIDATION.md) reported **196 passed / 1 skipped** and zero Pyright errors in its specific environment and revision. These are **not newly rerun CI results**. The [2026-10-06 `navigation-integrity-v1` work](docs/INTEGRITY_UPGRADE.md) improved seed initialization, curriculum normalization, per-episode counting, and history-preserving output paths. **It did not retrain or independently validate the archived successful PPO runs.**

Older scripts can overwrite named historical artifacts. Use an isolated checkout for training and read the [reproduction guide](docs/USAGE.md) for teacher-data generation, BC, curriculum, and recording.

## 7. Conclusion and future experiments

This repository currently delivers the **first adapter** of MoRA-Train: an inspectable Go2W hierarchical-navigation experiment bed plus an independently testable C++ S0 control core. Its value lies in **hierarchy interfaces, control/learning division, experiment diagnosis, and reproducible records**, not a new RL algorithm or validation of the whole MoRA-Train system.

Next work proceeds at the system level:

1. implement a task-adapter API outside Go2W (`reset / step / goal / progress / success`);
2. add short-term/episodic memory and a progress head to S1, with memory/progress ablations;
3. connect the training event stream to RL-Sentinel for cross-task, multi-seed early-stop reliability;
4. add multi-seed, domain-randomization, single-factor ablations, and matched checkpoint/normalization manifests;
5. complete the ROS2 P1 vertical slice before discussing real-robot boundaries.

## 8. S0 component: C++ / ROS2 stack (P0)

Within MoRA-Train, this section is the **S0 component plus control-layer validation for the first adapter**, not the whole training system. A **ROS-agnostic C++17 control core** now sits beneath the existing Python/MuJoCo stack as an independently testable System 0. P0 delivers the C++ core plus a Python adapter; the ROS2 vertical slice (P1) is not implemented yet.

- Floating-base FK/IK and wheel Jacobians (1e-9 / 1e-6 vs MuJoCo fixtures)
- Centroidal mass/COM/inertia (1e-9 / 1e-8)
- SRBD linear MPC (OSQP, N=20, 50 Hz)
- Contact-force → wheel-torque mapping and a safety state machine
- Scripted teacher with optional A/B pre-braking
- pybind11 injection into the existing `HighLevelEnvWrapper`

**P0 evidence (fixed seeds 1000–1009):**

| Configuration | Success | mean_x | mean_time |
| --- | ---: | ---: | ---: |
| scripted, no pre-brake (historical baseline) | 0/10 | 4.7811 | 6.292 s |
| scripted, pre-brake | 10/10 | 10.6571 | 13.990 s |
| **C++ S0, pre-brake** | **10/10** | **10.6571** | **13.990 s** |

The C++ and scripted pre-brake runs match item by item, and the seed-1000 trajectory is identical step by step. MPC benchmark: 200/200 solves, mean 3.16 ms, p99 4.07 ms, hover vertical-force error 2.15%. Demo video: [cpp_stack_p0_demo.mp4](media/cpp_stack_p0_demo.mp4). Native Windows (MSVC 2026) validation: Python suite **225 tests OK (4 skipped)**, bindings 3/3, MPC 200/200 (mean 3.88 ms, p99 4.59 ms), multi-segment **10/10** (mean_x 10.6563, 14.000 s). See [Windows validation](docs/WINDOWS_NATIVE.md).

**Scope limits:** simulation only; no ROS2 nodes, EKF, full WBC/RNEA, vision/tactile, real robot, or sim-to-real; MPC is not yet closed-loop integrated with `Go2wEnv`; historical success rates are not retroactively claimed. See [C++ architecture](docs/CPP_ROS2_ARCHITECTURE.md) · [Control design](docs/CONTROL_DESIGN.md) · [Calibration](docs/CALIBRATION.md) · [Validation](docs/VALIDATION.md) · [P0 protocol](docs/CPP_STACK_PROTOCOL.md) · [cpp/README](cpp/README.md).

## References

[1] Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

[2] Ross, S., Gordon, G., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS / PMLR 15, 627–635. [Paper](https://proceedings.mlr.press/v15/ross11a.html).

[3] Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS. [DOI](https://doi.org/10.1109/IROS.2012.6386109).

[4] Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268). [Paper](https://www.jmlr.org/papers/v22/20-1364.html).

**Primary project evidence:** [Architecture](docs/ARCHITECTURE.md) · [Experiments including failures](docs/EXPERIMENTS.md) · [Reproduction](docs/USAGE.md) · [Integrity and open issues](docs/INTEGRITY_UPGRADE.md)

## Contributions and licensing

**Anhao1314:** Project owner; original robotics/RL system, experiments, and archived evidence. **ChatGPT (OpenAI):** AI collaboration on repository review, the [experiment-integrity refactor](https://github.com/Anhao1314/Go2w-MoRA-navigation/pull/1), validation design, and documentation; this credit does not imply an independent human author or peer review. Original code is [MIT licensed](LICENSE); Go2W assets and other third-party materials retain their own [model license](models/go2w/LICENSE) and [third-party notices](docs/THIRD_PARTY_NOTICES.md).

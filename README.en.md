# Hierarchical Reinforcement Learning Navigation for Unitree Go2W: Methods, Experiments, and Validity Analysis

*An evidence-based engineering research report for known-path simulation tasks.*

**Document type:** Open-source engineering research report (not peer-reviewed)  
**Platform:** Unitree Go2W · MuJoCo · PPO / BC / DAgger  
**Scope:** Bounded simulation studies; no real-robot deployment claim

[中文版](README.md) · [Architecture and interfaces](docs/ARCHITECTURE.md) · [Experiments](docs/EXPERIMENTS.md) · [Reproduction](docs/USAGE.md) · [Integrity upgrade](docs/INTEGRITY_UPGRADE.md)

## Abstract

**Background.** Archived Go2W curve-navigation experiments with a 61-dimensional observation and six low-level PPO actions exhibited stationary behavior, limited progress, or failure. **Objective.** Investigate whether a hierarchy that reduces the learned control interface can support selected curve, multi-stage, and junction tasks under known-path simulation conditions. **Methods.** Combine System 2 rule-based goal/branch selection, System 1 PPO high-level commands, and System 0 scripted tracking and PD control; study behavior cloning (BC), DAgger, curriculum learning, and exploratory comparisons. **Results.** Under their respective archived fixed simulation protocols, the **complete systems** recorded **20/20** curve, **20/20** A→B, and **40/40** junction successes. Failed early low-level PPO, BC/DAgger attempts, and the first multi-stage version remain documented. **Conclusion.** These observations support bounded feasibility of the complete architecture, but do not isolate a causal benefit from PPO, BC, or any individual change. They do not demonstrate unseen-scene generalization or reliable hardware deployment.

**Keywords:** hierarchical control; robot navigation; reinforcement learning; PPO; imitation learning; MuJoCo; reproducibility

## 1. Introduction and research questions

Curved and multi-stage navigation couples stability, path following, speed regulation, and arrival/stop constraints. Learning all of these directly as low-level actions may produce stationary or ineffective policies. The initial recorded formulation used **61-dimensional observations and six-dimensional actions**, with unsuccessful curve-navigation attempts ([archived experiments](docs/EXPERIMENTS.md)).

This work does **not** introduce a new PPO algorithm. It investigates an **engineering hypothesis**: delegate known geometric tracking and braking to a deterministic controller, restrict the learned interface to a few high-level navigation commands, and provide explicit goal and task-progress context.

The study asks:

- **RQ1:** Can a complete rule + PPO + controller hierarchy carry out the specified curve, intermediate-stop, and junction tasks?
- **RQ2:** What do stationary policies, target overshoot, and failed curriculum stages suggest about action interfaces, rewards, and state distributions?
- **RQ3:** Do the existing comparisons independently identify the contributions of PPO, BC, DAgger, and scripted control?

**Scope and naming:** “MoRA-inspired” means an independent engineering interpretation of layered control, **not an official MoRA model or training reproduction**. System 2 currently consists of deterministic rules, not learned VLM/VLA planning or open-world semantic navigation. No real Go2W deployment was evaluated.

## 2. Methods

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
## 3. Archived experimental results

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

This repository documents a bounded hierarchical-navigation engineering study: initial low-level PPO failures, a revised division of control responsibilities, and archived success and failure experiments on specified simulated tasks. Its value lies in **inspectable system design and evidence-based failure analysis**, not a newly invented RL algorithm or proven hardware generalization.

Next work should prioritize matched multi-training-seed comparisons against controller-only baselines, held-out tasks and domain randomization, isolated BC/docking ablations, publication of policy-normalization-checkpoint pairs with exact manifests, and only then studies of perception, latency, safety, and real-robot deployment.

## References

[1] Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

[2] Ross, S., Gordon, G., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS / PMLR 15, 627–635. [Paper](https://proceedings.mlr.press/v15/ross11a.html).

[3] Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS. [DOI](https://doi.org/10.1109/IROS.2012.6386109).

[4] Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268). [Paper](https://www.jmlr.org/papers/v22/20-1364.html).

**Primary project evidence:** [Architecture](docs/ARCHITECTURE.md) · [Experiments including failures](docs/EXPERIMENTS.md) · [Reproduction](docs/USAGE.md) · [Integrity and open issues](docs/INTEGRITY_UPGRADE.md)

## Contributions and licensing

**Anhao1314:** Project owner; original robotics/RL system, experiments, and archived evidence. **ChatGPT (OpenAI):** AI collaboration on repository review, the [experiment-integrity refactor](https://github.com/Anhao1314/Go2w-MoRA-navigation/pull/1), validation design, and documentation; this credit does not imply an independent human author or peer review. Original code is [MIT licensed](LICENSE); Go2W assets and other third-party materials retain their own [model license](models/go2w/LICENSE) and [third-party notices](docs/THIRD_PARTY_NOTICES.md).

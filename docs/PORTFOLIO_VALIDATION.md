# Portfolio validation — 2026-09-12

Source baseline: `98faf7d72997cf189988aefcb5064238646777b5`, followed by the Portfolio documentation/CI update. Local environment: macOS, Python 3.12.14, MuJoCo 3.11.0, Stable-Baselines3 2.9.0.

- `python -m unittest discover -s tests -p 'test_*.py'`: 197 discovered; 196 passed, 1 skipped, 0 failures/errors. The skipped offline balance replay requires an uncommitted `rl/runs/balance/seed00/eval_log.csv`; no fabricated log was added.
- A subprocess ResourceWarning was emitted by the suite. A passing suite does not certify subprocess lifecycle behavior in every environment.
- Pyright 1.1.408 initially found two missing imports for optional trajectory plotting (`matplotlib`). The declared dependency was added; rerun `pyright rl scripts mujoco_demos` completed with **0 errors, 0 warnings, 0 informations** (using the project virtual environment).

The full checkout has third-party filenames differing only by case. Local macOS verification cannot establish faithful asset checkout on a case-sensitive system; the Linux CI run is a separate check. The terrain asset collision is not an intended source change and must not be committed.

Historical navigation metrics remain report-based, not newly reproduced training outcomes. Final successful PPO checkpoints and complete training logs are absent from this snapshot. Tests do not replace independent task/generalization evaluation.

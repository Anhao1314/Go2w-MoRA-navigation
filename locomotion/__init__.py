"""Phase-2 locomotion package: joint-level RL for Unitree Go2 (M1/M2).

Home of the code that executes the pre-registered experiment protocol in
``docs/EXPERIMENT_DESIGN.md``.  Subpackages:

- ``locomotion.isaac``: Isaac Lab task configurations (S1/S2/S2b).  Imports
  Isaac Lab and therefore must only be imported inside the ``.venv311``
  environment.
- ``locomotion.eval``: frozen seed set and metric implementations.  Pure
  Python; safe to import anywhere (CI included).
- ``locomotion.sim2sim``: S4 Isaac -> MuJoCo replication (design stage).
- ``locomotion.baselines``: C0 scripted PD and the C1 convex-MPC slot.

This top-level ``__init__`` deliberately imports nothing so that
``locomotion.eval`` stays usable without Isaac Sim installed.
"""

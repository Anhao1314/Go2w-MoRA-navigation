"""Frozen evaluation metrics for the M1/M2 locomotion protocol.

Implements the metric definitions of ``docs/EXPERIMENT_DESIGN.md`` §3.2 as
pure functions over per-episode arrays so they can be unit-tested (and reused
by the Isaac-side and MuJoCo-side evaluation loops) without Isaac Sim or
MuJoCo installed.

Freeze contract: these definitions are frozen by the pre-registered protocol
(v1.0, 2026-10-22).  If a metric proves ill-defined mid-experiment, the run is
marked ``protocol_violation`` and excluded from acceptance claims -- the code
here is not silently redefined.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

TRANSIENT_SKIP_S = 0.5
"""Seconds excluded from tracking metrics after episode start (reset transient)."""

VX_RECOVERY_TOL_MPS = 0.1
"""``recovery_time`` counts as recovered once the vx error stays below this."""


def _rmse_after_transient(
    time_s: np.ndarray,
    commanded: np.ndarray,
    measured: np.ndarray,
    transient_skip_s: float,
) -> float:
    """RMSE of ``measured`` vs ``commanded`` on samples with ``time_s > transient_skip_s``."""
    time_s = np.asarray(time_s, dtype=float)
    commanded = np.asarray(commanded, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if time_s.shape != commanded.shape or time_s.shape != measured.shape:
        raise ValueError(
            "time_s, commanded and measured must share one shape, got "
            f"{time_s.shape}, {commanded.shape}, {measured.shape}"
        )
    mask = time_s > transient_skip_s
    if not np.any(mask):
        raise ValueError(
            f"no samples after the {transient_skip_s} s transient; episode too short"
        )
    error = commanded[mask] - measured[mask]
    return float(np.sqrt(np.mean(error**2)))


def tracking_rmse(
    time_s: np.ndarray,
    commanded: np.ndarray,
    measured: np.ndarray,
    *,
    transient_skip_s: float = TRANSIENT_SKIP_S,
) -> float:
    """Generic command-tracking RMSE over the full episode after the reset transient."""
    return _rmse_after_transient(time_s, commanded, measured, transient_skip_s)


def vx_rmse(
    time_s: np.ndarray,
    commanded_vx: np.ndarray,
    measured_vx: np.ndarray,
    *,
    transient_skip_s: float = TRANSIENT_SKIP_S,
) -> float:
    """Forward-velocity tracking RMSE [m/s] (protocol metric ``vx_rmse``)."""
    return _rmse_after_transient(time_s, commanded_vx, measured_vx, transient_skip_s)


def wz_rmse(
    time_s: np.ndarray,
    commanded_wz: np.ndarray,
    measured_wz: np.ndarray,
    *,
    transient_skip_s: float = TRANSIENT_SKIP_S,
) -> float:
    """Yaw-rate tracking RMSE [rad/s] (protocol metric ``omega_z_rmse``)."""
    return _rmse_after_transient(time_s, commanded_wz, measured_wz, transient_skip_s)


def fall_rate(terminated_by_fall: Sequence[bool]) -> float:
    """Fraction of episodes terminated by a base-height/contact violation."""
    flags = np.asarray(terminated_by_fall, dtype=bool)
    if flags.size == 0:
        raise ValueError("terminated_by_fall must be non-empty")
    return float(np.mean(flags))


def recovery_time(
    time_s: np.ndarray,
    commanded_vx: np.ndarray,
    measured_vx: np.ndarray,
    disturbance_start_s: float,
    *,
    tol_mps: float = VX_RECOVERY_TOL_MPS,
) -> float | None:
    """Seconds from the disturbance onset until vx error < ``tol_mps`` (S2 only).

    The protocol calls the disturbance onset "disturbance offset".  Returns
    ``None`` when the vx error never drops below ``tol_mps`` after the onset.
    Recovery is measured on the first crossing; persistence of the recovered
    state is the caller's reporting responsibility.
    """
    time_s = np.asarray(time_s, dtype=float)
    commanded_vx = np.asarray(commanded_vx, dtype=float)
    measured_vx = np.asarray(measured_vx, dtype=float)
    mask = time_s >= disturbance_start_s
    if not np.any(mask):
        raise ValueError("disturbance_start_s is beyond the end of the episode")
    error = np.abs(commanded_vx[mask] - measured_vx[mask])
    below = error < tol_mps
    if not np.any(below):
        return None
    first_idx = int(np.argmax(below))
    return float(time_s[mask][first_idx] - disturbance_start_s)


def success_rate(
    completed_without_fall: Sequence[bool],
    terminal_within_target: Sequence[bool],
) -> float:
    """Fraction of episodes with no fall AND terminal position within +/-0.5 m of target (S4 only)."""
    no_fall = np.asarray(completed_without_fall, dtype=bool)
    within = np.asarray(terminal_within_target, dtype=bool)
    if no_fall.shape != within.shape:
        raise ValueError("both inputs must describe the same episodes")
    if no_fall.size == 0:
        raise ValueError("episode lists must be non-empty")
    return float(np.mean(no_fall & within))


def energy_per_meter(energy_j: float, distance_m: float) -> float | None:
    """Mechanical energy per meter travelled [J/m].  Reported, not gated."""
    if distance_m <= 0.0:
        return None
    return float(energy_j / distance_m)

#!/usr/bin/env python3
"""M1/M2 pre-registered evaluation protocol: frozen seeds, thresholds, self-check.

This file is named by ``docs/EXPERIMENT_DESIGN.md`` §3.1 as the home of the
frozen evaluation seed set.  The seed tuple itself lives in
``locomotion/eval/seeds.py`` (single source of truth); this CLI exposes it,
mirrors the frozen acceptance thresholds of protocol §4, and provides
``--verify`` self-checks that any M1/M2 run can cite as a pre-flight gate.

Example:

    python scripts/eval_protocol.py --verify
    python scripts/eval_protocol.py --list-seeds
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locomotion.eval.metrics import (  # noqa: E402
    TRANSIENT_SKIP_S,
    VX_RECOVERY_TOL_MPS,
)
from locomotion.eval.seeds import EVAL_SEEDS, NUM_EVAL_EPISODES  # noqa: E402

PROTOCOL_VERSION = "v1.0"
PROTOCOL_FROZEN_DATE = "2026-10-22"


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Frozen acceptance thresholds (protocol §4).

    ``vx_rmse_max`` / ``wz_rmse_max`` gate S1 flat-ground tracking;
    ``s2_no_fall_min`` gates the S2 100 N / 0.5 s push recovery;
    ``s4_success_min`` gates the S4 Sim2Sim replication.  Multi-seed
    consistency and baseline deltas are reporting rules, not scalars.
    """

    vx_rmse_max: float = 0.15
    wz_rmse_max: float = 0.2
    s2_no_fall_min: float = 0.90
    s4_success_min: float = 0.90


THRESHOLDS = AcceptanceThresholds()


def verify() -> list[tuple[str, bool, str]]:
    """Run protocol self-checks; each entry is ``(check, ok, detail)``.

    The expected literals below intentionally duplicate the protocol text so
    that an accidental edit of ``THRESHOLDS``, the seed tuple or the metric
    constants is caught before a run starts, not after.
    """
    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "eval seed set is E={0..19}",
            EVAL_SEEDS == tuple(range(20)) and len(EVAL_SEEDS) == 20,
            f"EVAL_SEEDS={list(EVAL_SEEDS)}",
        )
    )
    checks.append(
        (
            "episode count matches seed count",
            NUM_EVAL_EPISODES == 20,
            f"NUM_EVAL_EPISODES={NUM_EVAL_EPISODES}",
        )
    )
    checks.append(
        (
            "thresholds match protocol §4",
            (
                THRESHOLDS.vx_rmse_max == 0.15
                and THRESHOLDS.wz_rmse_max == 0.2
                and THRESHOLDS.s2_no_fall_min == 0.90
                and THRESHOLDS.s4_success_min == 0.90
            ),
            f"{THRESHOLDS}",
        )
    )
    checks.append(
        (
            "tracking metrics skip the 0.5 s reset transient",
            TRANSIENT_SKIP_S == 0.5,
            f"TRANSIENT_SKIP_S={TRANSIENT_SKIP_S}",
        )
    )
    checks.append(
        (
            "recovery tolerance is 0.1 m/s",
            VX_RECOVERY_TOL_MPS == 0.1,
            f"VX_RECOVERY_TOL_MPS={VX_RECOVERY_TOL_MPS}",
        )
    )

    # metric smoke check on a toy episode: perfect tracking -> zero RMSE
    t = np.linspace(0.0, 4.0, 401)
    from locomotion.eval.metrics import fall_rate, vx_rmse

    checks.append(
        (
            "metrics compute on a toy episode",
            abs(vx_rmse(t, np.full_like(t, 0.5), np.full_like(t, 0.5))) < 1e-12
            and abs(fall_rate([False, False, True]) - 1 / 3) < 1e-12,
            "vx_rmse=0 on perfect tracking, fall_rate([F,F,T])=1/3",
        )
    )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list-seeds", action="store_true", help="print the frozen seed set, one per line"
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="run protocol self-checks (pre-flight gate for M1/M2 runs)",
    )
    args = parser.parse_args(argv)

    if args.list_seeds:
        for seed in EVAL_SEEDS:
            print(seed)
        return 0

    print(f"M1/M2 evaluation protocol {PROTOCOL_VERSION} (frozen {PROTOCOL_FROZEN_DATE})")
    print(f"EVAL_SEEDS: {list(EVAL_SEEDS)}")
    print(f"THRESHOLDS: {THRESHOLDS}")
    print()

    results = verify()
    failed = False
    for check, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed = True
        print(f"[{status}] {check} ({detail})")

    if failed:
        print("\nprotocol self-check FAILED; do not start a run from this state")
        return 1
    print("\nprotocol self-check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Submit and manage Go2W training jobs on the Windows host from macOS.

The Windows workspace is ``D:/work/go2w`` (override with GO2W_WIN_ROOT).  Code
is synchronized with a tar stream, jobs are launched through
``tools/win_job_runner.py`` in a detached process, and job state is stored on
the Windows side under ``rl/runs/_jobs/<job-id>/``.

Examples:

    python tools/win_train.py sync
    python tools/win_train.py submit --preset multi-segment --seed 0
    python tools/win_train.py submit --preset multi-segment --seed 0 --smoke
    python tools/win_train.py submit --preset high-level-curve --seed 0 \
        -- --total-steps 2000000 --envs 8
    python tools/win_train.py list
    python tools/win_train.py tail <job-id>
    python tools/win_train.py fetch <job-id>
    python tools/win_train.py stop <job-id>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = os.environ.get("GO2W_WIN_HOST", "winbox")
WIN_ROOT = os.environ.get("GO2W_WIN_ROOT", "D:/work/go2w").rstrip("/")
SENTINEL_ROOT = os.environ.get("GO2W_SENTINEL_ROOT", "D:/work/RL-Sentinel").rstrip("/")
WIN_PY = f"{WIN_ROOT}/.venv/Scripts/python.exe"
SENTINEL_PY = f"{SENTINEL_ROOT}/.venv/Scripts/python.exe"
RUNNER = f"{WIN_ROOT}/tools/win_job_runner.py"
JOBS_DIR = f"{WIN_ROOT}/rl/runs/_jobs"
WIN_TAR = "C:/Windows/System32/tar.exe"

PRESETS = (
    "multi-segment", "junction", "high-level-curve", "curve", "ppo", "sentinel", "custom",
)

TAR_EXCLUDES = (
    ".git", ".venv", "build", "deps", "rl/runs", "media", "reports",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".DS_Store", "*.pyc",
    ".conda", "*.egg-info",
)


def _run(cmd: list[str], *, input_text: Optional[str] = None,
         capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        input=input_text,
        text=True,
        capture_output=capture,
        check=False,
    )


def ssh_run(command: str) -> str:
    result = _run(["ssh", HOST, command])
    if result.returncode != 0:
        raise SystemExit(
            f"ssh failed ({result.returncode}): {command}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def win_path(path: str) -> str:
    return path.replace("/", "\\")


def remote_mkdir(path: str) -> None:
    """Create one directory level; parents must already exist."""
    ssh_run(f'cmd.exe /c "if not exist {win_path(path)} mkdir {win_path(path)}"')


def git_revision() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = _run(["git", "status", "--porcelain"]).stdout.strip()
    return {
        "commit": commit,
        "dirty": bool(dirty),
        "dirty_files": len(dirty.splitlines()) if dirty else 0,
        "host": platform.node(),
    }


def sync(force: bool = False) -> None:
    """Stream the working tree to the Windows host, excluding runtime dirs."""
    runner_exists = _run(
        ["ssh", HOST, f'cmd.exe /c "if exist {win_path(RUNNER)} echo yes"']
    ).stdout.strip() == "yes"
    if runner_exists and not force:
        try:
            statuses = json.loads(ssh_run(
                f'"{WIN_PY}" "{RUNNER}" status --jobs-dir "{JOBS_DIR}"'
            ))
            running = [j for j in statuses if j.get("alive")]
            if running:
                names = ", ".join(j["job_id"] for j in running)
                raise SystemExit(
                    f"refusing to sync while jobs are running: {names}\n"
                    "stop them or pass --force"
                )
        except (SystemExit, json.JSONDecodeError, ValueError):
            if force:
                pass
            else:
                # A malformed/missing runner is not a reason to block a sync;
                # only an explicit running-job check failure after a valid
                # response should block.
                pass

    tar_cmd = ["tar", "-czf", "-"]
    for pattern in TAR_EXCLUDES:
        tar_cmd += ["--exclude", pattern]
    tar_cmd.append(".")
    remote = f'"{WIN_TAR}" -xzf - -C "{WIN_ROOT}"'
    print(f"sync {REPO_ROOT} -> {HOST}:{WIN_ROOT}")
    producer = subprocess.Popen(tar_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE)
    consumer = subprocess.Popen(
        ["ssh", HOST, remote], stdin=producer.stdout,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False,
    )
    assert producer.stdout is not None
    producer.stdout.close()
    _out, err = consumer.communicate()
    producer_rc = producer.wait()
    if producer_rc != 0 or consumer.returncode != 0:
        raise SystemExit(
            f"sync failed: tar={producer_rc} ssh={consumer.returncode}\n"
            f"{err.decode('utf-8', errors='replace')}"
        )
    remote_mkdir(JOBS_DIR)
    rev = git_revision()
    print(f"sync ok  commit={rev['commit'][:12]} dirty_files={rev['dirty_files']}")


def run_root_for(preset: str, job_id: str) -> str:
    task_dir = {
        "multi-segment": "multi_segment",
        "junction": "junction",
        "high-level-curve": "high_level_curve",
        "curve": "traverse_curve_curriculum",
        "ppo": "ppo",
    }.get(preset, preset)
    return f"{WIN_ROOT}/rl/runs/{task_dir}/{job_id}"


def build_command(args: argparse.Namespace, job_id: str,
                  run_root: str) -> tuple[list[str], str, dict[str, str]]:
    """Return (command, cwd, env)."""
    env = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "MUJOCO_GL": "wgl",
        "GO2W_TRAIN_DEVICE": args.device,
    }
    for item in args.env or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got: {item}")
        env[key] = value

    extra = list(args.extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    if args.smoke:
        extra.append("--smoke")

    if args.preset == "custom":
        if not args.cmd:
            raise SystemExit("--preset custom requires --cmd")
        return shlex.split(args.cmd), args.cwd or WIN_ROOT, env

    if args.preset == "sentinel":
        out = f"{SENTINEL_ROOT}/artifacts/{job_id}"
        cmd = [SENTINEL_PY, "-u", "-m", "rl_risk_replay", "experiment", "--out", out]
        return cmd + extra, SENTINEL_ROOT, env

    base = [WIN_PY, "-u"]
    if args.preset == "multi-segment":
        cmd = base + [
            "scripts/train_multi_segment_curriculum.py",
            "--seed", str(args.seed), "--device", args.device,
            "--run-root", run_root,
        ]
    elif args.preset == "junction":
        cmd = base + [
            "scripts/train_junction_curriculum.py",
            "--seed", str(args.seed), "--device", args.device,
            "--run-root", run_root,
        ]
    elif args.preset == "high-level-curve":
        cmd = base + [
            "scripts/train_high_level_curve.py",
            "--seed", str(args.seed), "--device", args.device,
            "--run-dir", run_root,
        ]
    elif args.preset == "curve":
        env["GO2W_CURVE_RUN_ROOT"] = run_root
        env["GO2W_TRAIN_DEVICE"] = args.device
        cmd = base + ["scripts/train_curve_curriculum.py"]
    elif args.preset == "ppo":
        if not args.task:
            raise SystemExit("--preset ppo requires --task")
        cmd = base + [
            "rl/train.py", "--task", args.task,
            "--seed", str(args.seed), "--device", args.device,
            "--fresh", "--run-dir", run_root,
        ]
    else:
        raise SystemExit(f"unknown preset: {args.preset}")
    return cmd + extra, WIN_ROOT, env


def submit(args: argparse.Namespace) -> None:
    if not args.no_sync:
        sync(force=args.force_sync)

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    job_id = args.name or f"{args.preset}-seed{args.seed}-{stamp}-{uuid.uuid4().hex[:6]}"
    run_root = run_root_for(args.preset, job_id)
    command, cwd, env = build_command(args, job_id, run_root)
    source = git_revision()
    env["GO2W_SOURCE_COMMIT"] = source["commit"]
    env["GO2W_SOURCE_DIRTY"] = "true" if source["dirty"] else "false"
    job_dir = f"{JOBS_DIR}/{job_id}"
    log_path = f"{job_dir}/train.log"
    spec = {
        "job_id": job_id,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "cwd": cwd,
        "command": command,
        "log": log_path,
        "env": env,
        "task": args.preset,
        "seed": args.seed,
        "run_dir": run_root,
        "source": source,
    }

    if args.dry_run:
        print(json.dumps(spec, ensure_ascii=False, indent=2))
        return

    remote_mkdir(JOBS_DIR)
    remote_mkdir(job_dir)
    with tempfile.TemporaryDirectory() as tmp:
        local_spec = pathlib.Path(tmp) / "spec.json"
        local_spec.write_text(json.dumps(spec, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        scp = _run(
            ["scp", str(local_spec), f"{HOST}:{job_dir}/spec.json"],
            capture=False,
        )
        if scp.returncode != 0:
            raise SystemExit("scp of job spec failed")

    out = ssh_run(f'"{WIN_PY}" "{RUNNER}" launch --spec "{job_dir}/spec.json"')
    meta = json.loads(out)
    print(json.dumps({
        "job_id": job_id,
        "status": meta.get("status"),
        "pid": meta.get("launcher_pid"),
        "run_dir": run_root,
        "log": log_path,
        "host": HOST,
    }, ensure_ascii=False, indent=2))


def list_jobs(args: argparse.Namespace) -> None:
    out = ssh_run(f'"{WIN_PY}" "{RUNNER}" status --jobs-dir "{JOBS_DIR}"')
    items = json.loads(out)
    if args.json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return
    if not items:
        print("no jobs")
        return
    print(f"{'job':<44} {'status':<10} {'task':<18} {'seed':>5} {'dev':<5} "
          f"{'log':>10}")
    for item in items:
        print(
            f"{item.get('job_id',''):<44} {item.get('status',''):<10} "
            f"{str(item.get('task','')):<18} {str(item.get('seed','')):>5} "
            f"{str(item.get('device','')):<5} "
            f"{int(item.get('log_size',0)):>10}"
        )


def tail_job(args: argparse.Namespace) -> None:
    print(ssh_run(
        f'"{WIN_PY}" "{RUNNER}" tail --jobs-dir "{JOBS_DIR}" '
        f'--job "{args.job}" --lines {args.lines}'
    ), end="")


def stop_job(args: argparse.Namespace) -> None:
    print(ssh_run(
        f'"{WIN_PY}" "{RUNNER}" stop --jobs-dir "{JOBS_DIR}" --job "{args.job}"'
    ))


def fetch_job(args: argparse.Namespace) -> None:
    meta = json.loads(ssh_run(
        f'"{WIN_PY}" "{RUNNER}" status --jobs-dir "{JOBS_DIR}" --job "{args.job}"'
    ))[0]
    run_dir = args.run_dir or meta.get("run_dir")
    if not run_dir:
        # Older job metadata predates run_dir; read the immutable job spec.
        with tempfile.TemporaryDirectory() as tmp:
            spec_local = pathlib.Path(tmp) / "spec.json"
            spec_scp = _run(
                ["scp", "-q", f"{HOST}:{JOBS_DIR}/{args.job}/spec.json",
                 str(spec_local)],
                capture=False,
            )
            if spec_scp.returncode == 0 and spec_local.is_file():
                spec = json.loads(spec_local.read_text(encoding="utf-8"))
                run_dir = spec.get("run_dir")
    if not run_dir:
        raise SystemExit("job metadata/spec has no run_dir")
    run_dir = str(run_dir).replace("\\", "/").rstrip("/")
    expected_prefix = f"{WIN_ROOT.rstrip('/')}/rl/runs/"
    if not run_dir.startswith(expected_prefix):
        raise SystemExit(
            f"refusing to fetch a non-run directory: {run_dir}"
        )
    dest = pathlib.Path(args.dest).expanduser().resolve() if args.dest else (
        REPO_ROOT / "rl" / "runs" / "_fetched" / args.job
    )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetch {HOST}:{run_dir} -> {dest}")
    producer = subprocess.Popen(
        ["ssh", HOST, f'"{WIN_TAR}" -czf - -C "{run_dir}" .'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    consumer = subprocess.Popen(
        ["tar", "-xzf", "-", "-C", str(dest)], stdin=producer.stdout,
    )
    assert producer.stdout is not None
    producer.stdout.close()
    rc = consumer.wait()
    _out, err = producer.communicate()
    if producer.returncode != 0 or rc != 0:
        raise SystemExit(
            f"fetch failed: ssh={producer.returncode} tar={rc}\n"
            f"{err.decode('utf-8', errors='replace')}"
        )
    # Keep the job metadata and log with the fetched artifacts.
    job_dest = dest / "_job"
    job_dest.mkdir(exist_ok=True)
    _run(["scp", "-q", f"{HOST}:{JOBS_DIR}/{args.job}/spec.json",
          str(job_dest / "spec.json")], capture=False)
    _run(["scp", "-q", f"{HOST}:{JOBS_DIR}/{args.job}/meta.json",
          str(job_dest / "meta.json")], capture=False)
    _run(["scp", "-q", f"{HOST}:{JOBS_DIR}/{args.job}/train.log",
          str(job_dest / "train.log")], capture=False)
    print(f"fetched {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows training job manager")
    sub = parser.add_subparsers(dest="action", required=True)

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("--force", action="store_true")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--preset", choices=PRESETS, required=True)
    p_submit.add_argument("--seed", type=int, default=0)
    p_submit.add_argument("--task", default=None,
                          help="rl/train.py task for --preset ppo")
    p_submit.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                          default="cpu",
                          help="SB3 MlpPolicy is faster on CPU; cuda is available for comparison")
    p_submit.add_argument("--name", default=None)
    p_submit.add_argument("--smoke", action="store_true")
    p_submit.add_argument("--env", action="append", default=[],
                          help="extra environment KEY=VALUE (repeatable)")
    p_submit.add_argument("--cmd", default=None,
                          help="command string for --preset custom")
    p_submit.add_argument("--cwd", default=None,
                          help="working directory for --preset custom")
    p_submit.add_argument("--no-sync", action="store_true")
    p_submit.add_argument("--force-sync", action="store_true")
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.add_argument("extra", nargs=argparse.REMAINDER,
                          help="extra training args after --")

    p_list = sub.add_parser("list")
    p_list.add_argument("--json", action="store_true")

    p_tail = sub.add_parser("tail")
    p_tail.add_argument("job")
    p_tail.add_argument("-n", "--lines", type=int, default=60)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("job")

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("job")
    p_fetch.add_argument("--dest", default=None)
    p_fetch.add_argument("--run-dir", default=None,
                         help="Override the run directory; must be under rl/runs")

    args = parser.parse_args()
    if args.action == "sync":
        sync(force=args.force)
    elif args.action == "submit":
        submit(args)
    elif args.action == "list":
        list_jobs(args)
    elif args.action == "tail":
        tail_job(args)
    elif args.action == "stop":
        stop_job(args)
    elif args.action == "fetch":
        fetch_job(args)


if __name__ == "__main__":
    main()

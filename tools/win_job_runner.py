#!/usr/bin/env python3
"""Detached job runner for the Windows training host.

This file is copied to the Windows workspace by ``tools/win_train.py`` and is
executed there.  It intentionally uses only the standard library (plus psutil
when available) so it can launch jobs before a virtual environment is active.

Lifecycle:

    launch --spec <spec.json>   create the job directory and spawn ``run``
                                in a detached process
    run    --spec <spec.json>   execute the training command, write status
    status --all                print all job metadata as JSON
    tail   --job <id>           print the end of the log
    stop   --job <id>           terminate the process tree

The detached ``run`` process survives the SSH session that started it.  Job
state lives in ``<jobs-dir>/<job-id>/meta.json``; the training log lives next
to it as ``train.log``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback
from typing import Any, Optional


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        import psutil  # type: ignore[import-not-found]

        return psutil.pid_exists(int(pid))
    except ImportError:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/NH"],
                capture_output=True, text=True,
            )
            return str(int(pid)) in result.stdout
        try:
            os.kill(int(pid), 0)
            return True
        except OSError:
            return False


def creation_flags(detached: bool) -> int:
    flags = 0
    if os.name == "nt":
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if detached:
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
    return flags


def job_dir_from_spec(spec_path: pathlib.Path) -> pathlib.Path:
    return spec_path.resolve().parent


def win_path(value: str) -> str:
    return value.replace("/", "\\")


def _wmi_create(bat_path: str) -> int:
    """Start a batch file through WMI so it is not owned by the SSH job."""
    command_line = f'cmd.exe /c ""{bat_path}""'
    ps_literal = "'" + command_line.replace("'", "''") + "'"
    script = (
        "$ErrorActionPreference='Stop'; "
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
        f"-Arguments @{{CommandLine={ps_literal}}}; "
        "if ($r.ReturnValue -ne 0) { "
        "throw ('Win32_Process.Create failed: ' + $r.ReturnValue) }; "
        "$r.ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return int(result.stdout.strip().splitlines()[-1])


def _spawn_detached(cmd: list[str], spec: dict[str, Any],
                    job_dir: pathlib.Path, env: dict[str, str]) -> int:
    if os.name == "nt":
        launcher_log = job_dir / "launcher.log"
        bat = job_dir / "launch.bat"
        quoted = " ".join(f'"{part}"' for part in cmd)
        content = "\r\n".join([
            "@echo off",
            f'cd /d "{win_path(spec["cwd"])}"',
            f'{quoted} >> "{win_path(str(launcher_log))}" 2>&1',
        ]) + "\r\n"
        bat.write_text(content, encoding="utf-8")
        return _wmi_create(str(bat.resolve()))
    proc = subprocess.Popen(
        cmd, cwd=spec["cwd"], env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    return proc.pid


def load_spec(spec_path: pathlib.Path) -> dict[str, Any]:
    spec = read_json(spec_path)
    for key in ("job_id", "cwd", "command", "log"):
        if key not in spec:
            raise SystemExit(f"spec missing required key: {key}")
    return spec


def launch(spec_path: pathlib.Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    job_dir = job_dir_from_spec(spec_path)
    meta_path = job_dir / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        if meta.get("status") in ("launched", "running"):
            if pid_alive(meta.get("child_pid") or meta.get("pid")):
                raise SystemExit(f"job already running: {spec['job_id']}")
    python = spec.get("python") or sys.executable
    cmd = [str(python), str(pathlib.Path(__file__).resolve()),
           "run", "--spec", str(spec_path.resolve())]
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in spec.get("env", {}).items()})
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    launcher_pid = _spawn_detached(cmd, spec, job_dir, env)
    meta = {
        "job_id": spec["job_id"],
        "status": "launched",
        "launcher_pid": launcher_pid,
        "created_at": utc_now(),
        "spec": str(spec_path.resolve()),
        "log": spec["log"],
        "cwd": spec["cwd"],
        "command": spec["command"],
        "task": spec.get("task"),
        "seed": spec.get("seed"),
        "host": os.environ.get("COMPUTERNAME", ""),
    }
    write_json(meta_path, meta)
    return meta


def run(spec_path: pathlib.Path) -> int:
    spec = load_spec(spec_path)
    job_dir = job_dir_from_spec(spec_path)
    meta_path = job_dir / "meta.json"
    log_path = pathlib.Path(spec["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = read_json(meta_path)
        except (OSError, json.JSONDecodeError):
            meta = {}
    meta.update({
        "job_id": spec["job_id"],
        "status": "running",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "cwd": spec["cwd"],
        "command": spec["command"],
        "log": str(log_path),
        "task": spec.get("task"),
        "seed": spec.get("seed"),
        "device": spec.get("env", {}).get("GO2W_TRAIN_DEVICE"),
    })
    write_json(meta_path, meta)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in spec.get("env", {}).items()})
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        with open(log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"\n=== job {spec['job_id']} started {utc_now()} ===\n")
            log.write(f"cwd: {spec['cwd']}\n")
            log.write("command: " + json.dumps(spec["command"], ensure_ascii=False) + "\n")
            log.flush()
            proc = subprocess.Popen(
                spec["command"],
                cwd=spec["cwd"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags(detached=False) | (
                    0 if os.name == "nt" else getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                ),
                start_new_session=(os.name != "nt"),
            )
            meta["child_pid"] = proc.pid
            write_json(meta_path, meta)
            returncode = proc.wait()
        meta["exit_code"] = returncode
        meta["status"] = "finished" if returncode == 0 else "failed"
        meta["finished_at"] = utc_now()
        write_json(meta_path, meta)
        with open(log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(f"=== job {spec['job_id']} {meta['status']} "
                      f"exit={returncode} {utc_now()} ===\n")
        return returncode
    except BaseException:
        meta["status"] = "failed"
        meta["error"] = traceback.format_exc()
        meta["finished_at"] = utc_now()
        write_json(meta_path, meta)
        with open(log_path, "a", encoding="utf-8", errors="replace") as log:
            log.write(traceback.format_exc())
        return 1


def status(jobs_dir: pathlib.Path, job: Optional[str]) -> list[dict[str, Any]]:
    if job:
        meta_path = jobs_dir / job / "meta.json"
        if not meta_path.exists():
            raise SystemExit(f"job not found: {job}")
        items = [read_json(meta_path)]
    else:
        items = []
        if jobs_dir.is_dir():
            for meta_path in jobs_dir.glob("*/meta.json"):
                try:
                    items.append(read_json(meta_path))
                except (OSError, json.JSONDecodeError):
                    continue
    for item in items:
        pid = item.get("child_pid") or item.get("pid")
        item["alive"] = pid_alive(pid)
        if item["alive"] and item.get("status") in ("launched", "running"):
            item["status"] = "running"
        elif item.get("status") in ("launched", "running") and not item["alive"]:
            item["status"] = "lost"
        item["log_size"] = (
            pathlib.Path(item["log"]).stat().st_size
            if item.get("log") and pathlib.Path(item["log"]).exists() else 0
        )
    items.sort(key=lambda x: x.get("created_at") or "")
    return items


def tail(jobs_dir: pathlib.Path, job: str, lines: int) -> str:
    meta_path = jobs_dir / job / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"job not found: {job}")
    meta = read_json(meta_path)
    log_path = pathlib.Path(meta["log"])
    if not log_path.exists():
        return ""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.readlines()
    return "".join(content[-max(1, lines):])


def stop(jobs_dir: pathlib.Path, job: str) -> dict[str, Any]:
    meta_path = jobs_dir / job / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"job not found: {job}")
    meta = read_json(meta_path)
    pid = meta.get("child_pid") or meta.get("pid")
    if pid:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                           capture_output=True, text=True)
        else:
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
        deadline = time.time() + 5.0
        while pid_alive(int(pid)) and time.time() < deadline:
            time.sleep(0.1)
    meta["status"] = "stopped"
    meta["stopped_at"] = utc_now()
    write_json(meta_path, meta)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows detached training job runner")
    sub = parser.add_subparsers(dest="action", required=True)

    p_launch = sub.add_parser("launch")
    p_launch.add_argument("--spec", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--spec", required=True)

    p_status = sub.add_parser("status")
    p_status.add_argument("--jobs-dir", default=str(pathlib.Path(__file__).resolve().parent.parent / "rl" / "runs" / "_jobs"))
    p_status.add_argument("--job", default=None)

    p_tail = sub.add_parser("tail")
    p_tail.add_argument("--jobs-dir", default=str(pathlib.Path(__file__).resolve().parent.parent / "rl" / "runs" / "_jobs"))
    p_tail.add_argument("--job", required=True)
    p_tail.add_argument("--lines", type=int, default=60)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--jobs-dir", default=str(pathlib.Path(__file__).resolve().parent.parent / "rl" / "runs" / "_jobs"))
    p_stop.add_argument("--job", required=True)

    args = parser.parse_args()
    if args.action == "launch":
        print(json.dumps(launch(pathlib.Path(args.spec)), ensure_ascii=False))
    elif args.action == "run":
        return run(pathlib.Path(args.spec))
    elif args.action == "status":
        print(json.dumps(status(pathlib.Path(args.jobs_dir), args.job),
                         ensure_ascii=False, indent=2))
    elif args.action == "tail":
        print(tail(pathlib.Path(args.jobs_dir), args.job, args.lines), end="")
    elif args.action == "stop":
        print(json.dumps(stop(pathlib.Path(args.jobs_dir), args.job),
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

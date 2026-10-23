#!/usr/bin/env python3
"""Live training dashboard for the M1 three-seed run (no extra dependencies).

Serves a single page on http://127.0.0.1:8777 that polls ``/api/status`` every
two seconds.  The state is rebuilt by tail-parsing the training log, so the
dashboard never touches the trainer process and costs no GPU time.

What it shows:
- per-seed state (pending / training / done / failed), iteration progress,
  mean reward, velocity-tracking errors, elapsed and ETA
- overall progress across the three seeds
- GPU utilisation, memory and temperature
- a rolling reward curve and the last log lines

Usage::

    python scripts/monitor_training.py \
        --log "C:/Users/Administrator/.zcode/workspace/default/m1_training.log" \
        --port 8777

Then open http://127.0.0.1:8777 (it binds to 127.0.0.1 only).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SEED_START = re.compile(r"=== SEED (\d+) START ([\d:]+) ===")
SEED_DONE = re.compile(r"=== SEED (\d+) DONE rc=(\d+) ([\d:]+) ===")
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
MEAN_REWARD = re.compile(r"Mean reward:\s*(-?[\d.]+)")
ERR_XY = re.compile(r"error_vel_xy:\s*(-?[\d.]+)")
ERR_YAW = re.compile(r"error_vel_yaw:\s*(-?[\d.]+)")
TRAIN_TIME = re.compile(r"Training time:\s*([\d.]+) seconds")
TIME_ELAPSED = re.compile(r"Time elapsed:\s*(\d+):(\d+):(\d+)")
ETA_LINE = re.compile(r"ETA:\s*(\d+):(\d+):(\d+)")
ITER_TIME = re.compile(r"Iteration time:\s*([\d.]+)s")

MAX_HISTORY = 400
MAX_LINES = 120

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # run directly, sys.path[0] is scripts/, so the package root must be added
    # for the deferred ``from scripts.render_training_video import ...``
    sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = PROJECT_ROOT / "logs" / "rsl_rl" / "_snapshots"
MEDIA_DIR = PROJECT_ROOT / "media" / "training"
REPORTS_DIR = PROJECT_ROOT / "reports" / "training"


def list_reports() -> list[dict[str, Any]]:
    """Formal training reports written by scripts/train_report.py."""
    if not REPORTS_DIR.is_dir():
        return []
    reports = []
    for path in sorted(REPORTS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        reports.append(
            {
                "name": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            }
        )
    return reports


def list_videos() -> list[dict[str, Any]]:
    """MP4 recordings available for playback in the dashboard."""
    if not MEDIA_DIR.is_dir():
        return []
    videos = []
    for path in sorted(MEDIA_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = path.stat()
        videos.append(
            {
                "name": path.name,
                "size_mb": round(stat.st_size / 1e6, 2),
                "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            }
        )
    return videos

HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M1 训练监控 · Go2 速度跟踪</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --line: #262b36; --text: #e6e9ef;
    --muted: #8b93a7; --accent: #4c8dff; --ok: #3ddc97; --warn: #f5a524; --bad: #ff5c5c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 20px; }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .sub { color: var(--muted); font-size: 12px; }
  .grid { display: grid; gap: 12px; margin-top: 16px; }
  .cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  .cols-2 { grid-template-columns: 2fr 1fr; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .card h2 { font-size: 12px; letter-spacing: .04em; text-transform: uppercase;
             color: var(--muted); margin: 0 0 8px; font-weight: 600; }
  .big { font-size: 24px; font-variant-numeric: tabular-nums; }
  .mono { font-variant-numeric: tabular-nums; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .bar { height: 8px; background: #0b0d12; border-radius: 999px; overflow: hidden; margin-top: 10px; }
  .bar > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, #4c8dff, #3ddc97); transition: width .4s ease; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; }
  .tag { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; }
  .t-training { background: rgba(76,141,255,.16); color: #8db4ff; }
  .t-done { background: rgba(61,220,151,.14); color: var(--ok); }
  .t-pending { background: rgba(139,147,167,.14); color: var(--muted); }
  .t-failed { background: rgba(255,92,92,.14); color: var(--bad); }
  canvas { width: 100%; height: 160px; display: block; }
  pre { margin: 0; max-height: 220px; overflow: auto; font-size: 11.5px; color: #aab2c5;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }
  .muted { color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  @media (max-width: 900px) { .cols-4 { grid-template-columns: repeat(2, 1fr); } .cols-2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>M1 训练监控 <span class="sub">· Go2Loco-S1-Flat-Blind-v0 · 1500 iters × 2048 envs</span></h1>
      <div class="sub" id="statusline">连接中…</div>
    </div>
    <div class="sub mono" id="clock"></div>
  </header>

  <div class="grid cols-4" id="cards">
    <div class="card"><h2>总体进度</h2><div class="big mono" id="overall">—</div><div class="bar"><i id="overall-bar"></i></div></div>
    <div class="card"><h2>当前种子</h2><div class="big mono" id="curseed">—</div><div class="sub" id="curstate">—</div></div>
    <div class="card"><h2>迭代</h2><div class="big mono" id="iteration">—</div><div class="bar"><i id="iter-bar"></i></div></div>
    <div class="card"><h2>已用 / 预计剩余</h2><div class="big mono" id="eta">—</div><div class="sub" id="started">—</div></div>
  </div>

  <div class="grid cols-4">
    <div class="card"><h2>Mean reward</h2><div class="big mono" id="reward">—</div></div>
    <div class="card"><h2>error_vel_xy</h2><div class="big mono" id="errxy">—</div></div>
    <div class="card"><h2>error_vel_yaw</h2><div class="big mono" id="erryaw">—</div></div>
    <div class="card"><h2>GPU</h2><div class="big mono" id="gpu">—</div><div class="sub" id="gpumem">—</div></div>
  </div>

  <div class="grid cols-2">
    <div class="card">
      <h2>Reward 曲线（当前种子）</h2>
      <canvas id="curve" width="760" height="160"></canvas>
    </div>
    <div class="card">
      <h2>GPU 实时</h2>
      <div class="mono" id="gpu-live" style="font-size:20px">—</div>
      <div class="sub" id="gpu-sub">—</div>
      <div class="bar"><i id="gpu-bar"></i></div>
      <h2 style="margin-top:16px">日志新鲜度</h2>
      <div class="mono" id="logage">—</div>
    </div>
  </div>

  <div class="card" style="margin-top:12px">
    <h2>种子状态</h2>
    <table>
      <thead><tr><th>Seed</th><th>状态</th><th>迭代</th><th>Reward</th><th>error_xy</th><th>error_yaw</th><th>开始</th><th>完成</th><th>耗时</th></tr></thead>
      <tbody id="seeds"></tbody>
    </table>
  </div>

  <div class="card" style="margin-top:12px">
    <h2>训练报告（Markdown，训练结束时自动生成）</h2>
    <div class="sub" id="reports">暂无报告</div>
  </div>

  <div class="card" style="margin-top:12px">
    <h2>训练记录视频（MP4）</h2>
    <div class="sub" id="videos">暂无视频</div>
  </div>

  <div class="card" style="margin-top:12px">
    <h2>最近日志</h2>
    <pre id="tail"></pre>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d);
const dur = (s) => {
  if (s === null || s === undefined) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? h + "h " : "") + m + "m " + String(sec).padStart(2, "0") + "s";
};

function drawCurve(history) {
  const c = $("curve"), ctx = c.getContext("2d");
  const w = c.width, h = c.height, pad = 6;
  ctx.clearRect(0, 0, w, h);
  if (!history || history.length < 2) return;
  const xs = history.map(p => p.iteration), ys = history.map(p => p.reward);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  let ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const X = x => pad + (w - 2 * pad) * ((x - xmin) / Math.max(1, xmax - xmin));
  const Y = y => h - pad - (h - 2 * pad) * ((y - ymin) / (ymax - ymin));
  ctx.strokeStyle = "#262b36"; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const yy = pad + (h - 2 * pad) * i / 3;
    ctx.beginPath(); ctx.moveTo(pad, yy); ctx.lineTo(w - pad, yy); ctx.stroke();
  }
  ctx.strokeStyle = "#4c8dff"; ctx.lineWidth = 2; ctx.beginPath();
  history.forEach((p, i) => { const x = X(p.iteration), y = Y(p.reward); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  ctx.fillStyle = "#8b93a7"; ctx.font = "11px ui-monospace, monospace";
  ctx.fillText(fmt(ymax, 2), pad + 2, pad + 10);
  ctx.fillText(fmt(ymin, 2), pad + 2, h - pad - 2);
}

function render(d) {
  $("clock").textContent = "更新于 " + new Date().toLocaleTimeString();
  const running = d.log_age_s !== null && d.log_age_s < 90;
  $("statusline").innerHTML =
    `<span class="dot" style="background:${running ? "#3ddc97" : "#8b93a7"}"></span>` +
    (running ? "训练进行中" : "日志超过 90 秒未更新（可能已结束或暂停）") +
    ` · 日志 ${d.log_file || "—"}`;

  const done = d.seeds.filter(s => s.state === "done").length;
  const cur = d.seeds.find(s => s.state === "training");
  const overall = ((done + (cur ? (cur.iteration || 0) / (cur.max_iteration || 1500) : 0)) / d.seeds.length) * 100;
  $("overall").textContent = overall.toFixed(1) + "%";
  $("overall-bar").style.width = overall.toFixed(1) + "%";
  $("curseed").textContent = cur ? ("seed " + cur.seed) : (done === d.seeds.length ? "全部完成" : "—");
  $("curstate").textContent = cur ? "训练中" : (done ? "等待评估" : "—");
  $("iteration").textContent = cur ? `${cur.iteration}/${cur.max_iteration}` : "—";
  $("iter-bar").style.width = cur ? `${100 * cur.iteration / cur.max_iteration}%` : "0%";
  $("eta").textContent = cur ? `${dur(cur.elapsed_s)} / ${dur(cur.eta_s)}` : "—";
  $("started").textContent = cur ? "开始于 " + (cur.started || "—") : "—";
  $("reward").textContent = fmt(cur && cur.reward, 3);
  $("errxy").textContent = fmt(cur && cur.err_xy, 4);
  $("erryaw").textContent = fmt(cur && cur.err_yaw, 4);

  if (d.gpu) {
    $("gpu").textContent = (d.gpu.util ?? "—") + "%";
    $("gpumem").textContent = `${d.gpu.mem_used_mb} / ${d.gpu.mem_total_mb} MiB`;
    $("gpu-live").textContent = (d.gpu.util ?? "—") + "% · " + (d.gpu.temp_c ?? "—") + "°C";
    $("gpu-sub").textContent = `${d.gpu.mem_used_mb} MiB 已用 / ${d.gpu.mem_total_mb} MiB`;
    $("gpu-bar").style.width = (d.gpu.util || 0) + "%";
  }
  $("logage").textContent = d.log_age_s === null ? "—" : d.log_age_s.toFixed(1) + " 秒前写入";

  $("seeds").innerHTML = d.seeds.map(s => `
    <tr>
      <td class="mono">${s.seed}</td>
      <td><span class="tag t-${s.state}">${s.state}</span></td>
      <td class="mono">${s.iteration ? s.iteration + "/" + s.max_iteration : "—"}</td>
      <td class="mono">${fmt(s.reward, 3)}</td>
      <td class="mono">${fmt(s.err_xy, 4)}</td>
      <td class="mono">${fmt(s.err_yaw, 4)}</td>
      <td class="mono">${s.started || "—"}</td>
      <td class="mono">${s.finished || "—"}</td>
      <td class="mono">${dur(s.elapsed_s)}</td>
    </tr>`).join("");

  const videos = d.videos || [];
  const vr = d.video_renderer;
  let html = videos.length
    ? videos.map(v => `<a class="tag t-done" style="margin:2px 6px 2px 0;text-decoration:none"
         href="/media/${encodeURIComponent(v.name)}" target="_blank">${v.name} · ${v.size_mb} MB</a>`).join("")
    : "暂无视频";
  if (vr) {
    const detail = vr.last_error
      ? `<span style="color:#ff5c5c">渲染失败：${vr.last_error}</span>`
      : `后台渲染中 · 每 ${vr.interval_s}s 刷新 · 已完成 ${vr.renders} 次` +
        (vr.last_render_s ? ` · 上次耗时 ${vr.last_render_s}s` : "");
    html += `<div class="sub" style="margin-top:8px">${detail}</div>`;
  }
  $("videos").innerHTML = html;

  const reports = d.reports || [];
  $("reports").innerHTML = reports.length
    ? reports.map(r => `<a class="tag t-done" style="margin:2px 6px 2px 0;text-decoration:none"
         href="/reports/${encodeURIComponent(r.name)}" target="_blank">${r.name} · ${r.size_kb} KB · ${r.mtime}</a>`).join("")
    : "暂无报告（训练结束时由 train_locomotion.py 自动生成）";

  drawCurve(d.history || []);
  $("tail").textContent = (d.recent_lines || []).join("\\n");
}

async function tick() {
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    render(await r.json());
  } catch (e) {
    $("statusline").textContent = "无法连接监控服务：" + e;
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


class LogWatcher:
    """Incrementally tail-parse a training log into dashboard state."""

    def __init__(
        self, log_path: pathlib.Path, num_seeds: int = 3, snapshot_interval_s: float = 15.0
    ) -> None:
        self.log_path = log_path
        self.num_seeds = num_seeds
        self.snapshot_interval_s = snapshot_interval_s
        self._offset = 0
        self._lock = threading.Lock()
        self._last_snapshot_epoch = 0.0
        self._seeds: dict[int, dict[str, Any]] = {
            seed: {
                "seed": seed,
                "state": "pending",
                "iteration": 0,
                "max_iteration": 1500,
                "reward": None,
                "err_xy": None,
                "err_yaw": None,
                "started": None,
                "started_epoch": None,
                "elapsed_s": None,
                "eta_s": None,
                "history": [],
            }
            for seed in range(num_seeds)
        }
        self._lines: list[str] = []
        self._last_write: float | None = None

    def _parse_line(self, line: str) -> None:
        clean = ANSI.sub("", line).rstrip()
        if not clean:
            return
        self._lines.append(clean)
        if len(self._lines) > MAX_LINES:
            self._lines = self._lines[-MAX_LINES:]

        start = SEED_START.search(clean)
        if start:
            seed = int(start.group(1))
            record = self._seeds.setdefault(seed, {"seed": seed, "history": []})
            record.update(
                state="training",
                started=start.group(2),
                started_epoch=time.time(),
                iteration=0,
                reward=None,
                err_xy=None,
                err_yaw=None,
                history=[],
            )
            return

        done = SEED_DONE.search(clean)
        if done:
            seed = int(done.group(1))
            record = self._seeds.setdefault(seed, {"seed": seed, "history": []})
            record["state"] = "done" if int(done.group(2)) == 0 else "failed"
            record["finished"] = done.group(3)
            if record.get("started_epoch"):
                record["elapsed_s"] = time.time() - record["started_epoch"]
            return

        current = next((s for s in self._seeds.values() if s.get("state") == "training"), None)
        if current is None:
            if self.num_seeds == 1:
                # A direct launcher run has no "=== SEED n START ===" marker
                # (that marker only exists in the three-seed loop), so a
                # single-seed monitor adopts the first metric line as seed 0.
                current = self._seeds[0]
                current["state"] = "training"
                if not current.get("started"):
                    current["started"] = time.strftime("%H:%M:%S")
            else:
                return

        match = ITERATION.search(clean)
        if match:
            current["iteration"] = int(match.group(1))
            current["max_iteration"] = int(match.group(2))
            return
        match = MEAN_REWARD.search(clean)
        if match:
            current["reward"] = float(match.group(1))
            current["history"].append(
                {"iteration": current["iteration"], "reward": current["reward"]}
            )
            if len(current["history"]) > MAX_HISTORY:
                current["history"] = current["history"][-MAX_HISTORY:]
            return
        match = ERR_XY.search(clean)
        if match:
            current["err_xy"] = float(match.group(1))
            return
        match = ERR_YAW.search(clean)
        if match:
            current["err_yaw"] = float(match.group(1))
            return
        match = TRAIN_TIME.search(clean)
        if match:
            current["train_time_s"] = float(match.group(1))
            return
        # RSL-RL prints its own clock every iteration; prefer it over any
        # estimate derived from the monitor's own start time.
        match = TIME_ELAPSED.search(clean)
        if match:
            h, m, s = (int(part) for part in match.groups())
            current["elapsed_s"] = h * 3600 + m * 60 + s
            return
        match = ETA_LINE.search(clean)
        if match:
            h, m, s = (int(part) for part in match.groups())
            current["eta_current_s"] = h * 3600 + m * 60 + s
            return
        match = ITER_TIME.search(clean)
        if match:
            current["iter_time_s"] = float(match.group(1))

    def refresh(self) -> None:
        with self._lock:
            try:
                stat = self.log_path.stat()
            except OSError:
                return
            self._last_write = stat.st_mtime
            if stat.st_size < self._offset:  # truncated / rotated
                self._offset = 0
                self._lines = []
            if stat.st_size == self._offset:
                return
            with self.log_path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                for line in handle:
                    self._parse_line(line)
                self._offset = handle.tell()

    def _eta(self, record: dict[str, Any]) -> float | None:
        """Remaining wall-clock for all seeds, using RSL-RL's own clock."""
        if record["state"] != "training":
            return None
        elapsed = record.get("elapsed_s")
        eta_current = record.get("eta_current_s")
        if eta_current is None:
            return None
        pending = sum(1 for s in self._seeds.values() if s["state"] == "pending")
        # a not-yet-started seed is assumed to take as long as the current one
        per_seed = (elapsed or 0.0) + eta_current
        return float(eta_current) + pending * per_seed

    def snapshot(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            current_seed = next(
                (s for s in self._seeds.values() if s["state"] == "training"), None
            )
            current_history: list[dict[str, Any]] = []
            seeds = []
            for seed in sorted(self._seeds):
                record = dict(self._seeds[seed])
                record["eta_s"] = self._eta(self._seeds[seed])
                record.pop("started_epoch", None)
                history = record.pop("history", [])
                if current_seed is not None and seed == current_seed["seed"]:
                    current_history = history
                seeds.append(record)
            age = None if self._last_write is None else max(0.0, time.time() - self._last_write)
            payload = {
                "log_file": self.log_path.name,
                "log_age_s": age,
                "seeds": seeds,
                "history": current_history,
                "recent_lines": self._lines[-40:],
                "gpu": read_gpu(),
                "videos": list_videos(),
                "reports": list_reports(),
            }
            self._maybe_record_snapshot(payload)
            return payload

    def _maybe_record_snapshot(self, payload: dict[str, Any]) -> None:
        """Append a period snapshot to JSONL so the run can be replayed later."""
        now = time.time()
        if now - self._last_snapshot_epoch < self.snapshot_interval_s:
            return
        self._last_snapshot_epoch = now
        try:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            path = SNAPSHOT_DIR / f"training_{time.strftime('%Y%m%d')}.jsonl"
            record = {
                "t": round(now, 3),
                "log_age_s": payload["log_age_s"],
                "gpu": payload["gpu"],
                "seeds": [
                    {
                        key: seed.get(key)
                        for key in (
                            "seed",
                            "state",
                            "iteration",
                            "max_iteration",
                            "reward",
                            "err_xy",
                            "err_yaw",
                            "elapsed_s",
                            "eta_s",
                        )
                    }
                    for seed in payload["seeds"]
                ],
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # a snapshot failure must never break the dashboard


def read_gpu() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        return {
            "util": float(parts[0]),
            "mem_used_mb": int(float(parts[1])),
            "mem_total_mb": int(float(parts[2])),
            "temp_c": int(float(parts[3])),
        }
    except ValueError:
        return None


class VideoRenderer(threading.Thread):
    """Render the current run into an MP4 on a fixed interval.

    CPU work by design: it replays the log, never touches the simulator.  The
    interval is a deliberate compromise -- frequent enough to feel live, rare
    enough not to compete with the trainer for the CPU.
    """

    def __init__(
        self, log_path: pathlib.Path, out_path: pathlib.Path, interval_s: float, stride: int, fps: int
    ) -> None:
        super().__init__(daemon=True)
        self.log_path = log_path
        self.out_path = out_path
        self.interval_s = interval_s
        self.stride = stride
        self.fps = fps
        self._stop_event = threading.Event()
        self.last_render_s: float | None = None
        self.last_error: str | None = None
        self.renders = 0

    def run(self) -> None:
        self.render_once()
        while not self._stop_event.wait(self.interval_s):
            self.render_once()

    def render_once(self) -> None:
        """Render in a child process.

        matplotlib is not reliably thread-safe, so running it inside the
        dashboard process can hang the render thread (observed).  A child
        process is isolated, bounded by a timeout, and cannot stall the API.
        """
        script = PROJECT_ROOT / "scripts" / "render_training_video.py"
        command = [
            sys.executable,
            str(script),
            "--log",
            str(self.log_path),
            "--out",
            str(self.out_path),
            "--stride",
            str(self.stride),
            "--fps",
            str(self.fps),
        ]
        started = time.time()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().splitlines()
                self.last_error = tail[-1] if tail else f"exit {result.returncode}"
                return
            self.last_render_s = time.time() - started
            self.renders += 1
            self.last_error = None
        except subprocess.TimeoutExpired:
            self.last_error = "render timed out after 600 s"
        except Exception as exc:  # never let a render kill the dashboard
            self.last_error = f"{type(exc).__name__}: {exc}"

    def stop(self) -> None:
        self._stop_event.set()

    def status(self) -> dict[str, Any]:
        return {
            "active": True,
            "interval_s": self.interval_s,
            "renders": self.renders,
            "last_render_s": None if self.last_render_s is None else round(self.last_render_s, 1),
            "last_error": self.last_error,
            "file": self.out_path.name,
        }


def make_handler(
    watcher: LogWatcher, page: bytes, renderer: VideoRenderer | None = None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            if self.path.startswith("/api/status"):
                status = watcher.snapshot()
                if renderer is not None:
                    status["video_renderer"] = renderer.status()
                payload = json.dumps(status, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if self.path.startswith("/reports/"):
                name = self.path[len("/reports/"):]
                candidate = (REPORTS_DIR / name).resolve()
                if candidate.parent != REPORTS_DIR.resolve() or not candidate.is_file():
                    self.send_error(404)
                    return
                data = candidate.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path.startswith("/media/"):
                name = self.path[len("/media/"):]
                candidate = (MEDIA_DIR / name).resolve()
                if candidate.parent != MEDIA_DIR.resolve() or not candidate.is_file():
                    self.send_error(404)
                    return
                data = candidate.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            return  # keep the console quiet

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=15.0,
        help="seconds between JSONL snapshots written for later replay",
    )
    parser.add_argument(
        "--video-interval",
        type=float,
        default=300.0,
        help="seconds between MP4 refreshes; 0 disables background rendering",
    )
    parser.add_argument("--video-stride", type=int, default=12, help="log records per video frame")
    parser.add_argument("--video-fps", type=int, default=15)
    args = parser.parse_args(argv)

    watcher = LogWatcher(
        args.log, num_seeds=args.seeds, snapshot_interval_s=args.snapshot_interval
    )
    watcher.refresh()

    renderer: VideoRenderer | None = None
    if args.video_interval > 0:
        renderer = VideoRenderer(
            log_path=args.log,
            out_path=MEDIA_DIR / f"{args.log.stem}_progress.mp4",
            interval_s=args.video_interval,
            stride=args.video_stride,
            fps=args.video_fps,
        )
        renderer.start()

    page = HTML_PAGE.encode("utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(watcher, page, renderer))
    url = f"http://127.0.0.1:{args.port}"
    print(f"[monitor] watching {args.log}")
    print(f"[monitor] open {url}")
    print("[monitor] Ctrl+C to stop", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

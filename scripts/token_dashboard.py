#!/usr/bin/env python3
"""Token 消耗统计：监听 Codex 日志数据库，提取每次对话的 token 用量。

数据源：~/.codex/logs_2.sqlite 的 logs 表，response.completed SSE 事件含 usage。
历史记录从日志补足（可能不完整），新记录由面板后台线程实时采集并写入
~/.codex_token_history.jsonl，保证从启用时刻起完整。
"""

from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import threading
import time
from typing import Any, Optional

LOGS_DB = pathlib.Path.home() / ".codex" / "logs_2.sqlite"
HISTORY_FILE = pathlib.Path.home() / ".codex_token_history.jsonl"
POLL_INTERVAL = 5.0

_lock = threading.Lock()
_records: dict[str, dict] = {}
_last_scan_ts = 0.0


def _connect_ro() -> Optional[sqlite3.Connection]:
    """只读连接 Codex 日志库，避免与正在写入的进程冲突。"""
    if not LOGS_DB.exists():
        return None
    try:
        return sqlite3.connect(f"file:{LOGS_DB}?mode=ro", uri=True)
    except sqlite3.Error:
        return None


def _parse_usage(body: str) -> Optional[dict]:
    """从 'SSE event: {...}' 日志体解析 response.usage。"""
    m = re.search(r"SSE event: (.*)$", body, re.S)
    if not m:
        return None
    try:
        event = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return None
    response = event.get("response") or {}
    usage = response.get("usage")
    if not usage or not isinstance(usage, dict):
        return None
    rid = response.get("id")
    if not rid:
        return None
    return {
        "response_id": rid,
        "model": response.get("model") or "unknown",
        "ts": float(response.get("created_at") or event.get("created_at") or time.time()),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "cached_tokens": int(
            (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        ),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "reasoning_tokens": int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        ),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "source": "log_history",
    }


def load_history() -> None:
    """从 jsonl 加载已有记录（去重）。"""
    global _records
    records: dict[str, dict] = {}
    if HISTORY_FILE.exists():
        try:
            for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                records[rec["response_id"]] = rec
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    with _lock:
        _records = records


def append_record(rec: dict) -> None:
    """追加一条记录到 jsonl（按 response_id 去重）。"""
    with _lock:
        if rec["response_id"] in _records:
            return
        _records[rec["response_id"]] = rec
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def scan_logs(until_ts: Optional[float] = None) -> int:
    """扫描日志库，把新的 response.completed usage 记录写入 jsonl。返回新增条数。"""
    global _last_scan_ts
    conn = _connect_ro()
    if conn is None:
        return 0
    cutoff = until_ts if until_ts is not None else (_last_scan_ts or 0.0)
    added = 0
    try:
        rows = conn.execute(
            "SELECT ts, feedback_log_body FROM logs "
            "WHERE feedback_log_body LIKE '%response.completed%' "
            "AND feedback_log_body LIKE '%\"usage\"%' "
            "AND ts >= ? ORDER BY ts ASC",
            (int(cutoff),),
        ).fetchall()
        for ts, body in rows:
            rec = _parse_usage(body or "")
            if rec is not None:
                if rec["ts"] < cutoff and rec["source"] == "log_history":
                    rec["source"] = "live"
                append_record(rec)
                added += 1
        if rows:
            _last_scan_ts = float(rows[-1][0])
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return added


def bootstrap_history() -> int:
    """首次启动：从日志补足全部历史 usage 记录。返回补足条数。"""
    conn = _connect_ro()
    if conn is None:
        return 0
    added = 0
    try:
        rows = conn.execute(
            "SELECT ts, feedback_log_body FROM logs "
            "WHERE feedback_log_body LIKE '%response.completed%' "
            "AND feedback_log_body LIKE '%\"usage\"%' ORDER BY ts ASC"
        ).fetchall()
        for ts, body in rows:
            rec = _parse_usage(body or "")
            if rec is not None:
                with _lock:
                    if rec["response_id"] in _records:
                        continue
                append_record(rec)
                added += 1
        if rows:
            _last_scan_ts = float(rows[-1][0])
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return added


def compute_metrics(records: list[dict]) -> dict:
    """从记录计算累计与效率指标。"""
    n = len(records)
    if n == 0:
        return {
            "count": 0, "input": 0, "cached": 0, "output": 0,
            "reasoning": 0, "total": 0, "cache_hit_rate": None,
            "output_ratio": None, "reasoning_ratio": None,
            "avg_input": None, "avg_output": None,
        }
    total_in = sum(r["input_tokens"] for r in records)
    total_cached = sum(r["cached_tokens"] for r in records)
    total_out = sum(r["output_tokens"] for r in records)
    total_reasoning = sum(r["reasoning_tokens"] for r in records)
    total = sum(r["total_tokens"] for r in records)
    return {
        "count": n,
        "input": total_in,
        "cached": total_cached,
        "output": total_out,
        "reasoning": total_reasoning,
        "total": total,
        "cache_hit_rate": total_cached / total_in if total_in else None,
        "output_ratio": total_out / total_in if total_in else None,
        "reasoning_ratio": total_reasoning / total_out if total_out else None,
        "avg_input": total_in / n,
        "avg_output": total_out / n,
    }


def trend_24h(records: list[dict], now: float | None = None) -> list[dict]:
    """最近 24 小时逐小时消耗趋势。"""
    now = now if now is not None else time.time()
    hours: dict[int, dict] = {}
    for r in records:
        ts = float(r["ts"])
        if now - ts > 86400:
            continue
        hour = int(ts // 3600)
        h = hours.setdefault(hour, {"ts": hour * 3600, "input": 0, "output": 0, "total": 0, "count": 0})
        h["input"] += r["input_tokens"]
        h["output"] += r["output_tokens"]
        h["total"] += r["total_tokens"]
        h["count"] += 1
    return [hours[k] for k in sorted(hours)]


def snapshot() -> dict:
    """返回完整快照：明细 + 累计指标 + 24h 趋势。"""
    with _lock:
        records = list(_records.values())
    records.sort(key=lambda r: float(r["ts"]))
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": compute_metrics(records),
        "trend": trend_24h(records),
        "records": [
            {
                "ts": float(r["ts"]),
                "model": r["model"],
                "input": r["input_tokens"],
                "cached": r["cached_tokens"],
                "output": r["output_tokens"],
                "reasoning": r["reasoning_tokens"],
                "total": r["total_tokens"],
                "source": r.get("source", "live"),
            }
            for r in reversed(records)
        ],
    }


def refresh_once() -> int:
    """供面板后台线程调用：补足历史（首启）+ 扫描新增。返回新增条数。"""
    global _last_scan_ts
    with _lock:
        loaded = len(_records)
    if loaded == 0:
        bootstrap_history()
    return scan_logs()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Token 消耗统计")
    parser.add_argument("--once", action="store_true", help="只扫描一次并输出快照")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()
    load_history()
    added = refresh_once()
    snap = snapshot()
    if args.json:
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    else:
        m = snap["metrics"]
        print(f"Token 统计 {snap['time']}  本次新增 {added} 条")
        print(
            f"累计: {m['count']} 次  input={m['input']:,}  output={m['output']:,}  "
            f"total={m['total']:,}  cached={m['cached']:,}"
        )
        if m["cache_hit_rate"] is not None:
            print(
                f"缓存命中率 {m['cache_hit_rate']*100:.1f}%  "
                f"输出转化率 {m['output_ratio']*100:.2f}%  "
                f"推理占比 {m['reasoning_ratio']*100:.0f}%"
            )

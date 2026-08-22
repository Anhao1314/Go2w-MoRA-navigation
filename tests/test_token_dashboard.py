"""Token 统计纯函数测试：解析、指标、去重、趋势。"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from scripts import token_dashboard as td


def sample_body(rid, created_at, input_t, cached, output_t, reasoning):
    usage = {
        "input_tokens": input_t,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_t,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_t + output_t,
    }
    event = {
        "type": "response.completed",
        "response": {
            "id": rid,
            "object": "response",
            "created_at": created_at,
            "model": "deepseek-v4-flash",
            "usage": usage,
        },
    }
    return "SSE event: " + json.dumps(event)


class ParseUsageTest(unittest.TestCase):
    def test_parse_usage(self):
        body = sample_body("r1", 1786000000, 1000, 800, 200, 50)
        rec = td._parse_usage(body)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["response_id"], "r1")
        self.assertEqual(rec["input_tokens"], 1000)
        self.assertEqual(rec["cached_tokens"], 800)
        self.assertEqual(rec["output_tokens"], 200)
        self.assertEqual(rec["reasoning_tokens"], 50)
        self.assertEqual(rec["total_tokens"], 1200)

    def test_parse_invalid(self):
        self.assertIsNone(td._parse_usage("not json"))
        self.assertIsNone(td._parse_usage("SSE event: {\"type\":\"x\"}"))


class MetricsTest(unittest.TestCase):
    def test_compute_metrics(self):
        records = [
            {"input_tokens": 1000, "cached_tokens": 800, "output_tokens": 200,
             "reasoning_tokens": 50, "total_tokens": 1200},
            {"input_tokens": 2000, "cached_tokens": 1000, "output_tokens": 400,
             "reasoning_tokens": 100, "total_tokens": 2400},
        ]
        m = td.compute_metrics(records)
        self.assertEqual(m["count"], 2)
        self.assertEqual(m["input"], 3000)
        self.assertEqual(m["cached"], 1800)
        self.assertEqual(m["output"], 600)
        self.assertAlmostEqual(m["cache_hit_rate"], 0.6)
        self.assertAlmostEqual(m["output_ratio"], 0.2)
        self.assertAlmostEqual(m["reasoning_ratio"], 0.25)
        self.assertEqual(m["avg_input"], 1500)

    def test_empty_metrics(self):
        m = td.compute_metrics([])
        self.assertEqual(m["count"], 0)
        self.assertIsNone(m["cache_hit_rate"])


class HistoryTest(unittest.TestCase):
    def test_append_and_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            with mock.patch.object(td, "HISTORY_FILE", path), \
                 mock.patch.object(td, "_records", {}):
                td.load_history()
                td.append_record({"response_id": "r1", "ts": 1.0, "input_tokens": 10})
                td.append_record({"response_id": "r1", "ts": 1.0, "input_tokens": 99})
                td.append_record({"response_id": "r2", "ts": 2.0, "input_tokens": 20})
                td.load_history()
                self.assertEqual(set(td._records.keys()), {"r1", "r2"})
                self.assertEqual(td._records["r1"]["input_tokens"], 10)


class TrendTest(unittest.TestCase):
    def test_trend_24h(self):
        now = time.time()
        records = [
            {"ts": now - 3600, "input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100},
            {"ts": now - 7200, "input_tokens": 2000, "output_tokens": 200, "total_tokens": 2200},
            {"ts": now - 100000, "input_tokens": 999, "output_tokens": 1, "total_tokens": 1000},
        ]
        trend = td.trend_24h(records, now)
        self.assertEqual(len(trend), 2)
        self.assertEqual(sum(h["count"] for h in trend), 2)


if __name__ == "__main__":
    unittest.main()

"""Smoke tests for the eval_protocol CLI (frozen seeds + thresholds self-check)."""

from __future__ import annotations

import io
import contextlib
import unittest

from scripts.eval_protocol import PROTOCOL_VERSION, main, verify


class TestVerify(unittest.TestCase):
    def test_all_protocol_self_checks_pass(self) -> None:
        results = verify()
        self.assertTrue(results, "verify() must return at least one check")
        failed = [f"{check}: {detail}" for check, ok, detail in results if not ok]
        self.assertEqual(failed, [], "protocol self-checks must all pass")

    def test_verify_covers_the_frozen_axes(self) -> None:
        names = [check for check, _, _ in verify()]
        self.assertTrue(any("seed" in name.lower() for name in names))
        self.assertTrue(any("threshold" in name.lower() for name in names))
        self.assertTrue(any("transient" in name.lower() for name in names))


class TestMainCli(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_verify_exits_zero(self) -> None:
        code, out = self._run(["--verify"])
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)
        self.assertNotIn("[FAIL]", out)

    def test_list_seeds_prints_exactly_the_frozen_set(self) -> None:
        code, out = self._run(["--list-seeds"])
        self.assertEqual(code, 0)
        self.assertEqual([int(line) for line in out.split()], list(range(20)))

    def test_protocol_version_is_reported(self) -> None:
        _, out = self._run(["--verify"])
        self.assertIn(PROTOCOL_VERSION, out)


if __name__ == "__main__":
    unittest.main()

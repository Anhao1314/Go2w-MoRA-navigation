"""Ensure generated C++ and Python model constants are current."""

from __future__ import annotations

import unittest

from scripts import gen_model_constants as gen


class ModelConstantsSyncTests(unittest.TestCase):
    def test_generated_files_match_model(self) -> None:
        data = gen.build()
        self.assertEqual(gen.CPP_HEADER.read_text(), gen.emit_cpp(data))
        self.assertEqual(gen.PY_MODULE.read_text(), gen.emit_python(data))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for strict n781 scorer."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from score_n781_strict import score


class TestScoreStrict(unittest.TestCase):
    def test_reads_metrics_success_and_rejects_top_level(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            man = {
                "task_ids_sorted": [str(i) for i in range(781)],
            }
            man_path = root / "m.json"
            man_path.write_text(json.dumps(man))
            res = root / "res"
            # write 781 results; only metrics.success counts
            for i in range(781):
                d = res / f"t{i}"
                d.mkdir(parents=True)
                (d / "result.json").write_text(
                    json.dumps(
                        {
                            "identity": i,
                            "success": 1,  # decoy — must be IGNORED
                            "metrics": {"success": 1 if i < 309 else 0},
                        }
                    )
                )
            meta, by_task = score(res, man_path, name="t")
            self.assertEqual(meta["n_success"], 309)
            self.assertEqual(meta["n_completed"], 781)
            self.assertAlmostEqual(meta["sr_pct"], round(100 * 309 / 781, 4))

    def test_fails_on_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            man_path = root / "m.json"
            man_path.write_text(json.dumps({"task_ids_sorted": [str(i) for i in range(781)]}))
            res = root / "res"
            for i in range(780):
                d = res / f"t{i}"
                d.mkdir(parents=True)
                (d / "result.json").write_text(
                    json.dumps({"identity": i, "metrics": {"success": 0}})
                )
            with self.assertRaises(SystemExit):
                score(res, man_path, name="t")


if __name__ == "__main__":
    unittest.main()

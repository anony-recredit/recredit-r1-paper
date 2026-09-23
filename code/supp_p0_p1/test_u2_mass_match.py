#!/usr/bin/env python3
"""Unit tests for Geo-token U2 mass match (mock tokenizer, no HF required)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from build_mass_matched_uniform_stage1_parquet import (
    assert_mass_match,
    compute_c_tau_by_traj,
    geo_token_weights_for_response,
    transform_frame,
    verify_u2_against_geo,
)


class _MockTok:
    """Character-as-token tokenizer with offset_mapping."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        text = text or ""
        ids = list(range(len(text)))
        offsets = [(i, i + 1) for i in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}


class TestGeoTokenU2(unittest.TestCase):
    def test_grounding_uses_perc(self):
        tok = _MockTok()
        # response: "abXXXcd" with g_t="XXX" → chars 2,3,4 get perc=2, rest reas=1
        w, m = geo_token_weights_for_response(tok, "abXXXcd", "", "XXX", "", 2.0, 1.0)
        self.assertEqual(sum(m), 7)
        self.assertEqual(w[0], 1.0)
        self.assertEqual(w[2], 2.0)
        self.assertEqual(w[4], 2.0)
        self.assertEqual(w[6], 1.0)
        self.assertAlmostEqual(sum(wi for wi, mi in zip(w, m) if mi), 1 + 1 + 2 + 2 + 2 + 1 + 1)

    def test_c_tau_matches_geo_mass(self):
        tok = _MockTok()
        df = pd.DataFrame(
            [
                {
                    "traj_id": "T1",
                    "response": "abXYcd",
                    "think_t": "",
                    "g_t": "XY",
                    "a_t": "",
                    "perc_weight": 3.0,
                    "reas_weight": 1.0,
                },
                {
                    "traj_id": "T1",
                    "response": "ZZ",
                    "think_t": "",
                    "g_t": "",
                    "a_t": "",
                    "perc_weight": 3.0,
                    "reas_weight": 1.0,
                },
            ]
        )
        # step0: g=XY at [2,4) → perc on 2 tokens, reas on 4 → mass=3*2+1*4=10, n=6
        # step1: no g → all reas → mass=2, n=2
        # c=(10+2)/(6+2)=1.5
        c_map, audit = compute_c_tau_by_traj(df, tok)
        self.assertAlmostEqual(c_map["T1"], 1.5)
        assert_mass_match(audit, atol=1e-9)
        out, meta = transform_frame(df, tok, atol=1e-9)
        self.assertTrue((out["perc_weight"] == 1.5).all())
        self.assertTrue((out["reas_weight"] == 1.5).all())
        self.assertTrue(meta["verify"]["ok"])
        self.assertEqual(meta["verify"]["n_mass_matched"], 1)

    def test_q_t_match_would_fail_verify_spirit(self):
        """If we wrongly set c from q_t, verify_u2_against_geo must not all-match."""
        tok = _MockTok()
        geo = pd.DataFrame(
            [
                {
                    "traj_id": "T",
                    "response": "GGGGrrrr",
                    "think_t": "",
                    "g_t": "GGGG",
                    "a_t": "",
                    "perc_weight": 0.4,
                    "reas_weight": 1.2,
                    "q_t": 0.4,
                }
            ]
        )
        # Wrong U2: c = q_t = 0.4
        bad = geo.copy()
        bad["perc_weight"] = 0.4
        bad["reas_weight"] = 0.4
        v = verify_u2_against_geo(geo, bad, tok, atol=1e-6)
        self.assertFalse(v["ok"])
        self.assertLess(v["mean_u2_over_geo"], 0.95)


if __name__ == "__main__":
    unittest.main()

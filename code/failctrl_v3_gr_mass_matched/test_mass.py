#!/usr/bin/env python3
"""Mass-matching unit tests (no GPU)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from prepare_v3_parquets import M_MAX, assign_failure_weights


class MassTests(unittest.TestCase):
    def test_example_from_plan(self):
        df = pd.DataFrame({"q_t": [0.10, 0.90], "w_t": [1.0, 1.0]})
        # choose eta,S so m=0.20 without cap
        uni = assign_failure_weights(df, eta=0.2, scale=1.0, routing="uniform_half")
        typ = assign_failure_weights(df, eta=0.2, scale=1.0, routing="q_proxy")
        np.testing.assert_allclose(uni["m_t"], [0.2, 0.2])
        np.testing.assert_allclose(typ["m_t"], [0.2, 0.2])
        np.testing.assert_allclose(-uni["perc_weight"], [0.10, 0.10])
        np.testing.assert_allclose(-uni["reas_weight"], [0.10, 0.10])
        np.testing.assert_allclose(-typ["perc_weight"], [0.18, 0.02])
        np.testing.assert_allclose(-typ["reas_weight"], [0.02, 0.18])
        self.assertAlmostEqual(float(uni["perc_weight"].abs().sum() + uni["reas_weight"].abs().sum()),
                               float(typ["perc_weight"].abs().sum() + typ["reas_weight"].abs().sum()))

    def test_cap_before_route(self):
        df = pd.DataFrame({"q_t": [0.2], "w_t": [10.0]})
        uni = assign_failure_weights(df, eta=0.5, scale=1.0, routing="uniform_half")
        typ = assign_failure_weights(df, eta=0.5, scale=1.0, routing="q_proxy")
        self.assertEqual(float(uni["m_t"].iloc[0]), M_MAX)
        self.assertEqual(float(typ["m_t"].iloc[0]), M_MAX)
        self.assertTrue(bool(uni["mass_saturated"].iloc[0]))
        self.assertAlmostEqual(-float(uni["perc_weight"].iloc[0]), 0.25)
        self.assertAlmostEqual(-float(typ["perc_weight"].iloc[0]), 0.4)
        self.assertAlmostEqual(-float(typ["reas_weight"].iloc[0]), 0.1)

    def test_q_half_identical(self):
        df = pd.DataFrame({"q_t": [0.5], "w_t": [0.8]})
        uni = assign_failure_weights(df, eta=0.25, scale=1.0, routing="uniform_half")
        typ = assign_failure_weights(df, eta=0.25, scale=1.0, routing="q_proxy")
        np.testing.assert_allclose(uni["perc_weight"], typ["perc_weight"])
        np.testing.assert_allclose(uni["reas_weight"], typ["reas_weight"])

    def test_invert_swaps_type(self):
        df = pd.DataFrame({"q_t": [0.10, 0.90], "w_t": [1.0, 1.0]})
        typ = assign_failure_weights(df, eta=0.2, scale=1.0, routing="q_proxy")
        inv = assign_failure_weights(df, eta=0.2, scale=1.0, routing="q_proxy_invert")
        np.testing.assert_allclose(typ["m_t"], inv["m_t"])
        np.testing.assert_allclose(typ["perc_weight"], inv["reas_weight"])
        np.testing.assert_allclose(typ["reas_weight"], inv["perc_weight"])
        np.testing.assert_allclose(-inv["perc_weight"], [0.02, 0.18])
        np.testing.assert_allclose(-inv["reas_weight"], [0.18, 0.02])


if __name__ == "__main__":
    unittest.main()

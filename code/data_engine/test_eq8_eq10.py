#!/usr/bin/env python3
"""Unit tests: reward coefficients must match paper Eqs. (7)–(8)."""
from __future__ import annotations

import math
import unittest

from reward import gated_advantages, paper_eq8_coefficients, spatial_weights, type_scores


class Eq8Tests(unittest.TestCase):
    def test_success_lambda1_equals_wt(self):
        # Paper: on success, s_perc = s_reas = w_t, so A^k = w_t * (+1)
        qs = [0.2, 0.8]
        out = paper_eq8_coefficients(qs, R_L2=1, lam=1.0)
        w = out["w_t"]
        self.assertAlmostEqual(sum(w), 1.0, places=6)
        for i in range(2):
            self.assertAlmostEqual(out["s_t_perc"][i], w[i], places=6)
            self.assertAlmostEqual(out["s_t_reas"][i], w[i], places=6)
            self.assertAlmostEqual(out["A_t_perc"][i], w[i], places=6)
            self.assertAlmostEqual(out["A_t_reas"][i], w[i], places=6)
        # Must NOT be q^2 / sum(q) style
        bad = [w[i] * qs[i] for i in range(2)]
        self.assertFalse(
            all(math.isclose(out["A_t_perc"][i], bad[i], rel_tol=1e-9) for i in range(2))
        )

    def test_failure_independent_type_scores(self):
        qs = [0.1, 0.9]
        ws = spatial_weights(qs, R_L2=0)
        sp, sr = type_scores(qs, R_L2=0)
        # perc mass on low-q; reas mass on high-q
        self.assertGreater(sp[0], sp[1])
        self.assertGreater(sr[1], sr[0])
        self.assertAlmostEqual(sum(sp), 1.0, places=6)
        self.assertAlmostEqual(sum(sr), 1.0, places=6)
        ap, ar = gated_advantages(ws, sp, sr, A_traj=-1.0, lam=1.0)
        # λ=1 → A^k = s^k * A_t
        self.assertAlmostEqual(ap[0], -sp[0], places=6)
        self.assertAlmostEqual(ar[1], -sr[1], places=6)
        # Old buggy form was w*(1-q)*A etc.
        old_ap0 = ws[0] * (1.0 - qs[0]) * (-1.0)
        self.assertFalse(math.isclose(ap[0], old_ap0, rel_tol=1e-9, abs_tol=1e-12))

    def test_lambda0_type_agnostic(self):
        qs = [0.25, 0.75]
        out = paper_eq8_coefficients(qs, R_L2=1, lam=0.0)
        for i in range(2):
            self.assertAlmostEqual(out["A_t_perc"][i], out["w_t"][i], places=6)
            self.assertAlmostEqual(out["A_t_reas"][i], out["w_t"][i], places=6)

    def test_eq10_signed_ce_form(self):
        """Eq. (10) is token-weighted CE with signed A^k (trainer: CE * w).

        Positive A → standard weighted CE; negative A → signed push-down.
        """
        out = paper_eq8_coefficients([0.2, 0.8], R_L2=0, lam=1.0)
        self.assertTrue(all(a <= 0 for a in out["A_t_perc"]))
        self.assertTrue(all(a <= 0 for a in out["A_t_reas"]))
        out_ok = paper_eq8_coefficients([0.2, 0.8], R_L2=1, lam=1.0)
        self.assertTrue(all(a >= 0 for a in out_ok["A_t_perc"]))


if __name__ == "__main__":
    unittest.main()

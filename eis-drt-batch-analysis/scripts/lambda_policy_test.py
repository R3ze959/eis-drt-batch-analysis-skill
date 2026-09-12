#!/usr/bin/env python3
"""Synthetic regressions for primary lambda search and numerical scale covariance."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from pyimpspec import DataSet

import batch_drt as batch


F = np.logspace(5, -2, 41)
Z = 1.2 + 2 / (1 + 1j * F / 3000) + 5 / (1 + 1j * F / 3)


def options(**changes):
    values = dict(lambda_policy="consensus", lambda_min=1e-7, lambda_max=.1,
                  lambda_points=13, fixed_lambda=1e-4, derivative_order=1,
                  rbf_type="gaussian", rbf_shape="fwhm", shape_coeff=.5,
                  solver_timeout=30, num_procs=1, impedance_scaling="median")
    values.update(changes)
    return SimpleNamespace(**values)


def fixed_fixture(data, value, inductance, opt):
    """Controlled L-curve fixture; actual solver is covered separately below."""
    t = np.logspace(-6, 0, 41)
    amplitude = 1 / (1 + value * 1e4)
    gamma = amplitude * np.exp(-((np.log10(t) + 2) / .5) ** 2)
    return SimpleNamespace(
        get_impedances=lambda: data.get_impedances() * (1 + value ** .5),
        get_drt_data=lambda: (t, gamma), pseudo_chisqr=value,
    )


class LambdaPolicyTests(unittest.TestCase):
    def select_mock(self, returned, opt=None, chisqr=1e-4):
        opt = opt or options()
        with patch.object(batch, "fixed_tr_rbf", side_effect=fixed_fixture), patch(
            "numerical_conditioning.calculate_tr_rbf_conditioned",
            return_value=SimpleNamespace(lambda_value=returned, pseudo_chisqr=chisqr),
        ):
            return batch.select_lambda(DataSet(F, Z), Z, False, opt)

    def test_outside_candidate_is_not_clipped_into_an_optimum(self):
        selected, _, info, _ = self.select_mock(.3)
        self.assertEqual(info["selection_method"], "grid-lcurve-fallback")
        self.assertFalse(info["mgcv_start_independent"])
        self.assertFalse(info["lambda_was_clipped"])
        self.assertLess(selected, .1)
        self.assertEqual([r["returned_lambda"] for r in info["mgcv_candidates"]], [.3] * 3)
        self.assertTrue(all(not r["eligible_for_consensus"] for r in info["mgcv_candidates"]))

    def test_explicit_mgcv_rejects_outside_consensus(self):
        with self.assertRaisesRegex(RuntimeError, "in-range nonboundary"):
            self.select_mock(.3, options(lambda_policy="package-mgcv"))

    def test_nonpositive_nonfinite_or_user_endpoint_not_accepted(self):
        for value in [0., -1., float("nan"), float("inf"), 1e-7, .1]:
            with self.subTest(value=value):
                _, _, info, _ = self.select_mock(value)
                self.assertFalse(info["mgcv_start_independent"])
                json.dumps(batch.json_ready(info), allow_nan=False)

    def test_package_boundary_rejected_inside_wider_user_range(self):
        _, _, info, _ = self.select_mock(1., options(lambda_min=1e-8, lambda_max=10.))
        self.assertTrue(all(r["rejection_reason"] == "on-package-search-boundary"
                            for r in info["mgcv_candidates"]))

    def test_nonfinite_fit_metric_not_consensus(self):
        _, _, info, _ = self.select_mock(1e-4, chisqr=float("nan"))
        self.assertFalse(info["mgcv_start_independent"])

    def test_narrow_search_has_distinct_starts_and_incomplete_sensitivity(self):
        _, _, info, _ = self.select_mock(3e-6, options(
            lambda_min=1e-6, lambda_max=1e-5, fixed_lambda=3e-6))
        starts = [r["start_lambda"] for r in info["mgcv_candidates"]]
        self.assertEqual(len(set(starts)), 3)
        self.assertTrue(all(1e-6 < x < 1e-5 for x in starts))
        self.assertTrue(info["mgcv_start_independent"])
        self.assertFalse(info["full_sensitivity_range"])
        self.assertTrue(info["boundary_sensitive"])

    def test_fixed_boundary_is_not_full_neighborhood(self):
        _, _, info, _ = self.select_mock(1e-4, options(lambda_policy="fixed", fixed_lambda=1e-7))
        self.assertFalse(info["full_sensitivity_range"])
        self.assertTrue(info["boundary_sensitive"])

    def test_constant_lcurve_fallback_is_flagged(self):
        with patch.object(batch, "curve_metrics", return_value=(1., 1.)):
            _, _, info, _ = self.select_mock(.3)
        self.assertIn("undefined", info["selection_reason"])
        self.assertTrue(info["boundary_sensitive"])

    def test_straight_log_lcurve_does_not_supply_a_corner(self):
        residual = 10. ** np.arange(7)
        roughness = 10. ** np.arange(6, -1, -1)
        self.assertTrue(np.all(np.isnan(batch.lcurve_chord_distance(residual, roughness))))

    def test_actual_consensus_is_scale_covariant(self):
        opt = options()
        selected, result, info, _ = batch.select_lambda(DataSet(F, Z), Z, False, opt)
        small, small_result, small_info, _ = batch.select_lambda(DataSet(F, Z * 1e-4), Z * 1e-4, False, opt)
        # Allow tiny optimizer differences, not a lambda decade or a spurious peak.
        self.assertLess(abs(np.log10(small / selected)), .01)
        self.assertEqual(info["selection_method"], small_info["selection_method"])
        np.testing.assert_allclose(small_result.get_drt_data()[1] / 1e-4,
                                   result.get_drt_data()[1], rtol=.005, atol=.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Synthetic search-boundary and truncated-sensitivity counterexamples."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import advanced_drt as advanced
from control_validation import classify_results
from regression_test import spectrum


def curve_rows(residual, roughness):
    return [{"lambda": float(value), "residual_norm": float(r), "roughness_norm": float(g)}
            for value, r, g in zip(np.geomspace(1e-7, 1e-1, len(residual)), residual, roughness)]


def synthetic_distribution(value=1e-4):
    tau = np.geomspace(1e-4, 10, 80)
    gamma = np.exp(-np.log(tau/.1)**2)
    return {"status": "completed", "lambda": value, "num_tau": len(tau),
            "tau_s": tau, "gamma": gamma, "residual_rms_percent": .01,
            "support": {"absolute_area": float(np.trapezoid(gamma, np.log(tau)))}}


class LcurveSearchTests(unittest.TestCase):
    def test_first_interior_candidate_is_boundary_sensitive(self):
        rows = curve_rows([1, 1.01, 2, 3, 4, 5, 6], [100, 1.1, 1.08, 1.06, 1.04, 1.02, 1])
        diagnostics = advanced._lcurve_diagnostics(rows)
        self.assertEqual(advanced._lcurve_choice(rows), 1)
        self.assertEqual(diagnostics["status"], "edge-adjacent")
        self.assertTrue(diagnostics["boundary_sensitive"])
        self.assertGreater(diagnostics["selected_chord_distance"], 0)

    def test_last_interior_candidate_is_boundary_sensitive(self):
        rows = curve_rows([1, 1.02, 1.04, 1.06, 1.08, 1.1, 100], [6, 5, 4, 3, 2, 1.01, 1])
        self.assertEqual(advanced._lcurve_choice(rows), 5)
        self.assertTrue(advanced._lcurve_diagnostics(rows)["boundary_sensitive"])

    def test_bracketed_corner_is_not_boundary_sensitive(self):
        rows = curve_rows([1, 1.01, 1.05, 1.3, 2.5, 5, 10], [100, 99, 90, 10, 5, 3, 2])
        diagnostics = advanced._lcurve_diagnostics(rows)
        self.assertEqual(diagnostics["status"], "corner")
        self.assertFalse(diagnostics["boundary_sensitive"])
        self.assertEqual(len(diagnostics["chord_distances"]), 7)

    def test_flat_curve_is_degenerate(self):
        rows = curve_rows([1]*7, [2]*7)
        self.assertEqual(advanced._lcurve_choice(rows), 3)
        self.assertEqual(advanced._lcurve_diagnostics(rows)["status"], "degenerate")
        self.assertTrue(advanced._lcurve_diagnostics(rows)["boundary_sensitive"])

    def test_straight_log_curve_is_degenerate(self):
        rows = curve_rows(10.**np.arange(7), 10.**np.arange(6, -1, -1))
        self.assertEqual(advanced._lcurve_choice(rows), 3)
        self.assertEqual(advanced._lcurve_diagnostics(rows)["status"], "degenerate")

    def test_insufficient_or_duplicate_search_is_not_bracketed(self):
        for count in (1, 2):
            rows = curve_rows(np.arange(count)+1, np.arange(count, 0, -1))
            self.assertEqual(advanced._lcurve_diagnostics(rows)["status"], "insufficient-search")
        repeated = [{"lambda": 1e-4, "residual_norm": 1., "roughness_norm": 2.}]*7
        self.assertEqual(advanced._lcurve_diagnostics(repeated)["status"], "insufficient-search")


class SensitivityRangeTests(unittest.TestCase):
    def assess(self, value, lower=1e-7, upper=1e-1):
        result = synthetic_distribution(value)
        alternative = copy.deepcopy(result)
        with patch.object(advanced, "_solve_distribution", return_value=alternative):
            advanced._distribution_sensitivity(result, np.array([1., 10.]), np.ones(2), None,
                signed=True, derivative_order=1, include_series_c=True, lambda_min=lower, lambda_max=upper)
        return result["sensitivity"]

    def test_clipped_low_neighbor_is_not_stable_even_with_identical_curves(self):
        sensitivity = self.assess(1e-7)
        self.assertEqual(sensitivity["status"], "range-incomplete")
        self.assertFalse(sensitivity["full_neighborhood_available"])
        self.assertTrue(sensitivity["evaluated_cases_stable"])
        self.assertTrue(sensitivity["cases"][0]["range_clipped"])

    def test_clipped_high_neighbor_is_not_stable(self):
        sensitivity = self.assess(1e-1)
        self.assertEqual(sensitivity["status"], "range-incomplete")
        self.assertTrue(sensitivity["cases"][1]["range_clipped"])

    def test_full_one_decade_neighbors_can_be_stable(self):
        sensitivity = self.assess(1e-4)
        self.assertEqual(sensitivity["status"], "stable")
        self.assertTrue(sensitivity["full_neighborhood_available"])
        self.assertTrue(all(not row["range_clipped"] for row in sensitivity["cases"]))

    def test_partial_decade_clipping_is_also_range_incomplete(self):
        sensitivity = self.assess(3e-7)
        self.assertEqual(sensitivity["status"], "range-incomplete")
        self.assertNotEqual(sensitivity["requested_lambda_range"], sensitivity["evaluated_lambda_range"])


class AdvancedAcceptanceTests(unittest.TestCase):
    def test_signed_boundary_and_incomplete_range_require_review_but_preserve_curves(self):
        for key, value, reason in (("lambda_boundary_sensitive", True, "signed-lambda-boundary-sensitive"),
            ("sensitivity", {"status": "range-incomplete"}, "signed-lambda-range-incomplete")):
            with self.subTest(reason=reason):
                row = spectrum("signed")
                row["signed_gdrt"] = {"status": "completed", "gamma": [1., -.5], key: value}
                classify_results([row], advanced.assess_batch_validity([row]))
                self.assertEqual(row["status"], "completed")
                self.assertEqual(row["numeric_status"], "review-required")
                self.assertIn(reason, row["decision_basis"])
                self.assertEqual(row["signed_gdrt"]["gamma"], [1., -.5])

    def test_actual_signed_spectrum_completes_with_search_diagnostics(self):
        frequency = np.geomspace(1e5, 1e-3, 41)
        omega = 2*np.pi*frequency
        impedance = .8 + 3/(1+1j*omega*.02) + 1.5*(1j*omega*.5)/(1+1j*omega*.5)
        result = advanced.continuous_signed_gdrt(frequency, impedance, lambda_points=7)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["solver"]["converged"])
        self.assertLess(result["residual_rms_percent"], 1.)
        self.assertLess(min(result["gamma"]), -.05)
        self.assertIn(result["lambda_search"]["status"], {"corner", "edge-adjacent", "degenerate"})
        self.assertEqual(result["lambda_search"]["selected_lambda"], result["lambda"])
        self.assertIn("full_neighborhood_available", result["sensitivity"])

    def test_one_successful_candidate_is_retained_with_insufficient_search(self):
        frequency = np.geomspace(1e4, 1e-3, 31)
        impedance = .8 + 3/(1+1j*2*np.pi*frequency*.02)
        solver = advanced.lsq_linear
        calls = 0

        def fail_after_first(matrix, rhs, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return solver(matrix, rhs, **kwargs)
            return SimpleNamespace(x=np.zeros(matrix.shape[1]), success=False, status=-1,
                                   message="synthetic optimizer failure", optimality=1.)

        with patch.object(advanced, "lsq_linear", side_effect=fail_after_first):
            result = advanced._solve_distribution(frequency, impedance,
                lambda omega, tau: 1/(1+1j*omega[:, None]*tau), signed=True,
                lambda_grid=[1e-5, 1e-4, 1e-3], num_tau=80)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["lambda_search"]["status"], "insufficient-search")
        self.assertEqual(result["lambda_search"]["failed_candidate_count"], 2)
        self.assertEqual(result["lambda_search"]["successful_grid_indices"], [0])
        self.assertTrue(result["lambda_boundary_sensitive"])

    def test_ddt_rejects_unbracketed_search_without_marking_calculation_failed(self):
        frequency = np.geomspace(1e4, 1e-3, 31)
        impedance = .8 + 3*advanced.diffusion_kernel("transmissive-short", 2*np.pi*frequency, np.array([.2]))[:, 0]
        result = advanced.ddt_analysis(frequency, impedance, lambda_min=1e-4, lambda_max=2e-4, lambda_points=3)
        self.assertEqual(result["status"], "completed")
        candidates = [row for row in result["candidates"] if row["boundary"] != "semi-infinite"]
        self.assertTrue(all(row["status"] == "completed" for row in candidates))
        for candidate in candidates:
            self.assertTrue(candidate["lambda_boundary_sensitive"])
            self.assertIn(candidate["lambda_search"]["status"], {"edge-adjacent", "degenerate"})
            self.assertEqual(candidate["sensitivity"]["status"], "range-incomplete")
            self.assertFalse(candidate["eligible_for_distribution_ranking"])
            self.assertIn("lambda-search-boundary", candidate["rejection_reasons"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Development regressions for prediction-selected RC/signed competition.

All cases are analytic synthetic spectra, not hidden holdout data. The pure-RC
counterexample deliberately exposes signed negative lobes from discretization.
"""
import unittest
from unittest.mock import patch

import numpy as np

from advanced_drt import predictive_rc_kernel_competition, relative_complex_rms


class FitSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frequency = np.geomspace(1e4, .01, 61)
        s = 2j * np.pi * cls.frequency
        rc = 5 + 30 / (1 + s * .03)
        cls.spectra = {"RC": rc, "RC+L": rc + s * 1e-4,
                       "RC+RL": rc + 20 * s * .003 / (1 + s * .003),
                       "ZARC": 5 + 30 / (1 + (s * .03)**.8)}
        generator = np.random.default_rng(20260910)
        cls.spectra["noisy-RC"] = rc + .002 * np.abs(rc) * (generator.normal(size=61) + 1j * generator.normal(size=61))
        cls.results = {name: predictive_rc_kernel_competition(cls.frequency, z, lambda_min=1e-7)
                       for name, z in cls.spectra.items()}

    def test_rc_and_rc_l_prefer_positive_despite_smaller_signed_residual(self):
        for name in ("RC", "RC+L"):
            with self.subTest(name=name):
                result = self.results[name]
                self.assertEqual(result["selected_branch"], "predictive_positive_rc")
                positive, signed = (result["candidates"][k] for k in ("positive_rc", "signed_rc"))
                self.assertLess(positive["residual_rms_percent"], .2)
                self.assertLess(signed["residual_rms_percent"], positive["residual_rms_percent"])
                self.assertGreater(signed["support"]["negative_absolute_area_fraction"], .05)
                self.assertTrue(np.all(positive["gamma"] >= 0))
                self.assertFalse(signed["negative_lobes_are_mechanism_evidence"])

    def test_true_rl_requires_signed_reconstruction_but_does_not_assign_mechanism(self):
        result = self.results["RC+RL"]
        self.assertEqual(result["selected_branch"], "predictive_signed_rc")
        positive, signed = (result["candidates"][k] for k in ("positive_rc", "signed_rc"))
        self.assertLess(signed["prediction_rms_percent"], .2)
        self.assertGreater(positive["prediction_rms_percent"], 10)
        self.assertFalse(result["selection_is_physical_acceptance"])
        self.assertFalse(result["negative_lobes_are_mechanism_evidence"])

    def test_zarc_prefers_positive_and_predicts_well(self):
        result = self.results["ZARC"]
        self.assertEqual(result["selected_branch"], "predictive_positive_rc")
        self.assertLess(result["candidates"]["positive_rc"]["prediction_rms_percent"], .1)

    def test_noise_not_silently_removed_or_interpreted_as_negative_process(self):
        result = self.results["noisy-RC"]
        self.assertEqual(result["selected_branch"], "predictive_positive_rc")
        self.assertEqual(result["cross_validation"]["raw_points_removed"], 0)
        self.assertEqual(result["cross_validation"]["final_refit_point_count"], 61)
        self.assertFalse(result["cross_validation"]["independent_measurement_validation"])
        self.assertIn("not noise-variance", result["candidates"]["positive_rc"]["solver"]["weighting"])

    def test_impedance_scale_invariance_with_same_noise_realization(self):
        z = self.spectra["noisy-RC"]
        baseline = self.results["noisy-RC"]
        for scale in (1e-6, 1e6):
            with self.subTest(scale=scale):
                result = predictive_rc_kernel_competition(self.frequency, z * scale, lambda_min=1e-7)
                self.assertEqual(result["selected_branch"], baseline["selected_branch"])
                for branch in ("positive_rc", "signed_rc"):
                    fit, reference = result["candidates"][branch], baseline["candidates"][branch]
                    self.assertEqual(fit["lambda"], reference["lambda"])
                    np.testing.assert_allclose(fit["reconstructed"] / scale, reference["reconstructed"], rtol=1e-7, atol=1e-7)
                    np.testing.assert_allclose(fit["gamma"] / scale, reference["gamma"], rtol=1e-6, atol=1e-6)
                    self.assertAlmostEqual(fit["prediction_rms_percent"], reference["prediction_rms_percent"], places=7)

    def test_noise_level_ladder_does_not_convert_numerical_fit_into_acceptance(self):
        generator = np.random.default_rng(24601)
        rc = self.spectra["RC"]
        realization = generator.normal(size=61) + 1j * generator.normal(size=61)
        for fraction in (.001, .01, .03):
            with self.subTest(relative_noise=fraction):
                z = rc + fraction * np.abs(rc) * realization
                result = predictive_rc_kernel_competition(self.frequency, z, lambda_min=1e-7)
                self.assertEqual(result["selected_branch"], "predictive_positive_rc")
                selected = result["candidates"]["positive_rc"]
                self.assertLess(selected["prediction_rms_percent"], 300 * fraction + .2)
                self.assertFalse(result["selection_is_physical_acceptance"])
                if fraction == .03:
                    self.assertFalse(selected["reconstruction_reliable"])
                    self.assertIn("prediction-or-refit-above-2-percent-heuristic", selected["reliability_reasons"])

    def test_reconstructed_arrays_match_exported_tau_gamma_and_passive_nuisance(self):
        for name, result in self.results.items():
            for branch, fit in result["candidates"].items():
                with self.subTest(name=name, branch=branch):
                    omega = 2 * np.pi * self.frequency
                    tau = np.asarray(fit["tau_s"])
                    dln = np.mean(np.diff(np.log(tau)))
                    nuisance = fit["nuisance"]
                    expected = nuisance["R_inf_ohm"] + 1j * omega * nuisance["L_henry"]
                    expected += nuisance["C_reciprocal_scaled_ohm"] * min(omega) / (1j * omega)
                    expected += (dln / (1 + 1j * omega[:, None] * tau)) @ fit["gamma"]
                    np.testing.assert_allclose(expected, fit["reconstructed"], rtol=1e-12, atol=1e-12)
                    self.assertGreaterEqual(nuisance["R_inf_ohm"], 0)
                    self.assertGreaterEqual(nuisance["L_henry"], 0)
                    self.assertGreaterEqual(nuisance["C_reciprocal_scaled_ohm"], 0)
                    self.assertTrue(fit["solver"]["passive_nuisance_bounds"])
                    self.assertAlmostEqual(fit["residual_rms_percent"], relative_complex_rms(self.spectra[name], expected), places=9)

    def test_original_order_and_inputs_preserved(self):
        order = np.random.default_rng(910).permutation(len(self.frequency))
        f, z = self.frequency[order].copy(), self.spectra["RC"][order].copy()
        before_f, before_z = f.copy(), z.copy()
        result = predictive_rc_kernel_competition(f, z, lambda_min=1e-7)
        np.testing.assert_array_equal(f, before_f)
        np.testing.assert_array_equal(z, before_z)
        for branch, fit in result["candidates"].items():
            np.testing.assert_array_equal(fit["measured_frequency_hz"], f)
            np.testing.assert_allclose(fit["reconstructed"], self.results["RC"]["candidates"][branch]["reconstructed"][order], rtol=1e-7, atol=1e-7)

    def test_folds_hold_duplicate_frequencies_together_and_keep_endpoints(self):
        f = np.repeat(self.frequency[::3], 2)
        z = 5 + 30 / (1 + 2j * np.pi * f * .03)
        result = predictive_rc_kernel_competition(f, z, lambda_points=3, num_tau=40)
        held = []
        for fold in result["cross_validation"]["folds"]:
            train, test = fold["train_indices"], fold["heldout_indices"]
            self.assertFalse(set(f[train]) & set(f[test]))
            self.assertIn(min(f), f[train])
            self.assertIn(max(f), f[train])
            self.assertEqual(set(train) | set(test), set(range(len(f))))
            held.extend(test)
        self.assertEqual(sorted(held), np.flatnonzero((f > min(f)) & (f < max(f))).tolist())

    def test_user_lambda_range_not_extended_or_boundary_hidden(self):
        for fit in self.results["RC"]["candidates"].values():
            self.assertTrue(fit["lambda_boundary_sensitive"])
            self.assertIn("lambda-search-boundary", fit["reliability_reasons"])
            self.assertEqual(fit["sensitivity"]["status"], "range-incomplete")
            self.assertFalse(fit["sensitivity"]["individual_peak_stability_assessed"])
            self.assertTrue(all(1e-7 <= row["lambda"] <= .1 for row in fit["lambda_grid"]))
            self.assertTrue(all(1e-7 <= row["lambda"] <= .1 for row in fit["sensitivity"]["cases"]))
            self.assertTrue(any(row["range_clipped"] for row in fit["sensitivity"]["cases"]))
            for case in fit["sensitivity"]["cases"]:
                self.assertEqual(len(case["gamma"]), len(fit["gamma"]))
                self.assertEqual(len(case["reconstructed"]), len(self.frequency))
                np.testing.assert_array_equal(case["tau_s"], fit["tau_s"])
                np.testing.assert_array_equal(case["measured_frequency_hz"], self.frequency)

    def test_cv_lambda_is_selected_from_predictions_not_training_error(self):
        for result in self.results.values():
            for fit in result["candidates"].values():
                rows = fit["lambda_grid"]
                best = min(row["prediction_rms_percent"] for row in rows)
                eligible = [row for row in rows if row["prediction_rms_percent"] <= 1.05 * best]
                self.assertEqual(fit["lambda"], max(row["lambda"] for row in eligible))
                self.assertTrue(fit["cv"]["all_folds_converged"])
                self.assertTrue(fit["full_refit_guard"]["converged"])
                self.assertFalse(fit["cv"]["endpoints_validated"])

    def test_optimizer_failure_cannot_be_recommended(self):
        with patch("advanced_drt.lsq_linear", side_effect=ValueError("intentional optimizer failure")):
            result = predictive_rc_kernel_competition(self.frequency, self.spectra["RC"], lambda_points=3, num_tau=40)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["selected_branch"])
        for fit in result["candidates"].values():
            self.assertEqual(fit["status"], "failed")
            self.assertNotIn("reconstructed", fit)
            self.assertTrue(all(not row["all_folds_converged"] for row in fit["lambda_grid"]))

    def test_invalid_inputs_and_ambiguous_configuration_refused(self):
        f, z = self.frequency, self.spectra["RC"]
        for frequency, impedance, kwargs in ((f[:10], z[:10], {}), (f.reshape(-1, 1), z, {}),
                                             (-f, z, {}), (f, z * np.nan, {}),
                                             (f, z, {"lambda_min": .1, "lambda_max": .1}),
                                             (f, z, {"folds": 2.5}), (f, z, {"num_tau": 40.5}),
                                             (f, z, {"lambda_points": 2}), (f, z, {"derivative_order": 0})):
            with self.subTest(kwargs=kwargs, shape=frequency.shape):
                with self.assertRaises(ValueError):
                    predictive_rc_kernel_competition(frequency, impedance, **kwargs)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Projection mathematics and consumed-development counterexamples, not holdout certification."""
from __future__ import annotations

import json
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import svd

from loewner_validation import (
    LEGACY_PROJECTION, LOCAL_PROJECTION, _descriptor_poles_residues,
    _loewner_matrices_all_points, _reduce_svd_right_vectors,
    _reliability_gate, fit_rational, pole_audit, relative_rms, validate_loewner,
)


def fixtures():
    f = np.logspace(4., -2., 50)
    rc = 2. + 30. / (1. + 2j*np.pi*f*.03)
    rng = np.random.default_rng(101)
    noisy_rc = rc + .02 * (rng.normal(size=f.size) + 1j*rng.normal(size=f.size))
    fr = np.logspace(3., -2., 70)
    s = 2j*np.pi*fr
    # Seed 1701 was consumed by first-round validation and is now explicitly a
    # regression fixture. It is NOT an unseen validation result for candidate2.
    fc = np.geomspace(1e5, 1e-2, 51)
    clean = 1. + 8. / (1. + (2j*np.pi*fc*.01)**.8)
    rng_consumed = np.random.default_rng(1701)
    consumed = clean + .003*abs(clean)*(rng_consumed.normal(size=51)+1j*rng_consumed.normal(size=51))
    values = {
        "exact-RC": (f, rc),
        "exact-RC-plus-series-L": (f, rc+2j*np.pi*f*1e-4),
        "exact-RC-plus-RL": (f, rc+4.*(2j*np.pi*f*.2)/(1.+2j*np.pi*f*.2)),
        "exact-RLC-resonance": (fr, 2.+300.*s/(s*s+3.*s+10000.)),
        "development-noisy-RC-seed-101": (f, noisy_rc),
        "consumed-round1-ZARC-seed-1701": (fc, consumed),
    }
    from peak_evidence_test import zarc
    broad = zarc(fc, [(8., .01, .65), (7., .04, .65)])
    rng_broad = np.random.default_rng(3714)
    broad += .003*abs(broad)*(rng_broad.normal(size=51)+1j*rng_broad.normal(size=51))
    fl = np.logspace(5., -3., 61)
    series_l = .8+3./(1.+2j*np.pi*fl/(2*np.pi*30.))+2j*np.pi*fl*2e-6
    rng_l = np.random.default_rng(3723)
    series_l += .003*abs(series_l)*(rng_l.normal(size=61)+1j*rng_l.normal(size=61))
    values["consumed-round2-overlap-3714"] = (fc, broad)
    values["consumed-round2-series-L-3723"] = (fl, series_l)
    return values


class LoewnerProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixtures()
        cls.results = {}
        cls.seconds = {}
        for name, (f, z) in cls.inputs.items():
            started = time.monotonic()
            cls.results[name] = validate_loewner(f, z)
            cls.seconds[name] = time.monotonic()-started

    def test_truncated_complex_pencil_uses_Vh_conjugate_transpose(self):
        # Full-rank transfer equivalence cannot expose a wrong truncated basis:
        # both V and Vh are invertible at full rank. Check rank-k identities.
        rng = np.random.default_rng(101)
        L = rng.normal(size=(7, 6)) + 1j*rng.normal(size=(7, 6))
        Ls = rng.normal(size=(7, 6)) + 1j*rng.normal(size=(7, 6))
        v = rng.normal(size=7) + 1j*rng.normal(size=7)
        w = rng.normal(size=6) + 1j*rng.normal(size=6)
        k = 3
        U, _, _ = svd(np.hstack((L, Ls)), full_matrices=False)
        _, singular, Vh = svd(np.vstack((L, Ls)), full_matrices=False)
        Y, X = U[:, :k], Vh.conj().T[:, :k]
        E, A, B, C, _ = _reduce_svd_right_vectors(L, Ls, v, w, k)
        for actual, expected in ((E, -Y.conj().T@L@X), (A, -Y.conj().T@Ls@X),
                                 (B, Y.conj().T@v), (C, w.T@X)):
            np.testing.assert_allclose(actual, expected, atol=2e-13)
        stacked = np.vstack((L, Ls))
        optimal_error = np.sqrt(np.sum(singular[k:]**2))
        self.assertAlmostEqual(np.linalg.norm(stacked-stacked@X@X.conj().T), optimal_error, places=12)
        wrong_X = Vh[:, :k]
        self.assertGreater(np.linalg.norm(stacked-stacked@wrong_X@wrong_X.conj().T), optimal_error*1.05)

    def test_even_point_matrices_match_original_convention(self):
        from pyimpspec.analysis.drt.lm import _generate_loewner_matrices
        f, z = self.inputs["exact-RC"]
        local = _loewner_matrices_all_points(2j*np.pi*f, z)
        original = _generate_loewner_matrices(2j*np.pi*f, z)
        for a, b in zip(local, original):
            np.testing.assert_allclose(a, b, atol=1e-13)

    def test_odd_point_pencil_keeps_highest_endpoint(self):
        f, z = self.inputs["consumed-round1-ZARC-seed-1701"]
        order = np.argsort(f)
        s, z = 2j*np.pi*f[order], z[order]
        mats = _loewner_matrices_all_points(s, z)
        self.assertEqual(mats[0].shape, (50, 52))
        altered = z.copy()
        altered[-1] += .3+.1j
        changed = _loewner_matrices_all_points(s, altered)
        self.assertGreater(np.linalg.norm(mats[0]-changed[0]), 0.)
        self.assertGreater(np.linalg.norm(mats[3]-changed[3]), .1)
        result = self.results["consumed-round1-ZARC-seed-1701"]
        self.assertTrue(result["fit_uses_all_training_points"])
        for candidate in result["candidates"]:
            for fold in candidate.get("folds", []):
                self.assertEqual(fold["fit_input_point_count"], len(fold["train_indices"]))
                self.assertIn(0, fold["train_indices"])
                self.assertIn(len(f)-1, fold["train_indices"])
                self.assertFalse(set(fold["train_indices"]) & set(fold["test_indices"]))

    def test_descriptor_constant_part_does_not_require_invertible_E(self):
        E, A = np.diag([1., 0.]), np.diag([-2., -1.])
        B, C = np.array([1., 1.]), np.array([3., .7])
        poles, residues = _descriptor_poles_residues(E, A, B, C)
        finite = np.isfinite(poles)
        np.testing.assert_allclose(poles[finite], [-2.])
        np.testing.assert_allclose(residues[finite], [3.])
        self.assertEqual(np.isinf(poles).sum(), 1)
        s = 2j*np.pi*np.array([.1, 1., 10.])
        direct = np.array([C@np.linalg.solve(value*E-A, B) for value in s])
        np.testing.assert_allclose(direct, .7+residues[finite][0]/(s-poles[finite][0]))

    def test_indeterminate_descriptor_is_not_positive_kernel_evidence(self):
        poles, residues = _descriptor_poles_residues(np.diag([1., 0.]), np.diag([-2., 0.]),
                                                   np.ones(2), np.ones(2))
        audit = pole_audit(poles, residues, np.array([.1, 1.]), np.ones(2))
        gate = _reliability_gate(0., 0., audit)
        self.assertFalse(gate["kernel_evidence_reliable"])
        self.assertIn("singular-pencil-indeterminate-pole", gate["reliability_reasons"])

    def test_noiseless_RC_is_low_order_accurate_and_real_pole_consistent(self):
        result = self.results["exact-RC"]
        self.assertEqual(result["status"], "completed")
        self.assertLess(result["heldout_rms_percent"], 1e-6)
        self.assertLess(result["training_rms_percent"], 1e-6)
        self.assertLessEqual(result["selected_order"], 4)
        self.assertTrue(result["kernel_evidence_reliable"], result["reliability_reasons"])
        self.assertEqual(result["kernel_compatibility"], "real-pole-consistent")

    def test_noiseless_series_L_retains_predictive_accuracy(self):
        result = self.results["exact-RC-plus-series-L"]
        self.assertEqual(result["status"], "completed")
        self.assertLess(result["heldout_rms_percent"], .002)
        self.assertLess(result["training_rms_percent"], .002)
        self.assertNotEqual(result["kernel_compatibility"], "complex-or-unstable-pole-response")

    def test_noiseless_resonance_keeps_reliable_complex_poles(self):
        result = self.results["exact-RLC-resonance"]
        self.assertLess(result["heldout_rms_percent"], 1e-5)
        self.assertLess(result["training_rms_percent"], 1e-5)
        self.assertTrue(result["kernel_evidence_reliable"], result["reliability_reasons"])
        self.assertEqual(result["kernel_compatibility"], "complex-or-unstable-pole-response")

    def test_development_noise_uses_only_refit_audited_candidates(self):
        result = self.results["development-noisy-RC-seed-101"]
        self.assertEqual(result["status"], "completed")
        self.assertLessEqual(result["selected_order"], 32)
        self.assertLess(result["heldout_rms_percent"], 2.)
        self.assertLess(result["training_rms_percent"], 2.)
        self.assertTrue(result["selection_uses_full_data_refit_audit"])
        selected = [r for r in result["candidates"] if r["order"] == result["selected_order"]
                    and r["frequency_reference_quantile"] == result["selected_frequency_reference_quantile"]][0]
        self.assertTrue(selected["eligible"], selected)
        self.assertLessEqual(result["training_rms_percent"], 2.*result["heldout_rms_percent"]+.05)
        self.assertGreaterEqual(len(result["candidates"]), len(result["order_candidates"]))

    def test_consumed_1701_refit_collapse_is_removed_not_relabelled_as_holdout(self):
        result = self.results["consumed-round1-ZARC-seed-1701"]
        self.assertLess(result["selected_order"], 32)
        self.assertLess(result["heldout_rms_percent"], 2.)
        self.assertLess(result["training_rms_percent"], 2.)
        self.assertLessEqual(result["training_rms_percent"], 2.*result["heldout_rms_percent"]+.05)

    def test_poor_full_refit_cannot_inherit_good_holdout_pole_claim(self):
        f, z = self.inputs["exact-RC"]
        def fake_fit(training_f, training_z, order, query_f, **kwargs):
            pred = 2.+30./(1.+2j*np.pi*np.asarray(query_f)*.03)
            if len(training_f) == len(f):
                pred = pred*1.2
            return pred, np.array([-1./.03]), np.array([1000.])
        with patch("loewner_validation.fit_rational", side_effect=fake_fit):
            result = validate_loewner(f, z, orders=(2,))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["kernel_evidence_reliable"])
        self.assertEqual(result["kernel_compatibility"], "not-established")
        self.assertEqual(result["raw_pole_classification"], "real-pole-consistent")
        self.assertIn("full-refit-worse-than-heldout-envelope", result["reliability_reasons"])
        self.assertTrue(result["poles"])

    def test_poor_holdout_also_blocks_positive_kernel_claim(self):
        audit = pole_audit(np.array([-1.]), np.array([3.]), np.array([.1, 1.]), np.ones(2))
        result = _reliability_gate(3., .1, audit)
        self.assertFalse(result["kernel_evidence_reliable"])

    def test_best_cv_bad_refit_does_not_hide_next_eligible_order(self):
        f, z = self.inputs["exact-RC"]
        def fake_fit(training_f, training_z, order, query_f, **kwargs):
            pred = 2.+30./(1.+2j*np.pi*np.asarray(query_f)*.03)
            if order == 2 and len(training_f) == len(f):
                pred *= 1.2
            elif order == 4:
                pred *= 1.001
            return pred, np.array([-1./.03]), np.array([1000.])
        with patch("loewner_validation.fit_rational", side_effect=fake_fit):
            result = validate_loewner(f, z, orders=(2,4), reference_quantiles=(.5,))
        self.assertEqual(result["selected_order"], 4)
        self.assertTrue(result["kernel_evidence_reliable"])
        rejected = next(r for r in result["candidates"] if r["order"] == 2)
        self.assertFalse(rejected["eligible"])
        self.assertAlmostEqual(rejected["training_rms_percent"], 20.)

    def test_consumed_round2_refit_counterexamples_no_longer_collapse(self):
        for name in ("consumed-round2-overlap-3714", "consumed-round2-series-L-3723"):
            result = self.results[name]
            self.assertEqual(result["status"], "completed")
            self.assertLess(result["heldout_rms_percent"], 2., name)
            self.assertLess(result["training_rms_percent"], 2., name)
            self.assertTrue(result["kernel_evidence_reliable"], name)
            self.assertTrue(any(not row["eligible"] for row in result["candidates"]), name)

    def test_frequency_scale_covariance_in_noisy_refit_and_selection(self):
        for name in ("development-noisy-RC-seed-101", "consumed-round2-series-L-3723"):
            f, z = self.inputs[name]
            reference = self.results[name]
            for scalar in (1e-6, 1e6):
                result = validate_loewner(f*scalar, z)
                self.assertEqual(result["selected_order"], reference["selected_order"])
                self.assertEqual(result["selected_frequency_reference_quantile"], reference["selected_frequency_reference_quantile"])
                self.assertEqual(result["kernel_evidence_reliable"], reference["kernel_evidence_reliable"])
                self.assertAlmostEqual(result["heldout_rms_percent"], reference["heldout_rms_percent"], places=5)
                self.assertAlmostEqual(result["training_rms_percent"], reference["training_rms_percent"], places=5)

    def test_RC_RL_poles_and_residues_restore_frequency_and_impedance_units(self):
        f, z = self.inputs["exact-RC-plus-RL"]
        for fscale, zscale in ((1., 1.), (1e-6, 1e4), (1e6, 1e-4)):
            pred, poles, residues = fit_rational(f*fscale, z*zscale, 4, f*fscale)
            np.testing.assert_allclose(pred/zscale, z, rtol=1e-7, atol=1e-8)
            for expected_p, expected_r in ((-1./.03, 30./.03), (-1./.2, -4./.2)):
                distance = np.full(poles.shape, np.inf)
                finite = np.isfinite(poles)
                distance[finite] = abs(poles[finite]/fscale-expected_p)
                i = int(np.argmin(distance))
                self.assertAlmostEqual(float((poles[i]/fscale).real), expected_p, places=5)
                np.testing.assert_allclose(residues[i]/(fscale*zscale), expected_r, rtol=1e-6, atol=1e-6)

    def test_pure_R_and_C_degenerate_pencil_members_are_valid_predictions(self):
        f = np.geomspace(1e4, 1e-2, 31)
        for z in (np.full(f.size, 2., complex), 1./(2j*np.pi*f*.01)):
            pred, _, _ = fit_rational(f, z, 1, f)
            np.testing.assert_allclose(pred, z, rtol=1e-8, atol=1e-9)

    def test_exported_original_model_remains_legacy_unscaled_identity(self):
        from advanced_drt import loewner_rc_rl_analysis
        from pyimpspec import DataSet
        f, z = self.inputs["development-noisy-RC-seed-101"]
        exported = loewner_rc_rl_analysis(DataSet(f, z), num_procs=1)
        self.assertEqual(exported["status"], "completed")
        result = validate_loewner(f, z, exported_model=exported)
        audit = result["exported_model_audit"]
        self.assertEqual(result["projection"], LOCAL_PROJECTION)
        self.assertEqual(audit["status"], "completed", audit)
        self.assertEqual(audit["projection"], LEGACY_PROJECTION)
        self.assertFalse(audit["impedance_normalized"])
        self.assertFalse(audit["heldout_validated"])
        self.assertLess(audit["recreation_rms_percent"], 1e-4)

    def test_scale_covariance_and_input_arrays_preserved(self):
        f, z = self.inputs["development-noisy-RC-seed-101"]
        before_f, before_z = f.copy(), z.copy()
        reference = self.results["development-noisy-RC-seed-101"]
        for scalar in (1e-4, 1e4):
            result = validate_loewner(f, z*scalar)
            self.assertEqual(result["selected_order"], reference["selected_order"])
            self.assertEqual(result["kernel_compatibility"], reference["kernel_compatibility"])
            self.assertAlmostEqual(result["heldout_rms_percent"], reference["heldout_rms_percent"], places=6)
        np.testing.assert_array_equal(f, before_f)
        np.testing.assert_array_equal(z, before_z)


if __name__ == "__main__":
    suite = unittest.main(verbosity=2, exit=False)
    print(json.dumps({"development_benchmarks": [dict(
        name=name, elapsed_seconds=LoewnerProjectionTests.seconds[name],
        **{key:result.get(key) for key in ("selected_order", "selected_frequency_reference_quantile", "heldout_rms_percent",
           "training_rms_percent", "kernel_compatibility", "raw_pole_classification",
           "kernel_evidence_reliable", "reliability_status", "reliability_reasons")})
        for name, result in LoewnerProjectionTests.results.items()],
        "boundary": "Seed 101 is development; 1701,3714,3723 are already-consumed counterexamples. No new heldout noise seeds used."}, indent=2))
    raise SystemExit(not suite.result.wasSuccessful())

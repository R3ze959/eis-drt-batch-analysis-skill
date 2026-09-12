#!/usr/bin/env python3
"""Independent developer-review counterexamples; synthetic fixtures only.

Locks model/audit identity, the acceptance boundary, and recoverable rendering.
No public holdout or private experimental inputs are read.
"""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pyimpspec import DataSet

from advanced_drt import loewner_rc_rl_analysis
from batch_drt import run_render_isolated, write_json
from control_validation import classify_results
from loewner_validation import fit_rational, pole_audit, validate_loewner, LEGACY_PROJECTION
from model_recommendation import recommend_model
from result_contract import export_bundle, numerical_digest


class LoewnerModelIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frequency = np.logspace(4, -2, 50)
        rng = np.random.default_rng(101)
        cls.impedance = (2 + 30 / (1 + 2j * np.pi * cls.frequency * .03)
                         + .02 * (rng.normal(size=50) + 1j * rng.normal(size=50)))
        cls.exported = loewner_rc_rl_analysis(DataSet(cls.frequency, cls.impedance), 1)
        cls.validation = validate_loewner(cls.frequency, cls.impedance,
                                         orders=(2, 4, 6, 8, 12, 16), exported_model=cls.exported)

    def test_full_rank_weights_and_reduced_cv_have_distinct_audit_targets(self):
        original, result = self.exported, self.validation
        self.assertEqual(original["status"], "completed")
        self.assertEqual(result["status"], "completed")
        actual = result["exported_model_audit"]
        self.assertEqual(actual["status"], "completed")
        self.assertEqual(actual["model_order"], original["model_order"])
        self.assertGreater(actual["model_order"], result["selected_order"])
        self.assertNotEqual(result["validation_target"], actual["validation_target"])
        self.assertEqual(actual["validation_target"], "exported-full-data-matrix-rank-model")
        self.assertFalse(actual["heldout_validated"])
        self.assertLess(actual["recreation_rms_percent"], 1e-4)
        self.assertEqual(len(actual["poles"]), original["model_order"])
        self.assertEqual(len(result["poles"]), result["selected_order"])

    def test_exported_poles_are_actually_audited_not_copied_from_cv(self):
        _, poles, residues = fit_rational(self.frequency, self.impedance,
                                          self.exported["model_order"], self.frequency, normalize=False,
                                          projection=LEGACY_PROJECTION)
        independent = pole_audit(poles, residues, self.frequency, self.impedance)
        actual = self.validation["exported_model_audit"]
        self.assertEqual(actual["kernel_compatibility"], independent["kernel_compatibility"])
        np.testing.assert_allclose([r["real_per_s"] for r in actual["poles"]], poles.real)
        np.testing.assert_allclose([r["imag_per_s"] for r in actual["poles"]], poles.imag)
        # Original unscaled full-rank and normalized reduced fits can share the
        # aggregate label while having very different poles. Require the actual
        # pole inventory, not a conveniently different physical-sounding label.
        self.assertFalse(actual["heldout_validated"])
        self.assertEqual(actual["validation_target"], "exported-full-data-matrix-rank-model")

    def test_audit_is_refused_when_export_reconstruction_does_not_match(self):
        altered = copy.deepcopy(self.exported)
        altered["reconstructed"] = np.asarray(altered["reconstructed"]) + 1
        result = validate_loewner(self.frequency, self.impedance, orders=(2,), exported_model=altered)
        self.assertEqual(result["status"], "completed")
        actual = result["exported_model_audit"]
        self.assertEqual(actual["status"], "failed")
        self.assertFalse(actual["heldout_validated"])
        self.assertNotIn("poles", actual)
        self.assertIn("not transferable", actual["error"])

    def test_series_inductance_far_poles_are_not_unstable_measured_dynamics(self):
        f = self.frequency
        s = 2j * np.pi * f
        result = validate_loewner(f, 2 + 30 / (1 + s * .03) + s * 1e-4)
        self.assertEqual(result["status"], "completed")
        self.assertLess(result["heldout_rms_percent"], .002)
        self.assertNotEqual(result["kernel_compatibility"], "complex-or-unstable-pole-response")
        # A correct descriptor can now represent L with poles at infinity rather
        # than reproducing the old cancelling finite pair. Test that historical
        # finite-pair guard directly instead of requiring the improved fit to
        # generate an old numerical artifact.
        audit = pole_audit(np.array([6.75e6, -6.61e6, -1/.03], complex),
                           np.array([1e9, -1e9, 1000], complex), f,
                           2 + 30/(1+s*.03) + s*1e-4)
        self.assertNotEqual(audit["kernel_compatibility"], "complex-or-unstable-pole-response")
        nuisance = [r for r in audit["poles"] if r.get("material") and not r.get("within_expanded_frequency_window")]
        self.assertTrue(nuisance, "The analytic L fixture must exercise the outside-window safeguard")
        self.assertTrue(all(r["pole_role"] == "outside-window-or-nuisance-asymptote" for r in nuisance))


def numerical_result():
    measured = np.array([3-1j, 5-2j, 8-1j])
    return {"status": "completed", "spectrum_uid": "review-one", "source_uid": "source-one",
            "source": "/synthetic/review.csv", "source_sha256": "synthetic-hash",
            "spectrum_id": "review-one", "label": "review", "result_slug": "review-one",
            "impedance_basis": "ohm", "metadata": {"data_origin": "synthetic"},
            "drt": {"residual_rms_re_percent": .4, "residual_rms_im_percent": .3, "lambda": 1e-3},
            "kk": {"status": "screen-pass"}, "peaks": [],
            "curves": {"frequency_hz": [100., 10., 1.], "acquisition_index": [1, 2, 3],
                       "zreal_measured": measured.real.tolist(), "neg_zimag_measured": (-measured.imag).tolist(),
                       "zreal_reconstructed": (measured.real * 1.005).tolist(),
                       "neg_zimag_reconstructed": (-measured.imag * 1.005).tolist(),
                       "tau_s": [.01, .1], "gamma": [1., 2.]}}


def classify(result, evidence_status="supported"):
    state = "supported" if evidence_status == "supported" else "evidence-missing"
    validation = {"per_spectrum": [{"spectrum_uid": result["spectrum_uid"],
                    "evidence_status": evidence_status,
                    "controls": {key: state for key in ("repeat", "linearity", "stationarity")}}]}
    classify_results([result], validation)
    return validation


def attach_recommendation(result):
    recommendation = recommend_model(result)
    result.update(model_recommendation=recommendation,
                  recommended_branch=recommendation["recommended_branch"],
                  recommendation_status=recommendation["status"],
                  recommended_rms_percent=recommendation["recommended_rms_percent"])


class RecommendationAcceptanceTests(unittest.TestCase):
    def incompatible(self):
        result = numerical_result()
        result["loewner_validation"] = {"status": "completed",
            "kernel_compatibility": "complex-or-unstable-pole-response", "heldout_rms_percent": .01,
            "kernel_evidence_reliable": True}
        attach_recommendation(result)
        return result

    def test_reliable_rational_counterevidence_prevents_control_acceptance(self):
        result = self.incompatible()
        self.assertEqual(result["recommendation_status"], "exploratory-review-required")
        validation = classify(result)
        self.assertEqual(result["numeric_status"], "review-required")
        self.assertEqual(result["validity_status"], "review-required")
        self.assertEqual(result["display_role"], "diagnostic-only")
        self.assertEqual(validation["overall_verdict"], "review-required")
        self.assertIn("predictive-in-band-poles-not-RC-RL-compatible", result["decision_basis"])

    def test_unreliable_rational_poles_do_not_trigger_kernel_counterevidence(self):
        result = self.incompatible()
        result["loewner_validation"]["kernel_evidence_reliable"] = False
        attach_recommendation(result)
        self.assertNotIn("better-predicting-rational-model-has-non-RC-RL-poles",
                         result["model_recommendation"]["reasons"])
        classify(result)
        self.assertNotIn("predictive-in-band-poles-not-RC-RL-compatible", result["decision_basis"])
        self.assertEqual(result["numeric_status"], "supported")
        self.assertEqual(result["validity_status"], "accepted-with-supplied-controls")

    def test_generic_recommendation_warning_is_not_a_blanket_veto(self):
        result = numerical_result()
        result["model_recommendation"] = {"status": "exploratory-review-required",
                                           "reasons": ["signed-distribution-not-stable"]}
        classify(result)
        self.assertEqual(result["numeric_status"], "supported")
        self.assertEqual(result["validity_status"], "accepted-with-supplied-controls")

    def test_positive_recommendation_does_not_supply_missing_controls(self):
        result = numerical_result()
        attach_recommendation(result)
        self.assertEqual(result["recommendation_status"], "exploratory-numerical-candidate")
        classify(result, evidence_status="evidence-missing")
        self.assertEqual(result["validity_status"], "controls-incomplete-no-acceptance")
        self.assertNotEqual(result["display_role"], "included")

    def test_counterevidence_state_reaches_index_recommended_curves_and_figure_category(self):
        result = self.incompatible()
        classify(result)
        before = copy.deepcopy(result["curves"])
        with tempfile.TemporaryDirectory(prefix="drt-review-export-") as directory:
            out = Path(directory) / "output"
            out.mkdir()
            (out / "execution_contract.json").write_text("{}", encoding="utf-8")
            figure = Path(directory) / "synthetic.svg"
            figure.write_text("<svg>synthetic review fixture</svg>", encoding="utf-8")
            result["figures"] = [str(figure)]
            export_bundle(out, [result], [], [], {}, {})
            for name in ("00_overview/spectrum_index.csv", "03_results/recommended/curves.csv"):
                with (out / name).open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertTrue(rows)
                self.assertTrue(all(r["display_role"] == "diagnostic-only" for r in rows))
                self.assertTrue(all(r["validity_status"] == "review-required" for r in rows))
            self.assertTrue((out / "05_figures/diagnostic/synthetic.svg").is_file())
            self.assertFalse((out / "05_figures/scientific/synthetic.svg").exists())
        self.assertEqual(result["curves"], before)


class RenderingRecoveryTests(unittest.TestCase):
    def assert_preserved(self, original, returned, path):
        disk = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(returned["status"], "completed")
        self.assertEqual(returned["figure_status"], "failed")
        self.assertEqual(returned["curves"], original["curves"])
        self.assertEqual(disk["curves"], original["curves"])
        self.assertEqual(numerical_digest(returned), numerical_digest(original))
        self.assertTrue(returned["figure_errors"])

    def test_process_start_error_preserves_numerical_checkpoint_and_allows_export(self):
        result = numerical_result()
        original = copy.deepcopy(result)
        with tempfile.TemporaryDirectory(prefix="drt-review-render-") as directory, patch("batch_drt.mp.get_context") as context:
            root = Path(directory)
            path = root / "result.json"
            context.return_value.Process.return_value.start.side_effect = OSError("synthetic resource exhaustion")
            returned = run_render_isolated(result, root, path, 7)
            self.assert_preserved(original, returned, path)
            classify(returned, evidence_status="evidence-missing")
            output = root / "export"
            output.mkdir()
            (output / "execution_contract.json").write_text("{}", encoding="utf-8")
            export_bundle(output, [returned], [], [], {}, {})
            self.assertTrue((output / "00_overview/spectrum_index.csv").is_file())

    def test_unreadable_renderer_output_restores_checkpoint(self):
        result = numerical_result()
        original = copy.deepcopy(result)
        with tempfile.TemporaryDirectory(prefix="drt-review-render-") as directory, patch("batch_drt.mp.get_context") as context:
            root = Path(directory)
            path = root / "result.json"
            process = context.return_value.Process.return_value
            process.is_alive.return_value = False
            process.exitcode = 0
            process.join.side_effect = lambda timeout: path.write_text("{invalid JSON", encoding="utf-8")
            returned = run_render_isolated(result, root, path, 7)
            self.assert_preserved(original, returned, path)

    def test_renderer_cannot_change_numerical_arrays(self):
        result = numerical_result()
        original = copy.deepcopy(result)
        altered = copy.deepcopy(result)
        altered["curves"]["gamma"][0] = 999
        with tempfile.TemporaryDirectory(prefix="drt-review-render-") as directory, patch("batch_drt.mp.get_context") as context:
            root = Path(directory)
            path = root / "result.json"
            process = context.return_value.Process.return_value
            process.is_alive.return_value = False
            process.exitcode = 0
            process.join.side_effect = lambda timeout: write_json(path, altered)
            returned = run_render_isolated(result, root, path, 7)
            self.assert_preserved(original, returned, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

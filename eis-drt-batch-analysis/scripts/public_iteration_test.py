"""Regression counterexamples from the public-data development round."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from loewner_validation import validate_loewner, pole_audit
from model_recommendation import recommend_model
from batch_drt import source_parser_options, run_render_isolated


class IterationTests(unittest.TestCase):
    def test_rc_predictive_model(self):
        f = np.logspace(4, -2, 50)
        z = 2 + 30 / (1+2j*np.pi*f*.03)
        a = validate_loewner(f, z)
        self.assertEqual(a["status"], "completed")
        self.assertLess(a["heldout_rms_percent"], 1e-3)
        self.assertLessEqual(a["selected_order"], 4)
        for row in a["candidates"]:
            for fold in row.get("folds", []):
                self.assertFalse(set(fold["train_indices"]) & set(fold["test_indices"]))
                self.assertIn(0, fold["train_indices"])
                self.assertIn(len(f)-1, fold["train_indices"])

    def test_resonance_not_rc_poles(self):
        f = np.logspace(3, -2, 70)
        s = 2j*np.pi*f
        z = 2 + 300*s / (s*s+3*s+10000)
        a = validate_loewner(f,z)
        self.assertEqual(a["status"], "completed")
        self.assertLess(a["heldout_rms_percent"], 1e-5)
        self.assertEqual(a["kernel_compatibility"], "complex-or-unstable-pole-response")

    def test_noise_order_cap_and_scaling(self):
        f = np.logspace(4,-2,50)
        rng = np.random.default_rng(101)
        z = 2+30/(1+2j*np.pi*f*.03)+.02*(rng.normal(size=50)+1j*rng.normal(size=50))
        a,b = validate_loewner(f,z),validate_loewner(f,z*1e-4)
        self.assertLessEqual(a["selected_order"],32)
        self.assertEqual(a["selected_order"],b["selected_order"])
        self.assertAlmostEqual(a["heldout_rms_percent"],b["heldout_rms_percent"],places=5)
        self.assertGreater(a["heldout_rms_percent"],0)

    def test_series_l_asymptote_not_unstable_physics(self):
        f = np.logspace(4,-2,50)
        z = 2+30/(1+2j*np.pi*f*.03)+2j*np.pi*f*1e-4
        a = validate_loewner(f,z)
        self.assertEqual(a["status"],"completed")
        self.assertLess(a["heldout_rms_percent"],.002)
        self.assertNotEqual(a["kernel_compatibility"],"complex-or-unstable-pole-response")

    def test_invalid_not_perfect(self):
        self.assertEqual(validate_loewner([1]*20,[1j]*20)["status"],"not-assessable")

    def result(self, bad=True, sensitive=True):
        z = np.array([3-1j, 5-2j, 8-1j])
        primary = z + (4 if bad else .01)
        return {"curves":{"zreal_measured":z.real.tolist(),"neg_zimag_measured":(-z.imag).tolist(),
                          "zreal_reconstructed":primary.real.tolist(),"neg_zimag_reconstructed":(-primary.imag).tolist()},
                "kk":{"status":"screen-pass"},
                "signed_gdrt":{"status":"completed","reconstructed":z.tolist(),
                  "sensitivity":{"status":"sensitive" if sensitive else "stable"},
                  "support":{"supported_absolute_area_fraction":.99}}}

    def test_bad_rc_routes_without_hiding_sensitivity(self):
        r = self.result()
        before = copy.deepcopy(r)
        a = recommend_model(r)
        self.assertEqual(a["recommended_branch"],"signed_gdrt")
        self.assertEqual(a["status"],"exploratory-review-required")
        self.assertIn("signed-distribution-not-stable",a["reasons"])
        self.assertEqual(r,before)

    def test_good_rc_not_replaced_for_tiny_improvement(self):
        self.assertEqual(recommend_model(self.result(bad=False))["recommended_branch"],"primary_rc")

    def test_failed_signed_not_recommended(self):
        r = self.result()
        r["signed_gdrt"]["status"] = "failed"
        self.assertEqual(recommend_model(r)["recommended_branch"],"primary_rc")

    def test_manifest_boolean_mapping_no_leak(self):
        a = argparse.Namespace(text_headerless=False, text_columns=None)
        b = source_parser_options(a,{"text_headerless":"false","text_columns":'{"frequency":0,"zreal":1,"zimag":2}'})
        self.assertFalse(b.text_headerless)
        self.assertIsNone(a.text_columns)
        with self.assertRaises(ValueError):
            source_parser_options(a,{"text_headerless":"maybe"})

    def test_render_timeout_retains_checkpoint(self):
        r = {"status":"completed","curves":{"gamma":[1,2]},"numerical_sha256":"known"}
        with tempfile.TemporaryDirectory() as tmp, patch("batch_drt.mp.get_context") as context:
            process = context.return_value.Process.return_value
            process.is_alive.return_value = True
            path = Path(tmp)/"result.json"
            out = run_render_isolated(r,Path(tmp),path,7)
            process.join.assert_any_call(7)
            process.terminate.assert_called_once()
            self.assertEqual(out["status"],"completed")
            self.assertEqual(out["figure_status"],"timeout")
            self.assertEqual(json.loads(path.read_text())["curves"],{"gamma":[1,2]})


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Data-meaning counterexamples; original observations are never repaired to fit."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import batch_eis as e
import eis_parser as p
from result_contract import normalized_input_rows
from eis_branch_test import fixture


class UnitMeaning(unittest.TestCase):
    def test_MHz_not_mHz(self):
        self.assertEqual(p.frequency_scale("Frequency (MHz)", "auto")[0], 1e6)
        self.assertEqual(p.frequency_scale("Frequency (mHz)", "auto")[0], 1e-3)

    def test_megohm_not_milliohm(self):
        self.assertEqual(p.impedance_scale("Zreal (MΩ)", "auto"), (1e6, "ohm"))

    def test_microohm_not_ohm(self):
        self.assertEqual(p.impedance_scale("Zreal (µΩ)", "auto"), (1e-6, "ohm"))

    def test_prefixed_area_basis_preserved(self):
        self.assertEqual(p.impedance_scale("Zreal (mΩ cm²)", "auto"), (1e-3, "ohm_cm2"))

    def test_unsupported_prefix_not_silently_ohm(self):
        with self.assertRaises(ValueError):
            p.impedance_scale("Zreal (GΩ)", "auto")

    def test_explicit_sign_not_overridden_by_auxiliary_phase(self):
        result, _ = p.resolve_imaginary(np.ones(3), "Zimag (ohm)", "auto",
                                        modulus=np.full(3, 2), phase=np.full(3, -30))
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_no_phase_unit_guess_for_ambiguous_column(self):
        with self.assertRaises(ValueError):
            p.resolve_imaginary(np.ones(3), "component", "auto", modulus=np.full(3, 2), phase=np.full(3, -30))

    def test_unicode_minus_in_component_header(self):
        result, _ = p.resolve_imaginary(np.ones(3), "−Zimag (ohm)", "auto")
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_unit_separator_not_component_sign(self):
        result, _ = p.resolve_imaginary(np.ones(3), "Zimag (ohm-cm2)", "auto")
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_area_quotient_not_product(self):
        with self.assertRaises(ValueError):
            p.impedance_scale("Zreal (ohm/cm2)", "auto")

    def test_negative_component_aliases(self):
        for header in ("-Z_imag (ohm)", "-ImpedanceImaginary (ohm)", "−Z_imag (mΩ)"):
            with self.subTest(header=header):
                values, _ = p.resolve_imaginary(np.ones(3), header, "auto")
                np.testing.assert_array_equal(values, np.ones(3))

    def test_other_area_units_do_not_lose_dimension(self):
        for header in ("Zreal (ohm m²)", "Zreal (ohm/m²)", "Zreal (ohm mm2)"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                p.impedance_scale(header, "auto")


class PipelineSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eis-data-safety-")
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.csv"
        fixture(self.raw)
        self.out = self.root / "eis"

    def tearDown(self):
        self.temp.cleanup()

    def run_eis(self, *extra):
        return e.main([str(self.raw), "--output-dir", str(self.out), "--plots", "off", *extra])

    def test_handoff_can_export_normalized_data_not_just_reparse(self):
        self.run_eis()
        manifest = self.out / "06_reproducibility/drt_input_manifest.csv"
        args = e.parse_args([str(self.raw), "--output-dir", str(self.root / "later"), "--manifest", str(manifest)])
        exact, by_path = e.load_manifest(manifest)
        payloads, _, failures = e.acquire(args, [self.raw], exact, by_path, "test")
        self.assertEqual(failures, [])
        # This is the same normalized exporter that the DRT branch uses.
        rows = normalized_input_rows(payloads, [e.base_result(p) for p in payloads])
        self.assertEqual(len(rows), 32)
        self.assertEqual([r["source_line"] for r in rows], list(range(2, 34)))

    def test_good_kk_does_not_hide_bad_raw_row(self):
        with self.raw.open("a") as h:
            h.write("bad,2,3\n")
        with patch("batch_eis.isolated_check", return_value={"status": "completed", "screen_status": "screen-pass"}):
            self.run_eis("--kk", "on")
        result = json.loads(next((self.out / "results").glob("*.json")).read_text())
        self.assertEqual(result["numeric_status"], "review-required")
        self.assertEqual(result["input_quality_status"], "review-required")

    def test_area_input_not_normalized_twice(self):
        self.raw.write_text("Frequency (Hz),Zreal (mΩ cm²),Zimag (mΩ cm²)\n100,2,-3\n10,4,-5\n1,6,-7\n")
        manifest = self.root / "manifest.csv"
        manifest.write_text("path,area_cm2\nraw.csv,10\n")
        self.run_eis("--manifest", str(manifest), "--area-normalize")
        result = json.loads(next((self.out / "results").glob("*.json")).read_text())
        np.testing.assert_allclose(result["zreal"], [.002, .004, .006])
        self.assertEqual(result["impedance_basis"], "ohm_cm2")

    def test_manifest_cannot_override_parser_row_identity(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text('path,included_source_lines\nraw.csv,"[999]"\n')
        self.assertEqual(self.run_eis("--manifest", str(manifest)), 2)

    def test_changed_source_after_inventory_not_accepted(self):
        original = p.read_text
        def changed(*args):
            fixture(self.raw, scale=2)
            return original(*args)
        with patch("batch_eis.load_parser_module", return_value=p), patch.object(p, "read_text", side_effect=changed):
            self.assertEqual(self.run_eis(), 2)
        run = json.loads((self.out / "run_manifest.json").read_text())
        self.assertEqual(run["completed"], 0)

    def test_modulus_overflow_is_parse_failure_not_whole_batch_crash(self):
        self.raw.write_text("Frequency (Hz),Zreal (ohm),Zimag (ohm)\n10,1.7e308,1.7e308\n1,1.7e308,1.7e308\n")
        with np.errstate(over="ignore"):
            self.assertEqual(self.run_eis(), 2)
        self.assertTrue((self.out / "01_inputs/source_audits.json").exists())

    def test_nonmonotonic_acquisition_time_requires_review(self):
        self.raw.write_text("Frequency (Hz),Zreal (ohm),Zimag (ohm),Time (s)\n100,1,-1,10\n10,2,-2,20\n1,3,-3,15\n")
        self.run_eis()
        result = json.loads(next((self.out / "results").glob("*.json")).read_text())
        self.assertFalse(result["qc"]["time_monotonic_increasing"])
        self.assertEqual(result["qc"]["status"], "suspect")
        self.assertEqual(result["input_quality_status"], "review-required")


if __name__ == "__main__":
    unittest.main(verbosity=2)

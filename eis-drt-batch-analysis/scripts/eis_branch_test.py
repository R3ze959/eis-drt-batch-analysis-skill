#!/usr/bin/env python3
"""Synthetic EIS-only regression and real CLI tests; no private reference data."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import batch_eis as eis
from eis_plotting import display_scale, limits, make_figure, visual_qa, STYLE


def fixture(path, *, points=32, scale=1, inductance=0, reverse=False, zero=False):
    f = np.logspace(5, -2, points)
    z = scale * (2 + 15 / (1 + 2j * np.pi * f * .01) + 2j * np.pi * f * inductance)
    if zero:
        z[0] = 0
    if reverse:
        f, z = f[::-1], z[::-1]
    with path.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["Frequency (Hz)", "Zreal (ohm)", "Zimag (ohm)"])
        w.writerows(zip(f, z.real, z.imag))
    return f, z


class EISBranchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="eis-regression-")
        self.root = Path(self.temp.name)
        self.source = self.root / "raw.csv"
        self.f, self.z = fixture(self.source)
        self.output = self.root / "eis"

    def tearDown(self):
        self.temp.cleanup()

    def command(self, *extras):
        return [str(self.source), "--output-dir", str(self.output), "--plots", "off", *extras]

    def run_eis(self, *extras):
        return eis.main(self.command(*extras))

    def result(self):
        return json.loads(next((self.output / "results").glob("*.json")).read_text())

    def test_no_drt_or_circuit_solver_called(self):
        with patch("batch_drt.analyze_payload", side_effect=AssertionError("DRT forbidden")), \
             patch("batch_drt.main", side_effect=AssertionError("DRT main forbidden")), \
             patch("batch_eis.isolated_check", side_effect=AssertionError("Optional checks off")):
            self.assertEqual(self.run_eis(), 0)
        r = self.result()
        self.assertEqual(r["analysis_mode"], "eis-only")
        self.assertNotIn("drt", r)
        self.assertNotIn("tau_s", r)
        self.assertEqual(r["drt_status"], "not-requested")

    def test_quantities_source_rows_and_hash(self):
        sha = eis.sha256_file(self.source)
        self.run_eis()
        r = self.result()
        np.testing.assert_allclose(r["quantities"]["modulus"], abs(self.z))
        np.testing.assert_allclose(r["quantities"]["phase_deg"], np.angle(self.z, deg=True))
        self.assertEqual(r["source_sha256"], sha)
        self.assertEqual(eis.sha256_file(self.source), sha)
        with (self.output / "01_inputs/normalized_eis.csv").open(encoding="utf-8-sig") as h:
            rows = list(csv.DictReader(h))
        self.assertEqual(len(rows), len(self.f))
        self.assertEqual([int(r["source_line"]) for r in rows], list(range(2, len(self.f)+2)))

    def test_no_15_point_drt_gate_on_eis(self):
        fixture(self.source, points=5)
        self.assertEqual(self.run_eis(), 0)
        self.assertEqual(len(self.result()["frequency_hz"]), 5)

    def test_inductive_points_preserved(self):
        f, z = fixture(self.source, inductance=1e-4)
        self.run_eis()
        r = self.result()
        self.assertEqual(len(r["frequency_hz"]), len(f))
        np.testing.assert_allclose(r["neg_zimag"], -z.imag)
        self.assertGreater(r["qc"]["negative_neg_zimag_points"], 0)

    def test_ascending_acquisition_not_reordered(self):
        f, _ = fixture(self.source, reverse=True)
        self.run_eis()
        np.testing.assert_array_equal(self.result()["frequency_hz"], f)

    def test_zero_modulus_phase_undefined(self):
        fixture(self.source, zero=True)
        self.run_eis()
        r = self.result()
        self.assertIsNone(r["quantities"]["phase_deg"][0])
        self.assertFalse(r["quantities"]["phase_defined"][0])

    def test_unknown_units_fail_with_audit(self):
        self.source.write_text("freq;Z_prime;Z_double_prime\n1000;1;-2\n100;2;-3\n10;3;-1\n")
        self.assertEqual(self.run_eis(), 2)
        audits = json.loads((self.output / "01_inputs/source_audits.json").read_text())
        self.assertEqual(len(audits[0]["row_ledger"]), 4)
        self.assertFalse((self.output / "results").exists())

    def test_public_header_alias_with_explicit_units(self):
        self.source.write_text("freq;Z_prime;Z_double_prime\n1000;1;2\n100;2;-3\n10;3;-1\n")
        self.assertEqual(self.run_eis("--frequency-unit", "hz", "--impedance-unit", "ohm"), 0)
        self.assertEqual(self.result()["neg_zimag"], [-2, 3, 1])

    def test_area_normalization_and_handoff(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text("path,area_cm2\nraw.csv,2\n")
        self.run_eis("--manifest", str(manifest), "--area-normalize")
        r = self.result()
        self.assertEqual(r["impedance_basis"], "ohm_cm2")
        np.testing.assert_allclose(r["zreal"], self.z.real * 2)
        self.assertTrue(json.loads((self.output / "06_reproducibility/drt_handoff.json").read_text())["area_normalize"])

    def test_handoff_reimports_same_selected_raw_spectra(self):
        self.run_eis()
        manifest = self.output / "06_reproducibility/drt_input_manifest.csv"
        args = eis.parse_args([str(self.source), "--manifest", str(manifest), "--output-dir", str(self.root / "later")])
        exact, by_path = eis.load_manifest(manifest)
        payloads, _, failed = eis.acquire(args, [self.source], exact, by_path, "test")
        self.assertEqual(failed, [])
        self.assertEqual(len(payloads), 1)
        np.testing.assert_array_equal(payloads[0]["frequency_hz"], self.f)
        np.testing.assert_array_equal(payloads[0]["zreal"], self.z.real)

    def test_resume_export_only_does_not_check(self):
        self.run_eis()
        before = self.result()
        with patch("batch_eis.isolated_check", side_effect=AssertionError("No recalculation")):
            self.assertEqual(self.run_eis("--resume", "--export-only"), 0)
        self.assertEqual(self.result(), before)
        self.assertTrue((self.output / "_superseded").is_dir())

    def test_resume_changed_data_rejected(self):
        self.run_eis()
        fixture(self.source, scale=2)
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_eis("--resume")

    def test_checkpoint_tamper_rejected(self):
        self.run_eis()
        path = next((self.output / "results").glob("*.json"))
        value = self.result()
        value["quantities"]["phase_deg"][0] += 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "checkpoint changed"):
            self.run_eis("--resume")

    def test_wrong_mode_checkpoint_rejected(self):
        self.run_eis()
        path = next((self.output / "results").glob("*.json"))
        r = self.result()
        r["analysis_mode"] = "drt"
        r["eis_sha256"] = eis.result_digest(r)
        path.write_text(json.dumps(r))
        with self.assertRaises(ValueError):
            self.run_eis("--resume")

    def test_existing_directory_not_overwritten(self):
        self.output.mkdir()
        marker = self.output / "user.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_eis()
        self.assertEqual(marker.read_text(), "keep")

    def test_foreign_export_file_preserved(self):
        self.run_eis()
        marker = self.output / "04_plot_data/user.csv"
        marker.write_text("keep")
        self.run_eis("--resume", "--export-only")
        self.assertEqual(marker.read_text(), "keep")

    def test_requested_check_failure_keeps_data(self):
        with patch("batch_eis.isolated_check", return_value={"status": "failed", "error": "injected"}):
            self.assertEqual(self.run_eis("--kk", "on"), 2)
        self.assertEqual(self.result()["status"], "completed")
        self.assertTrue((self.output / "03_results/eis_quantities.csv").is_file())

    def test_kk_reconstruction_keeps_sorted_to_source_mapping(self):
        fixture(self.source, reverse=True, inductance=1e-5)
        self.run_eis()
        r = self.result()
        c = eis.calculate_check(r, "kk", 1.0, 3.0)
        self.assertEqual(c["status"], "completed")
        self.assertEqual(c["screen_status"], "screen-pass")
        np.testing.assert_array_equal(np.asarray(r["frequency_hz"])[np.asarray(c["source_point_indices"])-1], c["frequency_hz"])
        self.assertEqual(c["source_point_count"], c["screen_point_count"])
        self.assertTrue(c["add_inductance"])

    def test_short_or_duplicate_check_not_assessable(self):
        self.run_eis()
        r = self.result()
        r["frequency_hz"][1] = r["frequency_hz"][0]
        self.assertEqual(eis.calculate_check(r, "kk", 1.0, 3.0)["status"], "not-assessable")

    def test_no_spectra_always_nonzero_even_partial(self):
        self.source.write_text("freq;real;imag\n1;2;3\n")
        self.assertEqual(self.run_eis("--allow-partial"), 2)

    def test_plot_failure_keeps_checkpoint_and_exports(self):
        with patch("eis_plotting.render_isolated", return_value={"status": "failed", "error": "injected"}):
            code = eis.main([str(self.source), "--output-dir", str(self.output)])
        self.assertEqual(code, 2)
        self.assertEqual(self.result()["status"], "completed")
        self.assertTrue((self.output / "01_inputs/normalized_eis.csv").is_file())
        report = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(report["figure_report"]["error"], "injected")

    def test_outside_axis_tick_does_not_trigger_overlap(self):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator
        with plt.style.context(str(STYLE)):
            fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
            ax.set(xlim=(2.68, 27.32), ylim=(-40.61, 13.73))
            ax.xaxis.set_major_locator(FixedLocator([0, 10, 20, 30]))
            ax.yaxis.set_major_locator(FixedLocator([-45, -30, -15, 0, 15]))
            self.assertEqual(visual_qa(fig, ax)["text_overlap_pairs"], [])
            plt.close(fig)

    def test_custom_parser_identity_in_handoff(self):
        import shutil
        custom = self.root / "instrument_parser.py"
        shutil.copy2(eis.DEFAULT_PARSER, custom)
        self.run_eis("--parser-script", str(custom))
        handoff = json.loads((self.output / "06_reproducibility/drt_handoff.json").read_text())
        self.assertEqual(handoff["parser_script"], str(custom.resolve()))
        self.assertEqual(handoff["parser_sha256"], eis.sha256_file(custom))

    def test_narrow_bode_magnitude_has_numeric_ticks(self):
        import matplotlib.pyplot as plt
        self.run_eis()
        r = self.result()
        r["zreal"] = np.linspace(.04, .08, 32).tolist()
        r["neg_zimag"] = [0.001] * 32
        with plt.style.context(str(STYLE)):
            fig, info = make_figure(r, "bode_modulus")
            ax = fig.axes[0]
            lo, hi = ax.get_ylim()
            labels = [t.label1.get_text() for t in ax.yaxis.get_major_ticks() if lo <= t.get_loc() <= hi]
            self.assertGreaterEqual(len([s for s in labels if s]), 2)
            self.assertEqual(info["text_overlap_pairs"], [])
            plt.close(fig)

    def test_missing_controls_never_accepted(self):
        self.run_eis()
        self.assertEqual(self.result()["evidence_status"], "not-assessed")
        self.assertNotEqual(self.result()["validity_status"], "accepted")

    def test_milliohm_limits_no_ohm_padding(self):
        self.assertLess(np.ptp(limits([.001, .003])), .003)
        self.assertEqual(display_scale([.001], [-.002], "ohm"), (1000, "mΩ"))

    def test_plot_native_points_and_no_overlap(self):
        import matplotlib.pyplot as plt
        fixture(self.source, scale=.001, inductance=1e-4)
        self.run_eis()
        with plt.style.context(str(STYLE)):
            for kind in ("nyquist", "bode_modulus", "bode_phase"):
                fig, info = make_figure(self.result(), kind)
                self.assertEqual(info["plotted_points"], 32)
                self.assertEqual(info["text_overlap_pairs"], [])
                self.assertEqual(fig.axes[0].lines[0].get_linestyle(), "None")
                plt.close(fig)

    def test_spreadsheet_formula_escape_only_text(self):
        path = self.root / "safe.csv"
        eis.csv_table(path, [{"label": "=1+1", "number": -2.1}])
        with path.open(encoding="utf-8-sig") as h:
            row = next(csv.DictReader(h))
        self.assertEqual(row, {"label": "'=1+1", "number": "-2.1"})

    def test_real_cli_and_export_only(self):
        cmd = [sys.executable, str(Path(eis.__file__)), *self.command()]
        for extra in ([], ["--resume", "--export-only"]):
            run = subprocess.run(cmd + extra, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stderr)

    def test_cli_invalid_options(self):
        with self.assertRaises(SystemExit):
            eis.parse_args(self.command("--export-only"))
        with self.assertRaises(SystemExit):
            eis.parse_args(self.command("--check-warn-percent", "nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

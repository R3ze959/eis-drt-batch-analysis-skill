"""Release counterexamples: failure recovery, manifests and shared ingestion."""
import argparse
import csv
import json
from pathlib import Path
import sys
import importlib.metadata
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import batch_drt as drt
import batch_eis as eis
import eis_parser as parser
from eis_branch_test import fixture
import runtime_contract


class ReleaseCounterexamples(unittest.TestCase):
    def test_diagnostics_retained_but_trends_require_request(self):
        args = eis.parse_args(["raw.csv", "--output-dir", "out"])
        self.assertEqual(args.plots, "on")
        with patch.object(sys, "argv", ["batch_drt.py", "raw.csv", "--output-dir", "out"]):
            args = drt.parse_args()
        self.assertEqual(args.trend_plots, "off")
        self.assertFalse(hasattr(args, "plots"))  # No new diagnostic gate.

    def test_later_trend_request_preserves_numerical_resume_identity(self):
        from result_contract import execution_contract, initialize_run
        with patch.object(sys, "argv", ["batch_drt.py", str(self.raw), "--output-dir", str(self.out)]):
            args = drt.parse_args()
        first = execution_contract(args, [], Path(drt.__file__).parent)
        initialize_run(self.out, first)
        args.trend_plots, args.trend_dpi = "auto", 120
        second = execution_contract(args, [], Path(drt.__file__).parent)
        self.assertEqual(first["run_fingerprint"], second["run_fingerprint"])
        initialize_run(self.out, second, resume=True)
        args.fixed_lambda = 0.005
        changed = execution_contract(args, [], Path(drt.__file__).parent)
        self.assertNotEqual(first["run_fingerprint"], changed["run_fingerprint"])
        with self.assertRaises(ValueError):
            initialize_run(self.out, changed, resume=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prepublication-")
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.csv"
        fixture(self.raw)
        self.out = self.root / "out"

    def tearDown(self):
        self.temp.cleanup()

    def run_eis(self, *extra):
        return eis.main([str(self.raw), "--output-dir", str(self.out), "--plots", "off", *extra])

    def run_drt(self, *extra):
        argv = ["batch_drt.py", str(self.raw), "--output-dir", str(self.out),
                "--no-credible-intervals", "--trend-plots", "off", "--no-loewner",
                "--signed-gdrt", "off", "--ddt", "off", *extra]
        with patch.object(sys, "argv", argv):
            return drt.main()

    def read(self, name="run_manifest.json"):
        return json.loads((self.out / name).read_text())

    def test_final_eis_overview_is_same_as_root_and_hash_owned(self):
        with patch("eis_plotting.render_isolated", return_value={"status": "failed", "error": "synthetic failure"}):
            self.assertEqual(self.run_eis("--plots", "on"), 2)
        self.assertEqual(self.read(), self.read("00_overview/run_manifest.json"))
        owned = self.read("derived_eis_export_manifest.json")
        for name, sha in owned["files"].items():
            self.assertEqual(eis.sha256_file(self.out / name), sha, name)

    def test_failed_reexport_does_not_leave_completed_current_status(self):
        self.run_eis()
        before = next((self.out / "results").glob("*.json")).read_bytes()
        with patch("batch_eis.publish_tree", side_effect=OSError("synthetic export fault")):
            with self.assertRaises(OSError):
                self.run_eis("--resume", "--export-only")
        self.assertEqual(self.read()["export_status"], "failed")
        self.assertIn("synthetic export fault", self.read()["export_error"])
        self.assertEqual(next((self.out / "results").glob("*.json")).read_bytes(), before)
        self.assertFalse((self.out / ".eis-run.lock").exists())

    def test_unexpected_renderer_error_keeps_status_and_data(self):
        with patch("eis_plotting.render_isolated", side_effect=OSError("synthetic launch fault")):
            self.assertEqual(self.run_eis("--plots", "on"), 2)
        self.assertEqual(self.read()["figure_status"], "failed")
        self.assertEqual(self.read()["export_status"], "completed")
        self.assertEqual(self.read(), self.read("00_overview/run_manifest.json"))
        self.assertEqual(len(list((self.out / "results").glob("*.json"))), 1)

    def test_ordinary_resume_retries_failed_requested_check(self):
        with patch("batch_eis.isolated_check", return_value={"status": "timeout"}):
            self.assertEqual(self.run_eis("--kk", "on"), 2)
        with patch("batch_eis.isolated_check", return_value={"status": "completed", "screen_status": "screen-pass"}) as check:
            self.assertEqual(self.run_eis("--kk", "on", "--resume"), 0)
        check.assert_called_once()
        r = json.loads(next((self.out / "results").glob("*.json")).read_text())
        self.assertEqual(r["kk"]["status"], "completed")
        self.assertEqual(r["eis_sha256"], eis.result_digest(r))

    def test_export_only_never_retries_failed_diagnostic(self):
        with patch("batch_eis.isolated_check", return_value={"status": "timeout"}):
            self.run_eis("--kk", "on")
        original = next((self.out / "results").glob("*.json")).read_bytes()
        with patch("batch_eis.isolated_check", side_effect=AssertionError("No computation")):
            self.assertEqual(self.run_eis("--kk", "on", "--resume", "--export-only"), 2)
        self.assertEqual(next((self.out / "results").glob("*.json")).read_bytes(), original)

    def test_spectrum_row_cannot_silently_override_source_units(self):
        manifest = self.root / "manifest.csv"
        args = eis.parse_args([str(self.raw), "--output-dir", str(self.out)])
        spectra, _ = parser.read_text(self.raw, "raw", args)
        with manifest.open("w", newline="") as h:
            w = csv.writer(h)
            w.writerows([["path", "spectrum_id", "frequency_unit"], ["raw.csv", "", "hz"],
                         ["raw.csv", spectra[0].spectrum_id, "khz"]])
        self.assertEqual(self.run_eis("--manifest", str(manifest)), 2)
        self.assertEqual(self.read()["completed"], 0)

    def test_manifest_duplicate_column_not_silently_last_wins(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text("path,area_cm2,area_cm2\nraw.csv,1,100\n")
        with self.assertRaises(ValueError):
            drt.load_manifest(manifest)

    def test_manifest_ragged_extra_cell_not_silently_discarded(self):
        manifest = self.root / "manifest.csv"
        manifest.write_text("path,area_cm2\nraw.csv,1,unexpected\n")
        with self.assertRaises(ValueError):
            drt.load_manifest(manifest)

    def spectrum(self):
        args = eis.parse_args([str(self.raw), "--output-dir", str(self.out)])
        return parser.read_text(self.raw, "raw", args)[0][0]

    def test_shared_payload_rejects_parser_provenance_override(self):
        s = self.spectrum()
        with self.assertRaisesRegex(ValueError, "parser-owned"):
            drt.prepare_payload(s, "a" * 64, parser.qc_for(s), {"included_source_lines": "[999]"}, False)

    def test_shared_payload_rejects_nonfinite_after_area_conversion(self):
        s = self.spectrum()
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            drt.prepare_payload(s, "a" * 64, parser.qc_for(s), {"area_cm2": "1e308"}, True)

    def test_drt_source_is_atomic_if_later_spectrum_preparation_fails(self):
        self.raw.write_text("Frequency (Hz),Zreal (ohm),Zimag (ohm)\n100,1,-1\n10,2,-2\n1,3,-3\n100,1,-1\n10,2,-2\n1,3,-3\n")
        args = eis.parse_args([str(self.raw), "--output-dir", str(self.out)])
        spectra, _ = parser.read_text(self.raw, "raw", args)
        self.assertEqual(len(spectra), 2)
        manifest = self.root / "manifest.csv"
        with manifest.open("w", newline="") as h:
            w = csv.writer(h)
            w.writerows([["path", "spectrum_id", "area_cm2"], ["raw.csv", "", "2"],
                         ["raw.csv", spectra[1].spectrum_id, "0"]])
        self.assertEqual(self.run_drt("--manifest", str(manifest), "--area-normalize"), 2)
        self.assertEqual(self.read()["spectrum_count"], 0)
        self.assertEqual(self.read()["source_audits"][0]["selected_spectrum_count"], 0)

    def test_drt_changed_source_between_inventory_and_parse_not_accepted(self):
        fixture(self.raw, points=3)
        original = parser.read_text
        def changed(*args):
            fixture(self.raw, points=3, scale=2)
            return original(*args)
        with patch("batch_drt.load_parser_module", return_value=parser), patch.object(parser, "read_text", side_effect=changed):
            self.assertEqual(self.run_drt(), 2)
        self.assertEqual(self.read()["spectrum_count"], 0)

    def test_all_invalid_drt_has_final_failure_and_export_status(self):
        self.raw.write_text("Frequency (Hz),Zreal (ohm),Zimag (ohm)\n0,1,-1\n")
        self.assertEqual(self.run_drt(), 2)
        self.assertEqual(self.read()["status"], "failed")
        self.assertEqual(self.read()["export_status"], "completed")
        self.assertEqual(self.read(), self.read("00_overview/run_manifest.json"))

    def test_drt_nonfinite_threshold_is_configuration_error(self):
        with patch.object(sys, "argv", ["batch_drt.py", str(self.raw), "--output-dir", str(self.out), "--repeat-warn-percent", "nan"]):
            args = drt.parse_args()
        with self.assertRaises(ValueError):
            drt.validate_args(args)


class ConditionalRuntimeTests(unittest.TestCase):
    def environment(self, system):
        from packaging.markers import default_environment
        return {**default_environment(), "sys_platform": system, "extra": ""}

    def test_windows_timezone_dependency_is_exactly_pinned(self):
        self.assertEqual(runtime_contract.locked_versions(self.environment("win32"))["tzdata"], "2026.3")

    def test_nonwindows_does_not_require_unused_timezone_package(self):
        for system in ("darwin", "linux"):
            self.assertNotIn("tzdata", runtime_contract.locked_versions(self.environment(system)))

    def test_active_timezone_version_is_in_runtime_fingerprint_inputs(self):
        pins = runtime_contract.locked_versions(self.environment("win32"))
        original = importlib.metadata.version
        with patch.object(runtime_contract, "locked_versions", return_value=pins), patch.object(
                importlib.metadata, "version", side_effect=lambda name: "2026.3" if name == "tzdata" else original(name)):
            self.assertEqual(runtime_contract.actual_versions()["tzdata"], "2026.3")

    def test_installed_metadata_dependency_closure_for_platform_markers(self):
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        for system in ("darwin", "linux", "win32"):
            env = self.environment(system)
            pins = runtime_contract.locked_versions(env)
            for package in pins:
                # tzdata's official 2026.3 metadata declares no Requires-Dist;
                # its binary installation is not simulated on a macOS host.
                requires = [] if package == "tzdata" else (importlib.metadata.requires(package) or [])
                for raw in requires:
                    req = Requirement(raw)
                    if req.marker and not req.marker.evaluate(env):
                        continue
                    name = canonicalize_name(req.name)
                    self.assertIn(name, pins, (system, package, raw))
                    self.assertTrue(req.specifier.contains(pins[name]), (system, package, raw))

    def test_loose_or_duplicate_dependency_pins_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            lock = Path(temp) / "requirements.txt"
            for content in ("numpy>=2\n", "numpy==2.*\n", "numpy==2.4.6\nnumpy==2.4.6\n"):
                lock.write_text(content)
                with patch.object(runtime_contract, "LOCK", lock), self.assertRaises(ValueError):
                    runtime_contract.locked_versions()


if __name__ == "__main__":
    unittest.main(verbosity=2)

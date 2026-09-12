#!/usr/bin/env python3
"""Executable counterexamples from the beta.1 audit; synthetic inputs only."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import advanced_drt as advanced
import batch_drt
import bootstrap
import build_release
from control_validation import classify_results
from regression_test import F, spectrum
from release_test import fixture
from result_contract import export_bundle, export_wide, execution_contract
from plot_drt_trends import prepare_groups, render_trends

ROOT = Path(__file__).resolve().parents[1]


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class AcceptanceCounterexamples(unittest.TestCase):
    def test_large_series_resistance_does_not_hide_repeat_peak_shift(self):
        def row(name, peak, **metadata):
            r = spectrum(name, z=100 + 1/(1+1j*F/peak),
                         ac_amplitude_mv=metadata.pop("ac_amplitude_mv", 10),
                         rest_time_s=metadata.pop("rest_time_s", 1800), **metadata)
            r["peaks"] = [{"frequency_hz": peak, "area_impedance": 1., "relative_height": 1.}]
            return r
        rows = [row("r0", .1), row("r1", 1), row("r2", 10),
                row("amp", .1, ac_amplitude_mv=5),
                row("early", .1, rest_time_s=60), row("late", .1, rest_time_s=600)]
        validation = classify_results(rows, advanced.assess_batch_validity(rows))
        repeat = next(r for r in validation["repeat_assessments"] if "r0" in r["member_uids"])
        self.assertLess(max(p["complex_rms_percent"] for p in repeat["pair_assessments"]), .5)
        self.assertEqual(repeat["status"], "repeat-peak-disagreement")
        self.assertEqual(rows[0]["controls"]["repeat"], "contradicted")
        self.assertNotEqual(rows[0]["validity_status"], "accepted-with-supplied-controls")

    def test_minor_noise_peaks_do_not_reject_stable_major_peak(self):
        rows = [spectrum(str(i)) for i in range(3)]
        for i, r in enumerate(rows):
            r["peaks"].append({"frequency_hz": 10**(-1+i), "area_impedance": .01,
                               "relative_height": .01, "confidence": "minor-low-signal"})
        a = advanced.assess_batch_validity(rows)["repeat_assessments"][0]
        self.assertEqual(a["status"], "repeat-supported")

    def test_missing_peaks_are_not_repeat_support(self):
        rows = [spectrum(str(i)) for i in range(3)]
        for r in rows: r["peaks"] = []
        a = advanced.assess_batch_validity(rows)["repeat_assessments"][0]
        self.assertEqual(a["status"], "repeat-peak-not-assessable")

    def test_missing_or_nonfinite_rms_never_numeric_support(self):
        for value in (None, float("nan"), float("inf")):
            with self.subTest(value=value):
                r = spectrum("r")
                r["drt"]["residual_rms_re_percent"] = value
                classify_results([r], advanced.assess_batch_validity([r]))
                self.assertEqual(r["numeric_status"], "review-required")

    def test_actual_stability_tolerance_maximizes_valid_matches(self):
        def peaks(values):
            return [{"frequency_hz": 10**v, "area_impedance": 1.} for v in values]
        a = advanced.assess_peak_stability(peaks([0, .5, 1]), peaks([.4, .7, 1.5]))
        self.assertEqual(a["stable_match_count"], 2)
        self.assertTrue(a["accepted"])

    def test_area_incompatible_pair_cannot_displace_valid_pair(self):
        before = [{"frequency_hz": 1., "area_impedance": 1.},
                  {"frequency_hz": 1.1, "area_impedance": 10.}]
        after = [{"frequency_hz": 1., "area_impedance": 10.},
                 {"frequency_hz": 1.1, "area_impedance": 1.}]
        self.assertEqual(advanced.assess_peak_stability(before, after)["stable_match_count"], 2)


class ExportCounterexamples(unittest.TestCase):
    def assert_group_count(self, a, b, expected):
        groups, _ = prepare_groups([(Path("a"), a), (Path("b"), b)])
        self.assertEqual(len(groups), expected)
        with tempfile.TemporaryDirectory() as d:
            export_wide(Path(d), [a, b])
            rows = table(Path(d)/"column_mapping.csv")
            self.assertEqual(len({r["group_id"] for r in rows}), expected)
            self.assertEqual(len(list(Path(d).glob("origin_*.csv"))), expected)
            return rows

    def test_condition_fields_separate_both_plots_and_tables(self):
        for key, values in {"sample_id": ("A", "B"), "protocol_id": ("A", "B"),
                "ac_amplitude_mv": (5, 10), "rest_time_s": (60, 600),
                "pressure_mpa": (1, 10), "data_origin": ("synthetic", "experimental")}.items():
            with self.subTest(key=key):
                a, b = fixture("a"), fixture("b")
                a["metadata"][key], b["metadata"][key] = values
                rows = self.assert_group_count(a, b, 2)
                self.assertEqual({r[key] for r in rows}, {str(v) for v in values})

    def test_lower_case_temperature_does_not_pool_different_temperatures(self):
        a, b = fixture("a"), fixture("b")
        a["metadata"]["temperature_c"], b["metadata"]["temperature_c"] = 25, 60
        self.assert_group_count(a, b, 2)

    def test_equivalent_metadata_aliases_share_a_group(self):
        for canonical, alias, value in (("temperature_C", "temperature_c", 25),
                ("protocol_id", "protocol", "P1"), ("rest_time_s", "ocv_rest_s", 600),
                ("ac_amplitude_mv", "perturbation_mv", 10)):
            with self.subTest(alias=alias):
                a, b = fixture("a"), fixture("b")
                a["metadata"][canonical] = value
                b["metadata"][alias] = str(value)
                self.assert_group_count(a, b, 1)

    def test_control_protocol_id_cannot_cross_protocols(self):
        a, b = spectrum("a", protocol_id="A"), spectrum("b", protocol_id="B")
        self.assertEqual(advanced.assess_batch_validity([a, b])["repeat_assessments"], [])

    def test_reclassification_retires_old_scientific_assets_and_tables(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"output"; out.mkdir()
            (out/"execution_contract.json").write_text("{}")
            image = Path(d)/"result.svg"; image.write_text("<svg>synthetic</svg>")
            r = fixture("r", display_role="included", figures=[str(image)],
                        source_uid="source", drt={"lambda": 1e-4})
            export_bundle(out, [r], [], [], {}, {})
            (out/"05_figures/scientific/user-notes.txt").write_text("keep")
            r["display_role"] = "diagnostic-only"
            export_bundle(out, [r], [], [], {}, {})
            self.assertFalse((out/"05_figures/scientific/result.svg").exists())
            self.assertTrue((out/"05_figures/diagnostic/result.svg").is_file())
            self.assertTrue((out/"05_figures/scientific/user-notes.txt").is_file())
            self.assertEqual(len(list((out/"04_plot_data").glob("origin_*.csv"))), 1)
            self.assertTrue(list((out/"_superseded").rglob("result.svg")))

    def test_trend_skip_does_not_leave_old_scientific_images(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out/"results").mkdir()
            for i in range(2):
                (out/f"results/{i}.json").write_text(json.dumps(fixture(str(i), 3+i*.1, display_role="included")))
            a = render_trends(out, dpi=72, font="DejaVu Sans")
            target = out/"05_figures/trends/draft"
            self.assertEqual(len(a["output_files"]), 4)
            for path in (out/"results").glob("*.json"):
                r = json.loads(path.read_text()); r["display_role"] = "diagnostic-only"
                path.write_text(json.dumps(r))
            b = render_trends(out, dpi=72, font="DejaVu Sans")
            self.assertEqual(b["status"], "skipped-no-eligible-group")
            self.assertEqual(list((target/"scientific").glob("*")), [])
            self.assertEqual(list(target.glob("*_plotted_values.csv")), [])
            self.assertTrue(list((target/"_superseded").rglob("*.svg")))


class RuntimeCounterexamples(unittest.TestCase):
    def contract(self):
        return execution_contract(argparse.Namespace(parser_script=ROOT/"scripts/eis_parser.py", manifest=None), [], ROOT/"scripts")

    def test_transitive_numerical_dependency_changes_fingerprint(self):
        a = self.contract()
        original = importlib.metadata.version
        with patch.object(importlib.metadata, "version", side_effect=lambda name:
                          "99.0-audit" if name == "lmfit" else original(name)):
            b = self.contract()
        self.assertNotEqual(a["run_fingerprint"], b["run_fingerprint"])

    def test_python_version_is_part_of_contract(self):
        a = self.contract()
        self.assertIn("python_version", a["runtime"])
        with patch("platform.python_version", return_value="3.11.99"):
            b = self.contract()
        self.assertNotEqual(a["run_fingerprint"], b["run_fingerprint"])

    def test_optional_gui_does_not_change_numeric_fingerprint(self):
        a = self.contract(); original = importlib.metadata.version
        with patch.object(importlib.metadata, "version", side_effect=lambda name:
                          "99.0-audit" if name == "deareis" else original(name)):
            b = self.contract()
        self.assertEqual(a["run_fingerprint"], b["run_fingerprint"])

    def test_installer_ignores_destination_overrides_but_keeps_index(self):
        class StopBeforeInstall(Exception): pass
        commands = []
        def intercept(command, **kwargs):
            commands.append((command, kwargs["env"]))
            raise StopBeforeInstall()
        with tempfile.TemporaryDirectory() as d:
            bad = {"PIP_TARGET": str(Path(d)/"unrelated"), "PIP_PREFIX": str(Path(d)/"prefix"),
                   "PIP_USER": "1", "PIP_CONFIG_FILE": str(Path(d)/"pip.conf"),
                   "PIP_INDEX_URL": "https://packages.example.invalid/simple"}
            with patch.dict(os.environ, bad), patch.object(sys, "argv", ["bootstrap", "--venv", str(Path(d)/"new")]), \
                    patch.object(bootstrap.venv.EnvBuilder, "create"), \
                    patch.object(bootstrap.subprocess, "run", side_effect=intercept):
                with self.assertRaises(StopBeforeInstall): bootstrap.main()
            command, env = commands[0]
            for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER"): self.assertTrue(key not in env, key)
            self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
            self.assertEqual(env["PIP_INDEX_URL"], bad["PIP_INDEX_URL"])
            from pip._internal.commands import create_command
            with patch.dict(os.environ, env, clear=True):
                options, _ = create_command("install").parse_args(command[5:])
            self.assertIsNone(options.target_dir)
            self.assertIsNone(options.prefix_path)
            self.assertFalse(options.use_user_site)


class DerivedOutputSafetyTests(unittest.TestCase):
    def test_final_status_update_keeps_owned_hash_current(self):
        from derived_outputs import publish_tree, refresh_owned_file
        with tempfile.TemporaryDirectory() as d:
            out, stage = Path(d)/"out", Path(d)/"stage"
            (stage/"00_overview").mkdir(parents=True)
            name = "00_overview/run_manifest.json"
            (stage/name).write_text('{"export_status":"pending"}')
            publish_tree(stage, out, "export")
            (out/name).write_text('{"export_status":"completed"}')
            refresh_owned_file(out, "export", name)
            owned = json.loads((out/"derived_export_manifest.json").read_text())
            self.assertEqual(owned["files"][name], hashlib.sha256((out/name).read_bytes()).hexdigest())

    def test_conflicting_aliases_are_not_silently_selected(self):
        r = fixture("a")
        r["metadata"].update(temperature_C=25, temperature_c=60)
        with self.assertRaisesRegex(ValueError, "Conflicting metadata"):
            prepare_groups([(Path("a"), r)])

    def test_actual_voltage_can_differ_from_target_voltage(self):
        from metadata_contract import canonical_metadata
        a = canonical_metadata({"voltage_v": 3.09, "target_voltage_v": 3.1})
        self.assertEqual(a["voltage_v"], "3.09")
        self.assertEqual(canonical_metadata({"target_voltage_v": 3.1})["voltage_v"], "3.1")

    def test_standalone_wide_reexport_archives_obsolete_group(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); r = fixture("a")
            export_wide(out, [r]); old = next(out.glob("origin_*.csv"))
            r["display_role"] = "diagnostic-only"
            export_wide(out, [r])
            self.assertFalse(old.exists())
            self.assertEqual(len(list(out.glob("origin_*.csv"))), 1)
            self.assertEqual(len(list((out/"_superseded").rglob(old.name))), 1)

    def test_publisher_refuses_unowned_destination(self):
        from derived_outputs import publish_tree
        with tempfile.TemporaryDirectory() as d:
            out, stage = Path(d)/"out", Path(d)/"stage"; out.mkdir(); stage.mkdir()
            (out/"column_mapping.csv").write_text("user-owned")
            (stage/"column_mapping.csv").write_text("new")
            with self.assertRaises(FileExistsError): publish_tree(stage, out, "wide")
            self.assertEqual((out/"column_mapping.csv").read_text(), "user-owned")

    def test_publisher_refuses_symlink_destination(self):
        from derived_outputs import publish_tree
        with tempfile.TemporaryDirectory() as d:
            out, stage = Path(d)/"out", Path(d)/"stage"; out.mkdir(); stage.mkdir()
            external = Path(d)/"external.csv"; external.write_text("keep")
            (out/"column_mapping.csv").symlink_to(external)
            (stage/"column_mapping.csv").write_text("new")
            with self.assertRaises(ValueError): publish_tree(stage, out, "wide")
            self.assertEqual(external.read_text(), "keep")

    def test_managed_manifest_cannot_claim_raw_files(self):
        from derived_outputs import publish_tree
        with tempfile.TemporaryDirectory() as d:
            out, stage = Path(d)/"out", Path(d)/"stage"; out.mkdir(); stage.mkdir()
            raw = out/"raw.csv"; raw.write_text("keep")
            (out/"derived_wide_manifest.json").write_text(json.dumps({"owner": "wide", "files": {"raw.csv": "fake"}}))
            (stage/"column_mapping.csv").write_text("new")
            with self.assertRaises(ValueError): publish_tree(stage, out, "wide")
            self.assertEqual(raw.read_text(), "keep")

    def test_failed_publication_restores_previous_owned_files(self):
        from derived_outputs import publish_tree
        with tempfile.TemporaryDirectory() as d:
            out, stage = Path(d)/"out", Path(d)/"stage"; stage.mkdir()
            (stage/"column_mapping.csv").write_text("old")
            publish_tree(stage, out, "wide")
            prior = (out/"derived_wide_manifest.json").read_bytes()
            (stage/"column_mapping.csv").write_text("new")
            original = Path.replace
            def fail_once(path, target):
                if path == stage/"column_mapping.csv": raise OSError("injected publication failure")
                return original(path, target)
            with patch.object(Path, "replace", fail_once), self.assertRaises(OSError):
                publish_tree(stage, out, "wide")
            self.assertEqual((out/"column_mapping.csv").read_text(), "old")
            self.assertEqual((out/"derived_wide_manifest.json").read_bytes(), prior)
            self.assertFalse((out/".drt-wide.lock").exists())

    def test_heatmap_failure_does_not_publish_orphan_scientific_ridge(self):
        import plot_drt_trends as plots
        original = plots.save_checked
        def injected(builder, stem, dpi):
            if "heatmap" in stem.name: raise ValueError("injected heatmap failure")
            return original(builder, stem, dpi)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out/"results").mkdir()
            for i in range(2):
                (out/f"results/{i}.json").write_text(json.dumps(fixture(str(i), 3+i*.1, display_role="included")))
            with patch.object(plots, "save_checked", side_effect=injected):
                report = render_trends(out, dpi=72, font="DejaVu Sans")
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["output_files"])
            target = out/"05_figures/trends/draft"
            self.assertFalse(list(target.rglob("*.png")))
            self.assertFalse(list(target.rglob("*.svg")))


class ArchiveCounterexamples(unittest.TestCase):
    def test_unsafe_or_incomplete_archives_rejected(self):
        for name, mode in (("../outside.txt", 0o100644), ("shortcut", 0o120777),
                           ("unexpected.txt", 0o100644), ("/absolute.txt", 0o100644),
                           ("nested\\..\\outside.txt", 0o100644)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                p = Path(d)/"synthetic.zip"; data = b"synthetic"
                with zipfile.ZipFile(p, "w") as z:
                    z.writestr("eis-drt-batch-analysis/FILE_MANIFEST.json",
                               json.dumps({"sha256": {name: hashlib.sha256(data).hexdigest()}}))
                    info = zipfile.ZipInfo("eis-drt-batch-analysis/"+name)
                    info.create_system = 3; info.external_attr = mode << 16
                    z.writestr(info, data)
                with self.assertRaises(ValueError): build_release.verify_archive(p)

    def test_windows_private_paths_detected(self):
        self.assertTrue(build_release.privacy_findings(b"C:" + b"\\Us" + b"ers\\synthetic_person\\data.csv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

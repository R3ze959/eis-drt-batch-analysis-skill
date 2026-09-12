#!/usr/bin/env python3
"""Rendered artist/bounding-box tests for diagnostic plots; no private inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

import diagnostic_plotting as plotting


def fixture():
    f = np.geomspace(1e4, 1e-3, 45)
    gamma = .001*np.exp(-np.log(f/10)**2)+.0005*np.exp(-np.log(f/200)**2)
    z = .01+.002/(1+1j*f/10)
    signed = gamma-.0004*np.exp(-np.log(f/.1)**2)
    return {"result_slug": "synthetic", "spectrum_id": "synthetic", "label": "Synthetic 公开诊断",
        "impedance_basis": "ohm", "peaks": [], "model_reduce": {"status": "disabled"},
        "curves": {"frequency_hz": f.tolist(), "zreal_measured": z.real.tolist(), "neg_zimag_measured": (-z.imag).tolist(),
            "zreal_reconstructed": z.real.tolist(), "neg_zimag_reconstructed": (-z.imag).tolist(),
            "residual_re_percent": (.03*np.sin(np.arange(len(f)))).tolist(),
            "residual_im_percent": (.03*np.cos(np.arange(len(f)))).tolist(),
            "drt_frequency_hz": f.tolist(), "gamma": gamma.tolist(),
            "sensitivity_curves": [{"frequency_hz": f.tolist(), "gamma": (gamma*scale).tolist(), "selected": scale == 1,
                                    "lambda": value} for scale, value in ((1.1, 1e-5), (1, 1e-4), (.9, 1e-3))]},
        "lambda_selection": {"selected_lambda": 1e-4, "selection_method": "grid-lcurve-fallback", "grid": [
            {"lambda": float(value), "relative_residual_norm": float(x), "roughness_norm": float(y)}
            for value, x, y in zip(np.geomspace(1e-7, .1, 13), np.linspace(.0776, .0786, 13), np.geomspace(.00018, .000026, 13))]},
        "signed_gdrt": {"status": "completed", "frequency_hz": f.tolist(), "gamma": signed.tolist()},
        "loewner_rc_rl": {"status": "completed", "tau_rc_s": [.001, .1], "gamma_rc": [.002, .004],
                          "tau_rl_s": [.02], "gamma_rl": [.0005]},
        "ddt": {"status": "not-triggered"}}


class CompactTickTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_actual_narrow_log_overlap_is_removed_without_altering_coordinates(self):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120, layout="constrained")
        x, y = np.linspace(.0776, .0786, 13), np.geomspace(.00018, .000026, 13)
        line, = ax.loglog(x, y)
        ax.xaxis.set_major_locator(FixedLocator(x[::2]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value*100:.3f} × 10⁻²"))
        before = plotting.audit_tick_label_collisions(ax)
        self.assertTrue(before["overlaps"])
        limits = (ax.get_xlim(), ax.get_ylim())
        result = plotting.ensure_compact_ticks(ax)
        self.assertEqual(result["status"], "pass")
        self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        self.assertLessEqual(len(ax.get_xticklabels()), 4)

    def test_tick_helper_reports_unresolved_geometry_without_shrinking_font(self):
        fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=100)
        ax.plot([1, 2], [1, 2])
        ax.set_xlabel("An intentionally enormous axis label", fontsize=40)
        result = plotting.ensure_compact_ticks(ax)
        self.assertEqual(result["status"], "overlap")
        self.assertEqual(ax.xaxis.label.get_fontsize(), 40)


class DiagnosticFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_discrete_weights_and_continuous_density_have_independent_artists_and_axes(self):
        result = fixture()
        original = copy.deepcopy(result)
        fig, axes, qa = plotting.build_advanced_audit_figure(result, dpi=80)
        self.assertEqual(qa["text_qa"]["status"], "pass")
        self.assertEqual(result, original)
        self.assertIsNot(axes["signed"], axes["loewner"])
        self.assertFalse(axes["signed"].get_shared_y_axes().joined(axes["signed"], axes["loewner"]))
        signed_line = axes["signed"].lines[0]
        np.testing.assert_array_equal(signed_line.get_xdata(), result["signed_gdrt"]["frequency_hz"])
        np.testing.assert_array_equal(signed_line.get_ydata(), result["signed_gdrt"]["gamma"])
        for collection, tau, weights in zip(axes["loewner"].collections,
                (result["loewner_rc_rl"]["tau_rc_s"], result["loewner_rc_rl"]["tau_rl_s"]),
                (result["loewner_rc_rl"]["gamma_rc"], result["loewner_rc_rl"]["gamma_rl"])):
            np.testing.assert_array_equal(collection.get_offsets()[:, 0], 1/(2*np.pi*np.asarray(tau)))
            np.testing.assert_array_equal(collection.get_offsets()[:, 1], weights)
        loewner_limits = axes["loewner"].get_ylim()
        result["signed_gdrt"]["gamma"] = (100*np.asarray(result["signed_gdrt"]["gamma"])).tolist()
        _, modified_axes, _ = plotting.build_advanced_audit_figure(result, dpi=80)
        self.assertEqual(modified_axes["loewner"].get_ylim(), loewner_limits)

    def test_area_normalized_units_do_not_rescale_plotted_values(self):
        result = fixture()
        result["impedance_basis"] = "ohm_cm2"
        fig, axes, report = plotting.build_advanced_audit_figure(result, dpi=80)
        self.assertEqual(report["impedance_unit"], "Ω cm²")
        self.assertIn("cm²", axes["loewner"].get_ylabel())
        self.assertIn("cm²", axes["signed"].get_ylabel())
        np.testing.assert_array_equal(axes["signed"].lines[0].get_ydata(), result["signed_gdrt"]["gamma"])
        np.testing.assert_array_equal(axes["loewner"].collections[1].get_offsets()[:, 1], [.0005])

    def test_optional_statuses_have_distinct_rendered_labels_and_no_empty_numeric_axes(self):
        titles = set()
        for status in ("disabled", "not-triggered", "failed", "completed"):
            with self.subTest(status=status):
                result = fixture()
                result["signed_gdrt"] = {"status": status}
                result["loewner_rc_rl"] = {"status": status}
                result["ddt"] = {"status": status, "best": None}
                fig, axes, qa = plotting.build_advanced_audit_figure(result, dpi=80)
                self.assertEqual(qa["text_qa"]["status"], "pass")
                titles.add(axes["ddt"].get_title())
                for name in ("signed", "loewner", "ddt"):
                    self.assertFalse(axes[name].axison)
                    self.assertEqual(len(axes[name].lines), 0)
                    self.assertEqual(len(axes[name].collections), 0)
                self.assertNotIn("acceptance", axes["primary"].get_title().lower())
                plt.close(fig)
        self.assertEqual(len(titles), 4)

    def test_completed_ddt_candidate_uses_its_original_distribution(self):
        result = fixture()
        result["ddt"] = {"status": "completed", "best_boundary": "transmissive-short", "best": {
            "frequency_hz": [1, 10, 100], "gamma": [.001, .003, .002]}}
        fig, axes, report = plotting.build_advanced_audit_figure(result, dpi=80)
        self.assertTrue(axes["ddt"].axison)
        np.testing.assert_array_equal(axes["ddt"].lines[0].get_ydata(), [.001, .003, .002])
        self.assertEqual(report["text_qa"]["status"], "pass")

    def test_primary_uses_shared_typography_and_preserves_lcurve_values(self):
        result = fixture()
        original = copy.deepcopy(result)
        fig, axes, qa = plotting.build_primary_audit_figure(result, dpi=80)
        self.assertEqual(qa["text_qa"]["status"], "pass")
        self.assertEqual(qa["tick_qa"]["lcurve"]["status"], "pass")
        self.assertEqual(axes["lcurve"].xaxis.label.get_fontsize(), 36)
        self.assertTrue(all(label.get_fontsize() == 28 for label in axes["lcurve"].get_xticklabels()))
        np.testing.assert_array_equal(axes["lcurve"].lines[0].get_xdata(),
                                      [row["relative_residual_norm"] for row in result["lambda_selection"]["grid"]])
        self.assertEqual(result, original)

    def test_real_png_and_editable_svg_exports_record_qa_and_keep_numerical_arrays(self):
        from PIL import Image
        result = fixture()
        original_curves = copy.deepcopy(result["curves"])
        with tempfile.TemporaryDirectory(prefix="drt-diagnostic-plot-") as temporary:
            paths = plotting.save_primary_audit(result, temporary, dpi=72)+plotting.save_advanced_audit(result, temporary, dpi=72)
            for path in map(Path, paths):
                self.assertGreater(path.stat().st_size, 1000)
                if path.suffix == ".png":
                    with Image.open(path) as img:
                        self.assertGreater(img.width, 1000)
                        self.assertGreater(img.height, 700)
                        self.assertIsNotNone(img.convert("RGB").getbbox())
                else:
                    root = ElementTree.parse(path).getroot()
                    self.assertGreater(len(root.findall(".//{http://www.w3.org/2000/svg}text")), 10)
            self.assertEqual(result["curves"], original_curves)
            self.assertEqual(result["plot_quality"]["primary"]["text_qa"]["status"], "pass")
            self.assertEqual(result["plot_quality"]["advanced"]["text_qa"]["status"], "pass")

    def test_batch_overlay_retains_every_curve_and_serializes_editable_text(self):
        a, b = fixture(), fixture()
        b["label"] = "Second synthetic spectrum"
        with tempfile.TemporaryDirectory(prefix="drt-diagnostic-overlay-") as temporary:
            paths = plotting.save_batch_overlay([a, b], temporary, dpi=72)
            root = ElementTree.parse(paths[1]).getroot()
            text = " ".join(root.itertext())
            self.assertIn("Second synthetic spectrum", text)
            self.assertGreater(Path(paths[0]).stat().st_size, 1000)


def render_public_examples(paths, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        numerical_before = copy.deepcopy(result)
        outputs = plotting.save_primary_audit(result, output_dir, dpi=120)+plotting.save_advanced_audit(result, output_dir, dpi=120)
        qa = result.pop("plot_quality")
        records.append({"input_name": path.name, "outputs": outputs, "qa": qa,
                        "numerical_result_unchanged": result == numerical_before})
    (output_dir/"diagnostic_plot_report.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"public_examples": len(records), "output_dir": str(output_dir),
                      "all_numerical_results_unchanged": all(row["numerical_result_unchanged"] for row in records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-result", type=Path, action="append", default=[])
    parser.add_argument("--artifact-dir", type=Path)
    args, extra = parser.parse_known_args()
    outcome = unittest.main(argv=[__file__, *extra], verbosity=2, exit=False)
    if not outcome.result.wasSuccessful():
        raise SystemExit(1)
    if args.public_result:
        if args.artifact_dir is None:
            parser.error("--artifact-dir is required for public examples")
        render_public_examples(args.public_result, args.artifact_dir)

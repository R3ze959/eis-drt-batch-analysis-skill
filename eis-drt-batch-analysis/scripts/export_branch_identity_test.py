#!/usr/bin/env python3
"""Array-identity regressions: recommendation metadata must never relabel a table."""
from __future__ import annotations

import copy
import csv
import ast
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from result_contract import common_fields, export_bundle, export_wide


def fixture(selected="signed_gdrt"):
    frequency = [100., 10., 1.]
    primary_tau, primary_gamma = [.01, .1, 1.], [1., 2., 3.]
    primary_z, signed_z = [2-1j, 3-2j, 4-3j], [20+4j, 30-5j, 40+6j]
    result = {"status": "completed", "spectrum_uid": "array-owner-one", "source_uid": "source-one",
        "source": "/synthetic/branch-input.csv", "source_sha256": "fixture-sha256",
        "spectrum_id": "same-label", "label": "same-label", "impedance_basis": "ohm_cm2",
        "metadata": {"cell_id": "branch-cell", "direction": "charge", "voltage_v": 3.1, "data_origin": "synthetic"},
        "display_role": "diagnostic-only", "numeric_status": "review-required",
        "validity_status": "controls-incomplete-no-acceptance", "evidence_status": "evidence-missing",
        "recommended_branch": selected, "recommendation_status": "exploratory-review-required",
        "recommended_rms_percent": .1, "drt": {"lambda": 1e-3},
        "model_recommendation": {"status": "exploratory-review-required", "recommended_branch": selected,
                                 "recommended_rms_percent": .1, "reasons": ["synthetic-review-fixture"]},
        "curves": {"frequency_hz": frequency, "acquisition_index": [7, 8, 9],
                   "zreal_measured": [10., 11., 12.], "neg_zimag_measured": [-1., 2., 3.],
                   "zreal_reconstructed": [z.real for z in primary_z],
                   "neg_zimag_reconstructed": [-z.imag for z in primary_z],
                   "residual_re_percent": [1., 2., 3.], "residual_im_percent": [.1, .2, .3],
                   "tau_s": primary_tau, "gamma": primary_gamma,
                   "sensitivity_curves": [{"lambda": 1e-3, "selected": True,
                                           "frequency_hz": frequency, "gamma": [4., 5., 6.]}]},
        "peaks": [{"peak_id": "P1", "tau_s": .1, "area_impedance": 2.}],
        "peak_evidence": {"per_peak": [{"peak_id": "P1", "status": "noise-or-resolution-sensitive"}]},
        "signed_gdrt": {"status": "completed", "tau_s": [.015, .15, .5], "gamma": [-4., 5., -6.],
                        "reconstructed": signed_z,
                        "sensitivity": {"cases": [{"case": "lambda", "lambda": .01, "num_tau": 3,
                                                      "tau_s": [.015, .15, .5], "gamma": [-7., 8., -9.]}]}},
        "ddt": {"candidates": [{"boundary": "transmissive-short", "status": "completed",
                                 "tau_s": [.02, .2], "gamma": [12., 13.],
                                 "reconstructed": [5-1j, 6-2j, 7-3j]}]}}
    payload = {key: copy.deepcopy(result[key]) for key in
               ("spectrum_uid", "source_uid", "source", "source_sha256", "spectrum_id", "label", "impedance_basis", "metadata")}
    payload.update(acquisition_index=[7, 8, 9], frequency_hz=frequency,
                   zreal=[10., 11., 12.], neg_zimag=[-1., 2., 3.])
    return result, payload


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class ExportBranchIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="drt-export-branch-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "output"
        self.output.mkdir()
        (self.output / "execution_contract.json").write_text("{}", encoding="utf-8")
        self.result, self.payload = fixture()
        self.original = copy.deepcopy((self.result, self.payload))
        export_bundle(self.output, [self.result], [self.payload], [],
                      {"parameters": {"reference_profile": "synthetic-compatibility", "inductance_policy": "off"}}, {})

    def rows(self, filename):
        return read_table(self.output / filename)[1]

    def assert_common_unchanged(self, rows):
        for row in rows:
            for key in ("spectrum_uid", "source_uid", "source_sha256", "impedance_basis", "display_role",
                        "numeric_status", "validity_status", "evidence_status", "recommended_branch", "recommendation_status"):
                self.assertEqual(row[key], self.result[key])

    def test_primary_reconstruction_is_not_relabelled_as_recommended_signed(self):
        rows = self.rows("02_quality/impedance_reconstruction.csv")
        self.assert_common_unchanged(rows)
        self.assertEqual({r["reconstruction_branch"] for r in rows}, {"primary_rc"})
        self.assertEqual({r["recommended_branch"] for r in rows}, {"signed_gdrt"})
        for column in ("frequency_hz", "acquisition_index", "zreal_measured", "neg_zimag_measured",
                       "zreal_reconstructed", "neg_zimag_reconstructed", "residual_re_percent", "residual_im_percent"):
            np.testing.assert_array_equal([float(row[column]) for row in rows], self.result["curves"][column])

    def test_primary_drt_plot_long_and_compatibility_arrays_keep_primary_identity(self):
        for filename in ("03_results/primary_drt/curves.csv", "04_plot_data/drt_long.csv",
                         "03_results/reference_compatibility/curves.csv"):
            with self.subTest(filename=filename):
                rows = self.rows(filename)
                self.assert_common_unchanged(rows)
                self.assertEqual({r["distribution_branch"] for r in rows}, {"primary_rc"})
                np.testing.assert_array_equal([float(r["tau_s"]) for r in rows], self.result["curves"]["tau_s"])
                np.testing.assert_array_equal([float(r["gamma"]) for r in rows], [1, 2, 3])
                self.assertTrue(all(r["distribution_measure"] == "gamma_per_ln_tau" for r in rows))

    def test_primary_peaks_sensitivity_and_perturbation_evidence_identify_their_owner(self):
        for filename in ("02_quality/peaks_with_support.csv", "02_quality/peak_perturbation_evidence.csv"):
            with self.subTest(filename=filename):
                rows = self.rows(filename)
                self.assert_common_unchanged(rows)
                self.assertEqual({r["distribution_branch"] for r in rows}, {"primary_rc"})
                self.assertEqual(rows[0]["peak_id"], "P1")
        rows = self.rows("02_quality/all_sensitivity_curves.csv")
        primary = [row for row in rows if row["method"] == "primary"]
        signed = [row for row in rows if row["method"] == "signed_gdrt"]
        self.assertEqual({r["distribution_branch"] for r in primary}, {"primary_rc"})
        np.testing.assert_array_equal([float(row["gamma"]) for row in primary], [4, 5, 6])
        self.assertTrue(signed)
        self.assertTrue(all(row["distribution_branch"] != "primary_rc" for row in signed))
        np.testing.assert_array_equal([float(row["gamma"]) for row in signed], [-7, 8, -9])

    def test_recommended_signed_tables_explicitly_scope_selection_to_reconstruction(self):
        for filename, identity_field in (("03_results/recommended/curves.csv", "distribution_branch"),
                                         ("03_results/recommended/reconstruction.csv", "reconstruction_branch")):
            with self.subTest(filename=filename):
                rows = self.rows(filename)
                self.assert_common_unchanged(rows)
                self.assertEqual({r[identity_field] for r in rows}, {"signed_gdrt"})
                self.assertEqual({r["selected_branch"] for r in rows}, {"signed_gdrt"})
                self.assertEqual({r["selection_scope"] for r in rows}, {"reconstruction-only"})
                self.assertEqual({r["selection_is_physical_acceptance"] for r in rows}, {"False"})
        curves = self.rows("03_results/recommended/curves.csv")
        np.testing.assert_array_equal([float(r["gamma"]) for r in curves], [-4, 5, -6])
        reconstruction = self.rows("03_results/recommended/reconstruction.csv")
        np.testing.assert_array_equal([float(r["zreal_reconstructed"]) for r in reconstruction], [20, 30, 40])
        np.testing.assert_array_equal([float(r["neg_zimag_reconstructed"]) for r in reconstruction], [-4, 5, -6])
        self.assertEqual(self.rows("00_overview/model_recommendations.csv")[0]["selection_scope"], "reconstruction-only")

    def test_origin_mapping_names_primary_without_changing_numeric_wide_columns(self):
        mapping = self.rows("04_plot_data/column_mapping.csv")
        self.assert_common_unchanged(mapping)
        self.assertEqual(mapping[0]["distribution_branch"], "primary_rc")
        key = mapping[0]["group_id"]
        fields, wide = read_table(self.output / f"04_plot_data/origin_{key}.csv")
        uid = self.result["spectrum_uid"]
        self.assertEqual(fields, [uid + "_tau_s", uid + "_gamma"])
        np.testing.assert_array_equal([float(r[uid + "_gamma"]) for r in wide], [1, 2, 3])
        matrix_fields, matrix = read_table(self.output / f"04_plot_data/heatmap_exact_grid_{key}.csv")
        self.assertEqual(matrix_fields, ["tau_s", uid])
        np.testing.assert_array_equal([float(r[uid]) for r in matrix], [1, 2, 3])

    def test_advanced_method_identity_and_raw_input_arrays_are_not_relabelled(self):
        advanced = self.rows("02_quality/advanced_reconstructions.csv")
        self.assertEqual({r["method"] for r in advanced}, {"signed_gdrt", "ddt:transmissive-short"})
        signed = [r for r in advanced if r["method"] == "signed_gdrt"]
        np.testing.assert_array_equal([float(r["zreal_reconstructed"]) for r in signed], [20, 30, 40])
        header, normalized = read_table(self.output / "01_inputs/normalized_eis.csv")
        self.assertNotIn("reconstruction_branch", header)
        self.assertNotIn("distribution_branch", header)
        np.testing.assert_array_equal([float(r["zreal"]) for r in normalized], self.payload["zreal"])
        self.assertEqual((self.result, self.payload), self.original)

    def test_recommended_primary_tables_still_export_primary_arrays(self):
        result, payload = fixture(selected="primary_rc")
        export_bundle(self.output, [result], [payload], [], {}, {})
        rows = self.rows("03_results/recommended/curves.csv")
        self.assertEqual({r["selected_branch"] for r in rows}, {"primary_rc"})
        self.assertEqual({r["distribution_branch"] for r in rows}, {"primary_rc"})
        self.assertEqual({r["selection_scope"] for r in rows}, {"reconstruction-only"})
        np.testing.assert_array_equal([float(r["gamma"]) for r in rows], [1, 2, 3])

    def test_standalone_wide_export_uses_the_same_branch_contract(self):
        folder = Path(self.directory.name) / "standalone-wide"
        export_wide(folder, [self.result])
        _, rows = read_table(folder / "column_mapping.csv")
        self.assertEqual(rows[0]["distribution_branch"], "primary_rc")
        self.assertEqual(rows[0]["recommended_branch"], "signed_gdrt")

    def test_overview_explains_actual_branch_vs_recommendation(self):
        text = (self.output / "00_overview/README.md").read_text(encoding="utf-8")
        self.assertIn("reconstruction_branch and distribution_branch", text)
        self.assertIn("never relabels the primary arrays", text)
        self.assertIn("selection_scope=reconstruction-only", text)
        self.assertIn("not physical-model acceptance", text)


class RootCompatibilityBranchTests(unittest.TestCase):
    """Execute the production post-fit CSV block without rerunning inversion."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="drt-root-branch-")
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.result, _ = fixture()
        r = self.result
        r.update(kk={"status": "screen-pass"}, zhit={"status": "not-run"}, validity={}, qc={},
                 lambda_selection={"selection_method": "fixed"}, inductance_mode=False,
                 loewner_rc_rl={"status": "completed", "model_order": 3,
                     "tau_rc_s": [.1, .2], "gamma_rc": [90., 80.],
                     "tau_rl_s": [.3], "gamma_rl": [70.]},
                 loewner_validation={"status": "completed", "selected_order": 2,
                     "kernel_evidence_reliable": True, "heldout_rms_percent": .01})
        r["drt"].update(credible_intervals=True, credible_samples=1000,
                        residual_rms_re_percent=.4, residual_rms_im_percent=.3)
        r["curves"].update(drt_frequency_hz=[1 / (2 * math.pi * t) for t in r["curves"]["tau_s"]],
                           credible_lower=[.5, 1.5, 2.5], credible_upper=[1.5, 2.5, 3.5])
        r["signed_gdrt"]["frequency_hz"] = [1 / (2 * math.pi * t) for t in r["signed_gdrt"]["tau_s"]]
        ddt = r["ddt"]["candidates"][0]
        ddt["frequency_hz"] = [1 / (2 * math.pi * t) for t in ddt["tau_s"]]
        self.original = copy.deepcopy(r)
        import batch_drt
        tree = ast.parse(Path(batch_drt.__file__).read_text(encoding="utf-8"))
        body = next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        start = next(i for i, n in enumerate(body) if isinstance(n, ast.AnnAssign)
                     and isinstance(n.target, ast.Name) and n.target.id == "summary_rows")
        stop = next(i for i, n in enumerate(body[start:], start) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Tuple)
                    and [getattr(t, "id", None) for t in n.target.elts] == ["name", "rows"])
        block = ast.Module(body=body[start:stop], type_ignores=[])
        namespace = {"results": [r], "output_dir": self.output, "common_fields": common_fields,
                     "write_csv": batch_drt.write_csv, "math": math, "Any": object}
        exec(compile(block, str(batch_drt.__file__), "exec"), namespace)

    def rows(self, filename):
        return read_table(self.output / filename)[1]

    def test_root_primary_gamma_peaks_and_rms_have_actual_primary_identity(self):
        for filename in ("drt_curves.csv", "peaks.csv", "batch_summary.csv"):
            with self.subTest(filename=filename):
                rows = self.rows(filename)
                self.assertTrue(rows)
                self.assertEqual({r["distribution_branch"] for r in rows}, {"primary_rc"})
                self.assertEqual({r["recommended_branch"] for r in rows}, {"signed_gdrt"})
                self.assertEqual({r["spectrum_uid"] for r in rows}, {self.result["spectrum_uid"]})
        self.assertEqual(self.rows("batch_summary.csv")[0]["reconstruction_branch"], "primary_rc")
        curves = self.rows("drt_curves.csv")
        for column in ("tau_s", "gamma", "credible_lower", "credible_upper"):
            np.testing.assert_array_equal([float(row[column]) for row in curves], self.result["curves"][column])

    def test_root_legacy_discrete_weights_are_not_validated_by_the_cv_model(self):
        rows = self.rows("loewner_rc_rl.csv")
        self.assertEqual({r["model_identity"] for r in rows}, {"legacy-full-data-matrix-rank-model"})
        self.assertEqual({r["distribution_branch"] for r in rows}, {"loewner_rc_rl"})
        self.assertEqual({int(r["model_order"]) for r in rows}, {3})
        self.assertEqual({r["heldout_validated"] for r in rows}, {"False"})
        self.assertEqual([r["branch"] for r in rows], ["RC", "RC", "RL"])
        np.testing.assert_array_equal([float(r["gamma"]) for r in rows], [90, 80, 70])
        np.testing.assert_array_equal([float(r["tau_s"]) for r in rows], [.1, .2, .3])

    def test_root_signed_and_diffusion_arrays_keep_their_own_branch(self):
        signed = self.rows("signed_gdrt_curves.csv")
        self.assertEqual({r["distribution_branch"] for r in signed}, {"signed_gdrt"})
        np.testing.assert_array_equal([float(r["signed_gamma"]) for r in signed], [-4, 5, -6])
        diffusion = self.rows("ddt_curves.csv")
        self.assertEqual({r["distribution_branch"] for r in diffusion}, {"ddt:transmissive-short"})
        np.testing.assert_array_equal([float(r["gamma"]) for r in diffusion], [12, 13])

    def test_root_export_does_not_mutate_numerical_record_or_status(self):
        self.assertEqual(self.result, self.original)
        for filename in ("drt_curves.csv", "peaks.csv", "batch_summary.csv", "loewner_rc_rl.csv",
                         "signed_gdrt_curves.csv", "ddt_curves.csv"):
            for row in self.rows(filename):
                for field in ("spectrum_uid", "source_uid", "source_sha256", "impedance_basis", "display_role",
                              "validity_status", "numeric_status", "evidence_status", "recommendation_status"):
                    self.assertEqual(row[field], self.result[field])


if __name__ == "__main__":
    unittest.main(verbosity=2)

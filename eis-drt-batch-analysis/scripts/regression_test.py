#!/usr/bin/env python3
"""Executable regressions for audited failure modes; synthetic data only."""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import advanced_drt as advanced
import batch_drt as batch
from control_validation import classify_results
from peak_matching import match_positions
from result_contract import identities, initialize_run, inventory_inputs, export_wide, numerical_digest


F = np.logspace(5, -3, 61)
W = 2 * np.pi * F
Z = 0.8 + 3 / (1 + 1j * W * 0.02)


def spectrum(name, z=Z, f=F, **metadata):
    return {"spectrum_uid": name, "spectrum_id": name, "source_uid": name, "label": name,
            "status": "completed", "impedance_basis": "ohm", "metadata": {"cell_id": "C1", "state": "S50", **metadata},
            "kk": {"status": "screen-pass"}, "drt": {"residual_rms_re_percent": 0.01, "residual_rms_im_percent": 0.01},
            "curves": {"frequency_hz": f.tolist(), "zreal_measured": z.real.tolist(),
                       "neg_zimag_measured": (-z.imag).tolist()},
            "peaks": [{"peak_id": "P1", "frequency_hz": 10., "tau_s": 1/(20*np.pi), "area_impedance": 3.}]}


class ControlTests(unittest.TestCase):
    def test_uncontrolled_spectrum_not_accepted(self):
        spectra = [spectrum("r" + str(i), ac_amplitude_mv="10", rest_time_s="1800") for i in range(3)]
        spectra += [spectrum("amp", ac_amplitude_mv="5", rest_time_s="1800"),
                    spectrum("rest60", ac_amplitude_mv="10", rest_time_s="60"),
                    spectrum("rest600", ac_amplitude_mv="10", rest_time_s="600"),
                    spectrum("uncontrolled", cell_id="OTHER", state="S90")]
        validation = classify_results(spectra, advanced.assess_batch_validity(spectra))
        self.assertEqual(spectra[0]["validity_status"], "accepted-with-supplied-controls")
        self.assertEqual(spectra[-1]["validity_status"], "controls-incomplete-no-acceptance")
        self.assertNotEqual(validation["overall_verdict"], "accepted-with-supplied-controls")
        self.assertEqual(spectra[4]["controls"]["stationarity"], "evidence-missing")

    def test_no_common_grid_repeats(self):
        rows = [spectrum(str(i), f=F * (1 + .03*i)) for i in range(3)]
        assessment = advanced.assess_batch_validity(rows)["repeat_assessments"][0]
        self.assertEqual(assessment["status"], "not-assessable")
        self.assertEqual(assessment["anomaly_scores"], [])

    def test_no_common_grid_amplitude_is_not_nonlinearity(self):
        rows = [spectrum("a", ac_amplitude_mv="5"), spectrum("b", f=F*1.03, ac_amplitude_mv="10")]
        assessment = advanced.assess_batch_validity(rows)["linearity_assessments"][0]
        self.assertEqual(assessment["status"], "not-assessable")
        self.assertEqual(assessment["member_uids"], ["a", "b"])

    def test_partial_overlap_is_not_acceptance(self):
        self.assertEqual(advanced._pair_assessment(spectrum("a"), spectrum("b", f=F[:10], z=Z[:10]))["status"], "not-assessable")

    def test_units_cannot_be_compared(self):
        a, b = spectrum("a"), spectrum("b")
        b["impedance_basis"] = "ohm_cm2"
        self.assertEqual(advanced._pair_assessment(a, b)["status"], "not-assessable")

    def test_duplicate_terminal_rest_is_not_plateau(self):
        rows = [spectrum("a", rest_time_s="60"), spectrum("b", rest_time_s="600"),
                spectrum("c", z=2*Z, rest_time_s="1800"), spectrum("d", z=2*Z, rest_time_s="1800")]
        self.assertEqual(advanced.assess_batch_validity(rows)["stationarity_assessments"][0]["status"], "rest-dependent")

    def test_unmatched_terminal_rest_never_uses_earlier_pair(self):
        rows = [spectrum("a", rest_time_s="60"), spectrum("b", rest_time_s="600"),
                spectrum("c", f=F*1.03, rest_time_s="1800")]
        self.assertEqual(advanced.assess_batch_validity(rows)["stationarity_assessments"][0]["status"], "not-assessable")

    def test_explicit_group_cannot_cross_cell(self):
        rows = [spectrum("a", repeat_group="X"), spectrum("b", repeat_group="X", cell_id="OTHER")]
        self.assertEqual(advanced.assess_batch_validity(rows)["repeat_assessments"], [])

    def test_identical_labels_still_identify_outlier(self):
        rows = [spectrum("a"), spectrum("b", z=1.001*Z), spectrum("c", z=1.15*Z)]
        for r in rows:
            r["label"] = "same label"
        a = advanced.assess_batch_validity(rows)["repeat_assessments"][0]
        self.assertEqual([r["spectrum_uid"] for r in a["anomaly_scores"] if r["outlier"]], ["c"])

    def test_pair_metric_is_symmetric(self):
        a, b = spectrum("a"), spectrum("b", z=2*Z)
        self.assertAlmostEqual(advanced._pair_assessment(a,b)["complex_rms_percent"], advanced._pair_assessment(b,a)["complex_rms_percent"])

    def test_pair_number_does_not_define_route(self):
        rows = [spectrum("Pair 1"), spectrum("Pair 99")]
        classify_results(rows, advanced.assess_batch_validity(rows))
        self.assertEqual(rows[0]["display_role"], rows[1]["display_role"])
        self.assertTrue(rows[0]["ordinary_rc_eligible"])

    def test_numeric_and_evidence_states_separate(self):
        rows = [spectrum("a")]
        rows[0]["kk"]["downsampled_for_screen"] = True
        classify_results(rows, advanced.assess_batch_validity(rows))
        self.assertEqual(rows[0]["status"], "completed")
        self.assertEqual(rows[0]["numeric_status"], "review-required")
        self.assertEqual(rows[0]["evidence_status"], "evidence-missing")


class PeakTests(unittest.TestCase):
    def test_maximum_cardinality_not_greedy(self):
        result = match_positions(10**np.array([0., .3]), 10**np.array([.1, -.2]), .25)
        self.assertEqual(len(result["matches"]), 2)
        self.assertEqual(result["matches"][0][0], 1)

    def test_merged_lambda_peaks_not_both_robust(self):
        def peak(t):
            return {"tau_s": t, "frequency_hz": 1/(2*np.pi*t), "gamma": 2., "area_impedance": 1.}
        primary = [peak(.01), peak(.014)]
        sensitivity = [{"lambda": 1e-5, "peaks": [peak(.012)]},
                       {"lambda": 1e-4, "peaks": copy.deepcopy(primary)},
                       {"lambda": 1e-3, "peaks": [peak(.012)]}]
        result = batch.classify_peaks(primary, sensitivity, .001, 1e5, .7, .5, .02, F, -Z.imag)
        self.assertFalse(any(r["confidence"] == "robust-to-lambda" for r in result))
        self.assertEqual(sum(r["lambda_persistence"] for r in result), 4)
        self.assertTrue(all(r["lambda_split_merge_ambiguous"] for r in result))

    def test_empty_matches(self):
        self.assertEqual(match_positions([], [], .5)["matches"], {})

    def test_close_but_persistent_peaks_are_not_false_merges(self):
        result = match_positions([.01, .014], [.01, .014], .5)
        self.assertEqual(result["resolution_change_left"], [False, False])

    def test_threshold_is_hard(self):
        self.assertEqual(match_positions([1.], [100.], .5)["matches"], {})


class InversionTests(unittest.TestCase):
    def test_nonconverged_solver_not_completed(self):
        def failed(a,b,**kwargs):
            return SimpleNamespace(x=np.zeros(a.shape[1]), success=False, status=0,
                                   message="iteration limit (injected)", optimality=1.)
        with patch.object(advanced, "lsq_linear", side_effect=failed):
            result = advanced._solve_distribution(F,Z,lambda w,t: 1/(1+1j*w[:,None]*t),
                       signed=False, lambda_grid=[1e-4])
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["solver"]["converged"])

    def test_rank_one_distribution_rejected(self):
        with self.assertRaisesRegex(ValueError, "Rank-one"):
            advanced._solve_distribution(F,Z,lambda w,t: advanced.diffusion_kernel("semi-infinite",w,t),
                                         signed=False, lambda_grid=[1e-4])

    def test_scalar_warburg_recovers_amplitude(self):
        z = .8 + 1j*W*2e-6 + 2/np.sqrt(1j*W)
        result = advanced.fit_semi_infinite_amplitude(F,z)
        self.assertFalse(result["distribution_identifiable"])
        self.assertEqual(result["gamma"], [])
        self.assertIsNone(result["dominant_peak"])
        self.assertAlmostEqual(result["sigma_impedance_per_sqrt_s"], 2., places=7)
        self.assertAlmostEqual(result["nuisance"]["R_inf_ohm"], .8, places=7)
        self.assertAlmostEqual(result["nuisance"]["L_henry"], 2e-6, places=10)

    def test_finite_diffusion_known_time_recovery(self):
        for boundary in ("blocking-open", "transmissive-short", "gerischer"):
            with self.subTest(boundary=boundary):
                z = .8 + 3*advanced.diffusion_kernel(boundary,W,np.array([.2]))[:,0]
                result = advanced.ddt_analysis(F,z,lambda_points=7)
                candidate = next(r for r in result["candidates"] if r["boundary"] == boundary)
                self.assertEqual(candidate["status"], "completed")
                self.assertLess(abs(math.log10(candidate["dominant_peak"]["tau_s"]/.2)), .15)
                self.assertLess(candidate["residual_rms_percent"], 1.5)
                self.assertTrue(candidate["solver"]["converged"])
                self.assertGreaterEqual(candidate["nuisance"]["R_inf_ohm"], 0)
                self.assertGreaterEqual(candidate["nuisance"]["L_henry"], 0)
                self.assertEqual(len(candidate["sensitivity"]["cases"]), 3)

    def test_signed_negative_lobe_and_support(self):
        z = Z + 1.5 * (1j*W*.5)/(1+1j*W*.5)
        result = advanced.continuous_signed_gdrt(F,z,lambda_points=7)
        self.assertEqual(result["status"], "completed")
        self.assertLess(min(result["gamma"]), -.05)
        self.assertLess(result["residual_rms_percent"], 1.)
        self.assertEqual(len(result["sensitivity"]["cases"]), 3)
        self.assertGreater(result["support"]["supported_absolute_area_fraction"], .8)

    def test_unphysical_diffusion_has_no_forced_winner(self):
        z = -3 + .5/(1+1j*W*.02) + 1j*W*1e-6
        result = advanced.ddt_analysis(F,z,lambda_points=7)
        self.assertIsNone(result["best_boundary"])
        self.assertEqual(result["selection_status"], "no-distribution-candidate-accepted")

    def test_model_reduce_uses_original_domain_addback(self):
        from advanced_smoke_test import rc_branch
        z = .8 + rc_branch(F,3,30) + 1j*W*2e-6
        peaks = [{"frequency_hz":30., "area_impedance":3.}]
        def solve(target):
            return {"reconstructed": target*.999, "peaks": peaks}
        result = advanced.model_reduce_analysis(F,z,z-1j*W*2e-6,peaks,solve)
        for c in result["candidates"][1:]:
            if c.get("status") == "completed":
                expected = advanced.schlueter_quality_indicator(z,c["addback_reconstructed_impedance"])
                self.assertAlmostEqual(c["quality"]["q_log10"], expected["q_log10"])

    def test_model_reduce_real_solver_recovers_rc_and_l(self):
        from pyimpspec import DataSet
        from pyimpspec.analysis.drt import calculate_drt_tr_rbf
        z=.8+3/(1+1j*W/(2*np.pi*30))+1j*W*2e-6
        def solve(target):
            r=calculate_drt_tr_rbf(DataSet(F,target),mode="complex",lambda_value=1e-4,
                 cross_validation="",rbf_type="gaussian",derivative_order=1,
                 inductance=False,num_procs=1,timeout=30)
            t,g=r.get_drt_data()
            peaks=batch.annotate_peak_support(batch.peak_basins(t,g),t,g,min(F),max(F))
            return {"reconstructed":r.get_impedances(),"tau_s":t,"gamma":g,
                    "peaks":[p for p in peaks if p.get("area_supported")]}
        baseline=solve(z)
        result=advanced.model_reduce_analysis(F,z,baseline["reconstructed"],baseline["peaks"],solve)
        self.assertEqual(result["selected_model"],"L")
        self.assertEqual(result["status"],"accepted")
        fit_l=result["selected"]["parameters"]["L_henry"]
        self.assertLess(abs(fit_l/2e-6-1),.05)
        dominant=max(result["selected"]["peaks"],key=lambda p:p["area_impedance"])
        self.assertLess(abs(math.log10(dominant["frequency_hz"]/30)),.15)
        self.assertLess(abs(dominant["area_impedance"]/3-1),.15)


class IOTests(unittest.TestCase):
    def test_same_filename_and_content_do_not_collide(self):
        a,b = identities("/tmp/a/eis.txt","abc","run1"),identities("/tmp/b/eis.txt","abc","run1")
        self.assertNotEqual(a["spectrum_uid"],b["spectrum_uid"])
        self.assertEqual(a,identities("/tmp/a/eis.txt","abc","run1"))

    def test_numeric_checkpoint_hash_detects_curve_edit(self):
        r=spectrum("a")
        before=numerical_digest(r)
        r["figure_status"]="failed"
        self.assertEqual(before,numerical_digest(r))
        r["curves"]["zreal_measured"][0] += .1
        self.assertNotEqual(before,numerical_digest(r))

    def test_manifest_allowlist_is_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b = Path(temp)/"eis.csv",Path(temp)/"old.csv"
            a.write_text("unknown-header\n1,2,3\n"); b.write_text("old-export\n")
            selected, inventory = inventory_inputs([a,b],{}, {str(a.resolve()): {"include":"true"}},manifest=Path(temp)/"manifest.csv")
            self.assertEqual(selected,[a]); self.assertFalse(inventory[1]["selected"])

    def test_drt_and_ocp_header_classification(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b = Path(temp)/"reference.csv",Path(temp)/"ocp.csv"
            a.write_text("tau_s,gamma\n1,2\n"); b.write_text("Time (s),Voltage (V)\n1,3\n")
            selected, inventory = inventory_inputs([a,b],{}, {})
            self.assertEqual(selected,[])
            self.assertEqual([r["input_role"] for r in inventory],["supplied-drt-or-export","process-or-ocp"])

    def test_conflicting_manifest_rows_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"manifest.csv"
            path.write_text("path,label\neis.txt,A\neis.txt,B\n")
            with self.assertRaisesRegex(ValueError,"Duplicate manifest"):
                batch.load_manifest(path)

    def test_resume_requires_matching_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"out"
            initialize_run(output,{"run_fingerprint":"a"})
            initialize_run(output,{"run_fingerprint":"a"},resume=True)
            with self.assertRaisesRegex(ValueError,"Resume refused"):
                initialize_run(output,{"run_fingerprint":"b"},resume=True)
            self.assertEqual(json.loads((output/"execution_contract.json").read_text())["run_fingerprint"],"a")

    def test_numerical_result_survives_plot_failure(self):
        r=spectrum("a")
        with patch.object(batch,"plot_spectrum",side_effect=RuntimeError("injected renderer failure")):
            r=batch.render_result(r,Path("unused"))
        self.assertEqual(r["status"],"completed")
        self.assertEqual(r["figure_status"],"failed")

    def test_nonfinite_json_is_standard(self):
        self.assertIsNone(batch.json_ready(np.float64(np.nan)))
        self.assertIsNone(batch.json_ready(np.float64(np.inf)))

    def test_unequal_tau_grids_not_silently_interpolated(self):
        rows=[spectrum("a"),spectrum("b")]
        rows[0]["curves"].update(tau_s=[1.,2.],gamma=[3.,4.])
        rows[1]["curves"].update(tau_s=[1.,3.,4.],gamma=[5.,6.,7.])
        with tempfile.TemporaryDirectory() as temp:
            export_wide(Path(temp),rows)
            self.assertEqual(len(list(Path(temp).glob("origin_*.csv"))),1)
            self.assertEqual(list(Path(temp).glob("heatmap_*.csv")),[])


if __name__ == "__main__":
    unittest.main(verbosity=2)

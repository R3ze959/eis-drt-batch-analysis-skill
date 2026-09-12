#!/usr/bin/env python3
"""Public-development synthetic checks for conditional peak evidence, not accuracy certification."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from pyimpspec import DataSet

from batch_drt import annotate_peak_support, peak_basins
from numerical_conditioning import calculate_tr_rbf_conditioned
from peak_evidence import assess_peak_evidence
from peak_matching import match_positions


DEVELOPMENT_SEEDS = (101, 202, 303, 404)
FREQUENCY = np.geomspace(1e5, 1e-2, 51)
LAMBDA = 10 ** -2.5
SOLVER = dict(mode="complex", lambda_value=LAMBDA, cross_validation="", rbf_type="gaussian",
              derivative_order=1, rbf_shape="fwhm", shape_coeff=.5, inductance=False,
              credible_intervals=False, num_procs=1, timeout=10)


def zarc(frequency, processes, series=1.):
    z = np.full(len(frequency), series, complex)
    for resistance, tau, exponent in processes:
        z += resistance / (1 + (2j*np.pi*frequency*tau) ** exponent)
    return z


def sample(processes, noise, seed, scale=1.):
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError("This development suite is intentionally restricted to its declared seeds")
    z = zarc(FREQUENCY, processes)
    rng = np.random.default_rng(seed)
    return (z + noise*abs(z)*(rng.normal(size=len(z))+1j*rng.normal(size=len(z))))*scale


def evaluate(z, *, seed=101, num_bootstrap=8, max_seconds=30.):
    result = calculate_tr_rbf_conditioned(DataSet(FREQUENCY, z), **SOLVER)
    tau, gamma = result.get_drt_data()
    peaks = annotate_peak_support(peak_basins(tau, gamma), tau, gamma, min(FREQUENCY), max(FREQUENCY))
    for i, peak in enumerate(peaks):
        peak["peak_id"] = f"P{i+1}"
    evidence = assess_peak_evidence(FREQUENCY, z, tau, gamma, peaks,
            lambda_value=LAMBDA, reconstructed=result.impedances, seed=seed,
            num_bootstrap=num_bootstrap, max_seconds=max_seconds)
    return result, peaks, evidence


def score(evidence, true_tau, tolerance=.35):
    supported = [p for p in evidence["per_peak"] if p["status"] == "perturbation-supported"]
    assignment = match_positions(true_tau, [p["tau_s"] for p in supported], tolerance)
    retained = len(assignment["matches"])
    raw_matched = match_positions(true_tau, [p["tau_s"] for p in evidence["per_peak"]], tolerance)
    persistent = [p for p in evidence["per_peak"] if p.get("stress_status") == "persistent"]
    persistent_matched = match_positions(true_tau, [p["tau_s"] for p in persistent], tolerance)
    return {"true_component_count": len(true_tau), "supported_lobes": len(supported),
            "matched_components": retained, "unmatched_supported_lobes": len(supported)-retained,
            "primary_matched_components": len(raw_matched["matches"]),
            "stress_persistent_matched_components": len(persistent_matched["matches"]),
            "elapsed_seconds": evidence["elapsed_seconds"]}


def development_matrix():
    """Only these public synthetic cases/seeds are used for development."""
    cases = {
        "single-zarc": ([(8., .01, .8)], .003),
        "separated-two-zarc": ([(5., .0003, .85), (4., .1, .8)], .005),
        "weak-real-zarc": ([(10., .001, .85), (.8, .2, .9)], .001),
        "overlapping-broad-zarc": ([(8., .01, .65), (7., .04, .65)], .003),
        "high-noise-single-zarc": ([(8., .01, .8)], .03),
        "noise-only-resistor": ([], .01),
    }
    rows = []
    for name, (processes, noise) in cases.items():
        for seed in DEVELOPMENT_SEEDS:
            _, _, evidence = evaluate(sample(processes, noise, seed), seed=seed)
            rows.append({"case": name, "seed": seed, "noise_fraction_per_component": noise,
                         **score(evidence, [p[1] for p in processes]),
                         "statuses": [{k:p[k] for k in ("tau_s", "status", "reasons")} for p in evidence["per_peak"]]})
    # Regression limits on these declared development fixtures only, not a
    # statistical guarantee or a requirement to force all spurious lobes away.
    for name, expected in (("single-zarc", 4), ("separated-two-zarc", 8),
                           ("high-noise-single-zarc", 4)):
        family = [r for r in rows if r["case"] == name]
        assert sum(r["matched_components"] for r in family) == expected, family
    noise_only = [r for r in rows if r["case"] == "noise-only-resistor"]
    assert sum(r["supported_lobes"] for r in noise_only) == 0, noise_only
    weak = [r for r in rows if r["case"] == "weak-real-zarc"]
    assert sum(r["primary_matched_components"] for r in weak) == 8, weak
    assert sum(r["stress_persistent_matched_components"] for r in weak) == 8, weak
    return {"status": "development-regressions-pass", "development_seeds": list(DEVELOPMENT_SEEDS), "rows": rows,
            "configuration": {"num_bootstrap": 8, "subset_count": 4, "lambda_value": LAMBDA, "frequency_points": 51},
            "matching_boundary": "Lobe-to-nominal-component matches are not proof of resolving overlapping physical processes.",
            "claim_boundary": "Small development matrix; no held-out performance, universal threshold or true-peak probability claim."}


class PeakEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.z = sample([(5., .0003, .85), (4., .1, .8)], .005, 101)
        cls.result, cls.peaks, cls.evidence = evaluate(cls.z)

    def test_resolved_main_lobes_retained(self):
        result = score(self.evidence, [.0003, .1])
        self.assertEqual(result["matched_components"], 2, result)

    def test_all_primary_peaks_and_inputs_are_preserved(self):
        old_peaks = copy.deepcopy(self.peaks)
        old_z = self.z.copy()
        old_gamma = self.result.gammas.copy()
        evidence = assess_peak_evidence(FREQUENCY, self.z, self.result.time_constants,
            self.result.gammas, self.peaks, lambda_value=LAMBDA,
            reconstructed=self.result.impedances, num_bootstrap=4)
        self.assertEqual(self.peaks, old_peaks)
        np.testing.assert_array_equal(self.z, old_z)
        np.testing.assert_array_equal(self.result.gammas, old_gamma)
        self.assertEqual(len(evidence["per_peak"]), len(self.peaks))
        self.assertFalse(evidence["raw_or_primary_modified"])

    def test_small_and_large_impedance_scales_keep_labels_and_restore_statistics(self):
        for scalar in (1e-4, 1e4):
            with self.subTest(scale=scalar):
                _, _, scaled = evaluate(self.z*scalar)
                self.assertEqual([p["status"] for p in scaled["per_peak"]],
                                 [p["status"] for p in self.evidence["per_peak"]])
                for before, after in zip(self.evidence["per_peak"], scaled["per_peak"]):
                    np.testing.assert_allclose(after["reference_contrast_impedance"]/scalar,
                                               before["reference_contrast_impedance"], rtol=2e-5, atol=1e-7)

    def test_support_is_never_physical_confirmation(self):
        for p in self.evidence["per_peak"]:
            self.assertEqual(p["physical_confirmation"], "not-established")
            self.assertEqual(p["uniquely_resolved_process"], "not-established")

    def test_exhausted_budget_cannot_be_support(self):
        _, _, result = evaluate(self.z, max_seconds=1e-12)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(all(p["status"] == "not-assessable" for p in result["per_peak"]))
        self.assertTrue(all(c["status"] == "not-run" for c in result["cases"]))

    def test_point_limit_does_not_silently_downsample(self):
        result = assess_peak_evidence(FREQUENCY, self.z, self.result.time_constants,
            self.result.gammas, self.peaks, lambda_value=LAMBDA, max_points=30)
        self.assertEqual(result["status"], "not-assessable")
        self.assertIn("not downsampled", result["reason"])
        self.assertEqual(result["cases"], [])

    def test_invalid_or_misaligned_inputs_fail_closed(self):
        for frequency in (FREQUENCY[:-1], np.zeros_like(FREQUENCY)):
            with self.assertRaises(ValueError):
                assess_peak_evidence(frequency, self.z, self.result.time_constants,
                                    self.result.gammas, self.peaks, lambda_value=LAMBDA)
        with self.assertRaises(ValueError):
            assess_peak_evidence(FREQUENCY, self.z, self.result.time_constants,
                self.result.gammas, self.peaks, lambda_value=LAMBDA, reconstructed=self.z[:-1])

    def test_weak_real_peak_is_retained_while_noise_model_remains_unverified(self):
        _, _, evidence = evaluate(sample([(10., .001, .85), (.8, .2, .9)], .001, 101))
        counts = score(evidence, [.001, .2])
        self.assertEqual(counts["primary_matched_components"], 2, counts)
        self.assertEqual(counts["stress_persistent_matched_components"], 2, counts)
        self.assertEqual(evidence["residual_model"]["status"], "serial-structure-flagged")
        self.assertEqual(counts["matched_components"], 0, counts)

    def test_overlapping_broad_components_are_not_declared_uniquely_resolved(self):
        _, _, evidence = evaluate(sample([(8., .01, .65), (7., .04, .65)], .003, 101))
        broad = min(evidence["per_peak"], key=lambda p: abs(np.log10(p["tau_s"]/.02)))
        self.assertTrue(.01 < broad["tau_s"] < .04)
        self.assertEqual(broad["stress_status"], "sensitive")
        self.assertIn("wild-residual-position-sensitive", broad["reasons"])
        self.assertEqual(broad["uniquely_resolved_process"], "not-established")
        # This fixture can retain a persistent side lobe: support is not truth.
        self.assertEqual(score(evidence, [.01, .04])["unmatched_supported_lobes"], 1)

    def test_noise_only_resistor_does_not_acquire_supported_relaxations(self):
        for seed in DEVELOPMENT_SEEDS:
            with self.subTest(seed=seed):
                _, _, evidence = evaluate(sample([], .01, seed), seed=seed)
                self.assertEqual(score(evidence, [])["supported_lobes"], 0)

    def test_noisy_single_zarc_preserves_known_main_peak(self):
        _, _, evidence = evaluate(sample([(8., .01, .8)], .03, 303), seed=303)
        self.assertEqual(score(evidence, [.01])["matched_components"], 1)


def public_development_example(path):
    """Explicit opt-in single allowed development result; no adjacent-file scan."""
    result = json.loads(path.read_text())
    if Path(result.get("source", "")).name != "2XZARCequal.csv":
        raise ValueError("Only the declared 2XZARCequal.csv development example is eligible here")
    curves = result["curves"]
    z = np.asarray(curves["zreal_measured"]) - 1j*np.asarray(curves["neg_zimag_measured"])
    fit = np.asarray(curves["zreal_reconstructed"]) - 1j*np.asarray(curves["neg_zimag_reconstructed"])
    evidence = assess_peak_evidence(curves["frequency_hz"], z, curves["tau_s"], curves["gamma"], result["peaks"],
                                   reconstructed=fit, lambda_value=result["drt"]["lambda"])
    target = min(evidence["per_peak"], key=lambda p: abs(np.log10(p["tau_s"]/.0022345832493900247)))
    assert target["physical_confirmation"] == "not-established"
    evidence["development_counterexample"] = {
        "scope": "Known supplied spurious lobe; a conditional support label must not become physical confirmation.",
        "tau_s": target["tau_s"], "conditional_status": target["status"],
        "physical_confirmation": target["physical_confirmation"],
        "contrast_to_perturbation_sd": target["families"].get("wild-residual", {}).get("reference_contrast_to_perturbation_sd")}
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-matrix", action="store_true")
    parser.add_argument("--public-result", type=Path)
    args, remaining = parser.parse_known_args()
    if args.development_matrix:
        print(json.dumps(development_matrix(), indent=2))
    elif args.public_result:
        print(json.dumps(public_development_example(args.public_result), indent=2))
    else:
        unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)

#!/usr/bin/env python3
"""Synthetic scale-covariance counterexamples for the DRT conditioning wrapper."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
import warnings
from unittest.mock import patch

import numpy as np
from pyimpspec import DataSet
from pyimpspec.analysis.drt import calculate_drt_tr_rbf
from pyimpspec.analysis.drt.tr_rbf import TRRBFResult

from numerical_conditioning import (
    calculate_tr_nnls_conditioned,
    calculate_tr_rbf_conditioned,
    conditioning_info,
)


F = np.geomspace(1e5, 1e-2, 51)
OPTIONS = dict(mode="complex", lambda_value=1e-4, cross_validation="", rbf_type="gaussian",
               derivative_order=1, rbf_shape="fwhm", shape_coeff=0.5,
               inductance=False, credible_intervals=False, timeout=30, num_procs=1)


def circuit(frequency=F, inductance=0.0):
    return 1.2 + 2/(1+1j*frequency/3000) + 5/(1+1j*frequency/3) + 1j*2*np.pi*frequency*inductance


def dataset(z=None, frequency=F):
    return DataSet(frequencies=frequency.copy(), impedances=(circuit(frequency) if z is None else z.copy()),
                   path="synthetic.csv", label="synthetic-only")


def area(result):
    return float(np.trapezoid(result.gammas, np.log(result.time_constants)))


class ConditioningTests(unittest.TestCase):
    def assert_covariant(self, z, *, frequency=F, inductance=False):
        settings = {**OPTIONS, "inductance": inductance}
        base = calculate_tr_rbf_conditioned(dataset(z, frequency), **settings)
        for scalar in (1e-4, 1e4):
            with self.subTest(scale=scalar, inductance=inductance):
                result = calculate_tr_rbf_conditioned(dataset(z*scalar, frequency), **settings)
                np.testing.assert_array_equal(result.time_constants, base.time_constants)
                np.testing.assert_allclose(result.gammas/scalar, base.gammas, rtol=2e-5, atol=2e-8)
                np.testing.assert_allclose(result.impedances/scalar, base.impedances, rtol=2e-6, atol=2e-8)
                self.assertAlmostEqual(area(result)/scalar/area(base), 1.0, places=6)
                self.assertAlmostEqual(result.pseudo_chisqr, base.pseudo_chisqr, places=8)
        return base

    def test_noiseless_two_rc_scale_covariance(self):
        result = self.assert_covariant(circuit())
        self.assertLess(abs(area(result)-7)/7, .02)

    def test_two_rc_plus_series_l_scale_covariance(self):
        self.assert_covariant(circuit(inductance=2.2e-6), inductance=True)

    def test_noisy_spectrum_scale_covariance(self):
        rng = np.random.default_rng(20260909)
        z = circuit()
        noise = .003*np.abs(z)*(rng.normal(size=z.size)+1j*rng.normal(size=z.size))
        self.assert_covariant(z+noise)

    def test_uneven_frequency_grid_scale_covariance(self):
        frequency = np.sort(np.r_[np.geomspace(1e5, 1e-2, 26), np.geomspace(5, 15, 15)])[::-1]
        self.assert_covariant(circuit(frequency), frequency=frequency)

    def test_off_mode_reproduces_unwrapped_solver(self):
        data = dataset(circuit()*1e-4)
        expected = calculate_drt_tr_rbf(data, **OPTIONS)
        result = calculate_tr_rbf_conditioned(data, impedance_scaling="off", **OPTIONS)
        np.testing.assert_array_equal(result.gammas, expected.gammas)
        np.testing.assert_array_equal(result.impedances, expected.impedances)
        self.assertEqual(result.pseudo_chisqr, expected.pseudo_chisqr)

    def test_input_arrays_mask_and_metadata_are_unchanged(self):
        data = dataset(circuit()*1e-4)
        data.set_mask({0: True, 12: True})
        original_f = data.get_frequencies(masked=None).copy()
        original_z = data.get_impedances(masked=None).copy()
        original_mask = data.get_mask().copy()
        data._codex_fixed_cache = {"sentinel": 7}
        result = calculate_tr_rbf_conditioned(data, **OPTIONS)
        np.testing.assert_array_equal(data.get_frequencies(masked=None), original_f)
        np.testing.assert_array_equal(data.get_impedances(masked=None), original_z)
        self.assertEqual(data.get_mask(), original_mask)
        self.assertEqual(data.get_path(), "synthetic.csv")
        self.assertEqual(data.get_label(), "synthetic-only")
        self.assertEqual(data._codex_fixed_cache, {"sentinel": 7})
        self.assertEqual(result.frequencies.size, F.size-2)
        self.assertEqual(conditioning_info(data)["included_points"], F.size-2)

    def test_ci_and_auto_lambda_are_restored_without_rescaling_relative_metrics(self):
        data = dataset()
        scale = conditioning_info(data)["impedance_scale"]
        fake = TRRBFResult(time_constants=np.array([.01, .1]), frequencies=F.copy(),
            impedances=circuit()/scale, residuals=np.full(F.shape, .001+1j*.002),
            pseudo_chisqr=.012, gammas=np.array([2., 5.]), mean_gammas=np.array([2.1, 5.1]),
            lower_bounds=np.array([1.8, 4.8]), upper_bounds=np.array([2.3, 5.3]), lambda_value=.003)
        with patch("pyimpspec.analysis.drt.calculate_drt_tr_rbf", return_value=fake) as solver:
            result = calculate_tr_rbf_conditioned(data, cross_validation="mgcv", credible_intervals=True, num_samples=123)
        forwarded = solver.call_args.args[0]
        self.assertIsNot(forwarded, data)
        np.testing.assert_array_equal(forwarded.get_frequencies(), data.get_frequencies())
        np.testing.assert_allclose(forwarded.get_impedances(), data.get_impedances()/scale)
        self.assertEqual(solver.call_args.kwargs, {"cross_validation": "mgcv", "credible_intervals": True, "num_samples": 123})
        for name in ("impedances", "gammas", "mean_gammas", "lower_bounds", "upper_bounds"):
            np.testing.assert_allclose(getattr(result, name), getattr(fake, name)*scale)
        np.testing.assert_array_equal(result.residuals, fake.residuals)
        self.assertEqual(result.pseudo_chisqr, fake.pseudo_chisqr)
        self.assertEqual(result.lambda_value, fake.lambda_value)

    def test_real_sampled_ci_is_finite_ordered_and_scale_covariant(self):
        # Bound the whole real sampler as well as its package-level timeout.
        # No mock is installed in this independent process.
        probe = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--real-ci-probe"],
                               capture_output=True, text=True, timeout=30)
        self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
        evidence = json.loads(probe.stdout)
        self.assertEqual(evidence["real_ci_solves"], 3)
        self.assertEqual(evidence["num_samples_per_solve"], 1000)
        self.assertGreater(evidence["ci_points"], 0)
        print(f"real conditional CI: 3 x 1000 samples in {evidence['seconds']:.3f}s; "
              "finite/order/unit-restoration checks only, not coverage calibration")

    def test_secondary_nnls_is_scale_covariant_and_series_l_invariant(self):
        base = calculate_tr_nnls_conditioned(dataset(), mode="real", lambda_value=-2.)
        scaled = calculate_tr_nnls_conditioned(dataset(circuit()*1e-4), mode="real", lambda_value=-2.)
        with_l = calculate_tr_nnls_conditioned(dataset(circuit(inductance=2.2e-6)), mode="real", lambda_value=-2.)
        np.testing.assert_allclose(scaled.gammas/1e-4, base.gammas, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(with_l.gammas, base.gammas, atol=1e-6, rtol=1e-5)

    def test_nonfinite_and_zero_inputs_are_rejected(self):
        for value in (0., np.nan, np.inf, 1j*np.inf):
            for mode in ("median", "off"):
                with self.subTest(value=value, mode=mode), self.assertRaises(ValueError):
                    conditioning_info(dataset(np.full(F.shape, value, dtype=complex)), mode)
        with self.assertRaises(ValueError):
            conditioning_info(dataset(), "unknown")

    def test_nonfinite_solver_output_is_rejected(self):
        fake = TRRBFResult(time_constants=np.array([.1]), frequencies=F.copy(),
            impedances=circuit(), residuals=np.zeros(F.shape, dtype=complex), pseudo_chisqr=.001,
            gammas=np.array([np.nan]), mean_gammas=np.array([]), lower_bounds=np.array([]),
            upper_bounds=np.array([]), lambda_value=1e-4)
        with patch("pyimpspec.analysis.drt.calculate_drt_tr_rbf", return_value=fake), self.assertRaises(RuntimeError):
            calculate_tr_rbf_conditioned(dataset(), **OPTIONS)


def print_counterexample():
    physical = dataset()
    small = dataset(circuit()*1e-4)
    for name, solver in (("original", calculate_drt_tr_rbf), ("median-conditioned", calculate_tr_rbf_conditioned)):
        base, scaled = solver(physical, **OPTIONS), solver(small, **OPTIONS)
        shape_error = np.linalg.norm(scaled.gammas/1e-4-base.gammas)/np.linalg.norm(base.gammas)
        print(f"{name}: gamma scale-covariance error={shape_error:.6g}; area={area(base):.8g}, rescaled-small-area={area(scaled)/1e-4:.8g}")


def real_ci_probe():
    """Exercise the real package sampler; this is not a coverage experiment."""
    started = time.monotonic()
    seed = 20260909
    frequency = np.geomspace(1e3, 1e-1, 15)
    # A broad passive response avoids the costly nearly-degenerate truncated
    # posterior of isolated delta-like RC peaks; no data are privately sourced.
    impedance = 1.2 + 5 / (1 + (1j * frequency / 10) ** .6)
    scale = conditioning_info(dataset(impedance, frequency))["impedance_scale"]
    settings = {**OPTIONS, "lambda_value": .01, "credible_intervals": True,
                "num_samples": 1000, "timeout": 10}
    checks = unittest.TestCase()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        np.random.seed(seed)
        normalized = calculate_drt_tr_rbf(dataset(impedance / scale, frequency), **settings)
        for scalar in (1., 2. ** -14):
            # Binary scaling makes the normalized observations identical, so a
            # fixed seed isolates restoration from Monte Carlo variability.
            np.random.seed(seed)
            physical = calculate_tr_rbf_conditioned(dataset(impedance * scalar, frequency), **settings)
            tau, mean, lower, upper = physical.get_drt_credible_intervals_data()
            checks.assertGreater(tau.size, 0)
            np.testing.assert_array_equal(tau, physical.time_constants)
            checks.assertTrue(np.all(tau > 0))
            for values in (mean, lower, upper):
                checks.assertEqual(values.shape, tau.shape)
                checks.assertTrue(np.all(np.isfinite(values)))
            checks.assertTrue(np.all(lower <= mean))
            checks.assertTrue(np.all(mean <= upper))
            checks.assertTrue(np.any(upper > lower))
            for name in ("impedances", "gammas", "mean_gammas", "lower_bounds", "upper_bounds"):
                np.testing.assert_allclose(getattr(physical, name) / (scale * scalar),
                                           getattr(normalized, name), rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(physical.residuals, normalized.residuals, rtol=1e-8, atol=1e-10)
            checks.assertAlmostEqual(physical.pseudo_chisqr, normalized.pseudo_chisqr, places=12)
            checks.assertEqual(physical.lambda_value, normalized.lambda_value)
    print(json.dumps({"real_ci_solves": 3, "num_samples_per_solve": 1000,
                      "ci_points": int(tau.size), "seed": seed,
                      "seconds": time.monotonic() - started,
                      "claim_boundary": "Implementation and unit consistency only; coverage is untested."}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--real-ci-probe"]:
        real_ci_probe()
    else:
        print_counterexample()
        unittest.main(verbosity=2)

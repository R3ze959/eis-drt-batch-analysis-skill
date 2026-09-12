#!/usr/bin/env python3
"""Synthetic regression suite for advanced EIS preprocessing, GDRT, and DDT."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from advanced_drt import (
    assess_batch_validity,
    continuous_signed_gdrt,
    ddt_analysis,
    detect_diffusion_tail,
    model_reduce_analysis,
    sweep_drift_screen,
)
from batch_drt import high_frequency_inductance, nonseries_inductive_region


def rc_branch(frequency_hz: np.ndarray, resistance: float, peak_hz: float) -> np.ndarray:
    omega = 2.0 * math.pi * frequency_hz
    tau = 1.0 / (2.0 * math.pi * peak_hz)
    return resistance / (1.0 + 1j * omega * tau)


def rl_branch(frequency_hz: np.ndarray, resistance: float, peak_hz: float) -> np.ndarray:
    omega = 2.0 * math.pi * frequency_hz
    tau = 1.0 / (2.0 * math.pi * peak_hz)
    return resistance * (1j * omega * tau) / (1.0 + 1j * omega * tau)


def warburg_semi(frequency_hz: np.ndarray, sigma: float) -> np.ndarray:
    return sigma / np.sqrt(1j * 2.0 * math.pi * frequency_hz)


def curve_result(label: str, impedance: np.ndarray, frequency_hz: np.ndarray, **metadata: str) -> dict:
    return {
        "status": "completed",
        "label": label,
        "metadata": metadata,
        "curves": {
            "frequency_hz": frequency_hz.tolist(),
            "zreal_measured": impedance.real.tolist(),
            "neg_zimag_measured": (-impedance.imag).tolist(),
        },
        "peaks": [{"peak_id": "P1", "frequency_hz": 30.0, "area_impedance": 3.0}],
    }


def write_spectrum(path: Path, frequency_hz: np.ndarray, impedance: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frequency (Hz)", "Z' (ohm)", "-Z'' (ohm)"])
        for frequency, value in zip(frequency_hz, impedance):
            writer.writerow([f"{frequency:.12g}", f"{value.real:.12g}", f"{-value.imag:.12g}"])


def main() -> int:
    frequency = np.logspace(5.0, -3.0, 61)
    omega = 2.0 * math.pi * frequency
    rc = 0.8 + rc_branch(frequency, 3.0, 30.0)
    rc_l = rc + 1j * omega * 2.0e-6
    rc_rl = rc + rl_branch(frequency, 1.5, 3.0)
    rc_warburg = rc + warburg_semi(frequency, 0.8)

    series_l, _ = high_frequency_inductance(frequency, -rc_l.imag)
    assert series_l is True, "RC+L must trigger the high-frequency series-L detector"
    residual_inductive, series_screen = nonseries_inductive_region(frequency, -rc_l.imag)
    assert residual_inductive is False, series_screen
    assert series_screen["series_inductance_estimate"]["series_l_explainable"] is True
    finite_inductive, _ = nonseries_inductive_region(frequency, -rc_rl.imag)
    assert finite_inductive is True, "RC+RL must trigger the finite-frequency inductive detector"

    signed = continuous_signed_gdrt(frequency, rc_rl, lambda_points=7)
    assert signed["status"] == "completed"
    assert float(np.min(signed["gamma"])) < -0.05
    assert any(row["branch"] == "RL-negative" for row in signed["peaks"])

    diffusion_trigger = detect_diffusion_tail(frequency, rc_warburg)
    assert diffusion_trigger["detected"] is True
    ddt = ddt_analysis(frequency, rc_warburg, lambda_points=7)
    assert ddt["status"] == "completed"
    assert {row["boundary"] for row in ddt["candidates"]} == {
        "blocking-open", "transmissive-short", "semi-infinite", "gerischer"
    }
    assert all(row["status"] == "completed" for row in ddt["candidates"])

    peak = [{"peak_id": "P1", "frequency_hz": 30.0, "area_impedance": 3.0}]

    def exact_reduced_solver(reduced: np.ndarray) -> dict:
        return {
            "reconstructed": np.asarray(reduced, dtype=complex),
            "peaks": peak,
            "tau_s": np.asarray([1.0 / (2.0 * math.pi * 30.0)]),
            "gamma": np.asarray([3.0]),
        }

    reduction = model_reduce_analysis(
        frequency,
        rc_l,
        rc,
        peak,
        exact_reduced_solver,
        min_quality_improvement_log10=0.01,
    )
    assert reduction["status"] == "accepted"
    assert reduction["selected_model"] == "L"
    assert reduction["selected"]["peak_stability"]["accepted"] is True

    rng = np.random.default_rng(20260827)
    noisy = rc * (1.0 + 0.002 * (rng.normal(size=rc.size) + 1j * rng.normal(size=rc.size)))
    drifted = rc * (1.0 + np.linspace(-0.04, 0.04, rc.size))
    noise_screen = sweep_drift_screen(np.arange(rc.size), noisy, rc)
    drift_screen = sweep_drift_screen(np.arange(rc.size), drifted, rc)
    assert noise_screen["status"] == "no-strong-monotonic-drift"
    assert drift_screen["status"] == "structured-drift-suspect"

    repeats = [
        curve_result("repeat-1", rc, frequency, cell_id="C1", state="S50", replicate="1"),
        curve_result("repeat-2", rc * 1.001, frequency, cell_id="C1", state="S50", replicate="2"),
        curve_result("repeat-bad", rc * 1.15, frequency, cell_id="C1", state="S50", replicate="3"),
    ]
    amplitudes = [
        curve_result("amp-5", rc, frequency, cell_id="C2", state="S50", ac_amplitude_mv="5", linearity_group="L1"),
        curve_result("amp-10", rc * 1.003, frequency, cell_id="C2", state="S50", ac_amplitude_mv="10", linearity_group="L1"),
    ]
    rests = [
        curve_result("rest-60", rc * 1.05, frequency, cell_id="C3", state="S50", rest_time_s="60", stationarity_group="T1"),
        curve_result("rest-600", rc * 1.002, frequency, cell_id="C3", state="S50", rest_time_s="600", stationarity_group="T1"),
        curve_result("rest-1800", rc, frequency, cell_id="C3", state="S50", rest_time_s="1800", stationarity_group="T1"),
    ]
    validity = assess_batch_validity(repeats + amplitudes + rests)
    assert validity["repeat_assessments"][0]["status"] == "repeat-outlier-detected"
    outliers = [
        row["label"]
        for row in validity["repeat_assessments"][0]["anomaly_scores"]
        if row["outlier"]
    ]
    assert outliers == ["repeat-bad"], outliers
    assert validity["linearity_assessments"][0]["status"] == "linearity-supported"
    assert validity["stationarity_assessments"][0]["status"] == "rest-stable"

    script = Path(__file__).with_name("batch_drt.py")
    with tempfile.TemporaryDirectory(prefix="advanced-eis-drt-") as temporary:
        root = Path(temporary)
        rl_path = root / "rc_rl.csv"
        warburg_path = root / "rc_warburg.csv"
        output = root / "out"
        write_spectrum(rl_path, frequency, rc_rl)
        write_spectrum(warburg_path, frequency, rc_warburg)
        command = [
            sys.executable, str(script), str(rl_path), str(warburg_path),
            "--output-dir", str(output),
            "--lambda-policy", "fixed", "--fixed-lambda", "1e-4", "--lambda-points", "7",
            "--no-credible-intervals", "--signed-gdrt", "always", "--ddt", "always",
            "--model-reduce", "--solver-timeout", "45", "--spectrum-timeout", "240",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=540)
        if completed.returncode != 0:
            raise AssertionError(
                f"advanced batch returned {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        results = [json.loads(path.read_text(encoding="utf-8")) for path in (output / "results").glob("*.json")]
        assert len(results) == 2
        for result in results:
            assert result["status"] == "completed", result.get("error")
            assert result["loewner_rc_rl"]["status"] == "completed"
            assert result["signed_gdrt"]["status"] == "completed"
            assert result["ddt"]["status"] == "completed"
            assert result["model_reduce"]["enabled"] is True
        for required in (
            "analysis_report.md", "batch_validation.json", "model_reduce_candidates.csv",
            "loewner_rc_rl.csv", "signed_gdrt_curves.csv", "ddt_curves.csv",
        ):
            path = output / required
            assert path.is_file() and path.stat().st_size > 0, path

    print(
        "PASS: RC, RC+L, RC+RL, RC+Warburg, random noise, monotonic drift, "
        "repeat outlier, amplitude linearity, rest stability, model-reduce, Loewner, signed/GDRT, and DDT"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded conditional noise and sampling-resolution evidence for RC-DRT peaks.

This module does not remove peaks, change the primary curve, estimate a true-peak
probability, or establish experimental repeatability. Fixed-model perturbation
and interleaved deletion fits are diagnostics, not calibrated hypothesis tests.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from numerical_conditioning import calculate_tr_rbf_conditioned
from peak_matching import match_positions


METHOD = "paired-wild-residual-and-interleaved-frequency-stress-v1"
CLAIM_BOUNDARY = (
    "Conditional on the chosen RC kernel, lambda, residual model and sampled frequency window. "
    "Perturbation support is neither a true-peak probability nor proof of a unique process. "
    "Residuals can contain regularization bias/model error; no bootstrap coverage or p-value is claimed."
)


def _area(tau, gamma, lower, upper):
    x = np.log(tau)
    lo, hi = max(math.log(lower), x[0]), min(math.log(upper), x[-1])
    if hi <= lo:
        return 0.0
    sample = np.r_[lo, x[(x > lo) & (x < hi)], hi]
    return float(np.trapezoid(np.interp(sample, x, gamma), sample))


def _lag(values):
    if len(values) < 4 or np.std(values[:-1]) < 1e-12 or np.std(values[1:]) < 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _geometry(tau, gamma, peaks, measured):
    raw_indices = [int(np.argmin(abs(np.log(tau / float(p["tau_s"]))))) for p in peaks]
    centers = sorted(set(raw_indices))
    output = []
    for number, (p, index) in enumerate(zip(peaks, raw_indices)):
        position = centers.index(index)
        left_neighbor = centers[position - 1] if position else 0
        right_neighbor = centers[position + 1] if position + 1 < len(centers) else len(tau) - 1
        left = left_neighbor + int(np.argmin(gamma[left_neighbor:index + 1]))
        right = index + int(np.argmin(gamma[index:right_neighbor + 1]))
        lower = max(float(tau[left]), measured[0])
        upper = min(float(tau[right]), measured[1])
        center = float(p["tau_s"])
        reason = None
        if not measured[0] <= center <= measured[1]:
            reason = "peak-outside-measured-window"
        elif not (left < index < right and lower < center < upper):
            reason = "no-two-sided-in-window-peak-basin"
        contrast = float(gamma[index] - max(np.interp(np.log(lower), np.log(tau), gamma),
                                           np.interp(np.log(upper), np.log(tau), gamma)))
        if contrast <= 0 and reason is None:
            reason = "no-positive-two-sided-contrast"
        output.append({"peak_id": p.get("peak_id", f"P{number + 1}"), "tau_s": center,
                       "frequency_hz": 1 / (2 * math.pi * center),
                       "basin_tau_min_s": lower, "basin_tau_max_s": upper,
                       "reference_contrast_impedance": contrast,
                       "reference_basin_area_impedance": _area(tau, gamma, lower, upper) if upper > lower else 0.,
                       "unassessable_reason": reason, "status": "not-assessable", "reasons": [],
                       "families": {}, "stress_status": "not-assessable", "noise_model_status": "not-assessed",
                       "physical_confirmation": "not-established", "uniquely_resolved_process": "not-established"})
    return output


def assess_peak_evidence(frequency_hz, impedance, tau_s, gamma, peaks=None, *,
                         lambda_value, reconstructed=None, inductance=False,
                         solver_options=None, seed=101, num_bootstrap=12,
                         subset_count=4, max_seconds=30., max_points=250):
    """Return separate evidence rows; never modify caller arrays or peak records.

    ``reconstructed`` must be the primary fit aligned to ``frequency_hz``. When
    omitted, one extra fixed-model fit obtains it. ``solver_options`` may specify
    the primary RBF/derivative/conditioning settings; lambda is held fixed.
    Outputs include every attempted case, physical contrast/area statistics and
    per-peak reasons. Defaults are explicit development heuristics, not universal
    acceptance limits. Runtime is bounded by solve count, point cap and a wall
    budget checked between solves (one ongoing library solve is not interrupted).
    """
    started = time.monotonic()
    f, z = np.asarray(frequency_hz, float), np.asarray(impedance, complex)
    t, g = np.asarray(tau_s, float), np.asarray(gamma, float)
    if f.ndim != 1 or z.shape != f.shape or t.ndim != 1 or g.shape != t.shape:
        raise ValueError("Aligned one-dimensional frequency/impedance and tau/gamma arrays are required")
    if (len(f) < 15 or len(t) < 3 or not np.all(np.isfinite(f)) or np.any(f <= 0)
            or not np.all(np.isfinite(z)) or not np.all(np.isfinite(t)) or np.any(t <= 0)
            or not np.all(np.isfinite(g)) or len(np.unique(f)) != len(f) or len(np.unique(t)) != len(t)):
        raise ValueError("Finite unique positive abscissae and at least 15 EIS points are required")
    if not math.isfinite(lambda_value) or lambda_value <= 0:
        raise ValueError("lambda_value must be finite and positive")
    if (not 4 <= num_bootstrap <= 64 or not 2 <= subset_count <= 8
            or not math.isfinite(max_seconds) or max_seconds <= 0 or max_points < 15):
        raise ValueError("Invalid bounded perturbation configuration")
    order, tau_order = np.argsort(f)[::-1], np.argsort(t)
    f, z, t, g = f[order], z[order], t[tau_order], g[tau_order]
    scale = float(np.median(abs(z)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("A positive finite impedance scale is required")
    if peaks is None:
        peaks = [{"tau_s": float(t[i])} for i in find_peaks(g)[0]]
    else:
        peaks = [dict(p) for p in peaks]
    if any(not math.isfinite(float(p.get("tau_s", 0))) or float(p.get("tau_s", 0)) <= 0 for p in peaks):
        raise ValueError("Every peak must have finite positive tau_s")
    window = (1 / (2 * math.pi * max(f)), 1 / (2 * math.pi * min(f)))
    rows = _geometry(t, g, peaks, window)
    settings = dict(mode="complex", lambda_value=float(lambda_value), cross_validation="",
                    rbf_type="gaussian", derivative_order=1, rbf_shape="fwhm", shape_coeff=.5,
                    inductance=bool(inductance), credible_intervals=False, num_procs=1,
                    timeout=10, impedance_scaling="median")
    allowed = {"rbf_type", "derivative_order", "rbf_shape", "shape_coeff", "impedance_scaling"}
    if solver_options:
        if set(solver_options) - allowed:
            raise ValueError("Unsupported fixed-model solver option")
        settings.update(solver_options)
    result = {"method": METHOD, "status": "pending", "claim_boundary": CLAIM_BOUNDARY,
              "configuration": {"lambda_value": float(lambda_value), "seed": int(seed),
                    "num_bootstrap": int(num_bootstrap), "subset_count": int(subset_count),
                    "max_points": int(max_points), "max_seconds": float(max_seconds),
                    "solver": settings, "match_decades": .35,
                    "bootstrap_support_fraction": .8, "subset_support_fraction": .75,
                    "position_q90_limit_decades": .25, "area_relative_iqr_limit": .5},
              "raw_or_primary_modified": False, "per_peak": rows, "cases": [],
              "residual_model": {"status": "not-assessed"}}
    def finish(status, reason=None):
        result.update(status=status, reason=reason, elapsed_seconds=time.monotonic()-started)
        return result
    if len(f) > max_points:
        for row in rows:
            row.update(status="not-assessable", reasons=["point-budget-exceeded"])
        return finish("not-assessable", "point-budget-exceeded; primary data were not downsampled")
    from pyimpspec import DataSet
    def solve(case_f, case_z):
        fitted = calculate_tr_rbf_conditioned(DataSet(case_f, case_z), **settings)
        ct, cg = fitted.get_drt_data()
        return np.asarray(ct, float), np.asarray(cg, float), np.asarray(fitted.get_impedances(), complex)
    if reconstructed is None:
        _, _, fit_z = solve(f, z)
    else:
        fit_z = np.asarray(reconstructed, complex)
        if fit_z.shape != order.shape or not np.all(np.isfinite(fit_z)):
            raise ValueError("The aligned primary reconstruction must be finite")
        fit_z = fit_z[order]
    residual = z - fit_z
    normalized = residual / np.maximum(abs(z), scale * 1e-12)
    lag = max(abs(_lag(normalized.real)), abs(_lag(normalized.imag)))
    result["residual_model"] = {"method": "paired-complex Rademacher wild residuals about fixed primary fit",
        "relative_complex_rms_percent": float(100 * np.sqrt(np.mean(abs(normalized)**2))),
        "lag1_max_abs": lag, "serial_structure_flag": lag > .5,
        "status": "serial-structure-flagged" if lag > .5 else "no-large-serial-flag",
        "assumptions": "Mean-zero errors; independent frequency-point multipliers. Real/imaginary pair and local variance pattern retained.",
        "limitation": "Fitted residuals are not an unbiased noise estimate; overfit/bias/correlation can invalidate inferential readings."}
    rng = np.random.default_rng(seed)
    jobs = [("wild-residual", i, np.ones(len(f), bool), fit_z + residual*rng.choice([-1., 1.], len(f)))
            for i in range(num_bootstrap)]
    for offset in range(subset_count):
        keep = np.ones(len(f), bool)
        keep[np.arange(1, len(f)-1)[np.arange(len(f)-2) % subset_count == offset]] = False
        jobs.append(("interleaved-deletion", offset, keep, z))
    for family, number, keep, perturbed in jobs:
        case = {"family": family, "case": number, "point_count": int(keep.sum())}
        if time.monotonic()-started >= max_seconds:
            case.update(status="not-run", error="wall-budget-exhausted")
        elif keep.sum() < 15:
            case.update(status="not-run", error="insufficient-points-after-deletion")
        else:
            try:
                ct, cg, _ = solve(f[keep], perturbed[keep])
                cp = find_peaks(cg)[0]
                matching = match_positions([r["tau_s"] for r in rows], ct[cp], .35)
                observations = []
                for i, row in enumerate(rows):
                    match = matching["matches"].get(i)
                    values = np.interp(np.log([row["basin_tau_min_s"], row["tau_s"], row["basin_tau_max_s"]]), np.log(ct), cg)
                    contrast = float(values[1]-max(values[0], values[2]))
                    observations.append({"peak_id": row["peak_id"], "matched": match is not None,
                        "position_shift_decades": match[1] if match else None,
                        "contrast_impedance": contrast,
                        "basin_area_impedance": _area(ct, cg, row["basin_tau_min_s"], row["basin_tau_max_s"]),
                        "split_merge_ambiguous": matching["resolution_change_left"][i]})
                case.update(status="completed", peak_count=len(cp), peaks=observations)
            except Exception as exc:
                case.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        result["cases"].append(case)
    complete = [c for c in result["cases"] if c["status"] == "completed"]
    all_complete = len(complete) == len(jobs)
    for i, row in enumerate(rows):
        summaries = {}
        for family in ("wild-residual", "interleaved-deletion"):
            obs = [c["peaks"][i] for c in complete if c["family"] == family]
            if not obs:
                summaries[family] = {"completed_cases": 0}
                continue
            positions = [o["position_shift_decades"] for o in obs if o["matched"]]
            areas = np.asarray([o["basin_area_impedance"] for o in obs])
            contrasts = np.asarray([o["contrast_impedance"] for o in obs])
            summaries[family] = {"completed_cases": len(obs),
                "match_fraction": sum(o["matched"] for o in obs)/len(obs),
                "positive_contrast_fraction": float(np.mean(contrasts > max(scale*1e-12, 0.))),
                "contrast_q10_impedance": float(np.quantile(contrasts, .1)),
                "contrast_standard_deviation_impedance": float(np.std(contrasts, ddof=1)) if len(contrasts) > 1 else None,
                "reference_contrast_to_perturbation_sd": (float(row["reference_contrast_impedance"] / np.std(contrasts, ddof=1))
                    if len(contrasts) > 1 and np.std(contrasts, ddof=1) > scale*1e-12 else None),
                "position_shift_q90_decades": float(np.quantile(positions, .9)) if positions else None,
                "basin_area_median_impedance": float(np.median(areas)),
                "basin_area_relative_iqr": float(np.ptp(np.quantile(areas, [.25, .75]))/max(abs(row["reference_basin_area_impedance"]), scale*1e-12)),
                "split_merge_fraction": sum(o["split_merge_ambiguous"] for o in obs)/len(obs)}
        row["families"] = summaries
        reasons = []
        if row["unassessable_reason"]:
            reasons.append(row["unassessable_reason"])
        if not all_complete:
            reasons.append("incomplete-perturbation-ensemble")
        if lag > .5:
            reasons.append("serial-residual-model-unverified")
        for family, fraction in (("wild-residual", .8), ("interleaved-deletion", .75)):
            s = summaries[family]
            if not s["completed_cases"]:
                continue
            if s["match_fraction"] < fraction or s["positive_contrast_fraction"] < fraction:
                reasons.append(family + "-peak-not-persistent")
            if s["position_shift_q90_decades"] is None or s["position_shift_q90_decades"] > .25:
                reasons.append(family + "-position-sensitive")
            if s["basin_area_relative_iqr"] > .5:
                reasons.append(family + "-area-sensitive")
        status = "perturbation-supported" if not reasons else "noise-or-resolution-sensitive"
        if row["unassessable_reason"] or not all_complete:
            status = "not-assessable"
        stress_reasons = [reason for reason in reasons if reason != "serial-residual-model-unverified"]
        row.update(status=status, reasons=reasons,
                   stress_status="not-assessable" if row["unassessable_reason"] or not all_complete else
                                 "sensitive" if stress_reasons else "persistent",
                   noise_model_status=result["residual_model"]["status"],
                   physical_confirmation="not-established", uniquely_resolved_process="not-established",
                   interpretation="A persistent numerical lobe, not an identified or uniquely resolved physical process." if not reasons else CLAIM_BOUNDARY)
    return finish("completed" if all_complete else "incomplete")

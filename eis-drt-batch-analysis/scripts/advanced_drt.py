#!/usr/bin/env python3
"""Advanced, auditable preprocessing and generalized inversion helpers.

The functions in this module never overwrite raw impedance data.  They return
parallel diagnostic branches that the batch driver serializes beside the
ordinary non-negative RC-DRT result.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable, Iterable
from xml.etree import ElementTree

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment, lsq_linear


Array = np.ndarray


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _complex_stack(values: Array) -> Array:
    values = np.asarray(values, dtype=complex)
    return np.concatenate((values.real, values.imag))


def relative_complex_rms(reference: Array, comparison: Array) -> float:
    """Return RMS complex discrepancy as percent of pointwise |reference|."""
    reference = np.asarray(reference, dtype=complex)
    comparison = np.asarray(comparison, dtype=complex)
    floor = max(float(np.median(np.abs(reference))) * 1e-12, np.finfo(float).tiny)
    relative = np.abs(comparison - reference) / np.maximum(np.abs(reference), floor)
    return float(100.0 * np.sqrt(np.mean(np.square(relative))))


def sweep_drift_screen(
    acquisition_index: Array,
    measured: Array,
    reconstructed: Array,
    acquisition_order_available: bool = True,
) -> dict[str, Any]:
    """Screen for a monotonic within-sweep residual trend, distinct from random noise."""
    if not acquisition_order_available:
        return {
            "status": "not-assessable",
            "reason": "original acquisition order is unavailable",
            "claim_boundary": "No stationarity conclusion can be drawn from frequency-sorted points alone.",
        }
    acquisition_index = np.asarray(acquisition_index, dtype=float)
    measured = np.asarray(measured, dtype=complex)
    reconstructed = np.asarray(reconstructed, dtype=complex)
    if acquisition_index.size != measured.size or measured.size < 15:
        return {"status": "not-assessable", "reason": "insufficient aligned acquisition points"}
    order = np.argsort(acquisition_index)
    measured = measured[order]
    reconstructed = reconstructed[order]
    floor = max(float(np.median(np.abs(measured))) * 1e-12, np.finfo(float).tiny)
    residual = (measured - reconstructed) / np.maximum(np.abs(measured), floor)
    x = np.linspace(-0.5, 0.5, measured.size)
    design = np.column_stack((np.ones(x.size), x))
    coeff_re, *_ = np.linalg.lstsq(design, residual.real, rcond=None)
    coeff_im, *_ = np.linalg.lstsq(design, residual.imag, rcond=None)
    trend = design @ coeff_re + 1j * (design @ coeff_im)
    detrended = residual - trend
    trend_span_percent = float(100.0 * math.hypot(coeff_re[1], coeff_im[1]))
    detrended_rms_percent = float(100.0 * np.sqrt(np.mean(np.abs(detrended) ** 2)))
    residual_rms_percent = float(100.0 * np.sqrt(np.mean(np.abs(residual) ** 2)))
    trend_to_noise = trend_span_percent / max(detrended_rms_percent, np.finfo(float).tiny)
    suspect = trend_span_percent > 0.5 and trend_to_noise > 1.5
    return {
        "status": "structured-drift-suspect" if suspect else "no-strong-monotonic-drift",
        "trend_span_percent": trend_span_percent,
        "detrended_rms_percent": detrended_rms_percent,
        "residual_rms_percent": residual_rms_percent,
        "trend_to_noise_ratio": trend_to_noise,
        "criterion": "trend span >0.5% and trend/detrended-RMS >1.5",
        "claim_boundary": (
            "This detects a monotonic acquisition-order residual signature only. It cannot distinguish "
            "cell drift from temperature, contact, or model-mismatch causes without repeats/time controls."
        ),
    }


def schlueter_quality_indicator(preprocessed: Array, reconstructed: Array) -> dict[str, float]:
    """Schlüter-style weighted reconstruction quality indicator.

    The real and imaginary deviations are modulus-weighted, condensed by their
    Euclidean norms, and reported on a base-10 logarithmic scale.  The exact
    formula used here is stored in every result rather than hidden behind a
    generic "quality" label.
    """
    preprocessed = np.asarray(preprocessed, dtype=complex)
    reconstructed = np.asarray(reconstructed, dtype=complex)
    floor = max(float(np.median(np.abs(preprocessed))) * 1e-12, np.finfo(float).tiny)
    weights = 1.0 / np.maximum(np.abs(preprocessed), floor)
    delta_re = float(np.linalg.norm(weights * (preprocessed.real - reconstructed.real)))
    delta_im = float(np.linalg.norm(weights * (preprocessed.imag - reconstructed.imag)))
    combined = math.sqrt(delta_re * delta_re + delta_im * delta_im) / math.sqrt(
        max(2 * preprocessed.size, 1)
    )
    return {
        "q_log10": float(math.log10(max(combined, np.finfo(float).tiny))),
        "weighted_norm_re": delta_re,
        "weighted_norm_im": delta_im,
        "relative_complex_rms_percent": relative_complex_rms(preprocessed, reconstructed),
        "definition": (
            "log10(sqrt(||W(Re(Zpre-Zrec))||^2+||W(Im(Zpre-Zrec))||^2)/sqrt(2N)), "
            "W=1/max(|Zpre|,floor))"
        ),
    }


def _peak_frequency(peak: dict[str, Any]) -> float | None:
    value = finite_float(peak.get("frequency_hz"))
    if value is None or value <= 0.0:
        tau = finite_float(peak.get("tau_s"))
        if tau is not None and tau > 0.0:
            value = 1.0 / (2.0 * math.pi * tau)
    return value if value is not None and value > 0.0 else None


def _peak_area(peak: dict[str, Any]) -> float | None:
    for key in ("area_impedance", "in_window_area_impedance", "area", "gamma"):
        value = finite_float(peak.get(key))
        if value is not None:
            return abs(value)
    return None


def assess_peak_stability(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    position_limit_decades: float = 0.35,
    area_fraction_limit: float = 0.50,
) -> dict[str, Any]:
    """One-to-one peak matching and preprocessing acceptance assessment."""
    before_rows = [(i, p, _peak_frequency(p)) for i, p in enumerate(before)]
    after_rows = [(i, p, _peak_frequency(p)) for i, p in enumerate(after)]
    before_rows = [row for row in before_rows if row[2] is not None]
    after_rows = [row for row in after_rows if row[2] is not None]
    matches: list[dict[str, Any]] = []
    matched_before: set[int] = set()
    matched_after: set[int] = set()
    if before_rows and after_rows:
        from peak_matching import match_positions
        # Optimize only pairs that can actually pass. Loose diagnostic matches
        # must not consume peaks needed for a valid position/area assignment.
        area_allowed = np.ones((len(before_rows), len(after_rows)), dtype=bool)
        for i, (_, bp, _) in enumerate(before_rows):
            for j, (_, ap, _) in enumerate(after_rows):
                a, b = _peak_area(bp), _peak_area(ap)
                if a is not None and b is not None and a > 0:
                    area_allowed[i, j] = abs(b-a)/a <= area_fraction_limit
        assignment = match_positions([r[2] for r in before_rows], [r[2] for r in after_rows],
                                     position_limit_decades, allowed_pairs=area_allowed)
        for row_index, (col_index, distance) in assignment["matches"].items():
            bi, bp, bf = before_rows[row_index]
            ai, ap, af = after_rows[col_index]
            before_area = _peak_area(bp)
            after_area = _peak_area(ap)
            area_change = None
            if before_area is not None and after_area is not None and before_area > 0.0:
                area_change = abs(after_area - before_area) / before_area
            stable = distance <= position_limit_decades and (
                area_change is None or area_change <= area_fraction_limit
            )
            matches.append({
                "before_index": bi,
                "after_index": ai,
                "before_peak_id": bp.get("peak_id", f"B{bi + 1}"),
                "after_peak_id": ap.get("peak_id", f"A{ai + 1}"),
                "before_frequency_hz": bf,
                "after_frequency_hz": af,
                "position_shift_decades": distance,
                "before_area": before_area,
                "after_area": after_area,
                "area_change_fraction": area_change,
                "stable": stable,
            })
            matched_before.add(bi)
            matched_after.add(ai)
    unmatched_before = [
        p.get("peak_id", f"B{i + 1}") for i, p, _ in before_rows if i not in matched_before
    ]
    unmatched_after = [
        p.get("peak_id", f"A{i + 1}") for i, p, _ in after_rows if i not in matched_after
    ]
    stable_count = sum(bool(row["stable"]) for row in matches)
    denominator = max(len(before_rows), 1)
    stable_fraction = stable_count / denominator
    accepted = (
        stable_fraction >= 2.0 / 3.0
        and len(unmatched_after) <= max(1, int(math.ceil(0.25 * max(len(after_rows), 1))))
    )
    return {
        "status": "accepted" if accepted else "changed-materially",
        "accepted": accepted,
        "before_peak_count": len(before_rows),
        "after_peak_count": len(after_rows),
        "stable_match_count": stable_count,
        "stable_fraction_of_before": stable_fraction,
        "position_limit_decades": position_limit_decades,
        "area_fraction_limit": area_fraction_limit,
        "matches": matches,
        "unmatched_before": unmatched_before,
        "introduced_after": unmatched_after,
    }


def _difference_operator(size: int, order: int = 1) -> Array:
    matrix = np.eye(size)
    for _ in range(max(1, int(order))):
        matrix = np.diff(matrix, axis=0)
    return matrix


def _lcurve_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report whether an interior chord-distance choice is actually bracketed.

    Excluding endpoint solutions does not make their immediate neighbors safe
    from a truncated search. A flat/straight log-log curve also supplies no
    resolved corner, even though a midpoint can still produce a useful curve.
    """
    count = len(rows)
    lambdas = [row.get("lambda") for row in rows]
    distinct = len(set(lambdas)) if all(value is not None for value in lambdas) else count
    diagnostics = {
        "status": "insufficient-search",
        "selected_index": count // 2,
        "successful_candidate_count": count,
        "distinct_lambda_count": distinct,
        "eligible_indices": list(range(1, count - 1)),
        "boundary_sensitive": True,
        "edge_adjacent": False,
        "chord_distances": [],
        "selected_chord_distance": None,
        "log_span_tolerance": 1e-12,
        "normalized_chord_tolerance": 1e-8,
        "reason": "At least three distinct successful lambda candidates are needed to assess a corner.",
    }
    if count < 3 or distinct < 3:
        return diagnostics
    x = np.log10(np.maximum([row["residual_norm"] for row in rows], np.finfo(float).tiny))
    y = np.log10(np.maximum([row["roughness_norm"] for row in rows], np.finfo(float).tiny))
    x_span, y_span = float(np.ptp(x)), float(np.ptp(y))
    diagnostics.update(residual_log10_span=x_span, roughness_log10_span=y_span)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or min(x_span, y_span) <= diagnostics["log_span_tolerance"]:
        diagnostics.update(status="degenerate", reason="Residual or roughness has no resolved finite log-scale variation.")
        return diagnostics
    x = (x - min(x)) / x_span
    y = (y - min(y)) / y_span
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    denom = math.hypot(x1 - x0, y1 - y0)
    if denom <= np.finfo(float).tiny:
        diagnostics.update(status="degenerate", reason="L-curve endpoints do not define a resolved chord.")
        return diagnostics
    distances = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / denom
    diagnostics["chord_distances"] = distances.tolist()
    if float(np.max(distances[1:-1])) <= diagnostics["normalized_chord_tolerance"]:
        diagnostics.update(status="degenerate", reason="Normalized log-log L-curve is indistinguishable from its endpoint chord.")
        return diagnostics
    choice = int(1 + np.argmax(distances[1:-1]))
    edge_adjacent = choice in {1, count - 2}
    diagnostics.update(
        status="edge-adjacent" if edge_adjacent else "corner",
        selected_index=choice,
        selected_chord_distance=float(distances[choice]),
        edge_adjacent=edge_adjacent,
        boundary_sensitive=edge_adjacent,
        reason=("Selected corner touches the eligible interior boundary; expand the lambda search."
                if edge_adjacent else "Selected corner is separated from both eligible interior boundaries."),
    )
    return diagnostics


def _lcurve_choice(rows: list[dict[str, Any]]) -> int:
    """Keep the integer-selection API; detailed search evidence is separate."""
    return int(_lcurve_diagnostics(rows)["selected_index"])


def _solve_distribution(
    frequency_hz: Array,
    target: Array,
    kernel_builder: Callable[[Array, Array], Array],
    *,
    signed: bool,
    lambda_grid: Iterable[float],
    derivative_order: int = 1,
    tau_extension_decades: float = 0.7,
    include_series_l: bool = True,
    include_series_c: bool = True,
    num_tau: int | None = None,
) -> dict[str, Any]:
    """Convergence-checked, scaled inversion; passive nuisance terms for DDT."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    target = np.asarray(target, dtype=complex)
    if frequency_hz.size < 5 or target.shape != frequency_hz.shape:
        raise ValueError("At least five aligned EIS points are required")
    if not np.all(np.isfinite(frequency_hz)) or not np.all(frequency_hz > 0) or not np.all(np.isfinite(target)):
        raise ValueError("Nonfinite impedance or invalid frequency")
    order = np.argsort(frequency_hz)[::-1]
    frequency_hz, target = frequency_hz[order], target[order]
    omega = 2 * math.pi * frequency_hz
    measured_min, measured_max = 1 / max(omega), 1 / min(omega)
    num_tau = int(num_tau or np.clip(3 * frequency_hz.size, 80, 180))
    tau = np.geomspace(measured_min / 10**tau_extension_decades,
                       measured_max * 10**tau_extension_decades, num_tau)
    dln = float(np.mean(np.diff(np.log(tau))))
    kernel = np.asarray(kernel_builder(omega, tau), complex) * dln
    singular = np.linalg.svd(np.vstack((kernel.real, kernel.imag)), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * max(kernel.shape) * np.finfo(float).eps))
    if rank <= 1:
        raise ValueError("Rank-one kernel cannot identify a time distribution; fit a scalar amplitude")
    nuisance_columns = [np.ones(frequency_hz.size, complex)]
    nuisance_names = ["R_inf_ohm"]
    if include_series_l:
        nuisance_columns.append(1j * omega / max(omega))
        nuisance_names.append("L_scaled_ohm")
    if include_series_c:
        nuisance_columns.append(min(omega) / (1j * omega))
        nuisance_names.append("C_reciprocal_scaled_ohm")
    design_complex = np.column_stack([*nuisance_columns, kernel])
    scale = max(float(np.median(np.abs(target))), np.finfo(float).tiny)
    weights = 1 / np.maximum(np.abs(target), scale * 1e-6)
    design = np.vstack((design_complex.real * weights[:, None], design_complex.imag * weights[:, None]))
    response = _complex_stack(target * weights)
    nuisance_count = len(nuisance_names)
    diff = _difference_operator(num_tau, derivative_order) / dln**derivative_order * math.sqrt(dln)
    penalty = np.zeros((len(diff), nuisance_count + num_tau))
    penalty[:, nuisance_count:] = diff / scale
    lower = np.zeros(nuisance_count + num_tau)
    if signed:
        lower[0] = -np.inf  # algebraic offset; not a separately identified passive resistor
        lower[nuisance_count:] = -np.inf
    upper = np.full_like(lower, np.inf)
    rows, solutions = [], []
    for value in lambda_grid:
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Regularization values must be finite and positive")
        matrix = np.vstack((design, math.sqrt(value) * penalty))
        rhs = np.concatenate((response, np.zeros(len(diff))))
        column_scale = np.maximum(np.linalg.norm(matrix, axis=0), np.finfo(float).tiny)
        fit = lsq_linear(matrix / column_scale, rhs, bounds=(lower * column_scale, upper),
                         method="bvls", tol=1e-10, max_iter=1000)
        solution = fit.x / column_scale
        converged = bool(fit.success and np.all(np.isfinite(solution)))
        row = {"lambda": value, "optimizer_success": bool(fit.success),
               "optimizer_status": int(fit.status), "optimizer_message": str(fit.message),
               "optimality": float(fit.optimality), "converged": converged}
        if not converged:
            row.update(status="failed", residual_norm=None, roughness_norm=None)
            rows.append(row)
            solutions.append(None)
            continue
        reconstructed = design_complex @ solution
        gamma = solution[nuisance_count:]
        row.update(status="completed",
                   residual_norm=max(relative_complex_rms(target, reconstructed) / 100, np.finfo(float).tiny),
                   roughness_norm=max(float(np.linalg.norm(diff @ gamma) / scale), np.finfo(float).tiny))
        rows.append(row)
        solutions.append(solution)
    successful = [i for i, row in enumerate(rows) if row["converged"]]
    if not successful:
        return {"status": "failed", "error": "No regularization candidate converged",
                "lambda_grid": rows, "solver": {"converged": False}, "tau_s": [], "gamma": []}
    search = _lcurve_diagnostics([rows[i] for i in successful])
    choice = successful[search["selected_index"]]
    search.update(selected_grid_index=choice, selected_lambda=rows[choice]["lambda"],
                  successful_grid_indices=successful, failed_candidate_count=len(rows)-len(successful),
                  requested_lambda_min=min(row["lambda"] for row in rows),
                  requested_lambda_max=max(row["lambda"] for row in rows))
    solution = solutions[choice]
    nuisance = {name: float(solution[i]) for i, name in enumerate(nuisance_names)}
    if "L_scaled_ohm" in nuisance:
        nuisance["L_henry"] = nuisance["L_scaled_ohm"] / max(omega)
    if "C_reciprocal_scaled_ohm" in nuisance:
        reciprocal = nuisance["C_reciprocal_scaled_ohm"] * min(omega)
        nuisance["series_C_farad"] = 1 / reciprocal if reciprocal > 0 else None
    gamma = solution[nuisance_count:]
    absolute_area = float(np.trapezoid(np.abs(gamma), np.log(tau)))
    in_window = (tau >= measured_min) & (tau <= measured_max)
    supported_area = float(np.trapezoid(np.abs(gamma[in_window]), np.log(tau[in_window])))
    return {"status": "completed", "tau_s": tau, "frequency_hz": 1 / (2 * math.pi * tau),
            "gamma": gamma, "reconstructed": design_complex @ solution,
            "measured_frequency_hz": frequency_hz, "lambda": rows[choice]["lambda"],
            "lambda_selection": "normalized-interior-L-curve-chord", "lambda_grid": rows,
            "lambda_search": search,
            "lambda_boundary_sensitive": search["boundary_sensitive"],
            "residual_rms_percent": relative_complex_rms(target, design_complex @ solution),
            "nuisance": nuisance, "signed": signed, "num_tau": num_tau,
            "distribution_measure": "gamma per natural ln(tau); integral has impedance units",
            "support": {"tau_min_s": float(measured_min), "tau_max_s": float(measured_max),
                        "absolute_area": absolute_area,
                        "supported_absolute_area_fraction": supported_area / max(absolute_area, 1e-300)},
            "solver": {**rows[choice], "passive_nuisance_bounds": not signed,
                       "signed_offset_is_algebraic": signed,
                       "kernel_numeric_rank": rank, "kernel_columns": num_tau,
                       "kernel_condition_number": float(singular[0] / max(singular[-1], 1e-300)),
                       "weighting": "pointwise impedance modulus; scaled derivative in ln(tau)",
                       "nuisance_at_lower_bound": [n for i, n in enumerate(nuisance_names)
                                                  if lower[i] == 0 and solution[i] < scale * 1e-8]}}


def _distribution_sensitivity(result, frequency_hz, impedance, kernel, *, signed, derivative_order,
                              include_series_c, lambda_min, lambda_max):
    """Check lambda and an independent coarser tau grid, without altering data."""
    if result.get("status") != "completed":
        return result
    cases = []
    requested_low, requested_high = result["lambda"] / 10, result["lambda"] * 10
    low, high = float(np.clip(requested_low, lambda_min, lambda_max)), float(np.clip(requested_high, lambda_min, lambda_max))
    full_neighborhood = (math.isclose(low, requested_low, rel_tol=1e-12)
                         and math.isclose(high, requested_high, rel_tol=1e-12))
    tests = [("lambda-low", low, result["num_tau"], requested_low),
             ("lambda-high", high, result["num_tau"], requested_high),
             ("tau-grid-coarse", result["lambda"], max(40, int(result["num_tau"] * 0.75)), result["lambda"])]
    t0, g0 = np.asarray(result["tau_s"]), np.asarray(result["gamma"])
    peak0 = float(t0[np.argmax(np.abs(g0))])
    for name, value, count, requested_value in tests:
        alternative = _solve_distribution(frequency_hz, impedance, kernel, signed=signed,
                      lambda_grid=[value], derivative_order=derivative_order, num_tau=count,
                      include_series_c=include_series_c)
        row = {"case": name, "lambda": value, "requested_lambda": requested_value,
               "range_clipped": not math.isclose(value, requested_value, rel_tol=1e-12),
               "num_tau": count, "status": alternative["status"]}
        if alternative["status"] == "completed":
            t, g = np.asarray(alternative["tau_s"]), np.asarray(alternative["gamma"])
            shift = abs(math.log10(float(t[np.argmax(np.abs(g))]) / peak0))
            area0 = result["support"]["absolute_area"]
            area_change = abs(alternative["support"]["absolute_area"] - area0) / max(area0, 1e-300)
            row.update(dominant_peak_shift_decades=shift, absolute_area_change_fraction=area_change,
                       stable=bool(shift <= 0.35 and area_change <= 0.5),
                       tau_s=t, gamma=g, residual_rms_percent=alternative["residual_rms_percent"])
        cases.append(row)
    # This checks dominant-lobe/grid stability, not individual-process uniqueness.
    evaluated_cases_stable = all(r.get("stable") for r in cases)
    result["sensitivity"] = {
        "status": "range-incomplete" if not full_neighborhood else "stable" if evaluated_cases_stable else "sensitive",
        "full_neighborhood_available": full_neighborhood,
        "requested_lambda_range": [requested_low, requested_high],
        "evaluated_lambda_range": [low, high],
        "configured_lambda_range": [lambda_min, lambda_max],
        "evaluated_cases_stable": evaluated_cases_stable,
        "cases": cases,
        "claim_boundary": "Dominant absolute lobe and integrated mass only; a clipped lambda neighborhood cannot establish full plus/minus-one-decade stability or unique peak decomposition.",
    }
    return result


def _signed_local_peaks(tau: Array, gamma: Array, relative_threshold: float = 0.03) -> list[dict[str, Any]]:
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    rows: list[dict[str, Any]] = []
    for sign_name, values, sign_value in (("RC-positive", gamma, 1.0), ("RL-negative", -gamma, -1.0)):
        maximum = float(np.max(values)) if values.size else 0.0
        if maximum <= 0.0:
            continue
        for index in range(1, len(values) - 1):
            if values[index] < relative_threshold * maximum:
                continue
            if values[index] >= values[index - 1] and values[index] > values[index + 1]:
                rows.append({
                    "branch": sign_name,
                    "tau_s": float(tau[index]),
                    "frequency_hz": float(1.0 / (2.0 * math.pi * tau[index])),
                    "gamma": float(sign_value * values[index]),
                })
    rows.sort(key=lambda row: row["frequency_hz"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["peak_id"] = f"G{index}"
    return rows


def continuous_signed_gdrt(
    frequency_hz: Array,
    impedance: Array,
    lambda_min: float = 1e-7,
    lambda_max: float = 1e-1,
    lambda_points: int = 13,
    derivative_order: int = 1,
) -> dict[str, Any]:
    """Continuous signed RC-kernel GDRT; negative lobes encode RL-like terms."""
    def kernel(omega: Array, tau: Array) -> Array:
        return 1.0 / (1.0 + 1j * omega[:, None] * tau[None, :])

    result = _solve_distribution(
        frequency_hz,
        impedance,
        kernel,
        signed=True,
        lambda_grid=np.logspace(math.log10(lambda_min), math.log10(lambda_max), lambda_points),
        derivative_order=derivative_order,
    )
    result = _distribution_sensitivity(result, frequency_hz, impedance, kernel, signed=True,
             derivative_order=derivative_order, include_series_c=True,
             lambda_min=lambda_min, lambda_max=lambda_max)
    result.update({
        "method": "continuous-signed-RC-kernel-GDRT",
        "interpretation": (
            "positive gamma is RC-like; negative gamma is RL-like after the fitted R_inf offset. "
            "Sign alone is not a mechanism assignment."
        ),
        "peaks": _signed_local_peaks(result["tau_s"], result["gamma"]),
    })
    if result.get("status") == "completed":
        support = result["support"]
        for peak in result["peaks"]:
            inside = support["tau_min_s"] <= peak["tau_s"] <= support["tau_max_s"]
            distance = min(abs(math.log10(peak["tau_s"] / support["tau_min_s"])),
                           abs(math.log10(support["tau_max_s"] / peak["tau_s"])))
            peak.update(measured_window_supported=inside,
                        support_status="unsupported-outside-window" if not inside else
                                       "boundary-sensitive" if distance < 0.7 else "in-window",
                        mechanism_assignment="unassigned")
    return result


def predictive_rc_kernel_competition(
    frequency_hz: Array,
    impedance: Array,
    *,
    lambda_min: float = 1e-7,
    lambda_max: float = 1e-1,
    lambda_points: int = 9,
    num_tau: int | None = None,
    folds: int = 3,
    derivative_order: int = 1,
) -> dict[str, Any]:
    """Independent RC+R/L/C reconstruction candidates with frequency-point CV.

    Positive and signed gamma share the SAME grid, modulus weighting, passive
    nuisance bounds, folds and lambda grid. Only gamma's sign constraint changes.
    CV tests interpolation within one sweep, not independent measurement validity
    or the physical reality of lobes. All original points are used in the final
    fits; their order is preserved. Existing primary and L-curve paths are intact.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    measured = np.asarray(impedance, dtype=complex)
    if frequency.ndim != 1 or measured.shape != frequency.shape or frequency.size < 12:
        raise ValueError("Prediction selection requires at least twelve aligned EIS points")
    if not np.isfinite(frequency).all() or not (frequency > 0).all() or not np.isfinite(measured).all():
        raise ValueError("Prediction selection requires finite impedance and positive finite frequency")
    if (not math.isfinite(lambda_min) or not math.isfinite(lambda_max)
            or not 0 < lambda_min < lambda_max or int(lambda_points) != lambda_points or lambda_points < 3):
        raise ValueError("Use a positive increasing lambda range and at least three grid points")
    if int(folds) != folds or not 2 <= folds <= 5 or derivative_order not in (1, 2):
        raise ValueError("Use two to five folds and derivative order one or two")
    unique, inverse = np.unique(frequency, return_inverse=True)
    if unique.size < max(10, 2 * folds + 2):
        raise ValueError("Too few distinct frequencies for endpoint-preserving prediction folds")
    count = int(np.clip(2 * frequency.size, 60, 120)) if num_tau is None else int(num_tau)
    if count < 20 or (num_tau is not None and count != num_tau):
        raise ValueError("num_tau must be an integer of at least twenty")
    omega = 2 * math.pi * frequency
    tau_min, tau_max = 1 / max(omega), 1 / min(omega)
    tau = np.geomspace(tau_min / 10**.7, tau_max * 10**.7, count)
    dln = float(np.mean(np.diff(np.log(tau))))
    kernel = dln / (1 + 1j * omega[:, None] * tau[None, :])
    design = np.column_stack((np.ones(frequency.size), 1j * omega / max(omega),
                              min(omega) / (1j * omega), kernel))
    diff = _difference_operator(count, derivative_order) / dln**derivative_order * math.sqrt(dln)
    penalty = np.column_stack((np.zeros((len(diff), 3)), diff))
    grid = np.logspace(math.log10(lambda_min), math.log10(lambda_max), int(lambda_points))
    all_indices = np.arange(frequency.size)
    # Repeated frequencies stay together: no same-frequency training leakage.
    fold_rows = []
    for fold in range(int(folds)):
        held_groups = np.arange(1, unique.size - 1)[np.arange(unique.size - 2) % folds == fold]
        test = all_indices[np.isin(inverse, held_groups)]
        train = all_indices[~np.isin(inverse, held_groups)]
        fold_rows.append((train, test))

    def fit_at(indices: Array, value: float, signed: bool) -> dict[str, Any]:
        reference = measured[indices]
        scale = max(float(np.median(np.abs(reference))), np.finfo(float).tiny)
        normalized = reference / scale
        weights = 1 / np.maximum(np.abs(normalized), 1e-6)
        weighted = design[indices] * weights[:, None] / math.sqrt(indices.size)
        # Preserve the existing advanced all-point lambda scale while giving
        # folds the same data/penalty tradeoff despite fewer training points.
        matrix = np.vstack((weighted.real, weighted.imag, math.sqrt(value / frequency.size) * penalty))
        rhs = np.concatenate((_complex_stack(normalized * weights) / math.sqrt(indices.size),
                              np.zeros(len(penalty))))
        norms = np.maximum(np.linalg.norm(matrix, axis=0), np.finfo(float).tiny)
        lower = np.zeros(count + 3)
        if signed:
            lower[3:] = -np.inf
        try:
            fit = lsq_linear(matrix / norms, rhs, bounds=(lower, np.inf),
                             method="bvls", tol=1e-10, max_iter=1000)
            coefficients = fit.x / norms * scale
            prediction = design @ coefficients
            good = bool(fit.success and np.isfinite(coefficients).all() and np.isfinite(prediction).all())
            return {"status": "completed" if good else "failed", "coefficients": coefficients,
                    "reconstructed": prediction, "optimizer_success": bool(fit.success),
                    "optimizer_status": int(fit.status), "optimality": float(fit.optimality),
                    "scale_ohm": scale}
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    def mass(values: Array) -> tuple[float, float]:
        absolute = float(np.trapezoid(np.abs(values), np.log(tau)))
        negative = float(np.trapezoid(np.maximum(-values, 0), np.log(tau)))
        return absolute, negative

    candidates = {}
    for name, signed in (("positive_rc", False), ("signed_rc", True)):
        search_rows = []
        for value in grid:
            errors, fold_errors, successes = [], [], []
            for train, test in fold_rows:
                fit = fit_at(train, float(value), signed)
                success = fit["status"] == "completed"
                successes.append(success)
                if success:
                    floor = max(float(np.median(np.abs(measured[train]))) * 1e-6, np.finfo(float).tiny)
                    residual = np.abs(fit["reconstructed"][test] - measured[test]) / np.maximum(np.abs(measured[test]), floor)
                    errors.extend(np.square(residual).tolist())
                    fold_errors.append(float(100 * np.sqrt(np.mean(np.square(residual)))))
                else:
                    fold_errors.append(None)
            complete = all(successes)
            search_rows.append({"lambda": float(value), "status": "completed" if complete else "failed",
                                "prediction_rms_percent": float(100 * np.sqrt(np.mean(errors))) if complete else None,
                                "fold_rms_percent": fold_errors, "all_folds_converged": complete})
        successful = [i for i, row in enumerate(search_rows) if row["status"] == "completed"]
        if not successful:
            candidates[name] = {"status": "failed", "lambda_grid": search_rows,
                                "error": "No lambda completed every prediction fold"}
            continue
        best_index = min(successful, key=lambda i: search_rows[i]["prediction_rms_percent"])
        best_error = search_rows[best_index]["prediction_rms_percent"]
        # A declared engineering tie-break, NOT an estimated noise variance or CI.
        eligible = [i for i in successful if search_rows[i]["prediction_rms_percent"] <= 1.05 * best_error]
        selected_index = max(eligible)
        value = float(grid[selected_index])
        fit = fit_at(all_indices, value, signed)
        if fit["status"] != "completed":
            candidates[name] = {"status": "failed", "lambda_grid": search_rows,
                                "error": "Selected lambda failed the all-point refit"}
            continue
        coefficients = fit["coefficients"]
        gamma = coefficients[3:]
        total_mass, negative_mass = mass(gamma)
        inside = (tau >= tau_min) & (tau <= tau_max)
        supported_mass = float(np.trapezoid(np.abs(gamma[inside]), np.log(tau[inside])))
        boundary = selected_index in (0, len(grid) - 1)
        sensitivity = []
        for factor in (.1, 10.):
            requested = value * factor
            actual = float(np.clip(requested, lambda_min, lambda_max))
            alternative = fit_at(all_indices, actual, signed)
            row = {"case": "lambda-low" if factor < 1 else "lambda-high", "num_tau": count,
                   "lambda": actual, "requested_lambda": requested,
                   "range_clipped": not math.isclose(requested, actual, rel_tol=1e-12),
                   "status": alternative["status"]}
            if alternative["status"] == "completed":
                alternate_gamma = alternative["coefficients"][3:]
                alternate_mass, alternate_negative = mass(alternate_gamma)
                row.update(residual_rms_percent=relative_complex_rms(measured, alternative["reconstructed"]),
                           absolute_area_change_fraction=abs(alternate_mass - total_mass) / max(total_mass, 1e-300),
                           negative_absolute_area_fraction=alternate_negative / max(alternate_mass, 1e-300),
                           tau_s=tau.copy(), gamma=alternate_gamma,
                           measured_frequency_hz=frequency.copy(), reconstructed=alternative["reconstructed"])
            sensitivity.append(row)
        nuisance = {"R_inf_ohm": float(coefficients[0]),
                    "L_henry": float(coefficients[1] / max(omega)),
                    "C_reciprocal_scaled_ohm": float(coefficients[2]),
                    "series_C_farad": float(1 / (coefficients[2] * min(omega))) if coefficients[2] > 0 else None}
        refit_rms = relative_complex_rms(measured, fit["reconstructed"])
        prediction_rms = search_rows[selected_index]["prediction_rms_percent"]
        refit_guard = bool(refit_rms <= 2 * prediction_rms + .05)
        reliable = bool(refit_guard and refit_rms <= 2 and prediction_rms <= 2)
        reliability_reasons = []
        if not refit_guard:
            reliability_reasons.append("full-refit-degrades-versus-prediction")
        if max(refit_rms, prediction_rms) > 2:
            reliability_reasons.append("prediction-or-refit-above-2-percent-heuristic")
        if boundary:
            reliability_reasons.append("lambda-search-boundary")
        candidates[name] = {
            "status": "completed", "method": f"predictive-{'signed' if signed else 'positive'}-RC-RLC",
            "reconstruction_branch": f"predictive_{name}", "distribution_branch": f"predictive_{name}",
            "tau_s": tau.copy(), "frequency_hz": 1 / (2 * math.pi * tau), "gamma": gamma,
            "measured_frequency_hz": frequency.copy(), "reconstructed": fit["reconstructed"],
            "residual_rms_percent": refit_rms, "prediction_rms_percent": prediction_rms,
            "cv_rms_percent": prediction_rms,
            "cv": {"status": "completed", "all_folds_converged": True, "fold_count": folds,
                   "rms_percent": prediction_rms, "endpoints_validated": False,
                   "selection_not_independent_external_validation": True},
            "full_refit_guard": {"status": "pass" if refit_guard else "review-required",
                                 "converged": True, "point_count": int(frequency.size),
                                 "rule": "full-refit RMS <= 2*prediction RMS + 0.05 percentage point"},
            "reconstruction_reliable": reliable,
            "reliability_status": "usable-numerical-diagnostic" if reliable else "review-required",
            "reliability_reasons": reliability_reasons,
            "reliability_rule": "all folds and full refit converge; prediction and refit <= 2 percent; full-refit guard; lambda boundary reported separately; engineering guards, not acceptance or confidence limits",
            "lambda": value, "lambda_grid": search_rows,
            "lambda_selection": "largest-lambda-within-5-percent-of-minimum-interleaved-prediction-RMS",
            "lambda_search": {"selected_index": selected_index, "minimum_prediction_index": best_index,
                              "boundary_sensitive": boundary, "failed_candidate_count": len(grid) - len(successful)},
            "lambda_boundary_sensitive": boundary, "signed": signed, "num_tau": count, "nuisance": nuisance,
            "support": {"tau_min_s": float(tau_min), "tau_max_s": float(tau_max),
                        "absolute_area": total_mass, "negative_absolute_area_fraction": negative_mass / max(total_mass, 1e-300),
                        "supported_absolute_area_fraction": supported_mass / max(total_mass, 1e-300)},
            "sensitivity": {"status": "range-incomplete" if any(r["range_clipped"] for r in sensitivity) else "evaluated-not-peak-validated",
                            "cases": sensitivity, "individual_peak_stability_assessed": False},
            "solver": {"converged": True, "passive_nuisance_bounds": True,
                       "weighting": "pointwise impedance modulus; not noise-variance weights",
                       "conditioning": "training-only uniform median-modulus scalar; physical units restored",
                       "objective": "mean squared modulus-relative complex residual + lambda/N_full times squared ln-tau derivative norm",
                       "optimizer_status": fit["optimizer_status"], "optimality": fit["optimality"]},
            "distribution_measure": "gamma per natural ln(tau); integral has impedance units",
            "negative_lobes_are_mechanism_evidence": False,
        }
    completed = [name for name, result in candidates.items() if result["status"] == "completed"]
    chosen = None
    if completed:
        chosen = min(completed, key=lambda name: candidates[name]["prediction_rms_percent"])
        if "positive_rc" in completed:
            positive_error = candidates["positive_rc"]["prediction_rms_percent"]
            best_error = candidates[chosen]["prediction_rms_percent"]
            if positive_error <= best_error * 1.10 + .2:
                chosen = "positive_rc"
    return {
        "status": "completed" if chosen else "failed", "method": "same-kernel-positive-signed-frequency-prediction-competition",
        "selected_branch": f"predictive_{chosen}" if chosen else None, "selected_candidate": chosen,
        "selection_scope": "impedance-reconstruction-only; not physical-model acceptance",
        "selection_is_physical_acceptance": False, "negative_lobes_are_mechanism_evidence": False,
        "selection_rule": "Prefer positive RC within 10 percent plus 0.2 percentage point of the best frequency-prediction RMS; heuristic, not a statistical test",
        "candidates": candidates,
        "cross_validation": {"kind": "interleaved-interior-frequency-points", "fold_count": folds,
                             "folds": [{"train_indices": train.tolist(), "heldout_indices": test.tolist()} for train, test in fold_rows],
                             "frequency_duplicates_grouped": True, "endpoints_always_in_training": True,
                             "independent_measurement_validation": False, "raw_points_removed": 0,
                             "final_refit_point_count": int(frequency.size)},
        "claim_boundary": "In-sweep interpolation selects reconstruction regularization only; unvalidated endpoints, correlated noise, nuisance/tail confounding and non-unique signed lobes remain. No repeat, amplitude, stationarity or physical-peak acceptance is inferred.",
    }


def loewner_rc_rl_analysis(data: Any, num_procs: int = 1) -> dict[str, Any]:
    """Run pyimpspec's Loewner framework and expose both RC and RL branches."""
    try:
        from pyimpspec.analysis.drt import calculate_drt_lm

        result = calculate_drt_lm(data, model_order=0, model_order_method="matrix_rank", num_procs=num_procs)
        tau_rc, gamma_rc, tau_rl, gamma_rl = result.get_drt_data()
        tau_rc = np.asarray(tau_rc, dtype=float)
        gamma_rc = np.asarray(gamma_rc, dtype=float)
        tau_rl = np.asarray(tau_rl, dtype=float)
        gamma_rl = np.asarray(gamma_rl, dtype=float)
        rc_mask = np.isfinite(tau_rc) & (tau_rc > 0.0) & np.isfinite(gamma_rc)
        rl_mask = np.isfinite(tau_rl) & (tau_rl > 0.0) & np.isfinite(gamma_rl)
        tau_rc, gamma_rc = tau_rc[rc_mask], gamma_rc[rc_mask]
        tau_rl, gamma_rl = tau_rl[rl_mask], gamma_rl[rl_mask]
        reconstructed = np.asarray(result.get_impedances(), dtype=complex)
        return {
            "status": "completed",
            "method": "pyimpspec-Loewner",
            "model_order": int(len(result.time_constants)),
            "pseudo_chisqr": float(result.pseudo_chisqr),
            "tau_rc_s": tau_rc,
            "gamma_rc": gamma_rc,
            "tau_rl_s": tau_rl,
            "gamma_rl": gamma_rl,
            "reconstructed": reconstructed,
            "purpose": "discrete RC/RL signed-response diagnostic",
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def diffusion_kernel(boundary: str, omega: Array, tau: Array) -> Array:
    """Normalized finite diffusion/reaction-diffusion kernels."""
    s = np.sqrt(1j * omega[:, None] * tau[None, :])
    if boundary == "blocking-open":
        return 1.0 / (s * np.tanh(s))  # coth(s)/s
    if boundary == "transmissive-short":
        return np.tanh(s) / s
    if boundary == "semi-infinite":
        return 1.0 / s
    if boundary == "gerischer":
        return 1.0 / np.sqrt(1.0 + 1j * omega[:, None] * tau[None, :])
    raise ValueError(f"Unknown diffusion boundary: {boundary}")


def detect_diffusion_tail(frequency_hz: Array, impedance: Array) -> dict[str, Any]:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)
    order = np.argsort(frequency_hz)
    count = max(5, int(math.ceil(0.18 * frequency_hz.size)))
    indices = order[:count]
    low_f = frequency_hz[indices]
    low_z = impedance[indices]
    x = low_z.real
    y = -low_z.imag
    dx = float(np.ptp(x))
    dy = float(np.ptp(y))
    slope = dy / dx if dx > np.finfo(float).tiny else math.inf
    magnitude_slope = float(np.polyfit(np.log10(low_f), np.log10(np.maximum(np.abs(low_z), np.finfo(float).tiny)), 1)[0])
    detected = (
        np.median(y) > 0.0
        and (0.15 <= slope <= 8.0 or magnitude_slope < -0.20)
        and float(np.ptp(np.abs(low_z))) > 0.02 * max(float(np.median(np.abs(impedance))), np.finfo(float).tiny)
    )
    return {
        "detected": bool(detected),
        "low_frequency_points": count,
        "nyquist_tail_slope_abs": slope,
        "log_magnitude_vs_log_frequency_slope": magnitude_slope,
        "criterion": "low-frequency growth with capacitive/diffusive orientation; screening heuristic only",
    }


def _dominant_distribution_peak(tau: Array, gamma: Array) -> dict[str, float] | None:
    gamma = np.asarray(gamma, dtype=float)
    if gamma.size == 0 or float(np.max(gamma)) <= 0.0:
        return None
    index = int(np.argmax(gamma))
    return {
        "tau_s": float(tau[index]),
        "frequency_hz": float(1.0 / (2.0 * math.pi * tau[index])),
        "gamma": float(gamma[index]),
    }


def fit_semi_infinite_amplitude(frequency_hz, impedance):
    """Rank-one Warburg baseline: Z=R+jwL+sigma/sqrt(jw); no tau distribution."""
    frequency = np.asarray(frequency_hz, float)
    order = np.argsort(frequency)[::-1]
    frequency, target = frequency[order], np.asarray(impedance, complex)[order]
    omega = 2 * math.pi * frequency
    matrix = np.column_stack((np.ones(len(omega)), 1j * omega / max(omega), 1 / np.sqrt(1j * omega)))
    weights = 1 / np.maximum(np.abs(target), max(float(np.median(abs(target))) * 1e-6, 1e-300))
    design = np.vstack((matrix.real * weights[:, None], matrix.imag * weights[:, None]))
    fit = lsq_linear(design, _complex_stack(target * weights), bounds=(0, np.inf), method="bvls", tol=1e-10)
    converged = bool(fit.success and np.all(np.isfinite(fit.x)))
    reconstructed = matrix @ fit.x
    return {"status": "completed" if converged else "failed", "boundary": "semi-infinite",
            "model": "R + series-L + scalar-Warburg", "distribution_identifiable": False,
            "identifiability_reason": "K(omega,tau)=omega-factor*tau-factor has rank 1",
            "dominant_peak": None, "tau_s": [], "gamma": [], "frequency_hz": [],
            "sigma_impedance_per_sqrt_s": float(fit.x[2]),
            "nuisance": {"R_inf_ohm": float(fit.x[0]), "L_henry": float(fit.x[1] / max(omega))},
            "reconstructed": reconstructed, "measured_frequency_hz": frequency,
            "residual_rms_percent": relative_complex_rms(target, reconstructed),
            "solver": {"converged": converged, "optimizer_status": int(fit.status),
                       "optimality": float(fit.optimality), "optimizer_message": str(fit.message)},
            "eligible_for_distribution_ranking": False}


def ddt_analysis(
    frequency_hz: Array,
    impedance: Array,
    lambda_min: float = 1e-7,
    lambda_max: float = 1e-1,
    lambda_points: int = 13,
    derivative_order: int = 1,
) -> dict[str, Any]:
    """Compare conditional series-impedance diffusion models, with explicit gates."""
    detection = detect_diffusion_tail(frequency_hz, impedance)
    candidates = []
    for boundary in ("blocking-open", "transmissive-short", "semi-infinite", "gerischer"):
        try:
            if boundary == "semi-infinite":
                result = fit_semi_infinite_amplitude(frequency_hz, impedance)
            else:
                kernel = lambda omega, tau, name=boundary: diffusion_kernel(name, omega, tau)
                result = _solve_distribution(frequency_hz, impedance, kernel, signed=False,
                         lambda_grid=np.logspace(math.log10(lambda_min), math.log10(lambda_max), lambda_points),
                         derivative_order=derivative_order, include_series_l=True, include_series_c=False)
                result.update(boundary=boundary, distribution_identifiable=False)
                if result["status"] == "completed":
                    result = _distribution_sensitivity(result, frequency_hz, impedance, kernel, signed=False,
                             derivative_order=derivative_order, include_series_c=False,
                             lambda_min=lambda_min, lambda_max=lambda_max)
                    peak = _dominant_distribution_peak(result["tau_s"], result["gamma"])
                    support = result["support"]
                    # A finite kernel can still mimic its asymptote outside the measured turnover.
                    turnover_supported = bool(peak and support["tau_min_s"] * 10**0.35 <= peak["tau_s"]
                                               <= support["tau_max_s"] / 10**0.35)
                    reasons = []
                    if result["residual_rms_percent"] > 2.0:
                        reasons.append("reconstruction-above-2-percent-heuristic")
                    if not turnover_supported:
                        reasons.append("dominant-turnover-not-covered")
                    if result["sensitivity"]["status"] != "stable":
                        reasons.append("lambda-or-tau-grid-sensitive")
                    if support["supported_absolute_area_fraction"] < 0.8:
                        reasons.append("substantial-outside-window-mass")
                    if result["lambda_boundary_sensitive"]:
                        reasons.append("lambda-search-boundary")
                    result.update(dominant_peak=peak, turnover_supported=turnover_supported,
                                  distribution_identifiable=not reasons,
                                  identifiability_scope="conditional on selected series kernel and regularization; not a unique physical distribution",
                                  eligible_for_distribution_ranking=not reasons, rejection_reasons=reasons)
            candidates.append(result)
        except Exception as exc:
            candidates.append({"status": "failed", "boundary": boundary,
                               "eligible_for_distribution_ranking": False,
                               "error": f"{type(exc).__name__}: {exc}"})
    completed = [r for r in candidates if r["status"] == "completed"]
    eligible = [r for r in completed if r.get("eligible_for_distribution_ranking")]
    best = min(eligible, key=lambda r: r["residual_rms_percent"]) if eligible else None
    attempt = min(completed, key=lambda r: r["residual_rms_percent"]) if completed else None
    competitive = [r["boundary"] for r in eligible if best and
                   r["residual_rms_percent"] <= best["residual_rms_percent"] + max(0.2, best["residual_rms_percent"] * 0.2)]
    return {"status": "completed" if completed else "failed", "trigger": detection, "candidates": candidates,
            "best_boundary": best["boundary"] if best else None, "best": best,
            "competitive_boundaries": competitive,
            "boundary_resolution": "ambiguous" if len(competitive) > 1 else "one-eligible-numerical-best" if best else "unresolved",
            "lowest_residual_attempt_boundary": attempt["boundary"] if attempt else None,
            "selection_status": "eligible-numerical-candidate" if best else "no-distribution-candidate-accepted",
            "domain": "series diffusion-kernel distribution in impedance domain",
            "topology": "R + series L + integral of diffusion impedances; no separately identified RC branch",
            "claim_boundary": "Not Song-Bazant parallel admittance DDT. RC overlap can invalidate this topology. "
                              "Scalar semi-infinite baseline has no identifiable diffusion time. "
                              "No boundary mechanism or diffusion coefficient is established by residual ranking."}


def _nuisance_component(frequency_hz: Array, model: str, params: dict[str, float]) -> Array:
    omega = 2.0 * math.pi * np.asarray(frequency_hz, dtype=float)
    result = np.zeros(omega.shape, dtype=complex)
    if model.startswith("L+") or model == "L":
        result += 1j * omega * params["L_henry"]
    low_model = model[2:] if model.startswith("L+") else (None if model == "L" else model)
    if low_model == "C":
        result += 1.0 / (1j * omega * params["C_farad"])
    elif low_model == "warburg-semi":
        result += params["sigma"] / np.sqrt(1j * omega)
    elif low_model in {"warburg-short", "warburg-open"}:
        s = np.sqrt(1j * omega * params["tau_d_s"])
        if low_model == "warburg-short":
            result += params["R_w_ohm"] * np.tanh(s) / s
        else:
            result += params["R_w_ohm"] / (s * np.tanh(s))
    return result


def _nuisance_eligibility(frequency_hz: Array, impedance: Array) -> dict[str, Any]:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)
    high = frequency_hz >= float(np.max(frequency_hz)) / 10.0
    low = frequency_hz <= float(np.min(frequency_hz)) * (10.0 ** 1.5)
    high_inductive = int(np.sum(impedance.imag[high] > 0.0)) >= 3
    low_y = -impedance.imag[low]
    if np.sum(low_y > 0.0) >= 4:
        c_slope = float(np.polyfit(
            np.log10(frequency_hz[low][low_y > 0.0]),
            np.log10(low_y[low_y > 0.0]),
            1,
        )[0])
    else:
        c_slope = math.nan
    diffusion = detect_diffusion_tail(frequency_hz, impedance)
    return {
        "series_L": high_inductive,
        "series_C": bool(math.isfinite(c_slope) and -1.35 <= c_slope <= -0.65),
        "diffusion": bool(diffusion["detected"]),
        "series_C_log_slope": c_slope,
        "diffusion_evidence": diffusion,
    }


def _fit_nuisance_model(frequency_hz: Array, impedance: Array, model: str) -> dict[str, Any]:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)
    omega = 2.0 * math.pi * frequency_hz
    high = frequency_hz >= float(np.max(frequency_hz)) / 10.0
    low = frequency_hz <= float(np.min(frequency_hz)) * (10.0 ** 1.5)
    use_high = model.startswith("L")
    use_low = model != "L"
    mask = (high if use_high else np.zeros(high.shape, dtype=bool)) | (
        low if use_low else np.zeros(low.shape, dtype=bool)
    )
    names: list[str] = ["R_edge_ohm"]
    initial: list[float] = [float(np.median(impedance.real[mask]))]
    lower: list[float] = [-10.0 * max(float(np.median(np.abs(impedance))), 1.0)]
    upper: list[float] = [10.0 * max(float(np.median(np.abs(impedance))), 1.0)]
    if model.startswith("L"):
        estimates = impedance.imag[high] / omega[high]
        estimates = estimates[estimates > 0.0]
        l0 = float(np.median(estimates)) if estimates.size else 1e-7
        names.append("log_L")
        initial.append(math.log(max(l0, 1e-15)))
        lower.append(math.log(1e-15))
        upper.append(math.log(1e2))
    low_model = model[2:] if model.startswith("L+") else (None if model == "L" else model)
    if low_model == "C":
        q = -impedance.imag[low] * omega[low]
        q = q[q > 0.0]
        c0 = 1.0 / float(np.median(q)) if q.size else 1e-3
        names.append("log_C")
        initial.append(math.log(max(c0, 1e-15)))
        lower.append(math.log(1e-15))
        upper.append(math.log(1e6))
    elif low_model == "warburg-semi":
        sigma0 = float(np.median(np.maximum(impedance.real[low] - np.min(impedance.real[low]), 0.0) * np.sqrt(2.0 * omega[low])))
        sigma0 = max(sigma0, float(np.median(np.abs(impedance))) * math.sqrt(float(np.min(omega))) * 0.1, 1e-12)
        names.append("log_sigma")
        initial.append(math.log(sigma0))
        lower.append(math.log(1e-15))
        upper.append(math.log(1e12))
    elif low_model in {"warburg-short", "warburg-open"}:
        rw0 = max(float(np.ptp(impedance.real[low])), 0.1 * float(np.median(np.abs(impedance))), 1e-9)
        tau0 = 1.0 / (2.0 * math.pi * math.sqrt(float(np.min(frequency_hz)) * float(np.max(frequency_hz[low]))))
        names.extend(("log_Rw", "log_tau"))
        initial.extend((math.log(rw0), math.log(tau0)))
        lower.extend((math.log(1e-12), math.log(1.0 / (2.0 * math.pi * float(np.max(frequency_hz))) * 1e-2)))
        upper.extend((math.log(1e12), math.log(1.0 / (2.0 * math.pi * float(np.min(frequency_hz))) * 1e2)))

    def unpack(vector: Array) -> tuple[float, dict[str, float]]:
        r_edge = float(vector[0])
        output: dict[str, float] = {}
        for name, value in zip(names[1:], vector[1:]):
            if name == "log_L":
                output["L_henry"] = float(math.exp(value))
            elif name == "log_C":
                output["C_farad"] = float(math.exp(value))
            elif name == "log_sigma":
                output["sigma"] = float(math.exp(value))
            elif name == "log_Rw":
                output["R_w_ohm"] = float(math.exp(value))
            elif name == "log_tau":
                output["tau_d_s"] = float(math.exp(value))
        return r_edge, output

    denominator = np.maximum(np.abs(impedance[mask]), max(float(np.median(np.abs(impedance[mask]))) * 1e-12, np.finfo(float).tiny))

    def residual(vector: Array) -> Array:
        r_edge, parameters = unpack(vector)
        predicted = r_edge + _nuisance_component(frequency_hz[mask], model, parameters)
        return _complex_stack((predicted - impedance[mask]) / denominator)

    fit = least_squares(
        residual,
        np.asarray(initial, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=3000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    r_edge, parameters = unpack(fit.x)
    component = _nuisance_component(frequency_hz, model, parameters)
    residual_vector = residual(fit.x)
    rss = float(np.sum(np.square(residual_vector)))
    n = max(residual_vector.size, 1)
    k = len(fit.x)
    aic = float(n * math.log(max(rss / n, np.finfo(float).tiny)) + 2 * k)
    bic = float(n * math.log(max(rss / n, np.finfo(float).tiny)) + k * math.log(n))
    bounds_distance = np.minimum(
        (fit.x - np.asarray(lower)) / np.maximum(np.asarray(upper) - np.asarray(lower), np.finfo(float).tiny),
        (np.asarray(upper) - fit.x) / np.maximum(np.asarray(upper) - np.asarray(lower), np.finfo(float).tiny),
    )
    return {
        "status": "completed" if fit.success else "fit-warning",
        "model": model,
        "parameters": {"R_edge_fit_not_subtracted_ohm": r_edge, **parameters},
        "component": component,
        "fit_frequency_min_hz": float(np.min(frequency_hz[mask])),
        "fit_frequency_max_hz": float(np.max(frequency_hz[mask])),
        "fit_points": int(np.sum(mask)),
        "fit_relative_rss": rss,
        "aic": aic,
        "bic": bic,
        "parameter_at_bound": bool(np.any(bounds_distance < 0.01)),
        "optimizer_message": str(fit.message),
    }


def model_reduce_analysis(
    frequency_hz: Array,
    impedance: Array,
    baseline_reconstructed: Array,
    baseline_peaks: list[dict[str, Any]],
    drt_solver: Callable[[Array], dict[str, Any]],
    *,
    min_quality_improvement_log10: float = 0.02,
    peak_position_limit_decades: float = 0.35,
    peak_area_fraction_limit: float = 0.50,
) -> dict[str, Any]:
    """Quality-indicator-driven optional nuisance model fitting and subtraction."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    impedance = np.asarray(impedance, dtype=complex)
    eligibility = _nuisance_eligibility(frequency_hz, impedance)
    baseline_quality = schlueter_quality_indicator(impedance, baseline_reconstructed)
    candidates: list[dict[str, Any]] = [{
        "status": "baseline",
        "model": "none",
        "quality": baseline_quality,
        "score": baseline_quality["q_log10"],
        "accepted": True,
        "preprocessed_impedance": impedance,
        "reconstructed_impedance": baseline_reconstructed,
        "peaks": baseline_peaks,
        "removed_component": np.zeros(impedance.shape, dtype=complex),
    }]
    low_models: list[str] = []
    if eligibility["series_C"]:
        low_models.append("C")
    if eligibility["diffusion"]:
        low_models.extend(("warburg-semi", "warburg-short", "warburg-open"))
    model_names: list[str] = []
    if eligibility["series_L"]:
        model_names.append("L")
    model_names.extend(low_models)
    if eligibility["series_L"]:
        model_names.extend(f"L+{name}" for name in low_models)
    for model in model_names:
        try:
            fit = _fit_nuisance_model(frequency_hz, impedance, model)
            component = np.asarray(fit.pop("component"), dtype=complex)
            reduced = impedance - component
            solved = drt_solver(reduced)
            reconstructed = np.asarray(solved["reconstructed"], dtype=complex)
            reduced_quality = schlueter_quality_indicator(reduced, reconstructed)
            # Compare both candidates on the original data with identical weights.
            quality = schlueter_quality_indicator(impedance, reconstructed + component)
            stability = assess_peak_stability(
                baseline_peaks,
                solved.get("peaks", []),
                peak_position_limit_decades,
                peak_area_fraction_limit,
            )
            centered = impedance - np.median(impedance.real)
            removed_fraction = float(np.linalg.norm(component) / max(np.linalg.norm(centered), np.finfo(float).tiny))
            penalty = 0.0
            if fit["parameter_at_bound"]:
                penalty += 0.30
            if removed_fraction > 0.85:
                penalty += 0.50 + removed_fraction - 0.85
            if not stability["accepted"]:
                penalty += 0.25
            score = quality["q_log10"] + penalty
            candidates.append({
                **fit,
                "quality": quality,
                "reduced_domain_quality_diagnostic": reduced_quality,
                "addback_reconstructed_impedance": reconstructed + component,
                "score": score,
                "removed_norm_fraction": removed_fraction,
                "peak_stability": stability,
                "preprocessed_impedance": reduced,
                "reconstructed_impedance": reconstructed,
                "removed_component": component,
                "peaks": solved.get("peaks", []),
                "tau_s": solved.get("tau_s", []),
                "gamma": solved.get("gamma", []),
            })
        except Exception as exc:
            candidates.append({
                "status": "failed",
                "model": model,
                "error": f"{type(exc).__name__}: {exc}",
            })
    completed = [row for row in candidates[1:] if row.get("status") == "completed"]
    for row in candidates[1:]:
        improvement = baseline_quality["q_log10"] - row.get("quality", {}).get("q_log10", math.inf)
        row["quality_improvement_log10"] = improvement
        row["accepted"] = bool(row.get("status") == "completed"
            and improvement >= min_quality_improvement_log10
            and row.get("peak_stability", {}).get("accepted")
            and not row.get("parameter_at_bound", True)
            and row.get("removed_norm_fraction", math.inf) <= 0.85)
    eligible = [row for row in completed if row["accepted"]]
    best = min(eligible, key=lambda row: row["quality"]["q_log10"]) if eligible else None
    accepted = False
    if best is not None:
        improvement = baseline_quality["q_log10"] - best["quality"]["q_log10"]
        accepted = (
            improvement >= min_quality_improvement_log10
            and best["peak_stability"]["accepted"]
            and not best["parameter_at_bound"]
            and best["removed_norm_fraction"] <= 0.85
        )
        best["quality_improvement_log10"] = improvement
        best["accepted"] = accepted
    return {
        "status": "accepted" if accepted else "no-candidate-accepted",
        "enabled": True,
        "eligibility": eligibility,
        "baseline_quality": baseline_quality,
        "minimum_quality_improvement_log10": min_quality_improvement_log10,
        "selected_model": best.get("model") if accepted and best else "none",
        "selected": best if accepted else candidates[0],
        "best_attempt": min(completed, key=lambda row: row["quality"]["q_log10"]) if completed else None,
        "candidates": candidates,
        "claim_boundary": (
            "Only fitted nuisance terms are subtracted. The raw DRT remains authoritative. "
            "Acceptance requires reconstruction improvement and peak stability; a selected nuisance "
            "model is not proof of a physical mechanism."
        ),
    }


def _normalized_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    from metadata_contract import canonical_metadata
    return {key.lower(): value for key, value in canonical_metadata(metadata).items()}


def _first(metadata: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = metadata.get(name.lower())
        if value:
            return value
    return None


def _state_signature(metadata: dict[str, str], exclude: set[str]) -> tuple[tuple[str, str], ...] | None:
    keys = (
        "cell_id", "sample_id", "soc_percent", "state", "voltage_v", "potential_v", "target_voltage_v",
        "temperature_c", "cycle", "direction", "protocol_id", "ac_amplitude_mv", "amplitude_mv",
        "rest_time_s", "pressure_mpa", "data_origin", "impedance_basis",
    )
    rows = [(key, metadata[key]) for key in keys if key not in exclude and metadata.get(key)]
    state_present = any(
        key in dict(rows)
        for key in ("soc_percent", "state", "voltage_v", "potential_v", "target_voltage_v")
    )
    identity_present = any(key in dict(rows) for key in ("cell_id", "sample_id"))
    return tuple(rows) if state_present and identity_present else None


def _group_id(result: dict[str, Any], purpose: str) -> str | None:
    metadata = _normalized_metadata(result.get("metadata", {}))
    metadata["impedance_basis"] = str(result.get("impedance_basis", "unspecified"))
    amplitude = _first(metadata, ("ac_amplitude_mv", "amplitude_mv", "perturbation_mv"))
    rest_time = _first(metadata, ("rest_time_s", "rest_before_eis_s", "ocv_rest_s", "hold_time_s"))
    if amplitude is not None:
        metadata["ac_amplitude_mv"] = amplitude
        metadata.pop("amplitude_mv", None)
        metadata.pop("perturbation_mv", None)
    if rest_time is not None:
        metadata["rest_time_s"] = rest_time
    explicit = _first(metadata, (f"{purpose}_group", "state_group"))
    if purpose == "repeat":
        signature = _state_signature(metadata, {"replicate"})
    elif purpose == "linearity":
        signature = _state_signature(metadata, {"ac_amplitude_mv", "amplitude_mv", "replicate"})
    else:
        signature = _state_signature(metadata, {"rest_time_s", "replicate"})
    if signature is None:
        return None
    # Explicit names never override cell/state/protocol incompatibility.
    return (f"explicit:{explicit}:" if explicit else "") + repr(signature)


def _align_pair(first: dict[str, Any], second: dict[str, Any]) -> tuple[Array, Array, Array]:
    f1 = np.asarray(first["curves"]["frequency_hz"], dtype=float)
    z1 = np.asarray(first["curves"]["zreal_measured"], dtype=float) - 1j * np.asarray(first["curves"]["neg_zimag_measured"], dtype=float)
    f2 = np.asarray(second["curves"]["frequency_hz"], dtype=float)
    z2 = np.asarray(second["curves"]["zreal_measured"], dtype=float) - 1j * np.asarray(second["curves"]["neg_zimag_measured"], dtype=float)
    pairs: list[tuple[int, int]] = []
    if not f1.size or not f2.size or np.any(f1 <= 0) or np.any(f2 <= 0):
        return np.asarray([]), np.asarray([], dtype=complex), np.asarray([], dtype=complex)
    used: set[int] = set()
    for i, value in enumerate(f1):
        distances = np.abs(np.log10(f2 / value))
        j = int(np.argmin(distances))
        if j not in used and distances[j] <= 1e-5:
            pairs.append((i, j))
            used.add(j)
    if len(pairs) < 5 or len(pairs) / max(len(f1), len(f2)) < 0.8:
        return np.asarray([]), np.asarray([], dtype=complex), np.asarray([], dtype=complex)
    return (
        np.asarray([f1[i] for i, _ in pairs], dtype=float),
        np.asarray([z1[i] for i, _ in pairs], dtype=complex),
        np.asarray([z2[j] for _, j in pairs], dtype=complex),
    )


def _pair_assessment(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    from control_validation import uid
    base = {"first": first["label"], "second": second["label"],
            "first_uid": uid(first), "second_uid": uid(second),
            "log_frequency_tolerance_decades": 1e-5, "minimum_overlap_fraction": 0.8}
    if first.get("impedance_basis") != second.get("impedance_basis"):
        return {**base, "status": "not-assessable", "reason": "incompatible-impedance-basis"}
    frequency, z1, z2 = _align_pair(first, second)
    if not frequency.size:
        return {**base, "status": "not-assessable", "reason": "insufficient-common-frequency-grid"}
    magnitude = (np.abs(z1) + np.abs(z2)) / 2
    denominator = np.maximum(magnitude, max(float(np.median(magnitude)) * 1e-12, np.finfo(float).tiny))
    point_relative = 100.0 * np.abs(z2 - z1) / denominator
    if not np.all(np.isfinite(point_relative)):
        return {**base, "status": "not-assessable", "reason": "nonfinite-impedance-comparison"}
    from control_validation import repeat_peak_assessment
    return {**base, "status": "completed", "common_points": int(frequency.size),
            "overlap_fraction": float(frequency.size / max(len(first["curves"]["frequency_hz"]), len(second["curves"]["frequency_hz"]))),
            "frequency_min_hz": float(np.min(frequency)), "frequency_max_hz": float(np.max(frequency)),
            "complex_rms_percent": float(np.sqrt(np.mean(np.square(point_relative)))),
            "complex_median_percent": float(np.median(point_relative)), "complex_max_percent": float(np.max(point_relative)),
            "peak_stability": assess_peak_stability(first.get("peaks", []), second.get("peaks", [])),
            "repeat_peak_stability": repeat_peak_assessment(first, second, frequency)}


def _group_results(results: list[dict[str, Any]], purpose: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "completed":
            continue
        group = _group_id(result, purpose)
        if group:
            grouped.setdefault(group, []).append(result)
    return {key: rows for key, rows in grouped.items() if len(rows) >= 2}


def _amplitude(result: dict[str, Any]) -> float | None:
    metadata = _normalized_metadata(result.get("metadata", {}))
    return finite_float(_first(metadata, ("ac_amplitude_mv", "amplitude_mv", "perturbation_mv")))


def _rest_time(result: dict[str, Any]) -> float | None:
    metadata = _normalized_metadata(result.get("metadata", {}))
    return finite_float(_first(metadata, ("rest_time_s", "rest_before_eis_s", "ocv_rest_s", "hold_time_s")))


def assess_batch_validity(
    results: list[dict[str, Any]],
    repeat_warn_percent: float = 2.0,
    linearity_warn_percent: float = 2.0,
    stationarity_warn_percent: float = 2.0,
) -> dict[str, Any]:
    from control_validation import assess_controls
    return assess_controls(results, repeat_warn_percent, linearity_warn_percent, stationarity_warn_percent)


def plot_advanced_audit(result: dict[str, Any], figure_dir: Path) -> list[str]:
    """Render independent continuous-density and discrete-weight audit panels."""
    from diagnostic_plotting import save_advanced_audit
    return save_advanced_audit(result, figure_dir)

"""Scale pyimpspec DRT solves without changing their impedance objective.

A single positive scalar conditions the complete spectrum. It is not a
frequency-dependent weight, noise model, or change in the regularization
parameter. Results are returned in the input impedance units.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def conditioning_info(data: Any, mode: str = "median") -> dict[str, Any]:
    """Validate included observations and describe the scalar used by a solve."""
    if mode not in {"median", "off"}:
        raise ValueError("impedance_scaling must be 'median' or 'off'")
    frequency = np.asarray(data.get_frequencies(), dtype=float)
    impedance = np.asarray(data.get_impedances(), dtype=complex)
    if frequency.ndim != 1 or impedance.shape != frequency.shape or not frequency.size:
        raise ValueError("Conditioning requires nonempty aligned one-dimensional observations")
    if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("Conditioning requires finite positive frequencies")
    if not np.all(np.isfinite(impedance)):
        raise ValueError("Conditioning requires finite complex impedances")
    median = float(np.median(np.abs(impedance)))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Conditioning requires a finite positive median impedance magnitude")
    return {
        "mode": mode,
        "impedance_scale": median if mode == "median" else 1.0,
        "scale_definition": "median(abs(Z)) of included observations" if mode == "median" else "1; original numerical scaling",
        "included_points": int(frequency.size),
        "uniform_scalar": True,
        "pointwise_weighting_applied": False,
        "output_units": "same impedance units as input",
        "claim_boundary": "Numerical conditioning only; no noise weighting, interpolation, or change in lambda convention.",
    }


def _conditioned_data(data: Any, mode: str) -> tuple[Any, float]:
    info = conditioning_info(data, mode)
    scale = info["impedance_scale"]
    if mode == "off":
        return data, scale
    from pyimpspec import DataSet

    # Only included observations enter a pyimpspec solve. Constructing a new
    # object also keeps the caller's arrays, mask, metadata and cache untouched.
    conditioned = DataSet(
        frequencies=np.asarray(data.get_frequencies(), dtype=float).copy(),
        impedances=np.asarray(data.get_impedances(), dtype=complex).copy() / scale,
        path=data.get_path(),
        label=data.get_label(),
    )
    return conditioned, scale


def _restore_result(result: Any, scale: float, fields: tuple[str, ...]) -> Any:
    values = {name: np.asarray(getattr(result, name)) * scale for name in fields}
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise RuntimeError("Nonfinite DRT solution after restoring impedance units")
    if not np.all(np.isfinite(result.residuals)) or not np.isfinite(result.pseudo_chisqr):
        raise RuntimeError("Nonfinite DRT residuals or pseudo chi-squared")
    # Relative residuals, pseudo chi-squared, lambda and both abscissae retain
    # their dimensionless/time/frequency values. Empty CI arrays remain empty.
    return replace(result, **values) if scale != 1.0 else result


def calculate_tr_rbf_conditioned(data: Any, *, impedance_scaling: str = "median", **kwargs: Any) -> Any:
    """Run fixed-lambda, mGCV or conditional-CI TR-RBF in physical output units."""
    from pyimpspec.analysis.drt import calculate_drt_tr_rbf

    conditioned, scale = _conditioned_data(data, impedance_scaling)
    result = calculate_drt_tr_rbf(conditioned, **kwargs)
    return _restore_result(result, scale, ("impedances", "gammas", "mean_gammas", "lower_bounds", "upper_bounds"))


def calculate_tr_nnls_conditioned(data: Any, *, impedance_scaling: str = "median", **kwargs: Any) -> Any:
    """Apply the same input/output conditioning contract to secondary TR-NNLS.

    A real-only NNLS fit remains real-only: its package pseudo chi-squared is
    not a full complex reconstruction check, including when series L is present.
    """
    from pyimpspec.analysis.drt import calculate_drt_tr_nnls

    conditioned, scale = _conditioned_data(data, impedance_scaling)
    result = calculate_drt_tr_nnls(conditioned, **kwargs)
    return _restore_result(result, scale, ("impedances", "gammas"))

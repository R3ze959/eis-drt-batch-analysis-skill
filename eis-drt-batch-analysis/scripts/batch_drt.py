#!/usr/bin/env python3
"""Audit and batch-analyze EIS spectra with an adaptive TR-RBF DRT workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing as mp
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

import numpy as np


TEXT_SUFFIXES = {".csv", ".txt", ".tsv"}
PYIMPSPEC_SUFFIXES = {
    ".p00", ".dfr", ".dta", ".i2b", ".idf", ".ids", ".mpt", ".z",
    ".pssession", ".ods", ".xlsx",
}
SUPPORTED_SUFFIXES = {".irf", *TEXT_SUFFIXES, *PYIMPSPEC_SUFFIXES}
DEFAULT_PARSER = Path(__file__).with_name("eis_parser.py")

# Dataset-specific profiles are external JSON inputs, never bundled private cases.
REFERENCE_PROFILES = {}


@dataclass
class WorkerOptions:
    lambda_policy: str
    lambda_min: float
    lambda_max: float
    lambda_points: int
    fixed_lambda: float
    derivative_order: int
    rbf_type: str
    rbf_shape: str
    shape_coeff: float
    edge_decades: float
    peak_match_decades: float
    peak_threshold: float
    credible_intervals: bool
    credible_samples: int
    seed: int
    num_procs: int
    solver_timeout: int
    kk_max_points: int
    strict: bool
    kk_warn_percent: float
    kk_block_percent: float
    model_reduce: bool
    loewner: bool
    signed_gdrt: str
    ddt: str
    model_reduce_min_improvement: float
    preprocess_peak_position_decades: float
    preprocess_peak_area_fraction: float
    inductance_policy: str
    impedance_scaling: str = "median"
    peak_evidence: bool = True
    predictive_fit: str = "auto"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return json_ready(value.item())
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_slug(text: str, fallback: str = "spectrum") -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:120] or fallback


def load_parser_module(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Authoritative EIS parser not found: {path}. "
            "Restore scripts/eis_parser.py from the release, or pass --parser-script explicitly."
        )
    module_name = "_codex_eis_impedance_parser"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load parser module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_inputs(inputs: Iterable[str], output_dir: Path) -> list[Path]:
    discovered: list[Path] = []
    output_dir = output_dir.resolve()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                discovered.append(path)
            continue
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if not candidate.is_file() or candidate.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                try:
                    candidate.resolve().relative_to(output_dir)
                    continue
                except ValueError:
                    discovered.append(candidate.resolve())
            continue
        raise FileNotFoundError(f"Input does not exist: {path}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise ValueError("No supported EIS inputs were found")
    return unique


def load_manifest(path: Path | None) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    exact: dict[tuple[str, str], dict[str, str]] = {}
    by_path: dict[str, dict[str, str]] = {}
    if path is None:
        return exact, by_path
    manifest = path.expanduser().resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("Manifest must contain a 'path' column")
        names = [str(name).strip().lower() for name in reader.fieldnames]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Manifest column names must be nonempty and unique")
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Manifest row {row_number} has a different number of cells than its header")
            raw_path = (row.get("path") or "").strip()
            if not raw_path:
                raise ValueError(f"Manifest row {row_number} has an empty path")
            source = Path(raw_path).expanduser()
            if not source.is_absolute():
                source = (manifest.parent / source).resolve()
            else:
                source = source.resolve()
            cleaned = {str(k): str(v).strip() for k, v in row.items() if k is not None and v is not None}
            spectrum_id = cleaned.get("spectrum_id", "")
            if spectrum_id:
                if (str(source), spectrum_id) in exact:
                    raise ValueError(f"Duplicate manifest source/spectrum at row {row_number}")
                exact[(str(source), spectrum_id)] = cleaned
            else:
                if str(source) in by_path:
                    raise ValueError(f"Duplicate manifest source at row {row_number}")
                by_path[str(source)] = cleaned
    return exact, by_path


def manifest_metadata(
    source: Path,
    spectrum_id: str,
    exact: dict[tuple[str, str], dict[str, str]],
    by_path: dict[str, dict[str, str]],
) -> dict[str, str]:
    return {**by_path.get(str(source.resolve()), {}), **exact.get((str(source.resolve()), spectrum_id), {})}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def lag1_correlation(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size < 4 or float(np.std(values)) == 0.0:
        return None
    result = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    return result if math.isfinite(result) else None


def package_versions() -> dict[str, str]:
    from runtime_contract import actual_versions
    return actual_versions()


def prepare_payload(
    spec: Any,
    source_sha256: str,
    qc: dict[str, Any],
    metadata: dict[str, str],
    area_normalize: bool,
) -> dict[str, Any]:
    from metadata_contract import canonical_metadata
    metadata = canonical_metadata(metadata)
    reserved = set(metadata) & set(spec.metadata) - set(PARSER_FIELDS)
    if reserved:
        raise ValueError("Manifest cannot override parser-owned provenance: " + ", ".join(sorted(reserved)))
    zreal = np.asarray(spec.zreal, dtype=float).copy()
    neg_zimag = np.asarray(spec.neg_zimag, dtype=float).copy()
    basis = str(spec.impedance_basis)
    warnings: list[str] = []
    if area_normalize:
        area = finite_float(metadata.get("area_cm2"))
        if basis == "ohm_cm2":
            warnings.append("Input was already area-normalized; area_cm2 was not applied again")
        elif area is None or area <= 0.0:
            raise ValueError(
                f"--area-normalize requires a positive area_cm2 in the manifest for {spec.spectrum_id}"
            )
        else:
            zreal *= area
            neg_zimag *= area
            basis = "ohm_cm2"
            warnings.append(f"Impedance multiplied by declared area_cm2={area:g}")
    frequency = np.asarray(spec.freq_hz, dtype=float)
    acquisition = np.asarray(spec.acquisition_index, dtype=float)
    arrays = (frequency, zreal, neg_zimag, acquisition)
    if any(a.ndim != 1 for a in arrays) or not all(len(a) == len(frequency) > 0 for a in arrays):
        raise ValueError("Empty or mismatched normalized EIS arrays")
    if not all(np.all(np.isfinite(a)) for a in arrays) or np.any(frequency <= 0):
        raise ValueError("Normalized EIS requires finite values and positive frequency after unit/area conversion")
    if not np.all(np.isfinite(np.hypot(zreal, neg_zimag))):
        raise ValueError("Impedance modulus overflow after conversion; inspect units and area")
    if spec.metadata.get("acquisition_order_preserved") is False:
        warnings.append(
            "The pyimpspec source parser sorts points by descending frequency; "
            "original acquisition order is unavailable for this format"
        )
    from result_contract import identities
    return {
        **identities(spec.source, source_sha256, spec.spectrum_id),
        "spectrum_id": str(spec.spectrum_id),
        "label": metadata.get("label") or str(spec.label),
        "source": str(Path(spec.source).resolve()),
        "source_sha256": source_sha256,
        "acquisition_index": np.asarray(spec.acquisition_index, dtype=float),
        "frequency_hz": np.asarray(spec.freq_hz, dtype=float),
        "zreal": zreal,
        "neg_zimag": neg_zimag,
        "impedance_basis": basis,
        "metadata": {**dict(spec.metadata), **metadata},
        "qc": qc,
        "warnings": warnings,
    }


def read_pyimpspec_source(
    source: Path,
    label: str,
    args: argparse.Namespace,
    parser_module: Any,
) -> tuple[list[Any], dict[str, Any]]:
    """Import vendor/spreadsheet formats through pyimpspec, then normalize to Spectrum."""
    from pyimpspec import parse_data

    frequency_unit = args.frequency_unit if args.frequency_unit != "auto" else "hz"
    frequency_scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e-3}[frequency_unit]
    unit = args.impedance_unit
    if unit == "auto":
        unit = "ohm"
    scales = {
        "ohm": (1.0, "ohm"),
        "mohm": (1e-3, "ohm"),
        "kohm": (1e3, "ohm"),
        "ohm-cm2": (1.0, "ohm_cm2"),
    }
    scale, impedance_basis = scales[unit]
    datasets = list(parse_data(str(source)))
    spectra: list[Any] = []
    for index, dataset in enumerate(datasets, start=1):
        frequencies = (
            np.asarray(dataset.get_frequencies(masked=False), dtype=float) * frequency_scale
        )
        impedances = np.asarray(dataset.get_impedances(masked=False), dtype=complex) * scale
        all_frequencies = (
            np.asarray(dataset.get_frequencies(masked=None), dtype=float) * frequency_scale
        )
        if frequencies.size == 0:
            continue
        dataset_label = str(dataset.get_label() or "").strip()
        item_label = dataset_label or (label if len(datasets) == 1 else f"{label} #{index}")
        spectrum_id = f"{source.stem}:pyimpspec:run{index}"
        excluded = {}
        masked_count = int(all_frequencies.size - frequencies.size)
        if masked_count:
            excluded["masked_by_source_file"] = masked_count
        spectra.append(
            parser_module.Spectrum(
                spectrum_id=spectrum_id,
                label=item_label,
                source=source,
                acquisition_index=np.arange(1, frequencies.size + 1, dtype=float),
                freq_hz=frequencies,
                zreal=impedances.real,
                neg_zimag=-impedances.imag,
                raw_rows=int(all_frequencies.size),
                excluded=excluded,
                sign_basis="pyimpspec complex-impedance convention; reported ordinate is -Im(Z)",
                impedance_basis=impedance_basis,
                metadata={
                    "parser": "pyimpspec.parse_data",
                    "source_suffix": source.suffix,
                    "dataset_index": index,
                    "dataset_label": dataset_label,
                    "source_frequency_unit": frequency_unit,
                    "frequency_scale_to_Hz": frequency_scale,
                    "source_impedance_override": unit,
                    "acquisition_order_preserved": False,
                },
            )
        )
    if not spectra:
        raise ValueError(f"pyimpspec found no included EIS points in {source}")
    audit = {
        "source": str(source.resolve()),
        "sha256": sha256_file(source),
        "parser": "pyimpspec.parse_data",
        "source_suffix": source.suffix,
        "parsed_spectra": len(spectra),
        "supported_native_formats": sorted(PYIMPSPEC_SUFFIXES),
    }
    return spectra, audit


def high_frequency_inductance(freq: np.ndarray, neg_zimag: np.ndarray) -> tuple[bool, dict[str, Any]]:
    threshold = float(np.max(freq)) / 10.0
    mask = freq >= threshold
    count = int(np.sum(mask))
    scale = max(float(np.nanpercentile(np.abs(neg_zimag), 75.0)), np.finfo(float).tiny)
    sign_threshold = 0.002 * scale
    negative = int(np.sum(neg_zimag[mask] < -sign_threshold))
    detected = count >= 3 and negative >= 3
    estimate = estimate_series_inductance(freq, neg_zimag) if detected else {
        "status": "not-triggered",
        "estimated_series_inductance_h": 0.0,
        "fit_r_squared": None,
        "series_l_explainable": False,
    }
    return detected, {
        "highest_frequency_decade_points": count,
        "negative_neg_zimag_points": negative,
        "significance_threshold_impedance": sign_threshold,
        "criterion": (
            "at least three materially negative -Zimag points in the highest measured frequency decade"
        ),
        **estimate,
    }


def estimate_series_inductance(freq: np.ndarray, neg_zimag: np.ndarray) -> dict[str, Any]:
    """Estimate a plausible wiring/fixture series L from the highest measured decade.

    The intercept is retained during the regression but is not subtracted from the spectrum.
    A linear -Zimag versus angular-frequency trend is evidence for a series-L candidate,
    not proof that the complete high-frequency response is an ideal inductor.
    """
    freq = np.asarray(freq, dtype=float)
    neg_zimag = np.asarray(neg_zimag, dtype=float)
    mask = freq >= float(np.max(freq)) / 10.0
    if int(np.sum(mask)) < 5:
        return {
            "status": "insufficient-points",
            "estimated_series_inductance_h": 0.0,
            "fit_intercept_ohm": None,
            "fit_r_squared": None,
            "series_l_explainable": False,
        }
    omega = 2.0 * math.pi * freq[mask]
    values = neg_zimag[mask]
    slope, intercept = np.polyfit(omega, values, 1)
    predicted = slope * omega + intercept
    centered = values - float(np.mean(values))
    denominator = float(np.sum(np.square(centered)))
    r_squared = (
        1.0 - float(np.sum(np.square(values - predicted))) / denominator
        if denominator > np.finfo(float).tiny else None
    )
    inductance_h = max(0.0, -float(slope))
    explainable = bool(
        inductance_h > 0.0 and r_squared is not None and r_squared >= 0.90
    )
    return {
        "status": "completed",
        "estimated_series_inductance_h": inductance_h,
        "fit_intercept_ohm": float(intercept),
        "fit_r_squared": r_squared,
        "fit_frequency_min_hz": float(np.min(freq[mask])),
        "fit_frequency_max_hz": float(np.max(freq[mask])),
        "fit_points": int(np.sum(mask)),
        "series_l_explainable": explainable,
        "criterion": "positive L and high-decade linear-fit R-squared >= 0.90",
    }


def nonseries_inductive_region(
    freq: np.ndarray,
    neg_zimag: np.ndarray,
    series_inductance_h: float | None = None,
    impedance_magnitude: np.ndarray | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Detect finite/low-frequency inductive response after plausible series-L compensation."""
    order = np.argsort(freq)[::-1]
    sorted_freq = np.asarray(freq, dtype=float)[order]
    sorted_imag = np.asarray(neg_zimag, dtype=float)[order]
    raw_sorted_imag = sorted_imag.copy()
    estimate = estimate_series_inductance(sorted_freq, sorted_imag)
    if series_inductance_h is None:
        series_inductance_h = (
            float(estimate["estimated_series_inductance_h"])
            if estimate.get("series_l_explainable") else 0.0
        )
    series_inductance_h = max(0.0, float(series_inductance_h))
    if series_inductance_h > 0.0:
        sorted_imag = sorted_imag + 2.0 * math.pi * sorted_freq * series_inductance_h
    outside_high = sorted_freq < float(np.max(sorted_freq)) / 10.0
    scale = max(float(np.nanpercentile(np.abs(raw_sorted_imag), 75.0)), np.finfo(float).tiny)
    sign_threshold = 0.002 * scale
    thresholds = np.full(sorted_freq.shape, sign_threshold)
    noise_evidence = {"status": "not-assessed", "reason": "complex-modulus-not-supplied"}
    if impedance_magnitude is not None:
        magnitude = np.asarray(impedance_magnitude, dtype=float)
        if magnitude.shape != sorted_freq.shape or not np.isfinite(magnitude).all() or np.any(magnitude < 0):
            raise ValueError("Aligned finite nonnegative impedance magnitudes are required for noise-aware screening")
        magnitude = magnitude[order]
        floor = max(float(np.median(magnitude)) * 1e-12, np.finfo(float).tiny)
        magnitude = np.maximum(magnitude, floor)
        if len(freq) >= 7 and np.all(sorted_freq > 0) and len(np.unique(sorted_freq)) == len(freq):
            # Local interpolation residuals estimate roughness/noise only; these
            # derived values never replace observations in an inversion.
            x = np.log(sorted_freq)
            normalized = sorted_imag / magnitude
            left = (x[2:] - x[1:-1]) / (x[2:] - x[:-2])
            right = 1.0 - left
            residual = (normalized[1:-1] - left*normalized[:-2] - right*normalized[2:])
            residual /= np.sqrt(1.0 + left*left + right*right)
            noise_sigma = float(1.4826*np.median(np.abs(residual-np.median(residual))))
            thresholds = np.maximum(thresholds, 3.0*noise_sigma*magnitude)
            noise_evidence = {"status": "estimated", "relative_sigma": noise_sigma,
                "method": "MAD of normalized local-linear interpolation residuals with coefficient-norm scaling",
                "noise_multiplier": 3.0,
                "claim_boundary": "Engineering roughness floor, not calibrated significance. Curvature can inflate it; no detected run does not prove absence of a weak inductive process."}
    raw_flags = (raw_sorted_imag < -sign_threshold) & outside_high
    pre_noise_flags = (sorted_imag < -sign_threshold) & outside_high
    flags = (sorted_imag < -thresholds) & outside_high
    longest = 0
    current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    detected = longest >= 3
    return detected, {
        "criterion": (
            "after plausible series-L compensation, at least three consecutive materially negative "
            "-Zimag points below the highest measured frequency decade and above the estimated noise floor when available"
        ),
        "significance_threshold_impedance": sign_threshold,
        "pointwise_screen_threshold_impedance": thresholds.tolist(),
        "screen_frequency_hz": sorted_freq.tolist(),
        "noise_evidence": noise_evidence,
        "negative_points_before_noise_floor": int(np.sum(pre_noise_flags)),
        "noise_limited_negative_points": int(np.sum(pre_noise_flags & ~flags)),
        "raw_negative_points_outside_highest_decade": int(np.sum(raw_flags)),
        "negative_points_outside_highest_decade": int(np.sum(flags)),
        "longest_consecutive_run": int(longest),
        "series_inductance_compensated_for_screen_h": series_inductance_h,
        "series_inductance_estimate": estimate,
        "detected": detected,
        "required_route": "signed/generalized DRT or a justified physical inductive model",
    }


def reconstruction_quality(measured: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    measured = np.asarray(measured, dtype=complex)
    reconstructed = np.asarray(reconstructed, dtype=complex)
    scale = np.maximum(np.abs(measured), max(float(np.median(np.abs(measured))) * 1e-12, np.finfo(float).tiny))
    weighted = (reconstructed - measured) / scale
    rms_value = float(np.sqrt(np.mean(np.square(weighted.real) + np.square(weighted.imag)) / 2.0))
    return {
        "modulus_weighted_rms": rms_value,
        "q_log10": float(math.log10(max(rms_value, np.finfo(float).tiny))),
    }


def choose_primary_inductance_mode(
    data: Any,
    measured_z: np.ndarray,
    detected: bool,
    evidence: dict[str, Any],
    options: WorkerOptions,
) -> tuple[bool, dict[str, Any]]:
    """Select no-L versus series-L by a small auditable candidate competition."""
    policy = options.inductance_policy
    if policy == "off":
        return False, {
            "status": "forced-off",
            "policy": policy,
            "selected_model": "RC-DRT",
            "reason": "legacy compatibility override; not permitted as the sole scientific result",
            "candidates": [],
        }
    if policy == "always":
        return True, {
            "status": "forced-on",
            "policy": policy,
            "selected_model": "series-L+RC-DRT",
            "reason": "explicit user override",
            "candidates": [],
        }
    if not detected:
        return False, {
            "status": "not-triggered",
            "policy": policy,
            "selected_model": "RC-DRT",
            "reason": "no material high-frequency inductive trigger",
            "candidates": [],
        }

    probe_lambda = float(np.clip(1e-4, options.lambda_min, options.lambda_max))
    if options.lambda_policy == "fixed":
        probe_lambda = float(options.fixed_lambda)
    candidates: list[dict[str, Any]] = []
    probes = np.unique(np.clip([probe_lambda / 10, probe_lambda, probe_lambda * 10],
                               options.lambda_min, options.lambda_max))
    for name, use_l in (("RC-DRT", False), ("series-L+RC-DRT", True)):
        for value in probes:
            try:
                probe = fixed_tr_rbf(data, float(value), use_l, options)
                quality = reconstruction_quality(measured_z, np.asarray(probe.get_impedances(), dtype=complex))
                candidates.append({"model": name, "status": "completed", "probe_lambda": float(value), **quality})
            except Exception as exc:
                candidates.append({"model": name, "status": "failed", "probe_lambda": float(value),
                                   "error": f"{type(exc).__name__}: {exc}"})
    completed = {}
    for name in ("RC-DRT", "series-L+RC-DRT"):
        valid = [r for r in candidates if r["status"] == "completed" and r["model"] == name]
        if valid:
            completed[name] = min(valid, key=lambda r: r["q_log10"])
    no_l = completed.get("RC-DRT")
    with_l = completed.get("series-L+RC-DRT")
    explainable = bool(evidence.get("series_l_explainable"))
    improvement = None
    if no_l is not None and with_l is not None:
        improvement = float(no_l["q_log10"] - with_l["q_log10"])
    use_l = bool(
        with_l is not None
        and (
            no_l is None
            or (improvement is not None and improvement >= 0.02)
            or (explainable and improvement is not None and improvement >= -0.005)
        )
    )
    reason = (
        ("series-L selected by high-decade linearity and non-worse reconstruction" if explainable
         else "series-L selected by improved weighted reconstruction; high-decade physical support unresolved")
        if use_l else
        "series-L candidate lacked sufficient physical or reconstruction support"
    )
    return use_l, {
        "status": "selected",
        "policy": policy,
        "selected_model": "series-L+RC-DRT" if use_l else "RC-DRT",
        "reason": reason,
        "series_l_explainable": explainable,
        "estimated_series_inductance_h": evidence.get("estimated_series_inductance_h"),
        "fit_r_squared": evidence.get("fit_r_squared"),
        "quality_improvement_log10_with_L": improvement,
        "candidates": candidates,
    }


def fixed_tr_rbf(data: Any, lambda_value: float, inductance: bool, options: WorkerOptions):
    from numerical_conditioning import calculate_tr_rbf_conditioned as calculate_drt_tr_rbf

    cache = getattr(data, "_codex_fixed_cache", {})
    key = (float(lambda_value), bool(inductance), options.rbf_type, options.rbf_shape,
           options.derivative_order, options.shape_coeff, getattr(options, "impedance_scaling", "median"))
    if key in cache:
        return cache[key]
    result = calculate_drt_tr_rbf(
        data,
        impedance_scaling=getattr(options, "impedance_scaling", "median"),
        mode="complex",
        lambda_value=float(lambda_value),
        cross_validation="",
        rbf_type=options.rbf_type,
        derivative_order=options.derivative_order,
        rbf_shape=options.rbf_shape,
        shape_coeff=options.shape_coeff,
        inductance=inductance,
        credible_intervals=False,
        timeout=options.solver_timeout,
        num_procs=options.num_procs,
    )
    if not np.all(np.isfinite(result.get_impedances())) or not np.all(np.isfinite(result.get_drt_data()[1])):
        raise RuntimeError("Nonfinite TR-RBF solution")
    cache[key] = result
    setattr(data, "_codex_fixed_cache", cache)
    return result


def curve_metrics(result: Any, measured_z: np.ndarray, derivative_order: int) -> tuple[float, float]:
    reconstructed = np.asarray(result.get_impedances(), dtype=complex)
    denominator = max(float(np.linalg.norm(measured_z)), np.finfo(float).tiny)
    residual_norm = float(np.linalg.norm(reconstructed - measured_z) / denominator)
    _, gamma = result.get_drt_data()
    gamma = np.asarray(gamma, dtype=float)
    order = max(1, int(derivative_order))
    rough = np.diff(gamma, n=order)
    roughness = float(np.linalg.norm(rough) / math.sqrt(max(rough.size, 1)))
    return max(residual_norm, np.finfo(float).tiny), max(roughness, np.finfo(float).tiny)


def lcurve_curvature(lambda_values: np.ndarray, residuals: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    t = np.log10(lambda_values)
    x = np.log10(residuals)
    y = np.log10(roughness)
    if len(t) < 3:
        return np.full(t.shape, np.nan)
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    denom = np.power(dx * dx + dy * dy, 1.5)
    curvature = np.divide(
        np.abs(dx * ddy - dy * ddx),
        denom,
        out=np.full_like(denom, np.nan),
        where=denom > np.finfo(float).eps,
    )
    if curvature.size >= 2:
        curvature[0] = np.nan
        curvature[-1] = np.nan
    return curvature


def lcurve_chord_distance(residuals: np.ndarray, roughness: np.ndarray) -> np.ndarray:
    """Return normalized log-log distance from the chord joining L-curve endpoints."""
    x = np.log10(np.asarray(residuals, dtype=float))
    y = np.log10(np.asarray(roughness, dtype=float))
    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    if x_span <= np.finfo(float).eps or y_span <= np.finfo(float).eps:
        return np.full(x.shape, np.nan)
    x = (x - float(np.min(x))) / x_span
    y = (y - float(np.min(y))) / y_span
    ax, ay = float(x[0]), float(y[0])
    vx, vy = float(x[-1] - x[0]), float(y[-1] - y[0])
    denominator = math.hypot(vx, vy)
    if denominator <= np.finfo(float).eps:
        return np.full(x.shape, np.nan)
    distances = np.abs(vx * (y - ay) - vy * (x - ax)) / denominator
    # A straight log-log tradeoff has no identifiable corner; do not select a
    # floating-point ripple as if it were a resolved optimum.
    if float(np.max(distances)) <= 1e-8:
        return np.full(x.shape, np.nan)
    return distances


def select_lambda(
    data: Any,
    measured_z: np.ndarray,
    inductance: bool,
    options: WorkerOptions,
) -> tuple[float, Any, dict[str, Any], dict[float, Any]]:
    from numerical_conditioning import calculate_tr_rbf_conditioned as calculate_drt_tr_rbf

    if options.lambda_policy == "fixed":
        grid = np.unique(np.clip([options.fixed_lambda / 10, options.fixed_lambda,
                                 options.fixed_lambda * 10], options.lambda_min, options.lambda_max))
    else:
        grid = np.logspace(math.log10(options.lambda_min), math.log10(options.lambda_max), options.lambda_points)
    cache: dict[float, Any] = {}
    rows: list[dict[str, Any]] = []
    for value in grid:
        result = fixed_tr_rbf(data, float(value), inductance, options)
        cache[float(value)] = result
        residual, rough = curve_metrics(result, measured_z, options.derivative_order)
        rows.append({
            "lambda": float(value),
            "relative_residual_norm": residual,
            "roughness_norm": rough,
            "pseudo_chisqr": float(result.pseudo_chisqr),
        })
    residuals = np.asarray([row["relative_residual_norm"] for row in rows], dtype=float)
    roughness = np.asarray([row["roughness_norm"] for row in rows], dtype=float)
    curvature = lcurve_curvature(grid, residuals, roughness)
    chord_distance = lcurve_chord_distance(residuals, roughness)
    for row, value, distance in zip(rows, curvature, chord_distance):
        row["lcurve_curvature"] = float(value) if math.isfinite(float(value)) else None
        row["lcurve_chord_distance"] = (
            float(distance) if math.isfinite(float(distance)) else None
        )

    boundary_guard = max(1, int(math.ceil(0.15 * (grid.size - 1))))
    eligible = np.zeros(grid.size, dtype=bool)
    eligible[boundary_guard : grid.size - boundary_guard] = True
    finite_indices = np.flatnonzero(np.isfinite(chord_distance) & eligible)
    for index, row in enumerate(rows):
        row["lcurve_eligible"] = bool(eligible[index])
    if finite_indices.size:
        lcurve_index = int(finite_indices[np.argmax(chord_distance[finite_indices])])
        lcurve_status = (
            "maximum normalized log-log endpoint-chord distance after excluding the outer "
            f"{boundary_guard} lambda-grid point(s) at each boundary"
        )
    else:
        lcurve_index = int(np.argmin(np.abs(np.log10(grid) - math.log10(options.fixed_lambda))))
        lcurve_status = "fallback to nearest fixed lambda because L-curve chord distance was undefined"
    lcurve_lambda = float(grid[lcurve_index])

    mgcv_candidates: list[dict[str, Any]] = []
    if options.lambda_policy in {"consensus", "package-mgcv"}:
        # Three genuinely distinct interior starts even for a narrow user range.
        starts = np.geomspace(options.lambda_min, options.lambda_max, 5)[1:-1]
        for start in starts:
            try:
                result = calculate_drt_tr_rbf(
                    data,
                    impedance_scaling=getattr(options, "impedance_scaling", "median"),
                    mode="complex",
                    lambda_value=float(start),
                    cross_validation="mgcv",
                    rbf_type=options.rbf_type,
                    derivative_order=options.derivative_order,
                    rbf_shape=options.rbf_shape,
                    shape_coeff=options.shape_coeff,
                    inductance=inductance,
                    credible_intervals=False,
                    timeout=options.solver_timeout,
                    num_procs=options.num_procs,
                )
                proposed = float(result.lambda_value)
                valid = bool(math.isfinite(proposed) and options.lambda_min < proposed < options.lambda_max)
                package_boundary = bool(math.isfinite(proposed) and proposed > 0 and
                    (abs(math.log10(proposed / 1e-7)) < 1e-5 or abs(math.log10(proposed)) < 1e-5))
                eligible_candidate = valid and not package_boundary and math.isfinite(float(result.pseudo_chisqr))
                mgcv_candidates.append({
                    "start_lambda": float(start),
                    "returned_lambda": proposed,
                    "eligible_for_consensus": eligible_candidate,
                    "rejection_reason": ("outside-or-on-user-search-boundary" if not valid else
                                         "on-package-search-boundary" if package_boundary else
                                         "nonfinite-fit-metric" if not eligible_candidate else None),
                    "pseudo_chisqr": float(result.pseudo_chisqr),
                    "error": None,
                })
            except Exception as exc:
                mgcv_candidates.append({
                    "start_lambda": float(start),
                    "returned_lambda": None,
                    "eligible_for_consensus": False,
                    "rejection_reason": "solver-exception",
                    "pseudo_chisqr": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    returned = np.asarray(
        [row["returned_lambda"] for row in mgcv_candidates if row.get("eligible_for_consensus")],
        dtype=float,
    )
    mgcv_span = (
        float(np.ptp(np.log10(returned))) if returned.size >= 2 else math.inf
    )
    mgcv_converged = returned.size >= 2 and mgcv_span <= 0.25

    if options.lambda_policy == "fixed":
        selected = float(options.fixed_lambda)
        method = "fixed"
        reason = "user-specified fixed lambda"
    elif options.lambda_policy == "lcurve-grid":
        selected = lcurve_lambda
        method = "grid-lcurve"
        reason = lcurve_status
    elif options.lambda_policy == "package-mgcv":
        if not mgcv_converged:
            raise RuntimeError(
                "Package mGCV did not provide independent in-range nonboundary candidates; "
                "use the consensus policy or prespecify a fixed lambda"
            )
        selected = float(10 ** np.mean(np.log10(returned)))
        method = "package-mgcv"
        reason = f"package mGCV requested and candidates agreed within {mgcv_span:.3f} decades"
    else:
        if mgcv_converged:
            selected = float(10 ** np.mean(np.log10(returned)))
            method = "multi-start-mgcv"
            reason = f"mGCV candidates agreed within {mgcv_span:.3f} decades"
        else:
            selected = lcurve_lambda
            method = "grid-lcurve-fallback"
            reason = (
                "package mGCV did not demonstrate start-independent convergence; "
                f"used {lcurve_status}"
            )

    proposed_selected = selected
    selected = float(np.clip(selected, options.lambda_min, options.lambda_max))
    guard_indices = np.flatnonzero(np.isfinite(chord_distance) & ~eligible)
    guard_has_larger_distance = bool(
        guard_indices.size
        and math.isfinite(float(chord_distance[lcurve_index]))
        and float(np.max(chord_distance[guard_indices])) > float(chord_distance[lcurve_index])
    )
    lcurve_touches_eligible_boundary = bool(
        lcurve_index in {boundary_guard, grid.size - boundary_guard - 1}
    )
    lcurve_boundary_sensitive = bool(
        method in {"grid-lcurve", "grid-lcurve-fallback"}
        and (lcurve_touches_eligible_boundary or guard_has_larger_distance or not finite_indices.size)
    )
    full_sensitivity_range = bool(selected / 10 >= options.lambda_min * (1-1e-12)
                                  and selected * 10 <= options.lambda_max * (1+1e-12))
    boundary_reasons = []
    if lcurve_boundary_sensitive:
        boundary_reasons.append("L-curve corner is guarded, edge-adjacent or undefined")
    if not full_sensitivity_range:
        boundary_reasons.append("search limits truncate the full lambda +/- one-decade neighborhood")
    if selected != proposed_selected:
        boundary_reasons.append("proposed lambda was clipped")
    cache_key = min(cache, key=lambda value: abs(math.log10(value) - math.log10(selected)))
    if abs(math.log10(cache_key) - math.log10(selected)) < 1e-10:
        selected_result = cache[cache_key]
    else:
        selected_result = fixed_tr_rbf(data, selected, inductance, options)
        cache[selected] = selected_result
    diagnostics = {
        "policy": options.lambda_policy,
        "selected_lambda": selected,
        "proposed_selected_lambda": proposed_selected,
        "lambda_was_clipped": selected != proposed_selected,
        "requested_lambda_range": [options.lambda_min, options.lambda_max],
        "full_sensitivity_range": full_sensitivity_range,
        "selection_method": method,
        "selection_reason": reason,
        "grid": rows,
        "grid_lcurve_lambda": lcurve_lambda,
        "grid_lcurve_index": lcurve_index,
        "lcurve_boundary_guard_points": boundary_guard,
        "lcurve_touches_eligible_boundary": lcurve_touches_eligible_boundary,
        "lcurve_guard_has_larger_chord_distance": guard_has_larger_distance,
        "boundary_sensitive": bool(boundary_reasons),
        "boundary_reason": "; ".join(boundary_reasons) or None,
        "mgcv_candidates": mgcv_candidates,
        "mgcv_log10_span_decades": mgcv_span if math.isfinite(mgcv_span) else None,
        "mgcv_start_independent": mgcv_converged,
        "mgcv_optimizer_status": "not-exposed-by-pyimpspec; start agreement is not an optimizer success flag",
        "mgcv_package_search_bounds": [1e-7, 1.0],
        "warning": (
            None if mgcv_converged or options.lambda_policy in {"fixed", "lcurve-grid"}
            else "Do not describe the returned default lambda as an automatic optimum"
        ),
    }
    return selected, selected_result, diagnostics, cache


def peak_basins(time_constants: np.ndarray, gammas: np.ndarray) -> list[dict[str, Any]]:
    time_constants = np.asarray(time_constants, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    if time_constants.size < 3:
        return []
    order = np.argsort(time_constants)
    time_constants = time_constants[order]
    gammas = gammas[order]
    peak_indices = np.flatnonzero(
        (gammas[1:-1] > gammas[:-2]) & (gammas[1:-1] >= gammas[2:])
    ) + 1
    if gammas[0] > gammas[1]:
        peak_indices = np.insert(peak_indices, 0, 0)
    if gammas[-1] > gammas[-2]:
        peak_indices = np.append(peak_indices, gammas.size - 1)
    if not peak_indices.size:
        return []
    boundaries = [0]
    for left, right in zip(peak_indices[:-1], peak_indices[1:]):
        local = int(left + np.argmin(gammas[left : right + 1]))
        boundaries.append(local)
    boundaries.append(gammas.size - 1)
    output: list[dict[str, Any]] = []
    max_height = max(float(np.max(gammas[peak_indices])), np.finfo(float).tiny)
    for i, peak_index in enumerate(peak_indices):
        left = boundaries[i]
        right = boundaries[i + 1]
        if right <= left:
            right = min(left + 1, gammas.size - 1)
        area = float(np.trapezoid(gammas[left : right + 1], x=np.log(time_constants[left : right + 1])))
        output.append({
            "index": int(peak_index),
            "tau_s": float(time_constants[peak_index]),
            "frequency_hz": float(1.0 / (2.0 * math.pi * time_constants[peak_index])),
            "gamma": float(gammas[peak_index]),
            "relative_height_global": float(gammas[peak_index] / max_height),
            "area_impedance_full_diagnostic": max(area, 0.0),
            "basin_tau_min_s": float(time_constants[left]),
            "basin_tau_max_s": float(time_constants[right]),
        })
    return output


def integrate_gamma_interval(
    time_constants: np.ndarray,
    gammas: np.ndarray,
    tau_min: float,
    tau_max: float,
) -> float:
    order = np.argsort(time_constants)
    tau = np.asarray(time_constants, dtype=float)[order]
    gamma = np.asarray(gammas, dtype=float)[order]
    x = np.log(tau)
    lower = max(math.log(tau_min), float(x[0]))
    upper = min(math.log(tau_max), float(x[-1]))
    if upper <= lower:
        return 0.0
    interior = (x > lower) & (x < upper)
    sample_x = np.concatenate(([lower], x[interior], [upper]))
    sample_y = np.interp(sample_x, x, gamma)
    return max(float(np.trapezoid(sample_y, x=sample_x)), 0.0)


def annotate_peak_support(
    peaks: list[dict[str, Any]],
    time_constants: np.ndarray,
    gammas: np.ndarray,
    fmin: float,
    fmax: float,
) -> list[dict[str, Any]]:
    measured_tau_min = 1.0 / (2.0 * math.pi * fmax)
    measured_tau_max = 1.0 / (2.0 * math.pi * fmin)
    for peak in peaks:
        basin_min = float(peak["basin_tau_min_s"])
        basin_max = float(peak["basin_tau_max_s"])
        supported_min = max(basin_min, measured_tau_min)
        supported_max = min(basin_max, measured_tau_max)
        in_window_area = integrate_gamma_interval(
            time_constants,
            gammas,
            supported_min,
            supported_max,
        ) if supported_max > supported_min else 0.0
        full_area = float(peak["area_impedance_full_diagnostic"])
        basin_span = max(math.log(basin_max / basin_min), np.finfo(float).tiny)
        supported_span = max(math.log(supported_max / supported_min), 0.0) if supported_max > supported_min else 0.0
        peak_inside = fmin <= float(peak["frequency_hz"]) <= fmax
        peak.update({
            "area_impedance": in_window_area if peak_inside else None,
            "in_window_area_impedance": in_window_area,
            "area_supported": basin_min >= measured_tau_min and basin_max <= measured_tau_max,
            "area_supported_fraction": (
                min(in_window_area / full_area, 1.0) if full_area > 0.0 else None
            ),
            "basin_window_fraction": min(supported_span / basin_span, 1.0),
            "measured_tau_min_s": measured_tau_min,
            "measured_tau_max_s": measured_tau_max,
        })
    return peaks


def match_peak(
    target_tau: float,
    candidates: list[dict[str, Any]],
    max_decades: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    distances = [abs(math.log10(item["tau_s"] / target_tau)) for item in candidates]
    index = int(np.argmin(distances))
    return candidates[index] if distances[index] <= max_decades else None


def compare_peak_sets_one_to_one(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    max_decades: float,
) -> dict[str, Any]:
    secondary.sort(key=lambda row: row["frequency_hz"], reverse=True)
    for index, peak in enumerate(secondary, start=1):
        peak["secondary_peak_id"] = f"S{index}"
    distances = np.full((len(primary), len(secondary)), np.inf, dtype=float)
    for i, left in enumerate(primary):
        for j, right in enumerate(secondary):
            distances[i, j] = abs(math.log10(left["frequency_hz"] / right["frequency_hz"]))
    primary_candidates = np.sum(distances <= max_decades, axis=1) if distances.size else np.zeros(len(primary), dtype=int)
    secondary_candidates = np.sum(distances <= max_decades, axis=0) if distances.size else np.zeros(len(secondary), dtype=int)
    from peak_matching import match_positions
    assignment = match_positions([p["frequency_hz"] for p in primary],
                                 [p["frequency_hz"] for p in secondary], max_decades)
    matches = assignment["matches"]
    assigned_primary = set(matches)
    assigned_secondary = {v[0] for v in matches.values()}
    matched_rows: list[dict[str, Any]] = []
    for i, peak in enumerate(primary):
        match = matches.get(i)
        secondary_index = match[0] if match is not None else None
        matched_peak = secondary[secondary_index] if secondary_index is not None else None
        peak.update({
            "tr_nnls_match": matched_peak is not None,
            "tr_nnls_peak_id": matched_peak.get("secondary_peak_id") if matched_peak else None,
            "tr_nnls_frequency_hz": matched_peak.get("frequency_hz") if matched_peak else None,
            "tr_nnls_log_distance_decades": match[1] if match is not None else None,
            "tr_nnls_candidate_count": int(primary_candidates[i]),
            "tr_nnls_secondary_split_candidate": bool(primary_candidates[i] > 1),
            "tr_nnls_primary_merge_candidate": (
                bool(secondary_candidates[secondary_index] > 1)
                if secondary_index is not None else False
            ),
        })
        if matched_peak is not None:
            matched_rows.append({
                "primary_peak_id": peak.get("peak_id"),
                "secondary_peak_id": matched_peak["secondary_peak_id"],
                "log_distance_decades": match[1],
            })
    return {
        "matching_rule": assignment["matching_rule"],
        "matched_pairs": matched_rows,
        "unmatched_primary_peak_ids": [
            peak.get("peak_id") for i, peak in enumerate(primary) if i not in assigned_primary
        ],
        "unmatched_secondary_peak_ids": [
            peak["secondary_peak_id"] for j, peak in enumerate(secondary) if j not in assigned_secondary
        ],
        "split_or_merge_boundary": (
            "Candidate flags indicate local peak-count disagreement; a boolean match is not full algorithmic agreement"
        ),
    }


def classify_peaks(
    selected_peaks: list[dict[str, Any]],
    sensitivity: list[dict[str, Any]],
    fmin: float,
    fmax: float,
    edge_decades: float,
    match_decades: float,
    peak_threshold: float,
    raw_freq: np.ndarray,
    raw_neg_zimag: np.ndarray,
) -> list[dict[str, Any]]:
    log_raw = np.log10(raw_freq)
    order = np.argsort(log_raw)
    log_raw = log_raw[order]
    raw_imag = raw_neg_zimag[order]
    in_window_heights = [
        float(peak["gamma"])
        for peak in selected_peaks
        if fmin <= float(peak["frequency_hz"]) <= fmax
    ]
    max_in_window_height = max(in_window_heights, default=np.finfo(float).tiny)
    classified: list[dict[str, Any]] = []
    from peak_matching import match_positions
    assignments = [match_positions([p["tau_s"] for p in selected_peaks],
                                  [p["tau_s"] for p in item["peaks"]], match_decades)
                   for item in sensitivity]
    for peak_index, peak in enumerate(selected_peaks):
        matches: list[dict[str, Any]] = []
        split_merge = False
        for item, assignment in zip(sensitivity, assignments):
            split_merge |= assignment["resolution_change_left"][peak_index]
            match = assignment["matches"].get(peak_index)
            if match is not None:
                matches.append({"lambda": item["lambda"], **item["peaks"][match[0]]})
        frequencies = [item["frequency_hz"] for item in matches]
        areas = [
            float(item["area_impedance"])
            for item in matches
            if item.get("area_impedance") is not None
        ]
        position_span = (
            float(np.ptp(np.log10(frequencies))) if len(frequencies) >= 2 else None
        )
        area_cv = (
            float(np.std(areas, ddof=1) / np.mean(areas))
            if len(areas) >= 2 and float(np.mean(areas)) > 0.0 else None
        )
        fpeak = float(peak["frequency_hz"])
        outside = fpeak < fmin or fpeak > fmax
        relative_height = float(peak["gamma"] / max_in_window_height) if not outside else None
        edge_distance = min(
            abs(math.log10(fpeak / fmin)),
            abs(math.log10(fmax / fpeak)),
        ) if not outside else -1.0
        edge_sensitive = not outside and edge_distance < edge_decades
        inductive_overlap = False
        if fmin <= fpeak <= fmax:
            interpolated = float(np.interp(math.log10(fpeak), log_raw, raw_imag))
            inductive_overlap = interpolated < 0.0
        if outside:
            confidence = "unsupported-outside-window"
        elif edge_sensitive:
            confidence = "boundary-sensitive"
        elif inductive_overlap:
            confidence = "inductive-overlap"
        elif relative_height is not None and relative_height < peak_threshold:
            confidence = "minor-low-signal"
        elif split_merge:
            confidence = "split-merge-ambiguous"
        elif len(matches) >= 3 and len(matches) == len(sensitivity) and (position_span or 0.0) <= 0.25 and (area_cv is None or area_cv <= 0.35):
            confidence = "robust-to-lambda"
        elif len(matches) >= 2 and (position_span is None or position_span <= 0.5):
            confidence = "tentative"
        else:
            confidence = "unstable"
        item = dict(peak)
        item.update({
            "lambda_split_merge_ambiguous": split_merge,
            "lambda_match_rule": "threshold-aware maximum-cardinality one-to-one",
            "lambda_persistence": len(matches),
            "lambda_cases": len(sensitivity),
            "position_span_decades": position_span,
            "area_cv": area_cv,
            "edge_distance_decades": edge_distance,
            "inductive_overlap": inductive_overlap,
            "relative_height": relative_height,
            "relative_height_reference": "largest peak whose maximum lies inside the measured frequency window",
            "confidence": confidence,
            "mechanism_assignment": "unassigned",
            "mechanism_boundary": "Frequency position alone is not sufficient for process assignment",
        })
        classified.append(item)
    classified.sort(key=lambda row: row["frequency_hz"], reverse=True)
    for index, peak in enumerate(classified, start=1):
        peak["peak_id"] = f"P{index}"
    return classified


def plot_spectrum(result: dict[str, Any], figure_dir: Path) -> list[str]:
    """Render the shared-style primary audit with collision-checked L-curve ticks."""
    from diagnostic_plotting import save_primary_audit
    return save_primary_audit(result, figure_dir)


def analyze_payload(payload: dict[str, Any], options: WorkerOptions, figure_dir: Path) -> dict[str, Any]:
    from advanced_drt import (
        continuous_signed_gdrt,
        ddt_analysis,
        detect_diffusion_tail,
        loewner_rc_rl_analysis,
        model_reduce_analysis,
        plot_advanced_audit,
        sweep_drift_screen,
    )
    from pyimpspec import DataSet, perform_exploratory_kramers_kronig_tests, perform_zhit
    from numerical_conditioning import (calculate_tr_rbf_conditioned as calculate_drt_tr_rbf,
                                        calculate_tr_nnls_conditioned as calculate_drt_tr_nnls,
                                        conditioning_info)

    np.random.seed(options.seed)
    warnings = list(payload.get("warnings", []))
    freq = np.asarray(payload["frequency_hz"], dtype=float)
    zreal = np.asarray(payload["zreal"], dtype=float)
    neg_zimag = np.asarray(payload["neg_zimag"], dtype=float)
    if freq.size < 15:
        raise ValueError(f"Only {freq.size} valid points; at least 15 are required for screening DRT")
    if np.unique(freq).size != freq.size:
        raise ValueError("Duplicate frequencies are present; automatic averaging is intentionally disabled")
    order = np.argsort(freq)[::-1]
    freq = freq[order]
    zreal = zreal[order]
    neg_zimag = neg_zimag[order]
    acquisition_index = np.asarray(
        payload.get("acquisition_index", np.arange(freq.size)), dtype=float
    )[order]
    measured_z = zreal - 1j * neg_zimag
    decades = float(math.log10(float(np.max(freq)) / float(np.min(freq))))
    if decades < 2.0:
        warnings.append(f"Narrow frequency coverage ({decades:.2f} decades) limits DRT resolution")
    data = DataSet(freq, measured_z, label=payload["label"], path=payload["source"])
    kk_data = data
    kk_indices = np.arange(freq.size, dtype=int)
    if 0 < options.kk_max_points < freq.size:
        kk_indices = np.unique(
            np.rint(np.linspace(0, freq.size - 1, options.kk_max_points)).astype(int)
        )
        kk_data = DataSet(
            freq[kk_indices],
            measured_z[kk_indices],
            label=payload["label"],
            path=payload["source"],
        )

    kk: dict[str, Any]
    kk_gate = "not-run"
    try:
        _, suggestion = perform_exploratory_kramers_kronig_tests(
            kk_data,
            test="complex",
            admittance=None,
            add_capacitance=True,
            add_inductance=True,
            timeout=options.solver_timeout,
            num_procs=options.num_procs,
        )
        kk_result = suggestion[0]
        _, kk_re, kk_im = kk_result.get_residuals_data()
        kk_re = np.asarray(kk_re, dtype=float)
        kk_im = np.asarray(kk_im, dtype=float)
        kk_max_rms = max(rms(kk_re), rms(kk_im))
        if kk_max_rms <= options.kk_warn_percent:
            kk_gate = "screen-pass"
        elif kk_max_rms <= options.kk_block_percent:
            kk_gate = "warn"
            warnings.append(f"KK residual RMS reaches {kk_max_rms:.3g}%")
        else:
            kk_gate = "suspect"
            warnings.append(f"KK residual RMS reaches {kk_max_rms:.3g}% and exceeds the screening gate")
        kk = {
            "status": kk_gate,
            "representation": "admittance" if kk_result.was_tested_on_admittance() else "impedance",
            "num_RC": int(kk_result.get_num_RC()),
            "log_F_ext": float(kk_result.get_log_F_ext()),
            "pseudo_chisqr": float(kk_result.pseudo_chisqr),
            "residual_rms_re_percent": rms(kk_re),
            "residual_rms_im_percent": rms(kk_im),
            "residual_lag1_re": lag1_correlation(kk_re),
            "residual_lag1_im": lag1_correlation(kk_im),
            "source_point_count": int(freq.size),
            "screen_point_count": int(kk_indices.size),
            "downsampled_for_screen": bool(kk_indices.size < freq.size),
            "screen_point_selection": (
                "equal acquisition-index coverage including both endpoints"
                if kk_indices.size < freq.size
                else "all source points"
            ),
            "claim_boundary": (
                "KK consistency does not independently prove linearity or stationarity; "
                "a downsampled KK result is a batch-screening result and selected "
                "publication spectra require a full-point rerun"
            ),
        }
    except Exception as exc:
        kk = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        kk_gate = "failed"
        warnings.append("KK calculation failed")
    if options.strict and kk_gate in {"suspect", "failed"}:
        raise RuntimeError(f"Strict mode blocked DRT because KK status is {kk_gate}")

    try:
        zhit_result = perform_zhit(data, num_procs=options.num_procs)
        _, zhit_re, zhit_im = zhit_result.get_residuals_data()
        zhit = {
            "status": "completed",
            "pseudo_chisqr": float(zhit_result.pseudo_chisqr),
            "residual_rms_re_percent": rms(np.asarray(zhit_re)),
            "residual_rms_im_percent": rms(np.asarray(zhit_im)),
            "smoothing": str(zhit_result.smoothing),
            "interpolation": str(zhit_result.interpolation),
            "window": str(zhit_result.window),
        }
    except Exception as exc:
        zhit = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        warnings.append("Z-HIT calculation failed")

    inductance_detected, inductance_evidence = high_frequency_inductance(freq, neg_zimag)
    inductance, inductance_selection = choose_primary_inductance_mode(
        data, measured_z, inductance_detected, inductance_evidence, options
    )
    if inductance_detected and not inductance:
        warnings.append(
            "High-frequency inductance was detected but the no-L candidate remained selected. "
            "Inspect the recorded candidate qualities and use a fixture/dummy-cell control before interpretation."
        )
    screening_l = (
        float(inductance_evidence.get("estimated_series_inductance_h") or 0.0)
        if inductance_evidence.get("series_l_explainable") else 0.0
    )
    nonseries_inductance, nonseries_evidence = nonseries_inductive_region(
        freq, neg_zimag, screening_l, impedance_magnitude=np.abs(measured_z)
    )
    if nonseries_inductance and options.signed_gdrt == "off":
        if options.strict:
            raise ValueError(
                "Residual finite/low-frequency inductive response remains after plausible series-L "
                "compensation. Strict mode requires signed/GDRT or a justified physical model."
            )
        warnings.append(
            "Residual finite/low-frequency inductive response remains after plausible series-L "
            "compensation. The ordinary RC result is a baseline only because signed/GDRT was disabled."
        )
    elif nonseries_inductance:
        warnings.append(
            "Residual finite/low-frequency inductive response remains after plausible series-L "
            "compensation; signed/GDRT or Loewner RC/RL is required for the RC/RL comparison."
        )
    selected_lambda, selected_result, lambda_info, cache = select_lambda(
        data, measured_z, inductance, options
    )
    if lambda_info.get("boundary_sensitive"):
        warnings.append(str(lambda_info["boundary_reason"]))
    sensitivity_values = sorted({
        float(np.clip(selected_lambda / 10.0, options.lambda_min, options.lambda_max)),
        selected_lambda,
        float(np.clip(selected_lambda * 10.0, options.lambda_min, options.lambda_max)),
    })
    sensitivity: list[dict[str, Any]] = []
    for value in sensitivity_values:
        key = min(cache, key=lambda candidate: abs(math.log10(candidate) - math.log10(value)))
        if abs(math.log10(key) - math.log10(value)) < 1e-10:
            drt = cache[key]
        else:
            drt = fixed_tr_rbf(data, value, inductance, options)
            cache[value] = drt
        tau, gamma = drt.get_drt_data()
        tau_array = np.asarray(tau, dtype=float)
        gamma_array = np.asarray(gamma, dtype=float)
        sensitivity.append({
            "lambda": value,
            "selected": abs(math.log10(value / selected_lambda)) < 1e-10,
            "tau_s": tau_array,
            "frequency_hz": 1.0 / (2.0 * math.pi * tau_array),
            "gamma": gamma_array,
            "peaks": annotate_peak_support(
                peak_basins(tau_array, gamma_array),
                tau_array,
                gamma_array,
                float(np.min(freq)),
                float(np.max(freq)),
            ),
            "pseudo_chisqr": float(drt.pseudo_chisqr),
        })

    credible_completed = False
    if options.credible_intervals:
        try:
            selected_result = calculate_drt_tr_rbf(
                data,
                impedance_scaling=options.impedance_scaling,
                mode="complex",
                lambda_value=selected_lambda,
                cross_validation="",
                rbf_type=options.rbf_type,
                derivative_order=options.derivative_order,
                rbf_shape=options.rbf_shape,
                shape_coeff=options.shape_coeff,
                inductance=inductance,
                credible_intervals=True,
                num_samples=options.credible_samples,
                timeout=options.solver_timeout,
                num_procs=options.num_procs,
            )
            credible_completed = True
        except Exception as exc:
            warnings.append(
                f"Credible-interval calculation failed and the point estimate was retained: "
                f"{type(exc).__name__}: {exc}"
            )
    tau, gamma = selected_result.get_drt_data()
    tau = np.asarray(tau, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    selected_peaks = annotate_peak_support(
        peak_basins(tau, gamma),
        tau,
        gamma,
        float(np.min(freq)),
        float(np.max(freq)),
    )
    peaks = classify_peaks(
        selected_peaks,
        sensitivity,
        float(np.min(freq)),
        float(np.max(freq)),
        options.edge_decades,
        options.peak_match_decades,
        options.peak_threshold,
        freq,
        neg_zimag,
    )

    try:
        nnls = calculate_drt_tr_nnls(data, mode="real", lambda_value=-2.0,
                                   impedance_scaling=options.impedance_scaling)
        nnls_tau, nnls_gamma = nnls.get_drt_data()
        nnls_tau = np.asarray(nnls_tau, dtype=float)
        nnls_gamma = np.asarray(nnls_gamma, dtype=float)
        nnls_peaks = annotate_peak_support(
            peak_basins(nnls_tau, nnls_gamma),
            nnls_tau,
            nnls_gamma,
            float(np.min(freq)),
            float(np.max(freq)),
        )
        comparison = compare_peak_sets_one_to_one(
            peaks,
            nnls_peaks,
            options.peak_match_decades,
        )
        secondary = {
            "status": "completed",
            "method": "TR-NNLS-real-L-curve",
            "lambda": float(nnls.lambda_value),
            "pseudo_chisqr": float(nnls.pseudo_chisqr),
            "peak_count": len(nnls_peaks),
            "peak_frequencies_hz": [row["frequency_hz"] for row in nnls_peaks],
            "peaks": nnls_peaks,
            "comparison": comparison,
            "purpose": "algorithm-dependence diagnostic; not the primary reported DRT",
        }
    except Exception as exc:
        secondary = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        for peak in peaks:
            peak["tr_nnls_match"] = None
            peak["tr_nnls_peak_id"] = None
            peak["tr_nnls_frequency_hz"] = None
            peak["tr_nnls_log_distance_decades"] = None
            peak["tr_nnls_candidate_count"] = None
            peak["tr_nnls_secondary_split_candidate"] = None
            peak["tr_nnls_primary_merge_candidate"] = None

    reconstructed = np.asarray(selected_result.get_impedances(), dtype=complex)
    drift_screen = sweep_drift_screen(
        acquisition_index,
        measured_z,
        reconstructed,
        acquisition_order_available=payload.get("metadata", {}).get("acquisition_order_preserved") is not False,
    )
    residual_f, residual_re, residual_im = selected_result.get_residuals_data()
    residual_re = np.asarray(residual_re, dtype=float)
    residual_im = np.asarray(residual_im, dtype=float)
    credible_lower: np.ndarray = np.asarray([], dtype=float)
    credible_upper: np.ndarray = np.asarray([], dtype=float)
    credible_mean: np.ndarray = np.asarray([], dtype=float)
    if credible_completed:
        ci_tau, ci_mean, ci_lower, ci_upper = selected_result.get_drt_credible_intervals_data()
        if len(ci_tau) == len(tau):
            credible_mean = np.asarray(ci_mean, dtype=float)
            credible_lower = np.asarray(ci_lower, dtype=float)
            credible_upper = np.asarray(ci_upper, dtype=float)

    result = {
        "status": "completed",
        "spectrum_id": payload["spectrum_id"],
        "spectrum_uid": payload["spectrum_uid"],
        "source_uid": payload["source_uid"],
        "run_fingerprint": payload.get("run_fingerprint"),
        "result_slug": payload.get("result_slug"),
        "label": payload["label"],
        "source": payload["source"],
        "source_sha256": payload["source_sha256"],
        "impedance_basis": payload["impedance_basis"],
        "metadata": payload["metadata"],
        "qc": payload["qc"],
        "frequency_decades": decades,
        "validity": {
            "linearity": payload["metadata"].get("linearity_status", "unknown"),
            "stationarity": payload["metadata"].get("stationarity_status", "unknown"),
            "kk_consistency": kk_gate,
            "single_sweep_drift_screen": drift_screen,
            "note": "A single spectrum cannot establish perturbation linearity or time invariance",
        },
        "kk": kk,
        "zhit": zhit,
        "inductance_mode": inductance,
        "inductance_detected": inductance_detected,
        "inductance_used_in_primary_drt": inductance,
        "inductance_policy": options.inductance_policy,
        "inductance_evidence": inductance_evidence,
        "inductance_model_selection": inductance_selection,
        "nonseries_inductance_evidence": nonseries_evidence,
        "lambda_selection": lambda_info,
        "drt": {
            "method": "TR-RBF",
            "numerical_conditioning": conditioning_info(data, mode=options.impedance_scaling),
            "fit_weighting": "uniform absolute real/imaginary errors; scalar conditioning is not pointwise noise weighting",
            "fitted_series_L_henry": None,
            "fitted_R_inf": None,
            "nuisance_export_status": "pyimpspec TRRBFResult does not expose fitted nuisance coefficients; high-frequency L estimate is separate",
            "mode": "complex",
            "lambda": selected_lambda,
            "rbf_type": options.rbf_type,
            "rbf_shape": options.rbf_shape,
            "shape_coeff": options.shape_coeff,
            "derivative_order": options.derivative_order,
            "pseudo_chisqr": float(selected_result.pseudo_chisqr),
            "residual_rms_re_percent": rms(residual_re),
            "residual_rms_im_percent": rms(residual_im),
            "credible_intervals": credible_completed,
            "credible_samples": options.credible_samples if credible_completed else 0,
            "credible_interval_definition": (
                "pyimpspec conditional posterior 0.5%–99.5% quantiles; does not include lambda-selection or replicate uncertainty"
                if credible_completed else None
            ),
            "area_definition": (
                "basin integral of gamma over natural log(tau); area_impedance is clipped to the "
                "measured tau window and is null when the peak maximum is outside that window; "
                "area_impedance_full_diagnostic retains the extrapolated integral"
            ),
        },
        "secondary_method": secondary,
        "peaks": peaks,
        "warnings": warnings,
        "claim_level": (
            "diagnostic_only"
            if kk_gate in {"suspect", "failed"} or nonseries_inductance
            else "numerically_supported_mechanism_unassigned"
        ),
        "claim_boundary": (
            "Peak positions, areas, and stability are numerical evidence only. "
            "Mechanism assignment requires controlled SOC/temperature/cycle trends, repeats, "
            "electrode-side baselines, or independent chemical/structural evidence."
        ),
        "curves": {
            "acquisition_index": acquisition_index,
            "frequency_hz": np.asarray(residual_f, dtype=float),
            "zreal_measured": zreal,
            "neg_zimag_measured": neg_zimag,
            "zreal_reconstructed": reconstructed.real,
            "neg_zimag_reconstructed": -reconstructed.imag,
            "residual_re_percent": residual_re,
            "residual_im_percent": residual_im,
            "tau_s": tau,
            "drt_frequency_hz": 1.0 / (2.0 * math.pi * tau),
            "gamma": gamma,
            "credible_mean": credible_mean,
            "credible_lower": credible_lower,
            "credible_upper": credible_upper,
            "sensitivity_curves": [
                {
                    "lambda": item["lambda"],
                    "selected": item["selected"],
                    "frequency_hz": item["frequency_hz"],
                    "gamma": item["gamma"],
                }
                for item in sensitivity
            ],
        },
    }
    loewner = (
        loewner_rc_rl_analysis(data, options.num_procs)
        if options.loewner
        else {"status": "disabled", "purpose": "discrete RC/RL signed-response diagnostic"}
    )
    result["loewner_rc_rl"] = loewner
    if loewner.get("status") == "completed":
        from loewner_validation import validate_loewner
        try:
            result["loewner_validation"] = validate_loewner(freq, measured_z, exported_model=loewner)
        except Exception as exc:
            result["loewner_validation"] = {"status": "failed", "error": str(exc)}
    else:
        result["loewner_validation"] = {"status": "disabled", "reason": "Loewner-not-completed"}

    loewner_rl_trigger = False
    if loewner.get("status") == "completed":
        gamma_rc = np.asarray(loewner.get("gamma_rc", []), dtype=float)
        gamma_rl = np.asarray(loewner.get("gamma_rl", []), dtype=float)
        rc_scale = max(float(np.max(gamma_rc)) if gamma_rc.size else 0.0, np.finfo(float).tiny)
        loewner_rl_trigger = bool(gamma_rl.size and float(np.max(gamma_rl)) >= 0.02 * rc_scale)
    material_loewner_rl_trigger = bool(
        loewner_rl_trigger
        and (nonseries_inductance or not inductance_evidence.get("series_l_explainable"))
    )
    run_signed = options.signed_gdrt == "always" or (
        options.signed_gdrt == "auto" and (nonseries_inductance or material_loewner_rl_trigger
            or float(np.hypot(rms(residual_re), rms(residual_im))) > 2.0)
    )
    if run_signed:
        try:
            signed = continuous_signed_gdrt(
                freq,
                measured_z,
                lambda_min=options.lambda_min,
                lambda_max=options.lambda_max,
                lambda_points=options.lambda_points,
                derivative_order=options.derivative_order,
            )
            signed["trigger"] = {
                "policy": options.signed_gdrt,
                "nonseries_inductance": nonseries_inductance,
                "loewner_rl_raw_trigger": loewner_rl_trigger,
                "loewner_material_rl": material_loewner_rl_trigger,
                "series_l_already_compensated_in_screen": screening_l > 0.0,
                "primary_fit_above_2_percent": float(np.hypot(rms(residual_re), rms(residual_im))) > 2.0,
            }
        except Exception as exc:
            signed = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            warnings.append("Continuous signed/GDRT calculation failed")
    else:
        signed = {
            "status": "not-triggered" if options.signed_gdrt == "auto" else "disabled",
            "trigger": {
                "policy": options.signed_gdrt,
                "nonseries_inductance": nonseries_inductance,
                "loewner_rl_raw_trigger": loewner_rl_trigger,
                "loewner_material_rl": material_loewner_rl_trigger,
                "series_l_already_compensated_in_screen": screening_l > 0.0,
            },
        }
    result["signed_gdrt"] = signed

    if (options.predictive_fit == "auto" and options.signed_gdrt != "off"
            and options.inductance_policy != "off" and
            (run_signed or float(np.hypot(rms(residual_re), rms(residual_im))) > 2.0)):
        from advanced_drt import predictive_rc_kernel_competition
        try:
            result["predictive_rc_competition"] = predictive_rc_kernel_competition(
                freq, measured_z, lambda_min=options.lambda_min, lambda_max=options.lambda_max,
                lambda_points=options.lambda_points, derivative_order=options.derivative_order)
        except Exception as exc:
            result["predictive_rc_competition"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            warnings.append("Predictive RC kernel competition failed; original branches retained")
    else:
        result["predictive_rc_competition"] = {"status": "disabled-or-not-triggered",
            "reason": "Requires auto predictive fit, enabled inductance/signed diagnostics, and signed trigger or primary RMS above 2 percent"}

    diffusion_trigger = detect_diffusion_tail(freq, measured_z)
    run_ddt = options.ddt == "always" or (options.ddt == "auto" and diffusion_trigger["detected"])
    if run_ddt:
        try:
            ddt = ddt_analysis(
                freq,
                measured_z,
                lambda_min=options.lambda_min,
                lambda_max=options.lambda_max,
                lambda_points=options.lambda_points,
                derivative_order=options.derivative_order,
            )
            ddt["policy"] = options.ddt
        except Exception as exc:
            ddt = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "trigger": diffusion_trigger}
            warnings.append("DDT calculation failed")
    else:
        ddt = {
            "status": "not-triggered" if options.ddt == "auto" else "disabled",
            "policy": options.ddt,
            "trigger": diffusion_trigger,
        }
    result["ddt"] = ddt

    if options.model_reduce:
        def solve_reduced(reduced_impedance: np.ndarray) -> dict[str, Any]:
            reduced_impedance = np.asarray(reduced_impedance, dtype=complex)
            reduced_inductance, _ = high_frequency_inductance(freq, -reduced_impedance.imag)
            reduced_inductance = reduced_inductance and options.inductance_policy != "off"
            reduced_data = DataSet(
                freq,
                reduced_impedance,
                label=f"{payload['label']} model-reduced",
                path=payload["source"],
            )
            reduced_result = fixed_tr_rbf(
                reduced_data,
                selected_lambda,
                reduced_inductance,
                options,
            )
            reduced_tau, reduced_gamma = reduced_result.get_drt_data()
            reduced_tau = np.asarray(reduced_tau, dtype=float)
            reduced_gamma = np.asarray(reduced_gamma, dtype=float)
            reduced_peaks = annotate_peak_support(
                peak_basins(reduced_tau, reduced_gamma),
                reduced_tau,
                reduced_gamma,
                float(np.min(freq)),
                float(np.max(freq)),
            )
            return {
                "reconstructed": np.asarray(reduced_result.get_impedances(), dtype=complex),
                "tau_s": reduced_tau,
                "gamma": reduced_gamma,
                "peaks": reduced_peaks,
            }

        try:
            model_reduce = model_reduce_analysis(
                freq,
                measured_z,
                reconstructed,
                peaks,
                solve_reduced,
                min_quality_improvement_log10=options.model_reduce_min_improvement,
                peak_position_limit_decades=options.preprocess_peak_position_decades,
                peak_area_fraction_limit=options.preprocess_peak_area_fraction,
            )
        except Exception as exc:
            model_reduce = {"status": "failed", "enabled": True, "error": f"{type(exc).__name__}: {exc}"}
            warnings.append("Optional model-and-reduce calculation failed")
    else:
        model_reduce = {
            "status": "disabled",
            "enabled": False,
            "claim_boundary": "Raw EIS was not altered; enable --model-reduce to evaluate nuisance branches.",
        }
    result["model_reduce"] = model_reduce
    if options.peak_evidence:
        from peak_evidence import assess_peak_evidence
        try:
            result["peak_evidence"] = assess_peak_evidence(
                freq, measured_z, tau, gamma, peaks,
                lambda_value=selected_lambda, reconstructed=reconstructed,
                inductance=inductance, seed=options.seed,
                solver_options={key: getattr(options, key) for key in
                    ("rbf_type", "derivative_order", "rbf_shape", "shape_coeff", "impedance_scaling")},
            )
            by_peak = {p["peak_id"]: p for p in result["peak_evidence"].get("per_peak", [])}
            for peak in peaks:
                evidence = by_peak.get(peak.get("peak_id"), {})
                peak["perturbation_evidence_status"] = evidence.get("status", "not-assessable")
                peak["physical_confirmation"] = False
        except Exception as exc:
            result["peak_evidence"] = {"status": "failed", "error": str(exc)}
    else:
        result["peak_evidence"] = {"status": "disabled"}
    from model_recommendation import recommend_model
    result["model_recommendation"] = recommend_model(result)
    recommendation = result["model_recommendation"]
    result.update(recommended_branch=recommendation.get("recommended_branch"),
                  recommendation_status=recommendation.get("status"),
                  recommended_rms_percent=recommendation.get("recommended_rms_percent"))
    result["warnings"] = warnings
    result["figure_status"] = "pending"
    result["figures"], result["advanced_figures"] = [], []
    return result


def render_result(result: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    """Plotting is a recoverable stage and cannot erase a numerical checkpoint."""
    from advanced_drt import plot_advanced_audit
    errors = []
    try:
        result["figures"] = plot_spectrum(result, figure_dir)
    except Exception as exc:
        result["figures"] = []
        errors.append(f"Primary plot: {type(exc).__name__}: {exc}")
    if any(result.get(key, {}).get("status") == "completed" for key in ("signed_gdrt", "ddt")) or result.get("model_reduce", {}).get("status") == "accepted":
        try:
            result["advanced_figures"] = plot_advanced_audit(result, figure_dir)
        except Exception as exc:
            result["advanced_figures"] = []
            errors.append(f"Advanced plot: {type(exc).__name__}: {exc}")
    result["figure_status"] = "failed" if errors else "completed"
    result["figure_errors"] = errors
    return result


def worker_entry(
    payload: dict[str, Any],
    options_dict: dict[str, Any],
    figure_dir: str,
    result_path: str,
) -> None:
    try:
        result = analyze_payload(payload, WorkerOptions(**options_dict), Path(figure_dir))
        from result_contract import numerical_digest
        result["numerical_sha256"] = numerical_digest(result)
        write_json(Path(result_path), result)
    except Exception as exc:
        result = {
            "status": "failed",
            "spectrum_id": payload.get("spectrum_id"),
            "label": payload.get("label"),
            "source": payload.get("source"),
            "source_sha256": payload.get("source_sha256"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "warnings": payload.get("warnings", []),
        }
    for key in ("spectrum_uid", "source_uid", "run_fingerprint", "result_slug", "metadata", "impedance_basis"):
        result.setdefault(key, payload.get(key))
    write_json(Path(result_path), result)


def run_isolated(
    payload: dict[str, Any],
    options: WorkerOptions,
    figure_dir: Path,
    result_path: Path,
    spectrum_timeout: int,
    render_timeout: int = 180,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    process = context.Process(
        target=worker_entry,
        args=(payload, asdict(options), str(figure_dir), str(result_path)),
    )
    process.start()
    process.join(spectrum_timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        if result_path.is_file():
            checkpoint = json.loads(result_path.read_text(encoding="utf-8"))
            if checkpoint.get("status") == "completed" and checkpoint.get("run_fingerprint") == payload.get("run_fingerprint"):
                checkpoint["figure_status"] = "timeout"
                checkpoint["figure_errors"] = ["Rendering exceeded the spectrum time budget; numerical checkpoint retained"]
                write_json(result_path, checkpoint)
                return checkpoint
        failure = {
            "status": "failed",
            "spectrum_id": payload["spectrum_id"],
            "label": payload["label"],
            "source": payload["source"],
            "source_sha256": payload["source_sha256"],
            "error": f"Spectrum analysis exceeded {spectrum_timeout} s and was terminated",
            "warnings": payload.get("warnings", []),
        }
        write_json(result_path, failure)
        return failure
    if not result_path.is_file():
        failure = {
            "status": "failed",
            "spectrum_id": payload["spectrum_id"],
            "label": payload["label"],
            "source": payload["source"],
            "source_sha256": payload["source_sha256"],
            "error": f"Worker exited with code {process.exitcode} without a result file",
            "warnings": payload.get("warnings", []),
        }
        write_json(result_path, failure)
        return failure
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") == "completed":
        result = run_render_isolated(result, figure_dir, result_path, render_timeout)
    return result


def render_worker_entry(result_path: str, figure_dir: str) -> None:
    path = Path(result_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result = render_result(result, Path(figure_dir))
    write_json(path, result)


def run_render_isolated(result, figure_dir, result_path, render_timeout=180):
    """Independent rendering budget, including resumed checkpoints; never erase a fit."""
    write_json(result_path, result)
    process = mp.get_context("spawn").Process(target=render_worker_entry,
        args=(str(result_path), str(figure_dir)))
    try:
        process.start()
    except Exception as exc:
        result["figure_status"] = "failed"
        result["figure_errors"] = [f"Render process could not start: {type(exc).__name__}: {exc}"]
        write_json(result_path, result)
        return result
    process.join(render_timeout)
    if process.is_alive():
        process.terminate()
        process.join(10)
        result["figure_status"] = "timeout"
        result["figure_errors"] = [f"Rendering exceeded its separate {render_timeout} s budget; numerical checkpoint retained"]
    elif process.exitcode != 0:
        result["figure_status"] = "failed"
        result["figure_errors"] = [f"Render process exited with code {process.exitcode}; numerical checkpoint retained"]
    else:
        try:
            candidate = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            result["figure_status"] = "failed"
            result["figure_errors"] = [f"Rendered checkpoint unreadable: {exc}"]
            write_json(result_path, result)
            return result
        from result_contract import numerical_digest
        if numerical_digest(candidate) != numerical_digest(result):
            result["figure_status"] = "failed"
            result["figure_errors"] = ["Renderer altered numerical content; original checkpoint restored"]
        else:
            result = candidate
    write_json(result_path, result)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(dict.fromkeys(fields + [key for row in rows for key in row]))
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_batch_overlay(results: list[dict[str, Any]], figure_dir: Path) -> list[str]:
    completed = [result for result in results if result.get("status") == "completed"
                 and result.get("ordinary_rc_eligible", False)]
    if len(completed) < 2:
        return []
    bases = {result["impedance_basis"] for result in completed}
    if len(bases) != 1:
        return []
    from diagnostic_plotting import save_batch_overlay
    return save_batch_overlay(completed, figure_dir)


def md_number(value: Any, precision: int = 6) -> str:
    number = finite_float(value)
    return f"{number:.{precision}g}" if number is not None else "—"


def build_report(
    run: dict[str, Any],
    results: list[dict[str, Any]],
    batch_validation: dict[str, Any] | None = None,
) -> str:
    completed = [row for row in results if row.get("status") == "completed"]
    failed = [row for row in results if row.get("status") != "completed"]
    lines = [
        "# EIS-DRT batch analysis report",
        "",
        f"- Generated: {run['generated_at']}",
        f"- Sources inventoried: {run['source_count']}",
        f"- Spectra discovered: {run['spectrum_count']}",
        f"- Completed: {len(completed)}",
        f"- Failed or blocked: {len(failed)}",
        "- Primary method: complex TR-RBF with adaptive no-L versus series-L model selection",
        f"- Lambda policy: {run['parameters']['lambda_policy']}",
        f"- Reference compatibility profile: {run['parameters'].get('reference_profile', 'none')}",
        "",
        "> Evidence boundary: DRT is an ill-posed inversion. Peak position and area do not identify a mechanism by themselves. KK consistency is not proof of perturbation linearity or stationarity.",
        "",
        "## Spectrum summary",
        "",
        "| Spectrum | Status | Basis | KK | Lambda | Conditional CI | DRT RMS Re/Im (%) | In-window | Robust | Unsupported |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for result in results:
        if result.get("status") == "completed":
            in_window_count = sum(
                peak.get("confidence") != "unsupported-outside-window" for peak in result["peaks"]
            )
            robust_count = sum(
                peak.get("confidence") == "robust-to-lambda" for peak in result["peaks"]
            )
            unsupported_count = sum(
                peak.get("confidence") == "unsupported-outside-window" for peak in result["peaks"]
            )
            lines.append(
                f"| {result['label']} | completed | {result['impedance_basis']} | "
                f"{result['kk'].get('status')} | {result['drt']['lambda']:.3g} | "
                f"{result['drt']['credible_intervals']} | "
                f"{result['drt']['residual_rms_re_percent']:.3g}/{result['drt']['residual_rms_im_percent']:.3g} | "
                f"{in_window_count} | {robust_count} | {unsupported_count} |"
            )
        else:
            lines.append(f"| {result.get('label')} | failed | — | — | — | — | — | — | — | — |")

    lines.extend(["", "## Reconstruction recommendations", "",
        "These choices concern impedance reconstruction only, not physical RC/RL mechanisms. "
        "A signed negative lobe can be a model/regularization artifact, even when its sensitivity test is stable. "
        "Primary curves remain in primary_drt; explicitly selected curves are in 03_results/recommended.", "",
        "| Spectrum | Selected branch | Primary RMS (%) | Selected RMS (%) | Recommendation status |",
        "| --- | --- | ---: | ---: | --- |"])
    for item in completed:
        rec = item.get("model_recommendation", {})
        lines.append(f"| {item.get('label')} | {rec.get('recommended_branch')} | "
                     f"{md_number(rec.get('primary_rms_percent'))} | {md_number(rec.get('recommended_rms_percent'))} | {rec.get('status')} |")

    if batch_validation is not None:
        coverage = batch_validation.get("coverage", {})
        verdict = batch_validation.get("overall_verdict", "not-assessed")
        lines.extend([
            "",
            "## Full acceptance assessment",
            "",
            f"- Overall verdict: **{verdict}**",
            f"- Repeat evidence: {coverage.get('repeat_status')} ({coverage.get('repeat_groups', 0)} groups)",
            f"- Amplitude-linearity evidence: {coverage.get('linearity_status')} ({coverage.get('linearity_groups', 0)} groups)",
            f"- Rest/stationarity evidence: {coverage.get('stationarity_status')} ({coverage.get('stationarity_groups', 0)} groups)",
            "- Missing repeat, amplitude, or rest metadata is reported as missing evidence; it is never converted into a pass.",
        ])
        for title, key in (
            ("Repeat and anomaly confirmation", "repeat_assessments"),
            ("Amplitude linearity", "linearity_assessments"),
            ("Rest-time stability", "stationarity_assessments"),
        ):
            rows = batch_validation.get(key, [])
            lines.extend(["", f"### {title}", ""])
            if not rows:
                lines.append("- Not assessable from the supplied metadata/spectra.")
                continue
            for row in rows:
                member_text = ", ".join(row.get("members", []))
                detail = member_text or ", ".join(
                    str(value)
                    for value in row.get("amplitudes_mv", row.get("rest_times_s", []))
                )
                if key == "repeat_assessments":
                    pair_rms = [
                        pair.get("complex_rms_percent")
                        for pair in row.get("pair_assessments", [])
                        if pair.get("complex_rms_percent") is not None
                    ]
                    outliers = [
                        item.get("label") for item in row.get("anomaly_scores", []) if item.get("outlier")
                    ]
                    metrics = (
                        f"max pair RMS={md_number(max(pair_rms) if pair_rms else None)}%; "
                        f"threshold={md_number(row.get('threshold_percent'))}%; "
                        f"outliers={', '.join(outliers) if outliers else 'none'}"
                    )
                elif key == "linearity_assessments":
                    metrics = (
                        f"max pair RMS={md_number(row.get('maximum_pair_rms_percent'))}%; "
                        f"threshold={md_number(row.get('threshold_percent'))}%"
                    )
                else:
                    metrics = (
                        f"last consecutive RMS={md_number(row.get('last_pair_rms_percent'))}%; "
                        f"threshold={md_number(row.get('threshold_percent'))}%"
                    )
                lines.append(
                    f"- `{row.get('group')}`: **{row.get('status')}**; {detail}; {metrics}"
                )

    for result in completed:
        unit = "Ω cm²" if result["impedance_basis"] == "ohm_cm2" else "Ω"
        qc = result["qc"]
        kk = result["kk"]
        zhit = result["zhit"]
        validity = result["validity"]
        inductance = result["inductance_evidence"]
        inductance_selection = result.get("inductance_model_selection", {})
        secondary = result["secondary_method"]
        model_reduce = result.get("model_reduce", {})
        loewner = result.get("loewner_rc_rl", {})
        signed = result.get("signed_gdrt", {})
        ddt = result.get("ddt", {})
        lines.extend(["", f"## {result['label']}", ""])
        lines.extend([
            f"- Source: `{result['source']}`",
            f"- SHA256: `{result['source_sha256']}`",
            f"- Rows/points: {qc.get('included_points')}/{qc.get('raw_rows')} included; row accounting={qc.get('row_accounting_matches_raw')}",
            f"- Frequency range: {min(result['curves']['frequency_hz']):.6g}–{max(result['curves']['frequency_hz']):.6g} Hz; basis={result['impedance_basis']}",
            f"- Linearity/stationarity evidence: {validity.get('linearity')}/{validity.get('stationarity')}",
            f"- Within-sweep drift screen: {validity.get('single_sweep_drift_screen', {}).get('status')}; "
            f"trend span={md_number(validity.get('single_sweep_drift_screen', {}).get('trend_span_percent'))}%",
            f"- KK: {kk.get('status')}; RMS Re/Im={md_number(kk.get('residual_rms_re_percent'))}/{md_number(kk.get('residual_rms_im_percent'))}%",
            f"- Z-HIT: {zhit.get('status')}; RMS Re/Im={md_number(zhit.get('residual_rms_re_percent'))}/{md_number(zhit.get('residual_rms_im_percent'))}%",
            f"- Series inductance: detected={result.get('inductance_detected', result['inductance_mode'])}; "
            f"used in primary DRT={result.get('inductance_used_in_primary_drt', result['inductance_mode'])}; "
            f"policy={result.get('inductance_policy', 'auto')} "
            f"({inductance.get('negative_neg_zimag_points')}/{inductance.get('highest_frequency_decade_points')} negative -Zimag points in the highest decade)",
            f"- Inductance model competition: selected={inductance_selection.get('selected_model', '—')}; "
            f"estimated L={md_number(1e6 * inductance_selection.get('estimated_series_inductance_h'), 5) if inductance_selection.get('estimated_series_inductance_h') is not None else '—'} µH; "
            f"high-decade R²={md_number(inductance_selection.get('fit_r_squared'), 5)}; "
            f"weighted-quality improvement with L={md_number(inductance_selection.get('quality_improvement_log10_with_L'), 5)} log10",
            f"- Selected λ: {result['drt']['lambda']:.6g} ({result['lambda_selection']['selection_method']}); boundary-sensitive={result['lambda_selection'].get('boundary_sensitive')}",
            f"- mGCV multi-start: start-independent={result['lambda_selection'].get('mgcv_start_independent')}; span={md_number(result['lambda_selection'].get('mgcv_log10_span_decades'), 4)} decades",
            f"- Conditional credible interval: {result['drt']['credible_intervals']} ({result['drt']['credible_samples']} samples)",
            f"- Model-and-reduce: {model_reduce.get('status')}; selected={model_reduce.get('selected_model', 'none')}",
            f"- Loewner RC/RL: {loewner.get('status')}; model order={loewner.get('model_order', '—')}",
            f"- Separate Loewner frequency validation: {result.get('loewner_validation', {}).get('reliability_status')}; "
            f"training/heldout RMS={md_number(result.get('loewner_validation', {}).get('training_rms_percent'))}/"
            f"{md_number(result.get('loewner_validation', {}).get('heldout_rms_percent'))}%. "
            "This reduced model is not the exported full-data matrix-rank model.",
            f"- Conditional peak stress: {result.get('peak_evidence', {}).get('status')}; "
            "persistence is not physical confirmation; see 02_quality/peak_perturbation_evidence.csv.",
            f"- Continuous signed/GDRT: {signed.get('status')}; trigger={signed.get('trigger', {}).get('policy', '—')}",
            f"- DDT: {ddt.get('status')}; best boundary={ddt.get('best_boundary', '—')}",
        ])
        if model_reduce.get("status") == "accepted":
            selected = model_reduce.get("selected", {})
            stability = selected.get("peak_stability", {})
            lines.append(
                f"- Pre/post peak stability: {stability.get('status')}; "
                f"stable={stability.get('stable_match_count')}/{stability.get('before_peak_count')}; "
                f"quality improvement={md_number(selected.get('quality_improvement_log10'), 4)} log10 units"
            )
        elif model_reduce.get("enabled"):
            best = model_reduce.get("best_attempt") or {}
            stability = best.get("peak_stability", {})
            lines.append(
                "- Preprocessing decision: no nuisance subtraction was accepted; raw DRT remains authoritative. "
                f"Best attempt={best.get('model', 'none')}; "
                f"quality improvement={md_number(best.get('quality_improvement_log10'), 4)} log10; "
                f"peak stability={stability.get('status', 'not-assessed')}; "
                f"parameter-at-bound={best.get('parameter_at_bound', '—')}; "
                f"removed norm={md_number(best.get('removed_norm_fraction'), 4)}"
            )
        if qc.get("acquisition_time_available"):
            lines.append(
                f"- Acquisition time: duration={md_number(qc.get('sweep_duration_s'))} s; "
                f"median/max step={md_number(qc.get('time_step_median_s'))}/{md_number(qc.get('time_step_max_s'))} s; "
                f"local step anomalies={qc.get('time_step_anomaly_count')} (audit flag, not automatic drift proof)"
            )
        if secondary.get("status") == "completed":
            frequencies = ", ".join(md_number(value, 4) for value in secondary.get("peak_frequencies_hz", []))
            lines.append(
                f"- Secondary TR-NNLS: {secondary.get('peak_count')} peaks at [{frequencies}] Hz; one-to-one local matches do not imply full algorithm agreement"
            )
        else:
            lines.append(f"- Secondary TR-NNLS: failed ({secondary.get('error', 'unknown error')})")

        if result.get("warnings"):
            lines.extend(["", "Warnings:", ""])
            lines.extend([f"- {warning}" for warning in result["warnings"]])

        lines.extend([
            "",
            f"| Peak | f (Hz) | tau (s) | gamma ({unit}) | Relative height in window | Supported area ({unit}) | Full diagnostic area ({unit}) | Supported-area fraction | Confidence | NNLS one-to-one |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for peak in result["peaks"]:
            supported_fraction = peak.get("area_supported_fraction")
            supported_text = (
                f"{100.0 * supported_fraction:.1f}%" if supported_fraction is not None else "—"
            )
            match_text = "False"
            if peak.get("tr_nnls_match"):
                match_text = (
                    f"{peak.get('tr_nnls_peak_id')} at Δ={md_number(peak.get('tr_nnls_log_distance_decades'), 3)} dec"
                )
                if peak.get("tr_nnls_secondary_split_candidate") or peak.get("tr_nnls_primary_merge_candidate"):
                    match_text += " (split/merge candidate)"
            lines.append(
                f"| {peak['peak_id']} | {peak['frequency_hz']:.6g} | {peak['tau_s']:.6g} | "
                f"{peak['gamma']:.6g} | "
                f"{md_number(100.0 * peak['relative_height'], 4) + '%' if peak.get('relative_height') is not None else '—'} | "
                f"{md_number(peak.get('area_impedance'))} | "
                f"{md_number(peak.get('area_impedance_full_diagnostic'))} | {supported_text} | "
                f"{peak['confidence']} | {match_text} |"
            )

    if failed:
        lines.extend(["", "## Failed or blocked spectra", ""])
        for result in failed:
            lines.append(f"- **{result.get('label')}**: {result.get('error', 'unknown error')}")
    lines.extend([
        "",
        "## Interpretation rules",
        "",
        "- Compare process contributions using the measured-window-supported basin area, not peak height or a fully extrapolated basin.",
        "- Treat peaks outside the measured frequency window as unsupported; their full area is diagnostic only.",
        "- Treat edge-proximal peaks, inductive-overlap peaks, lambda-boundary selections, and split/merge algorithm comparisons as lower confidence.",
        "- Assign electrochemical mechanisms only after controlled SOC, temperature, cycle, replicate, and electrode-side baseline evidence.",
        "- Loewner RL terms and negative signed/GDRT lobes support an inductive mathematical contribution; they do not identify adsorption, wiring, or another mechanism by themselves.",
        "- DDT boundary selection by residual is a numerical comparison. Diffusion coefficients require independently justified geometry and diffusion length.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version",
                        version=Path(__file__).resolve().parents[1].joinpath("VERSION").read_text().strip())
    parser.add_argument("inputs", nargs="+", help="EIS files and/or directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--manifest-mode", choices=["allowlist", "metadata"], default="allowlist")
    parser.add_argument("--resume", action="store_true", help="Resume only an identical fingerprinted run")
    parser.add_argument("--export-only", action="store_true", help="Re-export matching completed checkpoints without numerical solving; requires --resume")
    parser.add_argument("--parser-script", type=Path, default=DEFAULT_PARSER)
    parser.add_argument("--profile-json", type=Path, help="Explicit external numerical profile, tracked by hash")
    parser.add_argument("--trend-plots", choices=["auto", "off"], default="off",
                        help="Voltage ridge/heatmap figures for metadata-matched RC-compatible groups")
    parser.add_argument("--trend-dpi", type=int, default=600)
    parser.add_argument("--imag-convention", choices=["auto", "neg-zimag", "zimag"], default="auto")
    parser.add_argument("--frequency-unit", choices=["auto", "hz", "khz", "mhz"], default="auto")
    parser.add_argument("--impedance-unit", choices=["auto", "ohm", "mohm", "kohm", "ohm-cm2"], default="auto")
    parser.add_argument("--text-columns", help='Explicit JSON role-to-column map; e.g. {"frequency":0,"zreal":1,"zimag":2}')
    parser.add_argument("--text-headerless", action="store_true")
    parser.add_argument("--text-delimiter", choices=["auto", "comma", "tab", "semicolon", "whitespace"], default="auto")
    parser.add_argument("--text-header-row", type=int, help="1-based header line, for explicit text mapping")
    parser.add_argument("--text-frequency-order", choices=["acquisition", "single-spectrum-descending"],
                        default="acquisition", help="Preserve sweep detection, or explicitly sort one unique-frequency text spectrum")
    parser.add_argument("--area-normalize", action="store_true")
    parser.add_argument(
        "--reference-profile",
        choices=["none", *REFERENCE_PROFILES],
        default="none",
        help="Apply a locked compatibility profile calibrated against an external reference DRT set",
    )
    parser.add_argument(
        "--inductance-policy",
        choices=["adaptive", "always", "off"],
        default="adaptive",
        help=(
            "Adaptively compare RC-DRT with series-L+RC-DRT, force series L on, or explicitly "
            "disable it. The off mode is intended only for declared legacy compatibility."
        ),
    )
    parser.add_argument("--lambda-policy", choices=["consensus", "lcurve-grid", "package-mgcv", "fixed"], default="consensus")
    parser.add_argument("--lambda-min", type=float, default=1e-7)
    parser.add_argument("--lambda-max", type=float, default=1e-1)
    parser.add_argument("--lambda-points", type=int, default=13)
    parser.add_argument("--fixed-lambda", type=float, default=1e-3)
    parser.add_argument("--impedance-scaling", choices=["median", "off"], default="median",
                        help="Uniform numerical conditioning of primary RBF/NNLS; off is an explicit legacy comparison")
    parser.add_argument("--derivative-order", type=int, choices=[1, 2], default=1)
    parser.add_argument("--rbf-type", default="gaussian")
    parser.add_argument("--rbf-shape", default="fwhm")
    parser.add_argument("--shape-coeff", type=float, default=0.5)
    parser.add_argument("--edge-decades", type=float, default=0.7)
    parser.add_argument("--peak-match-decades", type=float, default=0.5)
    parser.add_argument("--peak-threshold", type=float, default=0.02)
    parser.add_argument("--no-credible-intervals", action="store_true")
    parser.add_argument("--credible-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-procs", type=int, default=1)
    parser.add_argument("--solver-timeout", type=int, default=180)
    parser.add_argument("--spectrum-timeout", type=int, default=600)
    parser.add_argument("--render-timeout", type=int, default=180, help="Separate recoverable plotting budget per spectrum")
    parser.add_argument("--no-peak-evidence", action="store_true", help="Explicitly disable conditional noise/resolution perturbation audit")
    parser.add_argument(
        "--kk-max-points",
        type=int,
        default=0,
        help=(
            "Cap only the exploratory KK screening grid by deterministic equal-index "
            "coverage; 0 uses every point. Raw data and DRT remain full resolution."
        ),
    )
    parser.add_argument("--kk-warn-percent", type=float, default=0.5)
    parser.add_argument("--kk-block-percent", type=float, default=1.0)
    parser.add_argument(
        "--model-reduce",
        action="store_true",
        help="Optionally fit and remove eligible L/C/Warburg nuisance branches after quality and peak-stability checks",
    )
    parser.add_argument("--no-loewner", action="store_true", help="Disable the Loewner RC/RL diagnostic")
    parser.add_argument("--predictive-fit", choices=["auto", "off"], default="auto",
                        help="On difficult spectra compare positive/signed RC with matched R/L/C and interior-frequency CV; does not replace primary curves")
    parser.add_argument(
        "--signed-gdrt",
        choices=["off", "auto", "always"],
        default="auto",
        help="Run continuous signed/GDRT never, on an inductive trigger, or for every spectrum",
    )
    parser.add_argument(
        "--ddt",
        choices=["off", "auto", "always"],
        default="auto",
        help="Run multi-boundary DDT never, on a low-frequency diffusion trigger, or for every spectrum",
    )
    parser.add_argument("--model-reduce-min-improvement", type=float, default=0.02)
    parser.add_argument("--preprocess-peak-position-decades", type=float, default=0.35)
    parser.add_argument("--preprocess-peak-area-fraction", type=float, default=0.50)
    parser.add_argument("--repeat-warn-percent", type=float, default=2.0)
    parser.add_argument("--linearity-warn-percent", type=float, default=2.0)
    parser.add_argument("--stationarity-warn-percent", type=float, default=2.0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def apply_reference_profile(args: argparse.Namespace) -> None:
    if args.profile_json:
        path = args.profile_json.expanduser().resolve()
        profile = json.loads(path.read_text(encoding="utf-8"))
        allowed = {"name", "scope", "lambda_policy", "fixed_lambda", "derivative_order",
                   "rbf_type", "rbf_shape", "shape_coeff", "inductance_policy"}
        if not isinstance(profile, dict) or set(profile) - allowed:
            raise ValueError("External profile contains unknown fields")
        required = allowed - {"scope"}
        if not required <= set(profile) or not isinstance(profile["name"], str) or not profile["name"].strip():
            raise ValueError("External profile requires a name and all numerical options")
        if profile["inductance_policy"] not in {"adaptive", "always", "off"}:
            raise ValueError("Invalid profile inductance policy")
        if profile["lambda_policy"] not in {"fixed", "consensus", "lcurve-grid", "package-mgcv"}:
            raise ValueError("Invalid profile lambda policy")
        if profile["derivative_order"] not in (1, 2):
            raise ValueError("Invalid profile derivative order")
        for key in ("fixed_lambda", "shape_coeff"):
            if not isinstance(profile[key], (int, float)) or not math.isfinite(profile[key]) or profile[key] <= 0:
                raise ValueError("External profile numeric parameters must be finite and positive")
        for key in allowed - {"name", "scope"}:
            setattr(args, key, profile[key])
        args.reference_profile = profile["name"]
        args.reference_profile_parameters = profile
        args.profile_sha256 = sha256_file(path)
        return
    if args.reference_profile == "none":
        args.reference_profile_parameters = None
        return
    profile = dict(REFERENCE_PROFILES[args.reference_profile])
    for key in (
        "lambda_policy", "fixed_lambda", "derivative_order", "rbf_type", "rbf_shape", "shape_coeff"
    ):
        setattr(args, key, profile[key])
    args.inductance_policy = profile["inductance_policy"]
    args.reference_profile_parameters = profile


def validate_args(args: argparse.Namespace) -> None:
    for key, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{key.replace('_', '-')} must be finite")
    if args.export_only and not args.resume:
        raise ValueError("--export-only requires --resume")
    if args.num_procs < 1:
        raise ValueError("--num-procs must be positive")
    if not (0.0 < args.lambda_min < args.lambda_max):
        raise ValueError("Expected 0 < --lambda-min < --lambda-max")
    if args.lambda_points < 7:
        raise ValueError("--lambda-points must be at least 7")
    if not (args.lambda_min <= args.fixed_lambda <= args.lambda_max):
        raise ValueError("--fixed-lambda must lie within the lambda grid")
    if args.credible_samples < 1000 and not args.no_credible_intervals:
        raise ValueError("pyimpspec requires at least 1000 credible-interval samples")
    if args.spectrum_timeout <= 0 or args.solver_timeout <= 0 or getattr(args, "render_timeout", 180) <= 0:
        raise ValueError("Timeouts must be positive")
    if args.kk_max_points != 0 and args.kk_max_points < 30:
        raise ValueError("--kk-max-points must be 0 or at least 30")
    if not (0.0 <= args.peak_threshold < 1.0):
        raise ValueError("--peak-threshold must be in [0, 1)")
    if args.model_reduce_min_improvement < 0.0:
        raise ValueError("--model-reduce-min-improvement must be non-negative")
    if args.preprocess_peak_position_decades <= 0.0:
        raise ValueError("--preprocess-peak-position-decades must be positive")
    if not (0.0 <= args.preprocess_peak_area_fraction <= 5.0):
        raise ValueError("--preprocess-peak-area-fraction must be in [0, 5]")
    if min(args.repeat_warn_percent, args.linearity_warn_percent, args.stationarity_warn_percent) <= 0.0:
        raise ValueError("Repeat, linearity, and stationarity thresholds must be positive")


PARSER_FIELDS = ("imag_convention", "frequency_unit", "impedance_unit", "text_columns",
                 "text_headerless", "text_delimiter", "text_header_row", "text_frequency_order")


def source_parser_options(defaults, metadata):
    """Source-level mapping/units are provenance-bearing manifest overrides."""
    values = vars(defaults).copy()
    for key in PARSER_FIELDS:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        if key == "text_headerless":
            if str(value).lower() not in {"true", "false", "1", "0"}:
                raise ValueError("Manifest text_headerless must be true/false")
            value = str(value).lower() in {"true", "1"}
        elif key == "text_header_row":
            value = int(value)
            if value < 1:
                raise ValueError("text_header_row must be positive")
        elif key == "text_frequency_order":
            if value not in {"acquisition", "single-spectrum-descending"}:
                raise ValueError("Invalid manifest text_frequency_order")
            values["frequency_order"] = value
            continue
        values[key] = value
    return argparse.Namespace(**values)


def validate_spectrum_parser_options(source_options, spectrum_metadata):
    """Parsing precedes spectrum selection: per-spectrum settings cannot reinterpret it."""
    proposed = source_parser_options(source_options, spectrum_metadata)
    for key in PARSER_FIELDS:
        if spectrum_metadata.get(key) in (None, ""):
            continue
        attr = "frequency_order" if key == "text_frequency_order" else key
        if getattr(proposed, attr, None) != getattr(source_options, attr, None):
            raise ValueError(f"Per-spectrum parser option {key} conflicts with source parsing; "
                             "declare parser settings on a blank spectrum_id source row")


def main() -> int:
    args = parse_args()
    apply_reference_profile(args)
    validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    from result_contract import (inventory_inputs, execution_contract, initialize_run,
                                 row_selected, common_fields, export_bundle, identities, write_table, numerical_digest)
    result_dir, figure_dir = output_dir / "results", output_dir / "figures"
    parser_module = load_parser_module(args.parser_script.expanduser().resolve())
    sources = discover_inputs(args.inputs, output_dir)
    manifest_exact, manifest_by_path = load_manifest(args.manifest)
    sources, inventory = inventory_inputs(sources, manifest_exact, manifest_by_path,
                                         manifest=args.manifest, manifest_mode=args.manifest_mode)
    if not sources:
        raise ValueError("Input selection contains no EIS candidates; inspect the manifest and file roles")
    contract = execution_contract(args, inventory, Path(__file__).parent)
    initialize_run(output_dir, contract, args.resume)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    parser_args = argparse.Namespace(
        imag_convention=args.imag_convention,
        frequency_unit=args.frequency_unit,
        impedance_unit=args.impedance_unit,
        text_columns=args.text_columns, text_headerless=args.text_headerless,
        text_delimiter=args.text_delimiter, text_header_row=args.text_header_row,
        frequency_order=args.text_frequency_order,
    )
    source_audits: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    parse_failures: list[dict[str, Any]] = []
    used_slugs: set[str] = set()
    expected_hashes = {row["source"]: row["source_sha256"] for row in inventory if row["selected"]}
    for source in sources:
        pending = []
        try:
            before_hash = sha256_file(source)
            if before_hash != expected_hashes[str(source)]:
                raise ValueError("Input changed after inventory; use a stable snapshot and a fresh run")
            label = manifest_by_path.get(str(source), {}).get("label") or source.stem
            source_parser_args = source_parser_options(parser_args, manifest_by_path.get(str(source), {}))
            if source.suffix.lower() == ".irf":
                spectra, audit = parser_module.read_irf(source, label, source_parser_args.imag_convention)
            elif source.suffix.lower() in TEXT_SUFFIXES:
                spectra, audit = parser_module.read_text(source, label, source_parser_args)
            else:
                spectra, audit = read_pyimpspec_source(source, label, source_parser_args, parser_module)
            source_audits.append(audit)
            source_hash = audit.get("sha256") or sha256_file(source)
            if source_hash != before_hash or sha256_file(source) != before_hash:
                raise ValueError("Input changed while parsing; this source was not accepted")
            requested_ids = {key for (path, key), row in manifest_exact.items()
                             if path == str(source) and row_selected(row)}
            missing_ids = requested_ids - {str(spec.spectrum_id) for spec in spectra}
            if missing_ids:
                raise ValueError(f"Manifest spectrum IDs were not parsed from source: {sorted(missing_ids)}")
            audit["parsed_spectrum_count"] = len(spectra)
            audit["selected_spectrum_count"] = 0
            audit["spectrum_exclusions"] = []
            for spec in spectra:
                qc = parser_module.qc_for(spec)
                metadata = manifest_metadata(source, spec.spectrum_id, manifest_exact, manifest_by_path)
                if args.manifest and args.manifest_mode == "allowlist" and not metadata:
                    audit["spectrum_exclusions"].append({"spectrum_id": spec.spectrum_id, "reason": "not-selected-by-manifest"})
                    continue
                if not row_selected(metadata):
                    audit["spectrum_exclusions"].append({"spectrum_id": spec.spectrum_id,
                        "reason": metadata.get("exclusion_reason") or metadata.get("input_role") or "manifest-excluded"})
                    continue
                validate_spectrum_parser_options(source_parser_args, manifest_exact.get((str(source), str(spec.spectrum_id)), {}))
                payload = prepare_payload(spec, source_hash, qc, metadata, args.area_normalize)
                payload["run_fingerprint"] = contract["run_fingerprint"]
                base = safe_slug(payload["spectrum_id"])[:70] + "_" + payload["spectrum_uid"][:12]
                slug = base
                counter = 2
                while slug in used_slugs:
                    slug = f"{base}_{counter}"
                    counter += 1
                used_slugs.add(slug)
                payload["result_slug"] = slug
                pending.append(payload)
            payloads.extend(pending)
            audit["selected_spectrum_count"] = len(pending)
        except Exception as exc:
            failed_audit = getattr(exc, "audit", None)
            if isinstance(failed_audit, dict) and not any(a is failed_audit for a in source_audits):
                source_audits.append(failed_audit)
            for audit in source_audits:
                if audit.get("source") == str(source):
                    audit.update(selected_spectrum_count=0, source_preparation_failed=True)
            parse_failures.append({
                **identities(source, expected_hashes[str(source)], "__parse_failure__"),
                "status": "failed",
                "spectrum_id": None,
                "label": source.stem,
                "source": str(source),
                "source_sha256": expected_hashes[str(source)],
                "error": f"ParseError: {type(exc).__name__}: {exc}",
            })
    if not payloads:
        run = {
            **contract,
            "status": "failed", "export_status": "pending", "trend_figure_status": "not-assessable",
            "code_sha256": contract["code_sha256"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_count": len(sources),
            "spectrum_count": 0,
            "source_audits": source_audits,
            "parse_failures": parse_failures,
            "package_versions": package_versions(),
            "parameters": {**vars(args), "output_dir": str(output_dir), "manifest": str(args.manifest) if args.manifest else None},
        }
        write_json(output_dir / "run_manifest.json", run)
        (output_dir / "analysis_report.md").write_text(build_report(run, parse_failures), encoding="utf-8")
        try:
            export_bundle(output_dir, parse_failures, [], inventory, run, {})
            run["export_status"] = "completed"
        except Exception as exc:
            run["export_status"] = "failed"
            run["export_error"] = f"{type(exc).__name__}: {exc}"
            write_json(output_dir / "run_manifest.json", run)
            raise
        write_json(output_dir / "run_manifest.json", run)
        write_json(output_dir / "00_overview/run_manifest.json", run)
        from derived_outputs import refresh_owned_file
        refresh_owned_file(output_dir, "export", "00_overview/run_manifest.json")
        return 2

    options = WorkerOptions(
        lambda_policy=args.lambda_policy,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_points=args.lambda_points,
        fixed_lambda=args.fixed_lambda,
        derivative_order=args.derivative_order,
        rbf_type=args.rbf_type,
        rbf_shape=args.rbf_shape,
        shape_coeff=args.shape_coeff,
        edge_decades=args.edge_decades,
        peak_match_decades=args.peak_match_decades,
        peak_threshold=args.peak_threshold,
        credible_intervals=not args.no_credible_intervals,
        credible_samples=args.credible_samples,
        seed=args.seed,
        num_procs=args.num_procs,
        solver_timeout=args.solver_timeout,
        kk_max_points=args.kk_max_points,
        strict=args.strict,
        kk_warn_percent=args.kk_warn_percent,
        kk_block_percent=args.kk_block_percent,
        model_reduce=args.model_reduce,
        loewner=not args.no_loewner,
        signed_gdrt=args.signed_gdrt,
        ddt=args.ddt,
        model_reduce_min_improvement=args.model_reduce_min_improvement,
        preprocess_peak_position_decades=args.preprocess_peak_position_decades,
        preprocess_peak_area_fraction=args.preprocess_peak_area_fraction,
        inductance_policy=args.inductance_policy,
        impedance_scaling=args.impedance_scaling,
        peak_evidence=not args.no_peak_evidence,
        predictive_fit=args.predictive_fit,
    )
    results: list[dict[str, Any]] = list(parse_failures)
    for payload in payloads:
        result_path = result_dir / f"{payload['result_slug']}.json"
        cached = json.loads(result_path.read_text(encoding="utf-8")) if args.resume and result_path.is_file() else {}
        if cached and cached.get("run_fingerprint") != contract["run_fingerprint"]:
            raise ValueError(f"Checkpoint fingerprint mismatch: {result_path}")
        if cached.get("status") == "completed" and cached.get("numerical_sha256") != numerical_digest(cached):
            raise ValueError(f"Checkpoint numerical content changed: {result_path}")
        if args.export_only and cached.get("status") != "completed":
            raise ValueError(f"No completed checkpoint for export-only: {payload['spectrum_id']}")
        if cached.get("status") == "completed":
            result = cached
            if args.export_only or result.get("figure_status") != "completed" or not all(
                Path(p).is_file() for p in result.get("figures", []) + result.get("advanced_figures", [])):
                result = run_render_isolated(result, figure_dir, result_path, args.render_timeout)
        else:
            result = run_isolated(payload, options, figure_dir, result_path, args.spectrum_timeout, args.render_timeout)
        for key in ("spectrum_uid", "source_uid", "run_fingerprint", "result_slug", "metadata", "impedance_basis"):
            result.setdefault(key, payload.get(key))
        write_json(result_path, result)
        results.append(result)
        print(f"[{result.get('status')}] {result.get('label')}")

    from advanced_drt import assess_batch_validity

    batch_validation = assess_batch_validity(
        results,
        repeat_warn_percent=args.repeat_warn_percent,
        linearity_warn_percent=args.linearity_warn_percent,
        stationarity_warn_percent=args.stationarity_warn_percent,
    )
    from control_validation import classify_results
    batch_validation = classify_results(results, batch_validation)
    overall_verdict = batch_validation["overall_verdict"]
    for result in results:
        if result.get("result_slug"):
            write_json(result_dir / (result["result_slug"] + ".json"), result)
    write_json(output_dir / "batch_validation.json", batch_validation)

    try:
        overlay = plot_batch_overlay(results, figure_dir)
        overlay_error = None
    except Exception as exc:
        overlay, overlay_error = [], f"{type(exc).__name__}: {exc}"
    summary_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    model_reduce_rows: list[dict[str, Any]] = []
    signed_curve_rows: list[dict[str, Any]] = []
    loewner_rows: list[dict[str, Any]] = []
    ddt_curve_rows: list[dict[str, Any]] = []
    for result in results:
        identity = common_fields(result)
        summary = {
            **identity,
            "claim_level": result.get("claim_level"),
            "decision_basis": result.get("decision_basis"),
            "kk_source_points": result.get("kk", {}).get("source_point_count"),
            "kk_screen_points": result.get("kk", {}).get("screen_point_count"),
            "spectrum_id": result.get("spectrum_id"),
            "label": result.get("label"),
            "source": result.get("source"),
            "status": result.get("status"),
            "error": result.get("error"),
        }
        if result.get("status") == "completed":
            in_window_count = sum(
                peak.get("confidence") != "unsupported-outside-window" for peak in result["peaks"]
            )
            robust_count = sum(
                peak.get("confidence") == "robust-to-lambda" for peak in result["peaks"]
            )
            boundary_or_inductive_count = sum(
                peak.get("confidence") in {"boundary-sensitive", "inductive-overlap"}
                for peak in result["peaks"]
            )
            unsupported_count = sum(
                peak.get("confidence") == "unsupported-outside-window" for peak in result["peaks"]
            )
            summary.update({
                "reconstruction_branch": "primary_rc",
                "distribution_branch": "primary_rc",
                "frequency_min_Hz": min(result["curves"]["frequency_hz"]),
                "frequency_max_Hz": max(result["curves"]["frequency_hz"]),
                "points": len(result["curves"]["frequency_hz"]),
                "impedance_basis": result["impedance_basis"],
                "kk_status": result["kk"].get("status"),
                "kk_rms_re_percent": result["kk"].get("residual_rms_re_percent"),
                "kk_rms_im_percent": result["kk"].get("residual_rms_im_percent"),
                "zhit_status": result["zhit"].get("status"),
                "zhit_rms_re_percent": result["zhit"].get("residual_rms_re_percent"),
                "zhit_rms_im_percent": result["zhit"].get("residual_rms_im_percent"),
                "linearity_status": result["validity"].get("linearity"),
                "stationarity_status": result["validity"].get("stationarity"),
                "sweep_drift_status": result["validity"].get("single_sweep_drift_screen", {}).get("status"),
                "sweep_drift_trend_percent": result["validity"].get("single_sweep_drift_screen", {}).get("trend_span_percent"),
                "lambda": result["drt"]["lambda"],
                "lambda_method": result["lambda_selection"]["selection_method"],
                "lambda_boundary_sensitive": result["lambda_selection"].get("boundary_sensitive"),
                "drt_rms_re_percent": result["drt"]["residual_rms_re_percent"],
                "drt_rms_im_percent": result["drt"]["residual_rms_im_percent"],
                "credible_intervals": result["drt"]["credible_intervals"],
                "credible_samples": result["drt"]["credible_samples"],
                "peak_count": len(result["peaks"]),
                "in_window_peak_count": in_window_count,
                "robust_peak_count": robust_count,
                "boundary_or_inductive_peak_count": boundary_or_inductive_count,
                "unsupported_peak_count": unsupported_count,
                "inductance_mode": result["inductance_mode"],
                "inductance_detected": result.get("inductance_detected"),
                "inductance_used_in_primary_drt": result.get("inductance_used_in_primary_drt"),
                "inductance_policy": result.get("inductance_policy"),
                "inductance_selected_model": result.get("inductance_model_selection", {}).get("selected_model"),
                "estimated_series_inductance_uH": (
                    1e6 * result.get("inductance_model_selection", {}).get("estimated_series_inductance_h")
                    if result.get("inductance_model_selection", {}).get("estimated_series_inductance_h") is not None
                    else None
                ),
                "series_inductance_fit_r_squared": result.get("inductance_model_selection", {}).get("fit_r_squared"),
                "inductance_quality_improvement_log10": result.get("inductance_model_selection", {}).get("quality_improvement_log10_with_L"),
                "residual_nonseries_inductance_detected": result.get("nonseries_inductance_evidence", {}).get("detected"),
                "sweep_duration_s": result["qc"].get("sweep_duration_s"),
                "time_step_anomaly_count": result["qc"].get("time_step_anomaly_count"),
                "model_reduce_status": result.get("model_reduce", {}).get("status"),
                "model_reduce_selected": result.get("model_reduce", {}).get("selected_model"),
                "loewner_status": result.get("loewner_rc_rl", {}).get("status"),
                "loewner_model_order": result.get("loewner_rc_rl", {}).get("model_order"),
                "signed_gdrt_status": result.get("signed_gdrt", {}).get("status"),
                "ddt_status": result.get("ddt", {}).get("status"),
                "ddt_best_boundary": result.get("ddt", {}).get("best_boundary"),
            })
            for peak in result["peaks"]:
                peak_rows.append({
                    **identity,
                    "label": result["label"],
                    "impedance_basis": result["impedance_basis"],
                    **peak,
                    "distribution_branch": "primary_rc",
                })
            tau_values = result["curves"]["tau_s"]
            lower_values = result["curves"].get("credible_lower", [])
            upper_values = result["curves"].get("credible_upper", [])
            if len(lower_values) != len(tau_values):
                lower_values = [None] * len(tau_values)
            if len(upper_values) != len(tau_values):
                upper_values = [None] * len(tau_values)
            for values in zip(
                tau_values,
                result["curves"]["drt_frequency_hz"],
                result["curves"]["gamma"],
                lower_values,
                upper_values,
            ):
                curve_rows.append({
                    **identity,
                    "label": result["label"],
                    "impedance_basis": result["impedance_basis"],
                    "distribution_branch": "primary_rc",
                    "tau_s": values[0],
                    "frequency_Hz": values[1],
                    "gamma": values[2],
                    "credible_lower": values[3],
                    "credible_upper": values[4],
                })
            model_reduce = result.get("model_reduce", {})
            for candidate in model_reduce.get("candidates", []):
                quality = candidate.get("quality", {})
                stability = candidate.get("peak_stability", {})
                model_reduce_rows.append({
                    **identity,
                    "label": result["label"],
                    "model": candidate.get("model"),
                    "status": candidate.get("status"),
                    "accepted": candidate.get("accepted", False),
                    "score": candidate.get("score"),
                    "q_log10": quality.get("q_log10"),
                    "relative_complex_rms_percent": quality.get("relative_complex_rms_percent"),
                    "quality_improvement_log10": candidate.get("quality_improvement_log10"),
                    "removed_norm_fraction": candidate.get("removed_norm_fraction"),
                    "parameter_at_bound": candidate.get("parameter_at_bound"),
                    "peak_stability": stability.get("status"),
                    "stable_peak_fraction": stability.get("stable_fraction_of_before"),
                    "error": candidate.get("error"),
                })
            signed = result.get("signed_gdrt", {})
            if signed.get("status") == "completed":
                for tau_value, frequency_value, gamma_value in zip(
                    signed.get("tau_s", []), signed.get("frequency_hz", []), signed.get("gamma", [])
                ):
                    signed_curve_rows.append({
                        **identity, "label": result["label"],
                        "distribution_branch": "signed_gdrt",
                        "tau_s": tau_value, "frequency_Hz": frequency_value, "signed_gamma": gamma_value,
                    })
            loewner = result.get("loewner_rc_rl", {})
            if loewner.get("status") == "completed":
                for branch, tau_key, gamma_key in (
                    ("RC", "tau_rc_s", "gamma_rc"), ("RL", "tau_rl_s", "gamma_rl")
                ):
                    for tau_value, gamma_value in zip(loewner.get(tau_key, []), loewner.get(gamma_key, [])):
                        loewner_rows.append({
                            **identity, "label": result["label"],
                            "distribution_branch": "loewner_rc_rl",
                            "model_identity": "legacy-full-data-matrix-rank-model",
                            "model_order": loewner.get("model_order"),
                            "heldout_validated": False,
                            "branch": branch, "tau_s": tau_value,
                            "frequency_Hz": 1.0 / (2.0 * math.pi * tau_value), "gamma": gamma_value,
                        })
            ddt = result.get("ddt", {})
            for candidate in ddt.get("candidates", []):
                if candidate.get("status") != "completed":
                    continue
                for tau_value, frequency_value, gamma_value in zip(
                    candidate.get("tau_s", []), candidate.get("frequency_hz", []), candidate.get("gamma", [])
                ):
                    ddt_curve_rows.append({
                        **identity, "label": result["label"],
                        "distribution_branch": "ddt:" + str(candidate.get("boundary")),
                        "boundary": candidate.get("boundary"),
                        "selected_best": candidate.get("boundary") == ddt.get("best_boundary"),
                        "residual_rms_percent": candidate.get("residual_rms_percent"),
                        "tau_s": tau_value, "frequency_Hz": frequency_value, "gamma": gamma_value,
                    })
        summary_rows.append(summary)

    summary_fields = [
        "spectrum_id", "label", "source", "status", "error", "frequency_min_Hz",
        "frequency_max_Hz", "points", "impedance_basis", "kk_status",
        "kk_rms_re_percent", "kk_rms_im_percent", "zhit_status",
        "zhit_rms_re_percent", "zhit_rms_im_percent", "linearity_status",
        "stationarity_status", "sweep_drift_status", "sweep_drift_trend_percent",
        "lambda", "lambda_method", "lambda_boundary_sensitive",
        "drt_rms_re_percent", "drt_rms_im_percent", "credible_intervals",
        "credible_samples", "peak_count", "in_window_peak_count", "robust_peak_count",
        "boundary_or_inductive_peak_count", "unsupported_peak_count", "inductance_mode",
        "inductance_detected", "inductance_used_in_primary_drt", "inductance_policy",
        "inductance_selected_model", "estimated_series_inductance_uH",
        "series_inductance_fit_r_squared", "inductance_quality_improvement_log10",
        "residual_nonseries_inductance_detected",
        "sweep_duration_s", "time_step_anomaly_count",
        "model_reduce_status", "model_reduce_selected", "loewner_status",
        "loewner_model_order", "signed_gdrt_status", "ddt_status", "ddt_best_boundary",
    ]
    peak_fields = [
        "spectrum_id", "label", "impedance_basis", "peak_id", "frequency_hz", "tau_s", "gamma",
        "area_impedance", "in_window_area_impedance", "area_impedance_full_diagnostic",
        "area_supported", "area_supported_fraction", "basin_window_fraction",
        "relative_height", "relative_height_global", "confidence", "lambda_persistence",
        "lambda_cases", "position_span_decades", "area_cv", "edge_distance_decades",
        "inductive_overlap", "tr_nnls_match", "tr_nnls_peak_id", "tr_nnls_frequency_hz",
        "tr_nnls_log_distance_decades", "tr_nnls_candidate_count",
        "tr_nnls_secondary_split_candidate", "tr_nnls_primary_merge_candidate",
        "mechanism_assignment",
    ]
    write_csv(output_dir / "batch_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "peaks.csv", peak_rows, peak_fields)
    if curve_rows:
        write_csv(
            output_dir / "drt_curves.csv",
            curve_rows,
            ["spectrum_id", "label", "impedance_basis", "tau_s", "frequency_Hz", "gamma", "credible_lower", "credible_upper"],
        )
    if model_reduce_rows:
        write_csv(
            output_dir / "model_reduce_candidates.csv",
            model_reduce_rows,
            ["spectrum_id", "label", "model", "status", "accepted", "score", "q_log10",
             "relative_complex_rms_percent", "quality_improvement_log10", "removed_norm_fraction",
             "parameter_at_bound", "peak_stability", "stable_peak_fraction", "error"],
        )
    if signed_curve_rows:
        write_csv(output_dir / "signed_gdrt_curves.csv", signed_curve_rows,
                  ["spectrum_id", "label", "tau_s", "frequency_Hz", "signed_gamma"])
    if loewner_rows:
        write_csv(output_dir / "loewner_rc_rl.csv", loewner_rows,
                  ["spectrum_id", "label", "branch", "tau_s", "frequency_Hz", "gamma"])
    if ddt_curve_rows:
        write_csv(output_dir / "ddt_curves.csv", ddt_curve_rows,
                  ["spectrum_id", "label", "boundary", "selected_best", "residual_rms_percent",
                   "tau_s", "frequency_Hz", "gamma"])

    for name, rows in (
        ("repeat_assessment.csv", batch_validation.get("repeat_assessments", [])),
        ("linearity_assessment.csv", batch_validation.get("linearity_assessments", [])),
        ("stationarity_assessment.csv", batch_validation.get("stationarity_assessments", [])),
    ):
        flat_rows = []
        for row in rows:
            flat_rows.append({
                "group": row.get("group"),
                "status": row.get("status"),
                "member_count": row.get("member_count"),
                "members": " | ".join(row.get("members", [])),
                "levels": " | ".join(str(v) for v in row.get("amplitudes_mv", row.get("rest_times_s", []))),
                "maximum_or_last_rms_percent": row.get("maximum_pair_rms_percent", row.get("last_pair_rms_percent")),
                "threshold_percent": row.get("threshold_percent"),
            })
        if flat_rows:
            write_csv(output_dir / name, flat_rows,
                      ["group", "status", "member_count", "members", "levels",
                       "maximum_or_last_rms_percent", "threshold_percent"])
    run = {
        **contract,
        "schema_version": 5,
        "status": "completed" if all(r.get("status") == "completed" for r in results) else "partial",
        "export_status": "pending",
        "input_inventory": inventory,
        "overlay_error": overlay_error,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "parser_script": str(args.parser_script.expanduser().resolve()),
        "parser_script_sha256": sha256_file(args.parser_script.expanduser().resolve()),
        "package_versions": package_versions(),
        "parameters": {**vars(args), "output_dir": str(output_dir), "manifest": str(args.manifest) if args.manifest else None},
        "source_count": len(sources),
        "spectrum_count": len(payloads),
        "source_audits": source_audits,
        "parse_failures": parse_failures,
        "result_files": [str(path.resolve()) for path in sorted(result_dir.glob("*.json"))],
        "overlay_figures": overlay,
        "batch_validation_file": str((output_dir / "batch_validation.json").resolve()),
        "overall_acceptance_verdict": overall_verdict,
        "claim_boundary": "Automated DRT output is screening/quantification evidence, not automatic mechanism proof",
    }
    write_json(output_dir / "run_manifest.json", run)
    (output_dir / "analysis_report.md").write_text(
        build_report(run, results, batch_validation), encoding="utf-8"
    )
    try:
        export_bundle(output_dir, results, payloads, inventory, run, batch_validation)
        run["export_status"] = "completed"
    except Exception as exc:
        run["export_status"] = "failed"
        run["export_error"] = f"{type(exc).__name__}: {exc}"
        write_json(output_dir / "run_manifest.json", run)
        raise
    if args.trend_plots == "auto":
        try:
            from plot_drt_trends import render_trends
            trend = render_trends(output_dir, dpi=args.trend_dpi)
            run["trend_figure_status"] = trend["status"]
        except Exception as exc:
            run["trend_figure_status"] = "failed"
            run["trend_figure_error"] = f"{type(exc).__name__}: {exc}"
    else:
        run["trend_figure_status"] = "disabled"
    write_json(output_dir / "run_manifest.json", run)
    write_json(output_dir / "00_overview/run_manifest.json", run)
    from derived_outputs import refresh_owned_file
    refresh_owned_file(output_dir, "export", "00_overview/run_manifest.json")
    print((output_dir / "00_overview" / "README.md").resolve())
    failures = [result for result in results if result.get("status") != "completed"]
    return 0 if not failures or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())

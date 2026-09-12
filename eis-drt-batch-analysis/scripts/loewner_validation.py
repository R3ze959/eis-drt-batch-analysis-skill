"""Bounded frequency holdout and actual-pole audit for the pinned Loewner backend.

The backend's DRT conversion takes abs(-1/pole). This module inspects poles
BEFORE that conversion; an oscillatory rational response is not an RC peak.
No raw points are removed from the saved spectrum or primary reconstruction.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.linalg import block_diag, eig, svd


LOCAL_PROJECTION = "dimensionless-local-pencil-svd-all-input-points"
LEGACY_PROJECTION = "pyimpspec-5.1.3-original-projection"
REFERENCE_QUANTILES = (.25, .5, .75)


def _loewner_matrices_all_points(s, impedance):
    """Conjugate-completed rectangular pencil, without dropping odd endpoints.

    The original backend makes its square pencil even by dropping one input.
    A rectangular left/right partition instead uses every distinct frequency.
    """
    def paired(values):
        return np.column_stack((values, np.conj(values))).reshape(-1)
    s, z = np.asarray(s, complex), np.asarray(impedance, complex)
    right, left = paired(s[::2]), paired(s[1::2])
    w, v = paired(z[::2]), paired(z[1::2])
    denominator = left[:, None] - right[None, :]
    L = (v[:, None] - w[None, :]) / denominator
    Ls = (left[:, None] * v[:, None] - right[None, :] * w[None, :]) / denominator
    block = np.array([[1., 1j], [1., -1j]]) / np.sqrt(2.)
    pr = block_diag(*[block for _ in range(len(right)//2)])
    pl = block_diag(*[block for _ in range(len(left)//2)])
    return ((pl.conj().T @ L @ pr).real, (pl.conj().T @ Ls @ pr).real,
            (pl.conj().T @ v).real, ((w.T @ pr).T).real)


def _reduce_svd_right_vectors(L, Ls, V, W, order):
    """Petrov-Galerkin reduction from BOTH members of the Loewner pencil.

    Y is the dominant left subspace of [L Ls]; X the dominant right subspace
    of [L; Ls]. Using L alone loses constant/series-L terms in its nullspace.
    scipy.linalg.svd returns M = U @ diag(s) @ Vh, so X=Vh.conj().T[:, :k],
    not Vh[:, :k]. C is W.T @ X (a transfer-function row, not W.conj().T).
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
    Pencil compression: https://pmc.ncbi.nlm.nih.gov/articles/PMC11907484/
    """
    U, singular_values, _ = svd(np.hstack((L, Ls)), full_matrices=False)
    _, right_values, Vh = svd(np.vstack((L, Ls)), full_matrices=False)
    if order < 1 or order > min(len(singular_values), len(right_values)):
        raise ValueError("Loewner order exceeds available left/right dimensions")
    Y, X = U[:, :order], Vh.conj().T[:, :order]
    Yh = Y.conj().T
    return (-Yh @ L @ X, -Yh @ Ls @ X, Yh @ V, W.T @ X, singular_values)


def _descriptor_poles_residues(E, A, B, C):
    """Finite simple-pole residues without assuming E is invertible.

    For left/right generalized eigenvectors u,v, residue=(Cv)(u.H B)/(u.H E v).
    Descriptor poles at infinity can encode the constant/series-L polynomial
    part; they are retained as nonfinite diagnostics, never finite RC peaks.
    """
    homogeneous, left, right = eig(A, E, left=True, right=True, homogeneous_eigvals=True)
    alpha, beta = homogeneous
    poles = np.full(alpha.shape, complex(np.nan, np.nan))
    finite = beta != 0
    poles[finite] = alpha[finite] / beta[finite]
    poles[(beta == 0) & (alpha != 0)] = complex(np.inf, 0.)
    residues = np.full(poles.shape, complex(np.nan, np.nan))
    scale = max(float(np.linalg.norm(E)), np.finfo(float).tiny)
    for i, pole in enumerate(poles):
        denominator = left[:, i].conj() @ E @ right[:, i]
        if np.isfinite(pole) and np.isfinite(denominator) and abs(denominator) > np.finfo(float).eps * scale:
            residues[i] = ((C @ right[:, i]) * (left[:, i].conj() @ B) / denominator)
    return poles, residues


def relative_rms(measured, predicted):
    measured, predicted = np.asarray(measured, complex), np.asarray(predicted, complex)
    floor = max(float(np.median(abs(measured))) * 1e-12, np.finfo(float).tiny)
    return float(100 * np.sqrt(np.mean(abs((predicted - measured) / np.maximum(abs(measured), floor)) ** 2)))


def fit_rational(frequency, impedance, order, query_frequency, normalize=True, *,
                 projection=LOCAL_PROJECTION, frequency_reference_quantile=.5):
    # Keep the original model generator ONLY for reproducing exported backend
    # models. Corrected CV models must never be passed off as those exports.
    from pyimpspec.analysis.drt.lm import (_generate_loewner_matrices,
        _generate_reduced_order_model, _calculate_model_impedance)
    idx = np.argsort(frequency)
    f, z = np.asarray(frequency)[idx], np.asarray(impedance)[idx]
    scale = max(float(np.median(abs(z))), np.finfo(float).tiny) if normalize else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if projection == LEGACY_PROJECTION:
            mats = _generate_loewner_matrices(2j * np.pi * f, z / scale)
            E, A, B, C, _ = _generate_reduced_order_model(*mats, int(order))
            query_s = 2j * np.pi * np.asarray(query_frequency)
        elif projection == LOCAL_PROJECTION:
            if not np.isfinite(frequency_reference_quantile) or not 0 <= frequency_reference_quantile <= 1:
                raise ValueError("Frequency reference quantile must be finite in [0,1]")
            reference_hz = float(np.exp(np.quantile(np.log(f), frequency_reference_quantile)))
            # s'=s/omega0 is dimensionless. Changing frequency units or all
            # physical timescales leaves this internal problem unchanged.
            omega0 = 2 * np.pi * reference_hz
            mats = _loewner_matrices_all_points(1j * (f/reference_hz), z / scale)
            E, A, B, C, _ = _reduce_svd_right_vectors(*mats, int(order))
            query_s = 1j * (np.asarray(query_frequency)/reference_hz)
        else:
            raise ValueError("Unknown Loewner projection identity")
        pred = _calculate_model_impedance(query_s, E, A, B, C) * scale
        if projection == LEGACY_PROJECTION:
            poles, vectors = eig(A, E)
            residues = (np.linalg.solve(vectors, np.linalg.solve(E, B)).reshape(-1)
                        * (C @ vectors).reshape(-1)) * scale
        else:
            poles, residues = _descriptor_poles_residues(E, A, B, C)
            # Z(s)=Z0*H(s/omega0), so p=omega0*p', r=Z0*omega0*r'.
            # Do not multiply infinite descriptor markers into NaN complex values.
            finite = np.isfinite(poles)
            poles[finite] *= omega0
            finite = np.isfinite(residues)
            residues[finite] *= scale * omega0
    if not np.isfinite(pred).all():
        raise ValueError("Nonfinite rational predictions")
    return np.asarray(pred), poles, residues


def pole_audit(poles, residues, frequency, impedance):
    s = 2j * np.pi * np.asarray(frequency)
    scale = max(float(np.sqrt(np.mean(abs(impedance) ** 2))), np.finfo(float).tiny)
    rows = []
    for pole, residue in zip(poles, residues):
        if not np.isfinite(pole) or not np.isfinite(residue):
            rows.append({"finite": False, "material": False,
                         "pole_status": "singular-pencil-indeterminate-pole" if np.isnan(pole)
                         else "infinite-descriptor-pole" if np.isinf(pole)
                         else "finite-pole-residue-not-assessable"})
            continue
        contribution = float(np.sqrt(np.mean(abs(residue / (s - pole)) ** 2)) / scale)
        complex_pole = bool(abs(pole.imag) > 1e-3 * max(abs(pole.real), 1e-20))
        material = contribution >= .01
        # Very fast cancelling pole pairs can approximate the non-proper jwL
        # asymptote. Their sign/angle does not identify measured RC/RL dynamics.
        in_band = bool(2*np.pi*min(frequency)/10 <= abs(pole) <= 2*np.pi*max(frequency)*10)
        rows.append({"finite": True, "real_per_s": float(pole.real), "imag_per_s": float(pole.imag),
                     "residue_real": float(residue.real), "residue_imag": float(residue.imag),
                     "response_rms_fraction": contribution, "material": material,
                     "complex": complex_pole, "unstable": bool(pole.real > 0),
                     "within_expanded_frequency_window": in_band,
                     "pole_role": "in-band-diagnostic" if in_band else "outside-window-or-nuisance-asymptote"})
    material = [r for r in rows if r.get("material") and r.get("within_expanded_frequency_window")]
    incompatible = any(r["complex"] or r["unstable"] for r in material)
    return {"poles": rows, "materiality_threshold_fraction": .01,
            "pole_window_expansion_decades": 1.0,
            "kernel_compatibility": "complex-or-unstable-pole-response" if incompatible else
                "real-pole-consistent" if material else "not-established",
            "claim_boundary": "Pole classification is numerical; cancellations and nuisance asymptotes can obscure individual residues. Real poles do not establish unique physical RC/RL mechanisms."}


def _reliability_gate(heldout_rms, training_rms, audit):
    """Declared engineering guards, not calibrated statistical thresholds."""
    reasons = []
    if not np.isfinite(heldout_rms) or heldout_rms > 2.:
        reasons.append("heldout-reconstruction-above-2-percent-or-nonfinite")
    if not np.isfinite(training_rms) or training_rms > 2.:
        reasons.append("full-refit-reconstruction-above-2-percent-or-nonfinite")
    if np.isfinite(training_rms) and np.isfinite(heldout_rms) and training_rms > 2.*heldout_rms + .05:
        reasons.append("full-refit-worse-than-heldout-envelope")
    if any(p.get("pole_status") == "finite-pole-residue-not-assessable" for p in audit["poles"]):
        reasons.append("finite-pole-residue-not-assessable")
    if any(p.get("pole_status") == "singular-pencil-indeterminate-pole" for p in audit["poles"]):
        reasons.append("singular-pencil-indeterminate-pole")
    return {"kernel_evidence_reliable": not reasons,
            "reliability_status": "usable-numerical-diagnostic" if not reasons else "review-required",
            "reliability_reasons": reasons,
            "reliability_rule": "heldout and full-refit RMS <= 2%; full-refit <= 2*heldout + 0.05 percentage point; finite-pole residues assessable. Engineering guards, not confidence limits."}


def validate_loewner(frequency_hz, impedance, orders=(2, 4, 6, 8, 12, 16, 24, 32),
                     exported_model=None, reference_quantiles=REFERENCE_QUANTILES):
    f, z = np.asarray(frequency_hz, float), np.asarray(impedance, complex)
    if f.ndim != 1 or z.shape != f.shape or len(f) < 15 or not np.isfinite(f).all() or not np.isfinite(z).all() or np.any(f <= 0) or len(np.unique(f)) != len(f):
        return {"status": "not-assessable", "reason": "requires-at-least-15-unique-finite-positive-frequencies"}
    quantiles = sorted(set(float(q) for q in reference_quantiles))
    if not quantiles or any(not np.isfinite(q) or not 0 <= q <= 1 for q in quantiles):
        return {"status": "not-assessable", "reason": "requires-finite-reference-quantiles-in-unit-interval"}
    ix = np.argsort(f)[::-1]
    f, z = f[ix], z[ix]
    indices = np.arange(len(f))
    folds = [indices[(indices > 0) & (indices < len(f)-1) & (indices % 3 == k)] for k in range(3)]
    candidates, refits = [], {}
    for order in sorted(set(orders)):
        if order < 1 or order > min(len(f)-len(test) for test in folds) - 2:
            continue
        for quantile in quantiles:
            predictions, observations, checks = [], [], []
            row = {"order": int(order), "frequency_reference_quantile": quantile,
                   "full_refit_reference_hz": float(np.exp(np.quantile(np.log(f), quantile))),
                   "eligible": False, "full_refit_status": "not-run"}
            try:
                for test in folds:
                    train = np.setdiff1d(indices, test)
                    pred, _, _ = fit_rational(f[train], z[train], order, f[test],
                                              frequency_reference_quantile=quantile)
                    predictions.extend(pred)
                    observations.extend(z[test])
                    checks.append({"train_indices": train.tolist(), "test_indices": test.tolist(),
                                   "fit_input_point_count": len(train),
                                   "frequency_reference_hz": float(np.exp(np.quantile(np.log(f[train]), quantile))),
                                   "heldout_rms_percent": relative_rms(z[test], pred)})
                value = relative_rms(observations, predictions)
                row.update(heldout_rms_percent=value, folds=checks)
                row["full_refit_status"] = "failed"  # replaced only after every refit audit completes
                pred, poles, residues = fit_rational(f, z, order, f,
                                                     frequency_reference_quantile=quantile)
                audit = pole_audit(poles, residues, f, z)
                training_rms = relative_rms(z, pred)
                gate = _reliability_gate(value, training_rms, audit)
                row.update(status="completed", full_refit_status="completed",
                           training_rms_percent=training_rms,
                           raw_pole_classification=audit["kernel_compatibility"],
                           eligible=gate["kernel_evidence_reliable"], **gate)
                refits[(int(order), quantile)] = (pred, audit, gate)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}", folds=checks)
            candidates.append(row)
    usable = [r for r in candidates if r["status"] == "completed" and np.isfinite(r["heldout_rms_percent"])]
    if not usable:
        return {"status": "failed", "candidates": candidates, "reason": "no-finite-holdout-model"}
    eligible = [r for r in usable if r["eligible"]]
    pool = eligible or usable
    best = min(r["heldout_rms_percent"] for r in pool)
    # Fixed engineering tolerance and tie-break; no preference for real poles.
    chosen = min((r for r in pool if r["heldout_rms_percent"] <= best * 1.2 + .05),
                 key=lambda r: (r["order"], abs(r["frequency_reference_quantile"]-.5),
                                r["heldout_rms_percent"], r["frequency_reference_quantile"]))
    pred, audit, reliability = refits[(chosen["order"], chosen["frequency_reference_quantile"])]
    exported_audit = {"status": "not-supplied"}
    if exported_model and exported_model.get("status") == "completed":
        try:
            actual_order = exported_model["model_order"]
            actual_pred, actual_poles, actual_residues = fit_rational(
                f, z, actual_order, f, normalize=False, projection=LEGACY_PROJECTION)
            reference = exported_model["reconstructed"]
            reference = np.asarray([complex(v["real"],v["imag"]) if isinstance(v,dict) else v for v in reference])[ix]
            agreement = relative_rms(reference, actual_pred)
            if agreement > 1e-4:
                raise ValueError("Recreated model disagrees with exported reconstruction; pole audit not transferable")
            exported_audit = {"status": "completed", "model_order": actual_order,
                "recreation_rms_percent": agreement, **pole_audit(actual_poles,actual_residues,f,z),
                "projection": LEGACY_PROJECTION, "impedance_normalized": False,
                "validation_target": "exported-full-data-matrix-rank-model",
                "heldout_validated": False}
        except Exception as exc:
            exported_audit = {"status": "failed", "error": str(exc), "heldout_validated": False}
    training_rms = relative_rms(z, pred)
    raw_classification = audit["kernel_compatibility"]
    if not reliability["kernel_evidence_reliable"]:
        audit["kernel_compatibility"] = "not-established"
    return {"status": "completed", "method": "3-fold endpoint-preserving interleaved frequency holdout",
            "projection": LOCAL_PROJECTION, "fit_uses_all_training_points": True,
            "validation_target": "separate-reduced-order-model-not-exported-discrete-weights",
            "exported_model_audit": exported_audit,
            "sorted_frequency_hz": f.tolist(), "input_sort_indices": ix.tolist(),
            "order_candidates": list(orders), "selected_order": chosen["order"],
            "reference_quantiles": quantiles,
            "selected_frequency_reference_quantile": chosen["frequency_reference_quantile"],
            "selected_frequency_reference_hz": chosen["full_refit_reference_hz"],
            "selection_status": "eligible-candidate-selected" if eligible else "no-reliable-candidate-diagnostic-only",
            "eligible_candidate_count": len(eligible),
            "selection_uses_full_data_refit_audit": True,
            "heldout_rms_percent": chosen["heldout_rms_percent"], "training_rms_percent": training_rms,
            "selection_tolerance": "among refit-audited eligible candidates: smallest order within 1.2*best heldout RMS + 0.05 percentage point; then reference quantile closest to 0.5, then lower CV RMS. If none eligible, retain an explicitly unreliable diagnostic only",
            "conditioning_scope": "s'=s/(2*pi*reference_hz), Z'=Z/median|Z|; fixed log-frequency reference quantiles change truncated-pencil approximation geometry, not measured points. Physical poles and residues restored. Not denoising or calibrated noise weighting",
            "candidates": candidates, **audit, **reliability,
            "raw_pole_classification": raw_classification,
            "scope": "Frequency interpolation within one sweep only. Each fold excludes test impedances from fitting, but order/conditioning selection also uses a full-data-refit audit: selected CV error is not an independent final generalization estimate. This model is NOT the exported matrix-rank model. Never auto-publish Loewner weights as continuous DRT."}

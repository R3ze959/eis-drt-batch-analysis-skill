"""Separate a useful reconstruction recommendation from physical acceptance."""
import numpy as np
from loewner_validation import relative_rms


def complex_array(values):
    return np.asarray([complex(v['real'], v['imag']) if isinstance(v, dict) else v for v in values], complex)


def recommend_model(result):
    curves = result.get("curves", {})
    measured = np.asarray(curves.get("zreal_measured", [])) - 1j * np.asarray(curves.get("neg_zimag_measured", []))
    primary = np.asarray(curves.get("zreal_reconstructed", [])) - 1j * np.asarray(curves.get("neg_zimag_reconstructed", []))
    if measured.ndim != 1 or not measured.size or measured.shape != primary.shape or not np.isfinite(primary).all() or not np.isfinite(measured).all():
        return {"status": "unresolved", "recommended_branch": None, "reasons": ["primary-not-completed"]}
    base = relative_rms(measured, primary)
    candidates = [{"branch": "primary_rc", "rms_percent": base, "distribution_stability": "see-primary-peak-and-lambda-evidence"}]
    signed = result.get("signed_gdrt", {})
    if signed.get("status") == "completed":
        pred = complex_array(signed.get("reconstructed", []))
        if pred.shape == measured.shape and np.isfinite(pred).all():
            candidates.append({"branch": "signed_gdrt", "rms_percent": relative_rms(measured, pred),
                "distribution_stability": signed.get("sensitivity", {}).get("status", "not-assessed")})
    nonseries = bool(result.get("nonseries_inductance_evidence", {}).get("detected"))
    chosen = candidates[0]
    # Do not exchange interpretable RC for extra flexibility to shave a tiny residual.
    if len(candidates) > 1 and (nonseries or base > 2) and candidates[1]["rms_percent"] <= .8 * base:
        chosen = candidates[1]
    competition = result.get("predictive_rc_competition", {})
    selected_key = competition.get("selected_candidate")
    predictive = competition.get("candidates", {}).get(selected_key, {})
    if predictive.get("status") == "completed":
        pred = complex_array(predictive.get("reconstructed", []))
        if pred.shape == measured.shape and np.isfinite(pred).all():
            candidate = {"branch": predictive["reconstruction_branch"], "rms_percent": relative_rms(measured, pred),
                         "prediction_rms_percent": predictive.get("prediction_rms_percent"),
                         "distribution_stability": "individual-peaks-not-validated"}
            candidates.append(candidate)
            viable = predictive.get("full_refit_guard", {}).get("status") == "pass"
            near_positive = (selected_key == "positive_rc" and
                             candidate["rms_percent"] <= 1.1 * chosen["rms_percent"] + .2)
            if (viable and candidate["rms_percent"] <= .8 * base and
                    (candidate["rms_percent"] < chosen["rms_percent"] or near_positive)):
                chosen = candidate
    reasons = []
    if chosen["rms_percent"] > 2:
        reasons.append("recommended-reconstruction-above-2-percent-heuristic")
    if chosen["branch"] == "signed_gdrt":
        if chosen["distribution_stability"] != "stable":
            reasons.append("signed-distribution-not-stable")
        if signed.get("lambda_boundary_sensitive"):
            reasons.append("signed-lambda-boundary-sensitive")
        if signed.get("support", {}).get("supported_absolute_area_fraction", 0) < .8:
            reasons.append("signed-substantial-outside-window-mass")
    elif chosen["branch"].startswith("predictive_"):
        reasons.extend(predictive.get("reliability_reasons", []))
        reasons.append("predictive-individual-peaks-not-validated")
        if predictive.get("support", {}).get("supported_absolute_area_fraction", 0) < .8:
            reasons.append("predictive-substantial-outside-window-mass")
        if chosen["branch"] == "predictive_positive_rc" and nonseries:
            reasons.append("positive-RC-kernel-versus-nonseries-screen-requires-review")
    elif nonseries:
        reasons.append("ordinary-RC-kernel-incompatible")
    else:
        if result.get("lambda_selection", {}).get("boundary_sensitive"):
            reasons.append("primary-lambda-boundary-sensitive")
    lv = result.get("loewner_validation", {})
    if lv.get("kernel_evidence_reliable") is True and lv.get("kernel_compatibility") == "complex-or-unstable-pole-response" and lv.get("heldout_rms_percent", 100) < chosen["rms_percent"] * .8:
        reasons.append("better-predicting-rational-model-has-non-RC-RL-poles")
    if result.get("kk", {}).get("status") not in {"screen-pass", "passed", "pass"}:
        reasons.append("KK-not-screen-pass")
    return {"status": "exploratory-review-required" if reasons else "exploratory-numerical-candidate",
            "recommended_branch": chosen["branch"], "recommended_rms_percent": chosen["rms_percent"],
            "primary_rms_percent": base, "reconstruction_improvement_fraction": 1 - chosen["rms_percent"] / max(base, 1e-300),
            "selection_scope": "impedance-reconstruction-only; not a physical-model identification",
            "reasons": reasons, "candidates": candidates,
            "selection_basis": "Keep primary unless a viable candidate reduces RMS by at least 20%; predictive candidates preselected by matched frequency CV; prefer its positive kernel within 10 percent plus 0.2 percentage point of current reconstruction RMS; no full-fit worsening versus primary. Legacy signed RMS is not CV evidence. Stability reported separately.",
            "claim_boundary": "A reconstruction recommendation, not a validated physical distribution or mechanism. DDT topology candidates and Loewner weights remain separate diagnostics. Primary and recommended arrays are never silently exchanged."}

"""Control evidence is attached to spectra, never broadcast to an entire batch."""
from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


def uid(row):
    # Production always supplies spectrum_uid. The fallback supports standalone
    # diagnostic callers; it is not used as an on-disk primary key.
    return str(row.get("spectrum_uid") or (str(row.get("source", "")) + "::" +
               str(row.get("spectrum_id") or row.get("label", ""))))


def repeat_peak_assessment(first, second, frequency):
    """Resolved major peaks only; lack of such evidence is not a pass."""
    from advanced_drt import assess_peak_stability, _peak_frequency, finite_float
    excluded = {"minor-low-signal", "unsupported-outside-window", "boundary-sensitive",
                "inductive-overlap", "unstable", "split-merge-ambiguous"}
    def major(result):
        peaks = []
        for p in result.get("peaks", []):
            f = _peak_frequency(p)
            height = finite_float(p.get("relative_height"))
            if (f is not None and frequency.min() <= f <= frequency.max()
                    and p.get("confidence") not in excluded and (height is None or height >= .1)):
                peaks.append(p)
        return peaks
    a, b = major(first), major(second)
    if not a and not b:
        return {"status": "not-assessable", "reason": "no-resolved-major-peaks"}
    forward, reverse = assess_peak_stability(a, b), assess_peak_stability(b, a)
    stable = (len(a) == len(b) and forward["stable_match_count"] == len(a)
              and reverse["stable_match_count"] == len(b))
    return {"status": "accepted" if stable else "changed-materially",
            "forward": forward, "reverse": reverse,
            "rule": "All resolved in-window major peaks match in both directions; relative height >= 0.1."}


def assess_controls(results, repeat_warn_percent=2.0, linearity_warn_percent=2.0,
                    stationarity_warn_percent=2.0):
    from advanced_drt import _group_results, _pair_assessment, _amplitude, _rest_time

    repeat_rows, linearity_rows, stationarity_rows = [], [], []
    def base(group, members):
        return {"group": group, "member_count": len(members),
                "members": [m["label"] for m in members], "member_uids": [uid(m) for m in members]}

    for group, members in _group_results(results, "repeat").items():
        pairs = [_pair_assessment(a, b) for a, b in itertools.combinations(members, 2)]
        complete = [p for p in pairs if p["status"] == "completed"]
        scores = [p["complex_rms_percent"] for p in complete]
        assessable = len(complete) == len(pairs)
        status = "repeat-supported" if len(members) >= 3 else "pair-only"
        if not assessable:
            status = "not-assessable"
        elif max(scores) > repeat_warn_percent:
            status = "repeat-disagreement"
        elif any(p["repeat_peak_stability"]["status"] == "changed-materially" for p in complete):
            status = "repeat-peak-disagreement"
        elif any(p["repeat_peak_stability"]["status"] != "accepted" for p in complete):
            status = "repeat-peak-not-assessable"
        anomalies = []
        if assessable and len(members) >= 3:
            member_scores = []
            for m in members:
                vals = [p["complex_rms_percent"] for p in complete
                        if uid(m) in (p["first_uid"], p["second_uid"])]
                member_scores.append(min(vals) if len(members) == 3 else float(np.median(vals)))
            center = float(np.median(member_scores))
            mad = float(np.median(np.abs(np.asarray(member_scores) - center)))
            for m, value in zip(members, member_scores):
                mz = 0.6745 * (value - center) / mad if mad > 1e-12 else None
                # An outlier is a high-discrepancy member, never the best repeat.
                outlier = value > repeat_warn_percent and (
                    mz > 3.5 if mz is not None else value > max(repeat_warn_percent, 3 * center))
                anomalies.append({"label": m["label"], "spectrum_uid": uid(m),
                                  "nearest_or_median_pair_rms_percent": value,
                                  "modified_z": mz, "outlier": bool(outlier)})
            if any(a["outlier"] for a in anomalies):
                status = "repeat-outlier-detected"
        repeat_rows.append({**base(group, members), "status": status,
                            "threshold_percent": repeat_warn_percent,
                            "pair_assessments": pairs, "anomaly_scores": anomalies})

    for group, members in _group_results(results, "linearity").items():
        members = [m for m in members if _amplitude(m) is not None and _amplitude(m) > 0]
        levels = sorted({_amplitude(m) for m in members})
        if len(levels) < 2:
            continue
        pairs = []
        for a, b in itertools.combinations(members, 2):
            pair = _pair_assessment(a, b)
            pair.update(first_amplitude_mv=_amplitude(a), second_amplitude_mv=_amplitude(b))
            pairs.append(pair)
        complete = [p for p in pairs if p["status"] == "completed"]
        maximum = max((p["complex_rms_percent"] for p in complete), default=None)
        status = "not-assessable" if len(complete) != len(pairs) else (
            "linearity-supported" if maximum <= linearity_warn_percent else "amplitude-dependent")
        linearity_rows.append({**base(group, members), "status": status, "amplitudes_mv": levels,
                               "threshold_percent": linearity_warn_percent,
                               "maximum_pair_rms_percent": maximum, "pair_assessments": pairs})

    for group, members in _group_results(results, "stationarity").items():
        members = [m for m in members if _rest_time(m) is not None and _rest_time(m) >= 0]
        rests = sorted({_rest_time(m) for m in members})
        if len(rests) < 2:
            continue
        levels = {t: [m for m in members if _rest_time(m) == t] for t in rests}
        adjacent = []
        for t1, t2 in zip(rests[:-1], rests[1:]):
            for a, b in itertools.product(levels[t1], levels[t2]):
                pair = _pair_assessment(a, b)
                pair.update(first_rest_s=t1, second_rest_s=t2)
                adjacent.append(pair)
        terminal = [p for p in adjacent if p["second_rest_s"] == rests[-1]]
        terminal_repeats = [_pair_assessment(a, b) for t in rests[-2:]
                            for a, b in itertools.combinations(levels[t], 2)]
        checks = terminal + terminal_repeats
        assessable = all(p["status"] == "completed" for p in checks)
        last = max((p["complex_rms_percent"] for p in checks if p["status"] == "completed"), default=None)
        status = "not-assessable" if not assessable else (
            "two-rest-discrepancy-only" if len(rests) == 2 else
            "rest-stable" if last <= stationarity_warn_percent else "rest-dependent")
        # A late plateau does not certify the deliberately short-rest spectra.
        supported_uids = [uid(m) for t in rests[-2:] for m in levels[t]] if status == "rest-stable" else []
        stationarity_rows.append({**base(group, members), "status": status,
                                 "rest_times_s": rests, "supported_member_uids": supported_uids,
                                 "threshold_percent": stationarity_warn_percent, "last_pair_rms_percent": last,
                                 "consecutive_assessments": adjacent, "terminal_repeat_assessments": terminal_repeats,
                                 "claim_boundary": "Plateau evidence covers the final two distinct rest levels only."})
    groups = {"repeat": repeat_rows, "linearity": linearity_rows, "stationarity": stationarity_rows}
    adverse = {"repeat-disagreement", "repeat-peak-disagreement", "repeat-outlier-detected",
               "amplitude-dependent", "rest-dependent"}
    supported = {"repeat-supported", "linearity-supported", "rest-stable"}
    by_spectrum = []
    for result in results:
        controls, links = {}, {}
        for purpose, rows in groups.items():
            linked = [r for r in rows if uid(result) in r["member_uids"]]
            links[purpose] = [r["group"] for r in linked]
            positive = any(r["status"] in supported and (purpose != "stationarity" or
                           uid(result) in r.get("supported_member_uids", [])) for r in linked)
            controls[purpose] = ("contradicted" if any(r["status"] in adverse for r in linked)
                                 else "supported" if positive else "evidence-missing")
        state = ("contradicted" if "contradicted" in controls.values() else
                 "supported" if all(v == "supported" for v in controls.values()) else "evidence-missing")
        by_spectrum.append({"spectrum_uid": uid(result), "controls": controls,
                            "control_group_links": links, "evidence_status": state})
    return {"repeat_assessments": repeat_rows, "linearity_assessments": linearity_rows,
            "stationarity_assessments": stationarity_rows, "per_spectrum": by_spectrum,
            "coverage": {"repeat_groups": len(repeat_rows), "linearity_groups": len(linearity_rows),
                         "stationarity_groups": len(stationarity_rows),
                         "repeat_status": "available-3plus" if any(r["member_count"] >= 3 for r in repeat_rows) else "evidence-missing",
                         "linearity_status": "available" if linearity_rows else "evidence-missing",
                         "stationarity_status": "available-3plus" if any(len(r["rest_times_s"]) >= 3 for r in stationarity_rows) else "evidence-missing",
                         "per_spectrum_evidence_counts": dict(Counter(r["evidence_status"] for r in by_spectrum))},
            "scope_rule": "Group membership and matched metadata only; no implicit evidence transfer between cells or states."}


def classify_results(results, validation):
    """Separate execution, numerical quality, experimental evidence, and display."""
    evidence = {r["spectrum_uid"]: r for r in validation["per_spectrum"]}
    for result in results:
        e = evidence[uid(result)]
        reasons = []
        complete = result.get("status") == "completed"
        kk = result.get("kk", {})
        if not complete:
            reasons.append("computation-not-completed")
        if kk.get("status") not in {"screen-pass", "passed", "pass"}:
            reasons.append("KK-not-screen-pass")
        if kk.get("downsampled_for_screen"):
            reasons.append("KK-subset-needs-full-point-confirmation")
        if result.get("lambda_selection", {}).get("boundary_sensitive"):
            reasons.append("lambda-boundary-sensitive")
        drt = result.get("drt", {})
        from advanced_drt import finite_float
        residuals = [finite_float(drt.get(k)) for k in ("residual_rms_re_percent", "residual_rms_im_percent")]
        if any(v is None or v < 0 for v in residuals):
            reasons.append("primary-reconstruction-RMS-missing-or-nonfinite")
            rms = None
        else:
            rms = np.hypot(*residuals)
        if rms is not None and rms > 2.0:
            reasons.append("primary-reconstruction-RMS-above-2-percent-heuristic")
        if result.get("validity", {}).get("single_sweep_drift_screen", {}).get("status") == "structured-drift-suspect":
            reasons.append("structured-residual-drift")
        nonseries = bool(result.get("nonseries_inductance_evidence", {}).get("detected"))
        diffusion = bool(result.get("ddt", {}).get("trigger", {}).get("detected"))
        for branch, needed in (("signed_gdrt", nonseries), ("ddt", diffusion)):
            if needed and result.get(branch, {}).get("status") != "completed":
                reasons.append(branch + "-required-not-completed")
        if nonseries and result.get("signed_gdrt", {}).get("sensitivity", {}).get("status") == "sensitive":
            reasons.append("signed-distribution-sensitive")
        signed = result.get("signed_gdrt", {})
        if signed.get("status") == "completed":
            if signed.get("lambda_boundary_sensitive"):
                reasons.append("signed-lambda-boundary-sensitive")
            if signed.get("sensitivity", {}).get("status") == "range-incomplete":
                reasons.append("signed-lambda-range-incomplete")
        if diffusion and result.get("ddt", {}).get("selection_status") == "no-distribution-candidate-accepted":
            reasons.append("diffusion-topology-or-timescale-unresolved")
        if nonseries:
            reasons.append("ordinary-RC-kernel-incompatible")
        if result.get("inductance_detected") and result.get("inductance_policy") == "off":
            reasons.append("inductance-disabled-compatibility-only")
        if any(p.get("confidence") in {"unstable", "split-merge-ambiguous"} and
               (p.get("relative_height") or 0) >= 0.1 for p in result.get("peaks", [])):
            reasons.append("major-peak-unresolved")
        if "better-predicting-rational-model-has-non-RC-RL-poles" in result.get("model_recommendation", {}).get("reasons", []):
            reasons.append("predictive-in-band-poles-not-RC-RL-compatible")
        numerical = "failed" if not complete else "review-required" if reasons else "supported"
        # Perturbation persistence is separate from lambda stability; lack of a
        # valid noise model is uncertainty, not proof that a retained peak is fake.
        if any(p.get("perturbation_evidence_status") == "noise-or-resolution-sensitive"
               and (p.get("relative_height") or 0) >= .1 for p in result.get("peaks", [])):
            reasons.append("major-peak-noise-or-resolution-sensitive")
            numerical = "review-required"
        verdict = ("review-required" if numerical != "supported" or e["evidence_status"] == "contradicted"
                   else "accepted-with-supplied-controls" if e["evidence_status"] == "supported"
                   else "controls-incomplete-no-acceptance")
        display = "review-pending" if not complete else "diagnostic-only" if reasons or e["evidence_status"] == "contradicted" else "included-exploratory"
        if verdict == "accepted-with-supplied-controls":
            display = "included"
        result.update({**e, "numeric_status": numerical, "validity_status": verdict,
                       "ordinary_rc_eligible": complete and not nonseries and not (
                           result.get("inductance_detected") and result.get("inductance_policy") == "off"),
                       "signed_gdrt_required": nonseries, "ddt_required": diffusion,
                       "display_role": display, "decision_basis": reasons or ["numerical-screens-supported"],
                       "boundary_status": "boundary-sensitive" if any(p.get("confidence") in {
                           "boundary-sensitive", "unsupported-outside-window"} for p in result.get("peaks", [])) else "in-window",
                       "claim_level": "numerically_supported_mechanism_unassigned" if numerical == "supported" else "diagnostic_only"})
    verdicts = [r["validity_status"] for r in results]
    validation["overall_verdict"] = ("review-required" if not verdicts or "review-required" in verdicts else
        "accepted-with-supplied-controls" if all(v == "accepted-with-supplied-controls" for v in verdicts)
        else "controls-incomplete-no-acceptance")
    validation["verdict_counts"] = dict(Counter(verdicts))
    validation["verdict_rules"] = {"missing_controls": "never a pass; exploratory display remains possible",
                                    "numerical_RMS_review_threshold_percent": 2.0,
                                    "no_spectrum_label_rules": True}
    return validation

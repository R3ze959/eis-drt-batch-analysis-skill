"""Versioned provenance, input selection, resume validation, and classified exports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from metadata_contract import canonical_metadata, condition_key, GROUP_FIELDS

SCHEMA_VERSION = 5

# Rendering can be requested after data processing without changing its numerical
# identity. Actual presentation settings remain recorded in the current run report.
PRESENTATION_PARAMETERS = {"trend_plots", "trend_dpi"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def identities(source, content_hash, spectrum_id):
    source_uid = digest(str(Path(source).resolve()))[:24]
    return {"source_uid": source_uid,
            "spectrum_uid": digest([source_uid, content_hash, str(spectrum_id)])[:24]}


def numerical_digest(result):
    from batch_drt import json_ready
    keys = ("spectrum_uid", "run_fingerprint", "source_sha256", "impedance_basis", "metadata",
            "qc", "kk", "zhit", "drt", "peaks", "lambda_selection", "curves",
            "inductance_model_selection", "nonseries_inductance_evidence",
            "loewner_rc_rl", "signed_gdrt", "ddt", "model_reduce",
            "peak_evidence", "model_recommendation", "loewner_validation", "predictive_rc_competition")
    return digest(json_ready({key: result.get(key) for key in keys}))


def row_selected(row):
    role = str(row.get("input_role", "eis")).strip().lower()
    include = str(row.get("include", "true")).strip().lower()
    return include not in {"0", "false", "no", "exclude", "excluded"} and role in {"", "eis", "raw-eis", "spectrum"}


def inventory_inputs(sources, exact, by_path, *, manifest=None, manifest_mode="allowlist"):
    from batch_drt import sha256_file
    selected, inventory = [], []
    for source in sources:
        path = str(source.resolve())
        source_rows = [row for (p, _), row in exact.items() if p == path]
        if path in by_path:
            source_rows.append(by_path[path])
        reason, role = "candidate-EIS", "eis-candidate"
        choose = True
        if manifest and Path(manifest).resolve() == source.resolve():
            choose, reason, role = False, "input-manifest-not-spectrum", "manifest"
        elif manifest and manifest_mode == "allowlist" and not source_rows:
            choose, reason, role = False, "not-listed-in-manifest", "unselected"
        elif source_rows and not any(row_selected(r) for r in source_rows):
            choose, reason, role = False, " | ".join(r.get("exclusion_reason") or r.get("input_role") or "manifest-excluded" for r in source_rows), "manifest-excluded"
        # Header evidence, not folder names or Pair labels, identifies prior exports.
        if choose and not source_rows and source.suffix.lower() in {".csv", ".txt", ".tsv"}:
            with source.open("rb") as handle:
                sample = handle.read(16384).decode("utf-8-sig", errors="replace").lower()
            lines = sample.splitlines()[:20]
            header = " ".join(lines)
            frequency = bool(re.search(r"frequency|freq(?:uency)?[_\s(]|频率", header))
            impedance = bool(re.search(r"zreal|zre\b|z['′]|zreal_measured|real.*ohm|阻抗", header))
            derived = bool(re.search(r"gamma|弛豫|relaxation|tau_s|signed_gamma", header))
            process = bool(re.search(r"time|时间", header) and re.search(r"voltage|potential|ocp|电压", header))
            if "spectrum_uid" in header or ("source_sha256" in header and "spectrum_id" in header):
                choose, reason, role = False, "generated-result-schema", "generated-output"
            elif derived and not (frequency and impedance):
                choose, reason, role = False, "distribution-columns-without-EIS-components", "supplied-drt-or-export"
            elif process and not frequency:
                choose, reason, role = False, "time-potential-table-without-frequency", "process-or-ocp"
        sha = sha256_file(source)
        inventory.append({"source": path, "source_uid": identities(source, sha, "")["source_uid"],
                          "source_sha256": sha, "input_role": role, "selected": choose, "selection_reason": reason})
        if choose:
            selected.append(source)
    listed = {p for p, _ in exact} | set(by_path)
    discovered = {str(p.resolve()) for p in sources}
    missing = [p for p in listed - discovered if any(row_selected(r) for r in
               ([by_path[p]] if p in by_path else []) + [r for (q, _), r in exact.items() if q == p])]
    if missing:
        raise ValueError("Selected manifest paths are missing or outside supplied inputs: " + "; ".join(sorted(missing)))
    return selected, inventory


def execution_contract(args, inventory, script_dir):
    from batch_drt import sha256_file, package_versions, json_ready
    from runtime_contract import runtime_identity
    code = {str(p.resolve()): sha256_file(p) for p in sorted(Path(script_dir).glob("*.py"))}
    root = Path(script_dir).resolve().parent
    for resource in (root / "VERSION", root / "requirements.lock.txt",
                     root / "assets/battery_origin.mplstyle"):
        if resource.is_file():
            code[str(resource)] = sha256_file(resource)
    parser = Path(args.parser_script).expanduser().resolve()
    code[str(parser)] = sha256_file(parser)
    parameters = {k: v for k, v in vars(args).items() if k not in
                  {"output_dir", "resume", "export_only", "allow_partial"} | PRESENTATION_PARAMETERS}
    contract = json_ready({"schema_version": SCHEMA_VERSION,
                "release_version": (root / "VERSION").read_text().strip() if (root / "VERSION").is_file() else None,
                "parameters": parameters,
                "input_files": inventory, "code_sha256": code, "package_versions": package_versions(),
                "runtime": runtime_identity(),
                "manifest_sha256": sha256_file(Path(args.manifest)) if args.manifest else None})
    contract["run_fingerprint"] = digest(contract)
    return contract


def initialize_run(output, contract, resume=False):
    from batch_drt import write_json
    path = output / "execution_contract.json"
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"Output is nonempty; use a new directory or explicit --resume: {output}")
        if not path.is_file():
            raise ValueError("Legacy/nonmatching output cannot be resumed without an execution contract")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != contract["run_fingerprint"]:
            raise ValueError("Resume refused: inputs, metadata, parameters, code, or package versions changed")
    elif resume:
        raise ValueError("--resume requires an existing matching run")
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, contract)
    if not (output / "run_manifest.json").is_file():
        write_json(output / "run_manifest.json", {**contract, "status": "initialized", "result_files": []})


def common_fields(result):
    m = canonical_metadata(result.get("metadata", {}))
    failed = result.get("status") == "failed"
    return {"spectrum_uid": result.get("spectrum_uid"), "source_uid": result.get("source_uid"),
            "spectrum_id": result.get("spectrum_id"), "source": result.get("source"),
            "source_sha256": result.get("source_sha256"), "label": result.get("label"),
            "impedance_basis": result.get("impedance_basis"),
            "cell_id": m.get("cell_id"), "sample_id": m.get("sample_id"),
            "direction": m.get("direction"), "cycle": m.get("cycle"), "state": m.get("state"),
            "voltage_v": m.get("voltage_v", m.get("potential_v", m.get("target_voltage_v"))),
            "soc_percent": m.get("soc_percent"), "temperature_C": m.get("temperature_C", m.get("temperature_c")),
            **{k: m.get(k) for k in GROUP_FIELDS}, "data_origin": m.get("data_origin", "experimental-or-unspecified"),
            "display_role": result.get("display_role") or "review-pending",
            "validity_status": result.get("validity_status") or ("review-required" if failed else "not-assessed"),
            "numeric_status": result.get("numeric_status") or ("failed" if failed else "not-assessed"),
            "evidence_status": result.get("evidence_status") or "not-assessed",
            "recommended_branch": result.get("recommended_branch"),
            "recommendation_status": result.get("recommendation_status"),
            "recommended_rms_percent": result.get("recommended_rms_percent")}


def normalized_input_rows(payloads, results):
    """Join current verdicts by UID while retaining the payload observation arrays."""
    by_uid = {}
    for result in results:
        uid = result.get("spectrum_uid")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Classified export requires a spectrum_uid for every result")
        if uid in by_uid:
            raise ValueError(f"Duplicate result spectrum_uid: {uid}")
        by_uid[uid] = result
    rows, seen = [], set()
    for payload in payloads:
        uid = payload.get("spectrum_uid")
        if not isinstance(uid, str) or not uid or uid not in by_uid:
            raise ValueError(f"Normalized EIS has no matching result spectrum_uid: {uid!r}")
        if uid in seen:
            raise ValueError(f"Duplicate payload spectrum_uid: {uid}")
        seen.add(uid)
        result = by_uid[uid]
        for key in ("source_uid", "source_sha256", "spectrum_id", "source", "impedance_basis"):
            if payload.get(key) is not None and result.get(key) is not None and payload[key] != result[key]:
                raise ValueError(f"Payload/result {key} conflicts for spectrum_uid {uid}")
        arrays = [np.asarray(payload[key]) for key in ("acquisition_index", "frequency_hz", "zreal", "neg_zimag")]
        if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
            raise ValueError(f"Normalized EIS observation arrays are not aligned for spectrum_uid {uid}")
        common = common_fields(result)
        source_lines = payload.get("metadata", {}).get("included_source_lines")
        if source_lines is not None and (len(source_lines) != len(arrays[0]) or
                any(not isinstance(line, (int, np.integer)) or line < 1 for line in source_lines)):
            raise ValueError(f"Text source lines are not aligned for spectrum_uid {uid}")
        for point, (a, f, real, neg_imag) in enumerate(zip(*arrays)):
            rows.append({**common, "status": result.get("status") or "not-reported", "error": result.get("error"),
                         "verdict_join": "spectrum_uid", "acquisition_index": a,
                         "included_point_index": point + 1,
                         "source_line": source_lines[point] if source_lines is not None else None,
                         "frequency_hz": f, "zreal": real, "neg_zimag": neg_imag})
    return rows


def write_table(path, rows, fields=()):
    from batch_drt import json_ready
    fields = list(dict.fromkeys([*fields, *(k for r in rows for k in r)])) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = json_ready(row)
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in clean.items()})
    temporary.replace(path)


def complex_values(values):
    if isinstance(values, list) and values and isinstance(values[0], dict):
        return np.asarray([v["real"] + 1j * v["imag"] for v in values])
    return np.asarray(values, complex)


def export_bundle(output, results, payloads, inventory, run, validation):
    """JSON is authoritative. Tables are reproducible views, never raw-file edits."""
    from derived_outputs import publish_tree
    output = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix=".drt-export-", dir=output.parent) as temp:
        stage = Path(temp)
        _export_bundle(stage, results, payloads, inventory, run, validation,
                       output / "execution_contract.json")
        return publish_tree(stage, output, "export")


def _export_bundle(output, results, payloads, inventory, run, validation, contract_path):
    from batch_drt import write_json
    folders = ["00_overview", "01_inputs", "02_quality", "03_results/primary_drt", "03_results/recommended",
               "03_results/signed_gdrt", "03_results/predictive_rc", "03_results/diffusion_candidates", "03_results/model_reduce",
               "03_results/reference_compatibility", "04_plot_data", "05_figures/scientific",
               "05_figures/diagnostic", "05_figures/promotion", "06_reproducibility"]
    for name in folders:
        (output / name).mkdir(parents=True, exist_ok=True)
    write_table(output / "00_overview/input_inventory.csv", inventory)
    write_json(output / "00_overview/run_manifest.json", run)
    write_json(output / "01_inputs/source_audits.json", run.get("source_audits", []))
    source_rows = []
    for audit in run.get("source_audits", []):
        for row in audit.get("row_ledger", []):
            source = audit.get("source")
            content_hash = audit.get("sha256")
            if not source or not content_hash or not isinstance(row.get("source_line"), int):
                raise ValueError("Source row ledger needs source, hash and an integer 1-based source_line")
            if row["source_line"] < 1:
                raise ValueError("Source row ledger line numbers are one-based")
            identity = identities(source, content_hash, row.get("spectrum_id", "__source_line__"))
            source_rows.append({"source": source, "source_sha256": content_hash,
                                "source_uid": identity["source_uid"], **row})
    write_table(output / "01_inputs/source_row_ledger.csv", source_rows)
    write_json(output / "02_quality/batch_validation.json", validation)
    write_json(output / "06_reproducibility/execution_contract.json",
               json.loads(contract_path.read_text()))
    source_snapshot = output / "06_reproducibility/code"
    source_snapshot.mkdir(exist_ok=True)
    for path, expected in run.get("code_sha256", {}).items():
        # Basenames for the skill scripts are unique; parser lives in its own slot.
        source = Path(path)
        from batch_drt import sha256_file
        if sha256_file(source) != expected:
            raise ValueError("Source code changed during run; refusing a misleading reproduction snapshot: " + str(source))
        target = source_snapshot / ("parser_" + source.name if "eis-impedance-plotting" in path else source.name)
        shutil.copy2(source, target)
    normalized = normalized_input_rows(payloads, results)
    write_table(output / "01_inputs/normalized_eis.csv", normalized)
    index, eis, peaks, sensitivity, diffusion, reduction = [], [], [], [], [], []
    primary, signed, loewner, diffusion_curves, branch_reconstructions = [], [], [], [], []
    recommendations, recommended_curves, recommended_eis, evidence_rows, rational_rows = [], [], [], [], []
    predictive_curves, predictive_scores = [], []
    for r in results:
        common = common_fields(r)
        index.append({**common, "status": r.get("status"), "error": r.get("error"),
                      "claim_level": r.get("claim_level"), "ordinary_rc_eligible": r.get("ordinary_rc_eligible"),
                      "signed_gdrt_required": r.get("signed_gdrt_required"), "ddt_required": r.get("ddt_required"),
                      "boundary_status": r.get("boundary_status"), "decision_basis": r.get("decision_basis"),
                      "control_group_links": r.get("control_group_links"),
                      "kk_source_points": r.get("kk", {}).get("source_point_count"),
                      "kk_screen_points": r.get("kk", {}).get("screen_point_count"),
                      "figure_status": r.get("figure_status"),
                      "result_json": "results/" + str(r.get("result_slug")) + ".json" if r.get("result_slug") else None})
        if r.get("status") != "completed":
            continue
        c = r["curves"]
        recommendation = r.get("model_recommendation", {})
        recommendations.append({**common, **recommendation, "selection_scope": "reconstruction-only",
                                "selection_is_physical_acceptance": False})
        chosen = recommendation.get("recommended_branch")
        predictive = r.get("predictive_rc_competition", {}).get("candidates", {})
        distributions = {"primary_rc": c, "signed_gdrt": r.get("signed_gdrt", {}),
                         **{"predictive_" + name: branch for name, branch in predictive.items()}}
        if chosen in distributions:
            distribution = distributions[chosen]
            for t, g in zip(distribution.get("tau_s", []), distribution.get("gamma", [])):
                recommended_curves.append({**common, "selected_branch": chosen, "distribution_branch": chosen,
                    "selection_scope": "reconstruction-only", "tau_s": t, "gamma": g,
                    "distribution_measure": "signed_gamma_per_ln_tau" if chosen in {"signed_gdrt", "predictive_signed_rc"} else "gamma_per_ln_tau",
                    "selection_is_physical_acceptance": False})
            zfit = (np.asarray(c["zreal_reconstructed"]) - 1j*np.asarray(c["neg_zimag_reconstructed"]) if chosen == "primary_rc"
                    else complex_values(distribution.get("reconstructed", [])))
            if len(zfit) != len(c["frequency_hz"]):
                raise ValueError("Recommended reconstruction length differs from original frequency grid")
            for f, z in zip(c["frequency_hz"], zfit):
                recommended_eis.append({**common, "selected_branch": chosen, "reconstruction_branch": chosen,
                    "selection_scope": "reconstruction-only", "selection_is_physical_acceptance": False, "frequency_hz": f,
                    "zreal_reconstructed": z.real, "neg_zimag_reconstructed": -z.imag})
        for p in r.get("peak_evidence", {}).get("per_peak", []):
            evidence_rows.append({**common, **p, "distribution_branch": "primary_rc"})
        lv = r.get("loewner_validation", {})
        rational_rows.append({**common, **{k: lv.get(k) for k in ("status", "selected_order", "heldout_rms_percent",
                              "training_rms_percent", "kernel_compatibility", "scope", "validation_target")},
                              "exported_model_order": lv.get("exported_model_audit", {}).get("model_order"),
                              "exported_model_kernel_compatibility": lv.get("exported_model_audit", {}).get("kernel_compatibility"),
                              "exported_model_heldout_validated": False})
        for i, f in enumerate(c["frequency_hz"]):
            eis.append({**common, "reconstruction_branch": "primary_rc", "frequency_hz": f,
                       **{key: c.get(key, [None] * len(c["frequency_hz"]))[i]
                       for key in ("acquisition_index", "zreal_measured", "neg_zimag_measured",
                                   "zreal_reconstructed", "neg_zimag_reconstructed", "residual_re_percent", "residual_im_percent")}})
        for p in r.get("peaks", []):
            peaks.append({**common, **p, "distribution_branch": "primary_rc"})
        for t, g in zip(c["tau_s"], c["gamma"]):
            primary.append({**common, "distribution_branch": "primary_rc", "tau_s": t, "gamma": g,
                            "distribution_measure": "gamma_per_ln_tau",
                            "lambda": r["drt"]["lambda"]})
        for case in c.get("sensitivity_curves", []):
            for f, g in zip(case["frequency_hz"], case["gamma"]):
                sensitivity.append({**common, "method": "primary", "distribution_branch": "primary_rc",
                                    "case": "lambda", "selected": case["selected"],
                                    "lambda": case["lambda"], "tau_s": 1 / (2 * math.pi * f), "gamma": g})
        sg = r.get("signed_gdrt", {})
        for t, g in zip(sg.get("tau_s", []), sg.get("gamma", [])):
            signed.append({**common, "tau_s": t, "signed_gamma": g, "distribution_measure": "signed_gamma_per_ln_tau"})
        lm = r.get("loewner_rc_rl", {})
        for branch in ("rc", "rl"):
            for t, g in zip(lm.get(f"tau_{branch}_s", []), lm.get(f"gamma_{branch}", [])):
                loewner.append({**common, "branch": branch.upper(), "tau_s": t, "discrete_resistance": g,
                                "measure": "discrete_branch_weight_not_density",
                                "model_identity": "legacy-full-data-matrix-rank-model",
                                "model_order": lm.get("model_order"), "heldout_validated": False,
                                "pole_audit_status": r.get("loewner_validation", {}).get("exported_model_audit", {}).get("status"),
                                "kernel_compatibility": r.get("loewner_validation", {}).get("exported_model_audit", {}).get("kernel_compatibility")})
        continuous = [("signed_gdrt", sg)] + [("ddt:" + d["boundary"], d) for d in r.get("ddt", {}).get("candidates", [])]
        for name, branch in predictive.items():
            identity = "predictive_" + name
            continuous.append((identity, branch))
            for row in branch.get("lambda_grid", []):
                predictive_scores.append({**common, "reconstruction_branch": identity, **row,
                    "selected_lambda": branch.get("lambda"), "prediction_is_independent_validation": False})
            for t, g in zip(branch.get("tau_s", []), branch.get("gamma", [])):
                predictive_curves.append({**common, "distribution_branch": identity, "tau_s": t, "gamma": g,
                    "selected_for_recommendation": chosen == identity, "selection_is_physical_acceptance": False,
                    "individual_peak_stability_assessed": False})
        for method, branch in continuous:
            for case in branch.get("sensitivity", {}).get("cases", []):
                for t, g in zip(case.get("tau_s", []), case.get("gamma", [])):
                    sensitivity.append({**common, "method": method, "case": case["case"], "lambda": case["lambda"],
                                        "tau_s": t, "gamma": g, "num_tau": case["num_tau"]})
            frequencies = branch.get("measured_frequency_hz", c["frequency_hz"])
            for f, z in zip(frequencies, complex_values(branch.get("reconstructed", []))):
                branch_reconstructions.append({**common, "method": method, "frequency_hz": f,
                                               "zreal_reconstructed": z.real, "neg_zimag_reconstructed": -z.imag})
        for d in r.get("ddt", {}).get("candidates", []):
            diffusion.append({**common, **{k: d.get(k) for k in ("boundary", "status", "distribution_identifiable",
                    "eligible_for_distribution_ranking", "residual_rms_percent", "rejection_reasons",
                    "sigma_impedance_per_sqrt_s", "nuisance", "solver", "support", "error")},
                    "topology": r["ddt"].get("topology"), "selected_best": d.get("boundary") == r["ddt"].get("best_boundary")})
            for t, g in zip(d.get("tau_s", []), d.get("gamma", [])):
                diffusion_curves.append({**common, "boundary": d["boundary"], "tau_s": t, "gamma": g,
                                         "eligible": d.get("eligible_for_distribution_ranking"), "distribution_measure": "gamma_per_ln_tau"})
        for candidate in r.get("model_reduce", {}).get("candidates", []):
            arrays = {key: complex_values(candidate.get(key, [])) for key in (
                "preprocessed_impedance", "reconstructed_impedance", "removed_component", "addback_reconstructed_impedance")}
            for i, f in enumerate(c["frequency_hz"]):
                row = {**common, "model": candidate.get("model"), "status": candidate.get("status"),
                       "accepted": candidate.get("accepted"), "frequency_hz": f}
                for key, values in arrays.items():
                    if i < len(values):
                        row[key + "_real"], row[key + "_imag"] = values[i].real, values[i].imag
                reduction.append(row)
        category = "scientific" if r.get("display_role") == "included" else "diagnostic"
        for name in r.get("figures", []) + r.get("advanced_figures", []):
            source = Path(name)
            if source.is_file():
                shutil.copy2(source, output / "05_figures" / category / source.name)
    for filename, rows in (
        ("00_overview/spectrum_index.csv", index), ("02_quality/impedance_reconstruction.csv", eis),
        ("02_quality/peaks_with_support.csv", peaks), ("02_quality/all_sensitivity_curves.csv", sensitivity),
        ("03_results/primary_drt/curves.csv", primary), ("03_results/signed_gdrt/curves.csv", signed),
        ("03_results/signed_gdrt/loewner_discrete.csv", loewner),
        ("03_results/diffusion_candidates/candidates.csv", diffusion),
        ("03_results/diffusion_candidates/curves.csv", diffusion_curves),
        ("03_results/model_reduce/before_after_addback.csv", reduction),
        ("02_quality/advanced_reconstructions.csv", branch_reconstructions),
        ("00_overview/model_recommendations.csv", recommendations),
        ("02_quality/peak_perturbation_evidence.csv", evidence_rows),
        ("02_quality/loewner_frequency_validation.csv", rational_rows),
        ("03_results/recommended/curves.csv", recommended_curves),
        ("03_results/recommended/reconstruction.csv", recommended_eis),
        ("03_results/predictive_rc/curves.csv", predictive_curves),
        ("03_results/predictive_rc/prediction_scores.csv", predictive_scores),
        ("04_plot_data/drt_long.csv", primary)):
        write_table(output / filename, rows)
    if (run.get("parameters", {}).get("reference_profile", "none") != "none"
            and run.get("parameters", {}).get("inductance_policy") == "off"):
        write_table(output / "03_results/reference_compatibility/curves.csv", primary)
    for name in run.get("overlay_figures", []):
        source = Path(name)
        if source.is_file():
            shutil.copy2(source, output / "05_figures/diagnostic" / source.name)
    _export_wide(output / "04_plot_data", results)
    (output / "00_overview/README.md").write_text(
        "# DRT results · schema 5\n\nStart with spectrum_index.csv: every spectrum has separate execution, numerical, "
        "experimental-evidence and display states. Missing controls do not prevent exploratory inspection but never count as acceptance.\n\n"
        "01_inputs contains normalized, not raw, data; immutable raw sources are referenced by path and SHA256. "
        "source_row_ledger.csv and source_audits.json retain physical text-line decisions, including excluded rows and parse failures. "
        "Raw text fields are untrusted source strings: import as text, not spreadsheet formulas. "
        "02_quality contains measurements/reconstructions, supported peak basins and sensitivity curves. "
        "03_results separates primary, signed and diffusion models. 04_plot_data provides long and group-specific Origin wide tables. "
        "05_figures separates accepted scientific from diagnostic figures; promotion remains a separate empty destination unless requested. "
        "06_reproducibility stores code/environment/configuration.\n\n"
        "Branch identity: reconstruction_branch and distribution_branch describe the arrays in that row. "
        "recommended_branch is separate per-spectrum recommendation metadata; it never relabels the primary arrays. "
        "Primary reconstruction, primary DRT/peaks/sensitivity, drt_long and Origin column mappings remain primary_rc, "
        "including any declared series-L nuisance in the primary fit. Paired Origin tables and exact-grid matrices "
        "retain their numeric columns; column_mapping.csv identifies their primary_rc distribution. "
        "Recommended tables additionally name selected_branch and selection_scope=reconstruction-only: selecting "
        "a useful reconstruction is not physical-model acceptance, and does not validate a distribution or mechanism. "
        "Read recommendation_status, numerical status and supplied-control evidence separately.\n\n"
        "The authoritative numerical records remain results/*.json; root CSV files are compatibility views. "
        "No semi-infinite tau distribution is exported. Finite diffusion results are conditional series-impedance models, not parallel-admittance DDT. "
        "No interpolation, duplicate averaging, reference retuning or mechanism assignment is performed by export.\n",
        encoding="utf-8")


def export_wide(folder, results):
    """Origin-ready paired X/Y columns; unequal tau grids remain independent."""
    from derived_outputs import publish_tree
    folder = Path(folder).resolve()
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".drt-wide-", dir=folder.parent) as temp:
        stage = Path(temp)
        _export_wide(stage, results)
        return publish_tree(stage, folder, "wide")


def _export_wide(folder, results):
    groups = defaultdict(list)
    for r in results:
        if r.get("status") != "completed":
            continue
        key = condition_key(r)
        groups[key].append(r)
    mapping = []
    for key, rows in groups.items():
        group_id = digest(key)[:12]
        fields, values = [], []
        for r in rows:
            fields.extend((r["spectrum_uid"] + "_tau_s", r["spectrum_uid"] + "_gamma"))
            mapping.append({"group_id": group_id, "group_fields": key, **common_fields(r),
                            "distribution_branch": "primary_rc",
                            "x_column": fields[-2], "y_column": fields[-1],
                            "row_order": "input order; voltage is a label, not implicit chronology"})
        for i in range(max(len(r["curves"]["tau_s"]) for r in rows)):
            row = {}
            for r in rows:
                if i < len(r["curves"]["tau_s"]):
                    row[r["spectrum_uid"] + "_tau_s"] = r["curves"]["tau_s"][i]
                    row[r["spectrum_uid"] + "_gamma"] = r["curves"]["gamma"][i]
            values.append(row)
        write_table(folder / f"origin_{group_id}.csv", values, fields)
        # Only an exact shared grid can be a matrix without a resampling decision.
        tau = np.asarray(rows[0]["curves"]["tau_s"])
        if all(np.array_equal(tau, np.asarray(r["curves"]["tau_s"])) for r in rows):
            matrix = [{"tau_s": t, **{r["spectrum_uid"]: r["curves"]["gamma"][i] for r in rows}}
                      for i, t in enumerate(tau)]
            write_table(folder / f"heatmap_exact_grid_{group_id}.csv", matrix)
    write_table(folder / "column_mapping.csv", mapping)

#!/usr/bin/env python3
"""Data-first EIS-only analysis. No DRT or physical circuit inversion is invoked."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Reuse only ingestion/serialization helpers, never batch_drt.main/analyze_payload.
from batch_drt import (DEFAULT_PARSER, TEXT_SUFFIXES, discover_inputs, json_ready,
                       load_manifest, load_parser_module, manifest_metadata,
                       prepare_payload, read_pyimpspec_source, sha256_file,
                       source_parser_options, validate_spectrum_parser_options,
                       PARSER_FIELDS, write_json, lag1_correlation)
from result_contract import (digest, execution_contract, identities, initialize_run,
                             inventory_inputs, normalized_input_rows, row_selected)
from metadata_contract import canonical_metadata
from derived_outputs import checked_path, publish_tree, refresh_owned_file

ROOT = Path(__file__).resolve().parents[1]
EIS_SCHEMA = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version=(ROOT / "VERSION").read_text().strip())
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--manifest-mode", choices=("allowlist", "metadata"), default="allowlist")
    p.add_argument("--parser-script", type=Path, default=DEFAULT_PARSER)
    p.add_argument("--imag-convention", choices=("auto", "neg-zimag", "zimag"), default="auto")
    p.add_argument("--frequency-unit", choices=("auto", "hz", "khz", "mhz"), default="auto")
    p.add_argument("--impedance-unit", choices=("auto", "ohm", "mohm", "kohm", "ohm-cm2"), default="auto")
    p.add_argument("--text-columns", help="JSON mapping; same contract as batch_drt")
    p.add_argument("--text-headerless", action="store_true")
    p.add_argument("--text-delimiter", default="auto")
    p.add_argument("--text-header-row", type=int)
    p.add_argument("--text-frequency-order", choices=("acquisition", "single-spectrum-descending"), default="acquisition")
    p.add_argument("--area-normalize", action="store_true")
    p.add_argument("--kk", choices=("off", "on"), default="off")
    p.add_argument("--zhit", choices=("off", "on"), default="off")
    p.add_argument("--check-timeout", type=int, default=120, help="Wall seconds per requested diagnostic")
    p.add_argument("--check-warn-percent", type=float, default=1.0)
    p.add_argument("--check-suspect-percent", type=float, default=3.0)
    p.add_argument("--plots", choices=("on", "off"), default="on")
    p.add_argument("--plot-dpi", type=int, default=600)
    p.add_argument("--display-unit", choices=("auto", "ohm", "mohm"), default="auto")
    p.add_argument("--render-timeout", type=int, default=180)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args(argv)
    if a.export_only and not a.resume:
        p.error("--export-only requires --resume")
    if a.check_timeout <= 0 or a.render_timeout <= 0 or not 72 <= a.plot_dpi <= 600:
        p.error("Timeouts must be positive; plot DPI must be 72..600")
    if not (0 < a.check_warn_percent < a.check_suspect_percent < float('inf')):
        p.error("Require 0 < check-warn-percent < check-suspect-percent < infinity")
    if a.text_header_row is not None and a.text_header_row < 1:
        p.error("text-header-row must be positive")
    a.analysis_mode = "eis-only"
    a.eis_schema_version = EIS_SCHEMA
    return a


def acquire(args, sources, exact, by_path, fingerprint, expected_hashes=None):
    parser = load_parser_module(args.parser_script.expanduser().resolve())
    defaults = argparse.Namespace(**{k: getattr(args, k) for k in PARSER_FIELDS})
    defaults.frequency_order = args.text_frequency_order
    audits, payloads, failures = [], [], []
    for source in sources:
        # All-or-nothing source preparation prevents a late manifest/area error
        # from silently accepting the first half of a multi-spectrum file.
        pending = []
        try:
            before_hash = sha256_file(source)
            if expected_hashes is not None and expected_hashes.get(str(source)) != before_hash:
                raise ValueError("Input changed after inventory; create a stable input snapshot and a fresh run")
            metadata = by_path.get(str(source), {})
            options = source_parser_options(defaults, metadata)
            label = metadata.get("label") or source.stem
            if source.suffix.lower() in TEXT_SUFFIXES:
                spectra, audit = parser.read_text(source, label, options)
            elif source.suffix.lower() == ".irf":
                spectra, audit = parser.read_irf(source, label, options.imag_convention)
            else:
                spectra, audit = read_pyimpspec_source(source, label, options, parser)
            audits.append(audit)
            source_hash = audit.get("sha256") or sha256_file(source)
            if source_hash != before_hash or sha256_file(source) != before_hash:
                raise ValueError("Input changed while parsing; results from this source were not accepted")
            requested = {sid for (path, sid), row in exact.items() if path == str(source) and row_selected(row)}
            missing = requested - {str(s.spectrum_id) for s in spectra}
            if missing:
                raise ValueError(f"Selected spectrum IDs not found: {sorted(missing)}")
            audit.update(parsed_spectrum_count=len(spectra), selected_spectrum_count=0, spectrum_exclusions=[])
            for spec in spectra:
                meta = manifest_metadata(source, spec.spectrum_id, exact, by_path)
                if (args.manifest and args.manifest_mode == "allowlist" and not meta) or not row_selected(meta):
                    audit["spectrum_exclusions"].append({"spectrum_id": spec.spectrum_id,
                        "reason": meta.get("exclusion_reason") or meta.get("input_role") or "not-selected-by-manifest"})
                    continue
                meta = canonical_metadata(meta)  # Reject conflicting aliases before checkpointing.
                validate_spectrum_parser_options(options, exact.get((str(source), str(spec.spectrum_id)), {}))
                payload = prepare_payload(spec, source_hash, parser.qc_for(spec), meta, args.area_normalize)
                payload.update(run_fingerprint=fingerprint, result_slug=payload["spectrum_uid"],
                    input_metadata=meta,
                    parser_options={**{k: getattr(options, k) for k in PARSER_FIELDS},
                                    "text_frequency_order": options.frequency_order})
                pending.append(payload)
                base_result(payload)  # Validate converted arrays before accepting this source.
            audit["selected_spectrum_count"] = len(pending)
            payloads.extend(pending)
        except Exception as exc:
            failed_audit = getattr(exc, "audit", None)
            if isinstance(failed_audit, dict) and not any(a is failed_audit for a in audits):
                audits.append(failed_audit)
            for a in audits:
                if a.get("source") == str(source):
                    a.update(selected_spectrum_count=0, source_preparation_failed=True)
            sha = (expected_hashes or {}).get(str(source)) or (sha256_file(source) if source.is_file() else None)
            failures.append({**identities(source, sha, "__parse_failure__"),
                "source": str(source), "source_sha256": sha, "label": source.stem,
                "analysis_mode": "eis-only", "status": "failed", "numeric_status": "failed",
                "error": f"{type(exc).__name__}: {exc}"})
    return payloads, audits, failures


def base_result(payload):
    r = json_ready(payload)
    f, real, neg_imag = [np.asarray(payload[k], float) for k in ("frequency_hz", "zreal", "neg_zimag")]
    if any(v.ndim != 1 for v in (f, real, neg_imag)) or not (len(f) == len(real) == len(neg_imag) > 0):
        raise ValueError("Empty or mismatched EIS arrays")
    if not all(np.all(np.isfinite(v)) for v in (f, real, neg_imag)) or np.any(f <= 0):
        raise ValueError("EIS arrays require finite values and positive frequencies")
    magnitude = np.hypot(real, neg_imag)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("Impedance modulus overflow; inspect physical units and input values")
    phase = np.where(magnitude > 0, np.degrees(np.arctan2(-neg_imag, real)), np.nan)
    r.update(analysis_mode="eis-only", eis_schema_version=EIS_SCHEMA, status="completed",
        numeric_status="parsed-not-screened", evidence_status="not-assessed",
        validity_status="exploratory-no-experimental-acceptance", display_role="included-exploratory",
        drt_status="not-requested", circuit_fit_status="not-requested",
        controls={"repeat": "not-assessed", "amplitude_linearity": "not-assessed", "rest_stability": "not-assessed"},
        quantities=json_ready({"modulus": magnitude, "zimag": -neg_imag, "phase_deg": phase,
                               "phase_defined": magnitude > 0}),
        kk={"status": "not-requested"}, zhit={"status": "not-requested"})
    if np.any(magnitude == 0):
        r["warnings"].append("Zero impedance: phase undefined; zero modulus retained, omitted only from log-magnitude panel")
    if len(np.unique(f)) != len(f):
        r["warnings"].append("Duplicate frequencies retained; automatic averaging and consistency diagnostics disabled")
    if r["qc"].get("status") != "pass" or r["qc"].get("excluded_by_reason"):
        r["warnings"].append("Parser/QC findings require review; inspect the complete row ledger")
    qc = r["qc"]
    review = bool(qc.get("status") != "pass" or qc.get("excluded_by_reason")
                  or qc.get("duplicate_frequency_count") or not qc.get("row_accounting_matches_raw", False)
                  or r["metadata"].get("table_excluded_by_reason")
                  or qc.get("time_monotonic_increasing") is False or qc.get("time_step_anomaly_count", 0))
    r["input_quality_status"] = "review-required" if review else "screen-pass"
    if review:
        r["numeric_status"] = "review-required"
    r["claim_boundary"] = "EIS-only exports do not certify DRT suitability, linearity, stationarity or physical mechanisms."
    return r


def calculate_check(result, method, warn, suspect):
    from pyimpspec import DataSet, perform_kramers_kronig_test, perform_zhit
    f = np.asarray(result["frequency_hz"], float)
    z = np.asarray(result["zreal"]) - 1j * np.asarray(result["neg_zimag"])
    if len(f) < 10 or len(np.unique(f)) != len(f) or np.any(np.abs(z) == 0):
        return {"status": "not-assessable", "reason": "Requires >=10 distinct frequencies and nonzero impedance"}
    order = np.argsort(-f, kind="stable")
    data = DataSet(f[order], z[order], label=result["label"])
    if method == "kk":
        fit = perform_kramers_kronig_test(data, test="complex", admittance=None,
            add_capacitance=True, add_inductance=True, num_procs=1)
        extra = {"test": "complex linear KK", "representation": "admittance" if fit.was_tested_on_admittance() else "impedance",
                 "num_RC": int(fit.get_num_RC()), "log_F_ext": float(fit.get_log_F_ext()),
                 "add_inductance": True, "add_capacitance": True}
    elif method == "zhit":
        fit = perform_zhit(data, num_procs=1)
        extra = {"smoothing": str(fit.smoothing), "interpolation": str(fit.interpolation), "window": str(fit.window)}
    else:
        raise ValueError("Unknown EIS check")
    # Assert alignment explicitly; never attach a sorted reconstruction to source order.
    ff = np.asarray(fit.get_frequencies(), float)
    zz = np.asarray(fit.get_impedances(), complex)
    if not np.array_equal(ff, f[order]) or zz.shape != z.shape or not np.all(np.isfinite(zz)):
        raise ValueError("Diagnostic reconstruction frequencies or values are not aligned/finite")
    residual = (zz - z[order]) / np.abs(z[order]) * 100.0
    re_rms = float(np.sqrt(np.mean(residual.real ** 2)))
    im_rms = float(np.sqrt(np.mean(residual.imag ** 2)))
    maximum = max(re_rms, im_rms)
    gate = "screen-pass" if maximum <= warn else "warn" if maximum <= suspect else "suspect"
    return json_ready({"status": "completed", "screen_status": gate, **extra,
        "source_point_count": len(f), "screen_point_count": len(ff), "downsampled_for_screen": False,
        "source_point_indices": order + 1, "frequency_hz": ff, "zreal": zz.real, "neg_zimag": -zz.imag,
        "residual_re_percent": residual.real, "residual_im_percent": residual.imag,
        "residual_definition": "100*(reconstruction-measured)/abs(measured), complex components",
        "residual_rms_re_percent": re_rms, "residual_rms_im_percent": im_rms,
        "residual_lag1_re": lag1_correlation(residual.real), "residual_lag1_im": lag1_correlation(residual.imag),
        "thresholds_percent": {"warn": warn, "suspect": suspect},
        "claim_boundary": "Numerical consistency screen only; residual structure and experimental controls remain to be assessed. No raw data correction."})


def check_worker(result, method, warn, suspect, path):
    try:
        check = calculate_check(result, method, warn, suspect)
    except Exception as exc:
        check = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(Path(path), check)


def isolated_check(result, method, args):
    with tempfile.TemporaryDirectory(prefix="eis-check-") as temp:
        path = Path(temp) / "check.json"
        worker = mp.get_context("spawn").Process(target=check_worker,
            args=(result, method, args.check_warn_percent, args.check_suspect_percent, str(path)))
        worker.start()
        worker.join(args.check_timeout)
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
            if worker.is_alive():
                worker.kill()
                worker.join()
            return {"status": "timeout", "timeout_s": args.check_timeout}
        if worker.exitcode != 0 or not path.is_file():
            return {"status": "failed", "error": f"Diagnostic worker exited {worker.exitcode}"}
        return json.loads(path.read_text())


def result_digest(result):
    return digest(json_ready({k: v for k, v in result.items() if k != "eis_sha256"}))


def complete_checks(result, args, *, retry=False):
    """Retry only transient failures on ordinary resume; never refit successful checks."""
    for method in ("kk", "zhit"):
        if getattr(args, method) != "on":
            continue
        prior = result.get(method, {})
        if retry and prior.get("status") not in {"failed", "timeout"}:
            continue
        if retry:
            result.setdefault("diagnostic_attempt_history", []).append({
                "method": method, "previous_result": prior,
                "retried_at": datetime.now(timezone.utc).isoformat()})
        print(f"[checking {method}] {result['label']}", flush=True)
        result[method] = isolated_check(result, method, args)
    requested = [result[k] for k in ("kk", "zhit") if getattr(args, k) == "on"]
    if requested:
        result["numeric_status"] = "consistency-screen-pass" if result["input_quality_status"] == "screen-pass" and all(
            c.get("screen_status") == "screen-pass" for c in requested) else "review-required"
    result["eis_sha256"] = result_digest(result)


def csv_table(path, rows, fields=()):
    """Spreadsheet-safe text views; exact strings remain in authoritative JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys([*fields, *(k for row in rows for k in row)])) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = json_ready(row)
            for key, value in clean.items():
                if isinstance(value, (dict, list)):
                    clean[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    clean[key] = "'" + value
            writer.writerow(clean)


def export_data(output, results, payloads, audits, inventory, contract, args):
    run = {"analysis_mode": "eis-only", "eis_schema_version": EIS_SCHEMA,
           "generated_at": datetime.now(timezone.utc).isoformat(), "run_fingerprint": contract["run_fingerprint"],
           "completed": sum(r["status"] == "completed" for r in results),
           "failed": sum(r["status"] != "completed" for r in results),
           "drt_computed": False, "physical_circuit_fit_computed": False,
           "experimental_acceptance": "not-assessed", "raw_data_modified": False}
    with tempfile.TemporaryDirectory(prefix=".eis-export-", dir=output.parent) as temp:
        stage = Path(temp)
        csv_table(stage / "00_overview/input_inventory.csv", inventory)
        index = [{k: r.get(k) for k in ("spectrum_uid", "source_uid", "spectrum_id", "label", "source", "source_sha256",
                   "status", "numeric_status", "input_quality_status", "validity_status", "impedance_basis", "error")} |
                 {"kk_status": r.get("kk", {}).get("status"), "kk_screen": r.get("kk", {}).get("screen_status"),
                  "zhit_status": r.get("zhit", {}).get("status"), "zhit_screen": r.get("zhit", {}).get("screen_status"),
                  "points": len(r.get("frequency_hz", [])), "drt_status": "not-requested"} for r in results]
        csv_table(stage / "00_overview/spectrum_index.csv", index)
        write_json(stage / "00_overview/run_manifest.json", run)
        write_json(stage / "01_inputs/source_audits.json", audits)
        ledger = [{"source": a["source"], "source_sha256": a["sha256"],
                   "source_uid": identities(a["source"], a["sha256"], "")["source_uid"], **row}
                  for a in audits for row in a.get("row_ledger", [])]
        csv_table(stage / "01_inputs/source_row_ledger.csv", ledger)
        normalized = normalized_input_rows(payloads, results)
        by_uid = {r["spectrum_uid"]: r for r in results}
        for row in normalized:
            q = by_uid[row["spectrum_uid"]]["quantities"]
            i = row["included_point_index"] - 1
            row.update({key: value[i] for key, value in q.items()})
        csv_table(stage / "01_inputs/normalized_eis.csv", normalized)
        csv_table(stage / "03_results/eis_quantities.csv", normalized)
        checks = []
        for r in results:
            for method in ("kk", "zhit"):
                c = r.get(method, {})
                for i, f in enumerate(c.get("frequency_hz", [])):
                    checks.append({"spectrum_uid": r["spectrum_uid"], "method": method, "frequency_hz": f,
                        "impedance_basis": r["impedance_basis"], "included_point_index": c["source_point_indices"][i],
                        **{k: c[k][i] for k in ("zreal", "neg_zimag", "residual_re_percent", "residual_im_percent")}})
        csv_table(stage / "02_quality/consistency_reconstructions.csv", checks)
        write_json(stage / "02_quality/eis_quality.json", results)
        csv_table(stage / "04_plot_data/eis_long.csv", normalized)
        for p in payloads:
            uid = p["spectrum_uid"]
            rows = [{k: v for k, v in row.items() if k in ("frequency_hz", "zreal", "neg_zimag", "modulus", "phase_deg",
                       "included_point_index", "source_line", "impedance_basis")} for row in normalized if row["spectrum_uid"] == uid]
            csv_table(stage / f"04_plot_data/origin_{uid}.csv", rows)
        # A manifest of raw sources, not calculated EIS tables, for a later DRT run.
        transfer = []
        for source in dict.fromkeys(p["source"] for p in payloads):
            group = [p for p in payloads if p["source"] == source]
            transfer.append({"path": source, "spectrum_id": "", "include": False,
                             "exclusion_reason": "Selection restricted to explicit spectrum rows below",
                             **group[0]["parser_options"]})
            for p in group:
                transfer.append({**p["input_metadata"], "path": source, "spectrum_id": p["spectrum_id"],
                                 "input_role": "eis", "include": True, "label": p["label"]})
        # Manifest is machine input: no formula escaping, exact metadata. It must
        # not be opened as an executable spreadsheet; README states this explicitly.
        manifest = stage / "06_reproducibility/drt_input_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(["path", "spectrum_id", *(k for r in transfer for k in r)]))
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for r in transfer:
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in json_ready(r).items()})
        upgrade = {"requires_user_request": True, "inputs": list(dict.fromkeys(p["source"] for p in payloads)),
                   "manifest": str(output / manifest.relative_to(stage)), "area_normalize": args.area_normalize,
                   "parser_script": str(args.parser_script.expanduser().resolve()),
                   "parser_sha256": sha256_file(args.parser_script.expanduser().resolve()),
                   "output_guidance": "Use a separate sibling drt directory. Original raw sources must still exist.",
                   "warning": "Not a DRT acceptance allowlist; all selected spectra need fresh DRT assessment."}
        write_json(stage / "06_reproducibility/drt_handoff.json", upgrade)
        write_json(stage / "06_reproducibility/execution_contract.json", contract)
        for i, (path, sha) in enumerate(contract["code_sha256"].items()):
            if sha256_file(Path(path)) != sha:
                raise ValueError("Code changed during run; refusing inconsistent snapshot")
            target = stage / "06_reproducibility/code" / f"{i:03d}_{Path(path).name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (stage / "00_overview/README.md").write_text(
            "# EIS-only results\n\n"
            f"Parsed: {run['completed']}; failed: {run['failed']}. No DRT or physical circuit fitting was performed.\n\n"
            "Data first: 01_inputs preserves normalized observations, source hashes and row ledgers; "
            "03_results contains modulus and signed phase. 02_quality contains optional consistency reconstructions, "
            "never replacements for raw points. 04_plot_data contains long and per-spectrum Origin tables. "
            "05_figures/draft holds raw Nyquist/Bode figures; inspect plot_report.json separately.\n\n"
            "CSV text beginning with formula characters is apostrophe-escaped. Exact text remains in JSON. "
            "The DRT handoff manifest preserves exact machine-readable strings: import it as text, not as formulas.\n\n"
            "Use current indexes, not recursive searches through _superseded. Zero exits indicate processing, "
            "not experimental acceptance. Missing controls and structured residuals still require review. "
            "Do not delete inductive points or infer mechanisms from consistency checks.\n\n"
            "A later DRT request uses the original sources and 06_reproducibility/drt_input_manifest.csv "
            "in a fresh sibling drt output; retain area-normalize when recorded in drt_handoff.json. "
            "Do not feed normalized export tables back through the raw importer.\n", encoding="utf-8")
        publish_tree(stage, output, "eis_export")
    return run


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    sources = discover_inputs(args.inputs, output)
    exact, by_path = load_manifest(args.manifest)
    sources, inventory = inventory_inputs(sources, exact, by_path, manifest=args.manifest, manifest_mode=args.manifest_mode)
    contract = execution_contract(args, inventory, Path(__file__).parent)
    contract.update(analysis_mode="eis-only", eis_schema_version=EIS_SCHEMA)
    contract["run_fingerprint"] = digest({k: v for k, v in contract.items() if k != "run_fingerprint"})
    # Independent mode and digest prevent EIS checkpoints being resumed as DRT.
    initialize_run(output, contract, args.resume)
    lock = checked_path(output, ".eis-run.lock")
    with lock.open("x") as handle:
        handle.write("EIS run active; do not start a second writer.\n")
    run = {"analysis_mode": "eis-only", "run_fingerprint": contract["run_fingerprint"],
           "status": "running", "export_status": "pending", "figure_status": "pending"}
    try:
        write_json(checked_path(output, "run_manifest.json"), run)
        payloads, audits, failures = acquire(args, sources, exact, by_path, contract["run_fingerprint"],
            expected_hashes={r["source"]: r["source_sha256"] for r in inventory if r["selected"]})
        results = list(failures)
        for payload in payloads:
            path = checked_path(output, f"results/{payload['spectrum_uid']}.json")
            if args.resume and path.is_file():
                r = json.loads(path.read_text())
                if r.get("analysis_mode") != "eis-only" or r.get("run_fingerprint") != contract["run_fingerprint"] or r.get("eis_sha256") != result_digest(r):
                    raise ValueError("EIS checkpoint changed or belongs to a different run")
                if not args.export_only and any(r.get(k, {}).get("status") in {"failed", "timeout"}
                                               for k in ("kk", "zhit") if getattr(args, k) == "on"):
                    complete_checks(r, args, retry=True)
                    write_json(path, r)
            else:
                if args.export_only:
                    raise ValueError("Export-only requires all completed EIS checkpoints")
                r = base_result(payload)
                complete_checks(r, args)
                write_json(path, r)  # Checkpoint precedes tables and figures.
            results.append(r)
            print(f"[EIS exported] {r['label']}: {len(r['frequency_hz'])} points", flush=True)
        try:
            run = export_data(output, results, payloads, audits, inventory, contract, args)
        except Exception as exc:
            run.update(status="failed", export_status="failed", export_error=f"{type(exc).__name__}: {exc}")
            write_json(checked_path(output, "run_manifest.json"), run)
            raise
        if args.plots == "on":
            try:
                from eis_plotting import render_isolated
                report = render_isolated(output, [r for r in results if r["status"] == "completed"], args)
            except Exception as exc:
                report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        else:
            report = {"status": "not-requested"}
        run["figure_status"] = report["status"]
        run["figure_report"] = report
        run["export_status"] = "completed"
        failed_checks = any(r.get(k, {}).get("status") in {"failed", "timeout", "not-assessable"}
                            for r in results for k in ("kk", "zhit"))
        failed = bool(failures or not payloads or failed_checks or report["status"] == "failed")
        run["status"] = "failed" if not payloads else "partial" if failed else "completed"
        write_json(checked_path(output, "run_manifest.json"), run)
        write_json(checked_path(output, "00_overview/run_manifest.json"), run)
        refresh_owned_file(output, "eis_export", "00_overview/run_manifest.json")
        return 2 if not payloads or failed and not args.allow_partial else 0
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(checked_path(output, "run_manifest.json"), run)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

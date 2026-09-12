#!/usr/bin/env python3
"""Real CLI checks: manifest selection, same-name IDs, exports and safe resume."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from advanced_smoke_test import write_spectrum
from smoke_test import synthetic_impedance


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def check_owned_export(output):
    manifest = json.loads((output / "derived_export_manifest.json").read_text())
    assert manifest["owner"] == "export"
    for name, expected_hash in manifest["files"].items():
        path = output / name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, name
    return manifest


def run_checks(root):
    script = Path(__file__).with_name("batch_drt.py")
    source_root = root / "input"
    source_root.mkdir(parents=True, exist_ok=True)
    raw = [source_root / "cell_A" / "eis.csv", source_root / "cell_B" / "eis.csv"]
    f = np.logspace(5,-2,61)
    z = synthetic_impedance(f)
    for path in raw:
        path.parent.mkdir(exist_ok=True)
        write_spectrum(path,f,z)
    ocp, reference, unlisted = (source_root/n for n in ("ocp.csv","reference.csv","unlisted.csv"))
    ocp.write_text("Time (s),Voltage (V)\n0,3.1\n1,3.1\n")
    reference.write_text("tau_s,gamma\n.01,3\n.1,5\n")
    unlisted.write_text("not an EIS table; must not be parsed\n")
    original_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw+[ocp,reference,unlisted]}
    manifest = source_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=["path","label","cell_id","voltage_v","direction","include","input_role","exclusion_reason"])
        writer.writeheader()
        for i,path in enumerate(raw):
            writer.writerow({"path":str(path),"label":"identical display label","cell_id":f"C{i}",
                             "voltage_v":"3.100","direction":"charge","include":"true","input_role":"eis"})
        writer.writerow({"path":str(ocp),"include":"false","input_role":"ocp","exclusion_reason":"potential-time control, not EIS"})
    output = root / "output"
    command=[sys.executable,str(script),str(source_root),"--manifest",str(manifest),"--output-dir",str(output),
             "--lambda-policy","fixed","--fixed-lambda","1e-4","--lambda-points","7", "--no-credible-intervals",
             "--ddt","off","--signed-gdrt","off","--no-loewner","--solver-timeout","30","--spectrum-timeout","180"]
    logs=[]
    def execute(extra=(), expected=0):
        result=subprocess.run(command+list(extra),capture_output=True,text=True,timeout=420)
        logs.append({"extra":list(extra),"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr})
        assert (result.returncode == 0) == (expected == 0), logs[-1]
        return result
    execute()
    manifest_result=json.loads((output/"run_manifest.json").read_text())
    assert manifest_result["schema_version"] == 5
    assert manifest_result["export_status"] == "completed"
    assert manifest_result["spectrum_count"] == 2 and not manifest_result["parse_failures"]
    owned = check_owned_export(output)
    inventory_name = "00_overview/input_inventory.csv"
    assert inventory_name in owned["files"]
    inventory=table(output/"00_overview/input_inventory.csv")
    assert sum(r["selected"] == "True" for r in inventory) == 2
    index=table(output/"00_overview/spectrum_index.csv")
    assert len(index)==2 and len({r["spectrum_uid"] for r in index})==2
    assert len({r["source_sha256"] for r in index})==1
    assert all(r["evidence_status"] == "evidence-missing" for r in index)
    for file in ("batch_summary.csv","peaks.csv","drt_curves.csv","02_quality/impedance_reconstruction.csv",
                 "03_results/primary_drt/curves.csv","04_plot_data/drt_long.csv"):
        rows=table(output/file)
        assert rows and all(r.get("spectrum_uid") for r in rows),file
        assert {r["spectrum_uid"] for r in rows} == {r["spectrum_uid"] for r in index},file
    peak_rows=table(output/"02_quality/peaks_with_support.csv")
    assert any("basin" in k and "tau" in k for k in peak_rows[0])
    normalized=table(output/"01_inputs/normalized_eis.csv")
    assert len(normalized)==122
    for uid in {r["spectrum_uid"] for r in normalized}:
        assert len({r["acquisition_index"] for r in normalized if r["spectrum_uid"]==uid})==61
    result_files=list((output/"results").glob("*.json"))
    numeric_before={p.name:json.loads(p.read_text())["curves"] for p in result_files}
    figure_times={p: p.stat().st_mtime_ns for p in (output/"figures").glob("*_DRT_audit.png")}
    assert len(figure_times)==2
    # Owned edits must survive recoverably; unrelated user files are never claimed.
    inventory_path = output / inventory_name
    edited_inventory = inventory_path.read_bytes() + b"\n# synthetic user annotation\n"
    inventory_path.write_bytes(edited_inventory)
    unknown_files = {name: b"synthetic user file: preserve exactly\n" for name in (
        "00_overview/user_notes.txt", "01_inputs/user_raw.csv",
        "05_figures/scientific/user_notes.txt")}
    for name, content in unknown_files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    execute(["--resume"])
    assert all(p.stat().st_mtime_ns==timestamp for p,timestamp in figure_times.items())
    owned = check_owned_export(output)
    archive = output / owned["superseded_archive"]
    assert (archive / "files" / inventory_name).read_bytes() == edited_inventory
    archived = json.loads((archive / "archive_manifest.json").read_text())
    assert archived["files"][inventory_name]["modified_since_export"] is True
    assert archived["files"][inventory_name]["sha256"] == hashlib.sha256(edited_inventory).hexdigest()
    assert "superseded" in archived["reason"]
    for name, content in unknown_files.items():
        assert (output / name).read_bytes() == content
        assert name not in owned["files"] and name not in archived["files"]

    # Simulate a missing ownership entry: a classified destination now belongs to
    # an unknown writer. The actual CLI must stop without touching that file or
    # replacing any previously published classified assets.
    ownership_path = output / "derived_export_manifest.json"
    previous_ownership = ownership_path.read_bytes()
    previous_inventory = inventory_path.read_bytes()
    unowned = json.loads(previous_ownership)
    del unowned["files"][inventory_name]
    ownership_path.write_text(json.dumps(unowned), encoding="utf-8")
    conflict_content = b"unowned synthetic inventory; must never be overwritten\n"
    inventory_path.write_bytes(conflict_content)
    classified_before = {name: (output / name).read_bytes() for name in unowned["files"]}
    blocked = execute(["--resume", "--export-only"], expected=1)
    assert "Refusing to overwrite an unowned file" in blocked.stderr
    assert inventory_path.read_bytes() == conflict_content
    assert json.loads(ownership_path.read_text()) == unowned
    assert all((output / name).read_bytes() == content for name, content in classified_before.items())
    assert not (output / ".drt-export.lock").exists()
    # Restore only this test fixture, then verify a normal CLI recovery succeeds.
    inventory_path.write_bytes(previous_inventory)
    ownership_path.write_bytes(previous_ownership)
    execute(["--resume","--export-only"])
    check_owned_export(output)
    for path in result_files:
        result=json.loads(path.read_text())
        assert result["curves"]==numeric_before[path.name]
        assert result["figure_status"]=="completed",result.get("figure_errors")
    blocked=execute(["--resume","--fixed-lambda","0.0002"],expected=1)
    assert "Resume refused" in blocked.stderr
    # This is an isolated synthetic file; alteration is an intentional negative test.
    original_bytes=raw[0].read_bytes(); raw[0].write_bytes(original_bytes+b"\n")
    blocked=execute(["--resume"],expected=1)
    assert "Resume refused" in blocked.stderr
    raw[0].write_bytes(original_bytes)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in original_hashes.items())
    assert all((output / name).read_bytes() == content for name, content in unknown_files.items())
    (root/"integration_log.json").write_text(json.dumps({"status":"PASS","checks":[
        "manifest allowlist", "old exports and OCP not parsed", "same-name/same-content unique IDs",
        "complete classified exports", "raw order preserved", "resume without solving or rerendering",
        "export-only exact numerical identity", "parameter mismatch rejected", "input mismatch rejected",
        "synthetic inputs preserved", "first CLI publication owns inventory and all published hashes",
        "resume archives user-modified owned inventory with matching provenance",
        "unknown user files preserved and never claimed", "unowned CLI collision is non-destructive",
        "normal export-only recovers after ownership conflict"],"commands":logs},ensure_ascii=False,indent=2))
    print(f"PASS: 15 CLI/provenance/export/resume checks; {root}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",type=Path)
    args=parser.parse_args()
    if args.output_root:
        root=args.output_root.expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Integration output must be fresh")
        root.mkdir(parents=True,exist_ok=True)
        run_checks(root)
    else:
        with tempfile.TemporaryDirectory(prefix="drt-integration-") as temp:
            run_checks(Path(temp).resolve())

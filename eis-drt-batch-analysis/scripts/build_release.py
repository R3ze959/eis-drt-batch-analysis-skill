#!/usr/bin/env python3
"""Build a deterministic, allowlisted source ZIP; never publish or upload."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile
from urllib.parse import unquote_to_bytes

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def privacy_findings(data):
    """Generic release guard; no private project-name list is embedded in source.

    This heuristic complements allowlisting and manual review, not guaranteed
    anonymization of arbitrary datasets or third-party author attribution.
    """
    decoded = data
    for _ in range(2):
        decoded = unquote_to_bytes(decoded)
    decoded = re.sub(rb"\\u00(2[fF]|5[cC])",
                     lambda m: bytes([int(m[1], 16)]), decoded)
    low = decoded.lower().replace(b"\\\\", b"\\")
    needles = [b"/" + segment for segment in
               (b"users/", b"home/", b"volumes/", b"media/", b"mnt/",
                b"private/" + b"var/" + b"folders/", b"var/" + b"folders/")]
    needles += [b"wx" + b"id_", b"xwechat" + b"_files"]
    found = ["private-case-or-machine-path" for needle in needles if needle in low]
    if re.search(rb"[a-z]:[\\/](?:us" + rb"ers|documents and settings)[\\/]", low):
        found.append("private-case-or-machine-path")
    if re.search(rb"(?:sk" + rb"-|gh" + rb"p_|github_" + rb"pat_)[A-Za-z0-9_-]{20,}", decoded):
        found.append("possible-secret")
    if re.search(rb"(?:AK" + rb"IA|AS" + rb"IA)[A-Z0-9]{16}", decoded):
        found.append("possible-secret")
    if re.search(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE " + rb"KEY-----", decoded):
        found.append("possible-secret")
    if re.search(rb"https?://[^\s/@:]+:[^\s/@]+@", low):
        found.append("credential-bearing-url")
    return sorted(set(found))


def safe_member(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(ord(c) < 32 for c in name) or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or str(PurePosixPath(name)) != name):
        raise ValueError("Unsafe release member path: " + repr(name))
    if privacy_findings(name.encode("utf-8")):
        raise ValueError("Private release member path")
    return name


def collect(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/"release_manifest.json").read_text(encoding="utf-8"))
    paths = manifest["files"]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate release member")
    result = {}
    for name in paths:
        safe_member(name)
        p = root / name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError("Missing/unsafe release member: " + name)
        if any(part.startswith(".") or part == ".." for part in Path(name).parts):
            raise ValueError("Hidden/traversing release member: " + name)
        data = p.read_bytes()
        findings = privacy_findings(data)
        if findings:
            raise ValueError(f"Privacy scan rejected {name}: {findings}")
        result[name] = data
    return result


def verify_archive(path, root=ROOT):
    # The trusted local source allowlist is independent of the ZIP's own claims.
    root = Path(root)
    required = {"VERSION", "SKILL.md", "release_manifest.json"}
    trusted = set(json.loads((root / "release_manifest.json").read_text())["files"])
    if not required <= trusted:
        raise ValueError("Local release allowlist lacks required metadata")
    for name in trusted:
        safe_member(name)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP names")
        for entry in z.infolist():
            safe_member(entry.filename)
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or kind not in (0, stat.S_IFREG):
                raise ValueError("ZIP members must be regular files: " + entry.filename)
        prefix = "eis-drt-batch-analysis/"
        if set(names) != {prefix+n for n in trusted | {"FILE_MANIFEST.json"}}:
            raise ValueError("ZIP does not match the local release allowlist")
        raw_index = z.read(prefix+"FILE_MANIFEST.json")
        if privacy_findings(raw_index):
            raise ValueError("Private archive index")
        index = json.loads(raw_index)
        if set(index.get("sha256", {})) != trusted:
            raise ValueError("ZIP hash index does not match local allowlist")
        if index.get("version") != (root/"VERSION").read_text().strip():
            raise ValueError("ZIP release version mismatch")
        if z.read(prefix+"VERSION").decode().strip() != index["version"]:
            raise ValueError("ZIP VERSION disagrees with index")
        embedded_manifest = json.loads(z.read(prefix+"release_manifest.json"))
        if set(embedded_manifest.get("files", [])) != trusted:
            raise ValueError("Embedded release allowlist mismatch")
        expected = {f"eis-drt-batch-analysis/{n}" for n in index["sha256"]}
        if set(names) != expected | {"eis-drt-batch-analysis/FILE_MANIFEST.json"}:
            raise ValueError("ZIP contains unexpected members")
        if z.testzip():
            raise ValueError("ZIP CRC verification failed")
        for name, digest in index["sha256"].items():
            safe_member(name)
            data = z.read("eis-drt-batch-analysis/" + name)
            if sha(data) != digest or privacy_findings(data):
                raise ValueError("Hash/privacy failure: " + name)
    return {"status": "PASS", "files": len(expected), "archive_sha256": sha(Path(path).read_bytes()),
            "verification_scope": "Integrity, local release allowlist, regular files and privacy; not publisher authentication."}


def build(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    files = collect(root)
    version = files["VERSION"].decode().strip()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"eis-drt-batch-analysis-{version}.zip"
    if archive.exists():
        raise FileExistsError("Archive already exists; choose a new output directory")
    index = {"version": version, "license": "GPL-3.0-or-later",
             "source_only": True, "synthetic_examples_only": True,
             "sha256": {name: sha(data) for name, data in sorted(files.items())}}
    files["FILE_MANIFEST.json"] = (json.dumps(index, indent=2, sort_keys=True)+"\n").encode()
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(files.items()):
            info=zipfile.ZipInfo("eis-drt-batch-analysis/"+name, date_time=(2026,9,8,0,0,0))
            info.create_system=3
            info.external_attr=(0o100755 if name.endswith(".sh") else 0o100644) << 16
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    report=verify_archive(archive, root)
    (output/"release_check.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/(archive.name+".sha256")).write_text(report["archive_sha256"]+"  "+archive.name+"\n",encoding="utf-8")
    print(json.dumps({"archive": str(archive), **report},indent=2))
    return archive


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT.parent/"dist")
    p.add_argument("--check",action="store_true",help="Check the source allowlist without building")
    p.add_argument("--verify",type=Path,help="Verify an existing archive")
    a=p.parse_args()
    if a.verify:
        print(json.dumps(verify_archive(a.verify),indent=2))
    elif a.check:
        print(f"PASS: {len(collect())} allowlisted files; privacy scan clean")
    else:
        build(a.output_dir)

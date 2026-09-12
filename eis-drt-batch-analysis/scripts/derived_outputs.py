"""Replace owned derived files only; retain superseded versions recoverably.

Numerical checkpoints and raw inputs are never owned by this publisher. Unknown
files are neither removed nor overwritten. Staging completes before replacement.
"""
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import uuid


def checked_path(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or name.startswith("/") or any(p in {"", ".", ".."} for p in name.split("/"))
            or str(PurePosixPath(name)) != name):
        raise ValueError("Unsafe managed-output name: " + repr(name))
    p = root
    for part in name.split("/"):
        p = p / part
        if p.is_symlink():
            raise ValueError("Refusing a symlink in managed output: " + name)
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError("Managed output escapes its root")
    return p


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def owned_name(owner, name):
    if owner == "eis_export":
        return name.startswith(("00_overview/", "01_inputs/", "02_quality/", "03_results/",
                                "04_plot_data/", "06_reproducibility/"))
    if owner == "eis_plot":
        return name in {"plot_report.json", "battery_origin.mplstyle"} or bool(
            re.fullmatch(r"[a-f0-9]{24}_(?:nyquist|bode_modulus|bode_phase)\.(?:png|svg)", name))
    if owner == "plot":
        return name in {"plot_report.json", "plot_mapping.csv", "CAPTION.md", "battery_origin.mplstyle"} or bool(
            re.fullmatch(r"[a-f0-9]{12}_plotted_values\.csv", name) or
            re.fullmatch(r"(?:scientific|exploratory)/[a-f0-9]{12}_p[0-9]+_(?:ridge|heatmap_(?:absolute|row-max))\.(?:png|svg)", name))
    if owner == "wide":
        return name == "column_mapping.csv" or bool(re.fullmatch(
            r"(?:origin|heatmap_exact_grid)_[a-f0-9]{12}\.csv", name))
    if owner == "export":
        return (name.startswith(("00_overview/", "02_quality/", "03_results/", "04_plot_data/",
                                 "05_figures/scientific/", "05_figures/diagnostic/", "06_reproducibility/"))
                or name in {"01_inputs/normalized_eis.csv", "01_inputs/source_row_ledger.csv",
                            "01_inputs/source_audits.json"})
    return False


def refresh_owned_file(target, owner, name):
    """Record a final status update made by the same writer after publication."""
    target = Path(target)
    manifest = checked_path(target, "derived_" + owner + "_manifest.json")
    value = json.loads(manifest.read_text())
    if value.get("owner") != owner or name not in value.get("files", {}) or not owned_name(owner, name):
        raise ValueError("Cannot refresh an unowned output")
    value["files"][name] = file_hash(checked_path(target, name))
    from batch_drt import write_json
    write_json(manifest, value)


def publish_tree(stage, target, owner, legacy_files=()):
    stage, target = Path(stage), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    manifest_name = "derived_" + owner + "_manifest.json"
    manifest = checked_path(target, manifest_name)
    lock = checked_path(target, ".drt-" + owner + ".lock")
    # Refuse overlapping writers instead of interleaving classifications.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Derived publication in progress; do not modify this directory.\n")
    try:
        previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if previous and previous.get("owner") != owner:
            raise ValueError("Derived manifest owner mismatch")
        old = previous.get("files", {}) if previous else {name: None for name in legacy_files}
        new = {p.relative_to(stage).as_posix(): file_hash(p)
               for p in sorted(stage.rglob("*")) if p.is_file()}
        if manifest_name in new or any(n.startswith("_superseded/") for n in new):
            raise ValueError("Staging contains reserved output metadata")
        for name in set(old) | set(new):
            p = checked_path(target, name)
            if not owned_name(owner, name):
                raise ValueError("File outside this export owner's scope: " + name)
            if p.exists() and not p.is_file():
                raise ValueError("Managed file path is a directory: " + name)
            if name in new and p.exists() and name not in old:
                raise FileExistsError("Refusing to overwrite an unowned file: " + str(p))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:12]
        archive = checked_path(target, "_superseded/" + owner + "-" + stamp)
        moved, installed = [], []
        archived = {}
        try:
            for name in old:
                p = checked_path(target, name)
                if p.is_file():
                    dest = checked_path(archive, "files/" + name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    actual = file_hash(p)
                    archived[name] = {"sha256": actual,
                                      "modified_since_export": old[name] is not None and old[name] != actual}
                    p.replace(dest)
                    moved.append((dest, p))
            if moved:
                (archive/"archive_manifest.json").write_text(json.dumps({
                    "owner": owner, "reason": "superseded derived output; not current scientific evidence",
                    "previous_manifest": previous, "files": archived}, indent=2), encoding="utf-8")
            for name in new:
                dest = checked_path(target, name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                (stage/name).replace(dest)
                installed.append((dest, stage/name))
            value = {"owner": owner, "files": new,
                     "superseded_archive": archive.relative_to(target).as_posix() if moved else None}
            pending = stage / manifest_name
            pending.write_text(json.dumps(value, indent=2), encoding="utf-8")
            pending.replace(manifest)
        except Exception:
            # Restore the prior publication if a filesystem operation fails.
            for dest, original in reversed(installed):
                original.parent.mkdir(parents=True, exist_ok=True)
                dest.replace(original)
            for archived_path, original in reversed(moved):
                original.parent.mkdir(parents=True, exist_ok=True)
                archived_path.replace(original)
            raise
        return value
    finally:
        lock.unlink()

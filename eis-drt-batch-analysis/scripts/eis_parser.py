#!/usr/bin/env python3
"""Bundled IRF/text acquisition parser and QC, extracted from the local EIS parser."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np


@dataclass
class Spectrum:
    spectrum_id: str
    label: str
    source: Path
    acquisition_index: np.ndarray
    freq_hz: np.ndarray
    zreal: np.ndarray
    neg_zimag: np.ndarray
    raw_rows: int
    excluded: dict[str, int]
    sign_basis: str
    impedance_basis: str
    safety_events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TextParseError(ValueError):
    """A text failure with a source-row audit available to batch/export callers."""

    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("μ", "u").replace("µ", "u").replace("ω", "ohm").replace("Ω", "ohm")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def finite_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def resolve_imaginary(raw: np.ndarray, header: str, convention: str,
                      metadata_text: str = "", modulus: np.ndarray | None = None,
                      phase: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    if convention == "neg-zimag":
        return raw, "explicit --imag-convention neg-zimag"
    if convention == "zimag":
        return -raw, "explicit --imag-convention zimag"
    if convention != "auto":
        raise ValueError("imag_convention must be auto, neg-zimag or zimag")
    token = norm(header)
    lower_header = _prime_header(header).lower().replace(" ", "").replace("_", "")
    if any(mark in lower_header for mark in (
        "-z''", "-zim", "-im(z", "-imz", "negz", "minusz", "-zdoubleprime", "-impedanceimaginary",
        "-zdoubleprime", "-imag", "-im(",
    )) or token.startswith(("negzimag", "negimz", "negimag", "minusimag")):
        return raw, f"explicit negative-imaginary header: {header}"
    meta = metadata_text.lower().replace(" ", "")
    if "we.-z" in meta or "-z''" in meta or "negzimag" in norm(meta):
        return raw, "instrument metadata identifies -Z imaginary"
    # Auxiliary phase/modulus may have independent units or sign conventions.
    # Never infer radians from magnitude or let them silently override a component
    # header. Ambiguous components require a documented explicit convention.
    if (
        "zimag" in token or "imaginary" in token or "z''" in lower_header
        or "zdoubleprime" in token or _column_role(header) == "zimag"
        or token.startswith(("imz", "imagz"))
    ):
        return -raw, f"explicit Z-imaginary header: {header}"
    raise ValueError("Imaginary sign is ambiguous; provide --imag-convention neg-zimag or zimag")


def split_frequency_runs(freq: np.ndarray) -> list[slice]:
    if len(freq) < 2:
        return [slice(0, len(freq))]
    diffs = np.diff(np.log10(freq))
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if not len(nonzero):
        return [slice(0, len(freq))]
    descending = float(np.median(nonzero)) < 0
    cuts = [0]
    for i, (before, after) in enumerate(zip(freq, freq[1:]), start=1):
        restart = after > before * 1.2 if descending else after < before / 1.2
        if restart:
            cuts.append(i)
    cuts.append(len(freq))
    return [slice(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in con.execute(f"pragma table_info({quote_ident(table)})")]


def read_safety_events(con: sqlite3.Connection) -> list[dict[str, Any]]:
    present = con.execute("select 1 from sqlite_master where type='table' and name='SafetyStateEntries'").fetchone()
    if not present:
        return []
    columns = table_columns(con, "SafetyStateEntries")
    rows = con.execute(f"select * from {quote_ident('SafetyStateEntries')}").fetchall()
    return [dict(zip(columns, row)) for row in rows]


def procedure_summary(document: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"header_parameters": []}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("State", "SubState", "Version"):
                if key in value and key.lower() not in summary:
                    summary[key.lower()] = value[key]
            parameters = value.get("HeaderParameters")
            if isinstance(parameters, list) and not summary["header_parameters"]:
                summary["header_parameters"] = [
                    {name: item.get(name) for name in ("Label", "Value", "ShownUnit", "UnitSuffix")}
                    for item in parameters if isinstance(item, dict)
                ]
            distribution = value.get("FrequencyDistribution")
            if isinstance(distribution, list) and "programmed_frequency_values" not in summary:
                flattened = []
                for item in distribution:
                    candidate = item[0] if isinstance(item, list) and item else item
                    parsed = finite_float(candidate)
                    if parsed is not None:
                        flattened.append(parsed)
                if flattened:
                    summary["programmed_frequency_values"] = flattened
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return summary


def read_irf(path: Path, label: str, imag_convention: str) -> tuple[list[Spectrum], dict[str, Any]]:
    spectra: list[Spectrum] = []
    audit: dict[str, Any] = {"source": str(path), "sha256": file_hash(path), "archive_members": [], "databases": []}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        audit["archive_members"] = members
        db_names = [name for name in members if name.lower().endswith(".db")]
        if not db_names:
            raise ValueError(f"No database was found inside {path}")
        metadata_parts = []
        procedure_summaries = []
        for name in members:
            if name.lower().endswith((".json", ".txt")):
                try:
                    decoded = archive.read(name).decode("utf-8", errors="replace")
                    metadata_parts.append(decoded)
                    if name.lower().endswith(".json"):
                        try:
                            procedure_summaries.append({"member": name, **procedure_summary(json.loads(decoded))})
                        except json.JSONDecodeError:
                            procedure_summaries.append({"member": name, "json_parse_error": True})
                except Exception:
                    pass
        metadata_text = "\n".join(metadata_parts)
        audit["procedure_summaries"] = procedure_summaries
        run_state = next((item.get("state") for item in procedure_summaries if item.get("state")), None)
        programmed_frequency = next(
            (item.get("programmed_frequency_values") for item in procedure_summaries
             if item.get("programmed_frequency_values")), None
        )
        with tempfile.TemporaryDirectory(prefix="eis-irf-") as temp_dir:
            for db_number, db_name in enumerate(db_names, start=1):
                db_path = Path(temp_dir) / f"db_{db_number}.sqlite"
                db_path.write_bytes(archive.read(db_name))
                con = sqlite3.connect(db_path)
                try:
                    tables = [row[0] for row in con.execute("select name from sqlite_master where type='table'")]
                    measured = []
                    for table in tables:
                        columns = table_columns(con, table)
                        required = {"VerifiedAppliedFrequency", "ImpedanceReal", "ImpedanceImaginary"}
                        if required.issubset(columns):
                            measured.append((table, columns))
                    db_audit = {"member": db_name, "tables": tables, "measured_tables": [x[0] for x in measured]}
                    audit["databases"].append(db_audit)
                    safety = read_safety_events(con)
                    for table_number, (table, columns) in enumerate(measured, start=1):
                        optional = [name for name in ("Index", "ImpedanceModulus", "ImpedancePhase", "Time", "Valid") if name in columns]
                        selected = optional + ["VerifiedAppliedFrequency", "ImpedanceReal", "ImpedanceImaginary"]
                        order = f" order by {quote_ident('Index')}" if "Index" in columns else " order by rowid"
                        rows = con.execute("select " + ",".join(quote_ident(x) for x in selected) +
                                           f" from {quote_ident(table)}" + order).fetchall()
                        positions = {name: i for i, name in enumerate(selected)}
                        raw_count = len(rows)
                        excluded = Counter()
                        idx_values = []
                        freq_values = []
                        real_values = []
                        imag_values = []
                        mod_values = []
                        phase_values = []
                        for row_number, row in enumerate(rows, start=1):
                            valid = finite_float(row[positions["Valid"]]) if "Valid" in positions else 1.0
                            f = finite_float(row[positions["VerifiedAppliedFrequency"]])
                            zr = finite_float(row[positions["ImpedanceReal"]])
                            zi = finite_float(row[positions["ImpedanceImaginary"]])
                            if valid is not None and valid <= 0:
                                excluded["Valid=0"] += 1
                                continue
                            if f is None or zr is None or zi is None:
                                excluded["nonnumeric_or_nonfinite"] += 1
                                continue
                            if f <= 0:
                                excluded["nonpositive_frequency"] += 1
                                continue
                            idx_values.append(finite_float(row[positions["Index"]]) if "Index" in positions else row_number)
                            freq_values.append(f)
                            real_values.append(zr)
                            imag_values.append(zi)
                            mod_values.append(finite_float(row[positions["ImpedanceModulus"]]) if "ImpedanceModulus" in positions else math.nan)
                            phase_values.append(finite_float(row[positions["ImpedancePhase"]]) if "ImpedancePhase" in positions else math.nan)
                        freq = np.asarray(freq_values, dtype=float)
                        real = np.asarray(real_values, dtype=float)
                        imag = np.asarray(imag_values, dtype=float)
                        modulus = np.asarray(mod_values, dtype=float)
                        phase = np.asarray(phase_values, dtype=float)
                        neg_imag, sign_basis = resolve_imaginary(imag, "ImpedanceImaginary", imag_convention,
                                                                 metadata_text, modulus, phase)
                        slices = split_frequency_runs(freq)
                        schedule_check = None
                        if programmed_frequency and len(programmed_frequency) == len(freq):
                            programmed = np.asarray(programmed_frequency, dtype=float)
                            relative = np.abs(freq - programmed) / np.maximum(np.abs(programmed), 1e-30)
                            schedule_check = {
                                "point_count_match": True,
                                "max_relative_difference": float(np.max(relative)),
                                "matches_within_1pct": bool(np.max(relative) <= 0.01),
                                "note": "Program values are compared in the database VerifiedAppliedFrequency base unit; display ShownUnit labels are not applied a second time",
                            }
                        elif programmed_frequency:
                            schedule_check = {"point_count_match": False, "matches_within_1pct": False,
                                              "program_points": len(programmed_frequency), "measured_points": len(freq)}
                        for run_number, run in enumerate(slices, start=1):
                            spectrum_id = f"{path.stem}:db{db_number}:{table}:run{run_number}"
                            run_excluded = dict(excluded) if len(slices) == 1 else {}
                            spectra.append(Spectrum(
                                spectrum_id, label, path, np.asarray(idx_values, dtype=float)[run], freq[run], real[run],
                                neg_imag[run], raw_count if len(slices) == 1 else len(range(*run.indices(len(freq)))),
                                run_excluded, sign_basis, "ohm", safety,
                                {"database_member": db_name, "table": table, "run": run_number,
                                 "all_table_rows": raw_count, "table_excluded_by_reason": dict(excluded),
                                 "frequency_runs_in_table": len(slices), "run_state": run_state,
                                 "frequency_schedule_check": schedule_check},
                            ))
                    if not measured:
                        db_audit["warning"] = "no table contained all required EIS columns"
                finally:
                    con.close()
    if not spectra:
        raise ValueError(f"No EIS spectrum was found inside {path}")
    return spectra, audit


def detect_encoding(path: Path) -> str:
    data = path.read_bytes()[:65536]
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16", "latin-1"):
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"


def split_line(line: str, delimiter: str, *, strict: bool = False) -> list[str]:
    if delimiter == "whitespace":
        return re.split(r"\s+", line.strip()) if line.strip() else []
    return next(csv.reader([line], delimiter=delimiter, strict=strict))


def _prime_header(value: str) -> str:
    return value.replace("″", "''").replace("′", "'").replace("’", "'").replace("−", "-").replace("－", "-")


_TEXT_ROLES = {"frequency", "zreal", "zimag", "modulus", "phase", "valid", "index", "time"}
_REQUIRED_ROLES = {"frequency", "zreal", "zimag"}
_DELIMITERS = {"comma": ",", "tab": "\t", "semicolon": ";", "whitespace": "whitespace",
               ",": ",", "\t": "\t", ";": ";"}


def _column_role(value: str) -> str | None:
    """Reviewed header aliases identify components, never their units from values."""
    lower = _prime_header(value).lower()
    token = norm(lower)
    # Strip only recognized trailing units to permit Real (ohm), Imag_mohm,
    # Z_prime/Ohm and equivalent explicit component labels.
    component = re.sub(r"(?:kohm|mohm|ohm)(?:cm2)?$", "", token)
    if "frequency" in token or token in {"freq", "f", "频率hz", "频率"} or token.startswith("freq"):
        return "frequency"
    if (any(mark in token for mark in ("zreal", "impedancereal"))
            or ("z'" in lower and "z''" not in lower)
            or token.startswith(("rez", "realz"))
            or component in {"real", "re", "zre", "zprime"}):
        return "zreal"
    if (any(mark in token for mark in ("zimag", "impedanceimaginary", "negzimag", "zdoubleprime"))
            or "z''" in lower or token.startswith(("imz", "imagz", "negimz"))
            or component in {"imag", "im", "zim", "imaginary", "negimag", "minusimag"}):
        return "zimag"
    if "modulus" in token or "zmod" in token or "|z|" in lower:
        return "modulus"
    if "phase" in token or "相位" in token:
        return "phase"
    if token in {"valid", "有效"}:
        return "valid"
    if token in {"index", "序号"}:
        return "index"
    if token in {"time", "times", "elapsedtime", "elapsedtimes", "时间", "时间s"}:
        return "time"
    return None


def _header_mapping(row: list[str]) -> tuple[dict[str, int], list[str]]:
    mapping, duplicates = {}, []
    for index, value in enumerate(row):
        role = _column_role(value)
        if role is None:
            continue
        if role in mapping:
            duplicates.append(role)
        else:
            mapping[role] = index
    return mapping, duplicates


def detect_text_table(lines: list[str], *, delimiter: str = "auto", header_row: int | None = None
                      ) -> tuple[int, str, list[str], dict[str, int]]:
    best = None
    if delimiter != "auto" and delimiter not in _DELIMITERS:
        raise ValueError("text_delimiter must be auto, comma, tab, semicolon or whitespace")
    if header_row is not None and (isinstance(header_row, bool) or not isinstance(header_row, int)
                                   or not 1 <= header_row <= len(lines)):
        raise ValueError("text_header_row must be an existing one-based row number")
    delimiters = ("\t", ",", ";", "whitespace") if delimiter == "auto" else (_DELIMITERS[delimiter],)
    candidates = list(enumerate(lines[:300])) if header_row is None else [(header_row - 1, lines[header_row - 1])]
    for separator in delimiters:
        for line_number, line in candidates:
            row = split_line(line, separator)
            mapping, duplicates = _header_mapping(row)
            score = len(mapping) + (6 if {"frequency", "zreal", "zimag"}.issubset(mapping) else 0)
            if best is None or score > best[0]:
                best = (score, line_number, separator, row, mapping, duplicates)
    if best is None or not {"frequency", "zreal", "zimag"}.issubset(best[4]):
        raise ValueError("No frequency/Z-real/Z-imaginary header was found; headerless data requires explicit text_columns and text_headerless")
    if best[5]:
        raise ValueError(f"Ambiguous duplicate column roles {best[5]}; provide explicit text_columns")
    return best[1], best[2], best[3], best[4]


def _decode_column_map(value: Any) -> dict[str, int | str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        def unique_pairs(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"Duplicate text_columns role: {key}")
                result[key] = item
            return result
        value = json.loads(value, object_pairs_hook=unique_pairs)
    if not isinstance(value, dict) or not _REQUIRED_ROLES.issubset(value):
        raise ValueError("text_columns must map frequency, zreal and zimag to zero-based indices or exact header names")
    if set(value) - _TEXT_ROLES:
        raise ValueError(f"Unknown text_columns roles: {sorted(set(value) - _TEXT_ROLES)}")
    for selector in value.values():
        if (isinstance(selector, bool) or not isinstance(selector, (int, str))
                or isinstance(selector, int) and selector < 0
                or isinstance(selector, str) and not selector.strip()):
            raise ValueError("Column selectors must be nonnegative zero-based integers or nonempty exact header names")
    return value.copy()


def text_layout(lines: list[str], args: argparse.Namespace) -> tuple[int, str, list[str], dict[str, int]]:
    """Resolve explicit text_* options; header_row is one-based, column indices zero-based.

    Headerless input starts at the first physical row. Its mapping, delimiter,
    frequency/impedance units and imaginary convention must all be declared.
    Explicit mapped headed input defaults to header row 1 and never discards a
    numeric first record as an inferred header.
    """
    columns = _decode_column_map(getattr(args, "text_columns", None))
    headerless = getattr(args, "text_headerless", False)
    delimiter = getattr(args, "text_delimiter", "auto")
    header_row = getattr(args, "text_header_row", None)
    if not isinstance(headerless, bool):
        raise ValueError("text_headerless must be a boolean")
    if header_row is not None and (isinstance(header_row, bool) or not isinstance(header_row, int)
                                   or not 1 <= header_row <= len(lines)):
        raise ValueError("text_header_row must be an existing one-based row number")
    if headerless:
        if header_row is not None:
            raise ValueError("text_headerless conflicts with text_header_row")
        if columns is None:
            raise ValueError("Headerless input requires explicit text_columns")
        if any(not isinstance(value, int) for value in columns.values()):
            raise ValueError("Headerless text_columns requires zero-based integer indices")
        if any(getattr(args, key, "auto") == "auto" for key in ("frequency_unit", "impedance_unit", "imag_convention")):
            raise ValueError("Headerless input requires explicit frequency_unit, impedance_unit and imag_convention")
    if columns is None:
        return detect_text_table(lines, delimiter=delimiter, header_row=header_row)
    if delimiter not in _DELIMITERS:
        raise ValueError("Explicit text_columns requires text_delimiter: comma, tab, semicolon or whitespace")
    separator = _DELIMITERS[delimiter]
    first = next((line for line in lines if line.strip()), "") if headerless else (lines[(header_row or 1) - 1] if lines else "")
    header = split_line(first, separator)
    if not header:
        raise ValueError("No columns exist at the declared text table start")
    mapping = {}
    for role, selector in columns.items():
        if isinstance(selector, int):
            index = selector
        else:
            matches = [i for i, name in enumerate(header) if name.strip() == selector.strip()]
            if len(matches) != 1:
                raise ValueError(f"Column name {selector!r} must occur exactly once; found {len(matches)}")
            index = matches[0]
        if index >= len(header):
            raise ValueError(f"Column index {index} for {role} is outside the declared table")
        if index in mapping.values():
            raise ValueError("Each text_columns role must select a different physical column")
        mapping[role] = index
    if headerless:
        return -1, separator, [f"column_{i}" for i in range(len(header))], mapping
    try:
        # NaN/Inf are invalid observations, not evidence for a text header.
        for role in _REQUIRED_ROLES:
            float(header[mapping[role]])
        numeric_header = True
    except (TypeError, ValueError):
        numeric_header = False
    if numeric_header:
        raise ValueError("Declared header contains numeric EIS components; use text_headerless to retain the first observation")
    return (header_row or 1) - 1, separator, header, mapping


def frequency_scale(header: str, override: str) -> tuple[float, str]:
    scales = {"hz": 1.0, "khz": 1e3, "mhz": 1e-3}
    if override != "auto":
        if override not in scales:
            raise ValueError("Frequency unit is ambiguous; provide --frequency-unit")
        return scales[override], override
    # SI prefix case is physical information: MHz and mHz differ by 10^9.
    text = header.replace("μ", "u").replace("µ", "u")
    matches = re.findall(r"(?<![A-Za-z])([numkKM]?)[Hh][Zz](?![A-Za-z])", text)
    if len(matches) != 1:
        raise ValueError("Frequency unit is ambiguous; provide --frequency-unit")
    prefix = matches[0]
    factor = {"": 1.0, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6}[prefix]
    return factor, {"": "hz", "m": "mhz", "k": "khz", "K": "khz"}.get(prefix, prefix + "Hz")


def impedance_scale(header: str, override: str) -> tuple[float, str]:
    scales = {"ohm": (1.0, "ohm"), "mohm": (1e-3, "ohm"), "kohm": (1e3, "ohm"),
              "ohm-cm2": (1.0, "ohm_cm2")}
    if override != "auto":
        if override not in scales:
            raise ValueError("Impedance unit is ambiguous; provide --impedance-unit")
        return scales[override]
    text = header.replace("μ", "u").replace("µ", "u").replace("Ω", "ohm").replace("ω", "ohm")
    if re.search(r"(?<![A-Za-z])(?:mm|m)\s*(?:\^?[-−]?2|⁻?²)", text, re.I):
        raise ValueError("Unsupported area unit: normalize explicitly to ohm cm^2 before import")
    area = bool(re.search(r"cm\s*(?:\^?2|²)", text, re.I))
    # A quotient is a different physical basis, not ohm times cm^2.
    if re.search(r"/\s*cm|cm\s*(?:\^?-|⁻)", text, re.I):
        raise ValueError("Impedance per area is not supported as impedance times area")
    text = re.sub(r"cm\s*(?:\^?2|²)", "", text, flags=re.I)
    matches = re.findall(r"(?<![A-Za-z])([numkKM]?)[oO][hH][mM](?![A-Za-z])", text)
    if len(matches) != 1:
        raise ValueError("Impedance unit is ambiguous; provide --impedance-unit")
    factor = {"": 1.0, "n": 1e-9, "u": 1e-6, "m": 1e-3, "k": 1e3, "K": 1e3, "M": 1e6}[matches[0]]
    return factor, "ohm_cm2" if area else "ohm"


def acquisition_time_scale(header: str) -> tuple[float | None, str]:
    text = header.lower().replace("μ", "u").replace("µ", "u").replace("秒", "s")
    token = norm(text)
    if "ms" in token and "min" not in token:
        return 1e-3, "ms"
    if "us" in token:
        return 1e-6, "us"
    if "min" in token:
        return 60.0, "min"
    if "(s)" in text or "[s]" in text or token in {"times", "elapsedtimes", "时间s"}:
        return 1.0, "s"
    return None, "unknown"


def acquisition_time_metadata(
    values: np.ndarray,
    header: str,
    scale: float | None,
    unit: str,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    key = "acquisition_time_values_s" if scale is not None else "acquisition_time_values_raw"
    metadata: dict[str, Any] = {
        "acquisition_time_header": header,
        "acquisition_time_source_unit": unit,
        key: [float(value) if math.isfinite(float(value)) else None for value in values],
    }
    finite_values = values[finite]
    if finite_values.size < 2:
        return metadata
    diffs = np.diff(finite_values)
    positive = diffs[diffs > 0.0]
    median = float(np.median(positive)) if positive.size else math.nan
    anomaly_mask = diffs <= 0.0
    local_reference = np.full(diffs.shape, np.nan, dtype=float)
    for index, step in enumerate(diffs):
        start = max(0, index - 3)
        stop = min(len(diffs), index + 4)
        neighbor_indices = [candidate for candidate in range(start, stop) if candidate != index]
        neighbors = diffs[neighbor_indices]
        neighbors = neighbors[neighbors > 0.0]
        if neighbors.size:
            local_median = float(np.median(neighbors))
            local_reference[index] = local_median
            if step > 3.0 * local_median:
                anomaly_mask[index] = True
    anomaly_indices = np.flatnonzero(anomaly_mask)
    suffix = "_s" if scale is not None else "_raw"
    metadata.update({
        f"acquisition_time_start{suffix}": float(finite_values[0]),
        f"acquisition_time_end{suffix}": float(finite_values[-1]),
        f"sweep_duration{suffix}": float(np.max(finite_values) - np.min(finite_values)),
        f"time_step_median{suffix}": median if math.isfinite(median) else None,
        f"time_step_min{suffix}": float(np.min(diffs)),
        f"time_step_max{suffix}": float(np.max(diffs)),
        "time_monotonic_increasing": bool(np.all(diffs > 0.0)),
        "time_step_anomaly_rule": "nonpositive or >3x median of up to 3 neighboring steps on each side",
        "time_step_anomaly_count": int(anomaly_indices.size),
        "time_step_anomalies": [
            {
                "from_included_point": int(index + 1),
                "to_included_point": int(index + 2),
                f"delta{suffix}": float(diffs[index]),
                f"local_median_reference{suffix}": (
                    float(local_reference[index]) if math.isfinite(local_reference[index]) else None
                ),
            }
            for index in anomaly_indices[:20]
        ],
    })
    return metadata


def _text_row_accounting(audit: dict[str, Any]) -> None:
    ledger = audit.get("row_ledger", [])
    counts = Counter(row["status"] for row in ledger)
    audit["row_accounting"] = {
        "physical_source_lines": len(ledger) if audit.get("source_line_ledger_available") else None,
        "status_counts": dict(counts),
        "included_rows": counts["included"], "excluded_rows": counts["excluded"],
        "undelivered_rows": counts["unparsed"] + counts["parsed-not-delivered"],
        "all_source_lines_accounted": bool(audit.get("source_line_ledger_available")) and
                                      [row["source_line"] for row in ledger] == list(range(1, len(ledger) + 1)),
        "table_rows_accounted": (audit.get("rows_after_header") ==
                                  sum(counts[key] for key in ("included", "excluded", "unparsed", "parsed-not-delivered")))
                                  if "rows_after_header" in audit else None,
    }
    audit["excluded_by_reason"] = dict(Counter(row["reason"] for row in ledger if row["status"] == "excluded"))


def read_text(path: Path, label: str, args: argparse.Namespace) -> tuple[list[Spectrum], dict[str, Any]]:
    """Parse text without losing excluded rows, including whole-source failures.

    ``args.frequency_order`` is ``acquisition`` by default (legacy restart
    detection), or an explicit ``single-spectrum-descending`` declaration.
    Failed sources raise TextParseError with the same JSON-safe ``audit`` API.
    Raw line contents are strings, not executable spreadsheet expressions.
    """
    audit: dict[str, Any] = {"source": str(path), "row_ledger": [], "status": "parsing",
                             "source_line_ledger_available": False,
                             "raw_line_convention": "Original decoded physical line content, excluding its line terminator; raw file bytes remain identified by sha256."}
    try:
        audit["sha256"] = file_hash(path)
        encoding = detect_encoding(path)
        audit["encoding"] = encoding
        lines = path.read_text(encoding=encoding, errors="strict").splitlines()
        audit["source_line_ledger_available"] = True
        audit["row_ledger"] = [{"source_line": number, "raw_line": line,
                                 "raw_fields": None, "status": "unparsed", "reason": "not-yet-parsed"}
                                for number, line in enumerate(lines, start=1)]
        spectra = _read_text_rows(path, label, args, lines, encoding, audit)
        audit.update(status="completed", parsed_spectra=len(spectra))
        _text_row_accounting(audit)
        return spectra, audit
    except Exception as exc:
        audit.update(status="failed", parsed_spectra=0,
                     error=f"{type(exc).__name__}: {exc}")
        if not audit["source_line_ledger_available"]:
            audit["row_ledger_unavailable_reason"] = "Source could not be read/decoded into physical lines; file-level failure and available sha256 are retained, not a zero-row success."
        for row in audit["row_ledger"]:
            if row["status"] in ("included", "parsed-not-delivered"):
                row.update(status="parsed-not-delivered", reason="source-parse-failed-after-row-validation")
                row.pop("spectrum_id", None)
                row.pop("included_point_index", None)
            elif row["status"] == "unparsed":
                row["reason"] = "source-parse-failed-before-row-validation"
        _text_row_accounting(audit)
        raise TextParseError(str(exc), audit) from exc


def _exclude_text_row(entry: dict[str, Any], excluded: Counter, reason: str, detail: str | None = None) -> None:
    excluded[reason] += 1
    entry.update(status="excluded", reason=reason)
    if detail is not None:
        entry["detail"] = detail


def _read_text_rows(path: Path, label: str, args: argparse.Namespace, lines: list[str],
                    encoding: str, audit: dict[str, Any]) -> list[Spectrum]:
    header_line, delimiter, header, mapping = text_layout(lines, args)
    ledger = audit["row_ledger"]
    audit.update(delimiter=delimiter, header_line=header_line + 1 if header_line >= 0 else None,
                 header=header, headerless=header_line < 0, column_mapping=mapping.copy(),
                 first_data_line=header_line + 2, rows_after_header=len(lines) - header_line - 1)
    for row in ledger[:header_line + 1]:
        row.update(status="header" if row["source_line"] == header_line + 1 else "preamble",
                   reason="declared-or-recognized-header" if row["source_line"] == header_line + 1 else "before-table-header")
    if header_line >= 0:
        ledger[header_line]["raw_fields"] = list(header)
    order_policy = getattr(args, "frequency_order", "acquisition")
    audit["frequency_order_policy"] = order_policy
    if order_policy not in ("acquisition", "single-spectrum-descending"):
        raise ValueError("frequency_order must be acquisition or single-spectrum-descending")
    f_scale, f_unit = frequency_scale(header[mapping["frequency"]], args.frequency_unit)
    zr_scale, zr_basis = impedance_scale(header[mapping["zreal"]], args.impedance_unit)
    zi_scale, zi_basis = impedance_scale(header[mapping["zimag"]], args.impedance_unit)
    audit.update(source_frequency_unit=f_unit, frequency_scale=f_scale,
                 zreal_scale=zr_scale, imaginary_column_scale=zi_scale, impedance_basis=zr_basis)
    time_factor, time_unit = (
        acquisition_time_scale(header[mapping["time"]])
        if "time" in mapping else (None, "not-present")
    )
    if zr_basis != zi_basis:
        raise ValueError("Real and imaginary impedance columns use incompatible units")
    excluded = Counter()
    acquisition = []
    freq = []
    real = []
    imag = []
    modulus = []
    phase = []
    acquisition_time = []
    included_source_lines = []
    for source_line, line in enumerate(lines[header_line + 1:], start=header_line + 2):
        entry = ledger[source_line - 1]
        try:
            row = split_line(line, delimiter, strict=True)
        except csv.Error as exc:
            _exclude_text_row(entry, excluded, "csv_parse_error", str(exc))
            continue
        entry["raw_fields"] = row
        entry["raw_values"] = {role: row[index] if index < len(row) else None for role, index in mapping.items()}
        if not row or not any(value.strip() for value in row):
            _exclude_text_row(entry, excluded, "blank_row")
            continue
        if max(mapping["frequency"], mapping["zreal"], mapping["zimag"]) >= len(row):
            _exclude_text_row(entry, excluded, "short_row")
            continue
        valid = finite_float(row[mapping["valid"]]) if "valid" in mapping and mapping["valid"] < len(row) else None if "valid" in mapping else 1.0
        f = finite_float(row[mapping["frequency"]])
        zr = finite_float(row[mapping["zreal"]])
        zi = finite_float(row[mapping["zimag"]])
        if valid is None:
            _exclude_text_row(entry, excluded, "invalid_valid_flag")
            continue
        if valid <= 0:
            _exclude_text_row(entry, excluded, "Valid=0")
            continue
        if f is None or zr is None or zi is None:
            _exclude_text_row(entry, excluded, "nonnumeric_or_nonfinite")
            continue
        f *= f_scale
        zr *= zr_scale
        zi *= zi_scale
        if not all(math.isfinite(value) for value in (f, zr, zi)):
            _exclude_text_row(entry, excluded, "nonfinite_after_unit_conversion")
            continue
        if f <= 0:
            _exclude_text_row(entry, excluded, "nonpositive_frequency")
            continue
        index = finite_float(row[mapping["index"]]) if "index" in mapping and mapping["index"] < len(row) else source_line
        acquisition.append(index)
        included_source_lines.append(source_line)
        freq.append(f)
        real.append(zr)
        imag.append(zi)
        entry.update(status="parsed-not-delivered", reason="awaiting-source-validation",
                     acquisition_index=index,
                     normalized_values={"frequency_hz": f, "zreal": zr, "imaginary_column_scaled": zi})
        modulus.append((finite_float(row[mapping["modulus"]]) or math.nan) * zr_scale if "modulus" in mapping and mapping["modulus"] < len(row) else math.nan)
        phase.append(finite_float(row[mapping["phase"]]) if "phase" in mapping and mapping["phase"] < len(row) else math.nan)
        raw_time = finite_float(row[mapping["time"]]) if "time" in mapping and mapping["time"] < len(row) else None
        acquisition_time.append(
            raw_time * time_factor if raw_time is not None and time_factor is not None
            else raw_time if raw_time is not None else math.nan
        )
    freq_array = np.asarray(freq, dtype=float)
    real_array = np.asarray(real, dtype=float)
    imag_array = np.asarray(imag, dtype=float)
    time_array = np.asarray(acquisition_time, dtype=float)
    neg_imag, sign_basis = resolve_imaginary(imag_array, header[mapping["zimag"]], args.imag_convention,
                                             modulus=np.asarray(modulus, dtype=float), phase=np.asarray(phase, dtype=float))
    audit["imaginary_sign_basis"] = sign_basis
    detected_runs = split_frequency_runs(freq_array)
    direction_changes = np.diff(freq_array)
    nonmonotonic = bool(np.any(direction_changes > 0) and np.any(direction_changes < 0))
    warnings = []
    if nonmonotonic:
        warnings.append("Frequency restart detection is heuristic: unordered points and multiple sweeps can look alike; no automatic sorting or merging is performed in acquisition mode.")
    if order_policy == "single-spectrum-descending":
        if len(np.unique(freq_array)) != len(freq_array):
            raise ValueError("Explicit single-spectrum sorting refuses duplicate frequencies; repetitions may represent multiple sweeps")
        positions = [np.argsort(-freq_array, kind="stable")]
        warnings.append("Explicit single-spectrum declaration bypasses frequency-run splitting; source acquisition indices and original row values are retained.")
    else:
        positions = [np.arange(len(freq_array))[run] for run in detected_runs]
    audit.update(frequency_runs_before_sort=len(detected_runs), frequency_order_warnings=warnings,
                 sorting_applied=order_policy == "single-spectrum-descending")
    spectra = []
    raw_rows_after_header = len(lines) - header_line - 1
    for run_number, run in enumerate(positions, start=1):
        run_raw_rows = raw_rows_after_header if len(positions) == 1 else len(run)
        run_excluded = dict(excluded) if len(positions) == 1 else {}
        source_lines = [included_source_lines[index] for index in run]
        spectrum_id = f"{path.stem}:text:run{run_number}"
        for point_index, index in enumerate(run, start=1):
            entry = ledger[included_source_lines[index] - 1]
            entry.update(status="included", reason="valid-eis-row", spectrum_id=spectrum_id,
                         included_point_index=point_index)
            entry["normalized_values"]["neg_zimag"] = float(neg_imag[index])
        run_metadata = {
            "encoding": encoding, "delimiter": delimiter, "header_line": header_line + 1 if header_line >= 0 else None,
            "headerless": header_line < 0, "column_mapping": mapping.copy(),
            "column_mapping_basis": "explicit" if getattr(args, "text_columns", None) is not None else "recognized-header-aliases",
            "first_data_line": header_line + 2,
            "all_rows_after_header": raw_rows_after_header, "table_excluded_by_reason": dict(excluded),
            "source_frequency_unit": f_unit, "frequency_runs": len(positions),
            "frequency_order_policy": order_policy, "frequency_order_warnings": list(warnings),
            "original_frequency_runs_before_sort": len(detected_runs),
            "included_source_lines": source_lines,
            "included_raw_values": [dict(ledger[number - 1]["raw_values"]) for number in source_lines],
            "analysis_to_source_included_index": [int(index) for index in run],
            "source_row_ledger_scope": "source-audit: one row per physical source line; exclusions in multi-run text are source-level, not guessed into a run",
            "analysis_order_matches_acquisition": bool(source_lines == sorted(source_lines)),
        }
        if "time" in mapping:
            run_metadata.update(
                acquisition_time_metadata(
                    time_array[np.sort(run)], header[mapping["time"]], time_factor, time_unit
                )
            )
            values_key = "acquisition_time_values_s" if time_factor is not None else "acquisition_time_values_raw"
            run_metadata[values_key] = [float(value) if math.isfinite(float(value)) else None for value in time_array[run]]
            run_metadata["acquisition_time_diagnostics_order"] = "physical-source-line-order; value arrays remain aligned to analysis points"
            time_source_lines = [included_source_lines[index] for index in np.sort(run)
                                 if math.isfinite(float(time_array[index]))]
            run_metadata["acquisition_time_diagnostics_source_lines"] = time_source_lines
            for anomaly in run_metadata.get("time_step_anomalies", []):
                anomaly["from_source_line"] = time_source_lines[anomaly["from_included_point"] - 1]
                anomaly["to_source_line"] = time_source_lines[anomaly["to_included_point"] - 1]
        spectra.append(Spectrum(
            spectrum_id, label, path, np.asarray(acquisition, dtype=float)[run], freq_array[run],
            real_array[run], neg_imag[run], run_raw_rows, run_excluded, sign_basis, zr_basis, [],
            run_metadata,
        ))
    if not spectra or not len(freq_array):
        raise ValueError(f"No valid EIS points were found in {path}")
    return spectra


def qc_for(spec: Spectrum) -> dict[str, Any]:
    freq = spec.freq_hz
    diffs = np.diff(freq)
    duplicates = int(len(freq) - len(np.unique(freq)))
    direction_changes = int(np.sum(np.sign(diffs[1:]) != np.sign(diffs[:-1]))) if len(diffs) > 1 else 0
    indexes = spec.acquisition_index[np.isfinite(spec.acquisition_index)]
    index_gaps = int(np.sum(np.diff(indexes) > 1.000001)) if len(indexes) > 1 else 0
    accounted = len(freq) + sum(spec.excluded.values())
    return {
        "spectrum_id": spec.spectrum_id,
        "label": spec.label,
        "source": str(spec.source),
        "raw_rows": spec.raw_rows,
        "included_points": int(len(freq)),
        "excluded_by_reason": spec.excluded,
        "row_accounting_total": accounted,
        "row_accounting_matches_raw": accounted == spec.raw_rows,
        "frequency_min_Hz": float(np.min(freq)),
        "frequency_max_Hz": float(np.max(freq)),
        "duplicate_frequency_count": duplicates,
        "frequency_direction_change_count": direction_changes,
        "acquisition_index_gap_count": index_gaps,
        "acquisition_time_available": "acquisition_time_header" in spec.metadata,
        "sweep_duration_s": spec.metadata.get("sweep_duration_s"),
        "time_step_median_s": spec.metadata.get("time_step_median_s"),
        "time_step_max_s": spec.metadata.get("time_step_max_s"),
        "time_step_anomaly_count": spec.metadata.get("time_step_anomaly_count", 0),
        "time_monotonic_increasing": spec.metadata.get("time_monotonic_increasing"),
        "negative_neg_zimag_points": int(np.sum(spec.neg_zimag < 0)),
        "imaginary_sign_basis": spec.sign_basis,
        "impedance_basis": spec.impedance_basis,
        "safety_events": spec.safety_events,
        "status": "suspect" if (spec.safety_events or direction_changes or index_gaps or
                                  spec.metadata.get("time_monotonic_increasing") is False or
                                  (spec.metadata.get("run_state") not in (None, "Finished")) or
                                  (spec.metadata.get("frequency_schedule_check") and
                                   not spec.metadata["frequency_schedule_check"].get("matches_within_1pct", False))) else "pass",
        "metadata": spec.metadata,
    }

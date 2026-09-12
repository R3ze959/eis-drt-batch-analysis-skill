#!/usr/bin/env python3
"""Deterministic end-to-end smoke test for the EIS DRT batch workflow."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from batch_drt import nonseries_inductive_region


def synthetic_impedance(frequency_hz: np.ndarray) -> np.ndarray:
    omega = 2.0 * math.pi * frequency_hz
    r_inf = 1.2
    inductance_h = 2.2e-6
    processes = (
        (2.0, 3.0e3),
        (5.0, 3.0),
    )
    impedance = np.full(frequency_hz.shape, r_inf, dtype=complex) + 1j * omega * inductance_h
    for resistance, peak_frequency in processes:
        time_constant = 1.0 / (2.0 * math.pi * peak_frequency)
        impedance += resistance / (1.0 + 1j * omega * time_constant)
    return impedance


def closest_decades(values: list[float], target: float) -> float:
    return min(abs(math.log10(value / target)) for value in values)


def main() -> int:
    detector_frequency = np.logspace(4.0, -2.0, 30)
    detector_imag = np.ones(detector_frequency.shape)
    detector_imag[-4:] = -1.0
    detected, evidence = nonseries_inductive_region(detector_frequency, detector_imag)
    assert detected is True and evidence["longest_consecutive_run"] == 4

    script = Path(__file__).with_name("batch_drt.py")
    with tempfile.TemporaryDirectory(prefix="eis-drt-smoke-") as temporary:
        root = Path(temporary)
        source = root / "synthetic_two_rc.csv"
        source_repeat = root / "synthetic_two_rc_重复.csv"
        output = root / "out"
        frequency = np.logspace(5.0, -2.0, 85)
        impedance = synthetic_impedance(frequency)
        for path, scale in ((source, 1.0), (source_repeat, 1.05)):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Frequency (Hz)", "Z' (ohm)", "-Z'' (ohm)"])
                for f_value, z_value in zip(frequency, impedance * scale):
                    writer.writerow([f"{f_value:.12g}", f"{z_value.real:.12g}", f"{-z_value.imag:.12g}"])

        command = [
            sys.executable,
            str(script),
            str(source),
            str(source_repeat),
            "--output-dir",
            str(output),
            "--lambda-policy",
            "fixed",
            "--fixed-lambda",
            "1e-4",
            "--lambda-points",
            "7",
            "--no-credible-intervals",
            "--solver-timeout",
            "30",
            "--spectrum-timeout",
            "180",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=240)
        if completed.returncode != 0:
            raise AssertionError(
                f"batch_drt.py returned {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        for forbidden in ("does not have a glyph", "missing from font", "findfont: Failed"):
            assert forbidden not in completed.stderr, completed.stderr
        result_files = list((output / "results").glob("*.json"))
        assert len(result_files) == 2, result_files
        results = [json.loads(path.read_text(encoding="utf-8")) for path in result_files]
        for result in results:
            assert result["status"] == "completed", result.get("error")
            assert result["qc"]["included_points"] == 85
            assert result["qc"]["row_accounting_matches_raw"] is True
            assert result["inductance_mode"] is True
            assert result["inductance_model_selection"]["selected_model"] == "series-L+RC-DRT"
            assert result["inductance_model_selection"]["series_l_explainable"] is True
            assert result["nonseries_inductance_evidence"]["detected"] is False
            assert math.isclose(result["drt"]["lambda"], 1e-4, rel_tol=1e-10)
            peak_frequencies = [
                float(peak["frequency_hz"])
                for peak in result["peaks"]
                if peak["confidence"] != "unsupported-outside-window"
            ]
            assert closest_decades(peak_frequencies, 3.0e3) < 0.35, peak_frequencies
            assert closest_decades(peak_frequencies, 3.0) < 0.35, peak_frequencies
        for required in (
            output / "analysis_report.md",
            output / "batch_summary.csv",
            output / "peaks.csv",
            output / "drt_curves.csv",
            output / "run_manifest.json",
        ):
            assert required.is_file() and required.stat().st_size > 0, required
        svg_files = list((output / "figures").glob("*.svg"))
        png_files = list((output / "figures").glob("*.png"))
        assert len(svg_files) >= 3 and len(png_files) >= 3
        assert (output / "figures" / "DRT_batch_overlay.svg").is_file()
        for path in svg_files:
            root_element = ElementTree.parse(path).getroot()
            # Shared multi-panel diagnostic layouts need not be square. Matplotlib
            # exports physical SVG canvas dimensions in points; require a usable
            # finite canvas without imposing a layout-specific aspect ratio.
            dimensions = [float(root_element.attrib[name].removesuffix("pt"))
                          for name in ("width", "height")]
            assert all(math.isfinite(value) and value > 0 for value in dimensions), (
                f"SVG canvas dimensions are invalid: {path}"
            )
            # Editable-text SVG is the current package contract. Math glyphs may
            # still use paths, but normal labels must retain actual SVG text.
            editable_text = [node for node in root_element.iter()
                             if node.tag.rsplit("}", 1)[-1] == "text"
                             and "".join(node.itertext()).strip()]
            assert editable_text, f"SVG has no editable text labels: {path}"
        quicklook = shutil.which("qlmanage")
        if quicklook:
            render_dir = root / "quicklook"
            render_dir.mkdir()
            rendered = subprocess.run(
                [quicklook, "-t", "-s", "1200", "-o", str(render_dir), *map(str, svg_files)],
                text=True,
                capture_output=True,
                timeout=60,
            )
            assert rendered.returncode == 0, rendered.stderr
            thumbnails = list(render_dir.glob("*.png"))
            assert len(thumbnails) == len(svg_files), thumbnails
            assert all(path.stat().st_size > 0 for path in thumbnails), thumbnails
        print(
            "PASS: two spectra batch-processed; two RC peaks recovered near 3 kHz and 3 Hz; "
            "series inductance detected; overlay, reports, and figures verified"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

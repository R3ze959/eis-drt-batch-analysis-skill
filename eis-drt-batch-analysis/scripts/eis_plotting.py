"""Raw EIS figures: fixed shared typography, native points, scale-aware limits."""
from __future__ import annotations
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile
import numpy as np
from batch_drt import write_json, sha256_file
from derived_outputs import publish_tree, checked_path

STYLE = Path(__file__).resolve().parents[1] / "assets/battery_origin.mplstyle"


def display_scale(real, neg_imag, basis, requested="auto"):
    largest = float(np.max(np.hypot(real, neg_imag)))
    milli = requested == "mohm" or requested == "auto" and 0 < largest < 0.1
    unit = "mΩ" if milli else "Ω"
    if basis == "ohm_cm2":
        unit += " cm²"
    return (1000.0 if milli else 1.0), unit


def limits(values, fraction=0.06):
    values = np.asarray(values, float)
    lo, hi = float(np.min(values)), float(np.max(values))
    width = hi - lo
    # No 1-ohm absolute floor: tiny impedance values keep a truthful useful scale.
    pad = fraction * (width if width > 0 else max(abs(lo), np.finfo(float).tiny))
    if lo == hi == 0:
        return (-1e-6, 1e-6)
    return lo - pad, hi + pad


def visual_qa(fig, ax):
    """Check actual rendered tick/axis-label boxes, not guessed font sizes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_text = []
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        # Locators intentionally generate extra ticks outside the visible limits.
        # Matplotlib does not draw those labels; counting their boxes is a false alarm.
        tick_text.extend(t.label1 for t in axis.get_major_ticks()
                         if lo <= t.get_loc() <= hi and t.label1.get_visible() and t.label1.get_text())
        offset = axis.get_offset_text()
        if offset.get_visible() and offset.get_text():
            tick_text.append(offset)
    labels = [ax.xaxis.label, ax.yaxis.label]
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in tick_text + labels]
    overlaps = []
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i+1:]:
            if a.overlaps(b):
                overlaps.append([ta, tb])
    return {"text_overlap_pairs": overlaps, "text_overlap_status": "pass" if not overlaps else "review-required"}


def make_figure(result, kind, display_unit="auto"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter, LogLocator, NullFormatter, FixedLocator
    f = np.asarray(result["frequency_hz"], float)
    real = np.asarray(result["zreal"], float)
    neg_imag = np.asarray(result["neg_zimag"], float)
    scale, unit = display_scale(real, neg_imag, result["impedance_basis"], display_unit)
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    ax.tick_params(axis="both", which="major", pad=9)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 17
    color = "#194f78"
    kwargs = dict(linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8)
    omitted = []
    if kind == "nyquist":
        x, y = real * scale, neg_imag * scale
        ax.plot(x, y, **kwargs)
        ax.set(xlabel=f"Z′ ({unit})", ylabel=f"−Z″ ({unit})", xlim=limits(x), ylim=limits(y))
        ax.set_box_aspect(1)
        # Square panel, independently scaled labelled axes (not equal impedance units).
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Frequency (Hz)")
        mag = np.hypot(real, neg_imag)
        if kind == "bode_modulus":
            good = mag > 0
            omitted = (np.flatnonzero(~good) + 1).tolist()
            if np.any(good):
                ax.plot(f[good], mag[good] * scale, **kwargs)
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
                ax.yaxis.set_minor_formatter(NullFormatter())
                # A narrow modulus range may contain no powers of ten. Keep the
                # logarithmic geometry but provide useful labelled numeric ticks.
                low, high = ax.get_ylim()
                if np.log10(high / low) < 1.0:
                    ticks = MaxNLocator(nbins=4).tick_values(low, high)
                    ticks = ticks[(ticks >= low) & (ticks <= high) & (ticks > 0)]
                    ax.yaxis.set_major_locator(FixedLocator(ticks))
                    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            else:
                ax.set_ylim(0, 1)
                ax.text(.5, .5, "Zero |Z|\nlog undefined", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(f"|Z| ({unit})")
        elif kind == "bode_phase":
            good = mag > 0
            omitted = (np.flatnonzero(~good) + 1).tolist()
            if np.any(good):
                phase = np.degrees(np.arctan2(-neg_imag[good], real[good]))
                ax.plot(f[good], phase, **kwargs)
                ax.set_ylim(limits(phase))
            else:
                ax.set_ylim(-180, 180)
                ax.text(.5, .5, "Undefined phase\n(|Z| = 0)", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel("Phase (°)")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        else:
            plt.close(fig)
            raise ValueError("Unknown EIS figure kind")
        ax.margins(x=0.04)
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == "linear":
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3, 4))
            axis.set_major_formatter(formatter)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    qa = visual_qa(fig, ax)
    if qa["text_overlap_pairs"]:
        fig.set_size_inches(13, 10)
        qa = visual_qa(fig, ax)
    info = {"spectrum_uid": result["spectrum_uid"], "kind": kind,
            "source_points": len(f), "plotted_points": len(f) - len(omitted),
            "omitted_included_point_indices": omitted,
            "omission_reason": "Zero modulus has no logarithm or defined phase" if omitted else None,
            "display_scale": scale, "display_unit": unit, "stored_basis": result["impedance_basis"],
            "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
            "figsize_inches": list(fig.get_size_inches()), "panel_aspect": "square independent axes" if kind == "nyquist" else "horizontal",
            "raw_points_only": True, "smoothing": False, "connecting_lines": False,
            "phase_convention": "arg(Z) in degrees, not minus phase; no unwrap",
            "layout": "constrained", "axis_label_pad_pt": 17, "tick_pad_pt": 9, **qa}
    return fig, info


def render(output, results, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = font_manager.findfont("Arial", fallback_to_default=False)
    destination = checked_path(Path(output), "05_figures/draft")
    with tempfile.TemporaryDirectory(prefix=".eis-plot-", dir=Path(output).parent) as temp:
        stage = Path(temp)
        local_style = stage / "battery_origin.mplstyle"
        shutil.copy2(STYLE, local_style)
        report = {"status": "completed", "figures": [], "failures": [],
                  "style_sha256": sha256_file(local_style), "font_path": font,
                  "png_dpi": args.plot_dpi, "svg_text": "editable", "overrides": {},
                  "claim": "Draft raw EIS figures; no physical circuit fit or DRT."}
        with plt.style.context(str(local_style)):
            for r in results:
                for kind in ("nyquist", "bode_modulus", "bode_phase"):
                    fig = None
                    try:
                        fig, info = make_figure(r, kind, args.display_unit)
                        base = r["spectrum_uid"] + "_" + kind
                        for suffix in ("png", "svg"):
                            fig.savefig(stage / (base + "." + suffix), dpi=args.plot_dpi, bbox_inches="tight", pad_inches=0.1)
                        report["figures"].append({**info, "png": base + ".png", "svg": base + ".svg"})
                    except Exception as exc:
                        report["failures"].append({"spectrum_uid": r["spectrum_uid"], "kind": kind,
                            "error": f"{type(exc).__name__}: {exc}"})
                    finally:
                        if fig is not None:
                            plt.close(fig)
        if report["failures"]:
            report["status"] = "failed"
        elif any(f["text_overlap_pairs"] for f in report["figures"]):
            report["status"] = "review-required"
        write_json(stage / "plot_report.json", report)
        publish_tree(stage, destination, "eis_plot")
    return report


def render_worker(output, results, args, status_path):
    try:
        report = render(output, results, args)
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(Path(status_path), report)


def render_isolated(output, results, args):
    with tempfile.TemporaryDirectory(prefix="eis-render-") as temp:
        status = Path(temp) / "status.json"
        worker = mp.get_context("spawn").Process(target=render_worker, args=(str(output), results, args, str(status)))
        worker.start()
        worker.join(args.render_timeout * max(1, len(results)))
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
            if worker.is_alive():
                worker.kill()
                worker.join()
            return {"status": "failed", "error": "Rendering timeout; numerical data intact"}
        if worker.exitcode != 0 or not status.is_file():
            return {"status": "failed", "error": f"Renderer exited {worker.exitcode}"}
        return json.loads(status.read_text())

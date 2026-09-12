"""Data-preserving diagnostic plots with explicit measures and rendered-text QA."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np


def compact_axis_ticks(ax, axis="both", max_major_ticks=4):
    """Replace crowded tick labels, preserving axis scale, limits and all data.

    Narrow logarithmic ranges use a small set of readable values in the original
    units. Wider log ranges use decade labels. Minor ticks remain unlabelled.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, MaxNLocator, NullFormatter

    if axis not in {"x", "y", "both"} or max_major_ticks < 2:
        raise ValueError("Expected x/y/both and at least two major ticks")
    configuration = {}
    for name in ("x", "y") if axis == "both" else (axis,):
        obj = ax.xaxis if name == "x" else ax.yaxis
        limits = ax.get_xlim() if name == "x" else ax.get_ylim()
        scale = ax.get_xscale() if name == "x" else ax.get_yscale()
        low, high = sorted(map(float, limits))
        if not math.isfinite(low+high) or high <= low:
            continue
        wide_log = scale == "log" and low > 0 and math.log10(high/low) >= 1.2
        if wide_log:
            exponents = np.arange(math.ceil(math.log10(low)), math.floor(math.log10(high))+1)
            if len(exponents) > max_major_ticks:
                exponents = exponents[np.unique(np.rint(np.linspace(0, len(exponents)-1, max_major_ticks)).astype(int))]
            ticks = 10.**exponents
            formatter = LogFormatterMathtext(base=10, labelOnlyBase=True)
        else:
            ticks = MaxNLocator(nbins=max_major_ticks-1, min_n_ticks=2).tick_values(low, high)
            ticks = ticks[(ticks >= low) & (ticks <= high)]
            if len(ticks) < 2:
                ticks = np.linspace(low, high, 2)
            if len(ticks) > max_major_ticks:
                ticks = ticks[np.unique(np.rint(np.linspace(0, len(ticks)-1, max_major_ticks)).astype(int))]
            if scale == "log":
                ticks = ticks[ticks > 0]
            precision = 3
            while precision < 14 and len({format(float(v), f".{precision}g") for v in ticks}) != len(ticks):
                precision += 1
            formatter = FuncFormatter(lambda value, pos, digits=precision: format(float(value), f".{digits}g"))
        obj.set_major_locator(FixedLocator(ticks))
        obj.set_major_formatter(formatter)
        obj.set_minor_formatter(NullFormatter())
        obj.get_offset_text().set_visible(False)
        configuration[name] = {"scale": scale, "limits": list(limits), "major_ticks": ticks.tolist(),
                               "format": "decade" if wide_log else "compact-original-units"}
    return configuration


def _axis_text_entries(ax, renderer):
    entries = []
    artists = []
    for direction, obj in (("x", ax.xaxis), ("y", ax.yaxis)):
        if ax.axison:
            artists.extend((f"{direction}-tick", item) for item in obj.get_ticklabels(which="both"))
            artists.extend(((f"{direction}-label", obj.label), (f"{direction}-offset", obj.get_offset_text())))
    artists.append(("title", ax.title))
    artists.extend(("annotation", item) for item in ax.texts)
    legend = ax.get_legend()
    if legend is not None:
        artists.extend(("legend", item) for item in legend.get_texts())
    for role, artist in artists:
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        entries.append({"role": role, "text": artist.get_text(), "bbox": list(map(float, bbox.extents)),
                        "font_size_pt": float(artist.get_fontsize())})
    return entries


def _text_collision_report(fig, entries, pad_points):
    pad = pad_points*fig.dpi/72
    overlaps, clipped = [], []
    width, height = fig.bbox.width, fig.bbox.height
    for i, left in enumerate(entries):
        a = left["bbox"]
        if a[0] < -.5 or a[1] < -.5 or a[2] > width+.5 or a[3] > height+.5:
            clipped.append(i)
        for j in range(i+1, len(entries)):
            b = entries[j]["bbox"]
            if a[0] < b[2]+pad and b[0] < a[2]+pad and a[1] < b[3]+pad and b[1] < a[3]+pad:
                overlaps.append([i, j])
    return {"status": "pass" if not overlaps and not clipped else "overlap",
            "padding_points": pad_points, "overlaps": overlaps, "canvas_clipping": clipped,
            "text_artists": entries}


def audit_tick_label_collisions(ax, pad_points=2.0, draw=True):
    """Inspect actual rendered tick, axis-label, title and annotation boxes."""
    if draw:
        ax.figure.canvas.draw()
    entries = _axis_text_entries(ax, ax.figure.canvas.get_renderer())
    return _text_collision_report(ax.figure, entries, pad_points)


def ensure_compact_ticks(ax, axis="both", max_major_ticks=4, min_major_ticks=2, pad_points=2.0):
    """Reduce tick count until rendered labels clear, without shrinking fonts.

    Return an explicit unresolved overlap if canvas expansion is still needed.
    Call after layout changes; rerun after any subsequent canvas/layout change.
    """
    if not 2 <= min_major_ticks <= max_major_ticks:
        raise ValueError("Expected 2 <= min_major_ticks <= max_major_ticks")
    attempts = []
    for count in range(max_major_ticks, min_major_ticks-1, -1):
        ticks = compact_axis_ticks(ax, axis=axis, max_major_ticks=count)
        qa = audit_tick_label_collisions(ax, pad_points=pad_points)
        attempts.append({"maximum_major_ticks": count, "status": qa["status"]})
        if qa["status"] == "pass":
            break
    return {**qa, "ticks": ticks, "attempts": attempts, "data_modified": False}


def audit_figure_text(fig, pad_points=2.0):
    """Include cross-panel text collisions and clipping in the layout audit."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    entries = []
    for index, ax in enumerate(fig.axes):
        entries.extend({"panel_index": index, **item} for item in _axis_text_entries(ax, renderer))
    for artist in fig.texts:
        if artist.get_visible() and artist.get_text().strip():
            entries.append({"panel_index": "figure", "role": "figure-text", "text": artist.get_text(),
                            "bbox": list(map(float, artist.get_window_extent(renderer).extents)),
                            "font_size_pt": float(artist.get_fontsize())})
    return _text_collision_report(fig, entries, pad_points)


def _impedance_unit(result):
    basis = result.get("impedance_basis")
    if basis == "ohm":
        return "Ω"
    if basis in {"ohm_cm2", "ohm_cm^2", "ohm-cm2"}:
        return "Ω cm²"
    return "input impedance units"


def _diagnostic_font_families():
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    base = "Arial" if "Arial" in available else "DejaVu Sans"
    cjk = next((name for name in ("Arial Unicode MS", "Noto Sans CJK SC", "Hiragino Sans GB") if name in available), None)
    return list(dict.fromkeys([base, *([cjk] if cjk else []), "DejaVu Sans"]))


def _branch_state(branch, *, has_distribution=True):
    status = branch.get("status", "not-available")
    if status == "completed" and not has_distribution:
        return "Completed; no accepted candidate", "Calculation completed.\nNo distribution candidate accepted."
    presentations = {
        "disabled": ("Disabled", "Calculation disabled."),
        "not-triggered": ("Not triggered", "Calculation was not triggered."),
        "failed": ("Failed", "Calculation failed.\nSee numerical error diagnostics."),
        "completed": ("Completed", ""),
        "accepted": ("Accepted", ""),
        "no-candidate-accepted": ("No candidate accepted", "Calculation completed.\nNo reduction candidate accepted."),
    }
    return presentations.get(status, (str(status).replace("-", " ").capitalize(), "Result unavailable."))


def _empty_panel(ax, title, message):
    ax.set_title(title)
    ax.set_axis_off()
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, linespacing=1.5)


def build_advanced_audit_figure(result, *, dpi=110):
    """Return (figure, named axes, QA) with continuous and atomic measures apart.

    No values are interpolated, rescaled, normalized or dropped. Loewner RL
    weights remain their exported positive magnitudes, distinguished by marker.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont
    from result_contract import complex_values

    style_path = Path(__file__).resolve().parents[1]/"assets"/"battery_origin.mplstyle"
    try:
        findfont(FontProperties(family="Arial"), fallback_to_default=False)
        font, fallback = "Arial", None
    except ValueError:
        font, fallback = "DejaVu Sans", "Arial unavailable; bundled-style portability fallback"
    with plt.style.context(style_path), matplotlib.rc_context({"font.family": font, "svg.fonttype": "none"}):
        fig, grid = plt.subplots(2, 3, figsize=(24, 14), dpi=dpi, layout="constrained")
        axes = dict(zip(("eis", "primary", "signed", "loewner", "ddt", "measures"), grid.ravel()))
        unit = _impedance_unit(result)
        curves = result["curves"]
        reduce = result.get("model_reduce", {})
        selected = reduce.get("selected") or {}
        reduction_state, _ = _branch_state(reduce)

        ax = axes["eis"]
        ax.plot(curves["zreal_measured"], curves["neg_zimag_measured"], "o", mfc="white", color="#24566B", label="Measured")
        if reduce.get("status") == "accepted":
            for field, label, color in (("preprocessed_impedance", "Reduced EIS", "#BA632A"),
                                        ("reconstructed_impedance", "Reconstruction", "#56686E")):
                values = complex_values(selected.get(field, []))
                if values.size:
                    ax.plot(values.real, -values.imag, color=color, label=label)
        ax.set(xlabel=f"Z′ ({unit})", ylabel=f"−Z″ ({unit})", title=f"Raw EIS\nReduction: {reduction_state.lower()}")
        ax.legend(loc="best")

        ax = axes["primary"]
        ax.plot(curves["drt_frequency_hz"], curves["gamma"], color="#24566B", label="Primary DRT")
        if reduce.get("status") == "accepted" and len(selected.get("tau_s", [])):
            ax.plot(1/(2*np.pi*np.asarray(selected["tau_s"])), selected["gamma"], color="#BA632A", label="Reduced DRT")
        ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel=f"Density γ(ln τ) ({unit})",
               title=f"Primary DRT\nReduction: {reduction_state.lower()}")
        ax.legend(loc="best")

        signed = result.get("signed_gdrt", {})
        ax = axes["signed"]
        state, message = _branch_state(signed)
        if signed.get("status") == "completed" and len(signed.get("gamma", [])):
            ax.plot(signed["frequency_hz"], signed["gamma"], color="#24566B", label="Continuous signed density")
            ax.axhline(0, color="#555555", lw=1)
            ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel=f"Density γ(ln τ) ({unit})",
                   title=f"Continuous signed DRT\n{state}")
        else:
            _empty_panel(ax, f"Continuous signed DRT\n{state}", message or "Calculation completed.\nNo continuous distribution returned.")

        loewner = result.get("loewner_rc_rl", {})
        ax = axes["loewner"]
        state, message = _branch_state(loewner)
        modes = False
        if loewner.get("status") == "completed":
            for tau_key, weight_key, label, color, marker in (("tau_rc_s", "gamma_rc", "RC weight", "#24566B", "o"),
                    ("tau_rl_s", "gamma_rl", "RL weight", "#BA632A", "^")):
                tau = np.asarray(loewner.get(tau_key, []), dtype=float)
                weights = np.asarray(loewner.get(weight_key, []), dtype=float)
                if tau.size:
                    ax.scatter(1/(2*np.pi*tau), weights, s=64, facecolors="white", edgecolors=color,
                               marker=marker, linewidths=1.8, label=label)
                    modes = True
        if modes:
            ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel=f"Branch weight $R_k$ ({unit})",
                   title=f"Discrete Loewner modes\n{state}")
            ax.legend(loc="best")
        else:
            _empty_panel(ax, f"Discrete Loewner modes\n{state}", message or "Calculation completed.\nNo discrete modes returned.")

        ddt = result.get("ddt", {})
        best = ddt.get("best") if ddt.get("status") == "completed" else None
        has_distribution = bool(best and len(best.get("gamma", [])))
        state, message = _branch_state(ddt, has_distribution=has_distribution)
        ax = axes["ddt"]
        if has_distribution:
            ax.plot(best["frequency_hz"], best["gamma"], color="#24566B")
            ax.set(xscale="log", xlabel="Diffusion frequency (Hz)", ylabel=f"DDT density (ln τ; {unit})",
                   title=f"DDT: {state.lower()}\n{ddt.get('best_boundary') or best.get('boundary', 'Selected candidate')}")
        else:
            # Do not draw empty 0..1 numerical axes for an uncomputed or rejected branch.
            _empty_panel(ax, "DDT\n"+state, message)

        ax = axes["measures"]
        ax.set_axis_off()
        ax.set_title("Different distribution measures")
        ax.text(.02, .92, "Continuous density\n∫ γ(ln τ) d ln τ gives resistance.\n\nDiscrete Loewner weights\nSum within each RC or RL branch.\n\nHeights are not directly comparable.\nEach panel has its own y scale.\nRL points show positive magnitudes.\nDotted lines: measured frequency limits.",
                va="top", ha="left", transform=ax.transAxes, linespacing=1.35)

        measured_frequency = np.asarray(curves.get("frequency_hz", []), dtype=float)
        boundaries = []
        if measured_frequency.size and np.all(np.isfinite(measured_frequency)) and np.all(measured_frequency > 0):
            boundaries = [float(np.min(measured_frequency)), float(np.max(measured_frequency))]
            for name in ("primary", "signed", "loewner", "ddt"):
                ax = axes[name]
                if ax.axison:
                    limits = ax.get_xlim()
                    for value in boundaries:
                        ax.axvline(value, color="#707070", lw=1, linestyle=":")
                    ax.set_xlim(limits)

        tick_reports = {}
        for name, ax in axes.items():
            if ax.axison:
                tick_reports[name] = compact_axis_ticks(ax, max_major_ticks=4)
        fig.canvas.draw()
        # Keep typography fixed. Fewer ticks or a larger canvas resolve collisions.
        for name, ax in axes.items():
            if ax.axison:
                tick_reports[name] = ensure_compact_ticks(ax)
        qa = audit_figure_text(fig)
        for _ in range(3):
            if qa["status"] == "pass":
                break
            fig.set_size_inches(*(fig.get_size_inches()*1.12))
            fig.canvas.draw()
            for name, ax in axes.items():
                if ax.axison:
                    tick_reports[name] = ensure_compact_ticks(ax)
            qa = audit_figure_text(fig)
        report = {"text_qa": qa, "tick_qa": tick_reports, "style_path": str(style_path),
                  "font": font, "font_fallback": fallback, "figure_size_inches": fig.get_size_inches().tolist(),
                  "measured_frequency_boundaries_hz": boundaries, "display_points": "all original calculated samples; no interpolation or normalization",
                  "continuous_measure": "density per natural ln(tau); integral has input impedance units",
                  "loewner_measure": "discrete nonnegative RC/RL branch resistance weights; sum has input impedance units",
                  "shared_continuous_discrete_y_axis": False, "loewner_rl_display": "original positive magnitude",
                  "impedance_unit": unit}
    return fig, axes, report


def save_advanced_audit(result, figure_dir, *, dpi=300):
    """Save an audited PNG and editable SVG; preserve existing figure filenames."""
    import matplotlib
    import matplotlib.pyplot as plt

    fig, axes, report = build_advanced_audit_figure(result)
    try:
        if report["text_qa"]["status"] != "pass":
            raise RuntimeError("Advanced diagnostic figure retains text overlap after canvas expansion")
        width, height = fig.get_size_inches()
        export_dpi = min(int(dpi), int(math.sqrt(90_000_000/(width*height))))
        fig.set_dpi(export_dpi)
        report.update(requested_dpi=dpi, export_dpi=export_dpi, export_text_qa=audit_figure_text(fig),
                      dpi_reason="Diagnostic default is 300 dpi; explicit previews may use less. A 90-million-pixel ceiling bounds memory; SVG retains all vector data.")
        if report["export_text_qa"]["status"] != "pass":
            raise RuntimeError("Advanced diagnostic text failed QA at export resolution")
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        slug = str(result.get("result_slug") or "spectrum")
        png, svg = figure_dir/f"{slug}_advanced_audit.png", figure_dir/f"{slug}_advanced_audit.svg"
        with matplotlib.rc_context({"svg.fonttype": "none", "savefig.bbox": None}):
            fig.savefig(png, dpi=export_dpi, facecolor="white")
            fig.savefig(svg, facecolor="white")
        ElementTree.parse(svg)
        result.setdefault("plot_quality", {})["advanced"] = report
        return [str(png.resolve()), str(svg.resolve())]
    finally:
        plt.close(fig)


def build_primary_audit_figure(result, *, dpi=110):
    """Construct the primary diagnostic without modifying any numerical arrays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_path = Path(__file__).resolve().parents[1]/"assets"/"battery_origin.mplstyle"
    fonts = _diagnostic_font_families()
    curves = result["curves"]
    with plt.style.context(style_path), matplotlib.rc_context({"font.family": fonts, "svg.fonttype": "none"}):
        fig, grid = plt.subplots(2, 2, figsize=(20, 18), dpi=dpi, layout="constrained")
        axes = dict(zip(("eis", "drt", "residual", "lcurve"), grid.ravel()))
        ax_n, ax_d, ax_r, ax_l = axes.values()
        unit = _impedance_unit(result)
        ax_n.plot(curves["zreal_reconstructed"], curves["neg_zimag_reconstructed"], color="#BA632A", label="TR-RBF reconstruction")
        ax_n.plot(curves["zreal_measured"], curves["neg_zimag_measured"], linestyle="none", marker="o", mfc="white",
                  mec="#24566B", label="Measured")
        ax_n.axhline(0, color="#777777", lw=1)
        ax_n.set(xlabel=f"Z′ ({unit})", ylabel=f"−Z″ ({unit})", title="(a) Impedance")
        ax_n.legend(loc="best")

        for item in curves["sensitivity_curves"]:
            selected = bool(item["selected"])
            ax_d.plot(item["frequency_hz"], item["gamma"], color="#BA632A" if selected else "#6B7280",
                      lw=2.6 if selected else 2.0, linestyle="-" if selected else "--",
                      label=("Selected " if selected else "")+f"λ={item['lambda']:.2g}")
        drt_frequency = np.asarray(curves["drt_frequency_hz"], dtype=float)
        low, high = np.asarray(curves.get("credible_lower", [])), np.asarray(curves.get("credible_upper", []))
        if low.size == drt_frequency.size and high.size == drt_frequency.size:
            ax_d.fill_between(drt_frequency, low, high, color="#BA632A", alpha=.16, label="0.5–99.5%\nconditional band")
        measured_frequency = np.asarray(curves["frequency_hz"], dtype=float)
        for boundary in (float(np.min(measured_frequency)), float(np.max(measured_frequency))):
            ax_d.axvline(boundary, color="#944040", lw=1.2, ls=":")
        for peak in result["peaks"]:
            ax_d.plot(peak["frequency_hz"], peak["gamma"], marker="x" if peak["confidence"] == "unsupported-outside-window" else "o",
                      linestyle="none", mfc="white", color="#111111")
        ax_d.set(xscale="log", xlabel="Frequency (Hz)", ylabel=f"Density γ(ln τ) ({unit})", title="(b) DRT and lambda sensitivity")
        ax_d.set_xlim(min(np.min(drt_frequency), np.min(measured_frequency))/1.15,
                      max(np.max(drt_frequency), np.max(measured_frequency))*1.15)
        handles, labels = ax_d.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_d.legend(unique.values(), unique.keys(), loc="best")

        ax_r.semilogx(measured_frequency, curves["residual_re_percent"], color="#24566B", label="ΔZ′")
        ax_r.semilogx(measured_frequency, curves["residual_im_percent"], color="#BA632A", label="ΔZ″")
        ax_r.axhline(0, color="#555555", lw=1)
        ax_r.set(xlabel="Frequency (Hz)", ylabel="Relative residual (%)", title="(c) Reconstruction residuals")
        ax_r.legend(loc="best")

        selection = result["lambda_selection"]
        rows = selection["grid"]
        x = np.asarray([row["relative_residual_norm"] for row in rows])
        y = np.asarray([row["roughness_norm"] for row in rows])
        lambdas = np.asarray([row["lambda"] for row in rows])
        ax_l.loglog(x, y, "o-", color="#6B7280")
        chosen = int(np.argmin(abs(np.log10(lambdas)-math.log10(selection["selected_lambda"]))))
        ax_l.plot(x[chosen], y[chosen], marker="o", ms=13, mfc="none", mec="#BA632A", mew=2.2)
        ax_l.set(xlabel="Relative reconstruction norm", ylabel="DRT roughness norm",
                 title="(d) Lambda selection\n"+selection["selection_method"])
        fig.suptitle(result["label"], fontsize=30)
        for ax in axes.values():
            compact_axis_ticks(ax)
        fig.canvas.draw()
        tick_qa = {name: ensure_compact_ticks(ax) for name, ax in axes.items()}
        qa = audit_figure_text(fig)
        for _ in range(3):
            if qa["status"] == "pass":
                break
            fig.set_size_inches(*(fig.get_size_inches()*1.12))
            tick_qa = {name: ensure_compact_ticks(ax) for name, ax in axes.items()}
            qa = audit_figure_text(fig)
        report = {"text_qa": qa, "tick_qa": tick_qa, "style_path": str(style_path), "font_families": fonts,
                  "figure_size_inches": fig.get_size_inches().tolist(), "data_modified": False,
                  "impedance_unit": unit, "svg_text": "editable"}
    return fig, axes, report


def save_primary_audit(result, figure_dir, *, dpi=300):
    import matplotlib
    import matplotlib.pyplot as plt
    from batch_drt import safe_slug

    fig, axes, report = build_primary_audit_figure(result)
    try:
        if report["text_qa"]["status"] != "pass":
            raise RuntimeError("Primary diagnostic figure retains text overlap after canvas expansion")
        width, height = fig.get_size_inches()
        export_dpi = min(int(dpi), int(math.sqrt(90_000_000/(width*height))))
        fig.set_dpi(export_dpi)
        report.update(requested_dpi=dpi, export_dpi=export_dpi, export_text_qa=audit_figure_text(fig),
                      dpi_reason="Diagnostic default is 300 dpi; explicit previews may use less. A 90-million-pixel ceiling bounds memory; SVG retains all vector data.")
        if report["export_text_qa"]["status"] != "pass":
            raise RuntimeError("Primary diagnostic text failed QA at export resolution")
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        slug = safe_slug(result.get("result_slug") or result["spectrum_id"])
        png, svg = figure_dir/f"{slug}_DRT_audit.png", figure_dir/f"{slug}_DRT_audit.svg"
        with matplotlib.rc_context({"svg.fonttype": "none", "savefig.bbox": None}):
            fig.savefig(png, dpi=export_dpi, facecolor="white")
            fig.savefig(svg, facecolor="white")
        ElementTree.parse(svg)
        result.setdefault("plot_quality", {})["primary"] = report
        return [str(png.resolve()), str(svg.resolve())]
    finally:
        plt.close(fig)


def save_batch_overlay(results, figure_dir, *, dpi=300):
    """Style an already-selected diagnostic cohort without changing its members."""
    import matplotlib
    import matplotlib.pyplot as plt
    import textwrap

    style_path = Path(__file__).resolve().parents[1]/"assets"/"battery_origin.mplstyle"
    columns = 1 if len(results) <= 12 else 2
    height = max(9, 3+math.ceil(len(results)/columns)*.7)
    with plt.style.context(style_path), matplotlib.rc_context({"svg.fonttype": "none", "font.family": _diagnostic_font_families()}):
        fig, ax = plt.subplots(figsize=(15+5*columns, height), dpi=110, layout="constrained")
        try:
            palette = ["#24566B", "#BA632A", "#73844E", "#826B8D", "#8B783F", "#555555"]
            for index, result in enumerate(results):
                curves = result["curves"]
                ax.plot(curves["drt_frequency_hz"], curves["gamma"], color=palette[index % len(palette)],
                        label="\n".join(textwrap.wrap(result["label"], 30)))
            ax.set(xscale="log", title="Diagnostic RC-compatible overlay\nNot a matched-state trend", xlabel="Frequency (Hz)",
                   ylabel=f"Density γ(ln τ) ({_impedance_unit(results[0])})")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), ncol=columns)
            ensure_compact_ticks(ax)
            qa = audit_figure_text(fig)
            if qa["status"] != "pass":
                raise RuntimeError("Batch overlay has unresolved text overlap")
            figure_dir = Path(figure_dir)
            figure_dir.mkdir(parents=True, exist_ok=True)
            png, svg = figure_dir/"DRT_batch_overlay.png", figure_dir/"DRT_batch_overlay.svg"
            width, height = fig.get_size_inches()
            export_dpi = min(dpi, int(math.sqrt(90_000_000/(width*height))))
            with matplotlib.rc_context({"savefig.bbox": None}):
                fig.savefig(png, dpi=export_dpi, facecolor="white")
                fig.savefig(svg, facecolor="white")
            ElementTree.parse(svg)
            return [str(png.resolve()), str(svg.resolve())]
        finally:
            plt.close(fig)

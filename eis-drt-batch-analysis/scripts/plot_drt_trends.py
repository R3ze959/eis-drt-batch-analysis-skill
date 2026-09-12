#!/usr/bin/env python3
"""Audited voltage-node DRT ridges and native-grid heatmaps; no raw-data changes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import Bbox
import numpy as np

from result_contract import write_table, numerical_digest
import tempfile
from metadata_contract import canonical_metadata, condition_key, GROUP_FIELDS
from batch_drt import write_json, sha256_file

STYLE = Path(__file__).resolve().parents[1] / "assets/battery_origin.mplstyle"
ROLES = {"included", "included-exploratory"}
FIGURE_CONTRACT = "voltage-node-v1"


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def prepare_groups(results):
    """Only explicit eligible evidence and matched metadata enter RC trend panels."""
    groups, mapping, seen = defaultdict(list), [], set()
    for path, result in results:
        uid = result.get("spectrum_uid")
        if not uid or uid in seen:
            raise ValueError("Every result requires a unique spectrum_uid")
        seen.add(uid)
        m, c = canonical_metadata(result.get("metadata", {})), result.get("curves", {})
        voltage = number(m.get("voltage_v", m.get("potential_v", m.get("target_voltage_v"))))
        reason = None
        if result.get("status") != "completed":
            reason = "numerical-not-completed"
        elif result.get("ordinary_rc_eligible") is not True:
            reason = "not-ordinary-RC-eligible"
        elif result.get("display_role") not in ROLES:
            reason = "diagnostic-or-unreviewed-display-role"
        elif not (m.get("cell_id") or m.get("sample_id")):
            reason = "missing-cell-or-sample-identity"
        elif not m.get("direction"):
            reason = "missing-direction"
        elif voltage is None:
            reason = "missing-finite-voltage"
        elif result.get("impedance_basis") not in {"ohm", "ohm_cm2"}:
            reason = "unsupported-or-missing-impedance-basis"
        row = {"spectrum_uid": uid, "result_json": str(path), "source": result.get("source"),
               "source_sha256": result.get("source_sha256"), "voltage_v": voltage,
               "display_role": result.get("display_role"),
               "ordinary_rc_eligible": result.get("ordinary_rc_eligible"),
               "decision_basis": result.get("decision_basis"), "plotted": False}
        if not reason:
            try:
                t, g = np.asarray(c["tau_s"], float), np.asarray(c["gamma"], float)
                f = np.asarray(c["frequency_hz"], float)
                if t.ndim != 1 or g.shape != t.shape or f.ndim != 1:
                    raise ValueError("mismatched arrays")
                if (len(t) < 2 or len(f) < 2 or not np.all(np.isfinite(t)) or np.any(t <= 0)
                        or not np.all(np.isfinite(g)) or not np.all(np.isfinite(f)) or np.any(f <= 0)):
                    raise ValueError("invalid finite positive tau/frequency or gamma")
                order = np.argsort(t)
                t, g = t[order], g[order]
                if np.any(np.diff(t) <= 0) or np.any(g < 0):
                    raise ValueError("duplicate tau or negative gamma in ordinary RC panel")
                lo, hi = max(t[0], 1 / (2 * np.pi * f.max())), min(t[-1], 1 / (2 * np.pi * f.min()))
                if lo >= hi or np.count_nonzero((t >= lo) & (t <= hi)) < 2:
                    raise ValueError("no supported grid")
                group = condition_key(result)
                groups[group].append({"tau": t, "gamma": g, "lo": lo, "hi": hi,
                                      "voltage": voltage, "row": row, "result": result})
            except (KeyError, TypeError, ValueError) as exc:
                reason = "invalid-curve: " + str(exc)
        row["plot_reason"] = reason or "eligible-awaiting-group"
        mapping.append(row)
    return groups, mapping


def clip_group(curves):
    lo, hi = max(r["lo"] for r in curves), min(r["hi"] for r in curves)
    if lo >= hi:
        return None
    clipped = []
    for r in curves:
        mask = (r["tau"] >= lo) & (r["tau"] <= hi)
        if mask.sum() < 2:
            return None
        clipped.append({**r, "x": np.log10(r["tau"][mask]), "g": r["gamma"][mask],
                        "points_before": len(r["tau"]), "points_displayed": int(mask.sum())})
    return clipped, math.log10(lo), math.log10(hi)


def cell_edges(x, lo, hi):
    """Native sample-centred cells; no interpolation and no extrapolated colour."""
    mid = (x[1:] + x[:-1]) / 2
    return np.r_[max(lo, x[0] - (x[1] - x[0]) / 2),
                 mid, min(hi, x[-1] + (x[-1] - x[-2]) / 2)]


def tick_strings(curves):
    # Keep every repeat, including equal voltages. Mapping disambiguates by UID.
    return [f'{c["voltage"]:.3f} V' for c in curves]


def unit_label(basis):
    return r"$\Omega\,\mathrm{cm}^2$" if basis == "ohm_cm2" else r"$\Omega$"


def title_for(key):
    identity = key[0] or key[1]
    title = f"{identity} | {key[2]}"
    if key[-1] == "synthetic":
        title = "Synthetic demonstration | " + title
    if key[-2] == "included-exploratory":
        title += "\nExploratory: acquisition controls incomplete"
    return title


def axis_qa(fig, ax, three_d=False):
    axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if three_d else [])
    rendered_ticks = {axis: axis._update_ticks() for axis in axes}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_boxes, labels = [], []
    for axis in axes:
        # get_ticklabels() updates 3D tick positions back to unprojected data
        # coordinates. Inspect already-drawn artists directly after canvas.draw().
        items = [label for tick in rendered_ticks[axis] for label in (tick.label1, tick.label2)
                 if label.get_visible() and label.get_text()]
        tick_boxes.extend((t.get_text(), t.get_window_extent(renderer)) for t in items)
        if axis.label.get_visible() and axis.label.get_text():
            labels.append((axis.label.get_text(), axis.label.get_window_extent(renderer)))
    if ax.title.get_text():
        labels.append(("title", ax.title.get_window_extent(renderer)))
    labels.extend((t.get_text(), t.get_window_extent(renderer)) for t in fig.texts
                  if t.get_visible() and t.get_text())
    all_boxes = tick_boxes + labels
    overlap = []
    for i, (name, a) in enumerate(all_boxes):
        for other, b in all_boxes[i+1:]:
            if a.overlaps(b):
                overlap.append([name, other])
    intersections = []
    if three_d:
        for axis in axes:
            line = axis.line.get_path().transformed(axis.line.get_transform())
            for name, box in tick_boxes:
                if line.intersects_bbox(box.padded(1), filled=False):
                    intersections.append(name)
    clipping = [name for name, b in all_boxes if not fig.bbox.contains(b.x0, b.y0)
                or not fig.bbox.contains(b.x1, b.y1)]
    return {"text_overlaps": overlap, "tick_axis_intersections": intersections,
            "canvas_clipping": clipping,
            "pass": not (overlap or intersections or clipping)}


def plot_ridge(curves, key, lo, hi, zmax, scale=1):
    n = len(curves)
    width, height = 18 * scale, max(12, 10 + n * .22) * scale
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=.015, right=.84, bottom=.14, top=.84)
    ax.view_init(elev=40, azim=-65)
    ax.set_box_aspect((1.85, 2.15, 1))
    ax.set(xlim=(lo, hi), ylim=(-.8, max(n - .6, .4)), zlim=(0, zmax))
    ax.grid(False)
    norm = colors.Normalize(min(c["voltage"] for c in curves), max(c["voltage"] for c in curves))
    for i, c in enumerate(curves):
        color = matplotlib.colormaps["cividis"](norm(c["voltage"]))
        verts = [(c["x"][0], 0), *zip(c["x"], c["g"]), (c["x"][-1], 0)]
        poly = PolyCollection([verts], facecolors=[colors.to_rgba(color, .30)], edgecolors="none")
        ax.add_collection3d(poly, zs=[i], zdir="y")
        ax.plot(c["x"], np.full(len(c["x"]), i), c["g"], color=color, linewidth=1.5)
        ax.plot([c["x"][0], c["x"][-1]], [i, i], [0, 0], color=".7", linewidth=.6)
    powers = np.arange(math.ceil(lo), math.floor(hi) + 1, max(1, math.ceil((hi-lo)/8)))
    ax.set_xticks(powers, [rf"$10^{{{int(p)}}}$" for p in powers])
    ax.set_yticks(range(n), tick_strings(curves))
    for tick in ax.yaxis.majorTicks:
        tick.label1.set_horizontalalignment("left")
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_minor_locator(NullLocator())
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor(".65")
    ax.tick_params(axis="x", pad=9)
    ax.tick_params(axis="y", pad=18)
    ax.tick_params(axis="z", pad=16)
    ax.set_zlabel(r"$\gamma(\ln\tau)$ (" + unit_label(key[-3]) + ")", labelpad=42)
    fig.suptitle(title_for(key), y=.965, fontsize=30)
    xlabel = fig.text(.41, .06, r"Relaxation time, $\tau$ (s)", ha="center", fontsize=36)
    qa = axis_qa(fig, ax, three_d=True)
    return fig, qa


def plot_heatmap(curves, key, lo, hi, vmax, scale_mode="absolute", scale=1):
    n = len(curves)
    fig, ax = plt.subplots(figsize=(16 * scale, max(8, 3 + .55*n) * scale))
    fig.subplots_adjust(left=.14, right=.74, bottom=.18, top=.84)
    ax.set(xlim=(10**lo, 10**hi), ylim=(-.5, n-.5), xscale="log")
    norm = colors.Normalize(0, vmax if scale_mode == "absolute" else 1)
    for i, c in enumerate(curves):
        gamma = c["g"]
        if scale_mode == "row-max":
            gamma = gamma / gamma.max() if gamma.max() > 0 else gamma.copy()
        ax.pcolormesh(10**cell_edges(c["x"], lo, hi), [i-.5, i+.5], gamma[None, :],
                      shading="flat", cmap="cividis", norm=norm, rasterized=False)
    ax.set_yticks(range(n), tick_strings(curves))
    ax.yaxis.tick_right()
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="both", left=False, right=True, pad=14)
    ax.set_xlabel(r"Relaxation time, $\tau$ (s)", labelpad=18)
    ax.set_title(title_for(key), pad=22)
    ax.grid(False)
    cax = fig.add_axes([.90, .24, .020, .45])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap="cividis"), cax=cax)
    cb.set_label((r"$\gamma(\ln\tau)$ (" + unit_label(key[-3]) + ")")
                 if scale_mode == "absolute" else r"$\gamma/\max(\gamma)$", labelpad=22)
    qa = axis_qa(fig, ax)
    # Check colorbar text against depth labels and the image, including canvas bounds.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    cb_boxes = [t.get_window_extent(renderer) for t in cax.get_yticklabels() if t.get_text()]
    cb_boxes.append(cax.yaxis.label.get_window_extent(renderer))
    depth_boxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
    qa["colorbar_overlaps"] = sum(a.overlaps(b) for a in cb_boxes for b in depth_boxes)
    qa["colorbar_clipped"] = sum(not fig.bbox.contains(b.x1,b.y1) or not fig.bbox.contains(b.x0,b.y0)
                                 for b in cb_boxes)
    qa["pass"] = qa["pass"] and not qa["colorbar_overlaps"] and not qa["colorbar_clipped"]
    return fig, qa


def save_checked(builder, stem, dpi):
    last = None
    for scale in (1., 1.18, 1.40, 1.70, 2.1, 2.5):
        fig, qa = builder(scale)
        last = qa
        if qa["pass"]:
            width, height = fig.get_size_inches()
            # Large all-node panels keep fixed typography; avoid multi-GB raster
            # allocations while retaining an unthinned, fully vector SVG.
            effective_dpi = min(dpi, max(72, int(math.sqrt(90_000_000/(width*height)))))
            fig.savefig(stem.with_suffix(".png"), dpi=effective_dpi, bbox_inches="tight", pad_inches=.18)
            fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=.18)
            info = {"qa": qa, "figsize_inches": list(fig.get_size_inches()), "dpi": effective_dpi,
                    "requested_dpi": dpi, "raster_pixel_budget": 90_000_000,
                    "dpi_reduction_reason": "large-canvas-memory-limit; SVG remains vector" if effective_dpi < dpi else None,
                    "png": stem.with_suffix(".png").name, "svg": stem.with_suffix(".svg").name}
            plt.close(fig)
            return info
        plt.close(fig)
    raise ValueError("Text layout did not pass after canvas expansion: " + json.dumps(last))


def render_trends(run_dir, output_dir=None, dpi=600, max_curves=20, heatmap_scale="absolute", font=None):
    run_dir = Path(run_dir).resolve()
    target = Path(output_dir).resolve() if output_dir else run_dir / "05_figures/trends/draft"
    from derived_outputs import publish_tree
    # The old beta had a plotting inventory but no ownership manifest. Only its
    # specifically inventoried files can be adopted, never a directory glob.
    legacy = []
    if not (target/"derived_plot_manifest.json").exists() and (target/"plot_report.json").is_file():
        old = json.loads((target/"plot_report.json").read_text())
        legacy = list(old.get("output_files", [])) + ["plot_report.json", "plot_mapping.csv", "CAPTION.md", STYLE.name]
        legacy += [g["id"]+"_plotted_values.csv" for g in old.get("groups", [])]
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".drt-plot-", dir=target.parent) as temp:
        stage = Path(temp)
        report = _render_trends(run_dir, stage, dpi, max_curves, heatmap_scale, font)
        publish_tree(stage, target, "plot", legacy_files=legacy)
        return report


def _render_trends(run_dir, target, dpi, max_curves, heatmap_scale, font):
    if dpi < 72 or dpi > 1200 or max_curves < 2:
        raise ValueError("Require 72 <= dpi <= 1200 and max_curves >= 2")
    if heatmap_scale not in {"absolute", "row-max"}:
        raise ValueError("Unknown heatmap scaling")
    results = []
    for path in sorted((run_dir / "results").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("numerical_sha256") and r["numerical_sha256"] != numerical_digest(r):
            raise ValueError("Numerical checkpoint hash mismatch: " + str(path))
        results.append((path, r))
    groups, mapping = prepare_groups(results)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STYLE, target / STYLE.name)
    chosen_font = font or "Arial"
    try:
        font_manager.findfont(chosen_font, fallback_to_default=False)
    except ValueError:
        if font:
            raise
        chosen_font = "DejaVu Sans"
    summary = {"contract": FIGURE_CONTRACT, "status": "skipped-no-eligible-group",
               "rendering": {"elevation_degrees": 40, "azimuth_degrees": -65,
                             "box_aspect": [1.85, 2.15, 1],
                             "label_pt": 36, "tick_pt": 28, "title_pt": 30,
                             "x_tick_pad_pt": 9, "voltage_tick_pad_pt": 18,
                             "gamma_tick_pad_pt": 16, "gamma_label_pad_pt": 42,
                             "palette": "cividis", "fill_alpha": .30,
                             "max_curves_per_page": max_curves},
               "font": chosen_font, "font_fallback_disclosed": chosen_font != "Arial",
               "style_sha256": sha256_file(STYLE), "scale": heatmap_scale,
               "equal_depth_spacing": "categorical voltage nodes, not equal voltage increments or SOC",
               "no_interpolation": True, "no_repeat_reduction": True,
               "input_sha256": {str(p): sha256_file(p) for p, _ in results},
               "groups": [], "errors": [], "output_files": []}
    # Explicit portability override: no proprietary font is shipped or silently substituted.
    with plt.style.context(str(target / STYLE.name)), plt.rc_context({"font.family": chosen_font, "svg.fonttype": "none"}):
        for key, curves in sorted(groups.items()):
            if len(curves) < 2:
                for c in curves:
                    c["row"]["plot_reason"] = "only-one-compatible-spectrum"
                continue
            descending = key[2].strip().lower() in {"discharge", "discharging", "放电"}
            curves.sort(key=lambda c: ((-1 if descending else 1)*c["voltage"], c["row"]["spectrum_uid"]))
            clipped = clip_group(curves)
            if clipped is None:
                for c in curves:
                    c["row"]["plot_reason"] = "no-common-measured-window"
                continue
            curves, lo, hi = clipped
            group_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
            category = "scientific" if key[-2] == "included" else "exploratory"
            groupdir = target / category
            groupdir.mkdir(exist_ok=True)
            vmax = max(float(c["g"].max()) for c in curves)
            vmax = vmax if vmax > 0 else 1.
            entry = {"id": group_id, "metadata_key": key, "tau_support_s": [10**lo, 10**hi],
                     "gamma_max": vmax, "spectrum_count": len(curves), "pages": []}
            plotted_data = []
            for start in range(0, len(curves), max_curves):
                page = curves[start:start+max_curves]
                page_no = start // max_curves + 1
                try:
                    with tempfile.TemporaryDirectory(prefix=".page-", dir=target) as page_temp:
                        page_dir = Path(page_temp)
                        ridge = save_checked(lambda scale: plot_ridge(page, key, lo, hi, vmax*1.06, scale),
                                             page_dir / f"{group_id}_p{page_no}_ridge", dpi)
                        heat = save_checked(lambda scale: plot_heatmap(page, key, lo, hi, vmax, heatmap_scale, scale),
                                            page_dir / f"{group_id}_p{page_no}_heatmap_{heatmap_scale}", dpi)
                        for artifact in (ridge, heat):
                            for kind in ("png", "svg"):
                                (page_dir/artifact[kind]).replace(groupdir/artifact[kind])
                    entry["pages"].append({"page": page_no, "ridge": ridge, "heatmap": heat})
                    for c in page:
                        c["row"].update({"plotted": True, "plot_reason": "eligible-shown-no-averaging",
                                         "group_id": group_id, "page": page_no})
                    for node, c in enumerate(page):
                        c["row"].update({"depth_node": node, "points_before": c["points_before"],
                                         "points_displayed": c["points_displayed"], "tau_min_s": 10**lo, "tau_max_s": 10**hi})
                        plotted_data.extend({"spectrum_uid": c["row"]["spectrum_uid"], "page": page_no,
                                             "depth_node": node, "voltage_v": c["voltage"],
                                             "tau_s": float(10**x), "gamma": float(g)}
                                            for x,g in zip(c["x"],c["g"]))
                    for artifact in (ridge, heat):
                        summary["output_files"].extend(str((groupdir / artifact[k]).relative_to(target)) for k in ("png", "svg"))
                except Exception as exc:
                    summary["errors"].append({"group": group_id, "page": page_no, "error": str(exc)})
                    for c in page:
                        c["row"]["plot_reason"] = "render-failed-see-report"
            write_table(target / f"{group_id}_plotted_values.csv", plotted_data)
            summary["groups"].append(entry)
    if summary["errors"]:
        summary["status"] = "failed"
    elif summary["groups"]:
        summary["status"] = "completed"
    write_table(target / "plot_mapping.csv", mapping)
    summary["output_sha256"] = {name: sha256_file(target / name) for name in summary["output_files"]}
    write_json(target / "plot_report.json", summary)
    (target / "CAPTION.md").write_text(
        "Voltage-node DRT comparison. All eligible spectra, including repeats, are retained; "
        "excluded records and reasons are in plot_mapping.csv. Charge nodes increase and discharge "
        "nodes decrease in voltage. Equal depth spacing is categorical, not equal voltage increments "
        "or SOC. Curves are clipped to the common measured-frequency-supported interval. "
        "Heatmap colours use native sample-centred cells with no cross-state or within-row interpolation. "
        + ("Row-maximum scaling shows relative shape, not absolute impedance contribution. "
           if heatmap_scale == "row-max" else "Absolute colour limits are shared within each group and across its pages. ")
        + "No mechanism is assigned by peak position. Exploratory groups lack full control evidence. "
        "Synthetic demonstrations are labelled, and drafts require visual review before paper use.\n",
        encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--max-curves", type=int, default=20)
    p.add_argument("--heatmap-scale", choices=["absolute", "row-max"], default="absolute")
    p.add_argument("--font", help="Require an installed font; missing explicit fonts are errors")
    a = p.parse_args()
    report = render_trends(a.run_dir, a.output_dir, a.dpi, a.max_curves, a.heatmap_scale, a.font)
    print(json.dumps({"status": report["status"], "errors": report["errors"]}, indent=2))
    return 2 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

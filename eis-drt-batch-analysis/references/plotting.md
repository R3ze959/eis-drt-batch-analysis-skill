# Voltage-node DRT plot contract

The bundled `scripts/plot_drt_trends.py` consumes authoritative per-spectrum
`results/*.json`; it does not fit, retune, change routing, or read a private case.
The batch command calls it only with explicit `--trend-plots auto`; default is off.
First complete numerical processing and exports. If plotting was not requested,
ask afterward and wait for the answer, as specified in SKILL.md. Calling this
standalone renderer is itself a plotting action and requires that request.

## Eligibility and grouping

- Require completed numerical results, explicit ordinary-RC eligibility, and an
  included or included-exploratory display role. Record every other spectrum and reason.
- Require finite voltage, declared impedance basis and cell/sample identity plus direction.
- Separate identity, direction, cycle, temperature, protocol, AC amplitude, rest
  time, pressure, impedance basis, evidence/display role and synthetic provenance.
  Use the shared canonical metadata/condition key also used by Origin export;
  recognized aliases normalize identically and conflicting aliases are errors.
- Same-voltage repeats remain individual curves and individual heatmap rows.
  Any requested display reduction needs a separate evidence-backed mapping.
- Sort charge by ascending and discharge by descending voltage. Equal categorical
  node spacing improves readability; it is not equal voltage steps, time or SOC.
- Do not pool different groups' colour limits as if they share an experimental basis.
  Within one group, amplitude/colour limits are common across all pages.

## Data fidelity

Show only the intersection of each curve's measured-frequency support and calculated
grid. Do not add points at the boundaries. Preserve every calculated sample inside
the common interval; unsupported points remain in numerical exports.

Ridges are actual independent curves with lightly filled profiles, never fitted surfaces.
The heatmap uses native sample-centred cells per row, even for unequal grids. This
does not calculate an interpolated common-grid matrix. Gaps outside each row's grid
coverage remain blank. The pre-existing exact-grid matrix exporter remains unchanged.

Absolute gamma is the default colour quantity. Row-max normalization is optional and
explicitly labelled dimensionless; it describes relative shape, not process resistance.
Gamma is a density per natural ln(tau); infer a contribution from supported basin
integration, not peak height. No peak-to-mechanism labels are inserted.

## Visual standard and QA

Use the bundled copy of battery_origin.mplstyle: white background, no grid, Arial,
36 pt labels/28 pt ticks, restrained cividis palette. No proprietary fonts are shipped.
If Arial is absent, record a DejaVu Sans portability fallback; a missing explicitly
requested font is an error. Text remains text in SVG.

Draw voltage as `x.xxx V` on each depth-axis tick, without sequence numbers or
a long Voltage axis title. Left-align depth labels so they do not straddle the
axis. Keep gamma ticks and title separated. Use a 40-degree elevation for the
generic dense-node layout; the 3D form is explicitly user-requested.

Before saving, inspect already-rendered 3D tick artists, not get_ticklabels() after
projection: that Matplotlib call can reset positions and invalidate bounding-box QA.
Check tick/label collisions, axis-line intersections, canvas clipping and heatmap
colourbar clearance. Expand the canvas with fixed typography if needed. A failing
layout is an error, not a successful figure.

PNG defaults to 600 dpi. If fixed-type all-node rendering would exceed 90 million
pixels, record the reduced PNG DPI and reason; the editable SVG retains every
vector point. Use fewer curves per page for smaller canvases at 600 dpi.
There is no curve subsampling. Preview runs may explicitly request 72–120 dpi.

Outputs are drafts. Inspect at least one full-size PNG and SVG from every condition/
layout class before accepting. Automated bounding boxes are not full visual proof.

macOS Quick Look can crop a non-square SVG in its square thumbnail even when the
SVG canvas and editable text are intact. Successful thumbnail creation is not a
no-clipping test. Prefer the full PNG for immediate preview, and inspect the SVG
in a standards-compliant browser/vector editor before judging its contents.
Do not crop, stretch or change scientific curve geometry merely to satisfy a
platform thumbnail renderer.

## Outputs and failure handling

`05_figures/trends/draft` includes scientific/exploratory subfolders,
plot_mapping.csv, per-group plotted_values.csv, CAPTION.md, the local style and
plot_report.json with input/output hashes, geometry, DPI, font and QA.

No eligible groups is a recorded skip, not scientific failure. A rendering error
does not alter the numerical JSON; the batch run records trend_figure_status.
Re-render to a new output directory after changing plotting code/configuration.
When reusing an owned plot destination, the prior inventory is recoverably archived
under `_superseded` before publishing the new set, including a no-eligible-group
result. A page publishes its ridge and heatmap only after both succeed. Failed
pages do not leave an orphan scientific figure. Extra user files are preserved.

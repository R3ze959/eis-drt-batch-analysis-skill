# Input, manifest, command, and output contract

Schema 5 (2026-09-07). The numerical engine is unchanged pyimpspec TR-RBF; generalized branches use the bundled checked SciPy solvers.

## Inputs and selection

Discovery recognizes audited text `.csv/.txt/.tsv`, IRF, and pyimpspec vendor/spreadsheet `.P00/.dfr/.dta/.i2b/.idf/.ids/.mpt/.z/.pssession/.ods/.xlsx`.

Text requires identifiable frequency, real impedance and imaginary impedance columns, for example:

```text
Frequency (Hz);Z' (ohm);-Z'' (ohm)
freq_kHz,Zreal_mohm,Zimag_mohm
```

Resolve ambiguity from the actual instrument header before using `--imag-convention neg-zimag|zimag`, `--frequency-unit hz|khz|mhz`, or `--impedance-unit ohm|mohm|kohm|ohm-cm2`. CLI `mhz` means **mHz**, not MHz. Never infer sign, units or area from magnitude.

Native/spreadsheet parsing assumes Hz and ohm unless overridden. pyimpspec sorts by descending frequency, so vendor-format original acquisition order may be unavailable; it must remain disclosed. Text acquisition indices and normalized arrays are preserved.

0.2.1 text import preserves SI prefix case: `MHz` and `mHz`, `MΩ` and `mΩ` have different scales. Prefixes on ohm·cm² do not remove its area basis. Unsupported area dimensions or unknown prefixes are errors, not plain-ohm fallbacks. Explicit imaginary headers and documented instrument metadata control the sign; the previous auxiliary-phase magnitude heuristic is removed. These shared-parser fixes also apply to DRT ingestion; legacy reproduction requires its original software.

Text auditing records every physical line in `row_ledger`, including headers, blanks, excluded values and parse failures. `source_line` and `included_point_index` are one-based; the latter restarts within each returned spectrum. Normalized CSV includes both when available, so source UID + source line can join the original line ledger without guessing row order. Do not execute raw strings as spreadsheet formulas; import these fields as text.

Default `--text-frequency-order acquisition` retains acquisition-based sweep segmentation. Use `--text-frequency-order single-spectrum-descending` only after establishing that an unordered table is ONE physical spectrum; this opt-in stable sort retains original indices/source lines and rejects exact duplicate frequencies. It cannot reliably distinguish multiple sweeps on different grids from one unordered spectrum; review the recorded restart warnings. Manifest `text_frequency_order` overrides the CLI at source level. It is not a general multi-sweep repair.

Before parsing, inventory files by source path/hash and selection reason. DRT-only headers, time/potential tables without frequency, and generated-result schemas can be excluded by content evidence. Unknown candidate formats still receive an explicit parse failure; folder names and Pair numbers are not role evidence. For mixed raw/reference/export trees use a manifest.

## Manifest

A UTF-8 CSV requires `path`; relative paths resolve from its directory.

```csv
path,spectrum_id,label,input_role,include,exclusion_reason,cell_id,state,voltage_v,direction,cycle,temperature_C,area_cm2,ac_amplitude_mv,rest_time_s,replicate,repeat_group,linearity_group,stationarity_group
raw/eis_01.txt,,C1 charge 3.100V,eis,true,,C1,S10,3.100,charge,1,25,0.785,10,1800,1,,,
reference/01.txt,,,supplied-drt,false,comparison reference only,,,,,,,,,,,,,
ocp.csv,,,ocp,false,time-potential control not EIS,,,,,,,,,,,,,
```

- With `--manifest`, default `--manifest-mode allowlist` parses only selected listed paths and spectrum IDs. `include=false` or a non-EIS `input_role` excludes the row while retaining source hash/reason.
- `--manifest-mode metadata` explicitly restores broad discovery plus metadata annotation. It is not recommended for raw/reference/output mixtures.
- Blank `spectrum_id` selects/applies metadata to all spectra in the source. An exact ID overrides that source-level metadata. Duplicate source-level rows or duplicate source+ID rows are rejected; selected missing paths/IDs are errors.
- Extra columns are preserved. Keep identity, state, protocol, cycle, temperature, direction and acquisition controls explicit. Voltage labels do not establish SOC or chronological order.
- `--area-normalize` requires positive `area_cm2`; multiply only non-area-normalized data. Already normalized data are not multiplied again.

Every source has a path-derived `source_uid`; every spectrum has a `spectrum_uid` derived from source identity, content hash and source-internal ID. Identical filenames, labels or contents in distinct source paths do not collide. Use UID joins, not labels. The content hash separately identifies identical copies.

From 0.2.2, duplicate/empty manifest column names and row/header cell-count mismatches are errors. Parser settings belong on a source row with blank `spectrum_id`; an exact spectrum row may repeat an identical setting but cannot contradict the settings already used to parse its source. Both branches protect parser-generated provenance and validate finite converted arrays. A changed source hash or later spectrum-preparation failure rejects that whole source; other independent valid files can still complete. This is an ingestion transaction, not an experimental outlier rule.

## Commands

```bash
.venv/bin/python scripts/batch_drt.py \
  "/absolute/input-directory" \
  --manifest "/absolute/manifest.csv" \
  --output-dir "/absolute/new-output"
```

Useful independent options:

- `--no-credible-intervals`: omit conditional sampling, retain lambda sensitivity.
- `--kk-max-points 120`: declared approximate KK screening subset only; raw EIS and DRT remain full resolution. Default `0` uses all points. Publication subsets require full-point confirmation.
- `--lambda-policy fixed --fixed-lambda 1e-4`: compute the declared value and clipped ±1-decade neighbors, not the entire search grid.
- `--inductance-policy adaptive` (default): compare no-L and series-L across a small lambda grid on a material trigger. `always` is a forced study; `off` is declared legacy compatibility only.
- `--signed-gdrt auto|always|off`, `--ddt auto|always|off`, `--no-loewner`: independent diagnostics.
- `--predictive-fit auto|off`: default auto compares matched positive/signed RC-RLC reconstructions when primary RMS > 2% or signed is triggered; it respects explicit inductance/signed off. This independent CV branch uses the declared lambda range even when the primary lambda policy is fixed.
- `--model-reduce`: optional bounded L/C/Warburg subtraction; retain baseline, reduced and add-back reconstructions and all candidate decisions.
- `--profile-json /absolute/external-profile.json`: explicit private/reference parameters with a recorded hash. No dataset-specific profile is bundled.
- Per-spectrum diagnostic figures and the clearly labelled diagnostic batch overlay remain automatic; no new switch or confirmation is required. Numerical diagnostics and plot-data tables are always retained.
- `--trend-plots auto|off`: metadata-matched voltage-node 3D and native-row heatmaps; default off, independently enabled only for requested trend figures.
- `--trend-dpi 120`: explicit quick preview; the default is 600, subject to a recorded large-canvas raster budget.
- `--allow-partial`: retain numerical failures visibly and permit a partial numerical run's exit status.

### Resume and re-export

The output must be fresh unless `--resume` is explicitly supplied. Reuse the **same command and numerical parameters**, adding:

```text
--resume
```

The execution fingerprint covers input paths/hashes/selection, manifest bytes, parameters, all bundled Python scripts, the authoritative parser, actual versions of every locked dependency, and Python implementation/version/platform. Optional DearEIS is not part of the numerical fingerprint. Changed inputs, parameters, code, interpreter or dependencies require a fresh run directory; an old schema without a contract cannot be resumed. Numerical checkpoint hashes detect altered curves/results. Earlier-beta fingerprints are not migrated by beta.3.

Beta.3 adds `drt.numerical_conditioning` (mode, scale, included point count, uniform-scalar/pointwise-weighting distinction) and `drt.fit_weighting`; physical exported values are restored before classification/export. `lambda_selection` includes original mGCV proposals, rejection reasons, user/package ranges, clipping and full-neighborhood evidence. These remain numerical-checkpoint fields and are covered by its digest. The input inventory is first published with the owned export bundle, not pre-created as an unowned file.

For matched completed checkpoints, add:

```text
--resume --export-only
```

This re-exports without solving and retains automatic diagnostic rendering. After the user requests trend figures, add `--trend-plots auto`; trend enablement and trend DPI are excluded from the numerical fingerprint, but all numerical settings, sources, code and dependencies remain strictly checked. Current DRT run reports record these presentation settings. For a later trend-only request, prefer the standalone renderer into a fresh directory. Re-export rejects missing/failed checkpoints. Ordinary resume repairs diagnostic output and only explicitly enabled trend output. Numerical JSON is saved atomically **before** plotting; plot failures/timeouts do not erase completed inversion. The wrapper fails if its declared Python is missing; it does not silently use system Python.

## Classified output

```text
output/
├── 00_overview/
│   ├── README.md
│   ├── spectrum_index.csv
│   ├── input_inventory.csv
│   └── run_manifest.json
├── 01_inputs/
│   ├── normalized_eis.csv
│   ├── source_row_ledger.csv
│   └── source_audits.json
├── 02_quality/
│   ├── batch_validation.json
│   ├── impedance_reconstruction.csv
│   ├── peaks_with_support.csv
│   ├── all_sensitivity_curves.csv
│   └── advanced_reconstructions.csv
├── 03_results/
│   ├── primary_drt/curves.csv
│   ├── recommended/{curves,reconstruction}.csv
│   ├── predictive_rc/{curves,prediction_scores}.csv
│   ├── signed_gdrt/{curves,loewner_discrete}.csv
│   ├── diffusion_candidates/{candidates,curves}.csv
│   ├── model_reduce/before_after_addback.csv
│   └── reference_compatibility/   # populated only for a legacy profile
├── 04_plot_data/
│   ├── drt_long.csv
│   ├── origin_<condition-id>.csv
│   ├── heatmap_exact_grid_<condition-id>.csv
│   └── column_mapping.csv
├── 05_figures/
│   ├── scientific/
│   ├── diagnostic/
│   └── promotion/               # generated only on a separate user request
├── 06_reproducibility/
│   ├── execution_contract.json
│   └── code/
├── results/<legacy-id>_<uid>.json # authoritative numerical records
├── execution_contract.json
└── root CSV/JSON/report/figures   # backward-compatible views
```

Directory names use stable ASCII for scripts; the overview explains them. Empty branch tables do not imply a completed calculation—read status/index/JSON. Raw files remain external and untouched; `normalized_eis.csv` is a derived normalized table, **not a replacement for raw files**. Archive the referenced originals separately for a portable research package.

Root compatibility outputs remain `batch_summary.csv`, `peaks.csv`, `drt_curves.csv`, advanced branch curves, control-group tables, `batch_validation.json`, `run_manifest.json`, `analysis_report.md` and `figures/`. CSV exports now retain extra fields rather than silently dropping basin bounds or UID metadata.

Numerical quantities retain the declared `impedance_basis`. Gamma is a density per natural ln(tau); Loewner amplitudes are **discrete branch weights**, not densities. For area-normalized input, nuisance resistance/amplitude and L carry the corresponding area factor; legacy internal `*_ohm` / `*_henry` field names must be interpreted with the enclosing basis. Semi-infinite sigma uses impedance per square-root second and appears only in candidate/parameter exports, not tau/gamma rows.

Origin tables use paired X/Y columns per spectrum, preserving unequal tau grids without interpolation. Exact shared grids also get a matrix; otherwise no matrix is silently manufactured. Tables and trends use the same condition key: both cell and sample identity, direction, cycle, temperature, protocol, AC amplitude, rest, pressure, basis, display role and data origin. Absent identity does not merge unrelated sources. The column mapping retains all those conditions plus UID, voltage and label. Recognized aliases include `temperature_c`, `protocol`, `amplitude_mv`/`perturbation_mv`, and `ocv_rest_s`/`rest_before_eis_s`/`hold_time_s`; conflicting aliases are errors. Actual voltage may differ from target voltage; target is a fallback only when actual voltage is absent. No automatic SOC conversion, replicate averaging, duplicate removal, across-voltage P-number identity, or XLSX workbook is implied.

Derived publications have ownership/hash inventories (`derived_export_manifest.json`, `derived_plot_manifest.json`, or standalone `derived_wide_manifest.json`). Stage the new set first, archive previously owned files under `_superseded/<owner-attempt>/files/`, then publish current files; rollback on a filesystem publication error. Each archive records hashes and its superseded status. Raw sources and `results/*.json` are outside this operation. Unknown files, symlinks and conflicting unowned destinations are never overwritten. Archive storage grows across re-exports; retain/review or remove it only with the user's authorization. For current results use the current index/manifest, never a recursive glob including `_superseded`.

Diagnostic figures use 300 dpi PNG and editable-text SVG. Font rendering and large-canvas thumbnails depend on the viewer; Quick Look can crop and is not full-layout acceptance. Inspect the full PNG and QA, or render SVG with a standards-compliant renderer. Root batch overlay is explicitly diagnostic and limited to RC-compatible spectra; it is not a state-matched scientific trend. Accepted scientific and diagnostic figure copies are classified separately; promotion assets never replace annotated records.

The new `05_figures/trends/draft` figures are distinct: they preserve editable SVG
text and every eligible voltage-node curve. Their native-row heatmaps need no
common-grid interpolation and do not change the exact-grid matrix exporter.
Read [plotting.md](plotting.md) for grouping, scaling, drafts and QA.

## States and audit fields

| Field | Meaning |
| --- | --- |
| `status` | Numerical computation completed or failed; not scientific acceptance |
| `numeric_status` | Supported, review-required or failed from numerical screens |
| `evidence_status` | Supported, contradicted or evidence-missing supplied controls |
| `validity_status` | Per-spectrum combined acceptance verdict |
| `display_role` | Included, included-exploratory, diagnostic-only or review-pending |
| `figure_status` | Not-requested/pending/completed/failed/timeout independently of numerical status |
| `run.export_status` | Pending/completed/failed classified export stage |

Control metadata alone is not a pass: comparisons link member UIDs and matched state; explicit group names cannot override incompatible cell/state. Insufficient common frequency points return not-assessable. A late rest plateau does not validate shorter-rest records.

Inspect row accounting, sign/unit basis, KK point coverage/residual structure, reconstruction quality, lambda and tau-grid sensitivity, supported peak basins, branch status and scientific claim boundary. A solver completion and best residual never establish a mechanism or diffusion boundary.

Exit 0 means all numerical spectra completed (or `--allow-partial` was explicit), not scientific acceptance. Exit 2 marks parse/numerical failures without that override. Configuration, unsafe-resume and export errors stop with an exception. Rendering failure is recorded separately and recoverable; always inspect `figure_status` and `export_status`.

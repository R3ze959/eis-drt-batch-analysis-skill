# EIS-only branch (0.2.2)

## Data-meaning safeguards added in 0.2.1

Automatic text units preserve SI prefix case. `MHz` = 10^6 Hz, `mHz` = 10^-3 Hz; `MΩ` = 10^6 Ω, `mΩ` = 10^-3 Ω. Recognized n/µ prefixes are scaled explicitly. CLI `--frequency-unit mhz` retains its historical meaning mHz. Unit-bearing headers allow auto import of MHz without changing that CLI meaning. Unknown prefixes are not stripped into plain Hz/ohm. Area-normalized `mΩ cm²` stays area-normalized after scaling; unsupported m²/mm² and per-area units require an explicit, audited conversion before import.

Imaginary sign follows the explicit convention/header (including Unicode minus and negative aliases), not inferred angle magnitude. Auxiliary modulus/phase cannot override it. Ambiguous sign requires `--imag-convention`; the old heuristic phase-unit guess is removed. Raw auxiliary values remain in the row ledger, and exported Bode phase is derived from the complex components.

`input_quality_status` is separate from KK/Z-HIT results. Bad rows, repeated frequencies, failed source-row accounting and backward/anomalous timestamps require review; they are neither silently dropped nor accepted just because a consistency fit has low residuals. Timestamp warnings do not prove physical drift. Missing experimental controls remain unassessed.

The parser rejects a source whose hash changes between inventory and parsing. For instrument exports being actively written, obtain a stable read-only snapshot and use a fresh output directory; this does not authorize stopping an instrument. DRT handoff includes only original user metadata and source-level parser settings, not generated source-row/time arrays. User metadata cannot replace parser-owned provenance.

Use this branch for impedance data conversion, Nyquist/Bode plotting or EIS consistency screening without DRT. It shares the bundled parser and source identities with DRT, but never invokes its inversion, peak analysis, model-and-reduce, signed/GDRT or DDT. No additional dependencies are required. Read `input-output.md` for supported formats, manifest selection, explicit column/unit/sign overrides and source-row semantics; its DRT solver and DRT output sections do not apply here.

## Commands

From the Skill directory, using the existing validated environment:

```bash
.venv/bin/python scripts/batch_eis.py /absolute/raw-data \
  --manifest /absolute/manifest.csv --output-dir /absolute/project/eis
```

Windows PowerShell uses an invocation operator and one line (Bash backslash continuation does not apply):

```powershell
& .\.venv\Scripts\python.exe scripts/batch_eis.py "C:\data\raw" --manifest "C:\data\manifest.csv" --output-dir "C:\data\project\eis"
```

No shell wrapper is needed. Leave out `--manifest` for an unambiguous raw-only input. Use a new output directory. No installation is necessary when this Skill's environment already exists and passes `scripts/check_runtime.py`.

Default: parse/QC, derive modulus and signed phase, export CSV/JSON/Origin-ready data and three raw-observation diagnostic figures per spectrum. These automatic Nyquist/Bode figures support data understanding and need no confirmation. Additional publication layouts or presentation figures require a request; follow the post-processing question in SKILL.md. It does **not** run KK or Z-HIT merely to draw a plot.

- `--kk on --zhit on`: request full-point consistency diagnostics. Lin-KK includes capacitive and inductive terms and compares impedance/admittance representations; Z-HIT records its selected reconstruction settings. Both are separate from raw observations. Raw points are not smoothed, subtracted or corrected.
- `--check-timeout 120`: wall-time budget **per diagnostic per spectrum**. A failure does not remove data or block later spectra.
- `--check-warn-percent 1 --check-suspect-percent 3`: configurable engineering screens on the larger real/imaginary RMS, not universal experimental acceptance thresholds. Residuals are `100*(reconstructed − measured)/|measured|`. Retain frequency-resolved residuals and lag-1 correlation; a low RMS does not establish random residuals.
- `--plots off`: data and audit only. `--plot-dpi 120`: explicitly requested quick preview; normal PNG is 600 dpi, with editable SVG. `--display-unit mohm` or `ohm`: display-only units; stored values retain ohm or ohm cm². The batch render budget is `--render-timeout` (default 180 seconds) multiplied by the number of spectra.
- `--area-normalize`: requires declared area, never double-applied. Inspect `impedance_basis` and recorded source units.
- `--resume --export-only`: same command, same parameters and unchanged inputs/code/environment; regenerate derived outputs without repeating checks. Keep numerical checkpoints unchanged. Current output indexes exclude archived `_superseded` versions.

Ordinary `--resume` retries only failed/timed-out requested KK or Z-HIT diagnostics, preserving each previous attempt in the checkpoint history. It does not recompute successful or not-assessable checks. `--export-only` never retries checks: it can re-export preserved EIS values while a requested diagnostic remains failed, and keeps the nonzero/partial outcome explicit.

Final root and classified run reports agree after a successful export, including rendering failures. If export itself fails, the root manifest records the current failure while the prior classified publication stays recoverable; consult the root manifest before treating an old table as current. Unexpected renderer-launch errors preserve checkpoints and are recorded as figure failures.

Short spectra remain usable for raw EIS plots: the 15-point DRT gate is not applied. Optional checks require at least ten distinct frequencies and nonzero impedance; otherwise they return `not-assessable`, not a pass. Acquisition-order parsing can segment multiple sweeps. Explicit `single-spectrum-descending` is only for a verified single unordered spectrum; duplicates are not averaged.

## Data-first exports

```text
project/
  eis/
    00_overview/             index, input inventory and reading guide
    01_inputs/               normalized_eis.csv, source row ledger/audits
    02_quality/              optional KK/Z-HIT reconstructions and all statuses
    03_results/              eis_quantities.csv: Z′, −Z″, Z″, |Z|, phase
    04_plot_data/            long table and per-spectrum Origin-ready columns
    05_figures/draft/        Nyquist, Bode magnitude, Bode phase PNG + SVG
    06_reproducibility/      code snapshot, contract, DRT handoff manifest/JSON
    results/<uid>.json       authoritative EIS checkpoint, separate digest/schema
    run_manifest.json       final export and figure statuses
  drt/                      only created for a later explicit DRT request
```

`phase_deg = atan2(Im Z, Re Z)` in degrees; no sign reversal or phase unwrap. Zero impedance has undefined phase (`null`/blank); its raw value remains. Only the log-magnitude and phase figures omit undefined points, with exact indices in the plot report. Normalized observations are not byte copies of raw files; originals remain untouched and external. Archive originals separately when portability is required.

CSV text beginning with spreadsheet formula characters is apostrophe-escaped. Exact metadata and row text remain in JSON. The handoff manifest is exact machine input, not a spreadsheet: if opened in Excel, import all text columns explicitly as text.

## Plotting rules

Use the bundled exact Origin style snapshot, copied into the output. Fixed Arial typography (axis labels 36 pt, ticks 28 pt), white background, black 2.2 pt frame, outward ticks, no grid, open raw markers with no connecting line. The EIS renderer requires installed Arial; if unavailable it reports a figure failure while preserving all data. Install an appropriately licensed font or request a documented style adaptation; do not silently change fonts. Other legacy DRT renderers may use their documented fallback.

Each spectrum receives separate Nyquist and Bode panels, so different cells and protocols are never silently overlaid. UID filenames map to full labels/conditions in the index/JSON. Nyquist uses a square axes box with independently scaled, explicitly labelled axes; it does not imply equal geometric scaling of impedance units. Show all negative −Z″ points. Auto display changes to mΩ when the largest modulus is below 0.1 Ω; bounds depend on actual data span, not an absolute 1 Ω padding floor. No new peaks or interpolated smooth curves are created.

Inspect PNGs and `plot_report.json` for label overlap, omitted undefined points, limits and scale. Figures remain drafts: automated text checks do not certify every visual or scientific feature. Rendering failure cannot erase completed numerical data. `run_manifest.json` records failures independently from spectrum parsing.

## Later DRT analysis

When the user requests DRT, read `06_reproducibility/drt_handoff.json`. Supply its original `inputs`, the recorded `drt_input_manifest.csv` and a **fresh sibling** `drt/` directory to `batch_drt.py`. Preserve `--area-normalize` if true in that JSON. Reuse its `--parser-script` and verify `parser_sha256`, especially for explicit custom instrument parsers. If it has changed or is unavailable, stop to resolve provenance. Do not use `--resume` across modes, copy EIS checkpoints into DRT, or feed exported `normalized_eis.csv` as raw input. No DRT is automatically triggered by this file.

The handoff manifest selects the same raw spectra and parser settings; it is **not** an acceptance filter. DRT reassesses validity, inductance, model family, controls and peak stability. Missing experimental controls remain incomplete, and an EIS consistency pass alone does not establish DRT applicability.

Use raw-only input directories rather than a parent containing prior output trees. Raw/reference mixtures require a reviewed manifest; supplied DRT is not raw EIS.

## Exit status and limitations

Exit 0: parsing/exports and requested computations completed (or explicit partial mode), not scientific acceptance. Exit 2: parse failure, unavailable requested diagnostic or rendering failure unless `--allow-partial`; no usable spectra always fails. Unsafe resume/configuration/export conflicts stop with an exception. Check final manifest and per-spectrum status, even after exit 0.

There is no automatic equivalent-circuit fitting, resistance extraction by invented intercept, repeat averaging, outlier deletion, measured-control acceptance, DRT inversion or physical mechanism assignment in this branch. Numerical KK RC elements are consistency-test basis elements, not a fitted physical circuit.

Official method references: [pyimpspec Lin-KK](https://vyrjana.github.io/pyimpspec/guide_kramers_kronig.html) and [Z-HIT](https://vyrjana.github.io/pyimpspec/guide_zhit.html), checked for the pinned 5.1.3 API. No experimental datasets are bundled in the release.

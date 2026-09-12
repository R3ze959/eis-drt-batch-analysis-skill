---
name: eis-drt-batch-analysis
description: "Process impedance/EIS data with traceable exports and Nyquist/Bode plots, or run audited DRT, signed/GDRT and diffusion candidates when requested. Use for EIS-only processing, EIS-DRT validation and plotting; not automatic electrochemical mechanism assignment."
---

# EIS DRT Batch Analysis

Release **1.0.0**: DRT numerical **schema 5**, separate EIS-only **schema 1**. One self-contained Skill and environment, with independent computation entrypoints. DearEIS and Origin are optional. See [RELEASE_NOTES.md](RELEASE_NOTES.md).

## Data first; diagnostic figures are automatic, presentation figures are opt-in

For both EIS and DRT, prioritize numerical processing, quality checks and CSV/JSON/Origin-ready exports. Automatically retain the figures needed to understand and audit the data: raw EIS Nyquist/Bode, DRT/reconstruction/residual and sensitivity diagnostics, and clearly labelled diagnostic comparisons. These require no additional confirmation or new switch. A rendering failure must not erase numerical results.

Do not automatically generate presentation figures such as voltage-node 3D waterfalls, heatmaps, publication layouts or promotional images. DRT defaults to `--trend-plots off`. After presenting the completed data, diagnostic findings and output location, ask once: “数据和诊断图已导出，是否还需要绘制趋势图或论文展示图？” Wait for the answer; no answer is not permission. Do not ask before processing or block processing on this choice. If the user explicitly declines presentation figures, do not ask again. If they already requested “处理并画图” with a clear figure scope, fulfill it without another confirmation. Historical plotting requests do not authorize presentation figures for new batches.

For requested voltage ridges/heatmaps, use `--trend-plots auto` or the standalone `scripts/plot_drt_trends.py` on completed results. Prefer the standalone renderer into a fresh plot directory for a later plotting-only request; do not refit. Alternatively use the original numerical command plus `--resume --export-only --trend-plots auto`. Trend enablement/DPI may change on re-export, but inputs, numerical settings, code and environment must still match. Never weaken scientific display eligibility to produce a requested image. The existing EIS `--plots off` remains an explicit all-figures opt-out, not the default workflow.

## Select the requested branch first

- “只处理 EIS/阻抗”, “画 Nyquist/Bode/阻抗图”, data conversion or simple EIS QC: use `scripts/batch_eis.py`; read [references/eis-only.md](references/eis-only.md). Do not run DRT or physical equivalent-circuit fitting. Optional `--kk on --zhit on` is appropriate for a requested consistency audit, not required merely to plot.
- Explicit “DRT/弛豫时间分布/signed/GDRT/DDT”: use `scripts/batch_drt.py` and the DRT guidance below.
- Both requested: use one project root with separate `eis/` and `drt/` subdirectories. Each mode has its own execution contract. A later DRT request uses original raw paths plus the EIS handoff manifest, not derived tables or EIS checkpoints.
- An unclear request to “处理阻抗” defaults to EIS-only; explain the scope briefly. DRT suitability is reassessed per spectrum when DRT is actually requested, not inferred from filename or from an EIS consistency pass.

Both branches share the bundled parser, unit/sign rules and source identity. Output is data-first. Never erase raw inductive points, automatically average repeats or turn missing controls into acceptance. No second Skill or extra EIS environment is needed.

0.2.1 preserves SI prefix case and area basis: `MHz` is not `mHz`, `MΩ` is not `mΩ`. Do not infer a sign from phase magnitude or let auxiliary columns override an explicit imaginary-component convention. Ambiguous/unsupported units require clarification. EIS input QC and consistency checks have separate statuses; a low residual cannot erase input-quality warnings. Keep acquisition provenance out of user metadata and reuse only original user metadata in the DRT handoff.

## Read only what the task needs

0.2.2 validates a whole source before accepting its spectra in either branch; changed source hashes, conflicting per-spectrum parser settings, malformed manifests and nonfinite converted arrays are explicit failures. Set parser options on source rows (blank `spectrum_id`), not as conflicting spectrum annotations. EIS ordinary resume can retry failed/timed-out requested checks with history; export-only never recalculates. Read the root final `run_manifest.json`, especially after a failed export: older classified files are not evidence of a successful current attempt.

- [README.md](README.md): install, quick start, supported-platform boundary.
- [references/methodology.md](references/methodology.md): before interpreting peaks or modifying models.
- [references/method-landscape.md](references/method-landscape.md): official-tool comparison and reasons for adopting or deferring numerical methods.
- [references/public-data-iteration.md](references/public-data-iteration.md): public counterexamples, reconstruction recommendations, frequency holdout and new exports.
- [references/peak-evidence.md](references/peak-evidence.md): before interpreting conditional noise/resolution peak evidence.
- [references/input-output.md](references/input-output.md): new inputs, manifests, export fields or resume.
- [references/plotting.md](references/plotting.md): 3D voltage-node or heatmap rendering and visual QA.
- [references/software.md](references/software.md): dependency/runtime/GUI questions.

## Installation and execution

Use a dedicated CPython 3.11 environment. From the extracted Skill directory:

```bash
python3.11 scripts/bootstrap.py
.venv/bin/python scripts/check_runtime.py
.venv/bin/python scripts/batch_drt.py /absolute/input \
  --manifest /absolute/manifest.csv --output-dir /absolute/new-output
```

Windows uses `py -3.11 scripts/bootstrap.py` and `.venv\Scripts\python.exe`. Its entrypoints are provided but this release is locally validated only on macOS arm64. No shell script or second Skill is required when invoking Python directly. The POSIX wrapper uses the package-local environment or an explicit `DRT_PYTHON`; no silent system-Python fallback.

## DRT workflow and claim boundaries

1. Inventory inputs read-only and inspect sign, units, frequency support, cell identity and available controls. A manifest is a selection allowlist by default. Never treat supplied DRT or OCP files as raw EIS.
2. Preserve source hashes, exclusions, acquisition order when available and normalized data. Do not silently flip sign, delete inductive regions, average repeats, or interpolate raw data. Vendor-parser sorting is disclosed.
3. Use a fresh output directory. Strict `--resume` requires matching inputs, configuration, code, actual locked dependencies and interpreter identity. `--resume --export-only` regenerates completed results without recalculation; it is not a legacy migration. Every code upgrade changes fingerprints: retain the original release for earlier runs.
4. Keep the scientific default: adaptive no-L/series-L competition, consensus lambda selection and lambda sensitivity. TR-RBF and TR-NNLS use uniform median-modulus numerical conditioning, restoring impedance, gamma and conditional bands to physical units. This is not pointwise noise weighting, smoothing or deletion. Explicit `--impedance-scaling off` is for solver compatibility only. Reject out-of-range or boundary mGCV candidates instead of clipping them into an optimum; incomplete lambda neighborhoods cannot certify full sensitivity. Optional fixed lambda is explicit; no-L-only is for a declared compatibility comparison, never the only scientific interpretation.
5. Loewner RC/RL is enabled; signed/GDRT and DDT run on their triggers. These are numerical models, not proof of adsorption, diffusion boundaries or unique mechanisms. DDT is a series-impedance candidate family, not full parallel-admittance DDT.
6. `--model-reduce` is opt-in. Gate nuisance subtraction on bounded/converged fits, original-domain add-back reconstruction, removed magnitude and pre/post peak stability. Retain the raw branch even when a reduced candidate is preferred.
7. Keep numerical completion, numerical quality, experimental controls and display purpose separate. Repeats require impedance agreement and resolved major-peak consistency; a low total RMS alone is insufficient. Missing repeats/amplitude/rest controls permit labelled exploratory analysis, not full acceptance. A KK pass does not prove linearity or stationarity.
8. Diagnose each spectrum before inclusion. Filename, index, voltage position and local peak labels never establish routing, exclusion or cross-state mechanism identity. Group metadata must match; do not silently pool cells or protocols.
9. Report supported peak basin integrals over natural `ln(tau)`, not peak height. Peaks beyond measurement support are unsupported; boundary/split/merge/lambda-sensitive peaks require qualified interpretation. No automatic diffusion coefficient or mechanism assignment.
10. Inspect `00_overview/README.md`, `spectrum_index.csv` and per-spectrum JSON. Join by `spectrum_uid`, not labels. Numerical JSON is checkpointed before rendering; a plotting failure does not erase it. Re-export archives previously owned derived files in `_superseded`; those are historical, not current accepted evidence. Unknown files are preserved and never overwritten.
11. Only when requested, render voltage trend figures for eligible matched groups with metadata using `--trend-plots auto`. Every eligible curve/repeat is retained; no private voltage cutoff or hard-coded exclusion applies. Voltage nodes are categorical, not true SOC. Inspect the PNG and plot QA report before accepting a draft.
12. External reference configurations can be supplied with `--profile-json`; their contents and hash are recorded. Keep private reference data and train/holdout protocols outside the release. Never retune on a declared holdout.
13. Inspect conditional peak perturbation evidence separately from lambda stability. It does not certify true peaks; stable spurious peaks can remain. Do not erase weak peaks to improve a benchmark score.
14. Read `model_recommendation` before choosing a reconstruction. Poor RC fits may trigger signed/GDRT and matched positive/signed RC with interior-frequency CV. Selected arrays are exported separately, not substituted into ordinary RC trends. Predictive individual-peak stability is not certified by primary-branch tests. Disable the added comparison with `--predictive-fit off`; explicit inductance/signed-off also disables it. Loewner frequency-holdout/pole diagnostics do not establish unique physical RC/RL elements.
15. Text row ledgers include excluded lines and failed parsing. For a verified physically single unordered text spectrum, explicitly use `--text-frequency-order single-spectrum-descending` (or manifest `text_frequency_order`). It rejects exact duplicate frequencies, but cannot reliably distinguish multiple sweeps on different grids from one unordered spectrum. Read the run-splitting warnings and establish physical single-spectrum identity before this opt-in. Never silently sort to hide acquisition changes.

## Deliverables

- `00_overview`: source inventory and per-spectrum verdicts.
- `01_inputs`: normalized observations and provenance, not replacements for originals.
- `02_quality`: reconstructions, residuals, sensitivity, peak support and controls.
- `03_results`: separate primary DRT, signed, diffusion, reduction and compatibility branches.
- `04_plot_data`: CSV long tables, paired-X/Y Origin tables and exact-common-grid matrices.
- `05_figures`: automatic diagnostic figures; requested presentation figures under `trends/draft` with PNG, editable-text SVG, mappings and plotting configuration. The absence of trend figures is normal when not requested. Promotional images require a separate request.
- `06_reproducibility`: numerical execution contract and code/configuration snapshots.

Interpret in layers: measurement validity, numerical robustness, evidence-supported interpretation, then the next discriminating measurement. Missing control information should not halt useful exploratory computation.

## Maintenance and release verification

Run with the installed DRT environment:

```bash
.venv/bin/python scripts/regression_test.py
.venv/bin/python scripts/release_test.py
.venv/bin/python scripts/audit_regression_test.py
.venv/bin/python scripts/conditioning_test.py
.venv/bin/python scripts/lambda_policy_test.py
.venv/bin/python scripts/advanced_lambda_test.py
.venv/bin/python scripts/public_iteration_test.py
.venv/bin/python scripts/parser_export_test.py
.venv/bin/python scripts/peak_evidence_test.py
.venv/bin/python scripts/diagnostic_plotting_test.py
.venv/bin/python scripts/review_counterexample_test.py
.venv/bin/python scripts/loewner_projection_test.py
.venv/bin/python scripts/export_branch_identity_test.py
.venv/bin/python scripts/fit_selection_test.py
.venv/bin/python scripts/final_release_test.py
.venv/bin/python scripts/eis_branch_test.py
.venv/bin/python scripts/eis_data_safety_test.py
.venv/bin/python scripts/prepublication_test.py
.venv/bin/python scripts/integration_test.py --output-root /absolute/new-test-directory
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/advanced_smoke_test.py
```

Tests recover known RC peaks, series L, signed contributions and diffusion parameters, and exercise rejection, exports, resume, parser independence and 20-label visual clearance. Do not run private holdout evaluation during maintenance.

Package using `scripts/build_release.py`; it uses an explicit allowlist, privacy scanning, deterministic ZIP members and a SHA256 inventory. Do not share the parent development directory or its environments, private adapters or validation logs.

## Privacy and portability

Use only the raw input paths and optional external profiles supplied for the task.
Do not search a home directory, private archive, other Skill, or mounted research
drive to recover missing configuration. Request missing inputs explicitly.
The source package contains no personal case context or machine-specific runtime.
An installed `.venv`, caches and generated analysis outputs are machine-local:
never redistribute the installed folder wholesale. Build the allowlisted ZIP.
Analysis outputs intentionally record user-supplied paths and metadata for audit;
they are not automatically anonymized. Do not redact authoritative checkpoints
in place or describe a clean software ZIP as proof that experimental exports are
safe to publish. Review a separate sharing copy when requested.

# Release 1.0.0

First 1.x source release, dated 2026-09-11. The numerical methods and pinned dependencies are inherited from 0.2.3.dev2 without retuning. This release consolidates independent EIS/DRT entrypoints, data-first workflow, automatic diagnostic figures, opt-in presentation figures, and privacy-safe source distribution.

## Scope and support

- EIS-only: parser/unit/sign checks, source-row provenance, modulus/phase, optional KK/Z-HIT, raw Nyquist/Bode diagnostics; no automatic physical circuit fit or DRT.
- DRT: primary TR-RBF, TR-NNLS comparison, regularization and sensitivity, adaptive series inductance, peak support, experimental-control checks, Loewner, conditional signed/GDRT and diffusion candidates, optional model-and-reduce.
- Data exports: CSV/JSON/Origin-ready tables with separate model branches and strict checkpoints. Presentation figures require a request; lack of trend figures is expected by default.
- Qualified platform: macOS arm64 with CPython 3.11. Windows/Linux entrypoints and platform-dependent lock entries are provided, but are not actual OS validation.
- EIS figures require installed Arial. DRT renderers support a disclosed DejaVu Sans fallback. Proprietary fonts are not shipped.

## Verification

Release qualification must use the frozen source and final archive: clean-environment bootstrap/runtime checks, the full regression suite, real CLI integration, basic and advanced synthetic smoke tests, privacy/allowlist checks and repeatable archive builds. Detailed receipts belong outside the distributed source package. No private holdout is used for release tuning, and reused development cases are not new blind data.

```bash
.venv/bin/python scripts/check_runtime.py
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s scripts -p '*_test.py'
.venv/bin/python scripts/integration_test.py --output-root /absolute/new-integration
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/advanced_smoke_test.py
```

Passing tests validates software behavior, not experimental accuracy or a probability that a DRT peak is real. A fresh installation test does not establish behavior on another operating system.

## Important limitations

1. KK success is not proof of linearity/stationarity. Missing repeats, amplitude or rest-time controls remain unassessed. Numerical completion is not scientific acceptance.
2. Weak/close peaks, drift, correlated noise, low-frequency tails and signed artifacts remain difficult. Conditional intervals do not quantify every source of uncertainty; stable spurious peaks can remain.
3. DDT is a series-impedance candidate family, not full parallel-admittance DDT. No automatic diffusion coefficient or unique mechanism is assigned.
4. Primary RC, recommended, signed and diffusion results are separate. Recommended reconstructions are not silently substituted into primary RC trends. Diagnostic-only spectra are not promoted simply to obtain a plot.
5. Metadata-matched trends can legitimately be skipped, including for clean synthetic examples with sensitive peaks. No unconditional publication-ready plot is promised.
6. SVG appearance depends on viewer fonts; full PNG and plot QA are the visual reference. Quick Look thumbnails may crop and some SVG renderers show thin heatmap-cell seams.
7. Not every instrument export/version is tested. Inspect new file formats and normalized units against the original source.

## Distribution and privacy

Use the allowlisted source ZIP and its SHA256 sidecar. The source includes no private case context, personal runtime paths, experimental datasets, environments, caches or font binaries. Third-party authorship/license notices remain. Python and dependencies install separately; their binary redistribution needs a separate license review. Archive hashes validate integrity, not publisher identity.

Scientific outputs intentionally contain source paths, hashes and metadata; a clean software package does not make those outputs anonymous. Share only independently reviewed copies of experimental results.

## Upgrade compatibility

Version/code changes alter execution fingerprints. Retain original software and outputs for old results; start a fresh directory for 1.0.0. No forced migration of earlier checkpoints is offered. Trend enablement and trend DPI can change within a matching run without numerical recalculation; all numerical/data/environment guards remain.

This is a locally prepared release artifact, not evidence of upload to any repository, marketplace or website.

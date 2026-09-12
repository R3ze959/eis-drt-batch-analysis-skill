# Public-data development round

_Research release 0.1.0; validation-driven development, not neural-network training._

## 📋 Changes and boundaries

| Counterexample | Transferable change | Boundary |
| --- | --- | --- |
| Native public text rejected | Explicit role/column mapping, aliases, headerless handling | No guessed units or signs; MAT remains an audited external adapter |
| Stable spurious peaks | Conditional residual and sampling stress evidence | Does not establish true peaks or eliminate all artifacts |
| Poor RC but useful signed fit | Separate reconstruction recommendation and original RC records | Smaller error alone is not physical acceptance |
| High-order rational overfit | Three-fold frequency holdout and actual-pole audit | Separate CV and exported-model identities |
| Crowded L-curve ticks | Compact scientific labels and bounding-box QA | No alteration of samples or curves |
| Discrete weights overlaid with density | Separate labeled axes and measures | Loewner weights must not be integrated as density |
| Empty normalized-table verdicts | UID join with final per-spectrum status | Reject missing or conflicting identities |
| Disabled models labeled as accepted | Explicit branch status in diagnostics | Not-triggered, disabled and failed remain distinct |
| Rendering uses up compute timeout | Separate recoverable rendering subprocess budget | Checkpoint retained, including process launch failure |

## 🔍 Method decisions

Second-round correction: CV compression uses the paired Loewner/shifted-Loewner pencil: the left subspace of `[L Ls]` and right subspace of `[L; Ls]`. The right basis uses `Vh.conj().T`, consistent with the SVD definition.[^2] Compressing `L` alone can lose the subspace needed for constant/inductive polynomial terms. A rectangular conjugate-completed pencil retains odd-frequency endpoints. Descriptor residues do not assume an invertible E matrix. Low-quality or inconsistent full-data refits cannot provide a reliable kernel verdict. The original pyimpspec matrix-rank export is retained only as a separately identified, non-holdout-validated diagnostic; this local correction does not silently patch the installed dependency.

Automated regularization can split or merge features; library documentation cautions against blindly accepting the automatically selected result.[^1] The perturbation module preserves original peaks and reports conditional stability separately. Read [peak-evidence.md](peak-evidence.md) before interpretation. Stable false peaks can remain.

The pinned pyimpspec Loewner conversion maps poles to absolute time constants. We additionally inspect generalized eigenvalues before this mapping. Candidate orders 2, 4, 6, 8, 12, 16, 24 and 32 are evaluated by three-fold frequency holdout retaining measurement endpoints. Release 0.1.0 uses three fixed dimensionless frequency references (log-frequency quartiles .25/.5/.75), audits each candidate's full-data refit, and applies the smallest-order preference among viable candidates within `1.2 × best RMS + 0.05 percentage point`, with median-reference tie-breaking. These are engineering heuristics, not confidence limits. Selected CV is not an unbiased external-validation estimate. Very fast cancelling poles can approximate series inductance; poles outside a one-decade-expanded measurement window are recorded as unresolved/nuisance asymptotes, not proof of unstable material dynamics. The CV model is NOT the exported full-data matrix-rank model. The latter receives its own pole audit only when its reconstruction is reproduced numerically; its weights are not thereby holdout-validated.

Primary RC is retained unless its error exceeds 2% or a nonseries response is detected, and signed/GDRT improves reconstruction RMS by at least 20%. Both errors and sensitivity flags are exported. This is reconstruction routing, not proof that the distribution is uniquely true. DDT remains a separate conditional topology family.

Release 0.1.0 adds an independent matched positive/signed comparison: same tau grid, modulus weighting, passive R/L/C nuisance bounds and three interleaved interior-frequency folds. Lambda stays within the user's range; the strongest smoothing within 5% of minimum prediction RMS is selected. Positive gamma is preferred within `1.10 × best prediction RMS + 0.2 percentage point`. This avoids rewarding an almost negligible residual gain from unconstrained negative lobes. Final routing requires an acceptable full-refit guard and at least 20% improvement over primary; a CV-selected positive reconstruction may replace the legacy signed recommendation within `1.10 × current RMS + 0.2 percentage point`, never worsening primary RMS. Legacy signed training RMS is not CV evidence. All candidate arrays, sensitivities and decisions remain separate. Primary peak perturbation evidence does not validate these new peaks.

The nonseries screen now also uses a three-point log-frequency interpolation-residual MAD estimate, normalized by impedance modulus. The per-point threshold is the larger of the old amplitude screen and three times that robust scale. This is a roughness-aware engineering screen, not a calibrated noise significance test; spectral curvature can inflate it and weak RL may be missed. The measured spectrum is never smoothed or changed.

## 📦 Data and exports

`00_overview/model_recommendations.csv` explains branch choices. `03_results/recommended/` holds explicitly labeled selected density and reconstruction; `primary_drt/` is not overwritten. `03_results/predictive_rc/` contains matched kernel curves and CV scores; `02_quality/advanced_reconstructions.csv` includes their frequency-domain fits. `02_quality/peak_perturbation_evidence.csv` and `loewner_frequency_validation.csv` are audit views. Per-spectrum JSON retains cases, poles, frequency splits, refit reliability, nuisance terms and complete sensitivity arrays.

Headerless example, after verifying units and sign against the source:

```bash
.venv/bin/python scripts/batch_drt.py /absolute/input.csv --output-dir /absolute/fresh-output \
  --text-headerless --text-delimiter comma \
  --text-columns '{"frequency":0,"zreal":1,"zimag":2}' \
  --frequency-unit hz --impedance-unit ohm --imag-convention zimag
```

Column indices are zero-based; `--text-header-row` is one-based. Equivalent source-level manifest fields override CLI defaults and are hashed. Explicit mapping requires an explicit delimiter. Do not guess differing schemas within one file.

## ⚠️ Validation scope

Previously inspected examples are development/regression data. Reserve new cell/condition files before tuning; freeze source hashes before their evaluation. Reusing a failed holdout for tuning makes it development data. No public experimental data, private reference data or machine paths are bundled. Keep licensing, URLs, hashes and raw-to-adapted checks in separate validation records.

Run all unittest `*test.py` modules, CLI integration and synthetic smoke tests. Success is implementation evidence, not uncertainty calibration or experimental acceptance.

### Remaining observed limitations

- Fresh noisy synthetic cases can still have a poor Loewner full-data refit despite good heldout error. The joint-pencil correction is necessary but is not a universal cure for numerical conditioning. Inspect `kernel_evidence_reliable` and `reliability_reasons`; an unreliable model must keep `kernel_compatibility=not-established`. Passing the engineering gate does not confirm physical elements.
- A pure series-L plus RC case can still challenge the residual nonseries screen; the specific consumed noise counterexample is now suppressed by the roughness floor. It remains a screening flag, not proof of an RL process. Signed inversion can improve a known RC spectrum and produce a stable but artificial negative lobe. Never infer a physical inductive mechanism from selection or lambda stability alone.
- Perturbation-supported extra peaks remain possible, and broad overlapping processes may not yield distinct maxima. The present stress ensemble is not a calibrated false-discovery procedure. Keep uncertain and weak features visible and qualified; do not remove peaks to improve truth-matching statistics.
- Text formats with arbitrarily alternating frequency order can be split into multiple sweep segments by default. A verified single unordered table has an explicit audited opt-in order policy in 0.1.0; do not silently reorder raw observations. Text ledgers now include every physical source line, even failed parses. Vendor formats still depend on their parser's available acquisition metadata.

Freeze numerical changes before evaluating a reserved set. A subsequent metadata-only export repair or documentation update must be separately identified and regression-tested, with numerical equivalence checked; it must not be described as validation of an identical complete source tree.

[^1]: pyimpspec. Distribution of relaxation times guide. https://vyrjana.github.io/pyimpspec/guide_drt.html

[^2]: SciPy. scipy.linalg.svd: U, singular values and Vh. https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

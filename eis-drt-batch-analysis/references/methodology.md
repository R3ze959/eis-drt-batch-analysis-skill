# DRT methodology and evidence rules

## 1. What ordinary RC-DRT estimates

The primary model used here is

\[
Z(\omega)=R_\infty+j\omega L+\int_{-\infty}^{+\infty}
\frac{\gamma(\ln\tau)}{1+j\omega\tau}\,d\ln\tau.
\]

For this convention, `tau = 1/(2*pi*f_peak)`. A peak position is a characteristic timescale; the basin integral of `gamma` over `ln(tau)` is an impedance contribution. Height is affected by broadening, overlap, discretization, and regularization and is not a substitute for area.

DRT is less dependent on a preselected equivalent circuit, but it is not model-free in the absolute sense. Non-negativity, the RC kernel, basis functions, derivative penalty, lambda, treatment of inductance, and the finite frequency window are all assumptions.

## 2. Acquisition validity comes before inversion

The desired system is causal, sufficiently stable over the sweep, and locally linear under the applied perturbation. There is no universal AC amplitude, rest time, points-per-decade, KK residual, or lambda threshold that is valid for every chemistry and cell.

Use the fastest discriminating checks available:

1. Compare at least two perturbation amplitudes under the same state to test local linearity; harmonic or Lissajous analysis is stronger when the instrument exposes it.
2. Repeat the spectrum or compare forward/reverse/time-adjacent sweeps to test drift.
3. Check temperature, OCV, current/voltage limits, contact pressure, and safety events.
4. Run complex Lin-KK in both Z and Y representations and inspect residual magnitude plus residual structure.
5. Use Z-HIT as a complementary reconstruction check, not a universal pass/fail oracle.

Important boundary: KK consistency is necessary evidence for a valid linear-time-invariant interpretation, but a pass does not prove linearity. Urquidi-Macdonald et al. showed experimental insensitivity of KK transforms to some linearity violations; modern amplitude/harmonic checks remain necessary.

For a large densely sampled screening batch, exploratory KK may be run on a declared deterministic equal-index subset when full-point runtime is prohibitive. The raw spectrum and DRT remain full resolution, both frequency endpoints are retained, and the source/screen point counts must be reported. This is a screening result only; selected publication spectra require full-point KK confirmation.

## 3. Preprocessing rules

- Retain original frequency points and acquisition order when the source parser exposes it; otherwise disclose parser sorting as a provenance limitation.
- Require finite positive frequency and finite complex impedance.
- Resolve whether the source column is `Zimag` or `-Zimag` from an explicit header/metadata/identity check. If still ambiguous, stop and require an override.
- Resolve Hz/kHz/mHz and ohm/mohm/kohm/ohm-cm2 explicitly. Do not infer area normalization from the magnitude alone.
- Do not silently average duplicate frequencies; repeats may represent drift or separate spectra.
- Do not delete an inductive region merely to force a non-negative RC distribution.

### Optional quality-driven model-and-reduce

Adaptive primary model selection first compares RC-DRT with and without an explicit series `L`; this is model selection, not raw-data deletion. `--model-reduce` is a separate optional step that evaluates raw DRT first, then fits only nuisance branches supported by edge-shape screening: series `L`, pure series `C`, semi-infinite Warburg, finite transmissive/short Warburg, finite blocking/open Warburg, and eligible `L+low-frequency-branch` combinations. The fitted edge resistance is used only during nuisance fitting and is not subtracted.

For each candidate, the script records an explicit Schlüter-inspired modulus-weighted reconstruction indicator:

\[
q=\log_{10}\left(\frac{\sqrt{\|W\Delta Z'\|_2^2+\|W\Delta Z''\|_2^2}}{\sqrt{2N}}\right),
\qquad W_i=1/\max(|Z_i|,\epsilon).
\]

This implementation declares its exact formula in the JSON; do not imply it is a universal threshold. From schema 5, compare `q` on the **original EIS** after adding the removed branch back to the reconstructed reduced EIS, using the same original modulus weights as baseline. Reduced-domain quality remains diagnostic only. First gate every candidate on convergence, quality improvement, parameter bounds, removed norm at most 0.85 of centered raw impedance, and peak stability; then rank eligible candidates. A rejected best-looking fit must not hide an eligible runner-up. Peak matching is threshold-aware maximum-cardinality one-to-one log-frequency assignment; default acceptance limits are 0.35 decade position shift and 50% basin-area change, with at least two-thirds of raw peaks stable. The raw branch is never overwritten.

## 4. Primary inversion and lambda policy

### Numerical conditioning is not denoising

From beta.3, primary TR-RBF (including model competition/reduction and conditional sampling) and secondary TR-NNLS solve a copy of the included spectrum divided by `median(abs(Z))`. All reconstructed impedance, gamma, and conditional mean/bounds are restored by that same positive scalar. Frequencies, masks, relative residuals and lambda are not rescaled; originals remain unchanged. Nonfinite or zero-scale inputs are rejected. `--impedance-scaling off` explicitly reproduces the unwrapped solver path, not the scientific default.

This uniform conditioning addresses finite-precision solver scale dependence seen in synthetic counterexamples; it does not change the relative contribution of frequency points, estimate their variance, remove noise, or recover information lost in measurement. It is not a new per-point weighting model. The primary fit uses uniform absolute real/imaginary errors; advanced signed/DDT retains its separately declared modulus weights. The JSON records the scale and distinction. Real conditional-sampler tests verify unit restoration, not uncertainty coverage. See [official-method comparison and adoption decisions](method-landscape.md).

### Primary model and search

The skill uses complex TR-RBF with Gaussian RBFs, first-derivative Tikhonov regularization, and FWHM shape coefficient 0.5. On a material high-frequency inductive trigger, it probes no-L and series-L at three lambda values around the fixed/probe value and reuses cached solves. Selection combines weighted reconstruction with a high-decade regression of `-Zimag` against angular frequency; a positive L and R² at least 0.90 support a series-L candidate. A reconstruction-only decision is labeled as such. The regression estimate is **not the fitted TR-RBF L**: pyimpspec 5.1.3 TRRBFResult does not expose its nuisance coefficients, so fitted-R/L exports remain null with a reason. These are automation heuristics, not universal instrument specifications.

The regularization parameter controls the fit-resolution tradeoff. Too little regularization can manufacture sharp/noise peaks; too much can merge real processes. The default policy therefore:

1. evaluates a fixed log-spaced lambda grid;
2. records reconstruction and roughness norms;
3. tries package mGCV from three distinct interior logarithmic starting values within the user range;
4. accepts mGCV only when at least two finite, nonboundary, in-range returns agree within 0.25 decade; out-of-range returns are retained as rejected diagnostics, not clipped into an optimum;
5. otherwise uses an interior, boundary-guarded L-curve corner estimated by normalized log-log distance from the endpoint chord, while retaining curvature as a diagnostic;
6. repeats DRT at selected lambda divided/multiplied by 10.

Fixed mode skips the full search and computes only the declared lambda and clipped ±1-decade neighbors. Robust-to-lambda requires at least three distinct cases. A clipped sweep is not equivalent to full ±1 decade and is explicitly boundary-sensitive. Undefined or guarded L-curve corners are also flagged. pyimpspec 5.1.3 mGCV has internal bounds [1e-7, 1]; touching either bound cannot establish an interior optimum. Its returned result does not expose optimizer success: start agreement is a heuristic, not a convergence certificate. Frozen legacy solver reproduction additionally requires explicit `--impedance-scaling off`; old fingerprints are not migrated.

GCV/mGCV performed well in the synthetic comparisons of Maradesa et al., but neither the literature nor pyimpspec guarantees automatic selection for every spectrum. The script therefore records the full decision path rather than hiding it.

### External-reference compatibility profiles

A compatibility profile is a frozen parameter set calibrated against supplied reference curves. It is not machine learning in the sense of a transferable physical model, and its training agreement is not validation accuracy. Keep it separate from the scientific default, lock train/holdout groups before tuning, score only within the measured tau window, and report detected versus actually fitted inductance separately.

Dataset-specific parameter sets and reference data are not bundled. An explicit
`--profile-json` can supply a frozen external configuration; the configuration and
its hash enter the run fingerprint. Reference reproduction does not establish
scientific validity or replace adaptive inductance analysis.

## 5. Peak finding and confidence classes

The script finds local maxima, separates their basins at intervening minima, and integrates each basin over natural `ln(tau)`. It reports both the full diagnostic integral and the part supported by the measured frequency window. The publication-facing area is null when the peak maximum lies outside that window. It assigns:

- `robust-to-lambda`: persists in every sensitivity case with small position span and acceptable area variation;
- `tentative`: persists in at least two cases but varies more;
- `unstable`: insufficient persistence;
- `minor-low-signal`: below the configured relative-height screen even if numerically persistent;
- `boundary-sensitive`: inside but within the configured edge distance (default 0.7 decade) of a frequency boundary;
- `inductive-overlap`: maximum lies in a measured inductive region and cannot be interpreted as an ordinary RC relaxation;
- `unsupported-outside-window`: peak maximum lies outside the measured frequency window.

The 0.7-decade edge rule is a conservative automation heuristic motivated by sampling/localization limitations, not a universal physical constant. Always show the measured-frequency boundaries on the DRT plot.

TR-NNLS with real-part L-curve selection is run as a secondary algorithm-dependence check. Peaks are matched one-to-one by log-frequency distance, with unmatched and local split/merge candidates retained. Agreement increases numerical confidence but still does not identify a mechanism.

All lambda comparisons now use the same threshold-aware maximum-cardinality matcher. A peak cannot be reused for two primary peaks. Connected peak neighborhoods whose peak counts change are `split-merge-ambiguous`; merely having nearby persistent peaks does not automatically imply merging. Diagnostic basin values remain exported, but do not make resolved single-process area claims for ambiguous features. No automatic cross-voltage process tracking or mechanism identity is implied by local P numbers.

## 6. Induction and diffusion routing

An ordinary non-negative RC-DRT plus a series `L` can represent wiring/high-frequency inductance. Before calling a response non-series, the workflow estimates the plausible high-frequency L, compensates only that term for the routing screen, and asks whether a material consecutive negative region remains below the highest measured decade. This avoids misclassifying a wide-band measurement merely because a valid series-L contribution extends over more than one decade. The screen never overwrites raw data. A series-L model is not adequate for a sustained residual finite- or low-frequency inductive loop generated by adsorbate relaxation, instability, or other signed responses. The skill therefore supplies two real calculations:

- pyimpspec Loewner analysis, reported as discrete non-negative RC and magnitude-valued RL branches;
- a continuous signed, regularized RC-kernel GDRT in which positive `gamma` is RC-like and negative `gamma` is RL-like.

The continuous result includes an algebraic resistance offset, non-negative scaled series-inductance and reciprocal-capacitance nuisance columns. Its lambda is chosen from a normalized interior L-curve; fits are modulus weighted and the derivative penalty is scaled in natural log(tau). Convergence/optimality, kernel rank/conditioning, peak support and lambda/coarser-tau-grid sensitivity are exported. `lambda_search` distinguishes a bracketed corner, edge-adjacent corner, degenerate curve and insufficient search. An interior choice alone is not a boundary pass. Truncated ±1-decade sensitivity is `range-incomplete`, even if all evaluated curves agree. Completed curves remain available for exploration; signed boundary cases require review and DDT boundary cases cannot enter accepted ranking. The signed offset is not a separately identified passive resistor. Sensitivity currently tests dominant absolute lobe position and integrated absolute mass, not unique recovery of every process. Sign remains mathematical evidence, not a unique electrochemical mechanism.

A diffusion tail can create a distributed pattern rather than a single RC peak. The diffusion branch compares three non-negative regularized **series-impedance** distributions:

- blocking/open finite diffusion: `coth(s)/s`;
- transmissive/short finite diffusion: `tanh(s)/s`;
- Gerischer/reaction-diffusion: `1/sqrt(1+j*omega*tau)`;

where `s=sqrt(j*omega*tau)`. R and series L are constrained nonnegative; failed or nonfinite optimizer outputs cannot be completed candidates. A fourth, **scalar semi-infinite Warburg** baseline fits `Z=R+j*omega*L+sigma/sqrt(j*omega)`. Its separable kernel has rank one; it has no identifiable tau distribution, no diffusion-time peak, and no gamma export.

Finite distributions enter ranking only with RMS at most 2%, an interior dominant turnover at least 0.35 decade inside the measured window, at least 80% supported absolute mass, stable lambda/coarser-grid checks and no lambda-search boundary hit. These explicit screening thresholds are not universal acceptance limits. Similar fits within max(0.2 percentage point RMS, 20% of the best RMS) are listed as competitive boundaries. No eligible candidate and ambiguous boundaries are valid outcomes.

This implementation is **not Song–Bazant parallel-admittance DDT**. It fits a series diffusion superposition to full EIS, without a separately identified RC branch. RC overlap or an incorrect electrode topology can invalidate the model even at low residual. Choosing a parallel-admittance topology or jointly separating RC+diffusion requires additional model development and physical justification, not an automatic relabeling. No diffusion coefficient is inferred without geometry and diffusion length.

## 7. Uncertainty and replication

The pyimpspec conditional credible band is conditional on the chosen method, lambda, basis, and noise assumptions. It does not include:

- uncertainty in lambda selection;
- alternative algorithm/model uncertainty;
- raw-data preprocessing uncertainty;
- sample-to-sample or repeat-spectrum variability;
- state drift.

For publication claims, combine conditional bands with lambda sensitivity, algorithm comparison, and physical replicates as appropriate.

The batch acceptance layer adds three non-substitutable controls when metadata permit:

- repeats: pairwise complex RMS plus one-to-one resolved major-peak stability. Require all major peaks to match in both directions at the existing 0.35-decade/50%-area limits. Compare only common-window peaks with relative height at least 0.1 (when available); minor, boundary, unsupported, inductive-overlap, unstable and split/merge features are not resolved major-peak evidence. A major mismatch is `repeat-peak-disagreement`; no resolved peaks is `repeat-peak-not-assessable`, not a pass. This prevents a large series resistance from hiding relaxation changes while avoiding rejection based on tiny noise peaks. At least three repeats are needed to identify one anomalous spectrum automatically;
- amplitude linearity: spectra at two or more `ac_amplitude_mv` levels are compared at otherwise matched state; agreement supports only local linearity;
- rest stability: group by distinct `rest_time_s`; compare the final two distinct levels and repeat dispersion within those levels. At least three levels are required, and a late plateau covers only the last two levels, not deliberately short-rest spectra.

Control groups require matched cell/sample, state, protocol and impedance basis. Explicit group names do not override conflicting metadata. Comparisons need at least five exact-matched frequency points, log-frequency tolerance 1e-5 decade and at least 80% point coverage of the larger spectrum. The symmetric complex RMS denominator is the mean of both pointwise moduli. No raw interpolation occurs. Insufficient overlap is not-assessable, never evidence of nonlinearity. Evidence links are per spectrum; protocol-wide transfer is not inferred automatically.

Execution (`status`), numerical quality (`numeric_status`), experimental controls (`evidence_status`) and intended display (`display_role`) are separate. Missing controls can coexist with supported numerical screening and an exploratory figure; scientific acceptance requires the relevant controls as well. A computed curve with a failed plot remains numerically preserved.

The within-sweep drift screen separates a monotonic acquisition-order residual trend from detrended random residuals. A trend span above 0.5% and trend/detrended-RMS above 1.5 is flagged. It is a screening signature, not proof that the cell itself drifted; temperature, contact changes, and model mismatch remain alternatives.

## 8. Mechanism assignment ladder

Do not map a frequency band directly to SEI, grain boundary, charge transfer, or diffusion. Assign a process only when controlled perturbations support it:

- temperature series and activation behavior;
- SOC/potential series with direction held constant;
- cycle/aging series with repeat cells;
- symmetric-cell, blocking-electrode, reference-electrode, or component-removal baselines;
- thickness/pressure/composition dependence;
- independent spectroscopy, diffraction, microscopy, or chemistry evidence.

Until then, report `P1`, `P2`, ... with position, area, stability, and a mechanism hypothesis explicitly labeled as such.

## Primary web and paper sources

- pyimpspec 5.1.3, [Distribution of relaxation times](https://vyrjana.github.io/pyimpspec/guide_drt.html) and [Kramers-Kronig testing](https://vyrjana.github.io/pyimpspec/guide_kramers_kronig.html).
- Wang et al., 2025, DRTtools tutorial and theory, [DOI 10.1021/acselectrochem.5c00334](https://doi.org/10.1021/acselectrochem.5c00334).
- Plank et al., 2024, comprehensive DRT review, [DOI 10.1016/j.jpowsour.2023.233845](https://doi.org/10.1016/j.jpowsour.2023.233845).
- Saccoccio et al., 2014, regularization in DRT, [DOI 10.1016/j.electacta.2014.09.058](https://doi.org/10.1016/j.electacta.2014.09.058).
- Wan et al., 2015, RBF discretization and DRTtools, [DOI 10.1016/j.electacta.2015.09.097](https://doi.org/10.1016/j.electacta.2015.09.097).
- Maradesa et al., 2023, regularization-parameter selection, [DOI 10.1149/1945-7111/acbca4](https://doi.org/10.1149/1945-7111/acbca4).
- Effat and Ciucci, 2017, probabilistic/regularized DRT, [DOI 10.1016/j.electacta.2017.07.050](https://doi.org/10.1016/j.electacta.2017.07.050).
- Liu et al., 2020, Bayesian Hilbert transform, [DOI 10.1016/j.electacta.2020.136864](https://doi.org/10.1016/j.electacta.2020.136864).
- Urquidi-Macdonald et al., 1990, KK stability/linearity limits, [DOI 10.1016/0013-4686(90)80010-L](https://doi.org/10.1016/0013-4686(90)80010-L).
- Schönleber et al., 2014, robust linear KK workflow, [DOI 10.1016/j.electacta.2014.01.034](https://doi.org/10.1016/j.electacta.2014.01.034).
- Macdonald, 2000, sampling/localization limits, [DOI 10.1088/0266-5611/16/5/324](https://doi.org/10.1088/0266-5611/16/5/324).
- Goh et al., 2024, experimental linearity assessment, [DOI 10.1149/1945-7111/ad3581](https://doi.org/10.1149/1945-7111/ad3581).
- Danzer, 2019, generalized DRT/peak analysis, [DOI 10.3390/batteries5030053](https://doi.org/10.3390/batteries5030053).
- Song and Bazant, 2018, distribution of diffusion times, [DOI 10.1103/PhysRevLett.120.116001](https://doi.org/10.1103/PhysRevLett.120.116001).
- Schlüter et al., 2021, quality-indicator-guided preprocessing/model reduction, [DOI 10.1002/celc.202100173](https://doi.org/10.1002/celc.202100173).

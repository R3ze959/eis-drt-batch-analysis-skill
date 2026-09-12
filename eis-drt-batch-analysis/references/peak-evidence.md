# Conditional peak evidence: scope and assumptions

`scripts/peak_evidence.py` adds an evidence layer beside the primary DRT and its
lambda sensitivity. It never deletes a detected peak, modifies the primary DRT,
replaces raw EIS or assigns a mechanism. In particular,
`perturbation-supported` does **not** mean that a true physical process has been
confirmed. It means a numerical lobe survived the declared stress cases.

## What is calculated

With the primary reconstruction `Z_fit` and paired complex residuals
`e_i = Z_i - Z_fit_i`, each wild-residual case uses

`Z_i* = Z_fit_i + s_i e_i`, where `s_i` is independently +1 or -1.

The real and imaginary residual at a frequency use the same multiplier. This
keeps the observed local variance pattern and the real/imaginary pair together.
The frequency grid is fixed. The default 12 cases use a recorded random seed.
Every case uses the same lambda, RC model, RBF settings and numerical conditioning
as the primary calculation, with conditional CI sampling disabled.

Four additional deterministic stress cases omit every fourth interior frequency
point in turn. Both frequency endpoints are retained. These fits test sensitivity
to sampling density and basis placement, not experimental repeatability. They
are derived calculation inputs only; raw observations and the primary curve
remain unchanged. No raw interpolation is performed.

The reference peak and its two neighboring valleys define a fixed basin, clipped
to the measured tau window. Each refit is evaluated using:

- one-to-one peak matching within 0.35 log10 decade;
- the fixed-center contrast `gamma(center) - max(gamma(left), gamma(right))`;
- the basin integral over natural `ln(tau)`;
- the 90th percentile of matched position shifts;
- local split/merge counts, retained as diagnostics.

Interpolation here evaluates calculated DRT curves at fixed diagnostic tau
locations; it never interpolates measured EIS. Keeping the basin fixed prevents
moving valley boundaries from masquerading as changes in integrated response.
It does not establish that the basin contains exactly one physical process.

## Decision rules and budget

The default development heuristics require at least 80% match and positive
contrast among wild cases and 75% among deletion cases; the position-shift 90th
percentile must be at most 0.25 decade and the area interquartile range at most
50% of the original fixed-basin area. Failed, skipped or budget-limited cases
prevent a complete support verdict. Peaks without a two-sided measured-window
basin are not assessable. A maximum absolute lag-1 correlation above 0.5 in the
modulus-normalized real or imaginary residual is flagged because independent
frequency multipliers then have questionable interpretation.

`stress_status` separately records whether the lobe itself persisted, and
`noise_model_status` records the serial residual flag. For example, both true
lobes of a weak double ZARC can persist while regularization bias makes the
residual noise model questionable. Those peaks remain in the output; they are
not treated as absent or false merely because noise-model support is withheld.

These numerical screens were not calibrated as universal acceptance thresholds.
The estimated contrast standard deviation and contrast/standard-deviation ratio
are exported as diagnostics, not converted to a p-value or real-peak probability.
The default budget is 12 perturbation fits + 4 deletion fits, at most 250 original
frequency points, and 30 seconds checked between solves. An omitted primary
reconstruction costs one extra fixed-model solve. One ongoing library solve is
not interrupted by the elapsed-time check; the caller's spectrum subprocess
timeout remains the hard outer boundary. Exceeding the point budget returns an
explicit not-assessable result rather than silently thinning the input spectrum.

## Statistical boundary

Wild residual resampling is appropriate as a model-conditional stress mechanism
when errors are mean-zero and their per-point variances can differ. The official
[fANCOVA `wild.boot` documentation](https://search.r-project.org/CRAN/refmans/fANCOVA/html/wild.boot.html)
describes this local variance preservation and cites Wu (1986) and Mammen (1991).
This implementation adapts that multiplier idea to paired complex residuals; it
does not inherit a theorem establishing confidence coverage for nonnegative,
regularized, selected-peak DRT inversion.

Residuals can contain smoothing bias, kernel mismatch and drift; overfitting can
make them too small. Most importantly, a false peak already present in `Z_fit`
can persist when that same fit generates the perturbations. Interleaved deletion
also shares the measured spectrum and is not independent experimental evidence.
Thus neither high persistence nor agreement with lambda sensitivity proves the
peak is real. Independent frequency prediction/model checks, repeated spectra,
and controlled perturbations provide complementary evidence. A broad numerical
lobe can contain two unresolved physical processes even if all stress cases agree.

## Development and independent validation

Development uses only synthetic single/separated-double/weak-double/overlapping
ZARC cases, a high-noise single ZARC, a noisy resistor, and noise seeds
101, 202, 303 and 404. The allowed public `2XZARCequal.csv` example is a declared
development example, not a holdout. Its approximately 0.00223458 s spurious peak
can remain conditionally perturbation-supported; this is an explicit known
counterexample to interpreting persistence as truth, not hidden by a bespoke
height threshold. Its main lobes and all original peaks remain in the output.

`peak_evidence_test.py` exercises input preservation, scale covariance, budgets,
invalid inputs and declared claim limits. `--development-matrix` reports counts
of retained known components and unmatched supported lobes for the six case
families and four development seeds. Overlapping components are tracked
separately because failure to split an unresolved broad lobe is not equivalent to
losing a known isolated peak. No independent validation seeds are used here, and
these small development counts are not a transferable accuracy estimate.

The 24-case development matrix uses 51 points, eight wild cases and four deletion
cases per spectrum. It retained all 4 known single-ZARC, 8 separated-double-ZARC
and 4 high-noise single-ZARC main lobes as conditional support. It also left 4,
5 and 3 unmatched supported side lobes in those respective families. All 8 weak
double-ZARC true components were retained in the primary output and passed the
stress checks, while correlated residuals withheld the final support label.
All four noisy-resistor cases had zero supported lobes. These results demonstrate
useful stress detection and substantial remaining false-support risk simultaneously.
For overlapping broad ZARCs, nominal-component matching is not a meaningful
standalone measure of physical resolution: the merged/broad maximum can move
under perturbation, and a persistent side lobe may still remain. The regression
suite keeps this counterexample explicit instead of equating support with truth.

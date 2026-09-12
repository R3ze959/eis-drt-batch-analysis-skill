# Attribution and third-party components

## This source release

EIS DRT Batch Analysis contributors, 2026. The original workflow, bundled acquisition
parser snapshot, style snapshot, tests and release tooling are offered under
**GPL-3.0-or-later**; see the full LICENSE. This choice does not relicense third-party
components or claim ownership of their code, fonts, names or research methods.

This ZIP contains source code and documentation, not vendored pyimpspec, numerical
library binaries, a Python interpreter, DearEIS, Origin, fonts or experimental data.
The parser and style are snapshots of the author's local EIS analysis workflow;
they require no external Skill or private data at runtime.

## Direct scientific dependencies

| Component | Role | Upstream license/source |
| --- | --- | --- |
| pyimpspec 5.1.3 | DRT, KK, Z-HIT, Loewner and vendor reading | [GPL v3 or later; upstream source](https://github.com/vyrjana/pyimpspec) |
| CVXOPT 1.3.3 | Convex numerical solver | [GPL v3 or later; copyright notice](https://cvxopt.org/copyright.html) |
| NumPy 2.4.6 | Numerical arrays | [BSD-3-Clause and bundled notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy 1.17.1 | Optimization and signal processing | [BSD-3-Clause and bundled notices](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| Matplotlib 3.11.1 | Figures and vector/raster export | [Matplotlib license and bundled notices](https://matplotlib.org/stable/project/license.html) |

The exact installed package closure is in requirements.lock.txt. The machine-readable
`DEPENDENCIES.json` inventories distribution names, versions, declared license metadata
and project URLs from that baseline. Upstream metadata can use broad identifiers;
the actual installed distribution's license files remain authoritative.

Windows additionally installs the conditional pandas dependency
[tzdata 2026.3](https://pypi.org/project/tzdata/2026.3/) (declared Apache-2.0).
It is locked and included in active-platform runtime fingerprints, not vendored.

pyimpspec itself retains upstream algorithm/source attributions in its LICENSES
directory (including its Loewner/DRT sources). Do not discard those notices when
redistributing installed libraries or a bundled application. This source-only
package does not copy their implementation files into its own scripts.

The full GPL text and retained direct-component notices are included under
`licenses/` where applicable. Any redistribution of an environment, binary build,
container or modified dependency needs its own complete corresponding-source and
license review; this ZIP's source manifest is not an approval for arbitrary rebundling.

No font binaries are distributed. EIS-only plotting requires installed Arial and
reports a figure failure if it is unavailable; exported numerical data remain intact.
DRT trend rendering can record an explicit fallback to Matplotlib's bundled DejaVu
Sans as described in references/plotting.md. Mathematical plots and exported
experimental data are not relicensed merely by being produced here.

## Research attribution

Methodological papers and official package documentation are linked in
references/methodology.md. Cite the numerical methods actually used in a manuscript;
do not imply that using this Skill validates a unique mechanism or obtains author
endorsement. This release does not incorporate borrowed reference datasets.

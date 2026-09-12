# Runtime and dependency boundary

This release uses CPython 3.11 with the exact direct/transitive closure in
`requirements.lock.txt`. Bootstrap creates a new isolated environment; it does
not share system site-packages or inherit PYTHONPATH during dependency setup.
Bootstrap disables disk pip configuration and installation-destination environment
overrides. Explicit index/proxy/certificate environment settings are retained.

- pyimpspec 5.1.3: primary DRT, vendor parsing, KK, Z-HIT and Loewner.
- cvxopt 1.3.3: numerical dependency required for the tested inversion path.
- NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1: arrays, checked advanced models, rendering.
- DearEIS is optional for independent GUI review; do not include GUI installation in core requirements.
- Origin is not required; exported CSV tables may be imported into it.

Use `scripts/check_runtime.py` and `python -m pip check` after installation.
Do not silently switch numerical dependencies or fall back to system Python.
The POSIX wrapper defaults to the package-local .venv; an explicit DRT_PYTHON can
select a different validated interpreter. On Windows invoke Python scripts directly.

The installed macOS arm64/Python 3.11 environment is the release baseline, not a
claim of Linux/Windows or every instrument-format validation. The lock specifies
versions, not wheel hashes; pip downloads platform-appropriate distributions.
Fresh-run numerical fingerprints and test reports record every actual locked
dependency version and Python implementation/version/platform, not only the lock
file bytes. An optional GUI installation does not invalidate numerical resume.

No external Skill is required: the acquisition parser and the plotting style are
bundled snapshots. Their versions/hashes enter the numerical/plotting provenance.
Third-party libraries are installed, not embedded in the ZIP.

The lock contains 33 cross-platform pins plus `tzdata==2026.3` for
`sys_platform == "win32"` (pandas' platform dependency). Runtime checks and numerical
fingerprints evaluate markers and include every active installed version. Windows
and Linux marker closure is statically checked against dependency metadata; this
does not test platform wheels, GUI fonts or numerical behavior on those systems.

Official upstream documentation:

- [pyimpspec DRT](https://vyrjana.github.io/pyimpspec/guide_drt.html)
- [pyimpspec KK](https://vyrjana.github.io/pyimpspec/guide_kramers_kronig.html)
- [DearEIS API and batch distinction](https://vyrjana.github.io/DearEIS/guide_api.html)

See the root LICENSE and THIRD_PARTY_NOTICES.md for attribution and redistribution
boundaries. Installing an optional GUI does not change the scientific acceptance rules.

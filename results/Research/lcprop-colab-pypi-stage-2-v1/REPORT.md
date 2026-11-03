# PyPI-native interactive LC notebook — Stage 2

Implemented a separate notebook pinned to the qualified production distribution
`lcprop[colab]==0.3.0`. **Ready for local review; stop before hosted qualification.**
No commits, pushes, uploads, scientific simulations or engine changes.

## Files

- notebooks/lc_static_pypi_cpu.ipynb
- notebooks/lc_static_pypi_cpu.md
- tests/test_lc_pypi_notebook.py

Exact SHA-256 identities: candidate-identities.json. Existing Git-pinned notebook,
bootstrap and packaged scientific/controller helpers match HEAD exactly. Existing
unrelated tracked modifications and protected evidence remain untouched.

## Design

Two collapsed/form code cells: an embedded setup implementation and an eight-line
user-facing initializer. Run all defines the implementation, then prepares widgets;
it does not run the solver. The implementation stays inside the notebook to avoid
any external helper or bootstrap download. Ordinary Jupyter may expose code despite
Colab metadata. Intro explains the standard third-party-code warning without treating
it as a scientific error. No Drive mount, GPU, Qt or custom JavaScript.

Setup uses isolated pip with production PyPI and exact version 0.3.0 if the package
is absent. A different already-loaded engine requires restart. Compatible loaded
numerical libraries are reused; mismatched loaded/disk version metadata is rejected.
Routine pip output is captured and failures show bounded stderr. An existing 0.3.0
core-only installation receives actionable instructions for missing extras.

Before enabling a new interface, the production wheel metadata and downloaded bytes
must match SHA-256
`9ab34f4ec244716b3e6f0431e4cac1e0d8a1ebeb17a7064c6422f1ed6b4e3d60`.
All 267 installed runtime files must match that wheel; loaded LCProp modules must
resolve within its package path. No engine reconstruction or alternate solver.
Provenance prints exact version, package origin, digest and dependency versions.
Existing packaged export behavior is unchanged.

The controller is reused on repeated setup, preserving completed snapshots and edited
controls. Errors raise without replacing app. Prior displayed controls are not removed
by a subsequent failed setup; resolve errors before further execution. Arbitrary
in-memory monkeypatches outside setup are not covered by on-disk identity checks.

## Validation

**10 focused tests passed** on the actual notebook cells: real installed-controller
initialization; canonical request equality; repeated snapshot preservation; published
and downloaded hash errors; installed-file tampering; missing dependencies; loaded
engine/dependency mismatch; synthetic successful and failed installation dispatch;
short/collapsed setup and clean notebook outputs. Install-dispatch tests mock pip and
never install or solve. Tests use the retained Stage 5B wheel as an offline fixture.

A separate local check exercised actual anonymous production PyPI metadata and wheel
download against the already-qualified Stage 6A installed environment. All 267 files
verified, widgets initialized, canonical request matched, and repeated setup retained
the exact prior object/snapshot. Solver methods and unexpected pip installation were
trapped. See real-initialization.json. This check preceded moving the IPython display
import to after dependency setup; the final 10-test run covers the resulting cells.

No Qt/LaunchPlane/CuPy import; Python 3.10 grammar and whitespace checks pass. Existing
scientific parity is preserved by unchanged 267-file runtime identity and identical
qualified request, not a new simulation. Stage 6A supplies independent fresh production
wheel/sdist installation evidence. This milestone did not repeat those installations.

## Scope and remaining checks

All beam/voltage controls, fresh bias, longitudinal plots, width curves, budgets and
immutable snapshot/export behavior are the unchanged packaged implementation. Only
pressing Run requests a scientific calculation. Constructor/draft edits sample the
launch preview, which is not a solver execution. Initial scientific preset and
normalization are not reimplemented.

No hosted execution yet. Fresh Colab installation, browser collapsing/layout, runtime
reuse diagnostics and user-driven Run/export still require hosted qualification.
Synchronous execution, between-slice timeout boundaries, exploratory configurations
and temporary session storage retain existing limitations. No claim of stationary
soliton formation or arbitrary-parameter accuracy. An immutable public Open in Colab
link requires a separately approved commit/publication; this candidate is unpublished.

## Stage 2A acceptance

Accepted source identities were reverified unchanged before staging. The actual final
notebook cells passed live anonymous production PyPI metadata/download verification,
all 267 installed identities, packaged widget initialization, canonical request equality
and repeated app/snapshot ownership. Installation and solver calls were trapped.
All **10 focused tests passed again in 1.65 seconds**. No scientific simulation or
installation occurred. The historical Git-pinned notebook/bootstrap and packaged
controller/helpers match HEAD; unrelated tracked modifications remain untouched.

| Accepted file | SHA-256 |
|---|---|
| `notebooks/lc_static_pypi_cpu.ipynb` | `5ca07fb69d058cc57e734f869e0e621704cf8f1043dcdcfd85c4cafdf5658808` |
| `notebooks/lc_static_pypi_cpu.md` | `f8be1ca3a12a05480c3762cd0848e8a14c6f5b57b9707cb8b353a577195756a2` |
| `tests/test_lc_pypi_notebook.py` | `e47dce738b0624de8f63955eee295a11f7b3829f5168cdfa1701e884195ca452` |

Only the notebook, usage document, focused test file and this report belong to the
acceptance commit. Supplementary logs and JSON remain local. The focused tests require
the retained Stage 5B wheel fixture and installed qualified dependencies; they are not
a self-contained test bundle for a fresh clone. This does not affect notebook setup,
which obtains its wheel from production PyPI.

After separately authorized publication, hosted qualification must cover a fresh Colab
CPU runtime, setup and provenance, controls/browser layout, repeated setup preserving
a real completed snapshot, and an explicitly authorized preset Run/plot/export check.
No hosted qualification or new numerical parity claim is made here.

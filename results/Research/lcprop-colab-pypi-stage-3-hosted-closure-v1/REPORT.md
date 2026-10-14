# Stage 3 — PyPI-native Colab hosted acceptance closure

**Hosted setup and interface-initialization acceptance closed**, within the evidence
scope below. This is not a new numerical or scientific-workflow qualification.

## Release and notebook identity

- Published notebook commit: `8ca30a25242e31682bf55299d142f8d264a0d912`.
- Path: `notebooks/lc_static_pypi_cpu.ipynb`.
- SHA-256: `dc2860509f5f5caba8780dbba0f5e8fa04961cf777d62b2ffe6f21c6dab66151`.
- Package pin: `lcprop[colab]==0.3.0`.
- Notebook import: `from lcprop.notebook.interactive_lc import InteractiveLC`.

Anonymous raw GitHub retrieval returned HTTP 200 and matched the committed bytes
exactly during this closure review. The current local notebook also matches.
[Immutable Colab notebook](https://colab.research.google.com/github/mcroning/LCProp/blob/8ca30a25242e31682bf55299d142f8d264a0d912/notebooks/lc_static_pypi_cpu.ipynb).

## Hosted observations — user reported

The user reports that installation and initialization succeeded, reinitialization
preserved modified beam controls, only one interface was displayed, and initialization
did not execute the scientific solver. These are accepted hands-on hosted observations,
not independently instrumented results. No new hosted logs, screenshots, package
inventory, scientific arrays or completed-run exports accompanied this request.
Exact hosted Python/dependency versions and runtime date are therefore not established.

The observations validate the reported setup, control reuse and same-cell rendering
experience. They do not establish preservation of a completed scientific snapshot
in a hosted runtime: no such completed run was reported for this acceptance.

## Independently retained local evidence

Stage 3 tests: **9 passed in 1.89 seconds**; acceptance rerun: **9 passed in 1.98
seconds**. Core-only import check passed with optional imports blocked. This closure
reviews those retained results without rerunning simulations, tests or installations.

The tests exercise actual notebook cells with pip/solver calls trapped, canonical
request equality, compatible controller reuse, edited budget preservation, completed
snapshot sentinel retention, incompatible version/app handling, and one clear/display
sequence. The isolated candidate-source test proves lazy package initialization and
identical old/new class imports without optional notebook/Qt/CuPy imports. The
published 0.3.0 controller was used by the notebook initialization checks.

Evidence references and SHA-256 identities are in `qualification.json`. Detailed
local logs remain outside this commit; the prior Stage 3 report is already committed.
The production Stage 6A qualification and earlier scientific comparisons remain
unchanged. No new numerical comparison, convergence claim, or simulation was made.

## Stable import and future release boundary

`from lcprop.notebook import InteractiveLC` is committed source functionality intended
for a future package release. The exact qualified 0.3.0 wheel's notebook initializer
was inspected and does not expose it. This closure does not modify or republish that
wheel. The current notebook correctly retains the compatible submodule import.

## Limitations

- Version conflicts instruct users to export results and restart. No live module
  reload or automatic restart. Distribution metadata cannot detect every stale module
  after external/manual package replacement; restart after such changes.
- The app, modified controls and completed snapshot live in kernel memory. Restart,
  session expiration or disconnect can lose them; export before ending the session.
  Reinitialization reuses the object, not disk-backed checkpoint recovery.
- Clearing output and displaying the same widget addresses the current cell. Multiple
  copied cells may display multiple views; this is not global widget deduplication.
- Synchronous execution, between-slice time budgets, transient Colab storage and
  exploratory-configuration restrictions remain unchanged.
- No new hosted Run/export cycle or completed-result snapshot reuse was qualified.
  Such checks remain separate from this setup/control acceptance.

## Closure and preservation

Only this report and its compact qualification metadata are proposed for the closure
commit. No notebook, controller, scientific engine, dependency or version modification.
No builds, simulations, installations, package uploads or pushes. Unrelated tracked
changes and protected scientific evidence are preserved. Publishing this documentation
commit requires separate authorization; the accepted notebook is already public.
A future release exposing the shorter import remains separately gated.

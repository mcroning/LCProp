# Stage 3 — stable notebook interface

**Implemented and locally validated; unstaged and uncommitted.** No scientific
simulation, package installation/build/upload, commit or push occurred.

## Exact changes

- src/lcprop/notebook/__init__.py: PEP 562 lazy InteractiveLC attribute, __all__ and
  discoverable __dir__. Attribute lookup caches the same class from the existing
  submodule; unknown names raise AttributeError. No import occurs on package load.
- notebooks/lc_static_pypi_cpu.ipynb: short guarded install cell containing the exact
  `%pip install -q "lcprop[colab]==0.3.0"` command; initialization keeps the existing
  published import path. Version errors explicitly instruct export and restart.
  Clear current output, then display the reused app widget once.
- notebooks/lc_static_pypi_cpu.md: import/release compatibility and display limitations.
- tests/test_lc_pypi_notebook.py: execute IPython-transformed real cells; intercept pip
  magic without installation; verify request, snapshot/budget reuse, errors, display.
- tests/test_notebook_lazy_interface.py: isolated-process lazy/both-path checks with
  an import blocker for optional dependencies and unknown-attribute behavior.

No packaged controller implementation or scientific-engine file was changed.
The Git-pinned qualification notebook/bootstrap is byte-identical to the pre-task
state. All preexisting tracked bytes outside this scope, including the two unrelated
modifications, are unchanged. Existing untracked evidence was not altered.

## Validation

**9 tests passed in 1.89 seconds.** The actual notebook cells are transformed with
IPython's transformer; the pip magic is mocked, not replaced in notebook source.
Controller and LocalRunner solver entry points are trapped. The default request
matches the canonical packaged preset. Repeated initialization/full setup keeps
app identity, completed snapshot sentinel and edited time budget. Version/session
mismatches and incompatible apps reject without replacement. Installation failure
leaves the previous app intact. No Qt/LaunchPlane/CuPy imports.

The candidate lazy initializer was tested explicitly from the checkout in an isolated
child process, not accidentally against installed 0.3.0. Importing lcprop.notebook
leaves interactive_lc absent from sys.modules. Both imports yield the identical class:
`from lcprop.notebook import InteractiveLC` and the existing submodule path.
The same test passed in the fresh core-only Stage 6A environment without notebook
libraries. A meta-path blocker rejects any attempted ipywidgets, Matplotlib, IPython,
Qt, CuPy or LaunchPlane import, including during class resolution. Construction is
intentionally not attempted there; that requires optional UI dependencies.

Python 3.10 grammar (after IPython magic transformation) and whitespace checks pass.
The initial test-harness run had three NameErrors because plain exec lacked the
IPython get_ipython global; the fixture now supplies the actual shell instance.
This was test harness behavior, not a failure in a notebook kernel. Final tests pass.

## Display and version semantics

Use normal display, with clear_output(wait=True) targeting the current output area.
The focused test observes exactly one clear and one display of the same app.widget
on rerun. The same controller owns existing controls, completed snapshot, plotting
and export callbacks; no handlers are registered again. No custom display handle.
Ordinary Jupyter/Colab current-cell output semantics should avoid accumulation;
actual hosted browser rendering was not commissioned here. Copying initialization
into separate cells may show multiple views, outside this same-cell guarantee.

The small guard before %pip prevents replacing files beneath a detected incompatible
loaded distribution or notebook session. The initialization repeats the version
check and never reloads modules. Arbitrary external package replacement in a live
kernel cannot always be inferred from distribution metadata; users must export and
restart after such changes. Normal %pip honors pip configuration (fresh Colab uses
production PyPI); no custom dependency resolver or release-integrity code is added.

## Future releases and limits

Published 0.3.0 does not expose the new alias. This notebook deliberately retains
`from lcprop.notebook.interactive_lc import InteractiveLC` and its 0.3.0 pin. A future
reviewed package release can introduce the alias; then change the notebook pin and
import once. Do not update a published 0.3.0 artifact in place. The added initializer
changes package-file identity and needs future distribution qualification.

Beam/workflow/diagnostic controls can evolve inside InteractiveLC without changing
this installation/launch structure as long as the no-argument constructor and widget
property remain supported. This is an interface compatibility condition, not evidence
that those scientific features are implemented. No new beams, dragging, workflows,
solver APIs, equations or tolerance changes. Existing snapshot/export semantics are
unchanged. Future scientific additions still need their own qualification.

Remaining checks: hosted Colab install and current-cell rerender behavior; normal
Jupyter browser display; separately authorized future package release and notebook
publication. No scientific simulation was needed or performed for this milestone.

## Acceptance review

All recorded candidate hashes were verified unchanged. Nine focused tests passed
again in 1.98 seconds; the isolated import test also passed in the existing core-only
Stage 6A environment with optional imports blocked. The controller and Git-pinned
notebook/bootstrap remain byte-identical to HEAD. The notebook continues to use the
existing production 0.3.0 import path, not the unreleased alias. Tests verify canonical
request equality, reused app/controls/completed snapshot, version errors and one
clear/display sequence per initialization. No solver, installation or build occurred.

Accepted implementation identities (supporting logs/JSON remain local):

- `src/lcprop/notebook/__init__.py`: `ba9dccc656636b633424c2ccf789b438d6f41bb1eaa83829304ec73f13b76107`
- `notebooks/lc_static_pypi_cpu.ipynb`: `dc2860509f5f5caba8780dbba0f5e8fa04961cf777d62b2ffe6f21c6dab66151`
- `notebooks/lc_static_pypi_cpu.md`: `c23597725bb2cdb521428daa05048d0d672930f1989eaff22e4999bafeec4291`
- `tests/test_lc_pypi_notebook.py`: `c2d837affa2caa8f5fede577bd0b18f7d124d1b32ee4d5d75ad48a6d2b716657`
- `tests/test_notebook_lazy_interface.py`: `4e5c050b5e69da5164082630942aa6912790d0a31e9f0dd116972e63a135f6fb`

The acceptance commit contains only these five implementation/documentation/test
files plus this report. Hosted Colab/Jupyter visual output replacement and a later
qualified package release containing the alias remain separate steps. No push is
authorized by this acceptance.

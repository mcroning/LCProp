# PyPI-native LC notebook — researcher-facing simplification

**PASS local validation; candidate unstaged and uncommitted.**

Changed only notebooks/lc_static_pypi_cpu.ipynb, its .md usage guide and
 tests/test_lc_pypi_notebook.py. Exact hashes are in candidate-identities.json.

The advanced implementation is replaced by a 13-line ordinary pinned pip cell
and a 14-line initialization cell importing the packaged InteractiveLC directly.
Installation uses production PyPI and lcprop[colab]==0.3.0, including missing extras
on an existing core installation. Standard pip output/errors remain visible. No
custom wheel fetching, hash inventories, installed-file scans or numerical-dependency
checker remains. Release qualification evidence stays separate and unchanged.

Initialization reuses app when it is an InteractiveLC instance, preserving its
completed snapshot, edited controls and exports. Existing incompatible app objects
are rejected without replacement. A loaded incompatible LCProp distribution or a
recorded incompatible notebook-session version produces a restart instruction.
There is no automatic restart. As documented, manually replacing package files in a
live kernel can leave stale in-memory modules that ordinary version metadata cannot
identify; restart before using this notebook after manual package replacement.

## Validation

- Seven focused tests passed in 1.33 seconds, exercising the actual cells: pinned
  installation dispatch; canonical request equality; direct packaged initialization;
  repeated initialization and repeated full setup retaining snapshot and budget;
  incompatible loaded/session versions; installation failure preserving prior app;
  incompatible app preservation; short cells and absence of integrity machinery.
- Executed the actual installation cell in a fresh disposable Python 3.12 environment
  under .codex-work/colab-pypi-simple/env. Production PyPI successfully installed
  lcprop 0.3.0 with its colab extra (NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2,
  ipywidgets 8.1.9, IPython 9.17.1). No manual uploads or helper scripts.
- Actual initialization then passed with solver methods trapped; request exactly
  matched the packaged qualified preset. Repeated initialization preserved the same
  app and completed-snapshot sentinel. No solver was executed. The constructor
  samples the existing launch preview only. See fresh-setup.log.
- Headless tests imported no Qt, LaunchPlane or CuPy. Python 3.10 grammar and scoped
  whitespace checks passed. Index remains empty.

No scientific algorithms, packaged controller, requests, normalization, tolerances,
time budgets or export implementation changed. Existing scientific parity evidence
remains applicable by unchanged source and canonical request identity; there was no
new simulation. The Git-pinned qualification notebook/bootstrap and all preexisting
tracked file bytes outside the three-file scope are unchanged, including unrelated
modifications. Protected evidence and untracked user inputs were not modified.

Hosted Colab browser presentation and user-driven Run/export qualification remain
pending; local install success is not hosted commissioning. The existing public
notebook is unchanged until separately authorized publication. No commit or push.

## Stage 2C acceptance

The recorded implementation identities were verified before acceptance. Seven focused
tests passed again in 1.28 seconds with scientific execution trapped. The two setup
cells remain short and pin lcprop[colab]==0.3.0. Repeated initialization preserves the
same app, completed snapshot sentinel and selected budget. Incompatible loaded/session
versions are rejected. No external helper, embedded manifest or runtime integrity scan.
The packaged controller and Git-pinned notebook/bootstrap match HEAD unchanged.
No installation, simulation or build was repeated for this acceptance.

Only the usage guide's publication-status sentence was updated for the accepted commit.
The four accepted files are the notebook, its usage guide, focused tests and this
validation report; supporting logs and candidate-identities.json remain local evidence.
Final implementation identities:

- `notebooks/lc_static_pypi_cpu.ipynb`: `023c04f6b63882bd804961ea1af672685bd153c8bbc7b48cf9a133127db4cd09`
- `notebooks/lc_static_pypi_cpu.md`: `bd7beb7c47cb044503c38667a9c06971cbd5d538a6e27a87296e95e3aadb51fd`
- `tests/test_lc_pypi_notebook.py`: `0a2de2a540a15196837b9ae0add5d6e7547003e2b571221b431c23f6630cc710`

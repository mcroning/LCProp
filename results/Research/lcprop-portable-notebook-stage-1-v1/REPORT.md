# Portable notebook Stage 1 — validated

## Stage 1A provenance remediation — current validation

The review's single blocker is fixed within the notebook and its tests. The
sections below this addendum retain the original Stage 1 evidence; the current
candidate identities and validation are in `stage-1a/` (the original inventories,
logs and saved example have not been overwritten).

Changed files: `notebooks/pr_reduced_cpu.ipynb`,
`tests/test_portable_pr_notebook.py`, and this report. README and Product source
are unchanged. No architectural or scientific-engine changes were required.

Before execution the notebook deep-copies the exact request passed to the Local
runner. Only a successful completed run publishes a frozen `CompletedRun` record
containing detached copies of that request, the result, the Product `RunData`
arrays, environment metadata and elapsed time. Updating parameter or plot globals
cannot change those copies. An unsuccessful later run does not replace the last
successful record. This is a small-demo copy policy, not a new large-grid storage
architecture; nested snapshot objects are not a tamper-proof security boundary.

`save_demo()` captures that one record and reads no independent request/result,
plot-array, figure or environment globals. It regenerates the exported figure
using the same snapshot renderer as the display cell, then closes the export-only
figure. It checks for a completed record before creating any directory. The
notebook explicitly explains that exports belong to the latest successful run;
edited parameters require execution to become a new saved experiment.

Validation: **14 passed, 0 skipped** focused notebook tests, including actual
fresh-kernel CPU execution; **24 passed** existing PR execution/headless/transport
regressions. The five new controls cover:

- Changed parameters after execution: save retains the executed request/result.
- Repeated execution without plotting: save uses the new snapshot and regenerates
  the new figure, ignoring the previous displayed figure.
- Repeated plotting without execution: display and exports remain attached to the
  preceding completed run despite edited parameters.
- Mutated result, RunData, environment and figure globals: saved snapshot arrays,
  metadata and rendered meshes remain unchanged.
- Saving before execution: bounded error and no output directory creation.

The out-of-order tests reopen actual Product packages, compare complete encoded
result-array bytes, verify the exact request metadata and selected arrays, and
inspect the meshes/title supplied to the actual PNG export renderer. Existing
canonical-workflow parity, callback-off parity, persistence/integrity checks and
Qt/LaunchPlane/CuPy import blocking remain passing. All **9** encoded arrays from
the remediated fresh-kernel run also match the original Stage 1 hashes exactly.
All **12** saved artifact hashes verify. Expected Agg noninteractive-show warnings
are presentation-only. Python syntax, Python 3.10 grammar and whitespace pass.

Commands are the original commands below with evidence paths changed to
`stage-1a/executed.ipynb`, `stage-1a/tests.log` and `stage-1a/regressions.log`.
No new dependencies or existing-user-environment installation were needed.

Current SHA-256 identities:

| Candidate / artifact | SHA-256 |
|---|---|
| Notebook | `7f8c06d8a18fa685c192deaa8a48712433e4a1cd41ea8710b4ad8a33ba17e6b4` |
| Tests | `ed46d15025647c096c03819042892f1f90333e2269731ae7cafe2758b7da6261` |
| Saved selected-intensity NPZ | `ad0c78ef43c38d6251472f5e4cdfadfd07fcbdf0831a27903694b12f647db738` |
| Regenerated PNG | `50a504f2412913210f599dc8f66dd2d263bec248a9ebaa1dd160338c2140a7a4` |
| Product result-array NPZ | `0256f39bc46dd85623e24adabeba3d8ff9a8a72787aca8871c4d93b063cd47bd` |

See `stage-1a/candidate-inventory.json`, `stage-1a/scientific-array-identities.json`,
`stage-1a/saved-example/notebook-manifest.json`, and
`stage-1a/preservation-verification.json`. Scientific reproducibility does not
imply identical timing/environment metadata or archive bytes across environments.
The snapshot fix does not add a cross-platform numerical qualification.

All unrelated tracked modifications, original Stage 1 artifacts and protected
evidence are preserved. Changes remain unstaged/uncommitted; no cluster, CUDA,
commit or push. **Stage 1A provenance remediation validated — ready for review.**

## Original Stage 1 validation (historical)

Date: 2026-10-08. Product baseline and unchanged HEAD:
`985b80fbbc05616d4fe92f974ef0da24bf84e94c`.

Implemented the smallest reduced-PR CPU milestone from the architecture audit.
No existing Product module was modified. The notebook is a thin client of the
existing scientific engine, not a port of PRProp3D scientific code.

## Exact candidate inventory

Three new, unstaged files; exact SHA-256 identities are in `candidate-inventory.json`:

| File | Purpose |
|---|---|
| `notebooks/pr_reduced_cpu.ipynb` | Unexecuted portable notebook, five Python cells and explanatory Markdown |
| `notebooks/README.md` | Explicit local/Colab setup, compute-node boundary, saving/reopening and validation instructions |
| `tests/test_portable_pr_notebook.py` | Nine bounded tests including a real fresh-kernel execution |

Research artifacts in this directory are evidence only, not Product candidate files.
They include the executed notebook, saved example, test logs, array hashes and
preservation inventories. The source notebook contains no saved outputs.

## Architecture and scientific request

1. Standard-library dependency check, Python/version/module-path identification.
2. Existing `PRRunRequest`, physical material factory, beam/grid/backend/solver specs;
   existing headless `validate_pr_gui_request` preflight (despite its module name,
   this path does not import Qt or LaunchPlane).
3. `LocalRunner.run_operation(PR_TIMEDEPENDENT_OPERATION, request)` with a scalar-only
   progress callback. Elapsed time is operational metadata, never material time.
4. Existing Product `RunData` output intensity and optical-intensity volume supply
   Matplotlib xy and xz plots. The xz slice is an existing-array view; no replay,
   FFT or material reconstruction is added.
5. An explicitly invoked save function uses existing experiment and transport codecs,
   plus selected NPZ arrays/coordinates, a PNG and SHA-256/environment manifest.
   New destination folders are required. There is no automatic write to Drive.

Example: NumPy float64; reduced nonlinear `pr_timedependent`; one 1 mW, 0.633 µm,
normal-incidence Gaussian with 12 µm beam-normal radii; explicit coherence group;
64×48 transverse nodes on 80×60 µm; 20 µm propagation in 1 µm steps; index 2.4;
gain-length product 0.1; no scattering; dark-equivalent 0.01 W/cm²; background zero.
Physical normalization remains `pr_integral_total_illumination_mean_irradiance_v1`.
Integrator is the existing `semi_implicit_trapezoidal`; three steps of 0.001 give
accepted characteristic material time 0.003. No seconds conversion is claimed.
Default published frozen-material optical-first coupling and other material
parameters are retained and displayed. Preflight has no aperture/sampling warnings;
its physical-normalization timestep assessment remains `pending`, as implemented
by Product. This demonstration is not a new timestep qualification.

Plots explicitly distinguish numerical optical intensity in 1/µm² from physical
irradiance in W/cm² and the normalized transport source. Greek symbols and units
render with Matplotlib mathtext. The saved PNG was visually inspected.

## Validation and scientific parity

**9 passed, 0 skipped** in the focused notebook suite; **24 passed** in existing
PR execution, headless integration and reduced-TD transport tests.

The fresh Jupyter kernel completed 3/3 accepted steps at τ=0.003. Its Local runner
and product-conversion elapsed time was about 0.136 s on this host (diagnostic only).
Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.0; LCProp reports 0.1.0.
Exact environment provenance is retained in `saved-example/notebook-manifest.json`.

Evidence tested:

- The notebook request survives Product codec encode/decode with exact scientific
  metadata. The saved `.lcprop.json` reconstructs the same request.
- All **9 encoded result/checkpoint array entries** match byte-for-byte against
  the direct headless workflow under the same retention settings. All Product
  field-adapter arrays also match exactly, including plotted xy/xz products.
- A second independent kernel executes the actual notebook cells and saves through
  the real codecs; all 9 array entries match the Local runner output exactly.
- The five core arrays (`A_initial`, `A_final`, `E_initial`, `E_final`,
  `source_intensity_stack`) are byte-identical with progress disabled as well.
- Progress reports accepted steps 1, 2, 3 and nonnegative elapsed times. No callback
  field arrays are retained by notebook progress bookkeeping.
- A kernel import blocker denies PySide6, PyQt6, LaunchPlane and CuPy; none is loaded.
- Plotted mesh arrays equal the correctly transposed Product arrays; xz shares
  memory with its source volume. Actual PNG export and hash verification pass.
- Full request/result packages reopen with existing integrity verification; selected
  NPZ files load without pickle; existing output folders are rejected.
- Four missing-dependency controls (NumPy/SciPy/Matplotlib/LCProp) produce bounded
  actionable errors; nothing installs itself.
- Notebook schema, Python syntax/3.10 grammar and whitespace checks pass.

The first parity assertion intentionally exposed a Product retention distinction:
`run_pr_timedependent` records extra movie products when a progress callback exists.
The finalized test matches that contract before comparing complete inventories,
and separately proves callback-off scientific-array identity. No Product fix or
tolerance adjustment was made. An early fresh-kernel attempt was blocked by the
sandbox's localhost socket restriction; rerunning with local socket permission
passed. The Agg validation backend emits one expected noninteractive `show()` warning;
the saved figure is verified. These are not scientific failures.

Commands (repository root):

```sh
PYTHONDONTWRITEBYTECODE=1 IPYTHONDIR=/tmp/lcprop-nb-ipython MPLCONFIGDIR=/tmp/lcprop-nb-mpl PYTHONPATH=src MPLBACKEND=Agg LCPROP_NOTEBOOK_EXECUTED_PATH=$PWD/results/Research/lcprop-portable-notebook-stage-1-v1/executed.ipynb /tmp/lcprop-notebook-stage1-venv/bin/python -m pytest -q tests/test_portable_pr_notebook.py
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/lcprop-nb-mpl PYTHONPATH=src MPLBACKEND=Agg /tmp/lcprop-notebook-stage1-venv/bin/python -m pytest -q tests/test_pr_execution.py tests/test_pr_headless_integration.py tests/test_pr_timedependent_transport.py
git diff --check
```

See `tests-final.log`, `regressions.log`, `executed.ipynb` and
`scientific-array-identities.json`. Original unsuccessful development checks remain
in `tests.log`; they were corrected as described above. No wider suite was needed:
existing Product source is unchanged, and the affected existing tests all pass.

## Dependencies and portability

No existing user environment was installed into or changed. Validation used a
temporary `/tmp/lcprop-notebook-stage1-venv` with system-site scientific dependencies
from the existing developer environment, and nbclient/nbformat/ipykernel installed
only into that temporary environment. Kernelspec registration was temporary.

The README supplies explicit fresh-environment commands to install LCProp from the
exact public Git commit, Matplotlib and Jupyter. A matching wheel can substitute.
The notebook needs Python >=3.10, NumPy, SciPy, Matplotlib and LCProp; it does not need
desktop GUI extras. Dependency versions and actual LCProp import location are
reported; the baseline string itself is explicitly not a source attestation.

Hosted Colab: upload/open this new notebook, select CPU, explicitly install the
pinned engine in a separate setup cell if needed, restart as appropriate, then run
cells. Neither this notebook nor the pinned baseline assumes the other is already
installed. No write credentials or private research files are required. Download
outputs before session loss; Drive is optional and never mounted automatically.
Local-runtime Colab uses the same scientific notebook after user-managed runtime setup.

Tufts: the kernel must already be **inside a Slurm allocation on a compute node**,
never on a login node. No scheduler, SSH, allocation or retrieval logic is included.

Only local macOS CPU/fresh-kernel execution was exercised. Hosted Colab, local-runtime
Colab, Linux/Windows and an allocated Tufts kernel remain uncommissioned. Pinning
Product alone does not guarantee bitwise reproducibility across dependency versions.
The small full-result save path must not be mistaken for a large-grid retention design.
This milestone has no widgets/Stop button, GPU support, live accepted-state display,
image amplification, backpropagation, fanning studies, LC workflows or Slurm integration.

## Preservation and Stage 2

HEAD and index remain unchanged; index empty. All tracked source hashes are unchanged,
including both unrelated pre-existing modifications. All **40,808** inventoried
pre-existing files retain size/mtime; all **919** prior protected manifest SHA-256
identities were independently reverified. See `preservation-verification.json`.
No source edits, commits, pushes, cluster access, CUDA or existing-user-environment
installation occurred. Candidate files remain untracked/unstaged.

Recommended Stage 2: a small non-Qt request editor with explicit experiment loading,
scalar progress and cancellation, sharing Product validation/codecs. Qualify an actual
Colab CPU runtime separately before advertising runtime support. Do not copy legacy
PRProp3D scientific calculations or broaden this milestone's workflow claims.

**LCPROP PORTABLE NOTEBOOK STAGE 1 VALIDATED — READY FOR LOCAL REVIEW**

# Stage 3I — Hosted restart-free qualification preparation

2026-10-17. **NO-GO for removing the restart requirement: hosted Case A/Case B results are pending.** A separate minimal diagnostic notebook is prepared and locally checked. This environment has no connected hosted Colab CPU session or browser automation for those sessions; local checks are not represented as hosted qualification.

## Deliverable

`notebooks/lc_static_pypi_rc1_restart_free_diagnostic.ipynb`

SHA-256: `904c5c149dc3221b6300a2c4883c9a28dc02bfab03a5419a81a4304e361187c9`.

The accepted RC1 qualification notebook and production 0.3.0 notebook remain byte-identical to HEAD. No package/controller changes, dependency changes, installations, simulations, commits, pushes or uploads occurred locally. New files are the diagnostic notebook, this report and temporary local validation under `.codex-work/beam-stage3i/`.

## Procedure and evidence collection

Use Colab's Open notebook / Upload facility to open this local notebook manually; it is not published under this task. Select a hosted CPU runtime. Use a newly allocated runtime, not a restart of a runtime where RC1 is already installed.

1. Case A: leave CASE = "A"; run the four code cells in order without restarting. Capture baseline first. Do not execute unrelated cells or backend-list diagnostics beforehand.
2. Stop and retain evidence if installation or initialization fails. The observation cell can still be run after a failure. Do not apply cache manipulation or restart and label the result restart-free.
3. If initialization succeeds, visually check first and repeated Beam 2 activation, Move/Tilt, numeric edits and stable rendering. Rerun only initialization to check reuse. Never press Run during this experiment.
4. Fill PASS/FAIL and notes in the observations cell; retain screenshots and the JSON/log files using the Colab Files panel. The evidence directory is printed and uses a runtime temporary directory independent of notebook location. No automatic upload occurs.
5. Disconnect/delete that runtime. In a separate newly allocated CPU runtime choose CASE = "B", which deliberately imports Matplotlib before installation. Repeat. Do not simulate B by restarting A.

A baseline detects LCProp already installed or imported and rejects that trial. This is a useful freshness check, not proof that the runtime has never run other user code. Session provenance still depends on following the instructions. If Colab preloads Matplotlib or ipympl, record that fact rather than removing it: Case A promises no notebook-initiated imports before pip, not an empty Colab kernel.

## What is captured

Before installation: Python/platform/process identity, complete sys.modules name list, exposed loaded module versions and installed distribution versions. Reading the existing Matplotlib rcParams dictionary records the raw backend value without importing Matplotlib or invoking lazy backend selection. Backend entry-point/cache discovery is deliberately not queried, because measurement itself can prime the suspected cache. Case B records both the initial state and the state after its deliberate import; it does not additionally activate a backend.

During installation: exact release endpoint, both advertised artifact hashes, independently verified wheel bytes, production-PyPI-only dependency selection, pip --report JSON and stdout/stderr. No --upgrade or force-reinstall flags are added. Ordinary resolution may install dependencies required by RC1. All distribution versions are compared before/after and stored; pip's report supplements that comparison, including same-version installation actions. A mapped already-loaded distribution version change stops initialization. This does not claim perfect detection of every unusual distribution/module alias or same-version file replacement.

The expected RC1 wheel SHA-256 remains `2cc0ede68970b2d7de08dc385597861ab1706ad7b08fa4fa2169c5539f8e396e`; sdist metadata SHA-256 remains `7f3367ea5a2940b8cd2256ec9c93ab44bfae2f3907798dda1158c96b1fdf2738`. Version 0.4.0rc1, all 269 runtime files, import ownership, distribution verifier and pip check remain mandatory. The sdist is not installed. No unverified source fallback is introduced.

After installation: enable Colab's custom widget manager and widget backend, construct the existing controller and use the accepted direct Configure/Run/Results presentation. The initialization block is copied unchanged into exception/evidence capture. Installation and initialization exceptions are retained, and Python success is explicitly separate from browser acceptance. Process/baseline guards reject a lost session or repeated installation. No private registry changes, reloads or automatic restarts are present.

The diagnostic supports editing only; no cell invokes scientific execution. Completed scientific records and exports are outside this experiment.

## Local validation

`python3 .codex-work/beam-stage3i/check.py`: PASS after correcting the test harness to provide the traceback module normally imported by the baseline cell. Checks cover code compilation, empty outputs, artifact pins, 269-file gate, production index and pip report option, exact accepted initialization-block identity, missing-release failure/evidence capture, repeat-install rejection and process guard. Network and installer calls were mocked; no real installation or hosted initialization occurred. These checks do not substitute for real frontend or resolver evidence.

## Results matrix

| Observation | Case A | Case B | Accepted restart workflow |
| --- | --- | --- | --- |
| Pre-install environment | Pending | Pending | Not retained historically |
| Installed/replaced dependencies | Pending | Pending | Not retained historically |
| Backend activation without restart | Pending | Pending | Failed in original user session |
| Interactive canvas and editing | Pending | Pending | User-reported PASS after restart |
| Complete corrected presentation | Pending | Pending | User-reported PASS |

Previously reported post-restart environment: Python 3.13.6; NumPy 2.1.3; SciPy 1.16.3; Matplotlib 3.10.0; ipywidgets 7.7.1; IPython 7.34.0; ipympl 0.10.0. This is a comparison baseline, not a pin imposed on fresh Colab images.

## Gate

Retain the manual restart requirement. A successful Case A would support a bounded initialization-order proposal only for the recorded environment; Case B helps distinguish early import effects. Both may already inherit Colab backend discovery, so equality of outcomes does not disprove the cache mechanism. Require repeat fresh-runtime success and actual browser observations before modifying the accepted workflow. Failure must be retained with its dependency transaction and traceback rather than repaired through undocumented cache manipulation.

## Hosted Case A — user-reported result

The user supplied baseline and installation/initialization output from a fresh Case A hosted runtime. Evidence directory reported by that runtime: `/tmp/lcprop-stage3i-4z_ndm9y`. Raw JSON, pip transaction report and logs have not yet been downloaded or independently inspected here.

- Python: 3.13.16 (Oct 1 2026 build), GCC 13.3.0.
- Preloaded relevant modules: numpy, matplotlib, ipywidgets, IPython.
- Raw initial backend: `module://matplotlib_inline.backend_inline`.
- Exact RC1 setup succeeded and reported 269 installed runtime files verified; no scientific execution.
- Distribution-version changes reported only additions: ipympl 0.10.0, jedi 0.20.1 and lcprop 0.4.0rc1. No existing distribution version changes were reported. This does not independently exclude same-version reinstalls; the pip report remains to be inspected.
- Initialization failed at `get_ipython().run_line_magic("matplotlib", "widget")`, before controller construction. IPython attempted to assign Matplotlib's backend rcParam, and validation rejected `module://ipympl.backend_nbagg`; the listed supported backends included inline but not widget/ipympl.

**Case A result: FAIL for restart-free initialization.** Browser editing was not reached. This narrows the failure beyond a conservative restart warning: actual backend activation failed without any reported existing-package version replacement. The result is consistent with the Stage 3H cached-discovery reproduction, but private registry state was not measured and the precise internal state is not directly proven.

**Updated gate: NO-GO for removing the restart requirement.** Preserve the failed case files before deleting its runtime. Case B remains optional comparative evidence and must use a separately allocated fresh runtime; importing Matplotlib there may add little contrast because Colab already preloaded it in Case A. Do not repair Case A through undocumented cache changes or reinterpret a post-restart success as restart-free. The accepted restart-based workflow remains the supported qualification path.

## Retained archive review

The supplied `lcprop-stage3i-case-a.zip` was inspected in place without extraction or execution. SHA-256: `f8a4b23cdc0e053876fb087ca5087a9a3a72a440e2750aed31cd67d5aa325c54`. It contains evidence.json, pip-report.json, pip-stdout.txt and pip-stderr.txt. Member hashes and independently recomputed comparisons are in `case-a-archive-review.json`.

The pip report and successful-install log contain exactly three installation actions: lcprop 0.4.0rc1, ipympl 0.10.0 and jedi 0.20.1. All three were absent in the pre-install metadata. Comparing the full pre/post inventory reproduces the recorded changes exactly. There are no existing-distribution replacements or same-version reinstalls in this transaction report. LCProp's wheel digest is the qualified expected digest; dependency download URLs use files.pythonhosted.org. Stderr contains the standard root-user pip warning, not an installation failure.

The retained identity-check result reports all package/runtime checks passed. The reviewer did not rerun the hosted 269-file checks; the original hosted files are not bundled. The recorded backend exception matches the reported failure and occurs before controller construction. Browser observations are empty and scientific_execution is false.

**Baseline inconsistency:** before installation, loaded NumPy reports 2.1.3 while distribution metadata reports 1.26.4. Both remain unchanged afterward. Therefore this inconsistency was not caused by the recorded RC1 pip transaction. Its cause (for example duplicate metadata or pre-existing environment changes) cannot be established from these files. Do not describe this as a fully consistent scientific runtime or use it to qualify numerical results. Future scientific qualification must reconcile loaded and installed identities first.

All four snapshots in the archive have PID 1036. The earlier pasted traceback used PID 8864; consequently the archive is not established as the exact same execution as that paste. It independently documents the same failure pattern with continuous process identity across its own snapshots. This distinction and the baseline inconsistency are preserved rather than silently merged away.

**Final gate remains NO-GO for removing the restart requirement.** This archived transaction rules out dependency replacement during that installation as the immediate explanation. It supports, but does not directly inspect, the cached-backend-discovery mechanism. Preserve the accepted restart procedure; do not weaken integrity checks or modify caches. No package/controller/notebook changes, installations, simulations, commits or uploads were performed during archive review.

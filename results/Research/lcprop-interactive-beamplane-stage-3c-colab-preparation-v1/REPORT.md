# Stage 3C — RC1 hosted-Colab qualification preparation

**Notebook prepared; hosted installation BLOCKED pending TestPyPI publication of the exact artifacts.** Anonymous request to `https://test.pypi.org/pypi/lcprop/0.4.0rc1/json` returned HTTP 404 during preparation (2026-10-16). This does not establish project-name availability. No alternate release or source installation was substituted. No installation, scientific execution, upload, commit or push occurred.

## Deliverable

`notebooks/lc_static_pypi_rc1_colab_qualification.ipynb` is a separate, output-free qualification notebook. The production `notebooks/lc_static_pypi_cpu.ipynb` and its 0.3.0 pin are unchanged.

Expected immutable release source: **2e195579dc3d76f6477bf6c885fd537043acdf9d**.

| Artifact | Expected SHA-256 |
| --- | --- |
| lcprop-0.4.0rc1-py3-none-any.whl | 2cc0ede68970b2d7de08dc385597861ab1706ad7b08fa4fa2169c5539f8e396e |
| lcprop-0.4.0rc1.tar.gz | 7f3367ea5a2940b8cd2256ec9c93ab44bfae2f3907798dda1158c96b1fdf2738 |

These identities come from the committed Stage 3A artifact record. Source-to-artifact correspondence was verified in Stage 3B; the wheel itself need not carry a Git checkout/direct_url commit. The notebook records the expected source identity rather than claiming that pip independently recovers a Git revision.

## Setup design

1. Check the exact TestPyPI JSON endpoint. Require precisely the expected wheel/sdist inventory and both published hashes. Stop on missing release, changed inventory or mismatch.
2. Download only the wheel, independently hash its bytes before any installation. The sdist's published hash is checked, but its bytes are not downloaded or installed in this notebook.
3. Install that verified local wheel with `[colab]`; explicitly select **production PyPI** for dependencies. Remove inherited PIP_* variables and disable pip config via PIP_CONFIG_FILE=/dev/null. No extra index, source fallback or unpinned LCProp selection. Routine pip output is captured; failures show diagnostics.
4. Check installed version, every one of the **269 runtime files** against the verified wheel, active import location, installed-distribution verifier and pip check. No embedded 269-entry manifest or external helper script is needed.
5. Repeated setup verifies identities again but skips installation when the previously verified controller package is already loaded. Unverified/different loaded LCProp triggers an export/restart instruction. If installation changes an already imported dependency version, stop and request a manual restart; no automatic restart or live reload.

All checks gate `_rc_ready`. The setup cell is collapsed using Colab form metadata. This is a qualification notebook, so its integrity checks intentionally exceed the ordinary production notebook's setup.

## Initialization and fallback

Initialization uses `from lcprop.notebook import InteractiveLC`. Interactive mode enables `google.colab.output.enable_custom_widget_manager()` and selects the ipympl widget backend before constructing the controller. Subsequent initialization reuses `app_rc1`, preserves controls/completed snapshots and clears/replaces only that initialization cell's output. Backend or incompatible/disposed controller replacement requires export/restart.

Explicit fallback: export needed records, restart, set `MODE = "static"`, and rerun setup/initialization. The inline backend uses the existing complete static preview and numeric controls. A silent frontend failure cannot be inferred from Python canvas creation, so automatic browser failure detection is not claimed. Mouse Move/Tilt is unavailable in static mode. Static success alone does not qualify the interactive frontend.

A separate evidence cell prints release/artifact identity, installed module, Python/platform and numerical/widget dependency versions. The checklist covers first Beam 2 activation, repeated toggles, overlap selection, Move/Tilt, numeric commits, coherent/incoherent controls, cancellation, stable canvas/colorbar, no duplicate views, and repeated initialization. The complete executed request remains available through the existing controller UI after an explicitly authorized run.

Only the existing Run button may execute propagation. The notebook does not auto-run it. Completed-run isolation and cross-platform numerical comparisons must remain pending until a separately authorized bounded run supplies a completed record. Editing/graphics checks can proceed without science. No soliton or broad multibeam qualification is implied.

## Preparation validation

No setup cells were executed against the network or an installer. Standard-library mock checks passed for:

- code compilation and empty outputs;
- missing TestPyPI release stops before installation;
- mismatched metadata and downloaded wheel bytes stop;
- matching local retained wheel reaches only the production-index installation command, then the mock stops it;
- pip configuration isolation;
- repeated initialization reuses the exact controller and snapshot object.

Actual anonymous metadata retrieval was performed separately and returned 404. Full install/runtime checks are executable gates in the notebook, not newly passed hosted results. No changes to qualified wheel/sdist bytes. Temporary preparation checks: `.codex-work/beam-stage3c/check_notebook.py` and `checks.txt`. Notebook checksum and preservation checks are in `preparation-identities.json`.

## Next authorized gate

First publish the exact candidate artifacts to TestPyPI under separate explicit authorization and verify their public hashes. Then use this notebook in a fresh hosted CPU runtime. Do not work around the current 404 with production 0.3.0, Git installation, or another artifact. Hosted widget acceptance and any bounded scientific Run require their own recorded results; preparation alone is not qualification.

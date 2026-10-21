# Stage 5C — Disposable Colab deployment preparation

**GO for a separate publication review; not published or ready to open through a public link.** 2026-10-21.

## Candidate

Development version: `0.4.1.dev0+stage5c` (local development build, not a production release).
Wheel: `lcprop-0.4.1.dev0+stage5c-py3-none-any.whl` in this directory.
Bytes: 761039.
SHA-256: `51520c31974ecfd6c0a911a4000ba4c2c6786456e14cd4168a737afac81e66a1`.

Source base: production commit `dfc426a26d288a540228d3ab24c990cf33105b0e`. Reconstructed the exact 282-input production manifest from Git, then applied only the accepted `interactive_lc.py` and `_beam_position_editor.py` Stage 5C edits. An isolated pyproject version overlay distinguishes the disposable wheel; the working-tree package version was not changed. Unrelated PR source edits were never copied into the build.

Frozen, read-only source snapshot: `.codex-work/beam-stage5c-deploy/source-snapshot.tar`; its SHA-256 is in provenance.json. Built with the existing build environment using `build --wheel --no-isolation`. No dependency upgrades. The unpacked build directory is disposable; the frozen tar and source manifest define its inputs.

All 269 runtime files match the development manifest. Exactly two differ from production: `lcprop/notebook/interactive_lc.py` and `lcprop/notebook/_beam_position_editor.py`. **All other 267 runtime files, including scientific engine and request-schema files, are byte-identical to production 0.4.0.** Archive inventory excludes diagnostics, research evidence and unrelated work. No production artifacts were modified.

## Verification

- Strict `twine check --strict`: PASS for the wheel.
- Installed this wheel with `--no-deps --no-index --target` into disposable audit storage, using existing dependencies. Existing installed production packages were not replaced.
- Independently verified installed origin, development version and all 269 runtime hashes.
- Copied the bounded test harness outside the repository source and ran against the target-installed package: **63 passed in 34.11s**. Retained tests cover default/configured mouse radii, both beams and Move/Tilt, overlap/inactive-beam isolation, cancellation, canonical request parity, frozen requests and synthetic completed snapshots/exports. No propagation.
- Notebook setup verification and synthetic-first/one-attempt gates tested locally with Colab eval_js and magic mocked. Public tolerance is 30, Run is disabled and its callback removed on the disposable instance; programmatic button click did not execute science. A repeated BeamPlane attempt is rejected.
- Existing headless import isolation checked; no new dependency requirements. Local dependency consistency checked with pip check. Hosted resolver/browser behavior remains untested.

Build environment, installed versions, source/runtime/archive inventories, test identities and compact verification are retained alongside the wheel. Local install/build logs remain in `.codex-work/beam-stage5c-deploy/`. No new numerical simulation or parity run was needed because engine/schema bytes are unchanged.

## Minimal acceptance notebook

`notebooks/lcprop_ipad_stage5c_acceptance.ipynb`.
Prepared SHA-256: `9cfd325df934f17b79a01372108ce5353873547658223572da67a48bd358236c`.

Three code cells: exact-wheel setup/verification, synthetic input check, then one disposable BeamPlane attempt. Code is collapsed by default. No mode editing, diagnostic uploads, log downloads or event-log transfer is required. The adapter is embedded, keeps 25 Hz motion limiting and immediate terminal events, and uses only the existing scoped canvas path. Missing widget framing stops safely.

Setup will anonymously download the exact wheel from a reviewed immutable GitHub commit, enforce its SHA-256, and install its colab extra from production PyPI. A constraints file pins every existing dependency version; compatible installed packages are preserved and resolution must stop if an upgrade/downgrade would be needed. Missing dependencies may be added. Existing LCProp may be replaced only before import. No such hosted installation was performed during preparation.

After first installation the user restarts the session and reruns Setup. Verification checks all 269 files and loaded/disk dependency-version consistency. Synthetic testing requires two complete drag sequences plus explicit user acceptance of finger/Pencil, cancellation and scrolling observations. The checkbox is a user attestation, not device-type inference from Python events. The user removes the synthetic adapter before moving on.

BeamPlane uses a new `InteractiveLC(hit_tolerance_pixels=30)`, existing Active Beam/Move-Tilt controls, no live launch previews during drag, and a disabled/disconnected Run callback. Only Configure is shown. Existing completed scientific controllers are not attached or touched. Private `_position_editor` and Configure-child access remain the previously disclosed diagnostic presentation coupling; hit tolerance uses the new public API.

One bounded hosted attempt is enforced in-session for synthetic/BeamPlane setup. This is an operational bound, not a security barrier against editing cells or restarting. If any interaction remains unusable, stop and report the simple outcome; do not loop through reinstalls or backend workarounds.

## Publication boundary

`PUBLICATION_COMMIT` intentionally contains a pending-review sentinel. On a runtime without the verified development package, Setup stops before any download/install until it is replaced with a reviewed public 40-character commit. No placeholder URL is presented as working.

A separate authorized publication review should first audit and publish the clean development inputs, exact wheel and required compact evidence to a dedicated branch; then bind the artifact commit in the notebook and publish that notebook. Two commits avoid a self-referential immutable URL. Recheck the notebook's hash/gates after binding, audit all outgoing history, anonymously verify the wheel hash and notebook bytes, and provide the final immutable Colab URL. Neither step is authorized or performed here. No manual wheel upload or source checkout fallback is included.

## Simple iPad acceptance after publication

1. Open the future immutable notebook on the iPad. Run Setup, restart when instructed, rerun Setup.
2. Run Synthetic check; make one finger drag and one Pencil drag, releasing inside. Check cancellation and outside scrolling. If either fails, stop. Otherwise check acceptance and tap Finish synthetic.
3. Run BeamPlane once. Try **finger Move**, **Pencil Move**, **Pencil Tilt**. Report each as usable/unusable and whether response is acceptable. Stop if still unusable. No scientific Run or log transfer.

## Preservation

No commits, pushes, publication, production PyPI upload, scientific simulations, dependency upgrades or unrelated modifications. Existing production wheel/sdist, unrelated PR work, saved notebook outputs and protected evidence remain intact. The development wheel is the only new package artifact; no sdist was requested or built.

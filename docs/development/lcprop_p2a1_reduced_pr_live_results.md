# P2A-1 — Reduced PR TD Local live accepted-state Results

## Preflight and scope

Baseline: `d9010ae04ef9228023bde22a08ab5447ef8b88fb`, authoritative branch
`feature/pr-second-order-static`. Development uses isolated worktree
`/private/tmp/lcprop-p2a1-reduced-pr-live`, branch `feature/p2a1-reduced-pr-live`.
Equations/model changes: **N/A**. This implements the reduced-PR portion of the
preceding source-only P2A audit, following the
[post-Stage-1 residuals](lcprop_human_acceptance_post_stage1_residuals_2026-09.md),
[P1 polish](lcprop_human_acceptance_residual_polish_p1.md), and
[runtime presentation](lcprop_native_acceptance_runtime_presentation_followup.md).
The [architecture contract](../architecture/LCProp_Target_Architecture.md) keeps
PR state interpretation in PR and shared views material-neutral.

Pre-edit source verification confirmed: reduced TD already performs candidate
optical replay before cancellation checking and accepted-state assignment;
legacy progress copies full initial/current material and optical-source volumes;
PR GUI previously displayed only progress text; shared Workspace already owns
request/result attribution, subtab preservation, and partial-render failure.
Full-transverse optical/material timing differs and is excluded.

The actual authoritative historical inventory at preflight is **4,336 paths**,
fingerprint `d6eb804afc3a6a1ce0fe2ede3b35ea42102f1d0a829b2e5abefee650bfbdce64`.
An external snapshot records paths, modes, sizes, modification times, and content
hashes. No older inventory is restored. LaunchPlane remains
`add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`; Research remains
`b947fac51722eafa971e756a2cc2053be065294d`.

## Accepted-state architecture

The path is accepted reduced PR state → PR-owned bounded snapshot → PR live
adapter → shared RunData → existing Workspace. The implementation is in
[PR live products](../../src/lcprop/pr/live_results.py),
[reduced workflow](../../src/lcprop/pr/workflow.py), and
[PR GUI](../../src/lcprop/pr/gui/main_window.py).

`live_preview_policy` is an optional keyword-only execution/presentation option,
not a scientific request field. Default `None` preserves legacy callback
payloads. Local reduced Run and Local Continue opt into bounded snapshots.
Full-transverse execution and Slurm do not opt in. No runner protocol, transport
codec, request schema, checkpoint format, or final-result type changes.

All integrator calls, source passes, candidate replay, cancellation checks, and
accepted-state assignment retain their original order. Extraction happens only
after that assignment. No predictor or incomplete replay is exposed. Existing
replay remains per accepted step when a callback is present; the implementation
does not attempt to optimize it away or alter headless cancellation behavior.

The presentation policy is a 0.5-second minimum interval measured after the
previous synchronous delivery returns. First and final newly accepted steps
bypass that interval. On cancellation after a throttled step, the last accepted
state is published from the existing matching cached optical observation,
without an extra replay. Pre-cancelled and zero-step runs invent no accepted
update. Scalar progress continues every step, while a displayed scientific
frame keeps its own step/time between updates. Rendering still adds wall time;
this policy is not a scientific cadence or real-time throughput guarantee.

## Live products and bounded memory

Each snapshot contains:

- sampled output-plane optical intensity, preserving existing coherence groups;
- one sampled accepted E plane at the middle material z index, with its physical
  z coordinate explicitly named;
- fixed x-z/y-z optical cuts at the original grid samples nearest zero;
- sampled physical x/y/z coordinate vectors;
- cumulative/segment steps, requested total, normalized material time;
- one existing scalar row: material-change RMS and, where applicable, minimum
  normalized carrier density; no new scientific metric;
- model identity, existing model-validation classification, and visualization,
  sampling, normalization, and accepted-replay provenance.

Point sampling is explicit; it is neither block averaging nor a new solver
resolution. Longitudinal intensities use the established PR slice-average source
conversion `(source - background) * peak_reference`. Output intensity is the
coherence-aware optical output observation; it is not the PR-driving source.
Material E is dimensionless, coordinates are micrometres, material time is
normalized, and optical intensity uses the existing `1/µm²` convention.

For policy maxima X, Y, Z, the retained float64 arrays obey:

`8 * (2*X*Y + Z*X + Z*Y + X + Y + Z)` bytes.

Defaults X=128, Y=128, Z=256 give **790,528 bytes**. Four image buffers and three
coordinate vectors are detached and read-only. Smaller scientific grids reduce
the payload. Scalar metadata has a fixed number of entries; no material-time
history, checkpoint, immutable input volume, or scientific 3-D volume is included.
Shared products reference these buffers rather than duplicating them.

Backend advanced indexing selects bounded planes/cuts before host conversion.
The optical intensity calculation uses only sampled complex channel fields.
Extraction temporaries scale with the bounded images and channel count, plus
one-dimensional original x/y coordinate vectors. They do not scale as a new
full material volume. Existing solver/replay and final-result allocations remain
unchanged and can still be large. Previous/current frames and Matplotlib may
coexist briefly; the array bound describes one snapshot, not total process RSS.
Blocking progress delivery preserves existing queue backpressure.

The existing backend cut helper transfers complete full-resolution cuts. This
milestone uses the same normalization/nearest-zero convention but indexes both
z and transverse axes before transfer so the configured live byte bound holds.
No GPU execution or measured performance claim is made.

## Shared Results behavior

The existing ImagePane handles transverse products. Explicit paired-cut metadata
in FieldData coordinates lets LongitudinalPane consume generic compact x-z/y-z
pairs, retaining legacy Fast-cut behavior. Pair axes, quantity, units, source
identity, and z geometry are checked before use. Paired cuts are excluded from
the transverse selector because they do not imply a retained 3-D source volume.
Fixed cuts do not promise arbitrary transverse cut repositioning.

Repeated updates retain the selected field and valid longitudinal z/x/y indices
when the same product has unchanged coordinates. This is a small shared-view
correction, not a new PR viewer. Workspace remains responsible for ownership,
old-to-new pane replacement, selected-subtab preservation, and partial-render
failure. New requests retain Previous-result ownership until a valid snapshot;
Stop/failure ownership uses existing lifecycle handling.

Transverse and longitudinal scale stores remain separate. Both longitudinal
views use one scale through Auto/manual/locked transitions; fresh-run lock reset
is unchanged. Plot sizing and vertical scrolling are deliberately not included.

## Validation and self-review

Validation uses the established Python 3.12 environment with isolated `src` on
PYTHONPATH, offscreen Qt, bytecode/pytest cache disabled, and an external
Matplotlib cache. These are software regression checks, not scientific
commissioning or new authoritative numerical evidence.

Focused batch: **144 passed in 22.98s**. It includes the new live-results module,
PR execution/checkpoint and GUI lifecycle/main-window modules, generic and Fast
cuts, runtime presentation, Stage 1A ownership, and Stage 1B scaling.
Affected batch: **359 passed in 39.42s**, including Local/remote GUI dispatch,
PR products and transport, LC workflow GUI, shared views, and layout.
Complete `tests/`: **1,795 passed, 77 skipped in 498.27s**; zero failures and no
accepted baseline exceptions. Syntax, documentation links/anchors, whitespace
including untracked candidate files, and `git diff --check` passed. Protected
repository identities and the full 4,336-path historical inventory match preflight.

Development self-review: **Passed**. The candidate contains exactly eight files,
with an empty index. Public positional interfaces are unchanged; the new optional
keyword is backward compatible. Existing checkpoint, final-result, and transport
contracts and tests remain intact. Preview units and model provenance are
explicit. Twenty-one new regression cases cover the requested behavior; existing
regressions cover shared lifecycle and persistence. Native subjective acceptance
and real GPU execution remain unqualified by these offscreen/CPU tests. The
backend extraction tests verify transfer shapes without invoking CUDA. No new
architecture or scientific blocker was discovered. Authoritative numerical
commissioning manifest/schema gates are N/A for this software-only validation.

The focused regressions cover all three reduced integrators, exact equality of
final scientific arrays, matching optical source/replay call order and states,
preview-off headless equivalence, discarded candidates/predictors, incomplete
and failed replays, first/final/throttled/Stop identity, cumulative Continue,
pre-cancelled/zero-step runs, fixed byte bounds, no full-volume host conversion
for live extraction, model provenance, Local-only dispatch, unchanged scalar
remote telemetry, ownership/render failures, selected tabs/fields/cuts, and
independent transverse/linked longitudinal scale transitions.

Initial development checks found test-fixture setup mistakes and generic-cut
refresh/extent wiring defects. These were corrected before the passing focused
batch; no baseline failure exception is accepted. The current source uses the
same established scientific/cancellation sequence. No existing test assertion
was weakened.

Two preliminary mixed GUI batches aborted in existing worker-thread GUI tests
with Qt segmentation faults. Merely closing newly created test windows was
insufficient cleanup. The new tests now explicitly dispose their Qt widgets and
collect their Python/Matplotlib cycles on the GUI thread before later worker
threads run. The affected batch then passed (359 tests), and passed again after
the final fixed-cut z-readout correction. The aborted batches remain recorded;
they are not counted as passing validation or classified as baseline exceptions.

## Reproducible local validation commands

Use the established Python 3.12 interpreter (`python` below) with
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=src`, `QT_QPA_PLATFORM=offscreen`, and
`MPLCONFIGDIR=/private/tmp/lcprop-p2a1-mpl`, from the isolated worktree.

```sh
python -m pytest -p no:cacheprovider \
  tests/test_pr_live_results.py tests/test_pr_execution.py \
  tests/test_pr_checkpoint.py tests/test_native_runtime_presentation.py \
  tests/test_pr_fast_longitudinal_intensity_cuts.py tests/test_pr_gui_lifecycle.py \
  tests/test_pr_gui_main_window.py tests/test_human_acceptance_transparency.py \
  tests/test_stage1b_transparency.py -q

python -m pytest -p no:cacheprovider \
  tests/test_pr_live_results.py tests/test_pr_gui_*.py \
  tests/test_pr_results_progress_mpr_and_td_visualization.py \
  tests/test_pr_products.py tests/test_pr_static_products.py \
  tests/test_pr_checkpoint.py tests/test_pr_execution.py \
  tests/test_pr_fast_longitudinal_intensity_cuts.py \
  tests/test_pr_timedependent_transport.py \
  tests/test_pr_transverse_timedependent_transport.py \
  tests/test_human_acceptance_transparency.py \
  tests/test_native_runtime_presentation.py tests/test_stage1b_transparency.py \
  tests/test_results_fields_layout.py tests/test_workspace.py \
  tests/test_gui_static_execution.py tests/test_gui_timedependent.py \
  tests/test_remote_gui_execution.py tests/test_stage1c_new_user.py -q

python -m pytest -p no:cacheprovider tests/ -q
python -m pytest -p no:cacheprovider tests/ --collect-only -q
git diff --check
```

Collection sanity identified 1,871 tests while diagnosing a long pause in the
existing LC off-axis coupling regression; that regression was left unchanged.
The external final-check script compiles changed Python text in memory (no
bytecode output), checks relative Markdown links/anchors, scans trailing
whitespace including new files, verifies the eight-file boundary and empty
index, computes the sorted path/size/SHA-256 manifest, and compares the complete
historical inventory with its preflight snapshot.

## Exclusions and next gate

No equations, integrators, scientific defaults, scientific arrays, source cadence,
checkpoint/continuation compatibility, persistence, result retention, transport,
source deployment, or remote artifact lifecycle changes. Existing final TD
movie/history behavior is preserved; no new live history is retained or encoded.

Full-transverse live material/optical observations are deferred to P2A-2. No new
far fields, N-beam/image-screen redesign, Image Amplification migration, channel
gain/RMS/centroid/power metrics, integrator error estimates, or LaunchPlane work.
PR scattering remains **SCIENTIFIC HOLD / UNESTABLISHED**. Remote continuation
remains **DEFERRED / NOT AUTHORIZED**. No Slurm/SSH/CUDA, push, prompt archival,
or fanning activity occurs.

The stopping point is an unstaged candidate with an external exact manifest and
validation evidence, followed by a separately authorized strictly read-only
pre-commit review. No commit or integration is authorized by this development
record.


## Native acceptance follow-up: worker-path coverage and launch provenance

The follow-up uses the same isolated branch at committed P2A-1 HEAD
`40914d6d0a840199008f63e850ccd6ebfd47529b`; its parent and the unchanged
Authoritative Product are `d9010ae04ef9228023bde22a08ab5447ef8b88fb`.
The reported native symptom was material time 0.004, step 4/10, with
“No displayed result” and empty field selectors. This follow-up changes only
this record and the existing live-results test module. No production patch is
justified by the reproduced evidence below.

### End-to-end findings

With this isolated checkout's `src` selected on PYTHONPATH, a real Local Run
through the native window's `run_clicked`, LocalRunner, WorkflowWorker,
BlockingQueuedConnection, and unchanged `_on_progress` reaches
`Workspace.set_run_data(..., state="Current accepted state")` on the first
accepted step. RunData contains `live_output_intensity`, `live_material_plane`,
`live_optical_xz`, and `live_optical_yz`. Both transverse products and the generic
paired cuts render. The same path works for Local Continue with cumulative
step/time. No snapshot is lost at any of these boundaries in the tested source.

A controlled launch without PYTHONPATH using the established interpreter instead
resolves `lcprop.pr.gui.main_window` to the unchanged authoritative checkout;
`lcprop.pr.live_results` is absent there. That older source emits legacy dictionary
progress and updates the time/status labels without a live adapter/Workspace
call. It reproduces the reported symptom exactly at step 4/10: time 0.004,
“No displayed result”, zero transverse fields, and zero longitudinal selections.
This is a verified launch-provenance failure mode, not evidence that the bounded
snapshot was dropped by the candidate's worker. The original native session's
launch command/interpreter has been requested; attribution of that particular
session remains unconfirmed until its provenance is supplied.

The first bounded snapshot bypasses throttling and survives the real worker
signal. The GUI recognizes PRLiveSnapshot, the adapter produces four fields,
and shared views accept their metadata. Current request and displayed-result
attempt IDs agree. Later scalar-only updates intentionally do not relabel the
last displayed scientific frame. Existing accepted-state ordering, Stop flush,
0.5-second cadence, producer memory bound, scientific arrays, and all exclusions
remain unchanged.

### Regression and probe limitations

The new regression starts real window Run and Continue operations, uses the real
registered workflow/worker/progress slot, and observes Workspace delivery and
its time indicator after rendering. It checks the first intermediate accepted
steps (1/10 and 11/20), ownership, request identity, four fields, selectable
nonempty transverse and paired-cut images, GUI-thread rendering, and cumulative
time. It also observes the final accepted step for each segment. Assertions run
outside Qt callback exception handling so a missing delivery cannot silently
pass. Existing movie encoding is stubbed by the established test fixture;
scientific integration and GUI progress dispatch are not stubbed.

Two temporary tracing probes replaced/overrode the progress slot and caused a
BlockingQueuedConnection affinity deadlock; both were interrupted and aborted
with an active-QThread warning. These were probe failures, not reproductions of
the reported empty-Results symptom. The successful probes and regression leave
the production Qt slot untouched and observe Workspace methods on the GUI
thread. Test-owned widgets are cooperatively shut down and explicitly disposed.
No Qt cleanup or production ownership behavior was changed.

### Launch the isolated candidate for native acceptance

Select the source explicitly; merely changing the working directory does not
override an interpreter's installed/editable package resolution. From the
isolated worktree, using the established Python 3.12 interpreter as `python`:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python -c \
  'import lcprop.pr.gui.main_window as m; import lcprop.pr.live_results as l; print(m.__file__); print(l.__file__)'
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src" python -m lcprop.pr.gui.app
```

Both printed module paths must belong to the isolated checkout before native
acceptance. These launch instructions do not integrate, reinstall, or alter
Authoritative Product. Native visual acceptance remains pending; offscreen
regressions do not establish the provenance or outcome of the original session.


### Follow-up validation and candidate boundary

Using the same Python 3.12 environment and focused/affected/full commands above:

- live-results module: **22 passed in 3.90s**;
- focused batch: **145 passed in 24.59s**;
- affected batch: **360 passed in 41.46s**;
- complete Product suite: **1,796 passed, 77 skipped in 505.35s**.

No failure or Qt abort occurred in these validation runs. Syntax compilation in
memory, Markdown links/anchors, whitespace, and `git diff --check` pass. No
scientific or production code changes were made. Development self-review of the
two-file follow-up passes, with original native-session attribution and native
visual acceptance explicitly unresolved rather than claimed complete.

External evidence uses the `lcprop-p2a1-native-` prefix in `/private/tmp`, with
`focused-probe.log`, `focused.log`, `affected.log`, `full.log`, and
`final-checks.log`. The two-file unstaged candidate is recorded in
`lcprop-p2a1-native-candidate-manifest.json` relative to commit `40914d6...`.
A separate `lcprop-p2a1-native-combined-manifest.json` describes the combined eight
P2A-1 files relative to the authoritative baseline; it is not an eight-file
uncommitted candidate. The original reviewed manifest remains preserved.

Authoritative Product, LaunchPlane, and Research retain their preflight commits
and clean tracked worktrees/indexes. The historical protection check does **not**
match the earlier timestamp-sensitive fingerprint. All 4,336 paths, content
hashes, sizes, and modes are unchanged, but six `src/lcprop.egg-info` files have
new modification times around 2026-09-19 16:13:14 UTC: PKG-INFO, SOURCES.txt,
dependency_links.txt, entry_points.txt, requires.txt, and top_level.txt.
No install, packaging, or timestamp-reset operation was performed by this
follow-up; the origin of the timestamp changes is unverified. The current
fingerprint is
`cbed042a4576fd09c778ed67813799a6dc7cfac0a97eeb5bb00cf1adb63683e3`,
compared with the earlier
`d6eb804afc3a6a1ce0fe2ede3b35ea42102f1d0a829b2e5abefee650bfbdce64`.
This is an explicit protection exception for review, not an exact-preservation
pass; no attempt was made to restore timestamps or overwrite historical files.
Nothing is staged, committed, integrated, pushed, or archived by this follow-up.
The next gate is strictly read-only review of the trace evidence, regression,
qualified provenance conclusion, and this two-file candidate. No implementation
of P2A-2 or other excluded work is authorized.

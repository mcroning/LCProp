# LCProp Human Acceptance Punch List — September 2026

## Status, provenance, and scope

This document records the corrected, consolidated source audit following a
manual fresh-user acceptance session on macOS in a separate Python 3.12
environment with a normal non-editable Product installation. The session
covered LC and PR GUI workflows, documentation, persistence, results,
Local/Slurm configuration, soliton/existence behavior, scattering, and Help.

Audited Product identities:

- **LCProp:** `0ba38d4e175f51f4eaa40ecc4d6712f8f7eea854`, branch
  `feature/pr-second-order-static`.
- **LaunchPlane:** `add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`.

The authoritative basis is the corrected read-only Human Acceptance
Punch-List Audit, not the original conversational observation numbering.
HA-01 through HA-22 are stable, consolidated work-item identifiers. The
document depends only on public Product contracts and source, not private
Research material. Relative source links refer to the audited LCProp baseline;
LaunchPlane links are pinned to its audited commit.

The audit inspected source without running applications, scientific
calculations, tests, or remote jobs. Visual observations remain human-reported
unless source establishes their cause. Prior release qualification and manual
successes are recorded as prior evidence, not as tests rerun by this audit.
The proposed acceptance checks below are future checks, not completed work.

**Scientific non-change gate:** this is documentation only. It does not
authorize or implement changes to source, tests, packaging, equations,
defaults, GUI behavior, requests, schemas, codecs, or active operational
prompts. In particular, no convergence tolerance, mixing parameter, scattering
default, solver, or physical definition should change merely to improve a
display or make a status flag look more favorable.

## Priorities and classifications

- **P1:** immediate human-acceptance remediation.
- **P2:** subsequent bounded Product improvement.
- **P3:** separate architecture or scientific-design work, including Research
  capabilities where identified.

Items distinguish confirmed Product defects, documentation/usability defects,
scientific-presentation defects, architectural limitations, missing Product
capabilities, Research capabilities, observations requiring investigation,
and already-correct behavior. Mixed items separate their immediate and later
portions explicitly.

## Consolidated punch list

### HA-01 — Distinguish general two-beam coupling from the specialized image experiment

**Priority:** P1. **Classification:** confirmed usability defect; architectural
investigation, narrower than universal validation leakage.
**Owner:** LCProp PR.

**Verified finding and correction:** ordinary **Gaussian beams** mode calls
`build_pr_request()` directly. **Image amplification** mode instead constructs
`PRImageAmplificationExperimentRequest`, whose validation requires the image
screen, pump/signal roles, carriers in the x-z plane, and symmetric x carriers.
Save Experiment and Run Planning both construct the selected request, so the
specialized mode explains these errors in both operations. Source does not
establish that ordinary Gaussian-mode requests universally enter image
validation. Reproducing those errors in ordinary mode would establish an
additional defect.

The immediate problem is the route offered to a user seeking general TBC.
Make the existing distinction explicit, for example **General beams /
two-beam coupling** and **Image amplification — specialized setup**. Merely
renaming the restrictive mode “Two-beam coupling / Image amplification” would
preserve the problem. Keep the specialized image checks intact; do not delete
them one by one.

**Acceptance:** supported screen-free and asymmetric general requests pass
planning and persistence, while invalid specialized image requests remain
rejected. Assess crossing orientation and screen arrangements against actual
solver/model capability; general TBC does not imply rotational equivalence
for reduced x-only transport. Preserve valid image-request Save/Open.

**Evidence:** [request dispatch](../../src/lcprop/pr/gui/main_window.py),
[image request construction](../../src/lcprop/pr/gui/image_input_panel.py),
[specialized validation](../../src/lcprop/pr/image_amplification.py).

### HA-02 — Identify which request owns every displayed result

**Priority:** P1. **Classification:** confirmed Product and
scientific-presentation defect. **Owner:** LCProp shared GUI with LC/PR
lifecycle integration.

Failure handlers record failure without marking displayed products as
belonging to a previous request. PR progress also leaves existing fields
displayed while updating progress text. Thus “Final Authoritative…” beside
step 1 can be an old completed result beside new-run progress, rather than a
new intermediate field mislabeled by its product converter.

Retain useful old results but visibly label them **Previous run**, including
after validation failure. Distinguish **current accepted state**, **completed
result**, and **state at stop**. Use concise scientific field names and retain
authoritative/replay/provenance details in metadata.

**Acceptance:** after a successful run followed by an invalid or failed
request, the previous fields cannot be mistaken for the new request's result.

**Evidence:** [PR lifecycle and progress](../../src/lcprop/pr/gui/main_window.py),
[LC failure handling](../../src/lcprop/lc/gui/main_window.py).

### HA-03 — Preserve the selected Results subtab during refresh

**Priority:** P1. **Classification:** confirmed Product defect.
**Owner:** LCProp shared GUI.

`Workspace.set_run_data()` explicitly selects Fields when fields exist,
otherwise Curves. LC live updates repeatedly call this method, directly
explaining why Console loses selection.

Preserve the selected Results subtab during progress and result refreshes.
Any automatic initial selection should occur only when appropriate for a new
result, not on every update.

**Acceptance:** Console, Diagnostics, Request, and Curves remain selected
throughout refreshes when the user chooses them.

**Evidence:** [workspace refresh](../../src/lcprop/gui/workspace.py).

### HA-04 — Make start, failure, and console transitions legible

**Priority:** P1. **Classification:** documentation/usability defect;
immediate-status observation partly qualified by source.
**Owner:** LCProp LC/PR GUI and shared console.

Both applications set running state before starting the worker thread; PR
explicitly sets **Running…**. Request construction, description, cost
assessment, and some preflight occur synchronously beforehand. Source
therefore supports a possible preflight delay, not the blanket claim that
status always waits for the first scientific step.

Provide lifecycle-appropriate **Validating/Preparing → Starting → Running**
transitions. Expected configuration failures should show an actionable cause
in the primary GUI; preserve full tracebacks in Console/Diagnostics.

Console appends plain text without a common operation boundary. Add timestamped
headers for Run, Continue, Save/Open Experiment, Save/Load Checkpoint, and
remote submission. A **Clear Console** action should clear presentation only,
preserving scientific results and diagnostics.

**Acceptance:** preparation is visibly acknowledged, expected errors explain
the remedy, and successive operations can be distinguished in Console.

**Evidence:** [PR start](../../src/lcprop/pr/gui/main_window.py),
[LC start](../../src/lcprop/lc/gui/main_window.py),
[console append](../../src/lcprop/gui/workspace.py).

### HA-05 — Provide pre-run request inspection and capability validation

**Priority:** P1. **Classification:** missing Product capability and confirmed
validation-timing defect; LC precision default already correct.
**Owner:** LCProp, with LaunchPlane interaction limited to committing editor
values.

An executed-request summary and PR planning exist, but there is no equivalent
general explicit pre-run request inspection surface. Construction is not
strictly pure: BeamPanel commits pending numerical edits, and builders update
optical context. These are UI mutations, not scientific solves. Separate
**commit controls**, **construct/validate request**, and **inspect request**
so validation can be reused predictably.

The transverse eigensolver checks its one-channel restriction inside execution,
after runtime construction. Validate it before starting when transverse
refinement is selected. Do not impose that restriction indiscriminately on
all LC propagation or fixed-point workflows.

**Precision correction:** LC GUI already uses float64. Static requests inherit
`RuntimeOptions(precision="float64")`; TD construction specifies it explicitly.
No float64-default fix is required. Show precision in preview/Help. A precision
selector is a separate capability decision requiring workflow-specific support.

For remote runs, distinguish requested scientific settings from the selected
cluster/resource and resolved execution plan.

**Acceptance:** users inspect the constructed request before execution and
receive applicable capability failures before a worker starts, without changing
scientific settings or silently dispatching work during preview.

**Evidence:** [BeamPanel snapshot](../../src/lcprop/gui/panels/beam_panel.py),
[PR builder](../../src/lcprop/pr/gui/request_adapter.py),
[transverse check](../../src/lcprop/lc/workflows/soliton_trans.py),
[runtime defaults](../../src/lcprop/lc/requests.py),
[LC construction](../../src/lcprop/lc/gui/main_window.py).

### HA-06 — Explain soliton termination and expose member diagnostics

**Priority:** P1. **Classification:** confirmed scientific-presentation defect;
alternating flags require investigation of the actual run history.
**Owner:** LCProp LC products and shared results views.

**Important correction:** the GUI's existence sweep constructs a
`ParameterSweepRequest` containing a default `SolitonRequest`. It does not use
the separate `SolitonExistenceRequest` defaults.

| Entry path | Outer limit | Field mixing | Residual RMS tolerance | Residual maximum tolerance |
| --- | ---: | ---: | ---: | ---: |
| GUI existence sweep: default `SolitonRequest` | 100 | 0.5 | `5e-3` | `5e-2` |
| Separate `SolitonExistenceRequest` API | 80 | 0.25 | `1e-3` | `1e-2` |

Both use `tol_field=1e-4`, `tol_theta=1e-5`, and require all four strict checks
in the same outer iteration in `soliton.py`: field relative change, director
RMS update, director residual RMS, and director residual maximum. The GUI's
independent parallel starts use the first row. A completed nonconverged member
can simply have exhausted its iteration budget; this is neither physical
nonexistence nor a stability determination.

For equally normalized fields, `field_rel² ≈ 2 × (1 − overlap_abs)`. An overlap
of `0.999999` still gives field change about `1.414e-3`, failing `1e-4`. A rounded
near-unity display does not establish convergence.

Per-member results and histories are retained, but presentation is incomplete:

- Curves say **Converged** and **xs and ys**.
- Structured Samples exist; the generic diagnostics renderer shows only a row
  count instead of an inspectable table.
- Applicable tolerance reference lines are absent.
- Near-unity curves retain Matplotlib's default offset formatting.
- Generic sweep samples do not expose all termination information retained in
  the member result.

Use **Solver converged within configured limits**. Expose termination reason,
failed gates, actual tolerances, and completed/budgeted iterations. Provide a
member table with power, beta, residual RMS/max, field change, overlap, theta
update, and status. Rename widths **Transverse RMS widths**. Make overlap
readable; optionally offer clearly labeled `1 − overlap`.

Explain continuation as initialization from a previously converged member,
not a waiver of convergence checks. The fixed-point solver versus optional
transverse eigenpair refinement is intentional and documented. The transverse
solver has a fifth optical-residual gate; its final polishing does not update
the outer-loop convergence flag. Its returned arrays and mixed outer/final
diagnostics require clear provenance rather than conflation with the sweep
solver.

**Scientific decision gate:** do not change iteration limits, mixing,
tolerances, solver selection, or scientific convergence semantics as a
presentation fix. Such changes require separate numerical validation. The
reported smooth-looking alternating family cannot be diagnosed member by
member without its actual request and history.

**Acceptance:** a nonconverged completed member remains inspectable and shows
which configured gate failed, without being presented as physical nonexistence.

**Evidence:** [GUI sweep construction](../../src/lcprop/lc/gui/main_window.py),
[defaults](../../src/lcprop/lc/requests.py),
[four-condition solver](../../src/lcprop/lc/workflows/soliton.py),
[transverse solver](../../src/lcprop/lc/workflows/soliton_trans.py),
[sweep semantics](../../src/lcprop/lc/workflows/sweep.py),
[products](../../src/lcprop/lc/products.py),
[table rendering](../../src/lcprop/gui/workspace.py),
[curve rendering](../../src/lcprop/gui/views/curve_view.py),
[scientific contract](../science/lc_model_contracts.md).

### HA-07 — Reconcile backend, resource choice, and planning output

**Priority:** P1. **Classification:** confirmed usability/scientific-presentation
gap; part of the requested behavior already exists.
**Owner:** LCProp PR GUI and shared remote controls.

PR tracks whether backend selection is implicit or explicitly selected/loaded.
Switching to a GPU-capable Slurm resource automatically chooses CuPy only while
the selection remains implicit; explicit NumPy is preserved. CPU Slurm is
valid. Full-transverse GUI validation requires explicit NumPy or CuPy and
rejects `auto`.

Run Planning reports both Local Mac/NumPy and H200/CuPy estimates. It receives
the scientific request, not the selected cluster/resource, so the H200 estimate
is a comparison scenario, not evidence of the configured run's backend. Remote
validation rejects CuPy on an unsuitable CPU profile, but has no corresponding
warning for NumPy on an allocated GPU.

Show the actual planned execution separately from comparison estimates. Warn
that NumPy on a GPU allocation will not use the GPU for scientific computation.
Preserve expert choices and reversible defaults; never equate all Slurm with
CuPy. Resolve any Auto presentation against the selected model's supported
backend contract before execution.

**Acceptance:** Local, CPU Slurm, and GPU Slurm configurations show their actual
backend and precision; estimates cannot be mistaken for an execution plan.

**Evidence:** [backend context](../../src/lcprop/pr/gui/evolution_panel.py),
[model validation](../../src/lcprop/pr/gui/request_adapter.py),
[planning invocation](../../src/lcprop/pr/gui/main_window.py),
[estimate formatting](../../src/lcprop/pr/runtime_estimator.py),
[remote validation](../../src/lcprop/gui/remote_execution.py).

### HA-08 — Make installed-package remote deployment explicit

**Priority:** P1 guidance/preflight; P3 deployment design.
**Classification:** confirmed architectural limitation/missing Product
capability already acknowledged by release documentation.
**Owner:** LCProp transport/deployment.

Automatic deployment derives `local_source` from the installed module location
and requires a clean Git checkout. A normal installed package points into its
Python environment rather than a deployable checkout, explaining
`source_not_git_checkout`.

Existing alternatives include programmatic `local_source` selection and a
pre-staged remote source plus SHA, including environment overrides. The profile's
**Remote source root** is not a local-checkout selector. Release qualification
explicitly distinguishes installed-package Local execution from automatic
clean-checkout Slurm staging.

**Immediate acceptance:** detect the missing deployable source before submission
and provide actionable configuration guidance in the primary GUI.

**Later architecture decision:** compare an explicit local checkout selector,
a reproducible packaged deployment artifact, and the existing pre-staged route.
Preserve source identity and environment compatibility. Do not require editable
installation as a blanket workaround.

**Evidence:** [runner composition](../../src/lcprop/transport/defaults.py),
[source resolution](../../src/lcprop/runners/source_deployment.py),
[release qualification](lcprop_release_hygiene_rc_preparation.md).

### HA-09 — Complete ordinary-user installation and documentation entry points

**Priority:** P1. **Classification:** confirmed documentation/usability defect.
**Owner:** LCProp documentation; LaunchPlane owns its own installation guidance.

Quick Start clones LaunchPlane, then runs `pip install -e '.[gui]'` without
cloning LCProp or establishing its working directory. README assumes a
checkout. Historical **LaunchPane** wording remains in user-facing material
and import-error guidance; standardize display names on **LaunchPlane** without
casually renaming compatibility imports.

Provide a complete installation path from a known starting directory and
environment. Distinguish ordinary non-editable installation from developer
editable installation using the modes already qualified. Explain the remote
deployment limitation separately.

Built-in Help already renders Markdown with `QTextBrowser.setMarkdown`, but
references to full guides are literal repository paths rather than a clear
rendered-document journey. Supply a discoverable full-guide route for installed
users; a wholly new Markdown renderer is not necessarily needed.

Suggested exploration wording: **Use modest settings for quick exploration.
Verify grid, step-size, and model convergence before drawing quantitative
conclusions.**

**Acceptance:** a fresh user can follow installation and read the guide without
inferring a missing clone, directory change, or Markdown reader.

**Evidence:** [Quick Start](../user/quick_start.md), [README](../../README.md),
[import guidance](../../src/lcprop/gui/panels/beam_panel.py),
[Help renderer](../../src/lcprop/gui/help.py).

### HA-10 — Preserve user-local profiles and fill management/isolation gaps

**Priority:** P1 fresh-user verification; P2 bounded management improvements.
**Classification:** existing architecture largely correct; missing management
capabilities; personal-default observations require provenance investigation.
**Owner:** LCProp shared remote configuration and preferences.

The current multi-cluster catalog already persists outside the repository and
package in the platform's per-user configuration location: Application Support
on macOS, AppData on Windows, and XDG configuration elsewhere.
`LCPROP_CLUSTER_CONFIG` overrides the catalog location. An absent file gives an
empty catalog and a clear no-profile state. This path injects no personal
cluster endpoint.

| Capability | Current support or gap |
| --- | --- |
| Multiple named clusters | Supported in catalog and GUI selector. |
| Multiple resources per cluster | Supported. |
| Create/edit profiles | Supported by the shared remote dialog. |
| Rename cluster/resource | Supported through name edits; saving replaces the old identity. |
| Delete cluster | Supported. |
| Duplicate profile | No dedicated duplication action found. |
| Delete individual resource | No dedicated action found. |
| Personal default cluster | Catalog/API support exists; no explicit GUI make-default/clear-default control found. First save establishes a default. |
| Default resource | Supported, but saving an edited resource implicitly makes it the cluster default. |
| Select cluster/resource for a run | Supported; selection alone is not separately persisted as the last-session choice. |
| Session restoration | Catalog/defaults reload; complete last-session selection restoration is absent. |
| Ordinary reinstall survival | External user configuration survives ordinary package replacement. |
| Persistent versus per-run settings | Backend, precision, and retrieval are separate controls; polling and cleanup are persistent cluster settings. |

A fresh Python environment under the same operating-system account is not a
fresh preference environment. Existing user configuration plausibly explains
the observed personal cluster, but its exact provenance was not established.
Do not remove working profiles or redesign the catalog on that basis.

Save/Open passes an empty initial directory to native file dialogs; it does not
hard-code a developer experiment directory. Investigate native dialog/session
state and launch context for the observed location.

For profile-only clean testing, the existing catalog override can point to an
isolated absent catalog without touching real profiles. This is not a complete
clean-user mode, particularly for native dialogs. Document that limited
mechanism now. Later add explicit default/duplicate/resource-management actions,
preference-location visibility, and a comprehensive temporary isolation mode.
Personal configurations must remain user-owned and outside distributed Product
defaults.

**Acceptance:** no-profile launch is neutral; multiple profiles survive restart;
editing one resource does not unexpectedly change an unrelated choice; clean
testing does not delete real preferences. Investigate comprehensive isolation
separately from the existing profile-only mechanism.

**Evidence:** [catalog and defaults](../../src/lcprop/runners/cluster_profiles.py),
[profile editor and empty state](../../src/lcprop/gui/remote_execution.py),
[file dialogs](../../src/lcprop/gui/experiment_files.py),
[configuration documentation](../../src/lcprop/transport/README.md).

### HA-11 — Make layout responsive and recover Beam workspace area

**Priority:** P1/P2. **Classification:** human-confirmed usability defects with
source-supported causes; exact clipping remains platform-dependent.
**Owner:** LCProp shared/LC/PR layout; LaunchPlane for internal canvas interaction.

Execution controls are crowded into horizontal rows. PR forms use default
`QFormLayout` behavior without an explicit responsive growth policy. Several
labels already enable wrapping, so adding wrapping alone is insufficient:
allocation and size negotiation need review.

The LC Beam tab always constructs the Input Screen pane with a 360-pixel
minimum width and places the optical-boundary form above the editor. LC disables
input screens but retains their pane. This supports the wasted-space observation.

Use expandable selectors, reflowable execution controls, and collapsible or
tabbed secondary beam configuration. Restore basic readability immediately;
keep broader workspace rearrangement bounded. Validate supported Mac window
sizes rather than accumulating fixed-width patches.

LaunchPlane already supports dragging beam centers and tilt handles. Easier
manipulation is an improvement request, not absent functionality. Review hit
targets, selection visibility, and canvas allocation before redesigning it.

**Acceptance:** meaningful control text and guidance remain readable, and the
disabled screen pane does not permanently crowd out beam manipulation.

**Evidence:** [PR layout](../../src/lcprop/pr/gui/main_window.py),
[PR form](../../src/lcprop/pr/gui/evolution_panel.py),
[Beam layout](../../src/lcprop/gui/panels/beam_panel.py),
[LC configuration](../../src/lcprop/lc/gui/main_window.py),
[LaunchPlane interaction](https://github.com/mcroning/LaunchPlane/blob/add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8/src/launchplane/canvas.py).

### HA-12 — Add display-only scaling and temporal scale locking

**Priority:** P1. **Classification:** confirmed missing Product capability and
scientific-presentation defect. **Owner:** LCProp shared field views.

ImagePane computes new limits on every field refresh. Its `_scale_limits` cache
is written with new limits rather than reused to lock scale. Longitudinal views
also recompute limits when their field changes. This supports apparent TD
brightness jitter.

Provide Auto, fixed limits, and explicit temporal locking. Add percentile/log
options only where meaningful for the quantity. Share normalization coherently
across linked slices. Never change numerical arrays, quantitative curves, or
exported scientific values through display controls.

**Acceptance:** scientific data remain identical before/after display changes,
and unchanged values maintain brightness across material-time frames when locked.

**Evidence:** [image limits](../../src/lcprop/gui/views/image_pane.py),
[longitudinal limits](../../src/lcprop/gui/views/longitudinal_pane.py).

### HA-13 — Complete live accepted-state visualization by workflow

**Priority:** P1 for existing wiring; P2 for new bounded products.
**Classification:** mixed: LC capability exists; PR GUI wiring and
full-transverse optical progress products are missing.
**Owner:** LCProp LC/PR products, shared views, and transport for remote previews.

| Workflow | Verified finding |
| --- | --- |
| LC TD local | Emits current output intensity, intensity stack, and director state; GUI converts these to live x-y and longitudinal products. |
| PR reduced TD local | Emits optical/material state and source intensity, but GUI progress updates text instead of rendering these states. |
| PR full-transverse TD local | Emits `psi_current`; no corresponding live optical field product is supplied. |
| PR Slurm | Receives structured remote status; final preview/movie transport does not establish live field-preview transfer. |

Investigate why existing LC views were not useful during acceptance: selection
resets, default field choice, scaling, cadence, or workflow state. Do not build
replacements before assessing the existing path.

For PR, define bounded optical preview products tied to the correct accepted
material state. Do not display a predictor or an optical observation from a
different time index as current. New optical observation or transfer work must
make its cost and cadence explicit. Remote previews need bounded transfer;
do not stream full evolving volumes by default.

**Acceptance:** available accepted-state fields are discoverable and correctly
time-labeled. Distinguish existing-payload wiring from new optical/replay or
transport capabilities requiring separate validation.

**Evidence:** [LC progress](../../src/lcprop/lc/workflows/timedependent.py),
[LC live products](../../src/lcprop/lc/products.py),
[PR reduced progress](../../src/lcprop/pr/workflow.py),
[PR transverse progress](../../src/lcprop/pr/transverse/workflow.py),
[PR GUI progress](../../src/lcprop/pr/gui/main_window.py),
[remote status retrieval](../../src/lcprop/runners/slurm.py).

### HA-14 — Expose existing PR movies and distinguish them from scientific history

**Priority:** P2. **Classification:** existing capability with discoverability
and availability gaps; LC playback remains missing.
**Owner:** LCProp PR visualization and shared workspace.

PR already samples up to 36 material-time frames, downsamples output-plane
intensity to at most 128 × 128, and encodes an MP4 with a fixed scale across
all retained frames. Frame/time/normalization metadata are retained. The GUI
offers **Open downsampled TD preview** when the artifact exists.

Frames are collected when a progress callback is supplied. Encoding requires
ffmpeg; missing/failed encoding records a warning and produces no movie, hiding
the button. There is no obvious Evolution recording control or in-application
material-time scrubber.

Expose preview availability and failure reasons, explain what is recorded, and
reuse the existing artifact path. An MP4 is not full quantitative accepted-state
history. Retaining inspectable arrays or adding LC playback is a separate bounded
storage/product enhancement.

**Acceptance:** users can find an available preview and understand why one is
unavailable, including encoder failure, without mistaking it for missing science.

**Evidence:** [movie bounds and encoding](../../src/lcprop/pr/visualization.py),
[reduced collection](../../src/lcprop/pr/workflow.py),
[transverse collection](../../src/lcprop/pr/transverse/workflow.py),
[playback button](../../src/lcprop/gui/workspace.py).

### HA-15 — Extend canonical scattering to reduced static deliberately

**Priority:** P2. **Classification:** confirmed missing Product capability,
not merely hidden controls. **Owner:** LCProp PR requests, workflow,
persistence/transport, and GUI.

The GUI hides scattering only for reduced static. Full-transverse static
already accepts and applies canonical scattering. Reduced `PRStaticRunRequest`
has no scattering field, and its GUI builder does not supply one.

The reduced-static gap therefore requires request/workflow and persistence/
transport wiring plus numerical validation, not just a visible checkbox. Reuse
the canonical realization and physical-z partition contract. Review application
across coupled passes and final replay so one realization remains consistent.

Help should state the operational limitation without presenting it as a
scientific reason static fanning lacks scattering.

**Scientific decision gate and acceptance:** qualify no-scattering compatibility,
canonical realization reuse, and partition/replay behavior before offering the
new reduced-static capability. Preserve existing full-transverse support.

**Evidence:** [GUI branches](../../src/lcprop/pr/gui/request_adapter.py),
[reduced-static schema](../../src/lcprop/pr/static_workflow.py),
[full-transverse application](../../src/lcprop/pr/transverse/static_workflow.py).

### HA-16 — Explain scattering strength and make sampling warnings actionable

**Priority:** P1 documentation; P2 richer diagnostics.
**Classification:** observed zero default contradicted by source; confirmed
guidance gap. **Owner:** LCProp PR.

The GUI initializes strength to `1e-8`, correlation length to 2 µm, seed to 0,
and canonical slab spacing to 1 µm. Zero is explicitly permitted. The headless
specification requires epsilon rather than providing a zero default. Investigate
saved state or editing provenance for the observed zero; there is no supported
basis for replacing a current zero starter default.

Explain **enabled with zero strength: no scattering**, preserving zero as valid.
Document accumulated-phase-variance normalization, correlation, seed, slab
alignment, and algorithm identity. These are setup choices, not universal
material constants. Changing starter values requires scientific review.

The grating warning computes minimum coherent sampling but emits a generic
message. Retain the offending beam pair, limiting direction, estimated period,
spacing, and samples per period so the remedy is concrete.

**Acceptance:** defaults are described accurately; zero is explained rather than
silently replaced; a sampling warning identifies the affected configuration.

**Evidence:** [GUI defaults](../../src/lcprop/pr/gui/evolution_panel.py),
[canonical normalization](../../src/lcprop/pr/scattering.py),
[sampling analysis](../../src/lcprop/pr/geometry.py).

### HA-17 — Rewrite Help around scientific decisions and edge treatment

**Priority:** P1. **Classification:** confirmed documentation and
scientific-presentation defect. **Owner:** LCProp; LaunchPlane for its own
focus terminology.

Help already correctly distinguishes Sponge's distance-accumulated attenuation
rate from Tukey's discrete window. Preserve that distinction. The boundary
heading and selection wording obscure that FFT propagation remains periodic.
Use **Optical edge treatment** with **None / Sponge / Tukey**, explaining the
periodic domain and unchanged stationary-soliton restrictions. Change
presentation, not mathematics.

Use scientific names such as **Optical intensity volume**, **Optical intensity
at selected z**, and **Current accepted state at t=…**. Say **linked x-y, x-z
and y-z slices** rather than unexplained MPR; define multiplanar reconstruction
once if retained. Put authoritative/replay/provenance distinctions in diagnostics.

Organize Help as physical meaning → what to choose → relevant qualification.
PR Slurm guidance should explain choosing a cluster/resource and checking CPU/
GPU execution operationally; unrelated LC restrictions belong in LC guidance.
Explain carrier input/output powers and gain, including separation limitations,
rather than only saying “Fourier-space power.” Existing carrier diagnostics
retain unavailable-gain reasons that can support the explanation.

**Acceptance:** a scientist new to LCProp can identify the intended choice and
interpret results without understanding implementation jargon. Preserve the
existing boundary mathematics and scientific qualifications.

**Evidence:** [Help](../../src/lcprop/gui/help.py),
[User Guide](../user/user_guide.md),
[carrier diagnostics](../../src/lcprop/pr/carrier_power.py).

### HA-18 — Investigate physically meaningful launch trajectory overlays

**Priority:** P3. **Classification:** missing visualization capability requiring
a reviewed scientific convention. **Owner:** LaunchPlane rendering/model;
LCProp supplies host optical context.

Current arrow length uses an arbitrary scale of 500 times angle components or
transverse-wavevector components, not exit displacement. Optical context already
contains refractive index and interaction length.

A future overlay should use conserved transverse wavevector and the host's
actual angle/refraction/propagation convention. Label it a geometric/reference
trajectory and visibly indicate off-aperture endpoints. It must not imply a
prediction of nonlinear bending or beam evolution. Review drawing and inverse
drag-edit mapping together.

**Scientific decision gate:** qualify the convention and off-aperture behavior
before representing the arrow as a predicted exit location.

**Evidence:** [pinned LaunchPlane arrow implementation](https://github.com/mcroning/LaunchPlane/blob/add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8/src/launchplane/canvas.py).

### HA-19 — Review Gaussian profile simplification without acceptance-time schema changes

**Priority:** P3. **Classification:** usability proposal requiring scientific/API
compatibility design. **Owner:** LaunchPlane and LCProp launch adapter.

Focused and collimated profiles are distinct serialized choices. Focused
geometry computes entrance width and curvature from focus position; it is more
than a label. Improve waist/focus explanation immediately through HA-17.

Later unification must preserve old fields, launch meaning, migration behavior,
and focus-position semantics. Do not assume moving a fixed waist far away
reproduces every existing collimated launch.

**Scientific decision gate:** review physical equivalence and schema/API
compatibility before unifying profiles.

**Evidence:** [pinned LaunchPlane model](https://github.com/mcroning/LaunchPlane/blob/add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8/src/launchplane/model.py),
[LCProp launch](../../src/lcprop/optics/launch.py).

### HA-20 — Separate physical TD change, numerical accuracy, and experiment observables

**Priority:** P3, with P1 wording protection.
**Classification:** missing scientific diagnostics/capabilities requiring
scientific design. **Owner:** LCProp PR science and products.

Current scalar products include material-state-change RMS and, where available,
minimum carrier density. Source calls the first quantity change, not integration
error; preserve that correct interpretation.

An equation-defect diagnostic must match the actual algorithm. Reduced nonlinear
TD uses a linearly implicit predictor/corrector with source evaluations at
accepted and predicted states; full-transverse nonlinear TD offers IMEX Euler;
linearized paths use frozen-source modal evolution. A generic trapezoidal
residual is inappropriate across all workflows. A small discrete solve defect
does not by itself measure temporal truncation error.

Review optional sampled step-doubling separately, including cost and the full
step being compared. Do not impose it on every run.

For physical curves, reuse carrier-power analysis where applicable, but define
pump/signal roles, separation requirements, normalization, and zero-input
handling before adding `log(P_signal,out / P_signal,in)` versus time. Retain raw
signal/pump powers where appropriate. Fanning efficiency requires a recovered
or separately validated definition, including angular/integration regions;
do not invent it during GUI work.

**Scientific decision gate:** validate metric definitions, interpretation,
cadence, and cost independently from presentation changes.

**Evidence:** [TD curves](../../src/lcprop/pr/products.py),
[reduced integrator](../../src/lcprop/pr/evolution.py),
[transverse evolution](../../src/lcprop/pr/transverse/workflow.py),
[carrier power](../../src/lcprop/pr/carrier_power.py).

### HA-21 — Keep general metric registration outside the acceptance batch

**Priority:** P3. **Classification:** future architecture.
**Owner:** LCProp shared products with material-owned metric implementations.

`RunData`, `CurveData`, `DiagnosticData`, and artifacts already provide reusable
presentation structures. They are not the proposed registry declaring required
inputs, evaluation cadence, units/labels, and live/final suitability.

Use existing structures for bounded acceptance improvements. Design a general
derived-metric extension system separately when concrete requirements justify
it, so adding a metric can be a bounded extension rather than another broad
acceptance cycle. Do not introduce that architecture during immediate remediation.

**Evidence:** [product data model](../../src/lcprop/products/data_model.py).

### HA-22 — Preserve multichannel/vector LC eigensolitons as Research work

**Priority:** P3. **Classification:** Research capability.
**Owner:** LCProp-Research initially; later Product integration requires a
separate contract and public validation evidence.

The transverse eigensolver supports one optical channel. Earlier GUI rejection
is an acceptance fix; implementing a multichannel/vector eigenproblem is not.
It requires a physical formulation, solver design, normalization and coupling
conventions, and validation evidence. No private Research material is required
to act on the Product acceptance items in this document.

**Evidence:** [single-channel transverse solver](../../src/lcprop/lc/workflows/soliton_trans.py).

## Successful findings to retain

- **Valid experiment Save/Open works.** The manual session demonstrated valid
  PR Image-Amplification persistence. Current paths preserve scientific requests
  and LaunchPlane presentation state. Specialized validation failures are not
  evidence that persistence itself is broken.
- **Non-editable Local installation is valid.** Prior release qualification
  records installed-wheel checks; the manual session additionally reported
  LC/PR GUI startup in the separate environment. These were not rerun by the
  audit. Automatic Slurm deployment has a distinct source requirement.
- **LC already defaults to float64.** No corrective precision-default change
  is required.
- **Multi-cluster persistence already exists outside the package.** Extend
  management and clean-user testing rather than replacing the catalog.
- **LC live TD fields and PR bounded fixed-scale movies already exist.** Improve
  discoverability and fill workflow-specific gaps.
- **Soliton/existence results retain useful diagnostics and physical curves.**
  Improve access and interpretation without changing convergence rules to make
  flags look smoother. Preserve completed nonconverged members for inspection.
- **Existing scientific distinctions remain valuable.** Preserve specialized
  image validation, full-transverse static scattering, intentional fixed-point
  versus transverse-refinement semantics, and Sponge/Tukey mathematical behavior.

Prior qualification evidence: [release hygiene record](lcprop_release_hygiene_rc_preparation.md).

## Recommended remediation sequence

### Stage 1 — Human-acceptance remediation; no new physics

Address the narrow general-TBC versus specialized-image distinction in HA-01;
request/result identity and retained-result labeling in HA-02; subtab retention
in HA-03; status, actionable errors, and console boundaries in HA-04; request
inspection and applicable early validation in HA-05; convergence interpretation,
tables, thresholds, and labels in HA-06; and actual-plan versus comparison-estimate
presentation in HA-07.

Include HA-09 installation/documentation, immediate HA-10 fresh-user provenance
checks and documentation of existing profile isolation, HA-11 basic layout
readability, HA-12 display-only scaling/locking, and HA-17 scientific Help.
Include HA-08 guidance/preflight for the existing deployable-source requirement,
HA-16 accurate defaults/zero-strength guidance, and HA-20 protection against
calling state change an integration-error estimate. Diagnose existing LC live
view usability under HA-13 before replacing functionality.

**Boundary:** no new physics, solvers, scientific-default changes, relaxed
validation, or numerical convergence-semantic changes. Preserve all successful
acceptance findings.

### Stage 2 — Existing capability exposure and wiring

Expose existing LC live-state products and wire existing PR reduced-TD progress
payloads where accepted-state timing is already valid (HA-13). Expose existing
bounded PR movies, metadata, and unavailable/encoding-failure guidance (HA-14),
without treating MP4 previews as quantitative history.

Complete bounded profile-management actions on the existing catalog (HA-10):
explicit defaults, duplication, resource management, and predictable session
selection. Improve beam workspace allocation and existing manipulation
discoverability (HA-11). Add actionable sampling details derived from the
existing analysis (HA-16).

**Boundary:** reuse existing scientific capabilities and contracts. Work needing
new optical observations, retained-array products, or remote transfer contracts
belongs in Stage 3 rather than being disguised as GUI-only wiring.

### Stage 3 — Scientifically bounded Product extensions

Add reduced-static canonical scattering only with explicit request/workflow/
persistence/transport integration and numerical validation (HA-15), preserving
the existing full-transverse implementation and no-scattering behavior.

Separately scope new bounded full-transverse optical progress products, live
remote preview transfer, retained inspectable time history, or LC playback
where new products are needed (later portions of HA-13/14). Specify accepted-state
time provenance, cadence, memory/transfer limits, and any observation/replay cost.
Any future precision selector must respect qualified workflow support (HA-05).

**Boundary:** each extension receives its own scientific or numerical validation
plan and compatibility checks. Do not bundle speculative metric definitions or
unreviewed default changes into this stage.

### Stage 4 — Later architecture and Research

Keep packaged remote-deployment architecture (HA-08), comprehensive clean-user
preference isolation beyond the current catalog override (later HA-10), reviewed
trajectory overlays (HA-18), Gaussian-profile unification (HA-19), integrator-
quality diagnostics and fanning/TBC metric definitions (HA-20), general metric
registration (HA-21), and multichannel/vector LC solitons (HA-22) as separately
scoped work. Any proposed scientific-default or solver-semantic change requires
a separate design and validation decision rather than acceptance-driven tuning.

**Stage 4 does not block ordinary Product acceptance remediation.** This punch
list records the roadmap; creating it does not begin Stage 1 or authorize any
implementation.

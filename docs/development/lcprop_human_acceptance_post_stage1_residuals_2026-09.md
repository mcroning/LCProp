# Post-Stage-1 human acceptance reconciliation — September 2026

## Provenance and scope

This record preserves the supplied post-Stage-1 human acceptance observations
and reconciles them against Product source. It supplements, and does not replace,
the [canonical HA-01–HA-22 punch list](lcprop_human_acceptance_punch_list_2026-09.md).
Observation identifiers A1–M3 below retain the supplied report's grouping.

Source baseline: LCProp `a56fc08a9f2415eadc8184836a11d1955022716f`, branch
`feature/pr-second-order-static`. Context: LaunchPlane
`add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`; private Research
`7c04f57397b2e378cf4041d805bfdfea0a4a585b`. Research is not required to use this
Product document. Source references below describe this LCProp baseline unless
explicitly identified as LaunchPlane references.

The observations are human-reported hands-on evidence, not a new automated or
interactive acceptance run by this reconciliation. The acceptance program began
with a fresh-user macOS session, a separate Python 3.12 environment, and a normal
non-editable installation. This continuation concerns LC/PR Results, launch
inspection, configuration presentation, Help, and a reported fanning attempt.
No screenshot, saved request, timing measurement, or cluster-run evidence is
invented to fill gaps in that report. Source confirms implementation paths; it
does not reproduce native clipping, establish which result policy was selected,
or qualify a physical result.

This task performs source inspection and creates this document only. It runs no
tests, GUI, simulations, fanning, SSH, Slurm, CUDA, or remote queries. Existing
development records remain unchanged. Their historical test results are not
new validation results for this reconciliation.

**Classification does not authorize implementation.** P1 means bounded residual
acceptance remediation without new physics; P2 means a separately bounded
Product capability or investigation with direct user value; P3 means separate
scientific or architectural work. Priority does not imply scientific validity.
The classification vocabulary distinguishes ACCEPTED / CONFIRMED IMPROVEMENT,
RESIDUAL ACCEPTANCE DEFECT, NEW USABILITY REQUIREMENT, DEFERRED PRODUCT CAPABILITY,
SCIENTIFIC HOLD, and INVESTIGATION NEEDED.

## Stage 1A–1C acceptance outcome

- [Stage 1A](lcprop_human_acceptance_stage1a.md), commit
  `048075ac5e87f0dead3a5391008fb40b13f50970`, established request/result ownership,
  selected-subtab preservation, general-TBC versus Image-Amplification
  separation, and truthful Local-only PR Continue behavior. These safeguards
  remain requirements for every follow-up.
- [Stage 1B](lcprop_human_acceptance_stage1b.md), commit
  `bcdecf2509af80af16c717d4ac45482368d82751`, made retained convergence evidence,
  execution configuration, and display-only scaling accessible. Human evidence
  confirms their usefulness; primary summaries and control readability still
  need attention. GUI existence sweeps use the actual default `SolitonRequest`
  inside `ParameterSweepRequest`, not separate existence-request defaults.
- [Stage 1C](lcprop_human_acceptance_stage1c.md), commit
  `a56fc08a9f2415eadc8184836a11d1955022716f`, improved Beam space, layout,
  installation guidance, profile default controls, and Help. Native readability
  and progressive explanation remain residual concerns. Its recorded green
  suite does not invalidate subsequent human usability observations.

The outcome is substantial accepted improvement with residual presentation
work and separately scoped capability requests. It is not a claim that all
human acceptance items or later scientific work are complete.

## Accepted improvements

All entries in this section are **ACCEPTED / CONFIRMED IMPROVEMENT**; preserve
them rather than assigning new implementation priority.

| Observations | Human acceptance evidence | Source support and preservation requirement |
| --- | --- | --- |
| A1, M1 | LC TD live frames made evolution followable; scale locking was useful; final physical curves, including RMS width against cumulative material time, helped interpretation. | LC progress adapters feed shared Results [S1, S5]. Preserve Auto/manual/locked behavior and display-only semantics; this is not evidence for changing the default scale mode. |
| A2 | Console remained selected during progress. | Shared Workspace preserves Results selection [S1]. HA-03 applies to all future live refreshes and failures. |
| A3, M2 | Samples, sweep-member rows, exact gate values/tolerances/pass-fail, physical curves, and tolerance lines were useful. Near-unity overlap became readable without the misleading offset. Transverse RMS width labels improved interpretation. | Material-owned soliton presentation and shared tables/curves [S2] expose retained evidence. Preserve near-unity resolution and exact evidence access during formatting work. |
| A4 | Ordinary screen-free two-beam PR Inspect Request passed without specialized Image Amplification screen/symmetry/geometry/pump restrictions. | PR request construction and validation distinguish workflows [S3]. General TBC must not be redirected into the specialized experiment. |
| A5 | Configured Slurm/H200 Large/CuPy was distinct from comparison estimates; explicitly selecting NumPy warned that scientific computation would not use the allocated GPU. | Execution presentation [S3, S9] supports this distinction. This is configuration/UI evidence, not evidence of an H200 or remote run. |
| A6 | Beams, Optical edge treatment, and Input Screen tabs recovered substantial LaunchPlane space. | Shared Beam host [S6]. Retain this allocation while resolving individual clipping. |
| M3 | Completed PR results successfully used shared Results. | PR final-result handlers and product adapters [S3, S5]. Live deficiencies do not justify a separate PR final-result viewer. |

## Residual P1 acceptance and presentation work

Unless an entry says otherwise, these are **RESIDUAL ACCEPTANCE DEFECTS**, owned
by Product GUI/documentation maintainers. Existing values/state are sufficient;
no scientific computation, request/schema/codec change, or persistence change
is proposed. Acceptance criteria below describe future review gates, not work
performed by this task.

### B1 — Scientist-facing language (P1; HA-04/17)

The LC experiment panel literally says “Static and time-dependent propagation
are wired to LocalRunner.” [S4] This confirms the reported implementation jargon.
Use task language such as execution on this computer where appropriate. Audit
the immediate related presentation, without a global terminology rewrite or
removal of useful expert diagnostics. Acceptance: a scientist can understand
the supported operation without knowing internal runner class names.

### B2 — Meaningful result ownership (P1; HA-02/03)

Workspace visibly includes numeric request sequence identifiers in ownership
labels [S1]. These IDs are useful internally but human examples such as request
6/11/13 do not identify the scientific result intuitively. Lead with Completed
result, Previous result, or Current accepted state while retaining provenance.
Acceptance: labels remain truthful across populated-to-empty results, failed
request construction, and partial rendering failure; no old pane silently takes
new ownership and the selected subtab remains selected.

### B3 — More visible Console boundaries (P1; HA-04)

**Correction:** timestamped boundaries already exist. `operation_boundary`
appends a local ISO timestamp and operation; request and file-operation paths
call it [S1, S3]. The human finding is insufficient visual separation, not
absence of all timestamps or operation delimiters. Make boundaries easier to
scan and associate warnings/failures with the relevant operation. No exact
decorative glyph or logging redesign is prescribed.

### B4 — Quantity-aware display precision (P1; HA-06/12)

Reported examples include long coordinates, `0.005000000000000001`, lengthy
beta/material values, unwieldy charge density, and trailing zeros in limits or
time steps. Tables use `repr(float)` and several curve/scale controls use `.17g`
[S2]. Some readouts already use compact formatting: this is not a claim that
every numeric widget prints raw precision.

Define a shared display policy with quantity, units, magnitude, and diagnostic
resolution in mind. Illustrative compact forms are `0.005`, `0.428402`,
`9.99714e-5`, and `6.4e22`; they are examples, not universal significant-digit
limits. Preserve full values in inspection/export where appropriate, lossless
edit semantics, gate evaluation, serialization, and arrays. Do not round a
near-unity overlap into a misleading constant or turn an actual gate failure
into an apparent pass.

### B5 — Coordinate/readout layout (P1; HA-11)

The human reports x/y/z readouts crowding the 3D field selector in LC and PR.
The shared longitudinal view owns these controls [S2]. Reflow or separate them
and apply B4 without altering coordinates. Native clipping is reported, not
reproduced here. Acceptance: field selection and coordinate values remain
readable at supported window sizes without requiring a much larger window.

### B6 — Scale-control layout and terminology (P1; HA-11/12)

Human evidence reports cramped entries and confusing Auto/Fixed/Lock choices.
Source already has Auto, Fixed/manual, and Lock across frames, including finite
limit validation and shared same-source scale identity [S2]. Explain the three
concepts: automatic scaling for the current view/frame, explicit user limits,
and a common retained scale across frames. Make limit entries readable/editable.
Changing the evolving-view default to locked is only an investigation, not an
approved behavior change. Preserve the accepted functionality and session-only
state; do not alter data or save display limits as scientific parameters.

### B7 — Longitudinal availability wording (P1; HA-02/13)

The longitudinal pane contains the reported fixed “No longitudinal fields are
available for this experiment.” fallback [S2]. During a supported running
workflow it can be read as a capability denial. Distinguish not available yet,
not retained by this result policy, and unsupported content using actual
operation/product state. Do not promise a developing view that does not exist.
This is wording/state presentation; E2 is the separate data capability.

### B8 — Material-time step terminology (P1; HA-17)

PR labels the existing `Nt` control “Material steps in segment” [S4]. Prefer an
explanation such as “Material time steps this run,” with Run/Continue scope
clear. Keep the value, meaning, cumulative-time accounting, and Local-only
Continue contract unchanged.

### B9 — Empty Samples / Tables (P1; HA-06)

Workspace adds this tab unconditionally; the table pane consumes existing
diagnostic tables [S1, S2]. The human LC TD empty view is consistent with a
result lacking table products. Show a clear empty explanation, or disable/hide
appropriately without violating selected-subtab preservation. Exposing already
retained scalar rows is bounded presentation work if verified available.
Creating new samples, retained histories, or scientific diagnostics is P2/P3,
not permission to fabricate a meaningful-looking table.

### B10 — Remaining profile-selector clipping (P1; HA-11)

The report's “Legacy entrance Gaus...” control is owned by LaunchPlane:
`src/launchplane/launchpane.py`, profile combo text “Legacy entrance Gaussian,”
at the recorded LaunchPlane commit. LCProp hosts it through the shared Beam
panel [S6]. The string/ownership are source-confirmed; the clipping cause is
not reproduced. Owner: joint Product host-layout / LaunchPlane UI triage.
Determine whether available host width or the embedded inspector constrains
it before authorizing a repository-specific fix. Do not enlarge the whole
window indiscriminately or modify LaunchPlane under a Product-only scope.

### C1–C4 — Convergence interpretation and discovery (P1 except C3)

- **C1 — NEW USABILITY REQUIREMENT, P1 (HA-06):** supplement expert tables with
  a primary summary identifying member power/name, failed gate, actual value,
  tolerance, and termination/budget. The human examples were 0.2 mW with
  field-relative change about `5.17e-4` and 1 mW with about `1.19e-4`, each above
  the required strict `< 1e-4`; other displayed gates passed. These are supplied
  examples, not rerun values or hardcoded cases. Existing member/gate records
  support a summary [S2]. Missing evidence must remain explicitly unavailable.
- **C2 — RESIDUAL ACCEPTANCE DEFECT, P1:** failure to converge within the
  configured budget is not proof of physical nonexistence or instability.
  Explain which requirements remain unmet, without inventing an accuracy or
  physical-validity score. Small director residuals, near-unity overlap, and
  small director change do not override a failed field-relative gate or another
  required qualification. Final polishing must not rewrite the recorded outer
  convergence decision.
- **C3 — NEW USABILITY REQUIREMENT, P2:** distinguish nonconverged members on
  physical curves such as beta versus power using neutral markers/styles.
  Retain points and values; do not imply instability. Existing member status is
  available, but association of samples and curve styling needs explicit
  presentation wiring and review [S2]. No new solver is needed; a generic curve
  metadata extension may require product/serialization compatibility review.
- **C4 — RESIDUAL ACCEPTANCE DEFECT, P1:** clearly separate successful execution
  completion from solver convergence. Workspace operation status and material
  solver evidence are different existing inputs [S1, S2]. Acceptance: a returned
  nonconverged sweep is neither presented as an execution exception nor as a
  fully converged physical solution.

### G1 — Cross-material Open Experiment (INVESTIGATION NEEDED, P1)

**Correction:** source does not support a confirmed cross-material application
defect. The shared chooser can select either file, but both main windows call
`load_experiment(..., expected_material_id=...)` before applying GUI state.
Persistence raises `ExperimentMaterialError` before request decoding when the
material differs; the GUI has a material-specific rejection message [S7].
Existing tests explicitly cover rejection before state/checkpoint changes in
both directions; these were read, not run.

Clarify whether the human observed chooser selection, a rejection message, or
actual application of foreign settings. A future reproduction should retain
the file's declared material and installed version. Preserve settings and
checkpoints on rejection; status/Console may legitimately report the failed
open. No new schema or duplicate guard is justified by this evidence. Only
source-confirmed residual rejection/discoverability defects belong in polish.

### H1–H2 — Task-oriented Help (P1; HA-17)

The human found improved accuracy but dense Fanning, Slurm, and Results text.
Built-in Help and the guide contain the relevant explanation [S8]. H1 is a
RESIDUAL ACCEPTANCE DEFECT: organize meaning, action, interpretation, limitation,
then deeper guide links. Keep deployment/SHA/`local_source` mechanics available
where operationally necessary without making them the introductory explanation.
H2 is a NEW USABILITY REQUIREMENT within this P1 presentation scope: scannable
Fields, Curves, Samples/Tables, Diagnostics, Request, and Console sections,
followed by scaling, TBC, and convergence explanations. Preserve installed-Local
versus deployable-checkout Slurm guidance, scientific caveats, general TBC
distinction, and Local-only Continue. No scientific or persistence change.

## P2 Product capabilities and bounded investigations

These entries require separate authorization and focused design/validation.
They must preserve selected tabs, truthful ownership through partial failure,
bounded memory, configured-versus-actual execution truth, and display-only scale
semantics. Owners are identified per entry.

### D1 — Channel-generic diagnostics

**NEW USABILITY REQUIREMENT, P2.** Owner: material product adapters and shared
Results. Human reports that aggregate x/y RMS curves are ambiguous for multiple
beams. Existing products use total/coherence-aware optical intensity and expose
some aggregate curves [S5, S10]. Label aggregate metrics as total-intensity
metrics where that is their actual definition. Do not assert that all current
metrics already exist per channel.

Future generic N-channel products should use retained channel identity/name
and expose centroids x/y, widths x/y, power, and peaks where physically valid.
Propagated channel fields and coherent interference totals are distinct; no
two-channel special case or invented split of an inseparable coherent total.
Final amplitudes can support some derived products; bounded time histories
require deliberate calculation/retention. Product keys, identity mapping,
units, normalization, Fast/Full coverage, and transport/persistence need review.
Scientific definitions and validation precede presenting new metrics.

### D2 — Optional general-TBC pump/signal roles

**DEFERRED PRODUCT CAPABILITY, P2 with P3 metric-definition dependencies.**
Owner: PR request/workflow and experiment-product maintainers. Specialized
Image Amplification already has role controls [S3, S6]; that is not a general
TBC role contract. Permit optional user-owned roles if separately designed;
never require them for ordinary TBC, infer them from beam order, or assume
exactly two beams. Extensible identity, request validation, experiment codecs,
saved metadata, and downstream diagnostic definitions need explicit review.

### E1–E2 — Shared live and developing Results

**DEFERRED PRODUCT CAPABILITY, P2** (HA-13/14). Owner: PR/LC workflow progress
adapters and shared RunData/Results, not a new PR viewer.

E1: LC progress handling converts live static/TD states to RunData. PR progress
handling principally reports progress text, while completed PR results use
shared products [S1, S3, S5]. Thus existing final-view machinery is reusable;
equivalent live PR accepted-state wiring is incomplete. Availability of final
arrays does not establish availability of a safe bounded live payload.

Use accepted scientific state → material-owned bounded product adapter → shared
RunData → shared Workspace. Exclude predictor/intermediate states unless
explicitly distinguished. Review callback ownership, array lifetime/copies,
cadence, cancellation, result policy, and transport. Local wiring does not
authorize remote streaming or remote continuation. Remote cadence and bounded
artifact transfer would be separate capabilities.

E2: developing xz/yz views during optical propagation are a distinct bounded
product request. Reuse shared longitudinal views, but determine which accepted
slices exist at each update, their validity extents and coordinate ownership.
Do not copy full optical volumes on every TD frame. Progress/product/transport
wiring and numerical non-interference validation are needed; scientific state
must not be generated merely to fill an empty viewer.

### F1–F2 — Far-field availability and bounded time evolution

**F1: INVESTIGATION NEEDED / DEFERRED PRODUCT CAPABILITY, P2.** Owner: PR product
adapters and Results discovery. **Correction:** PR far fields are not universally
missing. Reduced Fast static and TD paths call `_fast_optical_run_data`, which
adds an output far field when wavelength/reference-index launch metadata exist.
Specialized Image Amplification adds linear/log fields; full-transverse product
code also calls the shared spectrum helper and exposes far-field products [S5].
Generic reduced Full conversion does not simply take the Fast helper path.
The human report of not readily finding a far field therefore needs workflow,
policy, metadata, selected field, and discoverability checks before adding code.

Retained output-plane complex amplitudes and launch context can support missing
final products, but coverage is not universal by assertion. Reuse the established
direction-cosine axes and normalization [S10]; do not call dimensionless axes
spatial coordinates or invent an angle convention. Preserve total/coherence and
channel distinctions, units, provenance, and missing-metadata explanations.
Exposure-only fixes may be small; new keys/retention/transport require review.

**F2: DEFERRED PRODUCT CAPABILITY, P2/P3.** Bounded material-time far-field frames
could support later fanning interpretation. Final far fields do not imply an
existing time-resolved complex-field history. Define accepted-state sampling,
FFT cost, cadence, retained size, result policy, and movie/scientific-data
distinction before request/codec/transport wiring. This is not authorization to
run fanning or use visual far fields to qualify the scattering model.

### I1 — Preserve launch intent in Inspect Request

**NEW USABILITY REQUIREMENT, P2.** Owner: LCProp request presentation with
LaunchPlane schema/convention ownership. PR preview explicitly lists resolved
phase gradients in rad/µm [S3]. The Beam host can access LaunchPlane state;
resolved channels alone do not necessarily retain the user's editor convention
[S6]. Where available, show user launch intent alongside the resolved quantity.
Do not infer an angle through an unverified index/refraction convention.
Determine whether intent survives saved/reopened experiments before choosing
presentation-only wiring versus additive provenance/schema work. Mathematics,
launch arrays, and request meaning remain unchanged.

### J1 — LC starter material-time duration

**INVESTIGATION NEEDED, P2; no default change approved.** Owner: LC workflow/GUI
and numerical validation. The LC GUI solver panel initializes `Nt=2` and
`dt=750 × 1e-6` [S4]; the human report of two steps is source-consistent. The
suggested exploratory value 20 is a proposal, not a qualified default.

Source-only cost estimate: increasing 2 to 20 multiplies repeated material-step
work by roughly ten if per-step cost remains comparable; initialization and
output costs need not scale equally. LC workflow passes Nt into the material
evolution driver [S11]. This does not establish that 20 is quick on a user's
Local hardware. Default-grid/backend timing, stability, physical time span,
memory/output cadence, and useful visible evolution require separately
authorized validation. No timing benchmark or calculation was run here.

## P3 scientific and architecture holds

### I2 — Geometric reference trajectory

**DEFERRED PRODUCT CAPABILITY, P3** (HA-18). Owner: LaunchPlane/LCProp scientific
convention review. Preserve the requirement for a future entrance-to-z=L
reference endpoint from the established wavevector/refraction convention.
Current tilt-arrow semantics must not be interpreted as a nonlinear exit-position
prediction. Existing launch context is relevant [S6, S10 and HA-18]; a qualified
overlay needs convention review, host length/index context, and off-aperture
handling. It is not residual layout polish and must not imply nonlinear bending
or self-action. Context/provenance interfaces may need extension.

### L1–L2 — Integrator-specific quality diagnostics

L1 is **INVESTIGATION NEEDED, P3**; L2 is **DEFERRED PRODUCT CAPABILITY / scientific
diagnostic, P3** (HA-20). Owner: material numerical-method maintainers, followed
by product/diagnostic presentation maintainers.

Human evidence found physical width curves but no clear integration-quality
measure. Source already distinguishes some state-change and residual products:
full-transverse PR exposes material-state-change RMS and can retain equilibrium
and TD RHS residual arrays [S5, S11]. Their existence does not establish a
timestep truncation-error estimator or universal coverage.

Keep three concepts separate: accepted-state change measures evolution;
equation/step defect assesses satisfaction of a specified discrete update;
truncation error requires a justified estimator such as appropriately designed
step doubling. Never relabel state change as integration error. Audit each
actual LC and PR integrator, its discrete update, inner-solver residuals,
cheap defect opportunities, estimator cost/cadence, units, and interpretation.
Do not impose a universal trapezoidal residual on different integrators.
New computation/history would require request/product/persistence review and
numerical validation, not just a generic RMS plot.

### Other retained P3 boundaries

Gaussian-profile unification (HA-19), fanning/TBC metric definitions (HA-20),
general metric registration (HA-21), and vector/multichannel LC eigensolitons
(HA-22) retain the original punch list's separate scientific/architecture
ownership. Per-channel propagation diagnostics do not implement vector LC
eigensolitons. Gaussian simplification must not silently change saved launch
definitions. These are deferred capabilities, not uncompleted P1 polish.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**. PR Continue remains
intentionally Local-only. Checkpoint, retention, provenance, and remote-state
lifecycle design must not enter any residual usability or live-view milestone.
This is not permanent rejection; reconsideration requires future operational
need and separate authorization. No remote systems were queried here.

## Explicit PR scattering scientific hold

### J2, K1 — Starter strength and equivalence are unqualified here

**SCIENTIFIC HOLD / INVESTIGATION NEEDED, P3.** Owner: PR scattering scientific
review, with Research ownership of legacy/paper equivalence evidence.

The current GUI initializes scattering strength to `1e-8` [S4], consistent with
this human report. It is not changed to make a short run visually interesting.
An earlier report of zero remains a provenance question, not justification for
an unsupported default correction.

The supplied acceptance session attempted to reconstruct a known PR fanning
calculation using epsilon, transverse correlation length, seed, canonical slab
delta-z, and `canonical_phase_slabs_v1`. The user questioned an independently
specified slab spacing when the remembered paper/legacy specification used
amplitude and correlation length. **The attempt does not qualify canonical
scattering, establish equivalence, or establish incorrectness.** Presence or
absence of expected fanning is not a correctness test by itself.

Source explicitly describes a white-in-z random-phase interpretation and
canonical slab increments, including epsilon as accumulated phase variance and
alignment of intervals to slabs [S12]. Those implementation statements are not
independent verification of the authoritative Photonics/legacy definition.
Equivalence was not established in this acceptance session. Existing canonical
partition tests, if any, would not alone recover the missing historical model
contract. No paper or legacy equivalence investigation is performed here.

### K2 — Separately authorized scientific audit

**SCIENTIFIC HOLD, P3.** Recover the authoritative paper and legacy implementation
rather than reconstructing them from memory. Compare amplitude normalization,
transverse correlation definition, longitudinal random-medium construction,
z dependence, optical steps/substeps, and phase/intensity conventions.

Determine whether canonical slab spacing is an invariant realization parameter,
a discretization with a convergent legacy limit, or additional unintended model
dependence. Mathematical and numerical evidence will be required. No new epsilon
value, slab spacing, scattering code, default, request schema, or fanning claim
is authorized here. Reduced-static canonical scattering remains a separate
Product extension; full-transverse static scattering already exists, as recorded
in HA-15. Neither fact resolves equivalence.

### K3 — No unsupported angle explanation

**INVESTIGATION NEEDED under the same SCIENTIFIC HOLD, P3.** Preserve the human
discussion's rejection of a modest common input angle as the assumed first-order
cause of missing robust fanning. Geometry can affect trajectories, aperture
interaction, periodic-edge proximity, and details; this record makes no causal
diagnosis. Do not attribute the discrepancy to angle without evidence, and do
not alter launch mathematics to produce a desired picture.

## Recommended action sequence

1. **Residual polish, P1, no new physics:** scope B1–B10, C1/C2/C4, and H1/H2 as
   presentation work. Include G1 only if further evidence establishes a real
   rejection/discovery defect. Preserve all Stage 1 safeguards. Do not absorb
   defaults, solvers, histories, remote continuation, or scattering work.
2. **Separately bounded Product capabilities, P2:** evaluate E1 live accepted
   states and E2 developing views; D1 channel diagnostics; F1 existing far-field
   discovery/coverage and F2 bounded evolution; optional D2 roles; C3 convergence
   markers; and I1 launch-intent provenance. Identify existing data before
   proposing new products. These need not share a commit. J1 remains a default
   investigation pending authorized numerical and timing validation.
3. **Separate scientific/architecture review, P3:** preserve the J2/K scattering
   hold, L integrator diagnostics, I2 trajectories, Gaussian unification,
   experiment metric definitions/registration, and vector LC solitons. Remote
   continuation remains deferred and unauthorized. None blocks ordinary P1
   acceptance remediation; scattering-dependent qualification claims must wait
   for the relevant scientific evidence.

This sequence is a recommendation only. No implementation stage is started or
approved by recording it.

## Source reference inventory

References use repository-relative paths and symbol names so they remain useful
without personal filesystem locations. All LCProp paths below were inspected
at the recorded baseline; line numbers are intentionally not treated as stable
identifiers. LaunchPlane ownership is explicitly separated in B10/I2.

- **S1 — Shared ownership and lifecycle:**
  [Workspace](../../src/lcprop/gui/workspace.py), especially `begin_request`,
  `mark_previous`, `operation_boundary`, `finish_attempt`, and result updates;
  [Results panel](../../src/lcprop/gui/panels/results_panel.py).
- **S2 — Numerical evidence and shared views:**
  [soliton presentation](../../src/lcprop/lc/soliton_presentation.py),
  [tables](../../src/lcprop/gui/views/table_pane.py),
  [curve pane](../../src/lcprop/gui/views/curve_pane.py),
  [curve view](../../src/lcprop/gui/views/curve_view.py),
  [display scales](../../src/lcprop/gui/views/display_scale.py), and
  [longitudinal view](../../src/lcprop/gui/views/longitudinal_pane.py).
- **S3 — PR orchestration:**
  [PR main window](../../src/lcprop/pr/gui/main_window.py), request preview,
  `_on_progress`, `_on_result`, loading, and Continue dispatch.
- **S4 — Existing text and defaults:**
  [LC experiment panel](../../src/lcprop/lc/gui/panels/experiment_panel.py),
  [LC solver panel](../../src/lcprop/lc/gui/panels/solver_panel.py), and
  [PR evolution panel](../../src/lcprop/pr/gui/evolution_panel.py).
- **S5 — Material product/live adapters:**
  [LC main window](../../src/lcprop/lc/gui/main_window.py),
  [LC products](../../src/lcprop/lc/products.py),
  [PR products](../../src/lcprop/pr/products.py), and
  [full-transverse PR products](../../src/lcprop/pr/transverse/products.py).
- **S6 — Launch hosting and roles:**
  [Beam panel](../../src/lcprop/gui/panels/beam_panel.py),
  [launch configuration](../../src/lcprop/optics/launch_configuration.py),
  [LaunchPlane adapter](../../src/lcprop/adapters/launchplane.py), and
  [PR image input](../../src/lcprop/pr/gui/image_input_panel.py).
- **S7 — Material-aware loading:**
  [experiment persistence](../../src/lcprop/persistence/experiments.py),
  [GUI experiment messages](../../src/lcprop/gui/experiment_files.py),
  [GUI loading tests](../../tests/test_experiment_gui.py), and
  [persistence tests](../../tests/test_experiment_persistence.py), plus S3/S5.
- **S8 — Help:** [built-in Help](../../src/lcprop/gui/help.py) and
  [user guide](../user/user_guide.md).
- **S9 — Execution summary:**
  [request transparency](../../src/lcprop/gui/request_transparency.py).
- **S10 — Shared optical definitions:**
  [far-field spectrum](../../src/lcprop/optics/farfield.py),
  [intensity/propagation](../../src/lcprop/optics/splitstep.py), and
  [launch context](../../src/lcprop/optics/launch.py).
- **S11 — Numerical workflow boundaries:**
  [LC TD workflow](../../src/lcprop/lc/workflows/timedependent.py),
  [reduced PR TD](../../src/lcprop/pr/reduced_linearized_timedependent.py), and
  [transverse TD reference](../../src/lcprop/pr/transverse/linearized_timedependent_reference.py),
  with existing residual products in S5. This is a location map for a future
  integrator audit, not completion of that scientific audit.
- **S12 — Current scattering contract:**
  [canonical scattering](../../src/lcprop/pr/scattering.py),
  `PRCanonicalScatteringSpec` and interval-alignment validation; GUI controls
  are in S4. Inspection was limited to classification of the scientific hold.

## Review boundary

Only this development record is a candidate. No source, tests, defaults,
scientific arrays, request/codec/persistence definitions, active prompts,
original punch list, or Stage 1 record is changed. Product remains at the
recorded commit, with this document unstaged. LaunchPlane and Research are
protected. A strictly read-only pre-commit review must verify observations,
source qualifications, candidate identity, and historical-material preservation
before any separate commit authorization.

Preservation qualification: before this document was created, the untracked/
ignored inventory already contained 4,332 paths rather than the prior 4,131.
The complete pre-edit fingerprint was
`a7b3f02a2043fb7def986e43529bcef06cade8d2a114c6e838d7daa829f32b25`.
All prior paths remained present, but `.DS_Store` and six `src/lcprop.egg-info`
files differed in content and/or recorded metadata from the old snapshot.
The original 4,131-path subset then hashed to
`4b4763cc14183e7523fc14c1b56c3a18de81d49fbae53f7f3aec576e72feb94f`,
not the historical
`2aba45157752b965e820b955e14ef0e65d2a24c5336290731df559eeaecc7e05`.
The extra paths include existing build output. Their origin is not established
by this audit. No restoration, deletion, or cleanup is authorized: preserve the
actual pre-edit material and report this pre-existing mismatch explicitly.

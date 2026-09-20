# P2B-2 — General PR configuration and optional image analysis

## Scope and baseline

Baseline: `cf9d7276ba361cd1c18603801581393e098734a8` on authoritative
`feature/pr-second-order-static`. Development uses isolated branch
`feature/p2b2-general-pr-configuration`. Equations/model changes: **N/A**.
This bounded implementation follows the
[general beams and carrier-power contract](../architecture/pr_general_beams_and_carrier_power_contract.md)
and preserves the
[continuation decision](../architecture/continuation_state_and_interventions_decision.md).

The authoritative Product checkout is protected throughout development.
Preflight historical inventory: **4,337 paths**, fingerprint
`7f599a03f7924083c4336e6d20729dc4d481436b05dd7e645f7a10ec98bd0595`.
The external inventory records path, mode, size, modification time and content
hash. LaunchPlane remains `add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`;
Research remains `b947fac51722eafa971e756a2cc2053be065294d`.

## Previous behavior and implementation disposition

Ordinary PR already supported arbitrary BeamStacks and declarative screens,
including multiple independently screened channels. The GUI's specialized
Image Amplification mode built a composite request whose validator required
exactly two suitable Pump/Signal channels, a signal screen and specialized
geometry. Those restrictions belonged to the experiment, not general PR.

The Input mode selector now exposes:

- **General beams / two-beam coupling**: the existing ordinary route;
- **Image amplification — optional analysis**: the same ordinary scientific
  request and selected workflow, with an optional post-processing attempt;
- **Historical Image Amplification — compatibility**: the established composite
  request/validation/result path for historical APIs and saved experiments.

The shared Beam workspace and Input Screen editor remain authoritative. No
second beam or screen configuration architecture is introduced. New optional
configurations support N >= 1, absent screens, screens on any supported beam,
multiple screens, supported asymmetric geometry/coherence groups and a
zero-power channel when total launch power is positive. Ordinary shared-
wavelength and selected material-model restrictions remain in force.

[PR GUI dispatch](../../src/lcprop/pr/gui/main_window.py) snapshots optional
roles on the GUI thread before worker execution and freezes configuration
controls during the run. The worker calls the existing ordinary dispatch once,
then the [optional analysis adapter](../../src/lcprop/pr/optional_image_analysis.py).
This retains the ordinary reduced Local live-results path, ordinary static
progress handling and existing execution-target/preflight behavior.

The historical post-processing body in
[image amplification](../../src/lcprop/pr/image_amplification.py) is extracted
behind a callable seam accepting an already executed base result. The legacy
wrapper still prepares, executes and post-processes its composite request with
its historical status semantics. The new optional adapter invokes that same
analysis without dispatching a second material run.

## Optional-analysis semantics and compatibility

Missing roles/screens, unsuitable channel count, geometry or coherence, and
unavailable existing carrier-separation evidence make optional analysis
**unavailable**, with a reason in Results Diagnostics and Console. The adapter
uses the current two-carrier diagnostic applicability decision; conservation
alone is not a new resolvability test. No N-carrier attribution is introduced.

Analysis exceptions, augmentation failures and cancellation after successful
propagation retain the ordinary result and its scientific status. Unsuccessful
base runs do not undergo optional analysis. Image products are added only after
successful analysis. Base diagnostics, including existing `carrier_gain`, are
preserved under their original keys. Optional status and role provenance occupy
a separate diagnostic. Existing image metrics and benchmark quantities retain
their meanings; the linearized model remains a uniform-reference tangent model,
with applicable analysis validation qualifications preserved.

Historical specialized request classes, old mode identifier, saved-file codecs,
normalization, role semantics and composite failure behavior remain unchanged.
Opening an old specialized experiment explicitly selects the compatibility
mode. The new ordinary mode does not silently reinterpret old saved files.

No persistent schema change is needed. **Save Experiment in optional-analysis
mode saves the ordinary scientific request and reopens as General beams.**
Optional analysis selection is session-only, disclosed beside the selector,
in the request summary and in the [user guide](../user/user_guide.md).
Roles use existing name/enabled-channel-index mapping and a frozen per-run
snapshot; they are not stable Product-owned identities. A Clear analysis roles
button allows ordinary propagation with neither role assigned. Future durable
identity and persistence require their own compatibility design.

Ordinary reduced Continue remains Local-only with unchanged checkpoint and
request compatibility checks. Optional analysis applies to fresh Run, not
Continue. The historical composite path still has no Continue action. No
continuation interventions are implemented.

## Validation and self-review

Validation uses Python 3.12, isolated `src` on PYTHONPATH, offscreen Qt,
bytecode/pytest cache disabled and external temporary cache directories.
These are software regressions on small fixtures, not scientific commissioning.
No Slurm, SSH or CUDA execution is performed.

The [new regression module](../../tests/test_pr_general_configuration.py)
exercises real GUI Run → worker → ordinary runner → progress/Workspace → final
Results delivery. Its 19 cases cover N=1/2/3/4; no/arbitrary/multiple screens;
asymmetric geometry; missing roles; a zero-power channel; exact ordinary-request
and scientific-array equivalence; screen attenuation without renormalization;
unchanged unscreened channels and carrier diagnostics; rejected wavelength and
continuation changes; historical saved-file round trips; ordinary Save/reopen;
existing specialized metrics; unavailable separation; injected analysis and
augmentation failures; post-run cancellation; and all four ordinary workflows.
The reduced TD cases verify first accepted live state and paired-cut fields.

During test development, fixture assertions were corrected to use `E_final`,
match the workflow's complex128 launch dtype, and use a patterned absorbing
raster: an all-positive constant raster normalizes to a transparent screen
under the existing screen contract. These were test-fixture corrections;
production normalization and screen semantics were not changed.

Focused new module: **19 passed in 9.63s**. Affected regression batch:
**455 passed in 50.10s**, including PR GUI, image analysis/readiness, all image
workflow variants, live Results, checkpoints, screens/launch, persistence,
experiment GUI and human-acceptance transparency. The earlier compatibility
batch passed 128 tests. Collection sanity: **1,891 tests collected**.
Complete `tests/`: **1,815 passed, 77 skipped in 514.29s**; zero failures,
no baseline exceptions. Development self-review: **Passed**.

In-memory Python syntax compilation, Markdown structure/local link targets,
whitespace checks (including new files) and `git diff --check` passed. No
public positional API is removed; existing scientific requests and codecs
remain unchanged. Protected repository identities and the complete historical inventory match
preflight, including modification times. The candidate boundary contains seven
files and the index
is empty. No unrelated implementation hunks or new architecture/scientific
blockers were found. Numerical commissioning manifests and schema gates are
N/A: this milestone creates software validation evidence only.

## Non-change and deferred work

No equations, optical kernels, coherence physics, launch normalization, screen
transmission, integrators, defaults, carrier-power algorithms or gain definitions
change. No request/codec/checkpoint/persistence schema or continuation checks
change. No remote execution or retention implementation changes. No scientific
source volume is added to progress delivery.

P2B-3 N-carrier powers/ratios/TD curves, stable persistent channel IDs, P2A-2
full-transverse live Results, new far-field capabilities and N-beam scientific
attribution remain deferred. **PR scattering equivalence: SCIENTIFIC HOLD /
UNESTABLISHED. Remote continuation: DEFERRED / NOT AUTHORIZED.**

Remaining native acceptance: inspect selector/help readability and wrapped
layout on macOS; configure and remove screens on different beams; observe
optional-analysis reasons in Diagnostics/Console and current live Results;
confirm historical versus ordinary Save/reopen presentation. Offscreen tests
do not qualify subjective native layout or cluster/GPU operation.

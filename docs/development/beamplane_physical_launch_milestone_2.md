# Physical launch Milestone 2 — shared resolver and Product integration

Status: Review remediation Development + Local Validation passed; awaiting
strictly read-only pre-commit review.
Native acceptance and coordinated integration remain pending.

This candidate implements the committed [physical launch contract](../architecture/beamplane_physical_launch_contract.md)
against LCProp `c317f49d34a9e16247ae6c567a5dcc42c3f27f30` and the exact
LaunchPlane Milestone 1 commit `c577b5b0053b078b083b08561d8f27ecb932c4cc`.
Authoritative repositories remain protected. No installed package is changed.

## Ownership and physical semantics

`optics/physical_launch.py` owns scalar refraction, rolled transverse frames,
external face quadratic forms, derived internal quadratic forms, analytic launch
normalization, and spectral qualification. `optics/launch.py` applies laboratory
screens after sampling. LC and PR workflows continue through that shared launch
entry point with their existing material index contexts. Propagation kernels,
material equations, integrators and acceptance ordering are unchanged.

External radii are authoritative. Tangential k is conserved across the interface;
internal slopes use kx/kz and ky/kz. Invalid forward modes are rejected. The
external Rodrigues frame and independent roll determine the unique face field.
Internal beam-normal dimensions are derived from that face, not substituted for
external radii. Phase is referenced to the entrance-axis intersection.

A is irradiance amplitude divided by the square root of total requested incident
power. Its analytic scale includes the external/internal cosine ratio and
pi*w1*w2/2. Sampling never rescales aperture loss. Per-channel zero power remains
valid; an all-zero stack remains invalid. Existing laboratory raster orientation,
placement and multiplication are unchanged. Focused physical launch and
beam-normal screens remain unsupported; no old-profile fallback is provided.

Capture and screen-throughput estimates are central-direction scalar flux,
separate from unweighted norm and requested power. Tolerances are explicit:
a minimum forward cosine of 1e-8 (numerically grazing modes are rejected),
1e-3 capture deviation, 1e-2 spectral relative cosine RMS, 1e-6 nonpropagating norm
fraction and 1e-6 norm in the outer 10% of the Nyquist range. A carrier beyond
Nyquist invalidates physical-power qualification even if its aliased spectrum
looks narrow. External as well as internal bandwidth is checked. Qualified
homogeneous scalar axial-current references use angular-spectrum kz/k weighting;
coherent groups are summed as fields before that reference is evaluated. This is
not an anisotropic/vector Poynting claim. Material phase evolution can preserve
unweighted norm without preserving this weighted reference.

LC output presentation labels per-channel summed currents as scalar lineage
references, excluding coherent cross terms. Existing PR carrier partitions,
separation logic and endpoint norm-ratio algorithms are preserved. Their mW
conversion is unavailable under the new convention because those algorithms
partition unweighted norm. Normalization provenance and the reason are retained.
Specialized image-analysis endpoint mW is similarly unavailable; qualified
post-screen estimates remain distinct from requested power. No carrier histories,
new gain definition, attribution algorithm or generic reconstruction is added.

Explicit prepared fields retain their caller-supplied units/provenance contract.
Launch capture metadata describes the configured physical beams before such an
override; it must not be read as certification of an arbitrary prepared array.

## Models, persistence and preview

Product beam channels now store the same physical intent as LaunchPlane schema 4.
Derived k properties are read-only consumer conveniences, not serialized inputs.
The Product experiment envelope advances from 1 to 2, LC request payload from
2 to 3, and PR request payload from 6 to 7. Obsolete experiment versions are
rejected. Checkpoint/transport beam decoding requires physical-intent fields;
old launch dictionaries cannot acquire new meaning through defaults. Unrelated
checkpoint state and continuation compatibility algorithms are unchanged.

The LaunchPlane seam accepts host-computed laboratory contour points and labelled
text. It performs only x-right/y-up display conversion. External intent arrows
remain unchanged. Scene reset signals invalidate host contours before Qt deletes
them; preview is recomputed after host changes. No physical intent schema change
beyond Milestone 1 is introduced.

## Validation and migration observations

Independent regressions cover Snell refraction, tangential k/grating periods,
arbitrary rolled quadratic forms, internal radius conversion, phase reference,
float32/float64 complex dtype, unclipped flux, finite-window loss, screens,
coherent interference, spectral availability and unchanged partition/ratio
algorithms. Existing workflow fixtures are migrated to explicit physical intent;
obsolete focused/uniform acceptance tests become rejection tests. Physical-flux
assertions use independent spectral formulas, not old L2-as-mW expectations.

The old readiness test used a 200 µm pump radius on a 64 × 32 µm window.
Removing finite-window renormalization changes the represented pump/signal
balance and relaxation time. Its explicit test duration increases from 250 to
1500 steps; all residual/error thresholds and Product solver defaults stay
unchanged. The separately committed positive-carrier material-gate oracle is not
regenerated: its historical numerical source is frozen independently of launch
construction, preserving the original bitwise material regression.

Validation uses Python 3.12.13 with the two isolated source trees on PYTHONPATH,
Qt offscreen, bytecode disabled and pytest's cache provider disabled. No package
installation, cluster access or GPU execution is involved.

- Pre-review focused power/GUI/resolver regressions: 131 passed.
- Pre-review affected launch/adapter/LC/PR/screen/persistence regressions:
  284 passed, 2 skipped.
- Unchanged LaunchPlane complete-suite evidence: 95 passed.
- Pre-review complete Product `tests/` suite: 1889 passed, 77 skipped in 510.13 s.
- Candidate Python compilation, Markdown fence/local-link checks, whitespace,
  `git diff --check` and empty-index checks pass.

Preliminary failures exposed obsolete constructor/schema fixtures, old
sample-renormalized power expectations, the scene-contour lifetime defect, and
stale GUI guidance/extent expectations. They were corrected without loosening
scientific residual thresholds. The offset-image extent oracle now independently
uses the external incidence projection w1/cos(theta_ext); no stored image is
cropped or resampled. A full run before the final reporting/guidance corrections
had 1888 passed, 77 skipped and one stale selector assertion; it is not the final
validation result. The missing/incompatible LaunchPlane regression requires the
new coordinated-package instruction rather than the obsolete install message.

The breadth of the Product candidate principally reflects constructor and
serialization migration across existing tests. Affected scattering/remote tests
only migrate beam fixtures and execute local mocks; no scattering algorithm or
remote execution capability is changed. Stationary LC soliton execution rejects
tilted launch with an explicit unqualified-input message; its eigenmode
normalization and material solver remain unchanged.

## Read-only review remediation

The pre-commit review identified two blocking P2 gaps. The incidence restriction
was applied only in the ordinary soliton function, and LC Console still printed
scalar lineage currents under unqualified physical-power labels.

The existing incidence rule and exact message now live in the shared
`_validate_stationary_launch` helper. Ordinary and direct transverse solvers call
it after request validation and before runtime construction; exported
`polish_soliton` forwards through the same transverse entry point. No
stationary solver equations, eigenmode normalization, tolerances or supported
normal-incidence numerical path changed.

LC Console now consumes the same power-diagnostics helper as Results. It retains
the numerical values and displays `scalar_lineage_axial_current_*_mW` plus the
existing qualification that these currents exclude coherent cross terms and are
not anisotropic/vector Poynting power. Legacy results retain their legacy labels
and qualification. No power, coherence, partition, separation or gain algorithm
changed.

Eleven focused remediation cases pass. They verify rejection before runtime or
normalization for ordinary/transverse/polish calls (including tiny nonzero
incidence), unchanged normal-incidence validation/dispatch, and real LocalRunner
results delivered through the LC window's result-display path to its Console.
Single-channel and identical overlapping coherent beams preserve arrays and
numerical values; the coherent case distinguishes group current from the
lineage sum. Existing normal-incidence solver regressions remain part of the
complete suite.

The first remediation full-suite run found one GUI error-precedence regression
(1899 passed, 77 skipped, 1 failed): placing the rule in general request validation
preempted the existing specific multi-channel refinement error. The rule was
moved to the shared execution-entry helper instead, preserving request-building
and GUI validation order. The existing regression was not weakened or changed.
Final remediation validation passed under the same Python 3.12.13/offscreen
source-tree environment:

- Remediation plus existing GUI transparency module: 42 passed in 19.23 s
  (includes all 11 new remediation cases).
- Previously defined Milestone 2 focused set: 131 passed in 21.41 s.
- Previously defined affected set: 284 passed, 2 skipped in 24.64 s.
- Complete Product `tests/` suite: 1900 passed, 77 skipped in 514.28 s.
- Syntax, Markdown/local links, candidate whitespace, `git diff --check`, and
  empty-index checks passed. Source/test hashes remained fixed during final runs.

Both blocking findings are corrected, subject to a new read-only review.
The remediation changes only this record, `lc/gui/main_window.py`,
`lc/workflows/soliton.py`, `lc/workflows/soliton_trans.py`, and the new
`tests/test_physical_launch_review_remediation.py`. General request validation
is unchanged from the reviewed candidate.
LaunchPlane source, tests and manifest remain unchanged; its existing 95-pass
evidence is retained, with Product-side integration covered by the reruns.

## Boundaries and eventual integration

Do not integrate or install either candidate during this milestone. After both
candidates pass review, integrate Product and LaunchPlane as one coordinated
release boundary while applications are stopped, then install both reviewed
versions into the same environment before restarting either application. Do not
run authoritative old Product with breaking LaunchPlane schema 4. Validate the
installed pair and native GUI before release; local offscreen tests are not GPU
commissioning or Research evidence.

Native checks still required include physical arrow/footprint distinction,
responsive editor layout, angle/roll edits, material-index updates, screen-preview
availability messages, and reopening current experiments. No Slurm/SSH/CUDA,
remote continuation, scattering/fanning, P2A-2, reconstruction, or power curves
are included. Remote continuation remains DEFERRED / NOT AUTHORIZED. PR
scattering equivalence remains SCIENTIFIC HOLD / UNESTABLISHED.

The protected reconstruction-contract candidate is not edited. A later bounded
amendment should describe canonical PR scalar angular-spectrum propagation and
external physical launch with conserved tangential k, while retaining its
unimplemented reconstruction status. The governing physical-launch architecture
contract itself does not require amendment.

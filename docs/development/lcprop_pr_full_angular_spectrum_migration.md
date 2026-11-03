# PR full angular-spectrum migration

Status: Development candidate; local validation recorded below. No commit,
integration, GPU commissioning or generic reconstruction is authorized here.

## Preflight and scientific boundary

Baseline: `a450b1ec5aaaef28b88c8c219b3fd2e6227550b9`.
Development branch: `feature/pr-full-angular-spectrum`, isolated from the
unchanged authoritative Product branch `feature/pr-second-order-static`.
The authoritative untracked reconstruction-contract candidate is excluded and
must remain untouched. The actual pre-development historical inventory is
4,338 paths, separately fingerprinted; no older inventory is restored.

This authorized pre-release scientific migration changes linear PR diffraction
from the carrier-removed Fresnel approximation to full scalar homogeneous
angular-spectrum dispersion. It does not change PR material equations,
normalization, optical-substep counts, source cadence, integrators, acceptance,
scattering, boundaries, checkpoints, transport codecs, continuation, or LC.
No general-user propagation selector, saved-data conversion or checkpoint
version machinery is added. Existing ordinary Fresnel behavior is not retained
as a canonical compatibility mode. Tests are local software regression fixtures,
not immutable-source scientific commissioning or research evidence.

Read with the [PR model contracts](../science/pr_model_contracts.md),
[general-beams contract](../architecture/pr_general_beams_and_carrier_power_contract.md),
[continuation decision](../architecture/continuation_state_and_interventions_decision.md),
and [P2B-2 record](lcprop_p2b2_general_pr_configuration.md).

## Scalar primitive and numerical contract

[scalar_angular_spectrum_kernel](../../src/lcprop/optics/splitstep.py) factors the
previous streaming reference mathematics into one shared primitive:

```text
k = 2*pi*n/lambda
q = 1 - (lambda/n)**2 * (fx**2 + fy**2)
H(d) = exp(i*k*d*sqrt(q)) for q >= 0; otherwise 0
```

Lambda is vacuum wavelength in um; fx/fy are unshifted fftfreq cycles/um.
Full longitudinal carrier phase is retained. Distance may be positive, zero or
negative. Grazing q=0 is retained; negative q is suppressed without an epsilon
or evanescent exponential. Even zero distance projects away nonpropagating
support. Negative distance inverts only retained homogeneous propagation,
not windows, screens, absorption or material evolution.

The backend namespace handles all arrays. q and phase are constructed in
backend float64 then cast to explicitly requested complex64/complex128.
Default output precision follows float32 versus other supplied frequency
arrays. The cutoff uses the supplied grid values, including their prior
rounding; algebraically equivalent floating-point cutoff expressions need not
agree at a theoretically exact grazing bin. No tolerance widens support.
There is no array-to-host conversion, scalar reduction or per-volume allocation
in construction. The same cached 2-D kernel is reused across substeps/passes;
FFT counts and asymptotic optical memory order are unchanged. Extra construction
work is bounded 2-D float64 square root, masking and phase evaluation.

The existing streaming scalar reference helper delegates to this primitive.
Its reference Lie ordering and window/noise semantics remain distinct.
Near-cutoff conditioning and GPU phase accuracy still need commissioning.
This operator plus existing local material phase screens is not a full-vector
or exact inhomogeneous Helmholtz model.

## Exact migrated call paths

| Source | Migrated behavior |
|---|---|
| [reduced TD](../../src/lcprop/pr/workflow.py) | Ordinary source passes, candidate observation, accepted/final optical replay and continuation's existing optical path share the constructed scalar substep kernel. Cached accepted replay and explicitly unpropagated cancellation fallback retain their existing semantics. Live snapshots extract existing accepted fields. |
| [reduced static](../../src/lcprop/pr/static_workflow.py) | Coupled local trials and independent final replay share scalar propagation, including partial completed-slice output. |
| [transverse TD](../../src/lcprop/pr/transverse/workflow.py) | Source passes and final accepted-material replay share scalar propagation. No acceptance-timing shortcut is added. |
| [transverse static](../../src/lcprop/pr/transverse/static_workflow.py) | Outer fixed-point/visibility passes and independent convergence/validation replay share scalar propagation. |
| [marching static](../../src/lcprop/pr/transverse/marching_static.py) | Causal marching trials, correctors and replay use scalar diffraction with the existing explicit optical/material precision boundary. No new canonical status is assigned to this reference workflow. |
| [streaming static](../../src/lcprop/pr/static_streaming.py) | Production nonlinear Strang uses scalar substeps; existing scalar Lie-reference branches share the primitive without changing order. Independent streaming validation retains its existing full rerun semantics. |
| [coupling trace](../../src/lcprop/pr/coupling.py) | Frozen-material trace and matched linear reference fields use the same scalar model as the migrated ordinary run. |

No new FFT propagation is introduced by general PR dispatch, screen assembly,
carrier power, far-field transforms, result conversion or live Results.

## Ordering and material non-change

The prepared-response function itself is unchanged:

1. material half-screen;
2. diffraction at dz/Nsub;
3. material half-screen;
4. per-substep sponge;
5. repeat Nsub;
6. existing once-per-call Tukey boundary;
7. workflow-defined scattering at its existing post-slice location.

The midpoint source remains the existing average of entrance and exit driving
intensities. Material solvers still consume the same normalized intensity
contract, with coherent grouping and incident-channel peak reference unchanged.
No phase operation is moved into a material trial, and no trial, accepted-state,
Stop or final-replay boundary is moved. The migration necessarily changes optical
fields and can change subsequent self-consistent material states. That is not
an authorized change to their equations or acceptance rules.

## Geometry and retained historical behavior

[PR geometry](../../src/lcprop/pr/geometry.py) now computes both components from
one kz: Kx/Kz and Ky/Kz. Crossing centers and both aperture preflight helpers use
these slopes. User phase gradients remain unchanged. Grazing or nonpropagating
carrier centers have no finite forward trajectory and are explicitly rejected
by the geometry helper; this is distinct from the kernel's broadband cutoff.
The historical single-axis `paraxial_kernel_slope` remains available under its
explicit name. Gaussian radius/focus construction remains paraxial: preflight
envelopes and nominal focus are approximate, not promises of exact scalar focus.

The shared `linear_kernel` retains its original Fresnel expression. LC static,
runtime and longitudinal soliton paths still reach it through
`optics.substeps.build_optical_substep_kernel`; no LC call site, existing
split-step operation, or LC scientific test is changed. The transverse soliton
mode solver is also untouched. This source trace qualifies any broader claim
that all pre-existing LC execution was already nonparaxial.

Historical [Image Amplification analysis](../../src/lcprop/pr/image_amplification.py)
keeps its paraxial backward and matched reference propagation. Only an explicit
qualification docstring is added there. It is not a matched scalar inverse of
new ordinary PR output. Its metrics, historical request/codec APIs and optional
analysis failure handling remain unchanged. General scalar reconstruction is
separate future work; exact reproduction of an entire pre-migration ordinary
run requires its old Product revision.

## Independent validation and changed expectations

The new [scalar regression module](../../tests/test_pr_scalar_diffraction.py)
uses independently evaluated longitudinal wavevectors for FFT-bin modes,
positive/negative/zero distance, complex64/complex128, exactly representable
cutoff/grazing cases, and retained-field round trips. It checks the fourth-order
paraxial-limit phase error after removing common phase; Gaussian packet slopes
for on-axis, x/y and asymmetric tilt; and broad curved/screened fields against
an independent spectral oracle. A namespace probe checks backend dispatch
without importing/executing CuPy. It is not GPU parity evidence.

Real executions of all six propagation workflows at Nsub=1 and 3 are checked
against analytic oblique-plane-wave output, including final/validation replay.
Reduced float32/float64 executions check optical output precision. Existing
regressions continue to exercise midpoint reuse, integrator stages, scattering,
replay, cancellation, live snapshots and boundaries. A direct ordering test
checks the shared half-screen/hop/sponge sequence.

Changes to existing tests are limited to five PR modules:

- midpoint intensity reuse, semi-implicit prototype, and marching-static
  independent replay fixtures now select the canonical scalar operator;
- two-beam crossing propagation now uses scalar diffraction; the explicitly
  historical paraxial slope helper assertion remains valid and unchanged;
- vertical-slice manual propagation uses the scalar operator; two uniform
  plane-wave oracles add analytic exp(i*k*actual_distance) to the unchanged
  material phase. Their material/intensity tolerances are unchanged.

No stored output was adopted as a new expected result and no production
scientific tolerance was relaxed. An initial affected run passed 887 tests with
76 skips and failed only the two carrier-phase oracles above. The initial new
broad-spectrum oracle placed bins at a floating-point grazing ambiguity; its
wavelength was moved off that boundary, while dedicated exact/adjacent cutoff
tests retained explicit coverage. This was a fixture correction, not a cutoff
policy change.

Validation uses the established Python 3.12 environment with isolated src on
PYTHONPATH, offscreen Qt, bytecode/pytest cache disabled and external cache/log
locations. No installation or environment mutation is required.

Focused scalar/geometry/replay and affected manual-operator regressions:
**140 passed, 3 skipped in 10.75s**. The final affected PR/optics/launch batch:
**953 passed, 76 skipped in 62.25s**. Collection sanity: **1,955 tests collected**.
The complete Product suite passed: **1,879 passed, 77 skipped in 501.52s**;
zero failures and no baseline exceptions. Skips remain unexecuted qualifications,
including unavailable GPU paths; no CUDA execution was performed.

Commands use `python -m pytest -p no:cacheprovider -q` with:

- focused: `tests/test_pr_scalar_diffraction.py`, `tests/test_pr_two_beam_coupling.py`,
  `tests/test_pr_vertical_slice.py`, `tests/test_pr_midpoint_intensity_reuse.py`,
  `tests/test_pr_semi_implicit_prototype.py`, `tests/test_pr_transverse_marching_static.py`;
- affected: `tests/test_pr* tests/test_optical* tests/test_launch*`;
- full: `tests/`;
- collection: `--collect-only -q tests/`.

An intermediate affected batch had one documentation-link failure because it
ran before this new record existed. The completed-document rerun above passed.
In-memory syntax compilation, Markdown fence/local-link validation and whitespace
checks passed. AST comparison verifies every pre-existing function/class in the
shared split-step module is identical to baseline. The diff leaves LC, material
operators, checkpoint/codec and scattering source untouched. `git diff --check`
and the empty-index check passed again with the final manifest.

Development self-review: **Passed**. The final 19-file boundary includes source,
local tests, the scientific contract/PR README and this record only. No public
request positional fields, persistence structures or continuation checks changed.
All numerical expectation changes have an analytic or declared operator-migration
basis; unchanged material and ordering contracts retain regression coverage.
GPU commissioning/provenance schemas are N/A for this local regression milestone.
The authoritative candidate and historical files retain bytes, sizes, modes
and modification times; protected Product/LaunchPlane/Research HEADs are unchanged.
Nothing is staged or committed.

## Reconstruction amendment after integration

Do not amend or copy the authoritative untracked reconstruction candidate into
this change. After integration it needs a separate documentation amendment:
replace its baseline finding that ordinary forward PR is Fresnel with the
integrated scalar call-path inventory and commit identity; identify the shared
primitive and full-phase/cutoff/dtype contract; record forward reconciliation
as satisfied for new canonical PR; retain historical Image Amplification's
paraxial qualification and the need to identify old-result dispersion. Remove
any implication that preserving accidental ordinary Fresnel checkpoints requires
new compatibility machinery. Generic reconstruction, actual plane provenance,
carrier resolvability, tie ownership and stable identity remain unimplemented
requirements. This milestone does not authorize them.

## Limits and disposition

Local regression evidence does not commission CuPy, GPU complex64 phase accuracy,
near-grazing broadband PR, or large-angle material-response validity. Those are
later bounded qualifications. No benchmark or cluster execution is performed.
No requests, schemas, codecs, continuation semantics, GUI selector, material
model, scattering algorithm, LaunchPlane or Research source is modified.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**. PR scattering
equivalence remains **SCIENTIFIC HOLD / UNESTABLISHED**. No generic reconstruction,
carrier curves, N-beam separation, P2A-2 or fanning work begins.

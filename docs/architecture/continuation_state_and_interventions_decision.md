# Continuation: accepted state and experimental interventions

Date: 2026-09-20. Status: **Accepted future design principle; implementation NOT
AUTHORIZED.** Source baseline: `91ae72b50c58c9c1e6389d080bc591fe61a83698` on
`feature/pr-second-order-static`.

## Decision and scope

**Continuation means preserving an accepted material state and using it as the
initial state of a new evolution segment.** It must not mean that the entire
original scientific request must remain identical except for the number of
additional time steps.

The future contract must distinguish immutable retained-state/grid/model
identity from mutable experimental driving parameters. Immutable identity
preserves what the saved state means and where it came from; it must not turn
every coefficient in a material or optical request into an immutable parameter.
Each material owns the scientific compatibility classification. A parameter
change alone is neither sufficient reason to reject nor sufficient reason to
permit continuation.

The preserved state is the accepted **material state**, not an obligation to
retain the old optical launch. A compatible intervention must reconstruct the
new segment's optical launch from the new request. It must not silently reuse
checkpoint `A0` when that launch differs. This is a future requirement; changed
`A0` reconstruction is not implemented or authorized here.

This decision concerns time evolution from an accepted state. It does not
redefine resumption of an unfinished static spatial solve. It records intent,
not current availability, a schema, or an implementation plan. Existing checks
remain authoritative until separately authorized work establishes the new
contract and validates it.

## Current source behavior

- Reduced PR's [`validate_pr_continuation`](../../src/lcprop/pr/checkpoint.py)
  first validates the checkpoint, then requires equality of grid, material,
  beams, launch elements, backend, scattering, optical boundary, material
  response, and solver options with `Nt` excluded. If supplied, `initial_A` must
  match saved `A0`, and `initial_E` must match saved `E_initial`, after dtype
  conversion. Checkpoint validation includes shapes/dtypes and normalized time
  consistent with completed steps and the original timestep.
- [`continue_pr_timedependent`](../../src/lcprop/pr/workflow.py) starts from
  accepted `E_current`, copies saved `A0`, retains original-request/origin-state
  provenance, and advances cumulative step/time. Thus changing PR gain or beam
  power is currently rejected; merely weakening request equality would still
  leave the old optical input in use.
- LC TD's [`_validate_continuation_request` and `continue_timedependent`](../../src/lcprop/lc/workflows/timedependent.py)
  require equality of grid, material, bias, beams, runtime, optical boundary,
  and solver options except `Nt`; they check saved shapes and time consistency.
  Continuation reuses saved `theta` and `A0`, original-request provenance, and
  prior width history. This is a broad compatibility check, not literal equality
  of every request field: output options are not in that comparison, and input
  arrays are replaced by checkpoint state.
- LC [`continue_static`](../../src/lcprop/lc/workflows/static.py) resumes the
  next uncomputed z slice after checkpoint validation and a serialized-request
  fingerprint check. That resume contract must remain distinct from a new
  temporal intervention experiment.
- Checkpoints are material-owned behind the [shared codec dispatcher](../../src/lcprop/persistence/__init__.py).
  Reduced [PR persistence](../../src/lcprop/pr/persistence.py) currently writes
  schema 4; [LC TD persistence](../../src/lcprop/lc/persistence/timedependent.py)
  writes schema 2. Neither is changed here. Full-transverse PR TD is not thereby
  granted a checkpoint/Continue contract.

The [PR GUI design](pr_gui_vertical_slice_design.md) describes compatibility
revalidation and retained checkpoints. [Stage 1A](../development/lcprop_human_acceptance_stage1a.md)
keeps PR Continue explicitly Local-only. [P2A-1](../development/lcprop_p2a1_reduced_pr_live_results.md)
adds bounded Local Run/Continue presentation without changing compatibility.
These are descriptions of implemented behavior, not restrictions on the future
principle accepted here.

Current [beam specifications](../../src/lcprop/core/beams.py) distinguish power,
phase, transverse center, phase-gradient tilt, and waist/profile/focus intent.
[Launch construction](../../src/lcprop/optics/launch.py) builds those optical
fields and applies [per-channel launch elements](../../src/lcprop/optics/screens.py)
after incident-power normalization, without renormalizing the transformed field.
Raster dimensions are source-image dimensions, not the simulation grid. These
existing building blocks do not establish arbitrary-beam GUI screen support or
intervention compatibility. The specialized [Image Amplification experiment](../../src/lcprop/pr/image_amplification.py)
still requires exactly two enabled channels. Reduced PR retains `E`, whereas
[full-transverse PR](../../src/lcprop/pr/transverse/workflow.py) evolves a
potential-based state; those arrays cannot be interchanged based on shape.

## Future compatibility classification

| Class | Required disposition | Basis and examples |
|---|---|---|
| State-incompatible | Reject without explicit scientifically validated conversion/remapping | Changes to Nx/Ny/Nz, physical coordinates/aperture, discretization, state representation, or model meaning/normalization that invalidate the retained state as an initial condition. Reduced PR state and full-transverse potential are not interchangeable. Matching array shape alone is insufficient. No implicit resampling or reinterpretation; conversion is not authorized here. |
| Compatible experimental intervention | Permit | Intentional changes to driving/optical parameters for which the retained material state remains scientifically meaningful. Required motivating cases are changing PR gain and switching one beam's power to zero. Recompute the new segment's driving inputs consistently with its request while preserving the accepted material state. |
| Scientifically ambiguous | Review explicitly before classifying | Changes whose effect on state meaning, normalization, constraints, or required history has not been established. Neither blanket equality rejection nor blanket permission is the scientific decision. |

Required experiments include changing PR gain and observing subsequent response;
setting one beam's power to zero and continuing to measure decay/relaxation;
and other compatible optical/driving interventions without discarding accepted
material state. These are future acceptance requirements, not claims that the
current GUI or validators support them.

Expected compatible interventions, subject to later material-specific validation
that state meaning/representation is preserved, include:

- beam power, including setting an individual beam to zero;
- beam phase, transverse center/position, and tilt/transverse wavevector;
- beam waist, profile, and focus parameters;
- adding, removing, replacing, or changing a per-beam image/screen and its launch
  parameters;
- PR gain and other driving parameters established to preserve the retained
  state's physical meaning.

These are intended optical/LaunchPlane intervention capabilities, not existing
continuation permissions or authorization to modify LaunchPlane. Future general
N-beam PR continuation should not inherently require identical channel count or
configuration unless the selected material model makes channel topology part
of its retained state. Adding/removing optical channels is distinct from changing
a screen on an existing channel and still requires explicit scientific review.

The following are **unclassified pending explicit scientific review**, not
universally allowed or forbidden: adding/removing channels; coherence-group
changes; wavelength changes; optical boundary/edge treatments; material
coefficients affecting normalization or constitutive meaning; applied bias;
timestep; integrator; backend; and precision changes. The governing question is:
**Does the retained material state remain a valid physical initial condition,
and what numerical/history reconstruction is required for the new segment?**

One beam off is not automatically equivalent to all illumination off. Current
[beam validation](../../src/lcprop/core/beams.py) allows nonnegative per-channel
power, while [launch construction](../../src/lcprop/optics/launch.py) requires
positive total beam power. PR also derives a peak-intensity reference from `A0`.
A future dark-relaxation limit needs an explicit normalization/launch contract;
this decision does not bypass the current constraints.

## Boundary provenance and scientific questions

Continuation provenance must identify the source accepted state/checkpoint,
its original request/model/grid identity, accepted time with units or
normalization, and cumulative and segment step/time identities. Preserve the
source record and separately record the new segment request and each intentional
change with old/new values, units, stable parameter or channel identity, and its
compatibility classification. Launch changes must identify stable beam/channel
identity across segments, not only positional indices. Include screen/image
addition, removal, or replacement, old/new source identity (including content
hash where applicable), and changed screen/launch parameters. Preserve enough
information to distinguish incident power from power after a passive screen.
Do not rewrite the original checkpoint/request to make an intervention appear
to be an uninterrupted unchanged experiment.

Illustrative future boundary record (not measured evidence or a schema):

```text
continued from accepted state at t=0.010 (normalized PR material time)
Beam 2 power: 1 mW -> 0 mW
gain-length product: 10 -> 4
Beam 1 screen: none -> image A (source content identity recorded)
```

Later scientific classification must resolve:

- Which gain, bias, optical, or material coefficients change only the subsequent
  driving, and which change the units, reference state, constitutive meaning,
  or physical constraints of saved `E`/`theta`?
- How are new optical inputs and normalization/reference intensities constructed
  after a power, phase, profile, or wavelength intervention? How is accepted
  physical state preserved if normalization changes?
- When are changes of model family, boundary conditions, channel/coherence
  configuration, backend/precision, integrator, or timestep compatible, and
  when is conversion, reinitialization, or rejection scientifically required?
- How are cumulative time and diagnostic histories represented across segments
  with different parameters? Current time checks assume completed steps times
  one timestep; they cannot simply be reused for variable-timestep segments.
- What accepted numerical history is required by a selected integrator, and how
  must an intervention invalidate/rebuild dependent caches or history without
  changing the retained physical state? The [PR numerics design](pr_second_order_numerics_design.md)
  distinguishes current one-step state from possible future multistep history.

These questions require later bounded scientific review and numerical
validation. They do not authorize an implementation in this task.

## Non-authorization and retained holds

No compatibility checks are loosened; no checkpoint schemas, equations,
defaults, continuation numerics, persistence, or GUI behavior change. Parameter
interventions and changed `A0` reconstruction are not implemented. This does
not authorize screens on arbitrary beams, generic gain, N-beam/Image-Amplification
redesign, LaunchPlane modifications, or scattering/fanning investigation.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**. PR scattering
equivalence remains **SCIENTIFIC HOLD / UNESTABLISHED**. The
[post-Stage-1 residual dispositions](../development/lcprop_human_acceptance_post_stage1_residuals_2026-09.md)
remain in force. Acceptance of this design principle is not authorization to
begin follow-on work.

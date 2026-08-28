# PR Transverse Time-Dependent Slurm Parity

**Date:** 2026-08-28

**Branch:** `feature/pr-second-order-static`

**Implementation base:** `b0ad0fbec26d2140d530880fd02d0640d50cd86b`

## Purpose

This record documents the local transport implementation required to make the
canonical `pr_transverse_timedependent` operation eligible for ordinary Slurm
execution and for Image Amplification composition over that remote operation.
The work adds serialization, registration, and local fake-remote validation.
It does not commission the operation on a cluster and does not alter PR
physics, integration, optical propagation, or image analysis.

## Codec Identity

The material-owned codec is implemented in
`lcprop.pr.transverse.timedependent_transport_codec`.

- Request codec: `pr.transverse_timedependent.request`, version 1.
- Result codec: `pr.transverse_timedependent.result`, version 1.
- Material/workflow key: `pr / pr_transverse_timedependent`.

The codec is type-specific to `PRTransverseRunRequest` and
`PRTransverseRunResult`. It does not reuse the reduced-TD or transverse-static
result codecs. It reuses only the established BeamStack encoder, portable
launch-element codec, and JSON/NPZ array-packing helpers.

Experiment persistence and remote-result transport remain distinct envelope
formats. The existing `.lcprop.json` experiment codec is unchanged.

## Request Contract

The request payload preserves:

- grid and BeamStack;
- material, transverse transport, dielectric, boundary, and optical projection
  profiles;
- material-time integrator, `Nt`, normalized timestep, and optical substeps;
- requested backend and precision;
- canonical scattering specification;
- ordered declarative channel launch elements;
- optional explicit `initial_A` and `initial_psi` arrays.

Decoding reconstructs the canonical dataclasses without supplying new defaults.
It validates array shapes and retains the existing prohibition against combining
an explicitly prepared `initial_A` with declarative launch elements.

## Result Contract

The result payload preserves every field currently owned by
`PRTransverseRunResult`:

- `A_initial` and `A_final`;
- `psi_initial` and `psi_final`;
- initial and final power;
- completed and requested material steps;
- normalized material time;
- grid, launch, backend, and resolved-profile provenance;
- completion/cancellation status;
- complete scientific diagnostics.

Array shapes, numeric kinds, paired initial/final dtypes, backend dtype
provenance, completed/requested steps, and normalized time are validated before
the canonical result is reconstructed.

Status is validated as a coherent state machine. A completed result must have
accepted every requested step and must not carry cancellation provenance. A
cancelled result must stop before the requested step count and must identify
one of the canonical workflow cancellation stages. These rules are enforced
on both encoding and decoding.

The canonical result does not contain a retained optical driving/source volume,
per-time optical history, checkpoint, or restart record. The transport does not
invent any of them. Remote continuation is therefore not supported by this
milestone.

## Status and Cancellation Semantics

Transverse TD is finite-time integration, not an equilibrium solve. Both
completed and cancelled result envelopes use `converged = null`.

- Completed: `scientific_status = completed` and termination reason
  `requested_material_steps_completed`.
- Cancelled: `scientific_status = cancelled` and termination reason
  `cancelled_at_accepted_material_boundary`.

Cancellation before the first accepted step and after an accepted step both
round-trip with the canonical last-accepted-state behavior. A cancellation
reason cannot be replaced by a convergence or residual reason.

## Backend and Device Provenance

The codec records the resolved backend from the canonical result's
`backend_summary`. NumPy and CuPy-style provenance are both preserved. The
shared Slurm result path remains responsible for combining the verified result
envelope with the scheduler-retrieved allocated-device provenance; no
transverse-TD-specific runner branch was added.

Local float32 validation preserved `complex64` optical fields, `float32`
material fields, and float32 backend provenance without promotion. This is not
a new claim of local CuPy execution.

## Slurm Registration and GUI Capability

The default transport registry and default operation composition now include
`pr / pr_transverse_timedependent` alongside LC static and the other three PR
operations. The shared GUI eligibility check therefore enables ordinary
transverse TD and Image Amplification over transverse TD when the configured
runner advertises the registered operation.

The selected operation is not replaced with reduced TD. Existing generic
resource-profile, partition, GPU, CPU, memory, wall-time, backend, source
deployment, status, retrieval, verification, reconstruction, and product
conversion paths are reused unchanged.

## Portable Launch Validation

A screened two-channel request was transported with an embedded deterministic
raster source. Validation preserved the source checksum, pixels, placement,
channel association, and element order. Local and fake-remote execution
produced identical initial fields. The unaffected channel remained identical
to an unscreened launch, while the selected channel changed, confirming reuse
of the shared exact-once launch path.

## Local and Fake-Remote Equivalence

Deterministic ordinary requests were executed directly and through the complete
in-process request encode/decode, operation execution, result encode/decode,
and product-conversion path. Canonical arrays, scalar state, nested diagnostics,
profile/backend provenance, RunData geometry, fields, and diagnostic collections
agreed exactly, including recursive diagnostic values.

The existing Image Amplification comparison was extended to transverse TD. It
preserved the composite request, declarative signal screen, explicit pump and
signal roles, base fields and status, carrier mask, isolated signal fields,
full-length back-propagated reconstruction, zero-response reconstruction,
gain, analytic gain, correlation, NRMSE, power drift, and specialized products.
The augmented local and fake-remote RunData also agreed in workflow status,
field identities, geometry and coordinates, field arrays and presentation
metadata, curves, and recursively compared scientific diagnostic values.
Independent wall-clock timing values were intentionally excluded from exact
comparison.
Post-processing remains local after retrieval of the ordinary canonical remote
result.

## Result Size

For the bounded `10 x 8 x 2`, two-channel float64 fixture, the uncompressed NPZ
array content was 7,696 bytes:

| Contribution | Bytes |
|---|---:|
| `A_initial` | 2,560 |
| `A_final` | 2,560 |
| `psi_initial` | 1,280 |
| `psi_final` | 1,280 |
| carrier-integral diagnostic vector | 16 |

The four canonical state arrays dominate the payload. Their retrieval is a
correctness requirement for ordinary products and Image Amplification. Sparse
or selective retrieval is a possible future optimization, not part of this
milestone. There is no checkpoint duplication.

## Failure Validation

Focused tests reject malformed request fields, missing result fields, invalid
result status, incompatible result shapes, dtype/backend inconsistencies, and
invalid accepted-step/time relationships with transport-specific errors. They
also reject completed results carrying cancellation provenance, completed
results that stop early, cancelled results that claim every requested step, and
cancelled results without a recognized workflow cancellation stage.
Existing shared transport tests continue to cover envelope codec-version
mismatch, checksums, manifests, request/result identity, backend compatibility,
and reconstruction/product-conversion failure handling.

## Scientific Non-Change

No transverse transport equation, electrostatic closure, time integrator,
optical kernel, tolerance, scattering physics, launch-screen transformation,
carrier isolation, back-propagation, gain, or fidelity calculation changed.
The production modifications are limited to a new material-owned transport
codec and default registration. All remaining changes are validation and this
engineering record.

## Commissioning Status

Local transport and fake-remote equivalence are complete. Real Slurm/H200
commissioning is explicitly deferred to a separate approval-gated task. No
cluster access or job submission occurred during this milestone.

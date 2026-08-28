# Reduced Time-Dependent PR Basic Slurm Parity

## Scope

This milestone adds portable finite-run transport for the canonical reduced
photorefractive time-dependent workflow, identified by
`pr / pr_timedependent`. It is transport, registration, and capability
plumbing only. The PR equations, material integrator, optical propagation,
launch-screen physics, convergence controls, and image-amplification analysis
are unchanged.

The codec identity is:

- request: `pr.timedependent.request`, version 1;
- result: `pr.timedependent.result`, version 1.

## Request Transport

The PR-owned codec reconstructs the canonical `PRRunRequest`. It transports:

- runtime grid and beam stack;
- material and normalized material-time solver options;
- requested backend and precision;
- ordered channel launch elements through the shared portable-launch codec;
- optional explicit `initial_A` and `initial_E` runtime arrays;
- optional canonical partition-independent scattering specification.

The existing conflict rule remains authoritative: declarative launch elements
cannot be combined with explicit `initial_A`. Request reconstruction remains
type-specific to PR; the portable-launch encoder is shared with experiment
persistence, but the transport and experiment envelopes remain distinct.

## Result and Checkpoint Transport

The result codec reconstructs the canonical `PRRunResult`. It preserves:

- `A_initial` and `A_final`;
- `E_initial` and `E_final`;
- the final accepted driving-intensity z-volume;
- initial/final normalized power;
- completed and requested material steps;
- normalized material time;
- grid, launch, status, diagnostics, backend, and device provenance;
- the complete canonical `PRTimeDependentCheckpoint`.

The codec validates declared grid/channel dimensions, array shapes and numeric
dtypes, finite scalar values, completed/requested-step consistency, status,
backend provenance, checkpoint validity, and exact agreement between duplicated
result/checkpoint state.

The completed checkpoint intentionally duplicates `A_initial`, `E_initial`,
and `E_final`. For the bounded 10 × 8 × 2, two-channel float64 validation case,
the unpacked NPZ arrays total 14,080 bytes. Checkpoint duplication accounts for
5,120 bytes; the other result arrays account for 8,960 bytes. This is accepted
as correctness payload for canonical continuation parity. Sparse or deduplicated
retrieval is deferred.

Completed checkpoint transport does not imply remote continuation dispatch is
supported.

Remote submission of continuation segments and recovery of partial state from
scheduler-cancelled jobs remain explicitly deferred.

## Cancellation

Cancellation is represented only at accepted material-step boundaries. Both
cancellation before the first accepted step and cancellation after an accepted
step encode the explicit transport termination reason
`cancelled_at_accepted_material_boundary`. The last accepted scientific arrays
and checkpoint remain authoritative; no scheduler partial-result recovery is
introduced.

Because this workflow integrates to a requested finite material time rather
than solving a convergence problem, the generic transport-envelope
`converged` field is `null` for both completed and cancelled runs. Status,
cancellation, and termination provenance remain explicit in their dedicated
fields.

## Execution Composition and GUI Capability

The default material-neutral Slurm composition now registers
`PR_TIMEDEPENDENT_OPERATION` with the dedicated codec. Existing LC static, PR
static, and PR transverse-static registrations are preserved. PR transverse TD
is not registered.

The PR GUI continues to determine Slurm eligibility from registered ordinary
operations. Consequently reduced TD is eligible without a workflow allowlist,
while transverse TD remains unavailable. Image Amplification over reduced TD
dispatches exactly one ordinary `pr_timedependent` operation and performs its
existing analysis locally after verified result reconstruction. Native in-job
TD progress is not streamed through Slurm; the existing scheduler/transport
lifecycle remains authoritative.

## Local Equivalence Validation

Bounded local fake-remote tests exercise:

1. request encoding and reconstruction;
2. canonical execution of the reconstructed request;
3. result encoding and reconstruction;
4. ordinary PR product regeneration.

They cover an ordinary coherent two-channel run, a declarative raster-screened
run, NumPy float32 dtype/provenance preservation, cancellation on both sides of
the first accepted boundary, checkpoint consistency, malformed payloads, and
operation registration. The Image Amplification composition test compares the
local and in-process transported paths for the prepared fields, carrier
isolation, reconstruction fields, gain, correlation, NRMSE, power drift, and
composite status.

## Deferred Work

- remote TD continuation dispatch;
- scheduler-cancelled checkpoint recovery;
- sparse or deduplicated result retrieval;
- native in-job material-time progress streaming;
- transverse-TD transport and GUI eligibility;
- cluster commissioning of this codec.

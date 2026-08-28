# Static PR Slurm Parity and Portable Launch Transport

## Purpose

This milestone completes the local transport foundation for the two canonical
static photorefractive workflows: `pr_static` and `pr_transverse_static`. It
also permits the existing Image Amplification composite experiment to select
either static workflow through Local or Slurm execution without introducing a
special remote Image Amplification operation.

This is execution and serialization plumbing only. It does not change PR
physics, static convergence, optical propagation, or image-analysis methods.

## Shared Portable Launch Encoding

`lcprop.pr.portable_launch` owns the reusable encoding of declarative PR launch
elements. Experiment persistence and both static transport codecs use the same
field encoder/decoder while retaining distinct experiment and remote-transport
envelopes.

The portable representation preserves:

- ordered channel associations and ordered raster screens;
- intensity-transmission interpretation and preprocessing policy;
- physical placement, size, exterior transmission, and boundary policy;
- inversion and alpha-handling policy;
- source identity, basename, dimensions, decoded mode, and encoded format;
- canonical grayscale samples and their SHA-256 checksum;
- embedded original bytes and SHA-256 identity for user images.

Absolute local source paths are not authoritative. Malformed Base64, invalid
array metadata, and checksum mismatches fail during decoding before scientific
execution.

## Static 1D Transport

`lcprop.pr.static_transport_codec` provides a type-specific codec for
`PRStaticRunRequest` and `PRStaticRunResult`. Requests retain the grid, beam
stack, material, precision-aware static solver policy, backend, launch
elements, and optional runtime initial arrays. Results retain the complete
canonical optical and material state, residual/source volumes, iteration and
slice records, power, convergence/status, replay diagnostics, tolerance
provenance, launch/grid summaries, and backend/device provenance.

Decoded result arrays are checked against the declared grid, channel count,
and accepted-slice count. Status and convergence flags must agree. The remote
representation is not a reduced result type.

## Static 2D Transport

The existing `pr_transverse_static` codec now uses the same portable launch
encoding. Empty launch plans retain their previous behavior; nonempty plans,
including screened image-bearing channels, now round-trip through the ordinary
canonical request. Transverse solver physics and result semantics are
unchanged.

## Default Slurm Composition

The material-neutral default transport composition now registers:

- `lc / lc_static`;
- `pr / pr_static`;
- `pr / pr_transverse_static`.

No time-dependent PR operation is registered by this milestone.

## GUI Capability Rules

The PR GUI determines Slurm eligibility from the ordinary operation actually
registered by the active Slurm runner. It no longer uses a hard-coded request
type as a proxy for remote support.

- `pr_static` and `pr_transverse_static` are accepted by the default Slurm
  composition.
- `pr_timedependent` and `pr_transverse_timedependent` remain rejected with the
  selected operation identified in the error.
- Image Amplification inspects its selected base workflow and is accepted only
  when that ordinary base operation is registered.

Execution target and scientific backend remain independent GUI controls.

## Composite Dispatch and Local Analysis

Image Amplification remains a local composite orchestrator:

1. prepare the selected ordinary static PR request;
2. run exactly that registered operation through the selected runner;
3. retrieve, verify, and reconstruct the complete canonical base result when
   the runner is remote;
4. perform carrier isolation, back-propagation, metric calculation, and product
   augmentation locally.

The GUI forwards the selected shared runner arguments, including the resource
profile, to the ordinary base dispatch. It does not create or register an
Image-Amplification-specific Slurm operation.

The existing post-processing contract is unchanged: it consumes `A_initial`,
`A_final`, grid/launch summaries, powers, status, and source/channel
provenance. Carrier masks, reconstructed and zero-response images, gain,
correlation, NRMSE, and power-drift calculations are unchanged.

## Status, Failure, and Cancellation

The existing remote lifecycle remains authoritative:

```text
Submitting -> Pending -> Running -> Retrieving -> Verifying
           -> Reconstructing -> Completed / Failed / Cancelled
```

Scheduler completion alone is not GUI-ready completion. Image analysis begins
only after the ordinary result has been verified and reconstructed. Existing
base cancellation/failure handling and the bounded local analysis cancellation
checks are preserved. No partial remote checkpoint recovery or native
in-job iteration streaming was added.

## Validation

Local validation covers:

- shared launch encoding through experiment and remote-transport envelopes;
- empty, single-channel, multi-channel, rectangular, offset, inverted, and
  embedded-user-image screen plans;
- `pr_static` request/result transport and product regeneration;
- malformed result-array rejection and cancelled accepted-boundary transport;
- screened `pr_transverse_static` request transport;
- default operation/codec discovery;
- GUI capability selection and resource-profile forwarding;
- in-process transport equivalence for ordinary static results;
- Image Amplification Local versus in-process-remote equivalence over both
  static base workflows;
- relevant persistence, transport, runner, GUI, failure, and cancellation
  regressions.

Cluster commissioning is intentionally deferred. No claim of Slurm-site or GPU
commissioning is made by this local milestone.

## Deferred Work

- time-dependent PR transport and GUI Slurm support;
- remote TD continuation;
- sparse/selective retrieval;
- per-z complex optical histories;
- result-transfer optimization;
- cluster commissioning of the new static transport paths.

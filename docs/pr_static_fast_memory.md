# Retention-aware reduced Static result construction

Execution retention is separate from `PRStaticRunRequest`. `run_pr_static`
accepts a keyword-only `result_policy` (default `full`). The registered reduced
Static operation opts into `WorkflowOperation.supports_result_policy`.
`LocalRunner` passes its operational `_result_policy` only to opted-in operations;
the headless executor supplies that value from the existing request envelope.
Other workflows, including TD, retain their existing execution behavior.
No Slurm deployment or scientific request schema changes are involved.

## Lifetime and scientific compatibility

The primary solve, independent replay and bounded completion diagnostics execute
unchanged for both policies. Only subsequent result construction branches.

Previously, both policies copied four complete real volumes to host before the
transport codec removed them for Fast. At 8192 x 4096 x 80 float64 this was 80 GiB.
Now Fast drops solver-owned references to the initial/final material, primary
source/residual and replay residual volumes after diagnostics, before endpoint
transfer. The replay source remains only until Fast cuts/preview are built and
is then released. No pool flush is used: dead allocations may remain reserved
in CuPy's pool, available for reuse; reservation is not a live array reference.
Other transverse solver temporaries live until function return. Caller-owned
initial conditions, if supplied, remain owned by the caller.

Fast copies only initial/final channel optical fields to host. Those fields
supply the unchanged input/output intensities and far-field calculation through
the existing product adapter; no optical longitudinal volume is constructed.
The existing backend cut extractor retains the samples nearest x=0 and y=0.
The existing preview helper transfers one selected replay plane at a time,
using the same z indices, contiguous block averages, float64 normalization,
float32 preview payload, block boundaries and coordinates as post-pruned Full.
The launch intensity normalization is calculated from the host endpoint using
the same NumPy helper as the original codec.

Construction uses completed replay data, including after cancellation. Zero
completed slices retain endpoints and metadata but no longitudinal preview/cuts,
matching the existing contract. Convergence, iteration history, bounded replay
diagnostics, scattering provenance and all scalar metadata are retained.

The four Fast-omitted fields are `None` from construction onward. The codec
reuses retained cuts/preview rather than requiring a full source volume. It
rejects labeling a Fast-constructed result as Full. Full still constructs and
retains all four real volumes and the same endpoint fields; it remains expensive.

## Incremental host-memory bound

Let P=Nx*Ny, Z=completed slices, C=beam-channel count, and c=complex item size.
Define E=2*C*P*c (both retained endpoints), L=8*Z*(Nx+Ny) (upper bound for both
cuts), V<=4 MiB (preview), F=8*P (one float64 source plane), and
T=8*min(Nx,96)*Ny (preview block-reduction workspace).

A conservative live-array construction allowance is
`3*E + 3*L + 2*V + 4*F + 2*T`, plus linear-size coordinates and retained
iteration/provenance records. This includes endpoint-normalization work,
cut-transfer copies and overlapping preview-plane temporaries. No term is
proportional to Z*Nx*Ny. Preview construction may transfer every selected plane
(80 planes here), but never assembles their full-resolution host volume.

At the native one-channel geometry, E=1 GiB and F=0.25 GiB; this construction
allowance is approximately 4.03 GiB, not an 80-GiB host-volume requirement.
It is an analytical live-array allowance, not a measured RSS or GPU peak.
Python/NumPy/CuPy allocator reservations and library workspace add overhead.
Subsequent codec packing copies only retained E/L/V arrays; product creation
uses endpoints and retained cuts/preview. These remain O(E+L+V+P), not a material
volume. FFT workspace is backend-dependent. This change does not reduce the
six device volumes required during solve/replay or change retrieval products.

## Validation and commissioning

Small-grid tests require exact encoded array/metadata equality against Full
followed by the old Fast projection, covering float32/float64, nonlinear and
linearized response, scattering off/V2 on, and cancellation at 0/1/2 slices.
They also compare endpoint/far-field and longitudinal product arrays.
Transfer guards reject any full material-volume export. Weak references verify
that five volume allocations die before endpoint conversion and the sixth dies
before return. An executor/package round trip verifies that envelope policy
reaches construction; non-opted-in operations receive no new run keyword.
Optional CuPy equivalence checks remain native commissioning requirements.

The known four-volume host construction blocker is removed for Fast. Retrying
the native job still requires reviewed/committed installation and separately
authorized H200 commissioning of both this path and bounded completion
reductions. No native memory/timing guarantee or submission is implied.

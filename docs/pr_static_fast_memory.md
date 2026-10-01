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

Fast/selected construction exports only requested endpoint fields. For retained
cuts and previews, a presentation-only helper reproduces the historical Full
projection reference: NumPy `abs(A)**2`, maximum per channel in the launch's
real dtype, then NumPy summation in channel order. It views an already-retained
host launch if available. When the request supplies an exact-execution-dtype
NumPy launch, acceptance computes and carries this scalar from owned launch
bytes: the existing owned NumPy copy on CPU, or bounded owned host snapshots
used for GPU upload. Each upload completes before its snapshot is released.
Later caller mutation cannot alter the accepted launch or presentation scalar.
No launch readback is needed for that case. Otherwise it transfers launch chunks of at most
65,536 complex elements. It never retains an additional endpoint plane.

This reference is intentionally separate from the scientific `peak_reference`
used throughout source evaluation and propagation. Native diagnostic job 4679468
at commit eb471cfd45bece3257a51fdedef3ca9326d22a08 found that NumPy presentation
recomputation was one binary64 ULP below the CuPy scientific reference despite
identical launch bytes. This alone produced the Fast/Full cut mismatch; accepted
states, replay and raw cuts were bitwise equal. The presentation recomputation
uses the accepted host launch when present; the GPU-generated-launch fallback
occurs after solve, replay and completion diagnostics. Neither path changes
scientific normalization.

Fast transfers raw nearest-zero cuts and applies the same shared NumPy
subtraction/multiplication helper as Full projection. Previews use the same
presentation reference and the unchanged selected-plane transfers, float64
normalization/block averaging, float32 storage, coordinates and metadata.

Construction uses completed replay data, including after cancellation. Zero
completed slices retain policy-selected products and metadata but no longitudinal preview/cuts,
matching the existing contract. Convergence, iteration history, bounded replay
diagnostics, scattering provenance and all scalar metadata are retained.

Presentation-reference readback is permitted for either a complete result or a
cancelled/nonconverged partial result with at least one accepted scientific
slice, only when requested retained products require the exact NumPy reference
and no accepted-host scalar is already available. It runs during product
construction, after scientific execution has stopped accepting slices and after
the existing replay/completion diagnostics. It cannot start another iteration
or replay, change accepted state, or change cancellation/convergence status.
Zero accepted slices and exceptions before result construction do not invoke
the fallback. An exact accepted-host scalar also suppresses the fallback.

All current Fast/Interactive and Analysis policies include longitudinal cuts
and previews; bare `analysis` disables only additional exact products. Full
construction does not need this reference and performs no presentation-reference
readback (its explicitly retained scientific arrays still transfer normally).
A later Full-to-Fast projection uses the already retained host launch. There is
no current selected-product policy that omits these baseline presentation products.

The four Fast-omitted fields are `None` from construction onward. The codec
reuses retained cuts/preview rather than requiring a full source volume. It
rejects labeling a Fast-constructed result as Full. Full still constructs and
retains all four real volumes and the same endpoint fields; it remains expensive.

## Incremental host-memory and transfer bound

For one-channel A5 (8192 x 4096, 80 intervals, complex128/float64), whose prepared
request has `initial_A=None`, ordinary
Fast can transfer 536,870,912 launch bytes (512 MiB) for a complete or retained
partial result, in 512 chunks of at most
1,048,576 bytes (1 MiB). These bytes were not previously transferred for this
reference calculation. Only one chunk is live; the NumPy absolute-value and
square temporaries are each at most 524,288 bytes. Conservative additional
live launch-reference storage is 2 MiB, plus one scalar per channel. If a host
launch was already explicitly requested, this helper uses views and adds zero
device-to-host launch traffic. Zero completed slices never invoke the GPU
readback fallback; an accepted-host scalar may already have been captured.

Raw x-z and y-z transfers are respectively 5,242,880 and 2,621,440 bytes: 7.5 MiB
combined, unchanged in byte count from transferring normalized cuts. Retained
cuts also occupy 7.5 MiB. Host normalization may simultaneously hold a raw cut,
a normalized temporary and its owned result (at most 15 MiB for the x-z cut),
then releases temporaries. No source/replay volume is transferred. Float32
launch/cut traffic is half these amounts.

The additional launch traffic is one streamed launch (512 MiB), not the former
two-endpoint 1-GiB export, and no full launch plane is retained solely for
normalization. These 512 MiB are neither retained package storage nor Mac/network
retrieval traffic, and are not a longitudinal-volume transfer. There is no added
term proportional to Nz*Nx*Ny. Existing
preview transfers/storage and selected-product costs remain unchanged. For
explicit complex-input/output selection, the user's requested endpoint storage
remains necessary; the existing host launch is reused by this helper.

These are transfer and live-array bounds, not native RSS/GPU measurements.
Allocator reservations and FFT workspace remain backend dependent. The six
scientific device volumes and their lifetime through replay are unchanged.

## Validation and commissioning

Small-grid tests require exact retained cuts, previews and their metadata against
Full followed by the old Fast projection, including codec round trips. Coverage
includes float32/float64, nonlinear and field-linear response, scattering off/V2
on, and cancellation at 0/1/2 slices.
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

## Presentation-reference remediation validation

The focused regression deliberately separates the scientific reference by one
binary64 ULP and requires byte-identical cuts, previews and codec round trips
against post-pruned Full. Chunk-bound guards cover multiple channels, both
precisions, strided launches and nonmutation. Original exact-equality assertions
remain unchanged. Native CuPy re-certification is still required after review;
this remediation does not authorize an H200 job or the matched A5 calculation.

Exact-dtype supplied host launches incur zero additional device-to-host traffic
for this reference. GPU acceptance uses the necessary owned device launch plus
at most one 1-MiB host upload snapshot and one bounded device upload temporary;
it does not retain a new host plane. It synchronizes each upload before reusing
host storage. Only the scalar survives for presentation. CPU acceptance reuses
its existing owned copy. Host inputs needing dtype conversion retain the prior
acceptance and bounded-readback fallback. Result/package schemas and network
payloads are unchanged. Native upload/stream-lifetime certification is pending.

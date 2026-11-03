# Memory-Bounded Canonical PR Transverse-Static Workflow

## Status

This document records the local production-memory remediation prompted by
Slurm job `2993245`. The change preserves the canonical global coupled-static
formulation and has not yet been recommissioned on an H200.

## Failure Evidence

Job `2993245` ran `pr_transverse_static` at Git SHA
`bfa9ab59cb93c7e89b0c30f8a1f72f959f41859b` with CuPy, float64, and a
`1024 x 1024 x 400` grid. It failed in
`potential_rhs() -> fft2(flux_y)` while requesting 6,710,886,400 additional
bytes after CuPy had allocated 144,804,209,152 bytes. Sampled device use peaked
near 138,670 MiB; host MaxRSS was only about 366 MiB.

The failing `flux_y` had exact shape `(400, 1024, 1024)` and dtype float64.
Its complex128 Fourier transform is 6.25 GiB. The failure was therefore a
whole-longitudinal-volume FFT batch, not a single-plane FFT, scheduler error,
transport error, or Image Amplification error.

## Recovered Implementations and Formulation Choice

Two bounded-memory implementations already existed:

- `lcprop.pr.static_streaming` is the reduced A7 slice-local static workflow.
- `lcprop.pr.transverse.marching_static` is the causal zero-flux transverse
  marcher.

Neither can replace the canonical global `pr_transverse_static` workflow
mechanically. Both advance a causal accepted material state with the optical
field, whereas the canonical workflow performs global refreshed-source outer
iterations and a global coupled line search. Promoting either implementation
would change the material/optical coupling formulation.

The selected architecture therefore preserves the canonical outer algorithm
and changes storage/execution only:

1. Accepted potential, source, and final residual volumes are retained as
   NumPy host arrays.
2. Each frozen-source material plane is transferred to the selected backend,
   solved by the unchanged `_solve_plane()` Newton/PCG implementation, and
   copied back immediately.
3. Projection, state construction, residual evaluation, derivative-null
   diagnostics, and optical response construction operate on one transverse
   plane at a time.
4. The optical field and split-step kernel remain backend-native and advance
   sequentially through z.
5. Canonical scattering phases are generated one plane at a time on the
   selected runtime backend, copied immediately into a host cache, and
   transferred back one plane at a time during propagation. This preserves
   V1's historical backend-specific realization and V2's cross-backend
   realization contract without a persistent full-z GPU cache. Scattering
   provenance continues to identify the backend that actually generated V1.

This is a host-backed global coupled solve, not the separate causal marching
physics implemented by `marching_static.py`.

## Old Memory Model

For `Nz=400`, one retained real volume and one Fourier-sized complex volume
have the following sizes.

| Grid and precision | Real `(Nz,Nx,Ny)` | Complex `(Nz,Nx,Ny)` |
|---|---:|---:|
| 1024² float64/complex128 | 3.125 GiB | 6.250 GiB |
| 1024² float32/complex64 | 1.563 GiB | 3.125 GiB |
| 2048² float64/complex128 | 12.500 GiB | 25.000 GiB |
| 2048² float32/complex64 | 6.250 GiB | 12.500 GiB |

The old CuPy path simultaneously retained or generated several volumes:

| Data or workspace | Old shape | Persistence | Required role | New handling |
|---|---|---|---|---|
| accepted `psi` | `(Nz,Nx,Ny)` real | coupled solve | final material product | host |
| initial `psi` | `(Nz,Nx,Ny)` real | whole run | result provenance | host |
| midpoint source | `(Nz,Nx,Ny)` real | coupled solve | refreshed coupling | host |
| equilibrium residual | `(Nz,Nx,Ny)` real | coupled solve | convergence/result | host; recomputed for replay |
| TD residual | `(Nz,Nx,Ny)` real | coupled solve | diagnostic/result | host; recomputed for replay |
| material proposal and two residuals | three real volumes | material solve | Newton result | host outputs; unused residual pair released before line search |
| line-search direction/trial/source/residuals | multiple real volumes | trial | coupled globalization | host; dead trials released |
| PR state (`P`, `E_x`, `E_y`, scratch) | multiple real volumes | operation | material/electrostatic evaluation | one backend plane |
| `flux_x`, `flux_y`, RHS scratch | multiple real volumes | RHS | transport residual | one backend plane |
| FFT inputs/outputs/workspace | complex volumes | FFT call/pool | derivatives and inverse elliptic solve | 2-D backend FFT only |
| optical fields | `(Nchannel,Nx,Ny)` complex | optical pass | propagation state | backend, unchanged |
| scattering cache | `(Nz,Nx,Ny)` real | run/stage | deterministic reuse | host, one-plane transfer |

The observed approximately 135 GiB peak is plausible from only ten real
volumes (31.25 GiB), several 6.25 GiB complex transforms, FFT workspace, and
CuPy's cached allocator retaining overlapping Newton/PCG and line-search
allocations.

## Retention Policy

The established canonical result contract retains the final z-resolved
potential, midpoint source, authoritative equilibrium residual, and TD
diagnostic residual. These are final static products used by the ordinary GUI,
transport codec, and diagnostics; they are not an iteration-history volume.
They remain available and are stored on the host. Intermediate material,
line-search, replay, and continuation-stage volumes are not retained.

`record_iteration_history` continues to control compact scalar Newton/outer
records. Image Amplification continues to consume `A_initial`, `A_final`, and
ordinary summaries and does not add material-volume retention. No GUI request
implicitly requests a GPU-resident history.

This milestone deliberately does not change the public result or transport
schema. A future minimal-result policy would be required before 2048² x 400
can be considered safe on a 64 GiB host; that is separate from the H200 OOM
fixed here.

## Replay and Cancellation

Independent replay remains mandatory. It repropagates the backend optical
field through host-retained accepted potential planes, compares the final
field and source, and then recomputes authoritative residuals plane by plane.
Accepted residual volumes are released before replay so old and replayed
residual pairs do not overlap unnecessarily.

Cancellation remains observable only at established accepted outer
boundaries. A material proposal or line-search trial is never returned as
physical state. Visibility continuation retains the direct result required
for fallback and only the current continuation result; it no longer retains a
list of every full-volume stage result. The retained direct result can still
overlap an executing continuation stage and must be included in host-memory
planning.

## Expected Memory Scaling

Backend memory no longer contains an `Nz` factor. Its persistent state is the
channel optical field, split-step kernel, and one material plane. Transient
Newton/PCG/FFT scratch is also plane-sized. Conservative commissioning
projections, including FFT-plan/workspace uncertainty and allocator headroom,
are:

| Grid and precision | Persistent GPU state | Per-plane GPU workspace and safety margin | Expected GPU peak | Host final/working storage |
|---|---:|---:|---:|---:|
| 1024² float64 | <0.15 GiB | 1.5-3.0 GiB | <4 GiB | approximately 25-45 GiB direct; roughly 50-61 GiB with direct-fallback/continuation overlap; 18.75 GiB retained with scattering |
| 1024² float32 | <0.08 GiB | 0.8-1.8 GiB | <2.5 GiB | approximately 13-24 GiB direct; roughly 25-32 GiB with continuation overlap; 9.38 GiB retained with scattering |
| 2048² float64 | <0.6 GiB | 6-12 GiB | <16 GiB | exceeds a 64 GiB host under the current full-result contract |
| 2048² float32 | <0.3 GiB | 3-7 GiB | <9 GiB | approximately 50-64 GiB; marginal on a 64 GiB host |

These are architecture projections, not measured CuPy peaks. H200
recommissioning must measure the actual pool peak. The original 1024²
float64 case should no longer approach 100 GiB of device memory.

## Numerical Equivalence

A deterministic 24² x 2-slice NumPy float64 workflow was executed once from
the unmodified source archive at the milestone base SHA and once from the
memory-bounded implementation. SHA-256 hashes were identical for:

- `A_initial` and `A_final`;
- `psi_initial` and `psi_final`;
- midpoint source;
- authoritative equilibrium residual;
- TD diagnostic residual.

Convergence status, two accepted coupled iterations, replay equality, powers,
and all four scalar residual metrics were also exactly identical. A separate
unit comparison proves that the host-volume material adapter is bit-for-bit
identical to the original batched material solver for a deterministic
multi-plane case. A structural regression rejects any 3-D input reaching the
canonical material/electrostatic backend operations.

The same archived-snapshot comparison was repeated with V1 and V2 canonical
scattering on NumPy. Every listed array hash, coupled-iteration decision,
termination status, replay check, power, and scalar residual metric was
bit-for-bit identical for both algorithms. Focused float32 tests compare the
streamed precision-native reductions with the archived full-volume reduction
policy at passing and failing convergence boundaries. A complete float32
workflow comparison remains part of H200 recommissioning because the
production NumPy reference intentionally rejects float32 static solves.

The equations, tolerances, operation order within each plane, optical
propagator, canonical scattering realization, and convergence/replay gates are
unchanged. The change is a storage and dispatch-boundary correction only.
Float32 convergence, line-search, and physical-validity reductions retain
float32 accumulation; float64 reductions retain float64 accumulation. Host
aggregation does not silently promote precision-sensitive gates to float64.

## Validation and Recommissioning Gate

Local validation covers the transverse static solver/workflow, structural
slice locality, products, transport round trips, Image Amplification, replay,
and reduced `pr_static` regressions. CuPy tests remain local skips when no CUDA
device is present.

The next gate after review and commit is one bounded H200 recommissioning run
of the failed 1024² x 400 float64 request (or a checksummed scientifically
equivalent fixture). It must record actual device peak memory, numerical
status, residual/replay/power diagnostics, result transport, and Image
Amplification completion. No production-size success claim is made before
that gate passes.

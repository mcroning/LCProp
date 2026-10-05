# Local transverse-grid support and planning

The original scalable NumPy 96-per-axis check was a commissioning/support
boundary, not a numerical-method or memory-safety requirement. The local
qualification on Apple arm64 (10 cores, 16 GiB RAM; NumPy 2.5.3, SciPy 1.18.1)
used committed `bb6bbd0` arithmetic, with only the metadata ceiling exempted in
the diagnostic harness. Product equations, solvers, iteration policies,
precision, tolerances and persistence identities did not change.

## Current envelopes

| Path | Support | Advice |
|---|---|---|
| Scalable NumPy float64 | Through 512 per axis | Up to 256 per axis recommended for ordinary interactive use; larger qualified requests warn |
| Scalable NumPy state32 v3 | Existing 96 per axis | The parity-only float32 checks do not qualify all larger-grid use |
| Scalable CuPy | Existing state32 256 / state64 512 per axis | H200 envelope unchanged |
| Connected direct/reference | 12,288 active nodes | Algorithmic direct/reference guard unchanged |
| Reduced independent columns | Existing support | Not subject to the genuine-2D envelope |
| Reduced/full-transverse TD | Existing support | No new grid ceiling; existing timestep/physical checks still apply |

The explicit unbiased 384×32 scalable bridge remains unchanged. Shapes beyond
the qualified ordinary envelope remain unsupported, rather than silently
selecting a different backend, precision or solver. In particular 1024² was
not qualified. These bounds do not promise convergence for every illumination,
closure or gain.

## Evidence

All float64 ladder grids (96², 128², 192², 256², 384², 512²) passed the existing
physical gates with both unbiased PCG and prescribed-current GMRES. The ladder
used two optical cells, gain-length 0.1 and the retained H200 weak nonuniform
launch, with identical physical dimensions at each grid. At 512² the cases
took about 1.05 / 4.80 seconds, with process RSS peaks 327 / 735 MiB.
Matched retained H200 outputs at 128²/256² state32 and 512² state64 passed the
existing cross-backend contracts. Float64 local qualification at 128²/256²
is independent of those state32 parity comparisons.

A representative collimated Gaussian Static workload (200×200 µm aperture,
50 µm beam widths, 1 mW, wavelength 0.633 µm, index 2.4, normalized material
wavenumber 0.1/µm, dark 0.01, zero uniform background, zero flux, scattering
off, L=1000 µm, dz=10 µm, gain-length=3, float64) completed all 100 cells in
4.05 seconds at 128² and 15.96 seconds at 256². The fresh GUI/headless adapter
constructed the launch. This measures the scientific workflow with bounded
endpoint/cut products; it excludes GUI rendering and full-volume retrieval.

Full-transverse TD at 128²/256² completed the unchanged two-step fixture in
0.042 / 0.213 seconds, using spectral IMEX Euler, dt=1e-4, two z intervals and
published optical-first coupling. No TD policy changed. A reduced explicit
Euler fixture on a finer grid still rejects a timestep exceeding its physical
stability bound; preserving grid spacing by enlarging the aperture accepts
large optical grids. That distinction is intentional.

Detailed requests, traces, physical limits, timings, hashes and charts live in
`results/Research/pr-local-large-grid-qualification-v1/`. Timings used one BLAS
thread and include diagnostic overhead. They are examples, not runtime promises.

## Memory-aware advice

`resource_plan` adds a local assessment only for NumPy scalable execution.
The assessment sums the existing scientific/product workspace scenarios and
presentation estimates. Above 256 per axis is classified `large/slow`; an
estimate at or above 60% of detected physical RAM is `memory-risk`. The GUI
logs runtime advice and requires explicit confirmation for memory risk.
Existing large retained-volume confirmation remains in force.

Physical capacity is best-effort; when unavailable, the plan says assessment
is incomplete. Current free memory never enters scientific request identity,
serialization or validation. Resource classification is deterministic given
shape, estimator and supplied physical capacity. Estimates exclude process,
library and allocator overhead and are not measured peak guarantees. Users
must account for other running applications. Unsupported solver/backend shapes
continue to fail explicit validation.

Inspect Request and the solver guidance distinguish local qualification,
interactive advice, the direct bound and unchanged H200 support. There is no
automatic grid/precision substitution or new scientific execution during planning.

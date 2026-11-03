# Biased Full-Transverse Linearized Reference Backend Generalization

## Scope and baseline

This milestone generalizes the isolated frozen-intensity reference operator
committed as `5556410` (`Add biased transverse linearized PR reference`) from
NumPy-only execution to the repository's NumPy/CuPy backend convention. It
does not register or call the operator from production PR workflows, Image
Amplification, soliton code, the GUI, persistence, transport codecs, or Slurm.

The accepted response remains

\[
\widehat{\delta\psi} =
-\frac{a_M+iE_{\rm app}k_x}
{I_0\left[a_M(1+a_H)+iE_{\rm app}k_xa_H\right]}
\widehat{\delta I}.
\]

No Fourier-response or nonlinear PR mathematics changed.

## Backend API and ownership

`solve_pr_biased_linearized_reference` and
`biased_linearized_fourier_symbol` accept the repository's
`BackendSpec`, or a backend name string. Omitting the argument preserves the
validated NumPy/float64 behavior. A string selects float64; callers select
float32 explicitly with `BackendSpec`.

The implementation resolves `BackendSpec` with `lcprop.core.backend.get_backend`
and uses its `xp`, real dtype, complex dtype, and FFT implementation. It does
not introduce a PR-specific backend layer.

All returned array fields belong to the resolved backend:

- NumPy execution returns `numpy.ndarray` values;
- CuPy execution returns `cupy.ndarray` values;
- `total_fields_from_perturbation` preserves that ownership;
- callers explicitly use the repository's `asnumpy` helper if host arrays are
  required.

For CuPy, a NumPy input incurs one host-to-device transfer at `xp.asarray`.
An already device-native input incurs none. Input validity is reduced on the
device and transfers one Boolean to the host; this is the sole deliberate
array synchronization in the solver. Kernel construction, the forward FFT,
spectral products, inverse FFTs, and field reconstruction remain on device.
There is no implicit device-to-host conversion at the result boundary.
Conversely, selecting NumPy with a CuPy input performs one explicit
device-to-host conversion through the repository's `asnumpy` helper at the
input boundary; native NumPy input incurs no transfer.

## Dtype and provenance

Both backends support these explicit pairs:

- `float64` / `complex128`;
- `float32` / `complex64`.

Input dtype must match the selected real precision; the solver does not
silently coerce mismatched physical input. Result fields and spectral arrays
are cast to the selected pair. NumPy FFT internals may use NumPy's normal
higher-precision FFT behavior for float32 input, but returned ownership and
dtypes follow `BackendSpec`. CuPy float32 arrays remain device-native and are
not promoted to float64 by the solver.

Results record requested and resolved backend, real and complex dtype, GPU
status, `cuda:<device-id>` when applicable, model/profile identifiers,
applied field, reference intensity, and grid/material parameters. Reading the
CUDA device id does not copy a computational array.

## Fourier and null-mode identity

Both paths call the existing `spectral_wavevectors` helper with their resolved
`xp`. Therefore they share FFT axes `(-2, -1)`, default FFT normalization,
anisotropy orientation, and the canonical zeroing of each even-grid
first-derivative Nyquist symbol.

The resolved mask is `a_H > 0`. The constant gauge and every even-grid joint
derivative-null Nyquist combination are excluded before division. NumPy uses
its `out`/`where` divide; CuPy uses a safe denominator followed by the same
mask because CuPy does not support that combined ufunc form. Odd/odd,
even/odd, odd/even, and even/even classifications have paired tests.

## Reconstruction and operation structure

`delta_I`, its per-plane mean removal, FFT, response product, potential,
fields, carrier perturbation, and mean-current perturbation all use `xp`.
Signs and definitions match the validated solver. A 3-D input has shape
`(Nbatch, Nx, Ny)`: the leading index denotes independent frozen transverse
planes. It is not a propagation coordinate and no longitudinal volume is
retained. Validation errors use the same independent-batch terminology.

Each 2-D solve or batched call retains the reference operation structure:

- one batched forward 2-D FFT;
- four batched inverse 2-D FFTs;
- pointwise kernel and reconstruction arithmetic;
- no Newton solve, line search, PCG, or iterative material solve.

## Memory behavior

Storage is `O(Nbatch * Nx * Ny)` and never scales with retained longitudinal
propagation history. For float64/complex128, a conservative live-array
planning estimate is about 200 bytes per transverse sample, excluding
backend-dependent FFT workspace. This gives approximately:

| Grid | Estimated live arrays | FFT workspace |
| --- | ---: | --- |
| 256 x 256 | 12.5 MiB | backend-dependent, not measured locally |
| 512 x 512 | 50 MiB | backend-dependent, not measured locally |
| 1024 x 1024 | 200 MiB | backend-dependent, not measured locally |

These are architectural estimates, not measured peak-device-memory claims.
Actual GPU peak memory remains an H200 commissioning measurement.

## Equivalence and validation boundary

The focused suite preserves all existing NumPy analytic tests and adds:

- explicit backend selection and unavailable-backend behavior;
- NumPy float32/complex64 resolved multi-mode agreement with float64, bias
  reversal, unbiased and qualified reduced limits, and even-grid null modes;
- conditional CuPy float32/complex64 dtype and equivalence coverage;
- result ownership and provenance;
- all four parity combinations for null-mode classification;
- every existing analytic single-mode case on CuPy;
- CuPy bias reversal, unbiased response, and qualified y-independent reduced
  limit;
- a deterministic nonzero-mean, x/y/oblique, biased multi-mode fixture;
- batched NumPy/CuPy equivalence, including independent mean current;
- the independent nonlinear Taylor oracle on CuPy;
- an isolation assertion that the reference solve is not exported by the PR
  production package.

The local environment has no importable CuPy installation and no available
CUDA device. NumPy tests pass normally; CUDA-dependent tests skip narrowly.
Consequently no local GPU timing, measured device memory, or actual CuPy
numerical-equivalence claim is made. The existing NumPy nonlinear oracle
continues to demonstrate second-order Taylor remainder. The CuPy oracle is
implemented but deferred to real-device commissioning.

Local results:

- focused backend/reference suite: 35 passed, 13 CUDA tests skipped;
- low-level transport and static/reduced regressions: 69 passed, 3 skipped;
- complete `test_pr*.py` suite: 511 passed, 70 skipped;
- import/syntax compilation and `git diff --check`: passed.

## Deferred work

The next post-commit lifecycle is a small H200 commissioning gate for this
isolated operator. It must measure NumPy/CuPy agreement for the kernel,
material fields, zero/Nyquist modes, float64 and optional float32, runtime,
and peak GPU memory. Production integration remains deferred until that gate
passes.

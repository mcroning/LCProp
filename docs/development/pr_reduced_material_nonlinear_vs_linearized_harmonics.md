# Reduced material-only nonlinear versus linearized harmonics

## Scope and fixture

This bounded test compares the canonical reduced x-only nonlinear material
equation with its existing reduced linearized operator. It uses no optical
propagation, self-consistency, GUI, scheduler, or time evolution.

The prescribed total normalized intensity is

```text
I(x) = I0 [1 + m sin(kx)],
I0 = 1,  k = pi rad/um,  L = 2 um.
```

The endpoint-excluded periodic grid has `N=512` and `dx=L/N`. The production
material normalization is retained through `dx_normalized=k_D*dx`, with the
canonical `k_D=4.282893587390008 /um`. Applied field and explicit background
intensity are zero. Since `m<=0.95`, the prescribed intensity remains strictly
positive. Both solvers receive the identical frozen intensity and grid.

The nonlinear field is the converged root of the production `hopping_rhs`,
obtained with the existing batched cyclic Newton solver. The comparison field
comes from `solve_pr_reduced_linearized_intensity` with reference intensity
`I0=1`. All nonlinear cases converged in three or four iterations, with maximum
final RMS residual `2.04e-13` and maximum absolute residual `8.32e-13`.

## Fourier convention

For harmonic `n=1,2,3`, the exact periodic FFT bin is

```text
c_n = (1/N) sum_j [E_j - mean(E)] exp(-i 2*pi*n*j/N),
A_n = 2 |c_n|.
```

Thus `A_n` is the real-field sinusoidal amplitude. No windowing, interpolation,
or nearest-bin selection is used. The phase difference is
`wrap(arg(c_1,NL)-arg(c_1,lin))` on `[-pi,pi)`.

## Results

| m | `A1_NL` | `A1_lin` | ratio | `A2/A1` NL | `A3/A1` NL | delta phase (rad) | relative L2 error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.00476918424 | 0.00476905413 | 1.00002728 | 0.00273834 | 0.00000894713 | `8.88e-16` | 0.00273856 |
| 0.02 | 0.00953914930 | 0.00953810825 | 1.00010915 | 0.00547710 | 0.00003579395 | `0` | 0.00547890 |
| 0.05 | 0.0238615554 | 0.0238452706 | 1.00068293 | 0.01370010 | 0.00022395059 | `-8.88e-16` | 0.01372828 |
| 0.10 | 0.0478213465 | 0.0476905413 | 1.00274279 | 0.02745302 | 0.00089922859 | `-4.44e-16` | 0.02767933 |
| 0.20 | 0.0964448343 | 0.0953810825 | 1.01115265 | 0.05533695 | 0.00365307259 | `4.44e-16` | 0.05717485 |
| 0.40 | 0.199888823 | 0.190762165 | 1.04784312 | 0.11441653 | 0.0156073988 | `-4.44e-16` | 0.13014096 |
| 0.60 | 0.321442662 | 0.286143248 | 1.12336274 | 0.18304045 | 0.0398891553 | `-1.78e-15` | 0.24420561 |
| 0.80 | 0.490210692 | 0.381524330 | 1.28487400 | 0.27566164 | 0.0901815246 | `2.22e-15` | 0.47137731 |
| 0.95 | 0.744621348 | 0.453060142 | 1.64353753 | 0.40366741 | 0.191685979 | `4.44e-16` | 0.99827495 |

The nonlinear equation strengthens the Bragg-matched fundamental monotonically
over this fixture. The excess is negligible at small modulation but reaches
64.35% at `m=0.95`. At that endpoint, the nonlinear second and third harmonics
are respectively 40.37% and 19.17% of the nonlinear fundamental (`A2=0.30058`,
`A3=0.14273`).

## Small-modulation behavior

Log-log fits over `m=0.01,0.02,0.05,0.1` give:

| quantity | fitted power of m |
| --- | ---: |
| `A1_NL/A1_lin - 1` | 2.00217 |
| RMS of `E_NL-E_lin` | 2.00434 |
| relative L2 error | 1.00434 |
| `A2_NL` | 2.00214 |
| `A3_NL` | 3.00315 |

Therefore the absolute material-field difference is `O(m^2)`, while its
relative L2 error is `O(m)` because the linearized response itself is `O(m)`.
The nonlinear harmonics scale empirically as `A2=O(m^2)` and `A3=O(m^3)`.
The fundamental ratio tends to one quadratically. The phase differences never
exceed `2.22e-15` rad, so there is no resolved fundamental phase shift.

## Clean-source provenance and limitations

The authoritative calculation was run once from an isolated clean checkout at
exact commit `f534f0c7dabcae963688b8561bbfeb01b04f3869`. The finalized harness was
supplied separately with SHA-256
`b07fc221902173b42d5c3b27bd6bf86749d8efb924dee0d4f799f10d3a30ee61`.
The source status was empty before and after execution. Manifest SHA-256 is
`9e99324343d46d276b86e79f812ed87bf2db7c8966602ea3aa97cf552fd49d2a`;
it binds the exact summary, retained profile/spectrum rows, and five figures.

The clean summary CSV and the provisional dirty-tree CSV both have SHA-256
`9a91a09cc5cdf0c9cd9474b1b643d6c6ffba6016113a9220a7733f2feee2a10c`.
All requested metrics and fitted exponents are bitwise identical. Therefore the
uncommitted Newton line-search acceptance change did not affect this fixture or
its conclusions. The standalone illustrated report is
`docs/development/pr_reduced_material_nonlinear_vs_linearized_harmonics_report.md`.

## Local validation

- Focused harmonic-sweep tests against the clean production checkout: 7 passed.
- Committed reduced-linearized and nonlinear-static regressions in the clean
  checkout: 26 passed, 3 skipped for unavailable external/CuPy evidence.
- Syntax compilation: passed.
- Canonical scientific/profile/spectrum hashes, artifact hashes, retained
  profile FFT coefficients, figure readability, and Markdown links: passed.
- Candidate whitespace inspection and `git diff --check`: passed.

These conclusions apply to this one-period, zero-bias, material-only fixture.
They do not establish optical gain, propagation behavior, or a universal
nonlinear enhancement ordering. No production physics was modified.

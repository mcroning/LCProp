# Reduced x-only linearized PR production integration

## Scope and status

This milestone adds `material_response="linearized"` to the existing reduced
static workflow (`pr_static`). The existing reduced nonlinear response remains
the default and is unchanged. Reduced-linearized execution and Image
Amplification presentation are **Experimental**. Time-dependent linearized PR,
H200/CuPy commissioning, and soliton work are deferred.

## Canonical production equation recovered

The source of truth is `hopping_rhs` in `lcprop.pr.evolution`, not a new paper
transcription. In normalized variables it implements

```text
E_t = E_app I_b - (E I - I_x)(1 + E_x) + I E_xx .
```

Here `I` is the complete transport intensity (normalized optical intensity plus
dark and optional uniform background), while
`I_b = dark_intensity + uniform_background_intensity` is the explicit source
background. The terms are:

- applied-field source: `E_app I_b`;
- intensity-gradient source: `+I_x` after expanding the product;
- nonlinear drift/product: `-E I (1 + E_x)`;
- nonlinear gradient product: `+I_x E_x`;
- diffusion: `+I E_xx`;
- periodic centered first and second x differences;
- static closure: set the implemented RHS to zero independently on each frozen
  optical z plane.

There is no separate hopping parameter in this normalized production equation.
The material characteristic wavenumber enters through
`dx_normalized = k_D dx`.

## Direct linearization

Write

```text
I = I0 + delta_I
E = E0 + delta_E .
```

The uniform equilibrium is

```text
E0 = E_app I_b / I0 .
```

Keeping first-order terms in the implemented equation gives

```text
0 = -E0 delta_I - I0 delta_E + D_x delta_I
    - E0 I0 D_x delta_E + I0 D_xx delta_E,
```

or

```text
(1 + E0 D_x - D_xx) delta_E
    = (D_x - E0) delta_I / I0 .
```

Discarded second-order terms include `delta_E delta_I`, products of
`delta_E_x` with `E0 delta_I + I0 delta_E - delta_I_x`, and
`delta_I delta_E_xx`. Both `I0` and `I_b` remain explicit: `I0` sets the
linearization state, while `I_b` and `E_app` set `E0`.

For the centered-difference symbols

```text
k1 = sin(q) / dx
k2_squared = 4 sin(q/2)^2 / dx^2,
```

the implemented response is

```text
delta_E_hat / delta_I_hat =
    (i k1 - E0) / [I0 (1 + k2_squared + i E0 k1)] .
```

The denominator has positive real part. The operator therefore uses one
one-dimensional FFT and inverse FFT along x for every independent y row. It is
the exact spectral solution of the centered-difference linearization, including
the even-grid Nyquist behavior of the canonical reduced discretization.

## Relationship to the full-transverse y-independent limit

The full-transverse reference uses continuum spectral derivatives and a fixed
harmonic mean-field ensemble. For a zero-mean, y-independent perturbation and
the accepted equilibrium matching condition `I0 = I_b`, its field response is

```text
(i k - E_app) / [I0 (1 + k^2 + i E_app k)] .
```

The reduced response approaches this expression as the centered differences
approach continuum derivatives. It is not bitwise identical on a finite grid:
`k1` replaces `k` and `k2_squared` replaces `k^2`. A 257-point resolved mode-3
fixture gives relative L2 error `5.130280e-4` and maximum absolute error
`1.930922e-8`.

For that direct fixture, `dx=2*pi/257` gives nondimensional mode phase
`q=0.073345`, continuum `k=3`, centered `k1=2.99731100998037`, and centered
`k2_squared=8.99596615331214` instead of `k^2=9`. Substitution into the two
transfer functions predicts relative field difference
`5.1302802158582e-4`, compared with the observed `5.130280e-4`. The input
contains only the resolved mode-3 pair (and its inactive mean), so no
unresolved or Nyquist contribution enters this comparison.

The closures also differ for a nonzero mean intensity perturbation under bias.
The reduced fixed-source equation changes its mean field by
`-E0 mean(delta_I)/I0` at first order. The full-transverse fixed-mean-field
ensemble holds the harmonic mean field fixed. No unconditional equivalence is
claimed.

## Production architecture

- `PRStaticRunRequest.material_response` reuses the existing typed
  `nonlinear | linearized` axis and is appended after all prior fields, so old
  positional construction remains compatible.
- `nonlinear` dispatch is the prior Newton/coupled path without equation or
  tolerance changes.
- `linearized` dispatch calls the isolated reduced operator for each frozen
  source plane and retains the established optical/source outer correction and
  line search.
- Cancellation remains at real outer safe boundaries. The direct operator has
  no fabricated internal iteration checkpoints.
- Results retain `E`, optical fields, source intensity, and the linearized
  residual. `material_response_summary` records the operator, `I0`, `I_b`,
  `E_app`, `E0`, backend-independent solve identity, call count, and
  Experimental status. Nonlinear Newton-solver provenance is marked not
  applicable rather than fabricated for a linearized result.

## Backend and precision behavior

The operator accepts NumPy and CuPy through `BackendSpec` and preserves
float64/complex128 or float32/complex64 result types. Local NumPy tests cover
both precisions and resolved multimode inputs. Conditional CuPy float64 and
float32 parity tests are present but skipped until a GPU is available.

Reduced-linearized float32 uses precision-specific coupled residual defaults of
`5e-6` RMS and `2e-5` maximum. These reflect the directly measured residual
floor of the float32 centered-difference/FFT solve. Float64 defaults, all
explicit overrides, and all nonlinear tolerances are unchanged.

## Reference and applied-field policy

`reference_intensity` is mandatory, finite, positive, persisted, and
transported. It is never inferred from the optical beam or `I_b`.
`I_b` continues to come from `PRMaterialSpec.background_intensity`. The reduced
applied field remains solely `PRMaterialSpec.applied_field`; the
full-transverse electrical boundary profile is not imported into the reduced
workflow. Comparisons to the full-transverse limit state and enforce `I0 = I_b`
where required.

## GUI, persistence, products, and Image Amplification

The static GUI now shows the material-response selector for both reduced and
full-transverse transport. Reduced-linearized selection shows `I0`, hides the
full-transverse mean-field control, and is labeled Experimental. Reduced TD and
full-transverse TD remain nonlinear-only.

Schema-v3 experiment persistence and the existing reduced-static transport
codec carry the same material-response object. Payloads without the additive
field decode as nonlinear. The result transport codec retains the material
response summary, and shared products expose it in the summary diagnostics.

The reduced static workflow already feeds the common Image Amplification
postprocessor. Its output contract is unchanged, so reduced-linearized output
uses that same postprocessor without IA mathematics changes. The reduced
static IA capability, including the new response, remains Experimental.

## Local scientific comparisons

All figures below use deterministic, identical optical/numerical inputs within
each comparison. Relative errors are L2 norms unless stated otherwise.

### Reduced nonlinear versus reduced linearized

| regime | material field | final optical field | output intensity | linearized power drift |
| --- | ---: | ---: | ---: | ---: |
| weak (`I_b=20`) | `1.150869e-4` | `3.416636e-6` | `1.679443e-8` | `-4.440892e-16` |
| stronger (`I_b=0.2`) | `7.023272e-2` | `9.368947e-4` | `1.063511e-5` | `-4.440892e-16` |

The weak case confirms the tangent regime. The stronger case differs by more
than five times the weak error in both material and optical fields, confirming
that the selector changes the physics.

### Matched y-independent reduced versus full transverse linearized

The production fixture sets `E_app=0`, `I0=I_b=100`, and has no y variation.
The finite-grid centered-difference material difference is `7.813737e-3`; the
final optical-field difference is `1.142263e-7`. This is the predicted
finite-grid discretization difference, not a failure of the matching
condition.

Quantitatively, its 48-point mode-2 grid has `dx=0.2`, mode phase
`q=0.261799`, continuum `k=1.30899693899575`, centered
`k1=1.2940952255126`, and centered `k2_squared=1.70370868554658` instead of
`k^2=1.71347298630024`. The transfer functions therefore predict
`7.81373645361322e-3`, compared with the observed `7.813737e-3`. The larger
difference relative to the direct fixture comes from this larger
nondimensional mode phase (`0.261799` rather than `0.073345`), not
amplification by the outer optical/material iteration. Its initial source has
only the resolved mode-2 pair plus an inactive zero mode, and the agreement of
the final material difference with the symbol prediction excludes a material
effect from unresolved or Nyquist content at the reported precision.

Holding the normalized domain and mode fixed while refining the production
symbol comparison gives:

| `Nx` | predicted relative field difference |
| ---: | ---: |
| 48 | `7.813736e-3` |
| 96 | `1.953948e-3` |
| 192 | `4.885189e-4` |
| 384 | `1.221317e-4` |

Each doubling reduces the difference by approximately a factor of four, the
expected second-order convergence of the centered first and second
differences toward their continuum spectral symbols.

### Genuine two-dimensional reduced versus full transverse linearized

With finite y modulation and otherwise identical inputs:

| quantity | relative L2 difference |
| --- | ---: |
| material `E_x` | `3.387980e-2` |
| final optical field | `4.183779e-6` |
| output intensity | `5.063296e-8` |
| complex far field | `4.183779e-6` |

This measures the effect of allowing y carrier/current transport for this
fixture only; it is not a claim that transverse coupling always increases or
decreases a diagnostic.

### Three-way interpretation

For the same weak two-dimensional fixture:

| comparison | interpretation | material-field difference | optical-field difference |
| --- | --- | ---: | ---: |
| A reduced nonlinear vs B reduced linearized | linearization effect | `9.177934e-3` | `1.102235e-6` |
| B reduced linearized vs C full transverse linearized | physical transverse plus derivative-discretization effects | `3.387980e-2` | `4.183779e-6` |
| A reduced nonlinear vs C full transverse linearized | combined effect | `2.524714e-2` | `3.116251e-6` |

The B-versus-C value contains both physical reduced-versus-full transverse
model differences and centered-difference-versus-continuum-spectral
discretization differences. It cannot be assigned solely to physical
transverse transport. No full-transverse nonlinear calculation is part of
this milestone.

## Run-cost and validation status

The local run-cost guard classifies reduced-linearized static work separately
as a one-dimensional fixed-count FFT response per coupled pass. Its work score
is lower than reduced nonlinear and is not confused with full-transverse
nonlinear work. Local scientific coverage passes, but the production status
remains Experimental pending later isolated CuPy/H200 commissioning.

## Deferred work

- actual CuPy/H200 commissioning;
- reduced or full-transverse linearized time dependence;
- soliton existence or stability;
- large full-transverse nonlinear comparisons;
- new carrier geometry diagnostics.

## Local validation

Executed with the repository Python 3.12 environment and
`QT_QPA_PLATFORM=offscreen` where GUI tests were present:

- focused reduced-linearized production tests: `18 passed, 2 skipped` (the
  conditional CuPy float64/float32 cases);
- affected reduced nonlinear, full-transverse linearized, GUI/IA,
  persistence/transport, products, and run-cost/cancellation suites:
  `281 passed, 18 skipped`;
- complete `tests/test_pr_*.py` suite: `566 passed, 72 skipped`;
- syntax compilation of every changed Python source/test: passed;
- `git diff --check`: passed.

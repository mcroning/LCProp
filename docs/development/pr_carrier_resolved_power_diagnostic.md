# Carrier-resolved PR optical-power diagnostic

## Scope and status

This milestone adds a presentation-only diagnostic for two sufficiently
separated, mutually coherent PR input carriers. It applies to ordinary PR
RunData, including Fast results, and does not change material equations,
optical propagation, source/coherence physics, or Image Amplification (IA)
mathematics.

## Why channel power is not the coupling diagnostic

The canonical arrays retain one propagated field per input channel. Because
all channels experience the same lossless optical propagator, each lineage can
retain its own norm even while interference redistributes power between
angular components of the coherently combined physical field. Consequently,
raw per-channel norms do not measure two-beam energy transfer.

Carrier-resolved Fourier-space power is the quantitative two-beam coupling
diagnostic. Peak intensity or apparent brightness in independently autoscaled
output-plane images is not a quantitative measure of energy transfer:
autoscaling, beam-shape changes, focusing or defocusing, and spatial
redistribution can change a displayed peak without a corresponding one-to-one
change in carrier power.

For two channels in one coherence group, the diagnostic instead forms

```text
A_group = A_1 + A_2
```

at both input and output and partitions `fft2(A_group)` using the two original
input carrier directions. Output masks are never recentered on output peaks.
Exactly two channels in one shared coherence group are supported. Two channels
in different groups, missing carrier metadata, or other channel counts produce
an explicit `not_applicable` diagnostic rather than being silently combined.

## Carrier geometry and tie policy

The carrier centers are the launch phase gradients
`(tilt_x_rad_per_um, tilt_y_rad_per_um)`, matching the launch phase
`exp(i*(kx*x + ky*y))`. Fourier coordinates use the repository convention

```text
kx = 2*pi*fftfreq(Nx, dx_um)
ky = 2*pi*fftfreq(Ny, dy_um).
```

For centers `k_1` and `k_2`, a Fourier pixel belongs to carrier 1 when
`|k-k_1|^2 < |k-k_2|^2` and carrier 2 when the reverse inequality holds. This
is the perpendicular-bisector Voronoi partition and works without special
cases for x, y, diagonal, or oblique separation. Pixels exactly on the
bisector receive half weight in each region. Therefore the two weights sum to
one at every pixel and no spectral power is dropped or double counted.

The shared helper also supplies IA's established x-carrier mask. IA retains
its prior deterministic rule that bisector ties belong to the signal carrier;
its gain definition and reconstruction path are unchanged. On a controlled
two-mode fixture with no power on the tie boundary, IA isolation and the new
generic signal-region power agree to numerical precision.

## Fourier normalization and reported quantities

NumPy's unnormalized forward FFT gives the carrier-region integral

```text
P_c = dx_um*dy_um/(Nx*Ny) * sum_k w_c(k) * |fft2(A_group)(k)|^2.
```

Thus `P_1+P_2` equals the spatial integral of `|A_group|^2` by Parseval. The
diagnostic reports:

- input and output normalized optical integrals per carrier;
- `gain = P_out/P_in` for separated carriers with nonzero input power;
- `delta_power = P_out-P_in`;
- carrier sums and their drift;
- coherent-group spatial powers and the partition balance error;
- physical mW values when the launch summary supplies total physical power.

Normalized optical integrals and mW values are labeled separately. Physical
values use the production launch convention in which one normalized optical
integral corresponds to the incident total launch power in mW.

## Separation quality

The diagnostic evaluates each input channel's isolated spectrum against the
bisector and records the fraction lying on the other carrier's side. Its
quality is

```text
quality = 1 - max(wrong_side_fraction_1, wrong_side_fraction_2).
```

Quality of at least `0.9` is required to report numerical gains. Below that
threshold the status is `insufficient_separation`, a warning is emitted, and
gain values are omitted while the partition powers remain available. This
prevents nearly coincident or strongly overlapping carriers from being
presented as precise gain measurements.

Gain also requires the input carrier power to exceed a scale-aware stability
floor relative to the coherent group's total input power. The relative floor
is the larger of `1e-12` and 64 machine epsilons for the input field's real
precision (`1e-12` for float64 and approximately `7.63e-6` for float32). A
carrier at or below that floor retains its input/output power and delta, but
its gain is unavailable with an explicit
`insufficient_input_carrier_power` status. This avoids unstable ratios for
absent or numerically negligible inputs.

## Products, GUI, and Fast compatibility

The compact `carrier_power` RunData diagnostic contains the carrier centers,
partition method, tie policy, separation quality, input/output power, gain,
delta power, Parseval errors, and power-balance error. The GUI Diagnostics tab
renders a compact table:

```text
Carrier | Input Power (normalized) | Output Power (normalized) | Gain | Delta Power (normalized)
Pump    | ...                      | ...                       | ...  | ...
Signal  | ...                      | ...                       | ...  | ...
```

Carrier geometry and names are retained as compact launch-summary metadata.
The calculation requires only `A_initial`, `A_final`, grid spacing, coherence
groups, carrier centers, and total physical launch power. Full and Fast
transport therefore produce identical diagnostics without adding any 3-D
arrays to Fast retrieval.

## Deterministic validation

A synthetic Fourier fixture transfers normalized power from `(0.75, 0.25)`
to `(0.5, 0.5)`. The recovered gains are `(2/3, 2)`, deltas are
`(-0.25, +0.25)`, and the power-balance error is below `2e-15`. Float64 and
float32 tests verify that the carrier sum matches coherent spatial power at
precision-appropriate tolerances. Geometry tests cover x, y, diagonal, and
arbitrary oblique carrier pairs, including equal half-weight on the bisector.

The deterministic production fixture uses identical two-beam launch geometry
for all three locally supported comparison models. Its carrier-center
separation quality is `0.9999728769`:

| Model | Pump input | Pump output | Pump gain | Signal input | Signal output | Signal gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reduced nonlinear | `0.7499835021` | `0.7500863368` | `1.0001371160` | `0.2500167392` | `0.2499139045` | `0.9995886887` |
| Reduced linearized | `0.7499835021` | `0.7500850823` | `1.0001354433` | `0.2500167392` | `0.2499151590` | `0.9995937063` |
| Full-transverse linearized | `0.7499835021` | `0.7500511120` | `1.0000901485` | `0.2500167392` | `0.2499491293` | `0.9997295784` |

All three runs converge. Their maximum carrier-balance error is
`5.83e-16`. Raw channel-lineage powers remain conserved while the carrier
regions change, directly demonstrating the distinction the diagnostic is
intended to capture. A versus B contains the linearization effect; B versus C
contains both physical reduced-versus-full transverse differences and the
known centered-difference-versus-spectral discretization difference; A versus
C contains both effects.

The gain ordering in this weak fixture is fixture-specific and is not a
universal ordering of the three models. It does not contradict stronger GUI
experiments in which reduced nonlinear, reduced linearized, full-transverse
linearized, or full-transverse nonlinear responses may exhibit a different
coupling ordering.

## Limitations

- Only an exactly two-channel, single-coherence-group analysis is implemented.
- The bisector is a complete half-plane partition, not a finite-band aperture;
  strongly overlapping spectra are rejected by the quality gate.
- Carrier assignment is anchored to input launch directions and does not track
  shifted or newly generated output peaks.
- Physical mW conversion is unavailable when legacy result metadata lacks the
  total launch power.
- No full-transverse nonlinear calculation or cluster execution is part of
  this milestone.

## Local validation

Executed with the repository Python 3.12 environment and
`QT_QPA_PLATFORM=offscreen` for GUI coverage:

- focused carrier diagnostic: `17 passed`;
- affected IA, reduced/full linearized, products, GUI, and transport suites:
  `202 passed, 2 skipped`;
- complete `tests/test_pr_*.py` suite: `583 passed, 72 skipped`;
- syntax compilation of every changed Python source and test: passed;
- `git diff --check`, including supplemental checks for new untracked files:
  passed.

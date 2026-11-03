# PR Figure A5 Fanning-Ring and Optical-Substepping Study Preparation

**Date:** 2026-08-08

**Branch inspected:** `feature/pr-second-order-static`

**Git SHA inspected:** `7a0ccf1f77fd39574f7db034144b2a7c9b969b81`

**Status:** Preparation complete; launch remains blocked by unresolved Figure
A5 inputs. The common-scattering implementation is locally validated but still
awaits pre-commit review.

## Objective

Prepare, without launching, a three-case study of the Photonics (2025) Figure
A5 longitudinal-step artifact:

- A: 50 µm material spacing and one optical substep;
- B: 50 µm material spacing and 25 optical substeps;
- C: 2 µm material spacing and one optical substep.

The engineering question is whether B agrees with C while avoiding most of
C's PR material solves. The scientific reference is Figure A5; exact
reproduction and the controlled LCProp A/B/C comparison are distinct claims.

## Reference Audit

| Reference | Evidence |
|---|---|
| Paper PDF | `reference/prprop/photonics-12-00113-v3/photonics-12-00113-v3.pdf`; SHA-256 `72307a380993c257be5e13d6c269670f203be8d26311755a05ffdc434b683f1d` |
| Trusted implementation | `reference/prprop/prprop3d.py`; SHA-256 `0adb5c824459ebaa31adbf9cad541c9ef634ac63f2b83bf42e8c3ee5a62ed28f` |
| Official supplement | The archive lists inputs for Figures 1, 2c, 2d, 3/4/6, 9, 11, 13, 14a, 14b, and 16; it contains no Figure A5 input file. |

### Recovered Figure A5 Parameters

| Parameter | Value | Provenance | Status |
|---|---:|---|---|
| Wavelength | 0.633 µm | Figure A5 caption | Resolved |
| Beam waist | 200 µm | Figure A5 caption | Resolved |
| Interaction length | 4,000 µm | Figure A5 caption | Resolved |
| Aperture | 1,000 × 1,000 µm | Figure A5 caption | Resolved |
| Transverse grid | 8,192 × 4,096 | Figure A5 caption | Resolved |
| Case A `dz` | 50 µm | Figure A5 caption | Resolved |
| Case C `dz` | 2 µm | Figure A5 caption | Resolved |
| Case A color maximum | 4 normalized power per unit solid angle | Figure A5 caption | Presentation resolved |
| Case C color maximum | 6 normalized power per unit solid angle | Figure A5 caption | Presentation resolved |
| Case C fanning efficiency | 81% | Figure A5 caption | Reference result, not an input |
| Gain-length product | unknown | Not stated in Figure A5 or an A5 supplement | Blocking |
| Beam count and ratio | unknown | Not stated in Figure A5 or an A5 supplement | Blocking |
| Incidence angle and azimuth | unknown | Required by Equations (A12)-(A13), but not stated for A5 | Blocking |
| Beam longitudinal focus/crossing convention | unknown | Not serialized for A5 | Blocking |
| Noise type, strength, correlation, and seeds | unknown | Fanning requires a seed; A5 request is unavailable | Blocking |
| Dark/background intensity | unknown | Not stated for A5 | Blocking |
| Applied field | unknown | Not stated for A5 | Blocking |
| Refractive index and hopping-model constants | unknown | Not stated for A5 | Blocking |
| Tukey parameter | unknown | Not stated for A5 | Blocking |
| Static or transient material solve | unknown | Not stated in Figure A5 caption | Blocking |

The paper's Table 2 and official validated static-fanning inputs for Figure 9
use gain 10, one beam (`rat=0`), dark intensity 0.01, external incidence angle
0.3 rad, azimuth zero, volume-noise strength 0.02, correlation length 0.4 µm,
zero applied field, refractive index 2.4, relative permittivity 2500, mobile
charge density 6.4×10²² m⁻³, temperature 293 K, and Tukey parameter 0.2.
Those values are credible **surrogate fanning parameters**, not recovered
Figure A5 parameters. They require explicit scientific approval and must remain
labeled as surrogates if used.

## Current Optical-Substepping Semantics

The production path is

```text
run_pr_static_streaming()
  -> _run_streaming_pass()
  -> production_advance()
  -> advance_pr_slice_with_midpoint_source()
  -> half_step_response_from_E()
  -> advance_prepared_response()
```

For a nominal interval `dz_material` and `Nsub`:

1. The diffraction kernel is built for `dz_material/Nsub`.
2. `half_step_response_from_E()` builds a phase screen for half of that one
   optical substep.
3. `advance_prepared_response()` repeats symmetric
   response-half/diffraction/response-half ordering `Nsub` times.
4. One candidate PR state `E` is frozen during all substeps in a trial optical
   pass.
5. PR-driving intensities are evaluated before and after the entire nominal
   interval and averaged. They are not recomputed after individual optical
   substeps.
6. During coupled static convergence, every material candidate causes a fresh
   complete optical trial. Consequently, executed FFT hops equal `Nsub` times
   the number of trial optical passes, not simply `Nz*Nsub`.
7. After acceptance, one volume-noise phase screen is applied at the end of the
   nominal interval. Noise is not divided over optical substeps.

Case B therefore implements one accepted material state per 50 µm slice and 25
Strang optical hops through that state. It does not perform 25 material solves
or 25 source refreshes per accepted slice.

The named legacy-linearized and full-nonlinear Lie reference modes do not use
the production optical-substep path. A controlled A/B/C test must use the same
production nonlinear Strang mode for all three cases. Such a test can be
compared qualitatively with the paper, but it is not the same numerical method
as the published legacy calculation.

## Scattering-Reproducibility Resolution

LCProp's legacy mode draws one deterministic phase screen per material slice, with
amplitude proportional to `sqrt(epsilon/Nz)`, and selects it using either an
explicit per-slice seed or `SeedSequence(base_seed, z_index)`.

Using the same base seed for 80-slice and 2,000-slice calculations does not
produce the same physical three-dimensional scattering field. Case B also
applies noise only once per 50 µm interval, even though it performs 25 optical
substeps. Therefore the present public request cannot satisfy the requirement
that A, B, and C use an identical scattering realization independent of the
material partition.

The development milestone now provides `PRCanonicalScatteringSpec`. It models
the trusted `epsilon/Ns` screens as increments of a white-in-z random phase
measure on deterministic canonical physical-z slabs. Each 50 µm interval is
the sum of the corresponding twenty-five 2 µm increments. The generator is
coordinate-addressed, bounded-memory, backend-native, and carries compact
configuration and canonical-seed checksums.

The reduced readiness calculation found a coarse/fine integrated-phase
relative L2 difference of `2.1314e-16` and identical configuration and seed
checksums. This resolves the technical scattering blocker subject to
pre-commit review. Scattering remains applied after each complete nominal
material interval; the coarse and fine runs therefore share disorder but
retain the intended difference in longitudinal screen placement.

Compact numerical results and complete generator provenance are preserved in
[`metrics.json`](assets/pr_partition_independent_scattering_readiness_2026-08-08/metrics.json).

## Ring Prediction

For internal beam angle `theta_in`, refractive index `n`, wavelength `lambda`,
step `dz`, and harmonic `j`, the paper gives

```text
f_jx = cos(theta_in) * sqrt(n*j/(lambda*dz) - (j/dz)^2)
f_jy =                 sqrt(n*j/(lambda*dz) - (j/dz)^2)
```

in cycles/µm. Direction-cosine semiaxes are `lambda*f_jx` and
`lambda*f_jy`; angular spatial-frequency semiaxes are `2*pi*f_jx` and
`2*pi*f_jy` in rad/µm.

The angle and index are unresolved for Figure A5, so an immutable prediction
table cannot yet be produced. For scale only—not as a manifest value—using the
validated Figure 9 surrogate `n=2.4` gives these `f_jy` values for `dz=50 µm`:

| `j` | `f_jy` (cycles/µm) | `lambda*f_jy` | `2*pi*f_jy` (rad/µm) |
|---:|---:|---:|---:|
| 1 | 0.274644 | 0.173850 | 1.725640 |
| 2 | 0.387374 | 0.245208 | 2.433944 |
| 3 | 0.473168 | 0.299515 | 2.973004 |
| 4 | 0.544901 | 0.344923 | 3.423716 |
| 5 | 0.607575 | 0.384595 | 3.817504 |

The corresponding x semiaxes require multiplication by
`cos(theta_in)`. These values must be regenerated from the approved manifest,
not copied into final analysis.

## Resource Estimate

The transverse plane contains 33,554,432 samples. One plane occupies:

| Array type | Size |
|---|---:|
| float32 | 128 MiB |
| float64 | 256 MiB |
| complex64 | 256 MiB |
| complex128 | 512 MiB |

A retained float64 `E` volume would require 20 GiB for A/B and 500 GiB for C.
It must not be retained. The bounded-memory streaming path keeps one material
slice plus backend scratch, independent of `Nz`.

| Case | Material slices | Accepted optical hops | Expected GPU memory | Preliminary runtime | Scratch | Permanent archive |
|---|---:|---:|---:|---:|---:|---:|
| A | 80 | 80 | 25-40 GiB | 10-60 min | 8-15 GiB | <500 MiB |
| B | 80 | 2,000 | 25-40 GiB | 20-120 min | 8-15 GiB | <500 MiB |
| C | 2,000 | 2,000 | 25-40 GiB | 4-24 h | 8-15 GiB | <500 MiB |

These are planning bounds, not measured predictions. They account for
deterministic replay but cannot determine the production solver's coupled-pass
count at the unresolved A5 parameters. A same-aspect-ratio reduced GPU pilot is
required before freezing wall times. An 80 GB A100 is recommended for the
pilot and full study until a measured memory high-water mark establishes that
a 40 GB device is sufficient.

The accepted-hop counts understate actual FFT work because every coupled
candidate and backtracking trial replays a nominal optical interval. The final
report must use instrumented executed counts and synchronized timings.

## Diagnostics and Presentation Contract

Before submission, freeze one checksummed analysis specification containing:

- unit-integral far-field normalization and conversion to normalized power per
  unit solid angle;
- linear color maxima 4 for the A paper panel and 6 for the C paper panel,
  plus a common comparison scale declared independently of results;
- a fixed logarithmic floor and common physical/direction-cosine axes;
- fixed near-field crop and full far-field field of view;
- a predeclared carrier-exclusion mask matching the trusted fanning-efficiency
  definition;
- complex phase-alignment convention for B/C field error;
- centroid, RMS-width, radial-profile, and ring-contrast definitions;
- Equation (A12)-(A13) overlays computed without fitting;
- optical/material/diagnostic/I/O timing boundaries.

Existing LCProp validation scripts contain reusable analysis methods, but the
streaming result does not itself provide all requested fanning diagnostics or
separate timing categories. A research payload must collect them without
modifying the production solver.

## Candidate Case Matrix

This matrix is a preparation artifact, not an immutable launch manifest:

| Field | A | B | C |
|---|---:|---:|---:|
| `Nx`, `Ny` | 8192, 4096 | 8192, 4096 | 8192, 4096 |
| aperture | 1000 × 1000 µm | same | same |
| length | 4000 µm | same | same |
| wavelength | 0.633 µm | same | same |
| waist | 200 µm | same | same |
| `dz_material` | 50 µm | 50 µm | 2 µm |
| `Nsub` | 1 | 25 | 1 |
| production mode | nonlinear Strang | nonlinear Strang | nonlinear Strang |
| material slices | 80 | 80 | 2000 |
| nominal accepted optical hops | 80 | 2000 | 2000 |
| all remaining physical fields | unresolved | identical to A | identical to A |
| controlled scattering | unresolved | identical physical realization | identical physical realization |

No scheduler script or purported immutable manifest was created because the
required physical and scattering fields remain unresolved.

## Local Validation

The focused split-step and streaming-static tests were run from the exact
checkout with the environment's native Python and the checkout source path:

```text
PYTHONPATH=src /Users/mcroning/miniforge3/envs/lcprop/bin/python3.12 \
  -m pytest -q tests/test_splitstep.py tests/test_pr_static_streaming.py
```

Result: **38 passed, 1 skipped in 0.66 s**. The skipped test is an optional
CuPy path unavailable in this local CPU execution. No production source or test
file was changed.

## Readiness Recommendation

**Classification: Blocked before launch-manifest preparation.**

The substepping semantics support the intended Case B interpretation for the
production workflow, but the study is not ready for submission. Approval is
premature until:

1. the user approves either an authoritative A5 request or explicitly labeled
   surrogate fanning parameters;
2. the partition-independent scattering milestone passes pre-commit review;
3. a reduced GPU pilot measures production coupled-pass counts, memory, and
   timing;
4. quantitative B/C tolerances and the fanning mask are frozen;
5. concrete, checksummed A/B/C manifests are presented.

This stopping point preserves the central question. Launching now with only a
shared base seed would confound material spacing with a different scattering
field and could make any apparent ring suppression uninterpretable.

---

End of preparation record.

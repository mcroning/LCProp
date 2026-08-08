# Partition-Independent Photorefractive Scattering Design

**Date:** 2026-08-08

**Branch:** `feature/pr-second-order-static`

**Implementation status:** Locally validated; awaiting pre-commit review

## Purpose

This note records the physical interpretation, numerical design, and local
validation of LCProp's partition-independent photorefractive volume-scattering
representation. The immediate motivation is the Figure A5 comparison between
50 µm and 2 µm material grids, but the representation is PR-owned and is not
specific to that benchmark.

## Legacy and Reference Semantics

Trusted PRProp3D generates a normally distributed transverse phase array for
every longitudinal propagation step, filters it with a two-dimensional
Gaussian, and multiplies it by `sqrt(sigma_x*sigma_y)`. Before filtering, its
standard deviation is

```text
sqrt(epsilon/Ns * 4*pi),
```

where `Ns=L/dz` is the number of scattering screens. The resulting phase is
applied as `exp(i*phi)` after diffraction, the PR material update, the PR
optical response, and the Tukey window. There is no implemented longitudinal
filter: `sigmaz` is calculated in the trusted source but unused.

LCProp's legacy streaming mode preserves these choices. It uses either an
explicit seed per material slice or derives a seed from the base seed and
slice index. The phase scale contains `sqrt(epsilon/Nz)`, and the screen is
applied once after the complete material slice. It is not distributed across
optical substeps.

The normalized quantity is therefore the accumulated phase variance: each
independent screen contributes approximately `epsilon/Ns` after the trusted
transverse normalization, so the complete crystal contributes approximately
`epsilon`. The approximation reflects finite-grid and Gaussian-filter boundary
effects.

Equal legacy base seeds do not identify equal disorder on different z grids.
Both the seed address and the per-screen normalization change with `Nz`.

## Physical Interpretation

The source model specifies independent phase increments and no longitudinal
correlation length. It therefore does not uniquely define a smooth random
field sampled at points in z. The least-assumptive continuum interpretation is
a white-in-z random phase measure: the random object associated with an
interval is its integrated phase increment.

For controlled grid comparisons, “the same scattering realization” means
that a coarse interval receives the sum of the same fine physical-z increments
contained in that interval. It does not mean that coarse and fine simulations
apply their screens at identical z positions. Screen placement remains tied to
the material partition and is part of the discretization being studied.

## Alternatives Considered

### Coordinate-hashed point samples

Generating an unrelated field from each floating-point z coordinate would be
access-order independent, but coarse and fine intervals would not share an
integrated realization. It would reproduce the original confounding under a
different seed policy and was rejected.

### Smooth or interpolated three-dimensional field

A smooth field would require a physical longitudinal correlation length and a
choice between point sampling and interval averaging. The paper and trusted
implementation provide neither; inventing them would change the model. This
option was deferred until supported by a separate physical specification.

### Dense canonical three-dimensional volume

A stored 8192×4096×2000 float64 volume would require approximately 500 GiB.
It offers no scientific benefit over deterministic procedural generation and
was rejected.

### Canonical physical-z phase increments

This design preserves the white-in-z statistics, supports exact physical-z
addressing, composes fine increments into coarse intervals, and requires only
one transverse accumulator. It was selected.

## Implemented Representation

`PRCanonicalScatteringSpec` declares:

- total accumulated phase-variance parameter `epsilon`;
- transverse Gaussian correlation length;
- explicit realization seed;
- canonical physical-z slab spacing.

Canonical slab `i` occupies the half-open interval

```text
[i*canonical_dz_um, (i+1)*canonical_dz_um).
```

Its seed is derived directly as

```text
SeedSequence([realization_seed, LCPR_namespace_tag, i]).
```

No sequential RNG state is retained. A material interval must be an integer
union of canonical slabs. For each included slab, the generator adds a
backend-native Gaussian array with unfiltered variance

```text
epsilon * canonical_dz_um/L * 4*pi.
```

The accumulated raw interval field is passed once through the trusted
two-dimensional Gaussian filter and multiplied by
`sqrt(sigma_x*sigma_y)`. Filtering after summation is mathematically
equivalent to summing filtered increments and reduces filtering cost for a
coarse interval. Floating-point summation and filtering order can produce
roundoff-level, non-bitwise differences.

Memory remains `O(Nx*Ny)` regardless of crystal length or canonical slab
count. NumPy and CuPy generate arrays on their selected backends. The exact RNG
and filter library versions are part of provenance because NumPy and CuPy are
not asserted to produce identical random arrays from the same scalar seed.

## Configuration and Compatibility

The new representation is selected explicitly with
`PRStreamingStaticOptions.partition_independent_scattering`. It is mutually
exclusive with the existing `volume_noise_epsilon`,
`volume_noise_correlation_um`, `volume_noise_seed`, and
`volume_noise_seeds` controls.

The legacy controls, defaults, numerical formula, RNG selection, filtering,
and application order remain unchanged. Existing requests therefore continue
to reproduce legacy results unless they explicitly select the new dataclass.

The new screen remains after the complete nominal material slice. Optical
substep semantics are unchanged.

## Provenance Contract

Each result records:

- scattering mode and algorithm version;
- `epsilon`, correlation length, realization seed, and canonical spacing;
- physical z domain, transverse shape, and aperture;
- coordinate, interval-sampling, longitudinal-model, and normalization text;
- RNG backend and version;
- NumPy `SeedSequence` version;
- Gaussian-filter library and version;
- real dtype;
- canonical slab count, first and last derived seeds;
- SHA-256 of the complete little-endian uint32 canonical seed sequence;
- SHA-256 of the normalized configuration record.

This compact record regenerates the realization without retaining a 3-D
volume. Run manifests must additionally preserve the exact Git SHA and package
environment.

## Local Validation

Focused tests cover deterministic repeatability, access-order independence,
coarse/fine interval composition, white-in-z statistics, interval-variance
scaling, transverse spectral filtering, alignment validation, mutual
exclusion, legacy-formula preservation, provenance, production-workflow
replay, and an optional CuPy path.

The readiness calculation used a 32×16 grid over 64×32×20 µm, canonical
spacing 2 µm, and coarse spacing 10 µm with five optical substeps. Results:

| Quantity | Value |
|---|---:|
| Coarse/fine provenance equal | true |
| Canonical seed checksum | `76ed280e7fade2ddda907f1a50170b6b1843f3fff64fca34e0ebc4fe09a14262` |
| Integrated-phase relative L2 difference | `2.1314e-16` |
| Integrated-phase maximum difference | `1.6653e-16` |
| Coarse deterministic replay | passed |
| Fine deterministic replay | passed |
| Coarse/fine output-field relative L2 difference | `0.0229751` |

The output fields are not expected to coincide: the material spacing and
screen-application positions differ. The invariant is the underlying
integrated scattering realization.

## Limitations

- The model remains white in z. It does not represent a finite physical
  longitudinal correlation length.
- Material intervals and the complete z domain must align with the canonical
  spacing.
- Coarse screens are applied at coarse interval boundaries rather than at each
  canonical slab boundary.
- Backend RNG implementations are versioned in provenance but CPU/GPU random
  arrays are not promised to be identical.
- The implementation resolves the scattering-control requirement for Figure
  A5 only after review. Missing Figure A5 publication parameters remain a
  separate blocker.

---

End of design record.

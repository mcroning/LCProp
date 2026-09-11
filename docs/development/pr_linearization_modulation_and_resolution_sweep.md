# Headless PR linearization and grating-resolution sweep

## Scope and motivation

This milestone adds a deterministic, non-GUI sweep for separating three
questions that are otherwise confounded in independently autoscaled output
images:

1. whether the reduced nonlinear frozen-intensity material response approaches
   its linearization as modulation depth tends to zero;
2. whether the 2 um interference grating and its nonlinear harmonics are
   adequately represented at 256, 512, and 1024 samples across 200 um; and
3. how reduced x-only and full-transverse linearized material responses differ
   for an x-directed grating and an exact 45-degree grating.

Carrier-resolved Fourier power, not apparent image brightness or peak
intensity under independent autoscaling, is used for production optical
comparisons. No production equation, propagation kernel, material operator,
source/coherence rule, or carrier-power definition changes in this milestone.

## Physical reference experiment

The production subset is anchored to:

- 200 um by 200 um aperture;
- 1000 um interaction length, `Nz=100`, and 10 um optical steps;
- dark intensity `0.01`, no uniform background, and zero normalized applied
  field;
- gain-length product `3`, refractive index `2.4`;
- two mutually coherent beams totaling 2 mW at 0.633 um;
- 20 maximum coupled iterations and one optical substep per slice;
- NumPy float64.

The realistic subset uses 20 um circular Gaussian waists. The strict
linearization test is instead a prescribed broad-field intensity so Gaussian
envelopes and changing overlap do not become hidden perturbations. Every
retained material refinement uses the literal square grid `Nx = Ny = N`; the
y-independent x control is not evaluated on a collapsed surrogate grid.

The user's pending external full-transverse nonlinear job was not inspected,
altered, cancelled, or used. No cluster access or scheduler action occurred.
All retained calculations were regenerated from an isolated clean Git clone at
production commit `5cbdbeb149b5b2d6960d5fe7572fdf29195f97a3`. The finalized
sweep harness was supplied separately rather than copied from the production
checkout.

## Exact geometry

Both geometries have `|Delta k| = pi rad/um` and period 2 um.

### X-directed control

```text
beam 1 k = (-pi/2, 0) rad/um
beam 2 k = (+pi/2, 0) rad/um
Delta k  = (pi, 0) rad/um
```

For the production launch, paraxial center back-projection to a crossing at
half length gives symmetric input centers `(32.96875, 0)` um and
`(-32.96875, 0)` um.

### Exact 45-degree grating

```text
beam 1 k = (-pi/(2*sqrt(2)), +pi/(2*sqrt(2))) rad/um
beam 2 k = (+pi/(2*sqrt(2)), -pi/(2*sqrt(2))) rad/um
Delta k  = (pi/sqrt(2), -pi/sqrt(2)) rad/um
```

The symmetric input centers are `(23.3124266922, -23.3124266922)` um and its
negative. The carriers and beam centers remain inside the periodic aperture
through the full nominal crossing.

The x grating is exactly periodic on the 200 um aperture: its fundamental mode
is `(100, 0)`. The exact 45-degree components are
`(+/-100/sqrt(2))`, so they are deliberately off the periodic FFT bins by
about `0.2893` mode in both axes. The harness records this noncommensurability.
Material harmonic extraction for this case uses windowed demodulation, and
orientation trends are not interpreted as pure transport effects without
allowing for the off-grid boundary/discretization contribution.

## Fixed-power visibility construction

For visibility `m`, total power `T=2 mW` is fixed and powers are

```text
P1 = T/2 * (1 + sqrt(1-m^2))
P2 = T/2 * (1 - sqrt(1-m^2)).
```

This gives `2*sqrt(P1*P2)/(P1+P2) = m` without changing total launch power.
For `m=(0.5, 0.25, 0.125, 0.0625, 0.03125)`, the pump powers are respectively
`(1.8660254, 1.9682458, 1.9921567, 1.9980450, 1.9995116)` mW and the signal
powers are their complements to exactly 2 mW. No launch elements are used, so
attenuation does not change with `m`.

The frozen-intensity fixture samples

```text
I = I0 * (1 + m*(cos(K.r) - discrete_mean(cos(K.r))))
I0 = 1.01.
```

The tiny mean subtraction is zero for the commensurate x case and guarantees
the exact same discrete mean `I0` for the off-grid 45-degree case. Tests verify
the mean and fixed total power directly. Both nonlinear and linearized solves
receive the same `I0`. The applied field is zero, so the biased reduced/full
qualification involving `I0=I_b` is inactive here; no biased reduced/full
equivalence is claimed.

## Resolution gate

`underresolved` means fewer than two points per period, `marginal` means at
least two but fewer than four, and `resolved` means at least four.

| N | dx (um) | points/K | K | points/2K | 2K | points/3K | 3K |
| ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| 256 | 0.78125 | 2.56 | marginal | 1.28 | underresolved | 0.8533 | underresolved |
| 512 | 0.390625 | 5.12 | resolved | 2.56 | marginal | 1.7067 | underresolved |
| 1024 | 0.1953125 | 10.24 | resolved | 5.12 | resolved | 3.4133 | marginal |

An underresolved harmonic is recorded as unavailable rather than interpreted
from an alias. Thus 256 is not adequate for nonlinear-harmonic conclusions.
At 512 the fundamental is resolved but the second harmonic is only marginal
and the third is above Nyquist. A 512 nonlinear carrier gain therefore cannot
be declared grid-converged from sampling arguments alone.

## Frozen-intensity asymptotic result

The compact material sweep contains reduced nonlinear, reduced linearized,
and full-transverse linearized rows at every `(geometry, N, m)`, for 90 rows
total. Nonlinear-minus-linearized order is fitted from absolute material RMS;
the relative error is also retained but is not used for the order because the
linear response itself is proportional to `m`.

| Geometry | N | nonlinear-minus-linearized order | linear-response order | error/m^2 at m=0.03125 | error/m^2 at m=0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| x | 256 | 2.063345 | 1.000000 | 0.059033 | 0.071944 |
| x | 512 | 2.082206 | 1.000000 | 0.093927 | 0.121349 |
| x | 1024 | 2.084887 | 1.000000 | 0.093122 | 0.121224 |
| 45 deg | 256 | 2.073526 | 1.000000 | 0.078286 | 0.098468 |
| 45 deg | 512 | 2.074822 | 1.000000 | 0.091348 | 0.115325 |
| 45 deg | 1024 | 2.076045 | 1.000000 | 0.092251 | 0.116883 |

The small-`m` ratios approach grid-dependent constants and every fitted order
is close to two. The linearized response order is unity to displayed
precision. This closes the normalization gate for the prescribed material
fixture: nonlinear-minus-linearized material error is quadratic, first-order
response is linear, `I0` is shared, sampled mean intensity is fixed, launch
power is fixed, and no launch attenuation varies with `m`.

## Fundamental and harmonics

At `m=0.5`, the reduced nonlinear output is:

| Geometry | N | material error RMS | `|Ehat(K)|` | phase(K) | `|2K|/|K|` | `|3K|/|K|` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x | 256 | 0.0179861 | 0.0386220 | pi/2 | unavailable | unavailable |
| x | 512 | 0.0303373 | 0.1029673 | pi/2 | 0.186811 | unavailable |
| x | 1024 | 0.0303059 | 0.1220816 | pi/2 | 0.156255 | 0.030723 |
| 45 deg | 256 | 0.0246169 | 0.0655053 | pi/2 | unavailable | unavailable |
| 45 deg | 512 | 0.0288313 | 0.0978669 | pi/2 | 0.189752 | unavailable |
| 45 deg | 1024 | 0.0292207 | 0.1068489 | pi/2 | 0.175997 | 0.033686 |

The x material-error RMS changes by only about 0.1 percent from 512 to 1024,
but its fundamental magnitude changes by about 18.6 percent and its resolved
second-harmonic ratio changes by about 16.4 percent. The 45-degree error RMS
changes by about 1.35 percent while its fundamental changes by about 9.2
percent. Consequently scalar material error looks substantially more settled
than the spectrum that controls Bragg coupling. The newly available 3K values
at 1024 are still classified marginal and are not treated as converged.

Reduced/full linearized x differences fall from `2.3121` at 256 to `0.24895`
at 512 and `0.05407` at 1024, consistent with the known coarse centered-
difference versus continuum-spectral error. For the 45-degree case the values
are `0.39161`, `0.08007`, and `0.15388`; the nonmonotonic behavior prevents a
pure y-transport interpretation and is consistent with physical orientation
sensitivity mixed with the exact carrier's off-grid periodic representation.

## Production optical subsets and carrier gains

All production rows use the complete 1000 um reference length and `Nz=100`.
Carrier separation quality exceeds `0.9975`, every material/optical solve
converges, carrier balance errors are below `5e-16`, and normalized optical
power drift is below `3e-14`.

The realistic 20 um Gaussian, equal-power (`m=1`) N=256 results are:

| Geometry | Model | pump gain | signal gain | optical rel L2 vs linearized |
| --- | --- | ---: | ---: | ---: |
| x | reduced nonlinear | 0.642924 | 1.357076 | 0.201318 |
| x | reduced linearized | 0.886380 | 1.113620 | reference |
| 45 deg | reduced nonlinear | 0.444943 | 1.555057 | 0.258820 |
| 45 deg | reduced linearized | 0.810432 | 1.189568 | reference |

A three-point broad-waist production subset at N=256 records the complete
pump/signal normalized input and output powers, normalized power deltas, gains,
final optical and intensity errors, and final-slice material harmonics. The
schema names these fields explicitly with a `_normalized` suffix; the separate
physical launch total is `total_power_mW`. Signal gains are:

| Geometry | m | reduced nonlinear | reduced linearized |
| --- | ---: | ---: | ---: |
| x | 0.5 | 5.489107 | 3.949315 |
| x | 0.125 | 5.528158 | 4.936360 |
| x | 0.03125 | 5.546691 | 5.019878 |
| 45 deg | 0.5 | 10.790229 | 7.512436 |
| 45 deg | 0.125 | 12.139527 | 9.861713 |
| 45 deg | 0.03125 | 4.469103 | 4.212057 |

These N=256 production trends are measurements, not convergence claims. The
45-degree broad-waist sequence is particularly contaminated by the off-grid
periodic carrier and the grid's inability to represent 2K and 3K. The strict
quadratic oracle is the prescribed frozen-intensity fixture, not the
self-consistent propagation sequence in which envelopes, overlap, diffraction,
and accumulated optical feedback also vary.

No production carrier-gain calculation was run at 512 or 1024. Therefore this
milestone does not establish whether the user's large interactive 512 gain is
grid-converged. The harmonic gate instead gives a concrete negative result:
512 is insufficient to certify nonlinear convergence because 2K is marginal
and 3K is underresolved. A retained-endpoint 512/1024 production comparison is
deferred to an explicitly authorized larger local or H200 sweep.

## Harness and compact output policy

The reusable command-line harness is
`scripts/checks/pr_linearization_resolution_sweep.py`. It supports:

- configurable grid, visibility, and geometry lists;
- material-only or explicitly selected production study;
- optional full-transverse linearized material/production rows;
- broad or realistic Gaussian production fixtures;
- deterministic compact CSV/JSON and optional headless plots;
- no Slurm submission code and no cluster-specific path.

No 3-D material array is saved. The retained evidence is under
`results/pr_linearization_resolution_sweep_2026-09-10/`:

- `summary.csv` and `summary.json`: 90-row material sweep and manifest;
- material error, `error/m^2`, harmonic-resolution, and fundamental-phase
  plots;
- `production_broad_n256/`: 12 production rows and carrier-gain plot;
- `production_gaussian_x_n256/` and
  `production_gaussian_45deg_n256/`: realistic two-row comparisons.

Every JSON manifest records the clean production source commit
`5cbdbeb149b5b2d6960d5fe7572fdf29195f97a3`, confirms an empty production
source status, records Python/NumPy versions, and binds the separately supplied
harness SHA-256
`9109aeb8952dd1e444961268628d3ccc7eddfb22938514083eab9d72f77e00fa`.
Each manifest checksums its CSV and PNG artifacts and records a canonical
SHA-256 of its own scientific rows and summary. The committed JSON file is
additionally content-addressed by Git.

## Clean-source provenance comparison

Before replacement, the original package was preserved outside the repository
and compared row-for-row with the clean-source regeneration. Its four JSON
SHA-256 values were:

- material: `402167ed066527a2d5dc01d15c44c2c7ada2827b81b3b8ffdee0b5e3e293c737`;
- broad production: `560f520f13a4f1d652028be7568c876e38637451d36d0991591322d0f5eb5a7d`;
- Gaussian x production: `1da91f13e52e5abff5eef7e6401e78574779e2ae361583b808d42649200ce0c6`;
- Gaussian 45-degree production:
  `12e62f08ff8882e87cb31bc2c7dbc0f5d7cdfcc92304bff177b6192f7b3dec28`.

After mapping the six formerly ambiguous carrier-power names to their new
explicit `_normalized` names and excluding nondeterministic runtime, all 90
controlled material rows are exactly identical. The Gaussian 45-degree rows
are also exactly identical. Broad-production and Gaussian-x rows differ only
at floating-point roundoff: the largest substantive absolute difference is
`1.53655e-13` in `material_max_abs_vs_linearized`, corresponding to
`2.50388e-14` relative; the largest carrier-gain difference is
`1.77636e-15`; and the largest harmonic-magnitude difference is
`5.55112e-17`. Differences in near-zero power-balance/drift residuals remain
below `4.44090e-16` absolute. All seven PNG files are byte-identical.

Thus the dirty production changes present during the superseded execution,
including the unstaged static-solver maximum-residual acceptance guard, did
not affect any reported scientific conclusion. They did affect the exact
execution provenance and produced negligible last-bit changes in a subset of
production scalars, which is why the clean regeneration replaces that package.

## Limitations and deferred work

- The full-transverse nonlinear model is not required and was not run.
- Full-transverse linearized rows in the 256/512/1024 sweep are frozen-
  intensity material comparisons; no large full-transverse production sweep
  was attempted locally.
- The reduced x-only nonlinear model cannot diagnose physical y transport.
- Exact 45-degree carriers are not commensurate with the 200 um periodic FFT
  domain; windowed harmonic estimates and grid trends retain that limitation.
- Harmonics below the sampling gate are omitted rather than treated as data.
- Carrier gain is endpoint-only and does not imply a longitudinally resolved
  transfer history.
- The current results do not certify 512-grid nonlinear carrier-gain
  convergence.

## Local validation

Validation used the repository Python 3.12 environment:

- focused sweep tests against the isolated clean production source: `18 passed`;
- combined sweep, carrier-power, reduced-linearized, and full-transverse-
  linearized regressions against that clean source: `98 passed, 15 skipped`;
- complete committed PR suite in the clean production checkout:
  `570 passed, 70 skipped`;
- complete current `tests/test_pr_*.py` suite: `611 passed, 72 skipped`;
- bounded material sweep: 90 rows completed;
- realistic N=256 production subsets: four rows completed;
- broad-waist N=256 production subset: 12 rows completed;
- optional full-transverse-linearized production route: bounded three-row
  smoke completed against the isolated clean source;
- syntax compilation, JSON parsing, artifact consistency, and
  `git diff --check`: passed.

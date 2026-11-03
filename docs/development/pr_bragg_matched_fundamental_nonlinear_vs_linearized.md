# Bragg-matched PR fundamental: nonlinear versus linearized

## Scientific question and scope

This bounded follow-up asks whether the nonlinear material response at the
Bragg-matched grating wavevector is larger, smaller, or phase-shifted relative
to its linearization. It analyzes the compact evidence committed by the 02_30
headless sweep at exact commit
`1e5fb7b18961a8536d2cfa3904cc8db4d401c24f`; no material or optical
calculation was rerun.

The paired comparison is reduced x-only nonlinear versus reduced x-only
linearized for the same geometry, grid, visibility, and fixture. The analyzed
quantity is the reduced active material field `E`, which is the field supplied
to optical propagation. The retained full-transverse linearized rows contain
`delta_E_x`. Under the current profile, `g_x=1` and `g_y=0`, so this is exactly
the perturbation of the projected active field
`E_active = g_x E_x + g_y E_y`. They are not used as a surrogate for a
full-transverse nonlinear coefficient because no matching full-transverse
nonlinear sweep was run.

## Fourier convention and extraction

For every field, the coefficient convention is

```text
c(K) = sum[w (E - mean_w(E)) exp(-i K.r)] / sum(w).
```

The on-grid x geometry uses uniform weights and therefore the exact periodic
`K=(pi,0)` bin. The exact 45-degree vector
`K=(pi/sqrt(2),-pi/sqrt(2))` is off the periodic FFT grid; it reuses the
original sweep's separable Hann-window demodulation at the exact continuous
wavevector rather than sampling the nearest FFT pixel. Both models use the
identical convention. With this sign convention, the controlled material
response is nominally positive quadrature at `+pi/2`.

Complex coefficients are reconstructed from the retained magnitude and phase
as `c=|c| exp(i phase)`. Phase differences are
`wrap(arg(c_NL)-arg(c_lin))` in the principal interval `[-pi,pi)`.

## Primary 1024-square result

At the strongest retained controlled modulation, `m=0.5`, the fundamental is
well resolved. Values below use the reduced active field.

| Geometry | `|E_NL(K)|` | phase NL | Re NL | Im NL | `|E_lin(K)|` | phase lin | Re lin | Im lin | `R_K` | delta phase |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x | 0.1220815962 | +pi/2 | 7.48e-18 | 0.1220815962 | 0.1131129021 | +pi/2 | 6.93e-18 | 0.1131129021 | 1.0792897531 | 0 rad |
| 45 deg | 0.1068489440 | +pi/2 | 6.54e-18 | 0.1068489440 | 0.0993326711 | +pi/2 | 6.08e-18 | 0.0993326711 | 1.0756676819 | 0 rad |

Thus the nonlinear Bragg-matched fundamental is larger by about 7.93 percent
for x and 7.57 percent for 45 degrees. There is no measured phase shift in
this controlled frozen-intensity comparison. The enhancement is therefore a
magnitude effect, not a change in quadrature phase.

Raw amplitudes are not compared across geometries as a transport measure. The
45-degree result also contains orientation and off-grid window/discretization
effects.

## Resolution dependence at m=0.5

| Geometry | N | fundamental status | `|E_NL(K)|` | `|E_lin(K)|` | `R_K` | delta phase |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| x | 256 | marginal | 0.0386220348 | 0.0359977330 | 1.072901864 | 0 |
| x | 512 | resolved | 0.1029673065 | 0.0954632987 | 1.078606207 | 0 |
| x | 1024 | resolved | 0.1220815962 | 0.1131129021 | 1.079289753 | 0 |
| 45 deg | 256 | marginal | 0.0655053432 | 0.0610174381 | 1.073551188 | 0 |
| 45 deg | 512 | resolved | 0.0978668618 | 0.0910116883 | 1.075321903 | 0 |
| 45 deg | 1024 | resolved | 0.1068489440 | 0.0993326711 | 1.075667682 | 0 |

The absolute coefficients remain resolution-sensitive, especially for x.
The nonlinear/linearized ratio is much more stable from 512 to 1024: it
changes by about 0.063 percent for x and 0.032 percent for 45 degrees. The 256
rows support only a cautious ratio observation because K is marginal and the
nonlinear harmonics are underresolved. Primary interpretation therefore uses
1024.

## Modulation dependence and asymptotic consistency

At N=1024:

| Geometry | m | `|E_NL(K)|` | `|E_lin(K)|` | `R_K` | delta phase |
| --- | ---: | ---: | ---: | ---: | ---: |
| x | 0.50000 | 0.1220815962 | 0.1131129021 | 1.079289753 | 0 |
| x | 0.25000 | 0.0575544374 | 0.0565564511 | 1.017645844 | 0 |
| x | 0.12500 | 0.0283997557 | 0.0282782255 | 1.004297660 | 0 |
| x | 0.06250 | 0.0141542075 | 0.0141391128 | 1.001067585 | 0 |
| x | 0.03125 | 0.0070714402 | 0.0070695564 | 1.000266473 | 0 |
| 45 deg | 0.50000 | 0.1068489440 | 0.0993326711 | 1.075667682 | 0 |
| 45 deg | 0.25000 | 0.0505062594 | 0.0496663355 | 1.016911332 | 0 |
| 45 deg | 0.12500 | 0.0249355573 | 0.0248331678 | 1.004123098 | 0 |
| 45 deg | 0.06250 | 0.0124293088 | 0.0124165839 | 1.001024832 | 0 |
| 45 deg | 0.03125 | 0.0062098814 | 0.0062082919 | 1.000256015 | 0 |

`R_K` approaches one and the phase difference remains zero as `m` tends to
zero. For x, `(R_K-1)/m^2` decreases through `0.3172`, `0.2823`, `0.2751`,
`0.2733`, and `0.2729` as m decreases. For 45 degrees it decreases through
`0.3027`, `0.2706`, `0.2639`, `0.2624`, and `0.2622`. The small-modulation
values therefore tend toward geometry-specific constants while
`Delta_phi_K` tends to zero. This agrees with the previously established
first-order response and quadratic nonlinear-minus-linearized error.

## Relation to higher harmonics

At N=1024, nonlinear `2K/K` grows from `0.00909` to `0.15625` for x and from
`0.01024` to `0.17600` for 45 degrees as m increases from 0.03125 to 0.5.
Nonlinear `3K/K` grows from about `1.0e-4` to `0.03072` for x and from
`1.14e-4` to `0.03369` for 45 degrees. Over the same sequence, `R_K` increases
above one while delta phase stays zero. In this controlled case, higher-
harmonic growth accompanies fundamental enhancement, not depletion or a
fundamental phase rotation. This is an association within the solved model,
not a claim that harmonic growth causes the enhancement.

## Relation to carrier transfer

Carrier gains are available only for N=256 production fixtures, where the
fundamental is marginal and higher harmonics are underresolved. They are not
used for the primary physical conclusion.

| Fixture | Geometry | m | `R_K` | delta phase (rad) | signal gain NL | signal gain lin | pump gain NL | pump gain lin |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| broad | x | 0.5 | 1.8062 | -0.03946 | 5.4891 | 3.9493 | 0.67769 | 0.78824 |
| broad | x | 0.125 | 1.1500 | -0.00027 | 5.5282 | 4.9364 | 0.98216 | 0.98450 |
| broad | x | 0.03125 | 1.1196 | +0.00063 | 5.5467 | 5.0199 | 0.99888 | 0.99901 |
| broad | 45 deg | 0.5 | 1.2353 | +0.01319 | 10.7902 | 7.5124 | 0.27266 | 0.51617 |
| broad | 45 deg | 0.125 | 1.3350 | -0.08712 | 12.1395 | 9.8617 | 0.92822 | 0.94290 |
| broad | 45 deg | 0.03125 | 1.2186 | -0.08249 | 4.4691 | 4.2121 | 0.99046 | 0.99117 |
| Gaussian 20 um | x | 1.0 | 51.4237 | +0.09988 | 1.3571 | 1.1136 | 0.64292 | 0.88638 |
| Gaussian 20 um | 45 deg | 1.0 | 45.9074 | +0.15169 | 1.5551 | 1.1896 | 0.44494 | 0.81043 |

Every retained production pair has both a larger nonlinear fundamental and a
larger nonlinear signal gain, accompanied by stronger pump depletion. The
broad x phase differences are small, while some off-grid 45-degree and
Gaussian rows have measurable phase changes. The realistic Gaussian
coefficients are very small windowed final-slice projections and their large
ratios must not be compared directly with the controlled broad-field ratios.

The production evidence is consistent with magnitude enhancement contributing
to stronger transfer. It does not isolate causality: self-consistent optical
feedback, propagation history, envelope evolution, underresolution, and—in
the 45-degree rows—off-grid demodulation all remain. Whether a nonzero phase
change is more favorable cannot be inferred solely from these endpoint data
and sign conventions.

## Reproducibility and compact evidence

The read-only analyzer is
`scripts/checks/pr_bragg_fundamental_analysis.py`. It pairs retained reduced
nonlinear and linearized rows, reconstructs real and imaginary coefficients,
wraps phase differences, joins available carrier gains, and writes:

- `results/pr_bragg_fundamental_analysis_2026-09-10/summary.csv`;
- `results/pr_bragg_fundamental_analysis_2026-09-10/summary.json` with input
  and analysis-script SHA-256 provenance;
- plots of `R_K`, delta phase, and both magnitudes versus modulation;
- carrier-gain plots versus fundamental magnitude and phase difference.

The output contains 38 paired rows: 30 controlled material pairs and eight
production pairs. No large arrays are copied or saved.

The analyzer refuses to run unless repository `HEAD` equals the requested
sweep commit. Each input JSON must be byte-identical to the corresponding
path at that commit, must retain clean-production-source provenance, and must
pass its own scientific-payload and companion-artifact checksum checks. This
allows unrelated dirty-tree content to remain untouched without letting it
become an analysis input.

The authoritative bindings for this regeneration are:

- production/sweep commit:
  `1e5fb7b18961a8536d2cfa3904cc8db4d401c24f`;
- 02_31 analysis-script SHA-256:
  `847217d94efc7171980ff5a9907b6d411e5066f3f9456781b921c18850793a13`;
- material input JSON SHA-256:
  `297df63234bfb69f2689a3182adedc2a76507992be17758e7ebaf5df6380f247`;
- broad-production input JSON SHA-256:
  `7bfec9d649606fb5c4ae1e381b1afd24ff05b3dfb37a7fa4cb135af1ff54dbfd`;
- Gaussian-x input JSON SHA-256:
  `f6590ccfe8a59a533c7eb6d81b4a5feb3e8c5cd77a5a5489e6fbd0ee0bc4adfa`;
- Gaussian-45-degree input JSON SHA-256:
  `35aaf021378ab124dca6892fb87e4952b746b83c53a6053bf16b2c3d83498002`.

All four input manifests identify the 02_30 harness SHA-256 as
`9109aeb8952dd1e444961268628d3ccc7eddfb22938514083eab9d72f77e00fa`
and the clean production source as commit
`5cbdbeb149b5b2d6960d5fe7572fdf29195f97a3`. The regenerated JSON manifest
also records the exact SHA-256 of `summary.csv` and each of the five plots,
plus scientific-payload SHA-256
`077427b7403bab81bf1426980f9d514bdbfbc2d73c57a2f79cde02655fdfea42`.
The analysis environment was Python 3.12.13, NumPy 2.5.1, and Matplotlib
3.11.0; those versions are retained in the manifest.

## Comparison with the superseded package

The superseded `summary.json` had SHA-256
`7b0e8264de546b06b4023f9a200b8860c66b5ff61f03fa30b00444d71febc87e`.
All 38 row keys and all nonnumeric values match the regeneration. Of 526
numeric scalar values, 16 differ, all in the production-static rows and none
in the 30 controlled material rows. The largest absolute difference is
`6.821210263296962e-13` in the Gaussian-x `R_K`; the largest relative
difference among values with magnitude at least `1e-14` is
`1.347096178481491e-13`, in the broad 45-degree `m=0.5` phase difference.
These are floating-point roundoff differences and do not change any result or
interpretation. In particular, the two primary 1024-square ratios and their
zero phase differences reproduce exactly.

The regenerated `summary.json` has SHA-256
`fac7dc05998ae0335831a6476251f5ae2f6f04a6c8623baec7ae457f9df962bf`.
Its manifest is the authoritative per-artifact checksum inventory; the old
package is retained only as the explicitly identified comparison source and
is not an authoritative scientific input.

## Local validation

- Focused 02_31 analysis and 02_30 source-sweep tests: 31 passed.
- Python syntax compilation of the analyzer and its focused test: passed.
- Independent Git-content, input-payload, companion-artifact, output-artifact,
  CSV/JSON row-count, and primary-value consistency checks: passed.
- Candidate text whitespace inspection and `git diff --check`: passed.
- Fresh read-only review of the 10-file 02_31 boundary: no scientific,
  provenance, evidence-integrity, or scope blockers found.

## Limitations and verdict

- The controlled comparison is reduced nonlinear versus reduced linearized;
  no full-transverse nonlinear counterpart exists in the retained sweep.
- The 45-degree coefficient is an exact-wavevector windowed demodulation on an
  incommensurate periodic grid, not an exact FFT-bin coefficient.
- Carrier-transfer rows are N=256 endpoint measurements and cannot establish
  grid-converged transfer or phase causality.
- The m=1 rows are realistic Gaussian fixtures, not fabricated broad-beam
  asymptotic rows.

For both geometries, the reliable 1024-square, m=0.5 nonlinear fundamental is
stronger than the linearized fundamental. In the controlled fixture the phase
is identical at `+pi/2`, so phase change is not important to that enhancement.
Production rows show some phase differences, but available evidence cannot
classify those differences as more favorable for energy transfer.

No PR equation, material operator, optical propagation path, source/coherence
rule, carrier-power diagnostic, GUI component, or production workflow changes
in this milestone.

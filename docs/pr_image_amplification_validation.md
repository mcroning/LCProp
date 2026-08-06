# Photorefractive Image-Amplification Validation

**Date:** 2026-08-05

**Branch:** `feature/pr-second-order-static`

## Purpose

This record defines the first substantive LCProp image-amplification benchmark
after completion of the second-order semi-implicit PR material solver and its
bounded numerical-readiness gate. The physics reference is Section 3.2 of
“Three-Dimensional Scalar Time-Dependent Photorefractive Beam Propagation
Model,” Photonics 12, 113 (2025), DOI
[`10.3390/photonics12020113`](https://doi.org/10.3390/photonics12020113). The
implementation conventions are cross-checked against the frozen
`reference/prprop/prprop3d.py` source.

## Source Images

The original PRProp3D targets are maintained at
[`mcroning/sample_images`](https://github.com/mcroning/sample_images). The
validation inspection used repository commit
`d11989fd8fe4ba14f8fe27e7a32a7dfd4c9ef426`. The repository is MIT licensed.

The scaled Air Force chart run used:

- file: `AF Res Chart.png`;
- dimensions: 255 × 216 pixels, RGBA source converted to grayscale;
- SHA-256:
  `e424f1070587d79d2a91f4a3bd14e7c11e80adfe7c723e45432d9d4fc5bb85b7`.

Images are loaded only by the research check script. The headless PR package
accepts a real two-dimensional intensity array and therefore does not add
Pillow as a package dependency.

## Paper Benchmark Contract

The published Figure 4 small-signal case reports:

| Parameter | Value |
|---|---:|
| Saturated small-signal gain | 4000 |
| Input peak-intensity ratio | \(10^{-5}\) |
| Grid | 16,384 × 2,048 × 2,175 |
| Longitudinal step | 2 µm |
| Interaction length | 4.35 mm |
| Transverse aperture | 4 mm × 4 mm |
| Wavelength | 0.514 µm |
| Beam waists | 3.4 mm |
| External incidence angles | ±7.56° |
| Dark intensity | 0.01 |
| Tukey edge parameter | 0.05 |

On a 4 mm periodic aperture, 7.56° lies within 0.01° of Fourier mode 1024.
LCProp selects the exact integer mode, giving eight transverse samples per
two-beam grating period. The external carrier is mapped to the equivalent
internal phase gradient before constructing kernel-crossing beam centers.

One float64 E state at the paper grid requires 543.75 GiB. The production
semi-implicit update also requires source, predictor, and work arrays. The
current LCProp PR workflow is deliberately unbatched, so the exact Figure 4
case is presently a preserved parameter contract rather than an executable
ordinary benchmark. PRProp3D used batching and Tukey apodization; neither is
silently substituted into LCProp in this milestone.

## Gain and Image Measurements

The paper's finite-ratio absolute signal gain is

\[
G_0=\frac{(1+r)\exp(2\gamma_pL)}{1+r\exp(2\gamma_pL)},
\]

with \(G_{0\mathrm{sat}}=\exp(2\gamma_pL)\). LCProp calculates
\(\gamma_pL\) from its normalized grating wavenumber and the same plane-wave
coupling relation already validated by the two-beam benchmark.

The image measurement follows the trusted PRProp3D procedure:

1. Normalize the input image as an intensity transparency.
2. Apply its square root to the coherent signal field.
3. Propagate the pump and image-bearing signal through the evolved E state.
4. Partition the output Fourier plane by distance to the two known carriers.
5. Isolate the signal-carrier region.
6. Back-propagate that complex signal field through the inverse LCProp linear
   kernel to the input image plane.
7. Measure carrier-region power gain, intensity correlation, and normalized
   RMS image error.

No spatial division of overlapping beams is used. The zero-response replay
back-propagates to the input with correlation one to numerical precision.

## Scaled CPU Validation

The ordinary validation is explicitly a scaled analogue, not a reproduction
of Figure 4. It uses a 64 × 32 × 10 grid, a 64 µm × 32 µm aperture, 100 µm
interaction length, 0.633 µm wavelength, 12 µm waists, peak signal-to-pump
ratio \(10^{-3}\), and \(G_{0\mathrm{sat}}=10\). The production
`semi_implicit_trapezoidal` material integrator runs 500 steps of normalized
size 0.05.

Using the supplied Air Force chart gave:

| dz (µm) | Analytic plane-wave gain | Measured finite-image gain | Image correlation | Normalized image RMSE | Power drift |
|---:|---:|---:|---:|---:|---:|
| 20 | 9.91089108911 | 7.17061640552 | 0.974401300587 | 0.328746089801 | \(4.44\times10^{-16}\) |
| 10 | 9.91089108911 | 7.17128352532 | 0.974312569289 | 0.329595628035 | \(6.66\times10^{-16}\) |
| 5 | 9.91089108911 | 7.17163280385 | 0.974257079911 | 0.330138361064 | \(8.88\times10^{-16}\) |

The gain changes by less than \(1.5\times10^{-4}\) between successive
longitudinal refinements. Its finite-image value is below the plane-wave
prediction, as expected for finite beams, spatial modulation, incomplete
carrier orthogonality, and the generalized hopping model. Reversing the
coupling direction changes the scaled signal from amplification to depletion.

## Scope and Remaining Work

This milestone validates the image-transparency, coherent propagation,
carrier-separation, back-propagation, gain, fidelity, and power-measurement
chain on CPU. It does not claim quantitative reproduction of the published
4 mm aperture result.

The exact paper-scale comparison requires a PR-owned memory strategy such as
longitudinal batching or out-of-core E storage, a deliberate decision about
the paper's Tukey apodization versus LCProp's periodic boundaries, and a
research-scale GPU convergence run. Those changes should be reviewed as a
separate performance and boundary-condition milestone rather than hidden in
the benchmark.

## Figure 6 Saved-Run Contract

The subsequently supplied trusted-run parameters for Figure 6 supersede an
earlier caption-based inference. They are a distinct ratio-one case:

| Parameter | Saved value |
|---|---:|
| Gain-length product `gl` | 10 |
| Input peak-intensity ratio | 1 |
| Grid | 16,384 × 1,024 × 1,970 |
| Aperture | 3,000 µm × 1,000 µm |
| Interaction length | 3,940 µm |
| Longitudinal step | 2 µm |
| Wavelength | 0.5 µm |
| Beam waists | 600 µm |
| External half-angles | ±0.08543723722873033 rad |
| Refractive index | 2.4 |
| Dark intensity | 0.01 |
| Applied field | 0 |
| Relative permittivity | 2,500 |
| Mobile charge density | 2e22 m^-3 |
| Temperature | 293 K |
| Tukey alpha | 0.2 |
| Volume-noise epsilon | 0.02 |
| Noise correlation length | 0.4 µm |

LCProp replaces the saved `No Image` selector with the checksummed Air Force
resolution chart by explicit project direction. The supplied external angle
maps to periodic Fourier mode 512 on the 3 mm aperture, within 2e-6 rad, and
gives 16 samples per interference-grating period.

The complete saved configuration subsequently supplied all 1,970 generated
volume-noise seeds, exactly matching the number of longitudinal slices. LCProp
preserves that explicit sequence and selects seed `k` for slice `k`, matching
the trusted run's indexing. The sequence is validated as unsigned 32-bit
integers and results record its count, endpoints, and SHA-256 digest. A
base-seed derivation policy remains available for new reproducible fanning
experiments, but Figure 6 does not use it. Interpreted as little-endian uint32,
the historical sequence has SHA-256
`ade77c0e678bf3c2836131c4771e9df22774eba3cc3eb30e17150107adf2f32f`.

The full float64 E volume for this corrected contract would require about
246 GiB before source, residual, solver work, and optical arrays. The PR-owned
streaming workflow instead retains two transverse E slices and selected
diagnostics. Its production, full-nonlinear Lie reference, and legacy
linearized spectral/Lie modes remain explicitly distinct so agreement or
departure can be attributed to the material equation and optical ordering.

---

End of validation record.

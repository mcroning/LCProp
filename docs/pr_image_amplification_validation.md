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
implementation conventions were cross-checked against an archived trusted
reference implementation during development. That unpublished reference is
not part of the public package.

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

## Figure 6 Published Contract

A direct panel audit established that the paper caption and rendered panels,
rather than the mislabeled supplementary JSON, define the final Figure 6
request. The private run record and source figures are archived separately.
The public contract is the ratio-one large-signal case using the Figure 4
geometry:

| Parameter | Published value |
|---|---:|
| Saturated small-signal gain | 4,000 |
| Input peak-intensity ratio | 1 |
| Grid | 16,384 × 2,048 × 2,175 |
| Aperture | 4,000 µm × 4,000 µm |
| Interaction length | 4,350 µm |
| Longitudinal step | 2 µm |
| Wavelength | 0.514 µm |
| Beam waists | 3,400 µm |
| External half-angles | ±7.56° |
| Refractive index | 2.4 |
| Dark intensity | 0.01 |
| Applied field | 0 |
| Relative permittivity | 2,500 |
| Mobile charge density | 6.4e22 m^-3 |
| Temperature | 293 K |
| Tukey alpha | 0.05 |
| Volume noise | Disabled |
| Image transparency | Inverted Air Force chart |

The external angle maps to periodic Fourier mode 1024 on the 4 mm aperture and
gives eight samples per interference-grating period. The real-image
preprocessing normalizes the checksummed chart, inverts it, pads it to an even
square using transparent value one, rotates it into LCProp's `(x, y)` array
convention, resizes it with nearest-neighbor interpolation, and applies the
square root as a signal-amplitude transparency. Inversion before padding is
essential: it produces the published dark chart field with bright bars while
leaving the surrounding Gaussian beam unobstructed.

The supplementary `Figure_3_4_6.json` contains a separate no-image, noisy
3,000 µm × 1,000 µm configuration with `gl=10`. It is not a serialized Figure
6 request. Its historical seed sequence remains available for future fanning
validation but is not selected by `paper_figure6_spec()`.

The PR-owned streaming workflow retains only bounded transverse state and
selected diagnostics. Its production, full-nonlinear Lie reference, and legacy
linearized spectral/Lie modes remain explicitly distinct; correcting the
benchmark transparency does not change any of those numerical paths.

---

End of validation record.

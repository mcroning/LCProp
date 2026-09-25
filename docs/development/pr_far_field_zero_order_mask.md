# PR far-field zero-order mask diagnostics

## Preflight and boundary

Product baseline `5fad0e3383c977338ea130951fb50179ce59c8f8`;
worktree `/private/tmp/lcprop-pr-far-field-mask`, branch
`feature/pr-far-field-mask`. LaunchPlane remains
`3c99ca4c9dd86b3b463063b3a34383911a6e258f`.
Bounded Development + Local Validation: completed PR TD presentation products,
a reusable mask helper, independent regressions and documentation. Scientific
model/equations changed: N/A. Local regression/benchmark evidence only; no
commissioning or fanning-success evidence. No remote/GPU/native application work.

## Canonical and derived products

The canonical Output Far-Field Intensity is unchanged. The new
**Output Far-Field Intensity — zero-order masked** is a separate derived
presentation product in completed reduced/full-transverse TD, both Full and
Fast retrieval paths. The existing canonical transform, coherence grouping,
FFT shift, dx*dy factor, (n/lambda)^2 density, absolute in-medium axes and
x-right/y-up display remain unchanged. No window, recentering or output fitting
is added. The mask never feeds propagation, evolution, launch, screens,
checkpoints, continuation or saved scientific arrays.

[Mask helper](../../src/lcprop/pr/far_field_mask.py) consumes the canonical
2-D intensity and the result's retained `launch_summary.resolved_beams`.
This is authoritative [physical resolver](../../src/lcprop/optics/physical_launch.py)
geometry, not current editor state. Disabled editor beams never enter the
resolved launch; every retained enabled channel contributes an ellipse,
including coherent channels and zero-power enabled channels. Overlaps form a
union; neither intensity nor excluded area is counted twice.

Carrier center is `(kx,ky)/sqrt(kx²+ky²+kz_internal²)`. Retained tangential
components and internal longitudinal component determine the internal magnitude
without duplicating refraction. The retained entrance quadratic Q defines
amplitude `exp(-r.T Q r)`, so intensity spectral covariance is Q in k-space,
and Q/k_internal² on direction-cosine axes. This is the same analytic convention
used by [launch sampling](../../src/lcprop/optics/sampling.py). It accounts for
unequal external beam-normal radii, roll, oblique entrance projection and
arbitrary azimuth. It describes the unperturbed, unscreened, untruncated incident
Gaussian, not the observed spectrum or a fit to scattering.

Default exclusion is Mahalanobis radius <= **4 standard deviations** around
each carrier. An ideal 2-D Gaussian ellipse encloses `1-exp(-8)` of its lobe
(approximately 99.96645%). This radial criterion differs from separate marginal
4-sigma intervals in launch sampling. There is no pixel-radius parameter or
interactive tuning UI. The helper accepts an explicit positive multiplier;
all Product callers use the deterministic default and record it.

## Unavailable pixels, scale and metrics

The current generic Results path uses `np.asarray`, which would discard a
NumPy masked-array mask. Therefore the derived array uses **NaN to mean
carrier-excluded/unavailable**, with explicit metadata. It never writes zero
into excluded pixels and never mutates canonical data. Existing finite-only
Auto limits ignore NaNs; Matplotlib masks invalid values. All-excluded data
use the existing empty-data display range (0,1), not a measured range.
Manual/locked limits remain available. Distinct product keys keep canonical
and masked display settings separate. Selection back to canonical restores
its unchanged values and scale state.

Off-carrier diagnostics use the full canonical uniform FFT grid cell measure
`ds_x*ds_y`, including every cell (no trapezoid endpoint half weighting).
`off_carrier_field_norm = sum(I[not excluded])*ds_x*ds_y`;
`total_field_norm = sum(I)*ds_x*ds_y`; fraction is their ratio when total>0.
Zero total yields an unavailable fraction with an explicit reason. These are
sampled field-norm diagnostics, **not physical mW or fanning power**. Coarse
sampling can miss a lobe or weak feature; the analytic enclosed fraction is
not a guarantee for finite-grid, clipped, screened or evolved data.

Provenance records convention, multiplier, ideal enclosed fraction, every
carrier center/covariance, carrier count, excluded-pixel count, coordinate
measure, field norms, fraction and qualifications. Missing/incomplete historic
geometry produces an explicit unavailable diagnostic, never a guessed mask
or failure of ordinary propagation. Canonical far field remains available.

## Lifecycle and live support

Masking is **completed-only**. Reduced live snapshots retain intensity/axes but
not per-channel resolved covariance/geometry. Adding that provenance would
change the bounded snapshot interface and requires separate review. No mask
or metric is computed from decimated previews, and a prior completion's mask
is never reused in live Results. The existing execution-start invalidation
clears old derived products; each completion derives its own products from its
own result. Completed-only applies equally to stopped result presentation.
Helpers are reusable by future static presentation; static adapters and static
noise/scattering are unchanged here.

## Validation plan and evidence

Independent geometry tests use explicit trigonometric transverse bases, not
resolver helpers for expected values. Synthetic tests construct bright carrier
regions and known weak off-carrier content independently, exercising tilted,
separated and overlapping unions and the exact discrete integral. Results
tests cover Auto/manual limits, NaN display, canonical switching and invalidation.
A real worker A-to-B regression exercises disabled-beam filtering and verifies
no completed mask/metric leaks into waiting/live states. Reduced Full/Fast and
full-transverse Full/Fast adapters are covered.

A tiny gain-0/gain-10 linearized exact-modal regression retains identical
canonical v2 scattering configuration. Each spectrum is checked against its
own complex output FFT; each metric is checked against its own full-grid data.
Mask definitions and scattering provenance agree. There is no assertion that
gain 10 produces more fanning or a larger off-carrier fraction.

Initial focused run: 13 passed, one test assertion failed because the established
empty diagnostics panel contains `Workflow: pending`, not an empty string.
The assertion now checks that neutral text; production invalidation was unchanged.
Final focused set: **16 passed in 2.52s**. Affected PR GUI/products/Results,
TD lifecycle/live equivalence and launch-sampling set: **198 passed in 31.31s**;
the subsequently added independent rotated-ellipse exclusion test passed in the
final focused set and is included in the complete suite. The first complete suite returned **2022 passed, 77 skipped, 2 failed in
539.60s**. Both failures were stale presentation expectations: Fast transport's
exact field list omitted the new derived product; Image Amplification's
bit-for-bit presentation comparison used `np.array_equal`, which returns false
for matching NaNs. The field list now includes the new product, and only that
product uses shape/dtype/byte equality, including identical NaN positions and
payloads. Existing scientific array comparisons remain untouched. No production
change was required. Final expanded affected set: **283 passed in 36.34s**. Final complete Product suite: **2024 passed, 77 skipped in 536.10s (0:08:56)**. No failures. No LaunchPlane changes or suite rerun are needed.

A presentation-only NumPy benchmark on an existing 2048×64 spectrum, one carrier,
30 repetitions after warm-up measured median **0.659 ms**, maximum **1.133 ms**
for incremental mask plus metric construction. The derived float64 image retains
**1,048,576 bytes**; the boolean exclusion and distance scratch arrays scale with
a 2-D plane and are not retained. Axes are reused, no extra FFT is performed,
and no volume or live-payload bytes are added. This is not GPU qualification.
Commands/logs, checksummed benchmark harness, inventory and candidate manifest
are external `/private/tmp/pr-mask-*` artifacts and excluded from the candidate.

## Scientific non-change and protection

No changes to resolver, canonical far-field kernel, launch sampling/Nyquist gate,
material equations, gain, propagation/order, normalization, scattering/slabs,
screens, boundaries, schemas/codecs, continuation or LC physics. Research,
reconstruction candidate, installed environment, authoritative repositories and
prior worktrees are preserved. Fresh historical baseline: **4,351 paths**,
fingerprint `f9a194bbc051dbb780d900c3a678545435c303c4c092107fbef238ae64ef12cc`.
Installed-environment inventory: 19,737 files.

The 2048×64, 500×200 µm case with 0.4 µm scattering correlation length remains
a separate scientific sampling audit. No scattering changes or fanning claims
are authorized. Edge ownership, static noise, remote continuation, generic
reconstruction and persistence-cycle repairs remain excluded.

## Later native acceptance

After separate review/commit/integration/install, select completed masked and
canonical spectra; verify weak off-carrier scaling, tilted and two-carrier
exclusions, metadata and manual limits, then repeat the gain pair without a
preordained scientific interpretation. Native acceptance remains pending.

## Development self-review

The complete candidate consists of one pure presentation helper, two completed
TD adapter integrations, one focused regression module, two narrowly updated
presentation compatibility tests and two documentation files. Fast adapters guard on TD workflow identity; static paths remain unchanged.
No existing test expectation was weakened and no production propagation code
was edited. Optional missing metadata yields a diagnostic reason; supported
metadata uses the retained result, never mutable GUI configuration. Geometry
and spectral-integration expectations use independent formulas. Real offscreen
Qt tests cover selection, scales, disabled channels and actual worker delivery.
NaN availability semantics and non-mW units are explicit. No persisted or
transport schema changes, no public positional API changes, and no new scientific
model/status axis. Authoritative scientific evidence/schema gates are N/A:
this is local implementation regression and presentation-performance evidence.
Native acceptance and the deferred scattering-resolution audit remain separate.

Final syntax compilation in memory, Markdown/local links, candidate whitespace,
`git diff --check`, exact eight-file scope and empty-index checks pass. Protection
comparison confirms authoritative Product/LaunchPlane/Research, reconstruction
candidate, all 4,351 historical paths and 19,737 installed-environment files
remain unchanged. Tests used Python 3.12, offscreen Qt, isolated Product sources
and unchanged authoritative LaunchPlane sources, with bytecode and pytest-cache
writes disabled. No native or GPU qualification is claimed. Development + Local
Validation passed; the unstaged/uncommitted candidate is ready for independent
read-only pre-commit review.

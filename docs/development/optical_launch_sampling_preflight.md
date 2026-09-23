# Optical launch sampling preflight

## Scope and provenance

Product baseline `74f4df51c9f8d28eb0e98d0555b2ad4b1a24f6ff`; isolated branch
`feature/optical-launch-sampling-preflight`. Companion LaunchPlane baseline
`3c99ca4c9dd86b3b463063b3a34383911a6e258f` remains unchanged in its isolated
`test/optical-launch-sampling-preflight` worktree. This is Development + Local
Validation, not commissioned numerical evidence or human native acceptance.

The native observation was a plausible but aliased PR trajectory at external
30°, vacuum wavelength 0.633 µm, external index 1, internal index 2.4,
20 µm beam-normal radii and a 200 × 200 µm aperture. Nx=128 and 256 cannot
represent kx=4.96302157 rad/µm. Nx=512 can, independently of Ny=256 at ky=0.
The correctly sampled ray can reach an edge: periodic wrap and sponge attenuation
are legitimate, separately selected behaviors.

No equations, resolver, kernel, split-step order, normalization, screens, material
physics, carrier algorithms, persistence schemas or continuation are changed.
No cluster/GPU/remote access or installed-environment changes are authorized.
Edge-control relocation, focused launch, reconstruction, scattering/fanning and
the known persistence direct-import cycle are excluded.

## Shared ownership and hard gate

[Shared qualification](../../src/lcprop/optics/sampling.py) consumes the actual
`ResolvedBeamGeometry.kx/ky` used by `sample_resolved_beam`. BeamStack contains
only enabled channels after the existing LaunchPlane adapter; zero-power enabled
channels are still checked. Coherence groups do not exempt a carrier.

[RuntimeGrid](../../src/lcprop/core/grid.py) uses cell-centered samples with
`dx=Lx/Nx`, `dy=Ly/Ny`; the center offset does not change FFT Nyquist.
Each axis requires `abs(k) < pi/d`. Equality at either signed edge is rejected,
including the even-grid negative Nyquist bin, because the two directions coincide
on sampled points. Odd grids use the same sampling-frequency bound; a carrier
need not lie exactly on an FFT bin. No undocumented tolerance relaxes this gate.
The physical transmitted-mode cutoff remains in the existing resolver and is
not confused with numerical Nyquist.

The carrier-only integer minimum is `max(2, floor(abs(k)*L/pi)+1)`, checked using the same `pi/(L/N)` expression as the gate against
the strict floating-point inequality to handle rounding at an integer boundary.
Errors identify enabled-beam number/name, axis, signed carrier/magnitude, limit,
count, width, spacing and minimum. A separate finite-bandwidth recommendation
is not a power-of-two requirement.

[build_launch](../../src/lcprop/optics/launch.py) enforces the gate before sampling,
so ordinary direct LC/PR workflow calls cannot bypass GUI validation. Explicit
initial-field requests still qualify the configured launch, which those workflows
already construct. This is not a certificate for arbitrary supplied arrays.
Stationary LC's existing external-angle restriction remains separate.

PR GUI preflight combines shared sampling warnings with its existing geometry
report; LC propagating request preflight consumes the same calculation. Run and
Inspect Request evaluate the canonical request after pending editor commit and
block errors before worker dispatch. Inspect Request exposes failures and valid
sampling qualification without dispatch. [BeamPanel](../../src/lcprop/gui/panels/beam_panel.py)
keeps analytic footprints/rays when its independent preview sampling grid is too
coarse, reporting capture unavailable instead of sampling an aliased field.
LaunchPlane performs no Nyquist/refraction calculation and needs no source change.

## Spectral margin derivation

The supported collimated Gaussian entrance field has envelope
`A(r)=exp(-rᵀ Q r)`. Its Fourier amplitude is proportional to
`exp(-delta_kᵀ Q⁻¹ delta_k/4)`; spectral power is proportional to
`exp(-delta_kᵀ Q⁻¹ delta_k/2)`. Thus power covariance is **Q**, and the marginal
standard deviations are `sqrt(Qxx)` and `sqrt(Qyy)`, not fixed associations of
w1/w2 with x/y. Q comes from the existing physical external projection and roll.

Warn if `abs(kj)+4*sqrt(Qjj) >= pi/dj`. A four-sigma marginal interval leaves
`erfc(4/sqrt(2)) ≈ 6.334e-5` of Gaussian power outside each axis interval
(about 0.006334%). This is a documented significance criterion, not an arbitrary
fraction of Nyquist. Correlations do not change the marginal variances; the sum
of axis tails bounds power outside the rectangle. The practical integer count
applies the same strict-minimum calculation to `abs(kj)+4*sqrt(Qjj)`.

This analytic estimate is for the untruncated, unscreened profile. Finite apertures
and screens may add bandwidth; requests with launch elements receive an explicit
qualification. Existing sampled post-screen diagnostics remain unchanged. No
profile beyond the currently supported collimated Gaussian is newly authorized.

## Boundary warnings

The estimate uses the unwrapped central-ray endpoint `r0+L*(kx,ky)/kz_internal`
and a rigid two-radius entrance ellipse. Its projected half-width along j is
`2*sqrt((Q⁻¹)jj)`. Maximum absolute entrance/exit center plus that width is compared
with the half aperture, or the start of a sponge/Tukey edge region. This warns on
approach/crossing, allows execution and never clips the ray. It explicitly does
not predict diffraction or nonlinear envelope evolution. Existing PR geometry
warnings remain additional qualifications.

Periodic warnings explain possible opposite-side wrap; sponge/Tukey warnings
explain possible attenuation. Neither mode is mandated or forbidden. Spectral
errors and boundary warnings are independently accumulated. Beam owns intent;
Grid owns sampling; edge treatment owns boundary behavior. Controls remain in Beam.

**A physically valid launch angle is not necessarily numerically representable
on the selected transverse grid.** Increasing aperture at fixed N coarsens sampling.

## Regression strategy and preliminary findings

[Analytic tests](../../tests/test_launch_sampling.py) cover the native 128/256/512
case, signed theta, y-only/arbitrary azimuth, enabled/disabled/coherent beams,
strict/equal/neighboring Nyquist counts, broad/narrow/unequal/rolled profiles,
independent covariance projections, boundary modes and the sampled valid-grid
phase gradient. Tests exercise existing boundary operators without modifying them.
[GUI tests](../../tests/test_launch_sampling_gui.py) type pending theta then click
actual Run, check no dispatch on 128/256, retain the ellipse, inspect the failure,
raise Nx to 512 and check dispatch recovery. LC consumes the same gate.

Existing pending-editor/reciprocal worker tests previously used a 16 × 16 grid
for high-angle beams. Their shared lifecycle fixture now uses 512 × 256; all
original geometry, ownership and worker assertions remain. This migration makes
those success-path tests scientifically representable instead of exempting them.

Preliminary corrections: one equality test used a trigonometric construction
that rounded below Nyquist; it now constructs exactly representable resolved k=1.
One GUI assertion omitted the existing 'Execution status:' prefix; only the
assertion was corrected. A LaunchPlane test command initially used Product's
working directory and collected no tests; the correct companion command follows.
Two additional prior tests were migrated to the new contract: aliased launch
construction now expects an exception (rather than unavailable power metadata),
and the advisory grating-risk fixture uses kx=1.5 rather than 5 rad/µm on its
128-point/200 µm grid. Its waist/grating/boundary warning assertions are retained.
An initial full run started before fixture corrections and is preliminary only;
the final complete run executes the corrected candidate.

Focused new tests after the final equality correction: **31 passed in 2.91 s**. Combined lifecycle/preview focused
set: **75 passed in 12.37 s** (before five additional analytic test cases).
Affected Product set before the final equality correction: **281 passed in 46.64 s**.
Unchanged companion LaunchPlane affected set: **53 passed in 1.38 s**.
Self-review found that `pi*N/L` and `pi/(L/N)` differ by one ULP at some
non-power-of-two counts (for example N=7, L=200). The minimum-count helper now
uses exactly the same latter expression as the gate. Five regressions at counts
7, 31, 127, 255 and 327 require equality rejection and acceptance of the adjacent
representable value below the edge. Full validation started before this
correction is superseded by a new complete run; no pass from it is claimed for
the final identity. This correction changes only recommendation consistency,
not physical geometry or the strict carrier gate.
Final affected Product set after equality correction: **286 passed in 47.44 s**.
The superseded pre-equality full run was gracefully interrupted (864 passed,
31 skipped before interruption); it is not completed validation evidence.
## Final validation and self-review

Complete Product suite: **2001 passed, 77 skipped in 516.01s**. No failures.
Final focused: **31 passed**; affected Product: **286 passed**;
unchanged LaunchPlane affected: **53 passed**. Python 3.12 development environment,
offscreen Qt, isolated sources on PYTHONPATH, bytecode and pytest cache disabled.
Commands and logs are external `optical-sampling-*` artifacts. The new tests and
the full suite are local regression evidence, not commissioned science results.

Syntax compilation in memory, Markdown/local links, whitespace and
`git diff --check` pass. Self-review confirms no unrelated production changes,
no public positional API or schema changes, no changed physical kernel/resolver,
no persistence or continuation implementation, and no tolerance relaxation to
accept an unresolved carrier. The valid sampled phase-gradient regression agrees
with the resolver to 2e-14 rad/µm. LaunchPlane source and scientific display
ownership are unchanged; its complete suite is not rerun because no companion
source changed.

Fresh protection comparison preserves authoritative Product at
`74f4df51c9f8d28eb0e98d0555b2ad4b1a24f6ff`, LaunchPlane at
`3c99ca4c9dd86b3b463063b3a34383911a6e258f`, Research at
`b947fac51722eafa971e756a2cc2053be065294d`, the reconstruction candidate,
all 19,733 installed-environment files and the 4,349 pre-task historical Product
paths. Their historical fingerprint is
`b16d2243c369836603ceb24a992b4d2ff9ec88e5e7ca55f90f960074e991b398`.
Existing milestone worktrees/history are retained. Both isolated indexes are
empty; no staging, commit, installation, integration, native application launch,
remote/GPU access or follow-on work is included.

Development + Local Validation is complete. Independent strictly read-only
pre-commit review is next; human native acceptance remains pending.

## Later native acceptance

After separately approved review, commit, integration and installation: verify
30° at 200 µm blocks Nx=128/256, clears at 512 × 256, preserves periodic wrap or
sponge attenuation with warnings, retains the correct steep trajectory, checks
pending edits before Run, and handles y-only incidence per Ny. Automated tests
do not complete this human acceptance gate.

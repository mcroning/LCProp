# PR TD Results lifecycle and far-field diagnostics

## Preflight and scope

Baseline Product `1576632536b26da3f736daa14ec6301e11a9fba2`; isolated worktree
`/private/tmp/lcprop-pr-td-results`, branch `fix/pr-td-results-lifecycle`.
LaunchPlane remains `3c99ca4c9dd86b3b463063b3a34383911a6e258f`.
This is bounded Development + Local Validation for Results invalidation,
selection guides and TD far-field presentation. Equations/models changed: N/A.
No commissioning or authoritative fanning evidence is produced.

Candidate boundary: shared Results presentation, PR product adapters, the
existing reduced live-snapshot builder and its two metadata-only call sites,
regressions and documentation. No launch resolver, propagation, normalization,
material equation, scattering generator, persistence schema, continuation
contract or LaunchPlane source changes. No Slurm/SSH/CUDA/remotes or native GUI
launch. The installed environment and authoritative worktrees are protected.

## New execution invalidates displayed products

Previously, `begin_request` marked old content as Previous result but retained
its arrays. `_start_background` reset scales and started the worker without
invalidating those products. Thus A remained visible during B's initialization.

[PR dispatch](../../src/lcprop/pr/gui/main_window.py) now calls
[Workspace.invalidate_products](../../src/lcprop/gui/workspace.py) only after
validation/cost approval, at actual execution start. It publishes an empty
presentation record and **Waiting for current result**, clears field selectors
and image artists, and releases pane references to the prior product. This is
an unavailable state, not a synthetic zero scientific field. Saved results,
checkpoints and runner results are separate and are not destroyed by this
presentation operation. Existing fresh-run checkpoint policy is unchanged.

The first accepted B snapshot populates only B's arrays; subsequent snapshots
replace them, then completion publishes B. Scalar progress cannot resurrect A.
The selected Results subtab is retained. Product keys are retained as preferences
where available, including a far-field selection through live → completed.
Existing scale policy at new-run start remains unchanged. Invalid requests that
never start may still show explicitly identified previous results.

## Live guides and physical coordinates

The generic fixed-paired-cut path explicitly hid Show selection guides and
cleared its artists on every update. It did not publish physical cut positions
to the transverse pane. Its slider indices also cannot represent all fixed cut
coordinates on a downsampled preview grid.

[LongitudinalPane](../../src/lcprop/gui/views/longitudinal_pane.py) now keeps the
control visible and renders fixed-cut guides using the product's exact
`x_cut_um`, `y_cut_um` and selected z coordinate. The transverse vertical line
is at x_cut (the y-z cut), and its horizontal line is at y_cut (the x-z cut).
Fixed-cut refresh uses a dedicated guide signal rather than fabricated index
values. Linked volume views retain their existing index-guide API.
Coordinates need not coincide with a displayed preview pixel; no nearest-pixel
substitution changes the sampled cuts. z guides track the labeled z sample.
For fixed live products, the x-y output plane remains the output plane: moving
the z marker does not fabricate a movable x-y slice from unavailable volume data.
Completed volume products retain their real x/y/z slider reslicing.

[ImageView](../../src/lcprop/gui/views/image_view.py) supports persistent physical
coordinate guides in addition to index guides. Workspace synchronizes them after
product and selection changes. Spatial guides are cleared for angular products;
micrometre coordinates must never be drawn on direction-cosine axes. Guide-off
preferences are retained. Fixed-cut clicks cannot silently change retained cuts.
Canonical data remain `[x,y]`; display transpose and `origin='lower'` remain intact.

## Canonical far-field contract

The missing case was Full TD presentation. Existing Fast PR products and static
presentation already exposed angular spectra. Both reduced and full-transverse
**completed Full TD** adapters now expose `far_field_intensity`, reusing
[direction_cosine_spectrum](../../src/lcprop/optics/farfield.py) through a small
[PR presentation helper](../../src/lcprop/pr/far_field.py). Existing Fast/static
algorithms remain unchanged.

The transformed data are the complex `A_final[channel,x,y]` output plane. In live
reduced TD, they are the complex output of the already accepted state's complete
optical replay, before any spatial decimation. This is never an FFT of intensity.
For each coherence group, fields are summed before transforming; intensities of
incoherent groups are summed afterward.

- Transform: `fftshift(fft2(A_group)) * dx_um * dy_um`.
- Density: squared modulus multiplied by `(n_ref/lambda0_um)^2`.
- Axes: `s_x=lambda0_um*fftshift(fftfreq(Nx,dx_um))/n_ref`, likewise y.
- Reference: vacuum wavelength and scalar internal index from the retained launch
  metadata; full-transverse results use their resolved profile when those values
  are not included in the launch summary.
- No added window, carrier recentering, normalization to the spectral peak or
  propagating-disk clipping. Absolute canonical FFT coordinates are retained.
- Arrays use `[s_x,s_y]`; ImageView transposes at the display boundary only.
- Units are **normalized field norm / direction-cosine²**, not physical mW.
  The full-grid integral equals the coherence-aware transverse field norm by
  Parseval. It is not an axial-flux-weighted physical-power diagnostic.

No new carrier annotations or fanning metric are introduced. A cancellation
launch fallback is identified in field observation metadata; absent historical
wavelength/index metadata does not produce fabricated angular coordinates.

## Bounded live support and cost

Reduced Local Run/Continue already has a complete complex output plane at the
accepted boundary. The snapshot builder applies the canonical transform on that
plane's backend, then samples the real spectrum and coordinate vectors before
host transfer. No complex output volume or spectral history is retained.
Full-transverse live delivery is unchanged and not introduced here (P2A-2 remains
out of scope). Completed full-transverse TD has the new product.

The existing first/final/Stop publication and 0.5-second cadence are unchanged.
Only a due snapshot incurs the diagnostic FFT. The added temporary work scales
with a transverse plane and coherence groups, not Nz. Host payload adds at most
one 128×128 float64 image and two 128-element coordinate vectors: 133,120 bytes.
The default total bound increases from 790,528 to **923,648 bytes**. Existing
scientific-equivalence, cancellation, throttling, continuation and no-full-volume
transfer regressions are retained with updated product-count/payload assertions.

A NumPy presentation-only benchmark on a 2048×64 plane (200 z-coordinate samples,
one channel, 12 timed repetitions after warm-up) measured median snapshot times
0.565 ms without far field and 3.800 ms with it; maximum with it was 4.241 ms.
Actual payload was 508,480 bytes versus 441,408 without far field. This is local
performance evidence, not GPU qualification or a propagation/fanning calculation.
The live angular image is point-sampled from the full-grid spectrum. It can omit
narrow peaks and must not be integrated as a full-resolution power measurement;
use the completed full-resolution product for quantitative inspection.

## Regression evidence and qualifications

[New tests](../../tests/test_pr_td_results_lifecycle.py) exercise:

- Real worker A → B with B held before first progress: no A array survives in
  current rendered fields, refresh cannot reinsert it, and saved A remains intact.
  Distinct B fields/spectrum pass through the normal production progress slot.
- Physical fixed-cut guides across accepted replacements, product selection,
  z clicks/sliders, auto/manual scaling and resize; completed volume coordinates
  and sampled cuts remain linked. Angular views omit spatial guides.
- A non-square complex coherent field against an independent direct DFT,
  including axes, phase, normalization, Parseval, and incoherent grouping.
- Full-transverse completed output against its own complex output FFT.
- Backend full-plane FFT before bounded spectral transfer; exact payload bound.
- A tiny linearized/exact-modal gain-0/gain-10 pair with identical canonical v2
  scattering seed/settings. Each spectrum is checked against its own complex
  output, with identical coordinate/quantity contracts and retained deterministic
  scattering provenance. No expected fanning signature is asserted.

Three independent baseline probes fail for the expected reasons: stale A remains
at B startup, live guide artists are absent, and Full TD far field is absent.
These probes are external local test harnesses, not committed scientific evidence.
An initial test invocation referenced a nonexistent transverse-products module;
an affected invocation used an obsolete runtime-test filename. Neither collected
tests. Corrected commands are recorded externally. Initial new tests found two
presentation integration issues: completion's default MPR selection displaced a
far-field preference, and full-transverse angular inputs live in resolved_profile.
Both were corrected without scientific changes. Existing focused tests passed 28;
the first corrected combined focused set passed 33; the expanded affected set
passed 174. Self-review replaced a provisional dummy-index notification with a
separate guide-refresh signal; the superseded full run was interrupted after
680 passes and 1 skip and is not final validation evidence. The next full run
reported 2006 passes, 77 skips and two failures. One old Fast-cut test required
hidden guides; it now requires visible guides and verifies their exact physical
coordinates. The other caught a linked-volume index API compatibility regression;
Workspace now preserves that existing API for linked volumes, using physical
coordinate guides for fixed cuts and independent transverse images. The original
MPR test is retained unchanged. Final validation follows below.

## Deferred scientific questions

The representative commissioning grid has dy=3.125 µm and requested scattering
correlation length 0.4 µm. A future fanning audit must establish the canonical
parameter's interpretation relative to anisotropic optical sampling. This task
neither changes nor rejects scattering and does not claim fanning is fixed.
One-sided-looking y-z structure may be fixed cuts through tilted interference
fringes; no data are symmetrized. Static noise, edge-control relocation,
reconstruction, carrier-power curves, continuation changes and the pre-existing
persistence import cycle remain excluded.

## Protection and later native acceptance

Fresh pre-task Product historical inventory: **4,350 paths**, fingerprint
`17fb9756aa82d411b1e002ef9db2dfdd997c454153c482dbf741a89336c49509`.
The additional active prompt is part of this fresh baseline. Research remains
`b947fac51722eafa971e756a2cc2053be065294d`; installed-environment inventory has
19,735 files. Reconstruction candidate, all authoritative tracked/untracked
material and prior worktrees are protected. LaunchPlane source is unchanged;
no companion implementation or suite rerun is required.

After separate review, commit, integration and installation, human acceptance
must verify A clears immediately when B starts, waiting state precedes B's first
accepted product, guides remain tied to cuts, completed TD far field is present,
reduced live far field updates, and no previous spectrum survives into B.
Gain comparisons require identical axes and their own output spectra, not a
preordained fanning outcome. Automated offscreen tests do not complete native
acceptance.

## Final validation and self-review

Final new lifecycle/guide/far-field tests: **7 passed in 18.82s**.
Final affected set, including unchanged launch-sampling tests and the two full-suite
findings: **189 passed in 30.85s**. Complete Product suite: **2008 passed, 77 skipped in 531.26s (0:08:51)**.
No failures. Python 3.12 development environment, offscreen Qt, isolated Product
sources and unchanged authoritative LaunchPlane sources; bytecode and pytest cache
writes disabled. Validation commands and preliminary/final logs are external
`pr-td-results-*` artifacts. No native acceptance or GPU qualification is claimed.

Syntax compilation in memory, Markdown/local-link validation, candidate whitespace,
`git diff --check`, exact scope and empty-index checks pass. Protection comparison
passes for all authoritative repositories, historical material, reconstruction
candidate and installed-environment inventory.

Self-review: all hunks belong to presentation lifecycle, physical guide mapping,
canonical diagnostic exposure or their regressions/docs. Existing public positional
arguments remain compatible; optional live metadata/arrays are appended. Persisted
and transport schemas are untouched. Scientific units explicitly qualify field
norm rather than physical power; no new model/status axis is introduced. Existing
three-integrator exact scientific-array equivalence and cancellation/cadence tests
pass. Shared kernel, resolver, scattering, material physics, screen, normalization,
launch-sampling policy and continuation semantics remain unchanged. New physical
guide refresh uses its own signal; no fake cut indices are emitted. Source-only
review will independently assess the exact manifest. Evidence here is local
regression/performance evidence, not authoritative commissioning results.

Development + Local Validation is complete; everything remains unstaged and
uncommitted. Independent strictly read-only pre-commit review is next.

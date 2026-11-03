# Image Input / Results coordinate parity

Baseline: `5864c0362cf3311fd6d76edc452d7f29341f9df0`, branch
`feature/pr-second-order-static`.

Status: coordinate corrections implemented; local validation recorded below.
Candidate remains unstaged and uncommitted.

## Convention trace

`make_grid` in `src/lcprop/core/grid.py` constructs ascending cell-center x and
y coordinates. Canonical transverse arrays are `[x, y]`, shape `(Nx, Ny)`.
Index 0 increases toward physical right; index 1 increases toward physical up.

| Boundary | Shape and coordinate meaning | Transformation |
| --- | --- | --- |
| `decode_user_raster`, `gui/image_sources.py` | `(height, width)`; rows down, columns right | Qt applies image metadata orientation; RGB becomes grayscale without axis interchange |
| `prepare_intensity_raster_transmission`, `optics/screens.py` | Padded `(side, side)` raster | Max normalization, optional inversion, centered even-square padding; followed by explicit raster-to-physical conversion |
| v1 raster-to-grid mapping | `(side, side)` interpreted as `[x, y]` | Padded `[row, column]` goes to `[column, side - 1 - row]`; source left stays left and source top maps to positive y |
| Resampling | `(target_nx, target_ny)` | Nearest-neighbor zoom; counts derive from physical width/dx and height/dy |
| Placement | `(Nx, Ny)` | Nearest x/y center indices; centered insertion, unchanged exterior and reject/clip policy |
| Screen and launch | `(channel, Nx, Ny)` | Multiply normalized incident amplitude by square root of transmission; no post-screen renormalization |
| Post-screen preview input | `(Nx, Ny)` | Selected channel's squared magnitude; no scientific axis change |
| Qt preview pixels | `(Ny, Nx)`; rows down, columns right | Previously transpose only; corrected adapter places largest y in the top row |
| PR workflow launch | `(channel, Nx, Ny)` | `build_launch` then `_initial_fields`; retained `A_initial` copies those axes |
| Input Plane Intensity | `(Nx, Ny)`, axes `("x", "y")` | `pr_result_to_run_data` forms optical intensity without geometric remapping |
| Optical Intensity Volume | `(Nz, Nx, Ny)`, axes `("z", "x", "y")` | Reduced TD stores normalized endpoint-averaged slice intensity; product conversion removes background and restores intensity normalization, without transposition |
| Results plotting | `(Ny, Nx)` Matplotlib samples | `ImageView.set_field` transposes, uses `origin="lower"`, and ascending physical extents |

The observed preview/Results divergence begins at `_array_pixmap` in
`src/lcprop/gui/panels/input_screen_editor.py`. Its previous transpose placed
the smallest y at the top of a Qt image, while Results put it at the bottom.
The correction is `pixels[:, ::-1].T`, explicitly adapting ascending canonical
coordinates to Qt's top-down rows. This adapter changes only presentation.
Raw raster preview (`xy_axes=False`) is unchanged.

## Authorized raster correction and power contract

The other defect was at the raster-to-field boundary: `rot90(square)` mapped
`[row, column]` to `[side - 1 - column, row]`, a 180-degree physical rotation
relative to ordinary raster coordinates. The corrected `square[::-1, :].T`
maps to `[column, side - 1 - row]`. Both transformations now have explicit
coordinate contracts; neither compensates for an error elsewhere.

The user explicitly resolved the scope conflict in favor of physical
orientation correctness. Incident launch power and normalization remain
unchanged. Transmitted power may change solely because the corrected screen
has different overlap with nonuniform illumination. Tests require

`P_new - P_old = P_incident * sum(I_incident * (T_new - T_old)) * dx * dy`

for a unit-integral single-channel incident field. They also verify the full
post-screen complex field equals the unchanged incident field times
`sqrt(T_new)`, with no post-screen renormalization. Uniform illumination gives
unchanged throughput; the off-center Gaussian cases require a nonzero change.

The fix is in the shared laboratory-plane screen constructor. All its callers,
including historical Image Amplification wrappers, receive the same corrected
orientation. Existing saved raster inputs retain their schema and samples but
will be interpreted with the corrected orientation when rerun. Retained result
arrays and checkpoints are not rewritten. No schema migration, screen-plane
change, beam-normal projection, or frame rotation was introduced.

Even-square padding, nearest-neighbor interpolation, inversion, footprint
size/center, boundary handling, passive screen multiplication and incident
normalization are unchanged. Propagation equations and kernels are unchanged;
the scientific entrance field changes only as explicitly authorized.

## Regression coverage

`tests/test_image_input_display_parity.py` covers both Qt adapter modes and an
actual decoded 6-by-10 PNG with unequal quadrant markers and an asymmetric
interior. The screen occupies 10-by-20 micrometers on a 24-by-16 grid.
Uniform and off-center Gaussian illumination, each with normal and inverted
raster variants, check known physical coordinates, screen output,
actual preview dispatch, launch arrays, a real small NumPy PR workflow,
Input Plane Intensity, retained optical volume and the Results image artist.
Every preview pixel is checked against canonical coordinates, distinguishing
transposition, reflections and quarter-turns. No screenshot comparison or
mocked propagation is used.

The first reduced-TD volume slice is an endpoint-intensity average, not an
exact z=0 sample. Tests verify that average explicitly and use a short optical
step to bound its difference from launch intensity. Independent uniform-beam
and Gaussian overlap integrals verify incident/transmitted power and throughput. Rendering
does not mutate the input array.

These are development regressions, not commissioned scientific evidence.
No GPU, cluster, remote, reconstruction, continuation or scattering work is
included.

## Local validation

Python 3.12.13; NumPy CPU; `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`,
`QT_QPA_PLATFORM=offscreen`, external `MPLCONFIGDIR`.

Focused: `python -m pytest -p no:cacheprovider -q
tests/test_image_input_display_parity.py`: **6 passed**.

Affected: the same command with these modules: **237 passed**:

- `tests/test_image_input_display_parity.py`
- `tests/test_input_screen_editor.py`
- `tests/test_input_screens.py`
- `tests/test_pr_workflow_input_screens.py`
- `tests/test_pr_products.py`
- `tests/test_pr_static_products.py`
- `tests/test_image_pane.py`
- `tests/test_launch.py`
- `tests/test_pr_general_configuration.py`
- `tests/test_pr_image_amplification.py`
- `tests/test_pr_gui_image_amplification.py`
- `tests/test_pr_transverse_linearized_image_amplification.py`
- `tests/test_pr_timedependent_transport.py`
- `tests/test_pr_transverse_timedependent_transport.py`

Executing the new adapter regressions with the baseline `_array_pixmap`
function loaded in memory produced the expected **1 failed, 1 passed**:
canonical-coordinate preview fails; ordinary raster mode passes. No repository
file was reverted for this check. Loading only the baseline raster mapping
in memory makes all four end-to-end orientation cases fail (**4 failed,
2 adapter tests passed**), confirming that the new tests detect both defects.

The first workflow test attempt used a timestep above the existing Euler guard;
the fixture was corrected to a valid timestep, without changing production
validation. One affected-suite invocation named a nonexistent test module and
collected no tests; the corrected invocation above passed. Syntax compilation,
whitespace checks and `git diff --check` pass.

Final complete-suite validation on 2026-09-22 used the same Python 3.12.13
environment and ran `python -m pytest -p no:cacheprovider -q tests/`:
**1,894 passed, 76 skipped in 539.87 seconds**, with no failures and no
implementation or test changes. Skipped tests remain unexecuted qualifications.
The focused **6 passed** and affected **237 passed** results above still apply
to the identical implementation and tests. Only this validation record changed
after the complete-suite run.

Final syntax compilation, Markdown structure/local-reference checks, candidate
whitespace checks and `git diff --check` passed. The pre-task historical
inventory was captured from current state: 4,340 paths, all preserved with
unchanged contents, sizes, modes and mtimes. The separate reconstruction-contract
candidate remains byte-identical. The index remains empty; no commit, push,
remote access or follow-on implementation was performed.

# Native footprint and canvas annotation remediation

Product baseline: `16315734ad3223cdf5c0729a2d513a3314b6d7fa`.
LaunchPlane baseline: `1d2a0cece656fdd87113faf0fefbf7dd678793da`.
Status: Development + Local Validation and self-review passed; awaiting
strictly read-only pre-commit review. Native acceptance remains pending.

## Production path and findings

`BeamPanel._boundary_changed` builds the existing physical launch, takes the
resolved interface quadratic form, converts its eigenvalues/eigenvectors to
laboratory contour points, and calls `LaunchPlaneWidget.set_resolved_preview`.
LaunchPlane applies the existing y sign conversion and creates a dashed path.
Units, centers, radii and orientation in the delivered geometry were correct.
The contour was behind the opaque aperture: z=-0.5 versus background z=0.
The companion LaunchPlane fix places it at z=1 below the arrows at z=10 and
uses a brighter cosmetic dashed stroke.

A second disappearance route was exposed by the production refresh regression:
`BeamPanel.set_aperture` replaces the scene/stack, which correctly invalidates
old derived contours, but did not redeliver the current preview. It now calls
`_boundary_changed` after the final replacement. This does not change requests,
scientific arrays or resolver calculations.

LaunchPlane's two top annotations used fixed-pixel text at physical anchors on
the same row. They could overlap as the scene scaled. The companion changes
replace those scene annotations with separate wrapping labels above the view.

## Independent presentation checks

The new regressions construct the real PR window and use its actual Beam panel,
physical resolver and LaunchPlane delivery path. No synthetic replacement for
the production slot is used.

- Normal incidence, center (0,0), external radii 20/20 µm: circle radius 20 µm,
  including crossings (+/-20,0) and (0,+/-20).
- External theta 45 degrees, phi 0: principal radii 20/cos(45)=28.284271247 µm
  along x and 20 µm along y.
- Phi 90 rotates the long axis to y; phi 37 checks an arbitrary orientation.
- Changing internal index from 2 to 2.4 does not change the face contour.
- Circular-beam roll preserves the ellipse; unequal normal radii 10/20 with
  roll 90 give the independently expected swapped axes around center (7,11).
- Geometry residuals and coordinate comparisons use absolute tolerance 1e-12
  (micrometres for coordinates; dimensionless for the ellipse equation).
- Paths remain attached and visible after real host updates, aperture reset,
  screen-preview refresh, fitting and resizing. Stacking is above the opaque
  aperture and below the external arrow/center items. Normal arrow displacement
  remains zero; oblique arrows retain the external azimuth and bounded length.

## Preliminary failures and ownership qualification

Initial focused tests exposed missing host redelivery after aperture replacement.
Additional failures and two diagnostic-process crashes arose while the test
inspected parentless graphics items with `parentItem()`. A minimal probe of this
PySide build showed `shiboken6.ownedByPython(aperture)` change from False to True
when that getter returned None. Dropping the inspection wrapper then removed the
aperture from the scene (45 items became 44). This was a test-induced ownership
transfer, not evidence of a production preview callback deleting the aperture.

The test now identifies the aperture by its graphics-item type without calling
that getter. Trial production reference retention and explicit deletion were
removed; neither workaround remains in the candidate. Failed-run logs are
retained outside the repositories. The final test still verifies actual scene
membership, stacking, pens, geometry, refresh and window destruction.

## Validation and protection

Tests use the existing development Python 3.12 environment, Qt offscreen,
isolated source trees on PYTHONPATH, disabled bytecode and pytest cache, and
external temporary Matplotlib/font caches. No installation or native GUI is
performed. The installed `lcprop-new-user` environment is inventoried and
protected byte-for-byte including file modes and mtimes.

- Product real-host focused regressions: 5 passed.
- Product physical-launch/preview/adapter/screen/request/experiment GUI set:
  112 passed.
- LaunchPlane focused module: 5 passed; affected editor/model set: 67 passed;
  complete suite: 98 passed.
- Complete Product suite: 1905 passed, 77 skipped in 532.90 s.
- Baseline control with unchanged LaunchPlane: the new normal-circle test fails
  specifically because the contour is below the opaque aperture; this is the
  expected control failure, not part of the passing candidate validation.

Reproducible commands, from the corresponding isolated repository, with
`QT_QPA_PLATFORM=offscreen`, `PYTHONDONTWRITEBYTECODE=1`, temporary
`MPLCONFIGDIR`/`XDG_CACHE_HOME`, and the isolated LaunchPlane and Product `src`
directories first on `PYTHONPATH`:

```text
python -m pytest -p no:cacheprovider tests/test_host_footprint_preview.py -q
python -m pytest -p no:cacheprovider tests/test_host_footprint_preview.py tests/test_physical_launch_resolver.py tests/test_beam_panel_launchplane.py tests/test_launchplane_adapter.py tests/test_input_screen_editor.py tests/test_image_input_display_parity.py tests/test_pr_gui_request.py tests/test_experiment_gui.py -q
python -m pytest -p no:cacheprovider tests/ -q
```

LaunchPlane commands use the same environment with its isolated `src`:

```text
python -m pytest -p no:cacheprovider tests/test_resolved_preview.py -q
python -m pytest -p no:cacheprovider tests/test_resolved_preview.py tests/test_launchpane.py tests/test_angles.py -q
python -m pytest -p no:cacheprovider tests/ -q
```

Development self-review: the diff is limited to three presentation-source
files, two regression modules and two development records across the pair.
No public method signature, serialized value, scientific implementation or
installed file changes. Syntax, Markdown/local links, whitespace and diff checks
pass. Candidate identities and source hashes bind the retained local-validation
logs; these UI checks are not commissioning or scientific evidence.

Fresh preflight inventory: 4,344 historical Product paths, fingerprint
`b7c510ac50b0754097be225abb4c4c634104c74806aaa12979626e374bf97627`.
This includes the active remediation prompt. Authoritative Product and
LaunchPlane remain at the stated baselines; Research remains at
`b947fac51722eafa971e756a2cc2053be065294d`. The untracked reconstruction contract
is excluded and preserved. No historical timestamps are restored.

## Scientific non-change and native follow-up

Only Product preview redelivery and LaunchPlane rendering/layout change.
Resolver/refraction, external/internal k, physical power, capture, screen
semantics, propagation/material physics, diagnostics, schemas, coherence and
continuation are unchanged. Focused launch remains deferred. No cluster/GPU,
reconstruction, carrier-power curves, P2A-2, scattering or fanning work occurs.

Native acceptance is not complete. After separately approved integration and
installation, visually verify only the 20 µm normal circle, the 45-degree
x-oriented ellipse, phi=90 rotation, and annotation readability. Check compact
and ordinary widths, resize/zoom and footprint/arrow coexistence. Do not install
these candidates or continue into other acceptance work without authorization.

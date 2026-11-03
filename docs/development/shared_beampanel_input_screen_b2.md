# Shared BeamPanel Input-Screen Editor — Stage B2

## Status

Stage B2 adds a material-neutral input-screen editor to LCProp's shared
`BeamPanel`. It is a GUI and launch-composition milestone. It does not connect
screens to a material workflow and does not change optical or material physics.

## Ownership

The shared GUI owns selection and editing in
`lcprop.gui.panels.input_screen_editor`. The editor constructs declarative B1
`ChannelLaunchElements`; `lcprop.optics.screens` remains authoritative for
screen types, preprocessing, placement validation, runtime-grid preparation,
and passive transmission. `lcprop.optics.launch.build_launch()` remains
authoritative for screen application and power accounting.

LaunchPlane continues to own beam definitions and beam editing. It does not
own raster preprocessing, screen placement, or transmitted-power semantics.
The shared editor contains no LC or PR material roles.

## Channel Mapping

The channel selector lists canonical enabled-channel indices and beam names.
Assignments are resolved only after disabled LaunchPlane beams are filtered.
They follow unchanged beams through reordering and follow one unambiguous
immutable LaunchPlane beam edit. Assignments are removed when their beam is
deleted or disabled; ambiguous stale assignments are rejected rather than
silently redirected.

`BeamPanel.launch_configuration()` returns the canonical pair

```text
BeamStack + tuple[ChannelLaunchElements, ...]
```

and `BeamPanel.launch_elements()` exposes the plan alone. The editor never
stores transformed runtime fields as its declarative result.

## Screen and Source Handling

Stage B2 exposes only `None` and `Intensity image`. An image screen uses the B1
`even_square_nearest_transparent_v1` compatibility policy, nearest-neighbor
resampling, intensity transmission of one outside the footprint, and rejection
when the footprint extends beyond the aperture.

The shared bounded Qt raster decoder preserves the Stage A behavior:

- supported raster formats are selected from a conservative allow-list;
- compressed input and decoded pixel counts are bounded;
- RGBA is converted deterministically to grayscale with alpha ignored;
- source bytes receive a SHA-256 provenance digest;
- only the basename, format, dimensions, checksum, and decoded grayscale are
  retained; and
- the preview is bounded independently of the numerical grid.

No approved packaged image catalog exists in Stage B2. The Standard selector
therefore remains disabled with an explanation, while User image remains
available. No assets were added.

## Preview and Power

The editor presents the prepared intensity transmission and the selected
channel's post-screen intensity. Preview uses the current beam definitions, a
bounded rectangular runtime grid selected by `BeamPanel`, the declarative B1
plan, and `build_launch()`. It never invokes material propagation and never
changes the simulation grid in response to source dimensions.

Incident power, transmitted power, and throughput come directly from the
discrete launch result. There is no post-screen renormalization. An identity
path has throughput one, an absorbing image has throughput below one, and
unassigned channels remain unchanged.

## Placement

The editor exposes physical width, height, center x, and center y. Width and
height remain independent, so rectangular sources and rectangular numerical
grids are supported. A large host-side source is resampled onto the explicitly
selected preview or execution grid and cannot select `Nx` or `Ny`.

## Host Enablement Policy

An editable screen must never be shown in a host that discards its plan.

- The LC GUI currently builds requests from `BeamStack` only. Its shared editor
  is visibly disabled and states that LC requests do not yet carry launch
  elements.
- The ordinary PR GUI requests also carry only `BeamStack`. Its editor is
  visibly disabled pending Stage B3.
- PR Image Amplification continues to use the existing Stage A controls. Stage
  B2 does not enable the Beam tab for that workflow or alter pump/signal roles.

The enabled editor path is available through the shared `BeamPanel` API for
standalone validation and for a future host that explicitly consumes the plan.

## Validation

Focused coverage verifies enabled-channel mapping, disabled/reordered/edited
beams, add/replace/remove behavior, bounded user-image decoding, empty standard
catalog behavior, rectangular placement and previews, out-of-aperture
rejection, large-source/grid independence, declarative output, actual discrete
power accounting, and the absence of material propagation during preview.
The B1 optics suite and existing LC/PR BeamPanel and GUI suites remain the
scientific and compatibility regression gates.

## Deferred Work

Stage B3 will decide how PR Image Amplification selects material-specific
pump/signal roles and carries shared launch elements into its request. This
stage does not remove the Stage A image controls, add persistence or remote
transport, expose phase/complex screens, change LaunchPlane, or enable screens
in LC execution.

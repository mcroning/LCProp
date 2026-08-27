# PR Image Amplification Persistence and Transverse-View Zoom

## Scope

This milestone removes the C1-era restriction that prevented Image
Amplification experiments from using the shared **Save Experiment** and **Open
Experiment** actions. It also adds presentation-only zoom and pan controls to
the shared transverse image viewer. Slurm execution, remote transport, result
persistence, checkpoints, and scientific algorithms remain outside this
milestone.

## Persisted Experiment Contract

Image Amplification is stored as the registered composite operation
`pr_image_amplification`. Its material-owned payload records:

- the selected ordinary PR base operation and its complete request;
- the canonical beam stack and ordered channel launch elements;
- the pump and signal indices in canonical enabled-channel order; and
- the declarative intensity-screen source, placement, inversion,
  interpretation, sampling, and preprocessing policy.

The composite payload stores the launch-element plan once, under its
`LaunchConfiguration`, and restores that same immutable plan onto the decoded
base request. This avoids duplicating embedded image bytes while retaining the
base operation's ordinary runtime request semantics.

Loading rejects an unavailable base operation and invalid, identical, or
out-of-range pump/signal roles. The GUI restores the Image Amplification mode,
base algorithm, beams, screens, roles, material, grid, evolution controls, and
backend policy, then rebuilds the request and requires exact dataclass equality
with the decoded request.

The PR request-payload schema is version 2. Version 2 adds ordered
`launch_elements` to ordinary PR requests and defines the Image Amplification
composite payload. Version-1 ordinary PR files remain readable and are
interpreted as having an empty launch plan. Unsupported future versions fail
explicitly.

Runtime state is not part of an experiment definition. Prepared optical
fields, material fields, checkpoint state, result arrays, and output products
remain rejected or outside the experiment envelope.

## Portable User Images

For a user-selected raster, the experiment embeds the original encoded file
bytes rather than relying on its workstation path. The payload also records:

- source basename and display name;
- SHA-256 of the encoded bytes;
- encoded format;
- decoded width, height, and mode;
- alpha and preprocessing policies;
- canonical grayscale dtype, shape, bytes, and SHA-256; and
- stable asset ID when a future packaged image supplies one.

Both encoded and canonical grayscale checksums are verified while loading.
Consequently, deleting or moving the original image after saving does not
prevent the experiment from being reopened. Corrupt embedded bytes are
rejected before scientific execution. The standard-image catalog remains
empty; no assets were added.

## Transverse Image Navigation

The shared `ImageView` now supports the following physical-coordinate
interactions:

- scroll up/down zooms in/out about the cursor;
- right-button drag pans the current view;
- **Fit** restores the selected product's recommended framing extent; and
- **Full Aperture** restores the complete retained field extent.

Left-click selection retains its established behavior. Crosshair coordinates,
x-z/y-z selection synchronization, rectangular-grid aspect, axis orientation,
and physical coordinate labels are unchanged. Zoom and pan alter only axes
limits: they do not mutate field arrays or recompute color limits.

Selecting another product resets the shared viewer to that product's canonical
recommended extent, or to its full physical extent when no recommendation is
present. Per-product zoom history is intentionally not retained.

## Validation

The local validation covers all three currently compatible Image
Amplification base operations: reduced time-dependent, legacy static, and
canonical transverse static. It exercises exact request round trips, actual GUI
Save/Open buttons, original-file deletion, source and grayscale checksum
validation, launch placement and inversion, roles, base-operation selection,
backend/precision settings, runtime-state rejection, legacy version-1 ordinary
PR files, unsupported operations, and malformed roles.

Viewer regressions exercise rectangular and offset fields, cursor-centered
zoom, physical-coordinate pan, Fit, Full Aperture, product switching,
left-click selection, selection guides, and array immutability.

## Scientific Non-Change

This milestone changes declarative experiment serialization and shared result
presentation only. It does not change PR equations, material solvers,
convergence, optical propagation, launch-screen application, carrier
isolation, back-propagation, reconstruction, gain definitions, result arrays,
or optical-power semantics.

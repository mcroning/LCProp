# PR results, progress, MPR, and TD visualization

## Scope and outcome

This milestone adds presentation and retention products around the committed
eight-cell PR model matrix. It does not change a material equation, optical
propagator, integrator, scattering model, launch, boundary treatment, carrier
diagnostic, or Image Amplification calculation.

The primary volume interface is now a linked orthogonal MPR view. Its x-y,
x-z, and y-z planes share one `(x, y, z)` cursor. Mouse selection and sliders
both update that cursor, and labels use the coordinates belonging to the
selected field rather than assuming that all fields have the full grid.
Full results use the authoritative final optical-intensity volume. Fast results
use an explicitly labeled, visualization-only preview.

## Why the existing Fast cuts could be absent

The retention-to-GUI audit covered reduced static, reduced TD,
full-transverse static, and both full-transverse TD responses. For payloads
produced by the cut-bearing codecs committed in `f5569b2`, the chain is
complete: the workflow/source is projected by the transport codec, reconstructed
into the result, converted into the two named product fields, and offered by
the longitudinal selector. Local and Slurm reconstruction use the same product
adapter. The observed no-cut result therefore belonged to the supported older
payload generation from before `f5569b2`; it contained neither cut arrays nor
their coordinates. It was not a loss in the current GUI wiring.

The GUI and codecs now exercise three explicit compatibility states:

- older Fast payload: no cuts and no preview, with an unavailable-volume message;
- intermediate Fast payload: exact nearest-zero cuts but no preview;
- current Fast payload: exact cuts plus an interactive preview.

The exact cuts remain separate quantitative fields. They are full transverse
resolution at the actual samples nearest x=0 and y=0 and are not replaced by
arbitrary preview slices.

## Fast preview policy

The preview is generated from the final accepted PR-driving intensity while
that complete volume is available. Uniform material background is removed and
the launch peak-intensity reference is restored before reduction. Each retained
z plane is an actual computed plane; no longitudinal averaging occurs. Uniformly
spaced original z indices, including both endpoints, are retained. Within each
plane, deterministic contiguous x-y blocks are reduced by arithmetic mean.

The preview is float32 and bounded to 4 MiB. Each transverse dimension is at
most 96 samples and the z count is at most 113. Metadata records original and
requested grids, spacing, preview coordinates, retained z indices, x/y block
bounds, reduction and normalization conventions, dtype, payload size, byte
budget, quantity and unit, and `visualization_only: true`.

Representative raw preview overhead is:

| Authoritative `(Nx, Ny, Nz)` | Preview `(Nz, Nx, Ny)` | Raw overhead |
| --- | --- | ---: |
| 256 x 256 x 100 | 100 x 96 x 96 | 3.516 MiB |
| 512 x 512 x 400 | 113 x 96 x 96 | 3.973 MiB |
| illustrative 1024 x 1024 x 1000 | 113 x 96 x 96 | 3.973 MiB |

The cap is independent of authoritative precision and does not downcast an
authoritative array.

## TD products

Fast TD retains the final accepted 3-D preview, exact final nearest-zero cuts,
and real scalar observations. It never retains a material-time-by-z-by-x-by-y
preview history. Full TD retains the final authoritative source/intensity
volume, with propagation z kept distinct from material time.

At accepted material intervals, both reduced and full-transverse workflows
record material-state-change RMS. Nonlinear cells additionally record minimum
normalized carrier density; linearized cells do not fabricate that curve.
No curve is synthesized from endpoints. Exact-modal cells likewise receive no
invented iterative diagnostics.

When a progress observer exists (the normal local GUI and headless Slurm
executor paths), at most 36 accepted-time optical-intensity frames are selected,
including endpoints. Each frame uses contiguous block averaging to at most
128 x 128, and one global `[0, max]` color scale is used for the whole movie.
The preferred artifact is H.264 MP4 at 6 fps. Its raw grayscale input is at
most 0.563 MiB before compression. Metadata records cadence, retained indices
and material times, original/preview grids, reduction, fixed limits, quantity,
unit, codec/container, and visualization-only status. Encoding failure yields
a warning and leaves the completed scientific result and every other product
intact. The GUI offers a simple “Open downsampled TD preview” action rather
than implementing a custom player.

## Progress

Normal GUI console output is grouped by phase and ten-percent buckets. Status
labels still update for every callback, while starts, phase transitions,
warnings, replay, result preparation, and final status remain visible. This
prevents slice-by-slice and replay-slice floods without removing solver
diagnostics.

The headless executor atomically replaces a compact `progress.json`. Scientific
arrays are deliberately excluded. The file identifies schema, workflow,
phase/message, completed and total units, coordinate and elapsed time. Slurm
polling reads it only during the runner's existing polling cadence, tolerates a
missing or partially observed file, and exposes initialization, workflow
progress, packaging, retrieval, reconstruction, and completion through the
existing remote status channel. Progress writes are best-effort: creation,
writing, or atomic replacement failure is contained and cannot change the
scientific result, packaging outcome, cancellation semantics, or original
failure report. Temporary progress files are removed best-effort after a failed
update. Cancellation semantics are unchanged.

## Transport compatibility

Request codecs are unchanged. Result codecs advance as follows:

- reduced static: 2 to 3;
- reduced TD: 3 to 4;
- full-transverse static: 3 to 4;
- full-transverse TD: 2 to 3.

Each codec explicitly accepts its immediately previous result version and
defaults absent preview/movie/history fields to unavailable. Existing supported
legacy request migrations remain unchanged. Current payloads store MP4 bytes
as an NPZ array, and Fast payloads similarly store the float32 volume preview
there rather than as large JSON values. Unsupported future versions remain
rejected by the transport envelope.

## Scientific non-change

Paired runs with and without visualization callbacks produce bitwise-identical
accepted material states, final optical fields, and final driving-intensity
volumes. The new work observes, copies, averages, encodes, presents, and reports
already computed state only. Preview and movie products are non-authoritative;
the exact Fast cuts and all Full arrays retain their established quantitative
meaning.

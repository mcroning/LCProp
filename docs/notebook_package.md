# Optional installed LC notebook interface (0.3.0)

In an environment containing this candidate distribution and its `colab` extra:

```python
from IPython.display import display
from lcprop.notebook.interactive_lc import InteractiveLC
app = InteractiveLC()
display(app.widget)
```

Initialization samples the launch only. Press Run explicitly to solve. No helper
download, checkout, Qt, LaunchPlane or CuPy is required for canonical experiments.
The existing Configure label, voltage/beam controls, 120/180/300-second budgets,
immutable completed snapshots and exports are preserved. GUI editor metadata remains
subject to the separate LaunchPlane compatibility contract. No new scientific claim.

`lcprop.notebook.lc_static_demo` holds the unchanged shared preset and diagnostics,
and the existing standalone comparison entry point. The files in `examples/` are
compatibility wrappers; use them with this candidate installed, not an older wheel.
The historical Git-pinned bootstrap/notebook remains unchanged and uses its immutable
older helpers; do not substitute these wrappers into that historical hash contract.

The `colab` extra provides Matplotlib, ipywidgets and IPython. Core scientific imports
do not require those packages. Jupyter/Colab kernel provisioning is separate.
This guide describes version 0.3.0. For release qualification, install the verified
candidate wheel with its `colab` extra. Indexed installation requires publication
of that version to the selected package index.
The source distribution includes runtime source/data and this guide, both wrappers,
and the self-contained `tests/test_notebook_package.py` suite. Repository-only tests
and protected qualification artifacts are intentionally excluded from the sdist.

## Development candidate: one or two beams

The BeamPlane-integration candidate extends the packaged controller; these controls
are **not in the published 0.3.0 wheel or its pinned Colab notebook**. A separately
qualified future package release is needed before hosted users receive them. The
installation and launch structure above remains the same; no external helper is used.

The unchanged qualified preset starts with one beam. **Include Beam 2** adds an
exploratory Gaussian at x=20 µm, y=0, with 10 µm principal radii and 1 mW incident
power. It does not redistribute Beam 1 power. Position, radii, incidence, azimuth,
profile roll, wavelength, external index and phase are numeric controls. Angles are
in radians. Radii are beam-normal 1/e field radii (1/e² intensity radii). Beam 2 phase
is relative to Beam 1 at their respective beam-axis/entrance-plane intersections.
Removing Beam 2 from the draft is distinct from setting its power to zero.

LC requires the same wavelength for both beams. Enter matching wavelengths explicitly;
the interface never changes the other beam's wavelength silently. Existing Product
validation enforces positive radii, valid angles, nonnegative powers, positive total
power, and launch sampling. Invalid drafts retain the previous completed result.

**Preserve saved groups** retains the request's existing grouping, including legacy
group semantics. A newly added beam normally starts in a distinct group; when adding
to a legacy experiment its legacy stack mode is retained. The visible effective-group
list identifies the actual grouping. **Coherent** assigns a shared explicit group;
**Incoherent** assigns distinct explicit groups. These choices set canonical group
identities rather than relying on the legacy stack flag. Relative phase affects
interference within a group. The preview uses the engine's combined normalized
intensity, not a frontend renormalization or a physical-irradiance conversion.

Only Run executes propagation/material evolution. Additional beams, changed geometry,
and changed power remain exploratory, even when validation succeeds. Existing voltage,
progress, 120/180/300-second budgets and convergence tolerances are unchanged.
Restoring the preset resets every draft control without clearing the completed snapshot.

Results show total optical intensity and the shared director reorientation. Longitudinal
cuts use the nearest sampled coordinates to **Beam 1 of the completed request**.
**Combined RMS width** describes the entire intensity distribution, including any
interference, not either beam independently. There is no reconstruction of per-beam
longitudinal volumes. Plots and exports always use the completed snapshot, never later
control edits. Failure or timeout retains the previous completed snapshot and plots.

Canonical scientific-only one/two-beam LC static experiments round-trip within the
fixed grid/material/solver envelope. Names, channel order, phases, groups and loaded
values are retained. Desktop files containing optional LaunchPlane editor metadata
still require its validator; this interface does not strip that metadata or add Qt.
Desktop GUI representability is narrower than all native requests, so universal
editor-state round trips are not promised. Disabled editor beams are not scientific
channels. Existing scientific serialization and exports are unchanged.

Local validation is distinct from hosted acceptance. Check wrapping controls, phase
and coherence feedback, result ownership and export in a fresh hosted runtime after a
future authorized deployment. Execution remains synchronous and arbitrary exploratory
configurations are not convergence-qualified or evidence of soliton formation.

### Multibeam presentation refinement (development candidate)

**Beam coherence** appears above the individual beam controls. The three modes and
initial selection are unchanged. A wrapping text line repeats the selected option
so it remains readable when the native dropdown is narrow. Beam 2's **Relative phase
(rad)** means Beam 2 minus Beam 1 at their respective beam-axis/entrance-plane
intersections; the tooltip retains this definition.

The completed-run summary and exported plot titles derive coherence and relative
phase from the immutable completed request. They do not follow later draft edits.
Explicit saved groups take precedence over the legacy stack flag. Distinct groups
are labeled incoherent between groups, with no inter-group intensity effect from
relative phase. A single beam has no relative phase. The request does not record
which dropdown option produced it, so the summary does not claim that “Preserve
saved groups” was selected. True partially coherent/mixed grouping requires more
than two beams and is not newly enabled by these presentation changes.

### Persistent graphical positioning (Stage 2C development candidate)

A compatible active ipympl backend selects one persistent interactive canvas; otherwise a static launch preview is used. This feature is not in published LCProp
0.3.0. Use a kernel importing the reviewed development candidate; do not expect the
production-pinned Colab notebook to load it.

For an isolated local Jupyter test environment, optionally install
`python -m pip install "ipympl==0.10.0"` in that kernel environment. JupyterLab 4
requires the matching `jupyter-matplotlib` frontend supplied by ipympl; if server
and kernel environments differ, verify both. Do not alter `lcprop-new-user` or
system-wide Jupyter configuration for this test. Export existing results before
restarting the kernel. In the fresh kernel, **before creating the controller**, run:

```python
%matplotlib widget
```

Then run the existing reusable local initialization cell. There is no graphical-positioning checkbox. No solver runs on activation or drag. The controller
requires explicit backend activation because importing ipympl otherwise switches
the global Matplotlib backend. It does not silently change the backend. Missing
ipympl/setup leaves numeric editing and static preview available.

Select **Active Beam: Beam 1 / Beam 2**, then **Action: Move / Tilt**.
Move drags the center circle; Tilt drags the square arrow endpoint. Coincident
centers/endpoints remain unambiguous through explicit beam selection. At normal
incidence the tilt endpoint coincides with the center. Arrows show the internal
refracted entrance-to-exit ray across the configured propagation length, not an
external ray or a rescaled display arrow. Off-aperture endpoints may need numeric
angle editing; they are not shortened or silently clamped.

Numeric controls remain unchanged during motion. Optional **Live draft launch
(throttled)** samples a detached canonical request at most four times per second
and 20 times per gesture. Each provisional image identifies its target position;
it may lag the current marker. Samples exceeding 100 ms or rejected samples fall
back to marker-only movement for that gesture. Uncheck live preview for marker-only
operation. Escape/leave/cancel restores the last committed draft image. Release
atomically updates the selected position or incidence/azimuth pair and performs
one final canonical preview/advisory refresh. No propagation runs during editing.

Positions remain physical µm without snapping or y inversion. Tilt uses LCProp's
canonical inverse refraction mapping, retaining signed-theta/azimuth conventions;
unreachable external directions are rejected. Centers, radii, power, wavelength,
roll, phase and coherence remain untouched by Tilt. Numeric off-aperture values
remain permitted with the existing advisories.

Run, load and reset cancel gestures while retaining the same canvas. Events while busy or disposed are ignored; complete delayed gestures after editing resumes may change the draft only. Reuse the
same controller on notebook initialization reruns. Before intentionally replacing
it, call `demo.close()` to retire its interaction callbacks, then create/display the
replacement. Rebinding a Python variable cannot automatically close an older visible
widget; do not leave multiple active controllers in different cells.

Local Safari checks still required: move and tilt both beams, test overlap selection, compare
Python numeric values with marker positions, cancel at bounds, retain the same canvas across reset/load,
and rerun initialization without duplicating views or losing a completed snapshot.
Do not run the solver merely to test dragging. Optional Colab qualification is deferred;
its custom widget manager must also be enabled. No ipympl dependency is mandatory and
no package metadata or production release is changed by this candidate.

Run executes a detached, validated **kernel-accepted** request. Browser edits not yet
synchronized to Python are outside that guarantee; finish numeric submission or
mouse release before clicking Run. The completed result provides **View executed
parameters**, including the full frozen request and its SHA-256. Later editing does
not change completed plots or exports. A consumed Run button endpoint is replaced
with a fresh Run button after completion/failure; buffered duplicate clicks on the
old endpoint cannot execute again. There is no Prepare/Review or confirmation step.

### Unified launch preview (Stage 2D candidate)

Static and optional interactive previews now share intensity artists, labeled beam
centers, internal-ray-to-exit arrows, and active-beam highlighting. Active Beam
also selects the highlight in static mode; Move/Tilt mouse gestures require ipympl.
Zero-tilt arrows have zero length, with the endpoint square at the beam center.
All previews use the canonical combined launch intensity and coherence groups.
Numeric edits refresh either preview, without propagation. The static widget uses
complete PNG images; it does not depend on differential canvas frames.

For the static fallback, use an inline/Agg Matplotlib backend before creating a
controller. No ipympl installation is required. Export needed results before
restarting a kernel to change backends. Safari's intermittent interactive-frame
corruption remains unresolved; shared rendering does not claim to fix browser
frame delivery or painting. Completed-run figures and exports remain separate
from this editable launch preview.

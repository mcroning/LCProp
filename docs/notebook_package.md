# LCProp 0.4.0 notebook interface

The 0.4.0 runtime is byte-identical to the qualified 0.4.0rc1 implementation.
Both `from lcprop.notebook import InteractiveLC` and
`from lcprop.notebook.interactive_lc import InteractiveLC` are supported.
Only Run executes propagation. Initialization and editing sample the canonical launch.

## Installation and optional dependencies

Once 0.4.0 is published on production PyPI:

```python
%pip install "lcprop[colab]==0.4.0"
```

Before publication, use only separately verified candidate artifacts for release
qualification; do not substitute another version or a source checkout. The `colab`
extra includes Matplotlib, ipywidgets, IPython and ipympl>=0.10.0,<0.11. Core and GUI
extras do not depend on ipympl. Static plotting needs Matplotlib/ipywidgets/IPython
but no ipympl. Core imports need none of these or Qt/CuPy/LaunchPlane.
Jupyter/Colab kernel and frontend provisioning is separate from package installation.

## Hosted Colab: install, restart, verify, initialize

1. Use a hosted CPU runtime and install as above.
2. After first installation, select **Runtime → Restart session**, not disconnect/delete.
   Export any needed completed records first. No automatic restart or module reload.
3. Rerun this small verification cell after restart:

```python
from importlib import metadata
import subprocess, sys
assert metadata.version("lcprop") == "0.4.0"
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
```

4. Enable the custom widget manager and select the backend before creating the controller:

```python
from google.colab import output
output.enable_custom_widget_manager()
%matplotlib widget
```

5. Initialize once and reuse the controller. Colab's accordion view was unreliable
   in the qualified environment, so display its existing sections directly. No second
   controller or scientific state is created:

```python
import ipywidgets as W
from IPython.display import display, clear_output
from lcprop.notebook import InteractiveLC

if "app" not in globals():
    app = InteractiveLC()
    sections = (app.configure.children[0], *app.widget.children[1:])
    def expand_layout(widget):
        if widget is app.preview_host or isinstance(widget, W.Output):
            return
        if isinstance(widget, (W.Box, W.HTML)):
            widget.layout.height = "auto"
            widget.layout.max_height = "none"
            widget.layout.overflow = "visible"
            widget.layout.flex_shrink = "0"
        for child in getattr(widget, "children", ()):
            expand_layout(child)
    for section in sections:
        expand_layout(section)
    view = W.VBox(sections, layout=W.Layout(
        width="100%", max_width="1000px", height="auto", overflow="visible"))
if app.disposed:
    raise RuntimeError("Export needed records and restart before replacing this controller.")
clear_output(wait=True)
display(view)
```

Keep this in one initialization cell; do not create additional views in other cells.
Rerunning it preserves draft controls and completed snapshots. Ordinary initialization
reruns do not require another restart. Package replacement or backend changes require
exporting records and starting a new compatible session.

For a static fallback, export records and restart, select `%matplotlib inline` instead
of `%matplotlib widget`, then initialize. Numeric editing, centers, internal-ray arrows
and combined intensity remain available; mouse interaction does not. Static success
does not qualify the interactive frontend. The accepted production 0.3.0 notebook is
still pinned; this guide does not replace it or silently install 0.4.0 through it.

## Local Jupyter

Install into the active kernel environment, restart after installation, and select
`%matplotlib widget` before controller creation. Use `%matplotlib inline` for static
rendering. Local interfaces may display `app.widget` normally. Do not switch backends
beneath a live controller. If server/kernel environments differ, verify matching
ipympl frontend support. No external helper download or repository checkout is required.

## Beam controls and graphical editing

The qualified preset starts with one beam. Include Beam 2 adds a Gaussian centered
at (20, 0) µm, radii (10, 10) µm and power 1 mW, without redistributing Beam 1 power.
Radii are beam-normal 1/e field (1/e² intensity); angles are radians. Both beams must
use the same wavelength. Native validation retains positive radii, nonnegative powers,
positive total power and valid geometry. Modified settings remain exploratory.

Beam coherence offers Preserve saved groups, Coherent and Incoherent. Explicit saved
groups take precedence over the legacy stack mode; effective groups are displayed.
Beam 2 relative phase is Beam 2 minus Beam 1 at their respective beam-axis/entrance-plane
intersections. Phase changes interference only within shared groups. The preview uses
canonical combined normalization, never a second frontend beam model.

Active Beam selects the beam, including overlapping centers. Move drags its center;
Tilt drags its square endpoint. Arrows depict internal refracted entrance-to-exit rays,
not external incidence vectors or phase gradients. Positions use physical µm without
snapping or axis inversion. Unreachable graphical gestures cancel; off-aperture numeric
positions remain subject to existing advisories. No resizing or roll handles are added.

Optional Live draft launch updates are throttled; slow previews fall back to marker-only
motion. Release commits the numeric controls and final canonical preview. Run/load/reset
cancel gestures; busy/disposed events are ignored. Delayed complete gestures after editing
resumes may change only the draft. Only explicit Run can start scientific execution.

## Completed records and scientific scope

Run freezes the complete validated kernel-accepted request. Finish numeric submission
or mouse release before clicking Run; unsynchronized browser edits are not included in
its atomicity guarantee. View executed parameters exposes the complete frozen request
and SHA-256. Completed arrays, diagnostics, metadata, plots and exports do not follow
later draft edits. Failure or timeout retains the previous completed record. Budgets
are elapsed-time checks at progress callbacks, not hard process kill timers.

Longitudinal cuts use Beam 1 of the completed request. Combined RMS width describes
the entire distribution including interference, not separate beam widths. Results use
total intensity and shared director response. Self-consistent slice convergence does
not establish stationary soliton formation or stability.

RC1 hosted qualification passed one bounded single-beam CPU run: all nine retained
arrays within original tolerances, 100/100 slices converged in 35.38 seconds, and
byte-identical exports after draft edits. Graphical Beam 2/Move/Tilt/numeric editing
passed separately. Additional multibeam scientific configurations and arbitrary
parameters are not qualified by those results. The manual Colab restart is still required.

The production notebook should remain simple: installation, restart/verification,
initialization and explicit Run. Artifact manifests, reference arrays and qualification
helpers belong in separate validation notebooks, not the ordinary workflow. The
source distribution includes this guide, compatibility example wrappers and the bounded
package test module; protected research evidence is excluded.

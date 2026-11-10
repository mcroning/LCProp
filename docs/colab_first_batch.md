# Bounded LC and PR notebook interface

`InteractiveBatch` is a separate optional notebook frontend. The original
`InteractiveLC` teaching controller retains its preset and file rules. The
first batch uses the existing LC static and reduced nonlinear PR TD operations.
This is a bounded CPU preview. Earlier candidate builds passed the two local
preset comparisons and hosted checks for Stop with Refresh status, the PR
preset, single-canvas redraws and native ZIP reopen. These checks cover the
recorded presets and hosted environment. The [release notes](releases/0.5.0rc1.md) distinguish earlier preview checks from exact-wheel verification.

## Start in a fresh CPU kernel

Use the exact candidate wheel and its separately recorded SHA-256. Install its
`colab` extra into the new kernel, restart after installation and check the
installed version, wheel origin and dependencies. The companion
[notebook](../notebooks/colab_first_batch.ipynb) provides these cells. For a
local Jupyter kernel, use the same wheel and choose the widget backend before
constructing the controller. Colab requires the custom widget manager.

Rerun the display cell to show the existing controller. Export needed files
before restarting the kernel. In an active kernel, call
`await lcprop_batch.dispose()` before intentionally replacing the controller.
Disposal waits for the worker to acknowledge cancellation.

## Configure and run

The fixed LC grid is 128 × 128 with 100 z slices and the existing bounded
local self-consistent preset. Edit voltage and one or two enabled Gaussian beams using BeamPlane.
The fixed PR grid is 64 × 48 with 20 slices. Edit physical dark and uniform
irradiance, gain-length product, one to three normalized material steps, and
normalized dt up to 0.001. The integrator is semi-implicit trapezoidal.
Both editors preserve signed external tilt and beam phase, and allow explicit
coherence regrouping. Wavelength is fixed at 0.633 µm.

The draft input preview samples the entrance plane. Only **Run** admits
propagation. Finish submitting browser edits before Run. The kernel freezes
the validated request, disables draft edits and runs one background worker.
**Stop** requests cancellation at the next safe workflow boundary and stays
busy until acknowledgement. If Colab misses a background status update,
use **Refresh status** to request the current kernel state. The elapsed budgets are cooperative. A stopped
attempt has its own view and leaves the previous completed result available.

## Edit beams in BeamPlane

Select an active beam by name or by clicking its center in **Move** mode.
In **Tilt** mode, select its internal-ray endpoint. If handles overlap, choose
a beam from the overlap menu, then drag its handle. **Move** changes the
entrance center. **Tilt** uses the square endpoint at the exit face to edit
signed external angle and azimuth. Tilt switches to **Fit**. For short cells
such as the PR preset, Fit zooms to the reachable exit region. Release the
square inside the dashed circle. Out-of-range targets leave the draft unchanged. The numeric controls display the same draft.
A drag marker follows the pointer. The draft and combined intensity update on
release. Escape or leaving the canvas cancels the gesture. **Fit** frames
centers and ray endpoints. **Full aperture** shows the configured entrance face.

Add or duplicate to create a named beam. A new object starts disabled if two
are already enabled. Use **Enabled** or **Delete** to manage the stack. At
least one beam must stay enabled. Disabled objects remain editable, and the
draft file preserves their intent. Notebook-only disabled-object metadata
is for notebook round trips. Existing compatible LaunchPlane presentation
files retain the LaunchPlane format and require the pinned companion below.

The canvas needs the notebook widget backend. Numeric editing remains available
without it. The editor supports collimated Gaussian beams. Screen and other
object editing are deferred, and unsupported editable files are rejected
before replacing the draft. Completed package views retain their existing
read-only behavior.

## Browse and exchange files

Select Draft preview, Completed or Stopped / partial. Select a retained
field, XY/XZ/YZ plane and sample index when available. Missing cuts are
reported. Physical aspect uses matching physical scales and Square fits a
square plot. Executed request and diagnostics show the admitted hash and
provenance.

In hosted Colab, run this in a separate code cell and select one experiment
JSON or completed ZIP in Colab's native picker:

```python
lcprop_batch.open_from_colab()
```

The picker waits for a selection in the cell. This is the supported hosted
upload path for Colab. In local Jupyter, use Upload
and then Load / reopen under Save / load / export.
Batch-compatible desktop files can retain LaunchPlane schema-4 beam editor
metadata and local NumPy float64 execution intent. The notebook uses only the
headless LaunchPlane model and serializer. For editable desktop interchange only, after wheel verification, stage the
pinned companion source in the fresh kernel's working directory:

```bash
git clone https://github.com/mcroning/LaunchPlane.git LaunchPlane-interchange
git -C LaunchPlane-interchange checkout --detach b71d125eb08074b0992dc849a5f4fc39e45eabc2
```

Before the interface cell, add that source directory to the kernel import path:

```python
from pathlib import Path
import sys

companion_source = Path("LaunchPlane-interchange/src").resolve(strict=True)
sys.path.insert(0, str(companion_source))
from launchplane.serialization import SCHEMA_VERSION
assert SCHEMA_VERSION == 4
```

This source-only setup leaves pip's installed dependency graph unchanged.
Use a fresh kernel so an earlier LaunchPlane import cannot shadow the pin.

The notebook preserves saved retrieval preference for desktop round trips.
Reduced TD runs use the same local full-result path regardless of that saved
preference. Enabled presentation beams follow edits, and disabled editor
objects are retained. Unsupported scientific inputs, unknown presentation
fields, remote intent and unsupported retrieval policies fail editable load
atomically. Keep the original file. A
verified completed package may reopen read-only even when its request falls
outside the editor. Uploads are bounded at 128 MiB. Completed export remains
bound to the admitted request after draft edits.

If **Download completed** does not start a browser download in Colab, create
a ZIP in the runtime with this separate cell:

```python
from pathlib import Path
from uuid import uuid4
from lcprop.notebook.batch_files import archive_completed

folder = Path("/content") / f"lcprop-completed-{uuid4().hex[:8]}"
zip_path = archive_completed(lcprop_batch.session.export_completed(folder))
print(zip_path.name)
```

In Colab's Files pane, refresh the list and choose **Download** from the
ZIP's file menu. To reopen it, use the native picker above. The Files-pane procedure is a workaround. Browser download has not been qualified. Local Jupyter reports the saved path. The interface does not
offer continuation or checkpoint editing.

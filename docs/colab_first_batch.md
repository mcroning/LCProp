# Bounded LC and PR notebook interface

The separate LC static and reduced nonlinear PR TD notebooks are optional
frontends for the calculation engine. Both default to CPU. See the
[release notes](releases/0.5.0rc1.md) for the qualification scope.

## Start in a fresh Colab runtime

1. Open the [LC notebook](../notebooks/lcprop_lc_colab.ipynb) or
   [PR notebook](../notebooks/lcprop_pr_colab.ipynb) and select a CPU runtime.
2. Obtain the wheel and notebooks from the same release staging set (or
   release assets when published). Compare all three SHA-256 values with the
   supplied manifest before use.
3. Upload the wheel using Colab's Files pane so it appears under `/content`.
   Run the notebook's install cell, then restart the session. The cell
   installs the pinned wheel and its `colab` extra, plus Jedi for Colab's
   IPython dependency. The Colab T4 preview uses Colab's supplied CuPy.
4. After restart, run **Verify installation** and then the interface cell.
   Verification checks the installed wheel's origin, version and dependency
   consistency. The interface enables Colab's custom widget manager.

Rerun the interface cell to redisplay the existing controller. Export any
needed files before restarting. To intentionally dispose of the controller,
use `await lcprop_lc.dispose()` in the LC notebook or
`await lcprop_pr.dispose()` in the PR notebook, then restart before creating
another controller. Disposal waits for the worker to acknowledge cancellation.

## Configure and run

The fixed LC grid is 128 × 128 with 100 z slices and the existing bounded
local self-consistent preset. Edit voltage and one or two enabled Gaussian beams using BeamPlane.
The PR preset is 64 × 48 with a 1 mm cell and 1 µm longitudinal spacing
(1,000 slices). Saved 20 µm preset drafts retain their original length. Edit physical dark and uniform
irradiance, gain-length product, one to three normalized material steps, and
normalized dt up to 0.001. The integrator is semi-implicit trapezoidal.
Both editors preserve signed external tilt and beam phase, and allow explicit
coherence regrouping. Wavelength is fixed at 0.633 µm.

The PR **Device** preference defaults to **Auto**. **CPU** selects NumPy
float64 directly. With the default notebook setup, Auto uses CPU and shows
why, and GPU requests are rejected before a run starts. LC stays CPU-only.
Draft files preserve the requested preference. The executed request and
completed package record the resolved backend and precision. An admitted run
that fails keeps the previous completed result available. Each new attempt
requires **Run**.

The draft input preview samples the entrance plane. Only **Run** admits
propagation. Finish submitting browser edits before Run. The kernel freezes
the validated request, disables draft edits and runs one background worker.
**Stop** requests cancellation at the next safe workflow boundary and stays
busy until acknowledgement. If Colab misses a background status update,
use **Refresh status** to request the current kernel state. The elapsed budgets are cooperative. A stopped
attempt has its own view and leaves the previous completed result available.

## Opt in to the Colab T4 PR preview

Use the [PR notebook](../notebooks/lcprop_pr_colab.ipynb) in a fresh Colab
T4 runtime. Verify and install the reviewed wheel with the Colab extra, then
restart the kernel. Use the CuPy supplied by the Colab runtime. The qualified
setup used a Tesla T4 with CUDA 13.0 and CuPy 14.0.1 (`cupy-cuda13x`).
The wheel's optional CUDA 12 extra was outside this qualification. Avoid
installing it alongside Colab's CUDA 13 CuPy.

Set `ENABLE_COLAB_T4 = True` in the display cell before creating the
controller. The **GPU** preference requires a successful T4 probe.
**Auto** uses the T4 when the probe succeeds and shows the reason when
it falls back to CPU. **CPU** bypasses the probe. An admitted GPU run
executes on that device with float64/complex128 and has no automatic
CPU retry. Export any needed files and restart before changing the opt-in
setting on an existing controller.

The bounded three-step PR fixtures at 20 µm and 1 mm agreed with CPU
within the predeclared comparison tolerances. The 1 mm test also verified
in-work Stop and CPU reopen of the GPU ZIP. Qualification covers those
fixtures and the recorded T4 environment. The runs did not establish a
performance advantage.

## Edit beams in BeamPlane

Select the active beam using the **Active beam** dropdown. Canvas clicks
preserve that selection. Drag the selected beam's center in **Move** mode
or its internal-ray endpoint in **Tilt** mode. Overlapping handles act on
the selected beam. **Move** changes the
entrance center. **Tilt** uses the square endpoint at the exit face to edit
signed external angle and azimuth. Tilt preserves the selected view. Full aperture is the default. For short cells
such as the earlier 20 µm PR preset, Fit zooms to the reachable exit region. Release the
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

In the LC notebook, select **Completed**, then choose **Intensity** or
**Director response (Δθ)**. The **XY**, **XZ**, and **YZ** buttons choose the
plane. The cut dropdown lists physical coordinates along the remaining axis.
The selected coordinate appears in the plot title. **Physical** preserves the axis scale.
Choose **Square** for a wider view of a long, narrow cell. The entrance preview
contains XY data only.

Select Draft preview, Completed or Stopped / partial. Select a retained
field, XY/XZ/YZ plane and sample index when available. Missing cuts are
reported. Physical aspect uses matching physical scales and Square fits a
square plot. Executed request and diagnostics show the admitted hash and
provenance.

In hosted Colab, run the command for your notebook in a separate code cell.
Select one experiment JSON or completed ZIP in Colab's native picker:

- LC notebook: `lcprop_lc.open_from_colab()`
- PR notebook: `lcprop_pr.open_from_colab()`

The picker waits for a selection in the cell. This is the supported hosted
upload path for Colab. In local Jupyter, use Upload
and then Load / reopen under Save / load / export.
Batch-compatible desktop files can retain LaunchPlane schema-4 beam editor
metadata. Compatible local PR drafts also retain Auto, CPU, or GPU
float64 execution intent. The notebook uses the headless LaunchPlane model
and serializer. For editable desktop interchange only, after wheel verification, stage the
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

Under **Save / load / export**, use **Download draft** for an experiment
JSON or **Download completed** for the completed-result ZIP. If the draft
download stalls, run this in a separate cell and download the JSON from
Colab's Files pane:

```python
from pathlib import Path
from uuid import uuid4

app = lcprop_lc  # In the PR notebook, use app = lcprop_pr.
draft_path = Path("/content") / f"lcprop-draft-{uuid4().hex[:8]}.lcprop.json"
app.session.save_draft(draft_path)
print(draft_path.name)
```

If **Download completed** does not start a browser download in Colab, create
a ZIP in the runtime with this separate cell:

```python
from pathlib import Path
from uuid import uuid4
from lcprop.notebook.batch_files import archive_completed

folder = Path("/content") / f"lcprop-completed-{uuid4().hex[:8]}"
app = lcprop_lc  # In the PR notebook, use app = lcprop_pr.
zip_path = archive_completed(app.session.export_completed(folder))
print(zip_path.name)
```

In Colab's Files pane, refresh the list and choose **Download** from the
ZIP's file menu. To reopen it, use the native picker above. The Files-pane procedure is a workaround. Browser download has not been qualified. Local Jupyter reports the saved path. The interface does not
offer continuation or checkpoint editing.

On narrow Colab panes the plot colorbar can be clipped. Close the Files pane
or widen the browser view. Desktop feature parity, saved LC result acceptance,
and additional GPU workflows remain deferred.

# Bounded LC and PR notebook interface

`InteractiveBatch` is a separate optional notebook frontend. The original
`InteractiveLC` teaching controller retains its preset and file rules. The
first batch uses the existing LC static and reduced nonlinear PR TD operations.
Hosted acceptance and bounded CPU acceptance are pending.

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
local self-consistent preset. Edit voltage and one or two Gaussian beams.
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

## Browse and exchange files

Select Draft preview, Completed or Stopped / partial. Select a retained
field, XY/XZ/YZ plane and sample index when available. Missing cuts are
reported. Physical aspect uses matching physical scales and Square fits a
square plot. Executed request and diagnostics show the admitted hash and
provenance.

Expand Save / load / export for draft JSON and completed result ZIP files.
Batch-compatible desktop files can retain LaunchPlane schema-4 beam editor
metadata and local NumPy float64 execution intent. The notebook uses only the
headless LaunchPlane model and serializer. After wheel verification, stage the
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
bound to the admitted request after draft edits. Colab downloads explicitly
to your device. Local Jupyter reports the saved path. The interface does not
offer continuation or checkpoint editing.

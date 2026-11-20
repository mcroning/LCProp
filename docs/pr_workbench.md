# Reduced PR workbench (source preview)

This additive interface is available from this source revision; it is not in the
published rc3 wheel. Existing release notebooks and downloads are unchanged.
LC has a separate interface. Full-transverse PR transport is outside this preview.

## Open

1. In a clean environment, install this isolated checkout with `python -m pip install '.[colab]'`.
   Select that environment as your notebook kernel. In Colab use a disposable CPU
   runtime and install a separately reviewed build of this revision; the published
   rc3 wheel does not provide `PRWorkbench`. No private setup kit is required.
2. Open `notebooks/pr_workbench_source.ipynb` and run **Open PR workbench**.
3. Use **Beams**, **Results**, or **Files**. **Open synthetic example** displays a
   small demonstration without propagation. Only **Run** executes the solver.

The opening cell performs no installation or automatic restart. After installing
or changing package versions, save needed files, restart the kernel, and rerun
Open PR workbench. Rerunning it in the same live kernel redisplays the existing
controller, drafts, and results. A disconnected browser can reconnect and rerun
it; a deleted or replaced runtime cannot recover unsaved state. Save before ending
one. An incompatible/disposed controller requires an explicit restart rather
than silently replacing unsaved work.

## Controls and interpretation

**Evolution** selects reduced x-only Static or time-dependent PR. Each route keeps
its own draft. Beam position, tilt, power, coherence and other supported values
are editable; grid, material, solver, and resource information are under
**Advanced settings**. Request/diagnostic text is collapsed separately.

The beam-editing image uses an independently sampled, bounded entrance grid.
It does not alter scientific sampling or make an invalid scientific grid runnable.
Run continues to use the public validator, resource admission and device policy.
CPU is the default. Auto may require fallback confirmation. GPU remains restricted
to the existing frozen reduced TD requests and qualified runtime checks; Static
requires CPU. This preview neither installs CuPy nor extends GPU qualification.

Run identity, progress, Stop request/acknowledgement and failures remain visible.
A finished Static solve is labeled converged only when its result says so. A
partial result is identified separately. TD completion is not presented as global
convergence. The included example is synthetic, not a scientific calculation.

Results reuse existing field choices and XY/XZ/YZ views when the selected product
contains those axes. Physical and Square aspects reuse existing plotting behavior.
An unavailable longitudinal product is explained rather than fabricated.

## Files and unchanged limits

In Colab upload an experiment JSON or completed ZIP with the **Files sidebar**,
then **Files → Refresh file list → Open selected file**. The list shows files in
the current working directory; no picker opens automatically. Download draft and
completed-result actions reuse existing codecs. An optional Drive panel is
available but never mounted simply to open the workbench.

Public policies stay unchanged: 256 MiB retained per result; 2.5 GiB working plus
resident results; 512 MiB runtime reserve; 60-second cooperative PR run budget;
128 MiB ZIP import/export ceiling. Actual available memory can prevent admission.
Stop and timeout are cooperative. Oversize results require a separately reviewed
storage workflow; this preview does not include the private split-backup helper.
No notebook autosave promises to preserve live arrays.

## Verification scope

Local tests use fake runners, mocked sampling, and synthetic saved results.
They cover progress/Stop/error, route switching, draft preservation, redisplay,
independent preview versus scientific admission, result labels, plots, file limits,
and exact nine-member ZIP round trips. No new propagation, GPU or hosted acceptance
is implied. A fresh Colab CPU UI check of a reviewed build remains separate.

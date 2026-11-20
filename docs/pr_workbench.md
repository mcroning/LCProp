# Reduced PR workbench (source preview)

This additive interface is available from this source revision; it is not in the
published rc3 wheel. Existing release notebooks and downloads are unchanged.
LC has a separate interface. Full-transverse PR transport is outside this preview.

## Open

1. Connect a disposable Colab CPU runtime.
2. For this unpublished preview, upload the matching **LCProp-PR-workbench-setup.zip** through
   the Files sidebar. Keep it zipped; no result ZIP or separate wheel is needed.
3. Open the supplied notebook and run **Open PR workbench**. If asked, choose
   **Runtime → Restart session** once and run that same cell again. Then use the
   workbench; **Files → Open synthetic example** needs no propagation.

The single cell verifies the setup ZIP and exact wheel automatically. Installation
is confined to that disposable runtime. The opening cell pins Matplotlib 3.11.2
for compatibility with the widget backend; interrupted dependency setup resumes
from the same cell. It asks for a restart rather than clearing
live state itself. Re-running with an existing workbench redisplays the same
controller before considering installation, preserving committed drafts/results.
After a browser disconnect, rerun Open. Deleted/replaced runtimes cannot recover
unsaved arrays: save before ending one. The opening help is included directly in
the notebook, without a relative web link. Drive is never required for setup.

Developers can instead install this isolated checkout with
`python -m pip install '.[colab]'` in a clean environment and call
`open_pr_workbench(globals(), hosted=False)` after selecting the widget backend.
The published rc3 wheel does not contain this source-preview interface. The setup
ZIP is staged locally with `tools/stage_pr_workbench.py`; it is not published.

## Controls and interpretation

**Evolution** selects reduced x-only Static or time-dependent PR. Each route keeps
its own draft. Beam position, tilt, power, coherence and other supported values
are editable; grid, material, solver, and resource information are under
**Advanced settings**. Request/diagnostic text is collapsed separately.

The beam-editing image uses an independently sampled, bounded entrance grid.
It does not alter scientific sampling or make an invalid scientific grid runnable.
Run continues to use the public validator, resource admission and device policy.
If Run is unavailable, a short explanation appears directly below the buttons,
including the estimated result size and limit when that is the blocker. Detailed
estimates and advisory warnings remain under Advanced settings.
CPU is the default. Auto may require fallback confirmation. GPU remains restricted
to the existing frozen reduced TD requests and qualified runtime checks; Static
requires CPU. This preview neither installs CuPy nor extends GPU qualification.

Run identity, progress, Stop request/acknowledgement and failures remain visible.
A finished Static solve is labeled converged only when its result says so. A
partial result is identified separately. TD completion is not presented as global
convergence. The included example is synthetic, not a scientific calculation.

Results use a compact canvas that fits a narrow Colab panel, and reuse existing field choices and XY/XZ/YZ views when the selected product
contains those axes. Physical and Square aspects reuse existing plotting behavior.
An unavailable longitudinal product is explained rather than fabricated.

## Files and unchanged limits

In Colab upload an experiment JSON or completed ZIP with the **Files sidebar**,
then **Files → Refresh workbench files → Saved file → Open selected file**.
The workbench refresh is distinct from Colab’s sidebar refresh. Opening Files
automatically rescans; incomplete uploads require a later refresh. The list shows files in
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

## Local frontend corrections after rc4

The working source adds explicit CPU/Auto/GPU help without a device probe or policy
change, responsive controls, and a complete selected-field caption. Space-charge
products whose existing quantity is `normalized_space_charge_field` and unit is
`1` are labelled **Normalized field**, with dimensionless/model-normalization help;
array values and saved metadata are unchanged. Long plot titles use short explicit
lines to avoid vertical-colorbar overlap. Beam numeric controls show six significant
digits; full stored values remain in the request and tooltip. Redisplay does not
round saved beam values. Invalid numeric edits retain the prior request and explain
the error. Other frontends retain their default widgets/plots.

The published rc4 setup download is linked when it matches the notebook’s pinned
kit. A newly staged development kit explicitly requires the supplied matching ZIP,
and never links a different published kit. These frontend fixes are local source
changes, not a mutation of rc4 assets; testing them in Colab requires a separately
built/reviewed development wheel and matching kit. No hosted acceptance is implied.

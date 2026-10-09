# Interactive LC CPU notebook — self-installing setup

Open the published notebook in a **fresh Colab CPU runtime** and run its single setup
cell. It retrieves pinned helpers, installs headless LCProp and presentation dependencies,
verifies 264 installed package files and helper SHA-256 identities, and opens the controls.
No wheel, script or ZIP upload is required. Setup does not run the solver; press **Run**
separately. If imports from an earlier session conflict, restart the runtime. A verified
setup rerun preserves the existing application and completed snapshot.

**Publication pending:** both pinned URLs currently return 404. No moving revision is
substituted. The eventual immutable link is:

```text
https://colab.research.google.com/github/mcroning/LCProp/blob/<PUBLISHED_NOTEBOOK_COMMIT>/notebooks/lc_static_interactive_cpu.ipynb
```

Replace the placeholder only after this notebook revision is committed and published.
This is a link template, not a working deployment badge. The notebook itself pins:

- Engine: `52b00928978a9ec7862a357d38327fd0b9ae857c`.
- Helpers: `d69ef2bd5b93022dc3a4db6cc7f764d066ce078d`.

Installation builds from the immutable engine archive; it is not claimed to reproduce
the Stage 5 wheel ZIP hash. Installed source identities must match the qualified wheel
manifest. NumPy/SciPy are engine dependencies; Matplotlib/ipywidgets are installed for
presentation. No Qt, LaunchPlane or CuPy extras. Dependency versions are recorded by
the existing environment provenance; fresh hosted dependency combinations still need
qualification. Ordinary local Jupyter should use a fresh disposable kernel environment:
running setup intentionally installs into that kernel, not a separate environment.

Edit numeric x/y position, beam-normal 1/e field radii (µm) or incident power (mW), plus applied voltage (V).
Only the launch preview updates. Press Run for one bounded self-consistent static
calculation. The scientific preset and convergence tolerances are unchanged; altered
beam configurations are exploratory and may fail validation or convergence.
The launch preview is normalized intensity, not irradiance: changing single-beam
power need not change its shape or magnitude. LCProp retains physical power for the
material response. No frontend normalization is applied.

The four longitudinal panels show retained optical intensity XZ/YZ and director-reorientation
XZ/YZ cuts, with propagation z horizontal. Slices use nearest grid positions to the
completed run’s beam center, with actual coordinates in titles. Width curves appear
below the four panels. Convergence is not evidence
of a stationary soliton. Export writes a **new directory** with a canonical experiment,
nine selected arrays, complete convergence diagnostics/environment, a regenerated
snapshot figure and checksum manifest. Export uses the most recent successful snapshot,
not later control changes. No snapshot means export is rejected. Failed runs retain
previous completed ownership. The selected NPZ is not a checkpoint/full result codec.

Load experiment accepts canonical LC static requests matching the single-beam preset
except the six editable values (position, radii, power and voltage). No rounding of loaded floats is intended. Other
workflows/grids/material/solver options are rejected rather than silently simplified.
Optional LaunchPlane editor payloads may require the dependency's model validation;
this interface does not strip or recreate them. Use a canonical scientific experiment
without that optional payload. The original file remains unchanged. For full desktop
editor fidelity or multibeam exploration use the desktop GUI/Python APIs.

This synchronous milestone has progress updates but no responsive Stop guarantee.
A busy notebook kernel may delay browser events. No background workers/custom JavaScript
are used. The selectable between-slice 120/180/300 s wall-time budget remains; do not retry
with changed physics after failure. Plan at least 1 GiB available RAM and several
minutes for slower CPU systems. Store one completed snapshot; the snapshot uses
immutable NPZ bytes and decodes arrays on demand, so temporary copies during plot/export
add memory. No solver is called for plotting/export.

Hosted interaction has **not** been tested by this implementation task. Local clean-
kernel rendering/callback tests establish local operation, not Colab browser behavior.
Download data explicitly before session expiration. No automatic Drive mounting.

## Stage 1B controls

Configure, Run and Results are vertically grouped. Labels sit above full-width controls
and rows wrap; preview images are capped at 375 CSS pixels and results at 900 CSS pixels; both
shrink to fit narrower output areas rather than stretching across wide screens. Existing sampling
advisories are visible and retained in completed diagnostics. No new advisory criteria.

Applied voltage defaults to 0.9144 V. A new Run has no prepared director or optical state;
the existing LCProp workflow constructs bias afresh from the new BiasSpec. Changes
are exploratory, not newly qualified physical presets. Restore qualified preset resets
all draft inputs without altering a completed snapshot. No implicit run on edit/reset.

## Stage 1C presentation

Launch preview is approximately 375 px square including axes/colorbar, with physical
XY aspect preserved. Paths are capped at 300 px; buttons are 140–170 px. Configure
can collapse (automatically after success) so Run and Results get the viewport.
The compact 2×2 longitudinal grid uses two shared colorbars: nonnegative intensity
and symmetric, signed Δθ in radians. Δθ subtracts the same completed run’s retained
theta_bias on each slice; absent/invalid baselines are rejected. No director solve
or optical replay is used. Display coordinates have two decimals and colorbar ticks
about three significant figures. Metadata and arrays keep full precision. Width
curves are below. Narrow windows may still require vertical scrolling for legibility.

The Run section offers an elapsed **wall-time budget** of 120 (default), 180 or
300 seconds. This is execution configuration, separate from the scientific request
and solver iteration limits. It is checked after each completed slice, after updating
progress, so a long slice can exceed the selected budget. Timeout reports elapsed
time, slice count and limit; convergence is not assessed. Prior completed plots and
exports remain available. A larger budget does not guarantee convergence for an
exploratory configuration.

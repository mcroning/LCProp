# Interactive LC notebook — production PyPI 0.3.0

Open `lc_static_pypi_cpu.ipynb` in ordinary Jupyter or Colab and select a CPU runtime.
Use **Run all**: the collapsed implementation cell defines setup, then the short
**Prepare LCProp interface** cell installs and initializes the interface. Neither
cell runs the scientific solver. Review the preview and press the interface's
**Run** button explicitly. No helper/wheel/ZIP upload, Git checkout or Drive mount.

This candidate is not published or hosted-qualified yet. To review in Colab, open
this exact notebook file manually. An immutable Open in Colab URL requires a later
reviewed commit/publication; no moving-branch link is substituted. The historical
`lc_static_interactive_cpu.ipynb` and its Git-pinned bootstrap are unchanged.

Colab's warning that a notebook was not authored by Google is the standard
third-party-code warning. Inspect the source and trust the publisher before running
it. Code executes with access to its runtime and storage you explicitly connect.

Setup installs `lcprop[colab]==0.3.0` through production PyPI in the current kernel.
Use a disposable environment for local review. Compatible preinstalled numerical
libraries are reused; they are not explicitly upgraded. If LCProp 0.3.0 is already
installed without extras, setup gives an actionable missing-dependency message
rather than changing the environment under a running interface. Version conflicts
between loaded modules and installed metadata require a user restart; no automatic
restart occurs. Import failures and pip failures include bounded diagnostics.

Before initializing, setup retrieves the production wheel, verifies its fixed
qualified SHA-256, and compares all 267 installed runtime files byte-for-byte.
The wheel is checked in memory, not executed as an external helper. Loaded LCProp
modules must resolve within that installed package. The setup prints package path,
version, wheel identity and numerical/notebook dependency versions. This supplements
the packaged export's environment provenance; it does not alter exported requests
or arrays. Arbitrary in-memory monkeypatching by other notebook code is outside
this integrity check's scope; use a clean kernel for reproducibility.

Rerunning setup reuses `app` and preserves its completed snapshot, controls, figures
and exports. A failed verification raises visibly and does not replace the existing
object. It does not retroactively disable controls already displayed by an earlier
successful setup; resolve the failure before further execution.

All controls and scientific behavior come from the unchanged packaged controller:
beam position/radii/power, applied voltage and fresh bias construction, preset
restoration, nearest-coordinate longitudinal intensity/director-reorientation slices,
width diagnostics, 120/180/300-second budgets, immutable completed snapshots and
exports. Changes are exploratory and do not establish soliton formation. Execution
is synchronous and budgets are checked between slices. Kernel interruption is not
a checkpoint. Allow at least 1 GiB available memory; save exported evidence before
a Colab session expires. No new solver or frontend normalization exists here.

Local deployment tests cover actual notebook cells and the installed distribution.
Fresh hosted installation, browser layout and Run/export behavior remain a separate
qualification step; no scientific simulation was run during this implementation.

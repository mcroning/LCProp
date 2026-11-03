# LCProp

LCProp is a research package for scalar optical beam propagation in nonlinear
media. It provides one shared optical propagation and execution platform with
peer liquid-crystal (LC) and photorefractive (PR) material packages.

The package emphasizes reproducible numerical work, explicit material
ownership, and compatibility-preserving evolution. LC and PR retain their own
physical states, equations, solvers, requests, results, persistence codecs,
and applications while sharing beam launch, runtime grids, NumPy/CuPy backend
selection, optical propagation, execution, and presentation products.

## Try the interactive LC demonstration

**[Open the published interactive LC demonstration in Colab](https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb)**

Use a CPU runtime. Setup installs verified, immutable historical engine/helper revisions;
only **Run** executes a calculation. This existing qualified demonstration is independent
of the packaged release and does **not** install or qualify the latest package version. No manual helper uploads or
repository checkout are required. Colab runtime compatibility and resource availability
can change; retain exported results and provenance before the session expires.

## Interactive BeamPlane in 0.4.0

LCProp 0.4.0 preserves the qualified 0.4.0rc1 runtime: one/two-beam numeric
controls and optional graphical **Move/Tilt** editing through `InteractiveLC`.
Static and interactive previews share canonical combined intensity, beam centers,
internal-ray tilt arrows and active-beam highlighting. Coherence and relative
phase retain the engine's conventions. Editing never starts propagation.

Both `from lcprop.notebook import InteractiveLC` and
`from lcprop.notebook.interactive_lc import InteractiveLC` are supported.
The optional `colab` extra includes `ipympl>=0.10.0,<0.11`; core and GUI extras
remain independent of it. Local Jupyter users select `%matplotlib widget` before
creating the controller, or `%matplotlib inline` for the static preview.

**Hosted Colab requires installation → manual session restart → verification →
initialization.** Restart using Runtime → Restart session, then verify the installed
version before enabling Colab's custom widget manager and selecting the widget
backend. Do not delete the runtime or reload live numerical modules. Export needed
results before restarting. Restart-free installation is not qualified.

Reuse an existing compatible controller to preserve completed results. Run captures
the validated kernel-accepted request; completed plots, metadata and exports remain
attached to that frozen request after draft edits. Unsynchronized browser edits are
outside the capture guarantee. Only the explicit Run button executes science.

The RC1 implementation passed hosted graphical checks and the bounded single-beam
scientific/snapshot qualification. The existing production 0.3.0 notebook and
historical demonstration remain unchanged; a 0.4.0 ordinary-user notebook replacement
requires separate review. Qualification notebooks and evidence bundles are for
release validation, not required ordinary-user setup. See the
[0.4.0 notebook guide](https://github.com/mcroning/LCProp/blob/publication/colab-lc-0.4.0rc1/docs/notebook_package.md)
for setup and fallback details. That branch guide may lag the prepared release until
publication; the source distribution includes the guide matching its version.

## Scientific qualification scope

Retained hosted CPU qualification covers bounded presets for reduced PR time evolution,
LC fixed-director propagation and LC self-consistent nonlinear static propagation.
The 0.3.0 release family's packaged nonlinear LC preset passed nine scientific-array comparisons and all
100 slice-convergence checks. The 0.4.0rc1 runtime likewise passed all nine retained
array comparisons and 100/100 slices in 35.38 seconds, with unchanged exports after
draft edits. This covers the tested preset, not arbitrary multibeam settings. Qualification applies to the recorded scientific preset,
package identity and numerical environment; each release has its own validation scope.

These results do not establish stationary soliton formation, arbitrary beam/voltage
settings, all workflows, or every backend. Modified demonstration parameters are
**exploratory**. Broader LC/PR, multibeam, continuation and GPU/Slurm functionality exists,
with workflow-specific validation/commissioning boundaries; treat combinations lacking
matching evidence as unqualified. The following is an implementation inventory, not a
blanket numerical or environment qualification claim.

## Capabilities

- Multichannel Gaussian beam launch with independent wavelengths, powers,
  waists, positions, transverse phase gradients, phases, and coherence groups.
- Material-neutral angular-spectrum propagation through prepared complex
  response screens.
- NumPy CPU and optional CuPy GPU execution.
- LC static, time-dependent, soliton, continuation, and sweep workflows.
- PR time-dependent, coupled-static, and memory-bounded streaming workflows.
- Material-owned checkpoint persistence and shared explicit dispatch.
- Standalone PySide6 applications for LC and PR users.
- Shared field, curve, diagnostic, workspace, progress, and cancellation
  infrastructure.

Some modules retain historical import paths as compatibility shims. Canonical
LC ownership is under `lcprop.lc`; canonical PR ownership is under
`lcprop.pr`.

## Architecture

```text
beam definitions
      |
      v
shared launch, grids, backends, and optical fields
      |
      +----------------------+----------------------+
      |                                             |
      v                                             v
LC source -> director state -> LC response   PR source -> E state -> PR response
      |                                             |
      +----------------------+----------------------+
                             |
                             v
                advance_prepared_response()
                             |
                             v
             shared execution and presentation
```

The shared platform does not interpret a material state. Each material package
constructs its own optical source, evolves its own state, and converts that
state into the prepared response consumed by the optical engine.

The canonical architecture decision record is
[`docs/architecture/LCProp_Target_Architecture.md`](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/architecture/LCProp_Target_Architecture.md).
The [documentation index](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/README.md) distinguishes current references
from historical design records.

## Package layout

```text
src/lcprop/
    core/          shared beams, grids, backends, and execution records
    optics/        launch and material-neutral optical propagation
    products/      shared presentation data and optical diagnostics
    runners/       explicit workflow-operation dispatch
    persistence/   shared checkpoint-codec composition
    gui/           shared GUI framework and compatibility entry points
    lc/            liquid-crystal physics, workflows, codecs, products, GUI
    pr/            photorefractive physics, workflows, codecs, products, GUI
    algorithms/    genuinely shared numerical infrastructure and shims
    workflows/     historical LC workflow compatibility modules
```

## Installation

The ordinary installation commands below use **production PyPI** and apply to versions
published there. For a release candidate that is not on production PyPI, use its exact
checksum-verified artifacts and accompanying qualification instructions instead.
TestPyPI is a qualification service, not the primary installation route for researchers.

### Ordinary package installation

No Git checkout is required:

```bash
python -m pip install lcprop
```

For Jupyter/Colab widgets and plotting:

```bash
python -m pip install "lcprop[colab]"
```

For local Jupyter, select the desired backend before initialization. Hosted Colab
users must complete the restart/verification sequence above and enable its custom
widget manager first. In a compatible session, reuse the packaged interface:

```python
from IPython.display import display
from lcprop.notebook.interactive_lc import InteractiveLC
if "app" not in globals():
    app = InteractiveLC()
display(app.widget)
```

Initialization shows the launch preview; only **Run** starts the scientific solver.
No external helper download is required by this packaged interface.

For desktop GUI dependencies:

```bash
python -m pip install "lcprop[gui]"
```

**LaunchPlane is separate:** the GUI extra installs PySide6, pyqtgraph and Matplotlib;
it does **not** install [LaunchPlane](https://github.com/mcroning/LaunchPlane).
Install a compatible LaunchPlane separately where the desktop beam editor or saved
editor metadata requires it (schema 3 or newer). Core scientific and notebook use do
not require Qt, LaunchPlane or CuPy. GPU dependencies are separately optional via the
`gpu` extra (CUDA 12 CuPy); installing it does not qualify a GPU/driver/workflow combination.

### Python compatibility and qualification scope

The package declares Python **3.10 or newer**. This is not a claim that every Python
version/platform has been commissioned: recent release checks cover local macOS arm64
Python **3.12** and hosted Linux x86-64 Python **3.13** CPU environments. Desktop use
with current LaunchPlane requires Python **3.11 or newer** and separate GUI dependencies.
Select a Python version covered by the validation evidence for your workflow.

### Candidate artifacts and developer installation

For current candidate review, install the exact supplied wheel (verify its published
SHA-256 first), optionally selecting `[colab]` or `[gui]`. Candidate instructions and
identity records accompany the artifacts; do not assume an unpublished RC can be
obtained from production PyPI. Install into a fresh environment and record dependency
versions for reproducible research.

For developers only, from an LCProp source checkout:

```bash
python -m pip install .
# Editable desktop/test development, with LaunchPlane separately provisioned:
python -m pip install -e ".[gui,test]"
```

Source-checkout installation is optional, not a prerequisite for ordinary package use.
See the [source Quick Start](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/user/quick_start.md)
for desktop environment guidance. These online guides describe their published revision
and can lag newer candidate artifacts. Installed-distribution Slurm deployment validates
package resources and provenance; supported checkout deployment is another route.
Neither a Git checkout nor cluster access is required for local package execution.

## Graphical applications

Launch the LC application with:

```bash
python -m lcprop.lc.gui.app
```

Launch the standalone PR application with either:

```bash
lcprop-pr
```

or:

```bash
python -m lcprop.pr.gui.app
```

The applications share framework components where useful but retain separate
material controls and workflows.

Start with the [Quick Start](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/user/quick_start.md), then use the
[User Guide](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/user/user_guide.md) for model choices, execution, result
retention, visualization, and current limitations. Both applications also
provide a **Help** menu with concise topic guidance.

The LC application supports one- and multichannel propagation, including
angled or colliding coherent beams, through the shared launch editor. It also
provides single-power eigensoliton and bounded power-sweep workflows. LC result
products include transverse and longitudinal optical/director views,
convergence diagnostics, output far fields on in-medium direction-cosine axes,
and selectable optical/director profiles for completed sweep members.
Enabled LC channels currently share one wavelength. LC production workflows
execute with NumPy; the GUI uses float64, while headless static and TD requests
also support bounded NumPy float32 execution. See the canonical
[LC model contracts](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/science/lc_model_contracts.md) for the exact scope.

## Headless example

This small LC example uses canonical material-owned imports:

```python
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.lc import (
    BiasSpec,
    LCMaterial,
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    run_static,
)

request = StaticRunRequest(
    grid=GridSpec(
        Nx=64,
        Ny=64,
        dz_um=5.0,
        x_aperture_um=75.0,
        y_aperture_um=100.0,
        z_length_um=500.0,
    ),
    material=LCMaterial(),
    bias=BiasSpec(),
    beams=BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                power_mW=1.0,
                waist_x_um=3.0,
                waist_y_um=3.0,
                # These are phase gradients in rad/µm, not geometric angles.
                tilt_x_rad_per_um=0.0,
                tilt_y_rad_per_um=0.0,
            ),
        )
    ),
    solver=StaticSolverOptions(),
    output=OutputOptions(),
)

result = run_static(request)
print(result.status)
```

See [`src/lcprop/pr/README.md`](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/src/lcprop/pr/README.md) for the PR state,
normalization, numerical methods, and workflow contracts.

## Testing

After the development/test installation above, run the normal Product suite
with:

```bash
python -m pytest -q
```

Focused tests live alongside the relevant material or shared boundary in
`tests/`. CUDA tests additionally require the `gpu` extra and compatible CUDA
hardware. Cluster commissioning, historical saved-state checks, and other
explicitly supplied evidence remain conditional and are not part of the normal
local test environment.

## Documentation

These links use an immutable, anonymously accessible published documentation snapshot.
Its historical status and plans may precede the installed release; use the release
qualification record for the exact tested configuration.

- [Documentation index](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/README.md)
- [Published status snapshot](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/STATUS.md)
- [Canonical architecture](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/architecture/LCProp_Target_Architecture.md)
- [Published development-plan snapshot](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/development_plan.md)
- [Codex operational prompt library](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/codex/README.md)
- [PR package reference](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/src/lcprop/pr/README.md)
- [Quick Start](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/user/quick_start.md)
- [User Guide](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/user/user_guide.md)
- [PR model contracts](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/science/pr_model_contracts.md)
- [LC model contracts](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/docs/science/lc_model_contracts.md)
- [Third-party notices](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/THIRD_PARTY_NOTICES.md)

## Contributing

- Preserve the dependency direction defined by the architecture decision
  record.
- Keep material state, equations, source construction, solvers, requests,
  results, persistence payloads, and material GUI controls material-owned.
- Add shared abstractions only after more than one concrete implementation
  demonstrates the common contract.
- Preserve compatibility shims deliberately; do not add new canonical code to
  historical modules.
- Add focused regression tests and run the relevant complete material suite
  for every numerical change.
- Keep private research records, raw scheduler output, unpublished reference
  material, and one-off operational prompts outside the public package.

## Project status

LCProp supports two fundamentally different nonlinear material models through
the same optical propagation engine. The architecture migration that
established peer LC and PR ownership is complete; future work should build on
the canonical boundaries rather than reopen them incidentally.

## License

LCProp is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/LICENSE).

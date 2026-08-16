# LCProp

LCProp is a research package for scalar optical beam propagation in nonlinear
media. It provides one shared optical propagation and execution platform with
peer liquid-crystal (LC) and photorefractive (PR) material packages.

The package emphasizes reproducible numerical work, explicit material
ownership, and compatibility-preserving evolution. LC and PR retain their own
physical states, equations, solvers, requests, results, persistence codecs,
and applications while sharing beam launch, runtime grids, NumPy/CuPy backend
selection, optical propagation, execution, and presentation products.

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
[`docs/architecture/LCProp_Target_Architecture.md`](docs/architecture/LCProp_Target_Architecture.md).
The [documentation index](docs/README.md) distinguishes current references
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

LCProp requires Python 3.10 or newer. From a checkout:

```bash
python -m pip install -e .
```

Install the optional GUI or CUDA dependencies as needed:

```bash
python -m pip install -e '.[gui]'
python -m pip install -e '.[gpu]'
python -m pip install -e '.[gui,gpu]'
```

The `gpu` extra installs the CUDA 12 CuPy distribution. The host CUDA runtime
and driver must also be compatible with that package.

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

See [`src/lcprop/pr/README.md`](src/lcprop/pr/README.md) for the PR state,
normalization, numerical methods, and workflow contracts.

## Testing

Run the complete automated suite with:

```bash
python -m pytest -q
```

Focused tests live alongside the relevant material or shared boundary in
`tests/`. GPU tests skip when CuPy or a CUDA device is unavailable; cluster
commissioning is maintained as a separate, explicitly approved workflow.

## Documentation

- [Documentation index](docs/README.md)
- [Current status](docs/STATUS.md)
- [Canonical architecture](docs/architecture/LCProp_Target_Architecture.md)
- [Development plan](docs/development_plan.md)
- [Codex operational prompt library](docs/codex/README.md)
- [PR package reference](src/lcprop/pr/README.md)

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

LCProp is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

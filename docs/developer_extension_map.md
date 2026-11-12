# Developer extension map

These are existing composition points, not a stable general plugin SDK. Start in
a fork/new environment and preserve existing LC/PR model semantics. The canonical
[architecture decision](architecture/LCProp_Target_Architecture.md) remains the
source of ownership rules.

## Add a material response (for example Kerr)

- Shared propagation: `lcprop.optics.splitstep.advance_prepared_response`.
- Material-owned LC/PR workflows convert their state into prepared optical screens.
- Explicit local composition: `WorkflowOperation` in `lcprop.runners.base`, with
  registration in `lcprop.runners.local.LocalRunner`.

A separate material workflow can reuse these boundaries without editing LC or PR
physics. This does not make a correct Kerr solver automatic: define intensity and
coherence conventions, field normalization, units and when the field-dependent
response is recomputed. The prepared screen supplied to a call is fixed across
its substeps; do not mistake that for a self-updating nonlinearity. Remote codecs,
checkpoint semantics and GUI controls are additional integrations, not inherited
merely from local registration. Validate a bounded model before extending further.

## Add post-processing or visualization

`lcprop.products.data_model` defines `RunData`, `Geometry`, `FieldData`, `CurveData`,
`DiagnosticData` and `ArtifactData`. Material-owned adapters prepare these records.
A consumer can process existing outputs and construct separate derived products;
`lcprop.gui.workspace.Workspace.set_run_data` is a concrete GUI consumption point.

Respect axes, units, coordinates and quantity definitions. Frozen dataclasses do
**not** make contained arrays/collections immutable. Copy data before modifying
it, keep original results intact, and identify derived products separately. A
custom plot outside the application is simpler than promising automatic viewer
plugin discovery. New native viewer types need separately scoped integration.

After installing the engine with the [developer quickstart](developer_quickstart.md),
run the [transverse intensity profile example](../examples/transverse_intensity_profile.py)
from the checkout root:

```bash
.engine-env/bin/python examples/transverse_intensity_profile.py
```

The example uses synthetic XY intensity data and prints the x coordinates and
mean intensity over y as `[(-2, 2.0), (0, 3.0), (2, 4.0)]`. It returns a separate
`CurveData` and leaves the input intact. It averages samples with equal weight
and requires only the core installation.

### Process a completed PR ZIP

Use a compatible completed PR ZIP exported by LCProp, such as one saved
with **Download completed** in the PR notebook. See the
[Colab file guide](colab_first_batch.md) for export instructions. From the
checkout root, start `.engine-env/bin/python` and run the following, replacing
`completed-pr.zip` with the path to your archive.

```python
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

from lcprop.notebook.batch_files import unpack_completed
from lcprop.transport.defaults import default_transport_operations, default_transport_registry
from lcprop.transport.io import runner_result_from_package

archive = Path("completed-pr.zip")
mean_intensity_over_y = run_path(
    "examples/transverse_intensity_profile.py"
)["mean_intensity_over_y"]

with TemporaryDirectory() as staging:
    package = unpack_completed(archive.read_bytes(), Path(staging))
    opened = runner_result_from_package(
        package,
        registry=default_transport_registry(),
        operations=default_transport_operations(),
    )
    if opened.material_id != "pr" or opened.run_data is None:
        raise ValueError("Expected a completed PR result with retained products")
    run = opened.run_data
    field = run.fields.get("input_intensity")
    if field is None:
        raise ValueError("This PR result does not retain the input intensity plane")
    profile = mean_intensity_over_y(field, run.geometry)
    print(f"{len(profile.y)} samples", profile.units)
```

The loader validates the package and reconstructs its retained products.
This example selects the normalized XY input intensity plane and averages
its samples over y. The number of samples depends on the saved x grid.
The ZIP stays intact and the temporary extraction is removed on exit.
These imports work with the core install. The archive helper lives in the
notebook namespace and the recipe requires neither a running notebook
nor a GPU.

## Modify or replace a frontend

Desktop applications are composed under `lcprop.lc.gui`, `lcprop.pr.gui` and shared
`lcprop.gui` modules. GUI dependencies are optional; use workflows/runners and
shared products from a separate frontend instead of importing a window into the
calculation engine. The engine must remain independently usable.

Changing a request builder can change science without editing a solver. Check
request identities, defaults, units and persistence behavior as well as appearance.
Do not redirect the maintainer's launchers or editor paths. Frontend experiments
stay in your own environment and become upstream only through approved review.

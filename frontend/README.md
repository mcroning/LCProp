# PR capability workbench — developer preview

This standalone frontend uses the unchanged integrated engine **ce915366**. It does not replace `src/lcprop`, the desktop installation, public README links or a release. The ordinary opening flow remains one setup ZIP and one Open cell. LC is separate.

## Inputs and reproducible local build

One explicit prerequisite is required: either the exact integrated engine wheel or a supplied local capability setup kit containing that wheel. The wheel is not the published rc5 binary, and it is not downloaded, installed or published by the build tools. Obtain the supplied private/local kit through the project delivery process; record its full SHA-256. Historical accepted kit c2e57580… is a valid **engine input**, not acceptance of a new frontend.

The only allowed engine SHA-256 is:

```
ce9153666272780d3eac63f2c56d2777ae821a260bca6619e26852049cb78edd
```

From any checkout location, with paths you choose:

```
python tools/stage_pr_capabilities.py --input-kit /path/to/supplied-kit.zip --input-kit-sha256 FULL_VERIFIED_KIT_SHA256 --output /path/to/new-delivery
```

Alternatively use `--engine-wheel /path/to/lcprop-0.5.0rc3-py3-none-any.whl`. No sibling checkout, absolute project path or original LaunchPlane wheel is needed. The unchanged headless LaunchPlane source subset and license are included in `frontend/launchplane`; `PROVENANCE.json` pins every upstream file. The build validates these files and the engine, refuses to overwrite an existing destination, and creates the notebook, setup ZIP, unchanged wheel, PINS.json and SHA256SUMS deterministically. It never installs software or runs a solver.

Current tracked candidate notebook and manifest are under `delivery-v5/`. Setup ZIP and engine binaries remain **local supplied artifacts**, ignored by Git; they are not silently published with a source PR. Build to a new directory, verify its hashes against the tracked manifest, and set `PR_CAPABILITY_KIT` to that new ZIP. A source edit requires a newly identified kit; preserve older deliveries and their acceptance records.

## Local tests without scientific work

Use a disposable developer Python environment with the existing notebook/test dependencies: NumPy, SciPy, Pillow, pytest, IPython/ipykernel/traitlets, ipywidgets 7.7.1, ipympl 0.10.0 and Matplotlib 3.11.2. Matching widget dependencies are pinned in the notebook. Do not modify an existing desktop environment to run these checks.

Prepare an **import-only** engine target from the supplied kit; this command verifies all setup members and the engine wheel before extracting it. Nothing is installed or executed:

```
python tools/prepare_pr_frontend_tests.py --kit /path/to/supplied-kit.zip --expected-sha256 FULL_VERIFIED_KIT_SHA256 --target /path/to/new-engine-imports
```

Run from the checkout root, using that import target and the rebuilt current kit:

```
PYTHONPATH=/path/to/new-engine-imports:frontend PR_CAPABILITY_KIT=/path/to/new-delivery/LCProp-PR-capabilities-setup.zip MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/frontend -q
```

The suite prohibits native dispatch and makes CuPy unavailable. It tests synthetic request/codec behavior, screen ownership, independent form errors, CPU/Auto/GPU decisions, fake progress/Stop/failures, opening with mocked installation, and deterministic packaging. No scientific qualification follows from these checks. Without `PR_CAPABILITY_KIT`, opening/package tests clearly skip when no local current ZIP exists; synthetic tests need only the pinned engine import target, not private research results.

## Optional saved evidence

Historical saved-ZIP tests are optional and separately identified. Supply `--pr-saved-fixtures /path/to/map.json` (or `PR_SAVED_FIXTURES`) to run them. The JSON map has keys `legacy`, `pr_static`, `pr_transverse_static`, whose values are paths to the corresponding saved ZIPs. Accepted hashes are in `tests/frontend/saved-fixture-pins.json`. An explicitly supplied missing or wrong fixture fails; absence of a map clearly skips these three historical tests. No fixture path is tied to the author's computer. Original ZIPs are read-only; exports use new temporary names.

Do not add large private ZIPs, scientific arrays, the unchanged engine wheel or setup binaries to a public PR to make tests pass. A source PR can include source, the current notebook/pins and compact qualification summaries with hashes. Full historical evidence stays in its original local location and remains valid only for its exact recorded revision.

## Editor state and format boundaries

- Screens stay with unique beam names when preceding beams are disabled/deleted or their channel order changes. A deleted beam's screens are removed only from the new draft. Screens for disabled beams are retained as versioned editor metadata and restored on re-enabling, including after experiment save/load. Native enabled-channel launch records are unchanged.
- Experiment metadata uses native intent validation and native route-specific request validation. Supported float32 and full/analysis/fast policies round-trip. Unsupported native combinations still fail. Fixed-product operations reject incompatible saved retention explicitly rather than silently changing it.
- Inactive screens use the namespaced `notebook_inactive_launch_elements_v1` presentation key with native raster encoding. The engine receives no disabled screen or changed scientific format. An older interface that cannot preserve this extra editor metadata may reject the draft; active request fields remain native.

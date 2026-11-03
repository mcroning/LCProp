# Developer test tiers

Run from the repository root in a new environment prepared by the
[engine quick start](developer_quickstart.md) or
[desktop guide](user/quick_start.md). Install the `test` extra there before
running pytest. Verify that imports point to the intended source/install and
record its exact commit and dependencies. These examples select existing tests;
they are not a substitute for checks specific to a numerical change.

| Tier | Prerequisites and coverage | Limits |
|---|---|---|
| Focused metadata/ownership | Core dependencies plus `test`; request construction, canonical algorithm aliases, PR closure/specification validation | No solver run, GUI startup, device qualification or numerical correctness claim. Small metadata arrays do not establish resource feasibility. |
| Broader CPU subset | Same environment; grids, LC driving source, shared launch/optics, reduced PR material solves, unified operator/projection checks | Includes numerical work and is an explicit opt-in after focused checks. Not a full material/workflow suite, timing guarantee or scientific convergence study. GPU cases in mixed files are explicitly deselected. |
| Optional GUI | `gui` and `test` extras, compatible Qt platform, independently provisioned schema-4 LaunchPlane companion; ownership and PR request construction | Offscreen tests do not qualify native display, touch/Colab, end-to-end scientific execution or remote scheduling. |
| Optional GPU | `gpu` and `test` extras, compatible CuPy/CUDA/driver and accessible device; cyclic-solver/diffusion parity checks | Tests skip when CuPy/device is unavailable; a skip is not GPU qualification. Not H200/Slurm commissioning or all PR workflows. |

## Focused metadata and ownership

```bash
python -m pytest -q -p no:cacheprovider tests/test_imports.py tests/test_core_objects.py tests/test_lc_algorithm_ownership.py tests/test_pr_unified_specs.py
```

The listed modules have no fixture that submits jobs or runs a simulation during
collection. Alias tests import canonical implementations but do not execute
propagation. The specification tests construct/validate metadata. Do not assume
an arbitrary ownership or “smoke” file is similarly bounded.

## Broader CPU subset

```bash
python -m pytest -q -p no:cacheprovider -m 'not slow' -k 'not cupy and not native_projection_parity_when_available' tests/test_grid_metadata.py tests/test_lc_source.py tests/test_launch.py tests/test_splitstep.py tests/test_pr_static.py tests/test_pr_unified_operators.py tests/test_pr_unified_projection.py
```

This allowlist uses NumPy numerical fixtures. `test_pr_static.py` and
`test_pr_unified_operators.py` also contain CuPy cases; the `-k` expression
excludes those and the native projection parity cases in
`test_pr_unified_projection.py`. Keep the exclusions even on machines with a
GPU. The fixture in `test_pr_unified_operators.py` imports request helpers from
`test_pr_unified_static.py`; that module's tests are not selected by this command.
There is no repository-wide `cpu` or `gpu` marker. `-m 'not slow'` alone does
not make the entire suite CPU-only or inexpensive.

## Optional GUI and GPU

These tiers require separate resources and were not executed for this
documentation milestone.

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q -p no:cacheprovider tests/test_gui_ownership.py tests/test_pr_gui_request.py
```

```bash
python -m pytest -q -p no:cacheprovider tests/test_pr_cyclic_gpu.py
```

The GUI modules construct/import actual Qt/LaunchPlane components; provide
exactly the schema expected by the [adapter](../src/lcprop/adapters/launchplane.py).
The GPU file checks device availability inside its test bodies, so successful
collection alone establishes neither a device nor passing GPU tests.

## Exclusions and separate qualification

The allowlists intentionally exclude:

- The existing `slow` LC checks
  `tests/test_soliton_trans_td.py::test_seed_and_polished_solitons_in_coupled_td`
  and `tests/test_theta_equilibrium.py::test_polished_theta_is_td_equilibrium`:
  they include repeated polished soliton/TD work and require a separate budget.
- Saved Research evidence regressions such as `tests/test_pr_a5_diagnostics.py`:
  its frozen-reference cases require separately retained checksum-bound inputs
  (see `tests/fixtures/frozen_a5.txt`) and must not be represented as covered by
  this CPU subset. Missing fixtures can fail rather than skip; do not regenerate
  or replace them to obtain a passing suite.
- `tests/test_pr_published_static.py::test_material_math_bytes_unchanged`:
  compares source bytes to a historical Git object. It requires that history and
  matching baseline; it is not a generic test for an arbitrary modified checkout.
- Hosted/notebook evidence, full workflow simulations, large-grid/resource tests,
  packaging/build/install tests and Slurm commissioning. Some deployment unit
  tests use temporary fake transports; that does not establish live cluster access.
- Historical documentation assertions outside the tiers need separate review:
  for example, `tests/test_release_documentation.py` still expects version
  `0.3.0` and the older README installation layout, while the current package
  metadata declares `0.4.0`. Those pre-existing assertions are not repaired or
  represented as passing by this documentation-only milestone.

Do not start with unrestricted `python -m pytest -q`. A broader regression plan
must state selected workflows, fixture identities, optional dependencies, time/
memory budgets and omissions. Historical release counts and excluded checks
remain historical evidence; do not present them as results from a new checkout.

## What was checked for these instructions

At baseline `cb17b6be8c0621a57e13e0a1410f7d7201f5b424`, test configuration,
imports, module-level code, parameterization and helper/fixture behavior were
inspected. Focused and broader CPU commands were verified by collection only,
with bytecode/cache writes and third-party pytest plugin autoload disabled.
No focused/CPU numerical test bodies or scientific examples were executed.
Separate file/link-only documentation checks were run. GUI/GPU commands were
checked statically; those tiers were not run or device-qualified. Repeat safe
collection when the test files or dependencies change before relying on this map.

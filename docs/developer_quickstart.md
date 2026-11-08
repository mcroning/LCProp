# Engine installation from a Git checkout

Use a new checkout and a clean Python 3.12 environment. Python 3.12 on macOS
arm64 was used for the bounded engine wheel checks. The package declares
Python >=3.10, and support for other configurations requires separate checks.

```bash
git clone https://github.com/mcroning/LCProp.git LCProp-public-start
cd LCProp-public-start
git rev-parse HEAD
python3.12 -m venv .build-env
.build-env/bin/python -m pip wheel --no-deps --wheel-dir dist .
python3.12 -m venv .engine-env
.engine-env/bin/python -m pip install dist/lcprop-*.whl
.engine-env/bin/python -m pip check
```

Run these commands in an unused directory so the wheel pattern selects one
artifact. Record the commit, wheel SHA-256 and installed dependency versions.
For a specific reviewed revision, check it out before building. The build
and install can download dependencies.

From outside the checkout, use the new environment’s Python. Confirm the package location and
that the core result/propagation interfaces import without constructing a GUI:

```python
import lcprop
from lcprop.optics.splitstep import advance_prepared_response
from lcprop.products.data_model import RunData
print(lcprop.__file__)
print(advance_prepared_response.__name__, RunData.__name__)
```

For the non-editable installation above, the package location should be under
the new environment, not an unrelated source checkout. This is an installation/interface check, **not a calculation** or proof
of numerical correctness. No Qt, LaunchPlane or GPU dependency is required for
these imports.

For editable development, clone your fork into a separate new directory. A fork
that copied only the old default branch may not contain the tested commit. Fetch
the retained reference without changing your fork's remote configuration:

```bash
git fetch https://github.com/mcroning/LCProp.git publication/colab-lc-0.4.0rc1
git rev-parse FETCH_HEAD
```

Stop if `FETCH_HEAD` is not exactly
`4299ec618810f63ed39431ad1cb374f8f7e6cb93`; do not silently substitute a newer tip.
Then create your feature branch from that exact commit:

```bash
git switch -c my-change 4299ec618810f63ed39431ad1cb374f8f7e6cb93
```

Create a new virtual environment in that fork as above, activate it, and install
with `python -m pip install -e '.[test]'`. An editable installation should now
resolve `lcprop.__file__` inside **your fork's `src/lcprop`**, not the environment's
package directory and never the maintainer's working checkout. Check the source
commit and import location again before testing. Run checks relevant to your
change using the [test tiers](developer_testing.md); do not automatically run
large solvers or cluster workflows.

Desktop GUI work is a separate route: the `gui` extra supplies GUI dependencies,
but LaunchPlane is a separate companion with its own compatibility requirements.
Do not install an arbitrary companion version and infer a qualified GUI setup.
The [documentation index](README.md) and [extension map](developer_extension_map.md)
describe the boundaries; a desktop installation and acceptance are scoped separately.

For a small interactive calculation, the [historical notebook demonstration](https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb)
uses its own pinned versions and stated validation scope. It is not a test of
this release reference. Review its setup and limits before running it.

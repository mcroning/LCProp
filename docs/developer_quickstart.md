# Tested engine: isolated source quick start

This route installs the exact tested source reference, not an unspecified package
index release. It does not alter an existing installation. Choose a **new unused**
checkout directory and environment; stop if either already contains work.

Python 3.12 was used for the recorded macOS CPU qualification (3.12.14). The package
declares Python >=3.10, but that is not blanket platform qualification. Python,
Git and compatible NumPy/SciPy dependencies must be available; this is not an
offline standalone recovery kit or a fully locked dependency environment.

```bash
git clone --no-checkout https://github.com/mcroning/LCProp.git LCProp-public-start
cd LCProp-public-start
git checkout --detach 4299ec618810f63ed39431ad1cb374f8f7e6cb93
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

These install commands have dependency/network effects **only when you run them**.
Do not run them in the maintainer's preserved environment. Record installed
versions if you need reproducible results. A dependency index may supply versions
that differ from the historical qualification.

Verify the checkout with `git rev-parse HEAD`. Confirm the package location and
that the core result/propagation interfaces import without constructing a GUI:

```python
import lcprop
from lcprop.optics.splitstep import advance_prepared_response
from lcprop.products.data_model import RunData
print(lcprop.__file__)
print(advance_prepared_response.__name__, RunData.__name__)
```

The package location should be under the new environment, not an unrelated source
checkout. This is an installation/interface check, **not a calculation** or proof
of numerical correctness. No Qt, LaunchPlane or GPU dependency is required for
these imports.

For editable development, use your fork as the clone URL and create a feature
branch from the reference instead of remaining detached. In the new environment,
install with `python -m pip install -e '.[test]'`. Run checks relevant to your
change; do not automatically run large solvers or cluster workflows.

Desktop GUI work is a separate route: the `gui` extra supplies GUI dependencies,
but LaunchPlane is a separate companion with its own compatibility requirements.
Do not install an arbitrary companion version and infer a qualified GUI setup.
The [documentation index](README.md) and [extension map](developer_extension_map.md)
describe the boundaries; a fresh public desktop-user acceptance pass remains
separately scoped.

For a small interactive calculation, the [historical notebook demonstration](https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb)
uses its own pinned versions and stated validation scope. It is not a test of
this release reference. Review its setup and limits before running it.

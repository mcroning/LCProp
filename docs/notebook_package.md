# Optional installed LC notebook interface (0.3.0)

In an environment containing this candidate distribution and its `colab` extra:

```python
from IPython.display import display
from lcprop.notebook.interactive_lc import InteractiveLC
app = InteractiveLC()
display(app.widget)
```

Initialization samples the launch only. Press Run explicitly to solve. No helper
download, checkout, Qt, LaunchPlane or CuPy is required for canonical experiments.
The existing Configure label, voltage/beam controls, 120/180/300-second budgets,
immutable completed snapshots and exports are preserved. GUI editor metadata remains
subject to the separate LaunchPlane compatibility contract. No new scientific claim.

`lcprop.notebook.lc_static_demo` holds the unchanged shared preset and diagnostics,
and the existing standalone comparison entry point. The files in `examples/` are
compatibility wrappers; use them with this candidate installed, not an older wheel.
The historical Git-pinned bootstrap/notebook remains unchanged and uses its immutable
older helpers; do not substitute these wrappers into that historical hash contract.

The `colab` extra provides Matplotlib, ipywidgets and IPython. Core scientific imports
do not require those packages. Jupyter/Colab kernel provisioning is separate.
This guide describes version 0.3.0. For release qualification, install the verified
candidate wheel with its `colab` extra. Indexed installation requires publication
of that version to the selected package index.
The source distribution includes runtime source/data and this guide, both wrappers,
and the self-contained `tests/test_notebook_package.py` suite. Repository-only tests
and protected qualification artifacts are intentionally excluded from the sdist.

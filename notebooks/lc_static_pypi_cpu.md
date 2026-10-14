# Interactive LC optics from PyPI

Open `lc_static_pypi_cpu.ipynb` in Colab with a CPU runtime, or in a disposable local
Jupyter environment. Run **1. Install LCProp**, then **2. Open the interface**.
Alternatively choose Run all. No file uploads, checkout or helper downloads.

Installation uses production PyPI and pins `lcprop[colab]==0.3.0`. Normal pip output
shows installation progress and errors. The optional extra supplies the notebook
libraries; Qt, LaunchPlane and CuPy are not needed. No scientific calculation occurs
until you press **Run** inside the interface. Editing controls only samples the launch
preview. All physics, normalization and controls come from the packaged controller.

Colab's third-party notebook warning means Google did not author this code. Review
it and trust its publisher before execution. No Drive mount is performed.

Rerunning initialization preserves the same interface, controls and completed
snapshot. Exports remain bound to the last successful calculation. An incompatible
loaded LCProp version or existing interface produces a restart instruction instead
of replacing the result. If you manually replace packages within a live kernel,
restart it first: ordinary distribution-version checks cannot identify every stale
in-memory module. No automatic restart is performed.

The notebook intentionally contains no wheel download, hash manifest, package-file
scanner or custom numerical-dependency checks. Published release qualification is
separate from ordinary notebook use. The existing Git-pinned qualification notebook
and its verification machinery remain unchanged.

Beam/voltage controls, longitudinal intensity/director-reorientation plots, beam
widths, 120/180/300-second budgets, preset restoration and snapshot/export behavior
are unchanged. Modified settings are exploratory. Convergence does not establish
stationary soliton formation. Execution remains synchronous; budgets are checked
between completed slices. Save exports before a hosted session expires.

This simplified notebook awaits separately authorized publication. The earlier public
notebook remains unchanged until that publication. Fresh hosted Colab
browser/layout and user-driven Run/export qualification remain pending.

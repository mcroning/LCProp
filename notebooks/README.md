# Portable LCProp notebooks

`pr_reduced_cpu.ipynb` is a small NumPy float64 reduced nonlinear PR TD demonstration.
It uses the Product preflight, Local runner, result adapters and persistence codecs.
No Qt, LaunchPlane, GPU, notebook-specific scientific equations or automatic setup is used.

## Local Jupyter

Choose a new Python >=3.10 environment. Explicit setup commands (not notebook execution):

```sh
python3 -m venv .venv-notebook
source .venv-notebook/bin/activate
python -m pip install 'lcprop @ git+https://github.com/mcroning/LCProp.git@985b80fbbc05616d4fe92f974ef0da24bf84e94c' matplotlib jupyterlab ipykernel
python -m jupyter lab
```

Open this notebook from the candidate checkout or a downloaded copy and choose that
environment's kernel. The pinned scientific baseline does not contain this new notebook.
An exact baseline wheel may replace the Git installation. No repository credentials
or private research files are required. LCProp installs NumPy/SciPy; plotting adds
Matplotlib, notebook execution needs IPython/ipykernel and a Jupyter frontend.
The notebook reports actual module path and versions. Save its manifest: pinning
LCProp does not pin NumPy/SciPy or guarantee cross-version bitwise identity.

## Colab

Upload/open the notebook and explicitly select a CPU runtime. In a separate setup
cell, if needed, run:

```python
%pip install 'lcprop @ git+https://github.com/mcroning/LCProp.git@985b80fbbc05616d4fe92f974ef0da24bf84e94c' matplotlib
```

Restart the runtime after setup if required, then run the scientific cells in order.
The notebook does not install anything, mount Drive, allocate GPUs or connect to
remote systems. Check Python >=3.10 and actual dependency versions in its first cell.
Hosted storage is ephemeral: explicitly download the saved directory or copy it to
storage you control. Drive mounting is optional and outside this notebook.
For a Colab local runtime, prepare the same local environment and follow Colab's
current local-runtime connection guidance; the scientific cells are unchanged.

## Tufts compute-node kernel

Use a Jupyter kernel already running **inside a Slurm-allocated compute node**.
Never run the scientific cell on a Pax login node. Allocation, tunnels, authentication
and scheduler integration are outside this milestone; no cluster commands are embedded.
No hosted Colab, Windows/Linux or Tufts-runtime qualification is claimed here.

## Run, save, reopen

Run all cells in a clean kernel. The three-step 64×48 example reports scalar progress
and elapsed wall time, then plots retained xy and xz optical intensity. The physical
illumination inputs and dimensionless characteristic time are explicitly labeled.
The plotted numerical optical intensity has units 1/µm²; it is not the physical
W/cm² input or total normalized transport source.

Call `save_demo(Path("my-new-run-folder"))` after execution. Existing directories are
rejected. The function saves the experiment, full tiny Product result package, selected
arrays/coordinates, a PNG and a SHA-256 manifest. It does not upload anything.
Reopen the experiment with `read_experiment_file(..., registry=experiments)` and the
result with `lcprop.transport.io.read_result_package(..., registry=codecs)`, using the
same registries shown in the save function. Product package readers verify integrity.
Selected NPZ/PNG exports alone are not continuation checkpoints.

## Developer validation

In a disposable test environment with pytest, nbformat, nbclient and ipykernel:

```sh
PYTHONPATH=src MPLBACKEND=Agg python -m pytest -q tests/test_portable_pr_notebook.py
```

The clean-kernel test registers a temporary kernelspec only in its temporary test
directory. It compares actual notebook outputs to the headless engine, exercises
persistence, and blocks Qt/LaunchPlane/CuPy imports. No GPU or remote tests occur.
Stage 2 should add a thin non-Qt request editor and explicit experiment loading,
without duplicating scientific validation or expanding the qualified workflow scope.

# LCProp Quick Start

This guide is the shortest path from a fresh checkout to a small, inspectable
LC or photorefractive (PR) calculation. The examples are smoke-sized starting
points, not convergence evidence for a scientific conclusion.

## Install and launch

LCProp's headless core requires Python 3.10 or newer. The GUI requires Python
3.11 or newer because its beam editor is the separate
[LaunchPlane Product](https://github.com/mcroning/LaunchPlane), schema 3 or
newer. Until LaunchPlane has a separately released package version, install it
from its public source checkout in the same environment:

```bash
git clone https://github.com/mcroning/LaunchPlane.git
python -m pip install -e ./LaunchPlane
python -m pip install -e '.[gui]'
```

Launch the LC application:

```bash
python -m lcprop.lc.gui.app
```

Launch the PR application:

```bash
lcprop-pr
```

The equivalent module command is `python -m lcprop.pr.gui.app`.

## First LC calculation

1. In **Experiment**, select **Static propagation**.
2. In **Beam**, keep one enabled Gaussian beam. LaunchPane controls its power,
   wavelength, transverse position, phase, coherence group, profile, and focus.
3. In **Grid**, begin with `Nx=64`, `Ny=64`, a short interaction length, and a
   longitudinal step that gives several planes. Larger or finer grids require
   an explicit convergence study.
4. In **Physics**, review refractive indices, elastic constant, dielectric
   anisotropy, bias voltage, and boundary director angle.
5. In **Solver**, use **local_self_consistent** for coupled static propagation.
6. Choose **Periodic**, **Sponge**, or **Tukey** in the Beam tab's transverse
   optical-boundary panel. Sponge is usually the safer exploratory choice when
   diffracted light could reach an FFT boundary.
7. Leave **Execution** at **Local**, click **Run Static propagation**, and wait
   for the status to complete.
8. Inspect **Fields**, **Curves**, **Diagnostics**, **Request**, and **Console**
   in **Results**. The Request tab is the configuration that actually ran.

For time dependence, select **Time-dependent propagation**, choose a beam,
retained static result, or retained soliton as the initial condition, and set
`Nt` and `dt` in Solver. Static and time-dependent experiments can be saved and
reopened. Checkpoints are the mechanism for numerical continuation.

## First PR calculation

1. In **Input**, select the ordinary Gaussian-beam mode.
2. In **Beam**, keep one enabled Gaussian or add a second beam. Beams in the
   same explicit coherence group interfere; beams in separate groups add
   incoherently.
3. In **Grid**, start with `Nx=64`, `Ny=64`, `Optical dz=10 µm`, and a short
   interaction length such as `20 µm`.
4. In **Evolution**, choose the three physical axes independently:
   **Static** or **Time dependent**, **Reduced x-only** or **Full transverse**,
   and **Fully nonlinear** or **Linearized**.
5. For a quick local run, select **Reduced x-only**. Linearized material
   response requires an explicit positive reference intensity `I₀`.
6. A nonlinear time-dependent reduced run presents **Semi-implicit
   trapezoidal** and **Explicit Euler (reference)**. Linearized TD presents the
   exact modal update. Static selections hide material-time controls.
7. Keep **Backend=numpy**, **Precision=float64**, and **Execution=Local** for
   the first run. Click **Estimate current request** in **Run Planning** before
   launching a larger request.
8. Click **Run PR** and inspect Fields, Curves, Diagnostics, Request, and
   Console. A completed status does not replace inspection of convergence and
   physical diagnostics.

## Focused and collimated beams

- **Collimated Gaussian**: the displayed waists describe the entrance plane.
- **Focused Gaussian**: the waist is specified at its focus. `focus_z=0` is
  the interaction entrance plane, positive positions are inside/downstream,
  and negative positions are upstream. Elliptical x/y waist evolution is
  resolved independently.
- **Uniform**: a tilted uniform field must be an exact periodic Fourier mode
  on the consuming LCProp grid. LaunchPane cannot enforce that grid-dependent
  condition by itself.

LaunchPlane expresses beam intent and visualization. LCProp owns the runtime
grid, medium refractive index, interaction length, propagation, and material
response.

## Local, Slurm, Fast, and Full

**Execution** selects Local or Slurm. It is distinct from the scientific
NumPy/CuPy backend. PR may choose CuPy automatically for a GPU Slurm resource
only while the backend remains an automatic default; explicit user and loaded
experiment choices are preserved.

For PR Slurm retrieval:

- **Fast / Exploratory** keeps optical endpoints, compact diagnostics, exact
  full-resolution cuts nearest `x=0` and `y=0`, and a bounded downsampled MPR
  preview for visualization.
- **Full** retains the supported complete scientific volumes and costs more to
  package, transfer, and hold in memory.

LC Slurm execution is implemented only for canonical NumPy static propagation
on a CPU resource profile; real scheduler commissioning is pending. TD,
soliton, and soliton-existence remote execution remain unsupported. The LC GUI
does not expose the PR Fast/Full retrieval selector.

## Save and reproduce

Use **Save Experiment** before expensive runs. Experiment files preserve the
user-visible scientific request and migrate supported older schema versions.
Use checkpoints for continuation state; an experiment file is not a checkpoint.
Record the Product commit, experiment file, backend/precision, and result
retention policy with any scientific result.

Continue with the [LCProp User Guide](user_guide.md) before a large or
publication-facing run.

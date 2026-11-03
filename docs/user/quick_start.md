# LCProp Quick Start

This guide is the shortest path from a fresh checkout to a small, inspectable
LC or photorefractive (PR) calculation. The examples are smoke-sized starting
points, not convergence evidence for a scientific conclusion.

## Install and launch

LCProp's headless core requires Python 3.10 or newer and does not require
LaunchPlane or Qt. Start with the [tested engine quick start](../developer_quickstart.md)
for that route. Desktop use with [LaunchPlane](https://github.com/mcroning/LaunchPlane)
requires Python 3.11 or newer and **exactly schema 4**, including the host-resolved
preview interface. Schema numbers are serialization/API contracts, not package
version numbers; “schema 3 or newer” is not sufficient. The
[adapter](../../src/lcprop/adapters/launchplane.py) rejects any other schema.

Use new unused directories and a new environment. `main` is the evolving public
entry, not a qualification label. The example checks out LCProp's tested source
reference `4299ec618810f63ed39431ad1cb374f8f7e6cb93` after cloning `main`.
LaunchPlane is independently pinned to schema-4 source
`f96715e443cec17c82e23658d82a359052764194` ([serialization](https://github.com/mcroning/LaunchPlane/blob/f96715e443cec17c82e23658d82a359052764194/src/launchplane/serialization.py)).
That schema check is not a fresh qualification of this desktop pair; no package
version or current companion branch alone guarantees compatibility. A separate
fresh desktop installation/acceptance pass remains required.

```bash
mkdir lcprop-work
cd lcprop-work
python3.12 -m venv .venv
source .venv/bin/activate
git clone --no-checkout https://github.com/mcroning/LaunchPlane.git
git -C LaunchPlane checkout --detach f96715e443cec17c82e23658d82a359052764194
git clone --branch main https://github.com/mcroning/LCProp.git
git -C LCProp checkout --detach 4299ec618810f63ed39431ad1cb374f8f7e6cb93
python -m pip install ./LaunchPlane
cd LCProp
python -m pip install '.[gui]'
python -c 'from launchplane.serialization import SCHEMA_VERSION; assert SCHEMA_VERSION == 4, SCHEMA_VERSION'
```

Record both `git rev-parse HEAD` values and installed dependency versions. These
commands install dependencies only when you run them; do not use a preserved
working environment. The existing historical notebook links keep their own
pinned identities and do not qualify this desktop setup.

For **developer editable installation**, create a feature branch in your fork
from your stated baseline, then use the new environment and companion checkout:

```bash
python -m pip install -e ../LaunchPlane
python -m pip install -e '.[gui,test]'
```

See the [test tiers](../developer_testing.md) before selecting tests. Editable
installation is not required for ordinary Local execution. Slurm can deploy
supported verified installed distributions or clean source checkouts; see
[deployment prerequisites and limitations](user_guide.md#package-deployment-and-remote-environment-prerequisites).
A successful Local installation does not establish remote environment readiness.

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
2. In **Beam**, keep one enabled Gaussian beam. LaunchPlane controls its power,
   wavelength, transverse position, phase, coherence group, profile, and focus.
3. In **Grid**, begin with `Nx=64`, `Ny=64`, a short interaction length, and a
   longitudinal step that gives several planes. Use modest settings for quick
   exploration. Verify grid, step-size, timestep, precision, and model convergence
   as applicable before drawing quantitative conclusions.
4. In **Physics**, review refractive indices, elastic constant, dielectric
   anisotropy, bias voltage, and boundary director angle.
5. In **Solver**, use **local_self_consistent** for coupled static propagation.
6. Choose **None**, **Sponge**, or **Tukey** in **Beam → Optical edge treatment**.
   FFT propagation uses a periodic domain; None applies no edge attenuation. Sponge is usually the safer exploratory choice when
   diffracted light could reach an FFT boundary.
7. Leave **Execution** at **Local**, click **Run Static propagation**, and wait
   for the status to complete.
8. Inspect **Fields**, **Curves**, **Samples / Tables**, **Diagnostics**, **Request**, and **Console**
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
4. In **Evolution**, choose **Static** or **Time dependent** and **Reduced
   x-only** or **Full transverse**. Material choices follow the supported
   seven-choice matrix in the User Guide.
5. For a quick local run, select **Reduced x-only**. Static offers **Nonlinear
   reduced hopping** and **Field-linear (local intensity)**, the paper Eq. (5)
   approximation with no `I₀` control. Reduced TD supports nonlinear hopping
   only. The distinct full-transverse **Uniform-reference tangent** requires
   an explicit positive reference intensity `I₀`.
6. A nonlinear time-dependent reduced run presents **Semi-implicit
   trapezoidal** and **Explicit Euler (reference)**. Full-transverse tangent TD
   presents the exact modal update. Static selections hide material-time controls.
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
  on the consuming LCProp grid. LaunchPlane cannot enforce that grid-dependent
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
  full-resolution cuts nearest `x=0` and `y=0`, and a bounded downsampled orthogonal-slice
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

The GUI **Help** menu renders concise topics without an editor. Each topic links
to the [rendered User Guide](https://github.com/mcroning/LCProp/blob/main/docs/user/user_guide.md)
and [documentation index](https://github.com/mcroning/LCProp/blob/main/docs/README.md).
Online pages follow evolving public `main`; pinned installations may differ. Built-in
Help corresponds to your installation.

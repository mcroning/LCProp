# LCProp Desktop Quick Start

Use this guide to install and open the LC and photorefractive (PR) desktop
applications. For headless Python use, see the [engine quick start](../developer_quickstart.md).
The [Colab notebooks](../../README.md#optional-colab-notebooks) are an optional
browser interface with their own published release and qualification.

## Desktop source reference

The native LC and PR startup check used these sources on Apple silicon
with CPython 3.12.13 and Qt's macOS Cocoa backend.

| Component | Source revision | Distribution label |
| --- | --- | --- |
| LCProp | [`9d3fa6abe074a298843a800cbc4ece1226bb4ba6`](https://github.com/mcroning/LCProp/commit/9d3fa6abe074a298843a800cbc4ece1226bb4ba6) | `0.4.0` |
| LaunchPlane | [`b71d125eb08074b0992dc849a5f4fc39e45eabc2`](https://github.com/mcroning/LaunchPlane/commit/b71d125eb08074b0992dc849a5f4fc39e45eabc2) | `0.1.0.dev0` |

LaunchPlane supplies the beam editor. LCProp requires exactly schema 4 and the
matching preview API. Use the companion revision in the table. Package version
labels alone do not identify either checkout.

The maintainer visually checked the startup input-intensity preview and the
waiting state for longitudinal cuts in both applications. Both exited cleanly.
Minimum window size remains a layout follow-up. The check used staged LCProp
source, LaunchPlane files from the pinned companion build, and hash-verified
dependency wheels. A subsequent fresh install following the commands below
also passed on macOS 26.6.2 arm64 with Python 3.12.13. The maintainer opened and
closed both installed apps at defaults. Use the source installation below for
this pairing. Other operating systems, production solves, GPU use, and Slurm
are outside these startup checks.

## Install from source in a clean environment

These commands target macOS 14 or later on Apple silicon, as required by the
recorded dependency wheels. Use an arm64 Python 3.12.13 interpreter and run
the commands from a new, unused directory on local storage. Save the
[desktop runtime requirements](desktop-macos-arm64-requirements.txt) from this
guide alongside the two checkouts as `desktop-macos-arm64-requirements.txt`.
Pip checks each runtime wheel against the recorded SHA-256 and stops if a
matching wheel is unavailable. Stop if any command fails. The build tools used by pip are resolved
separately, so the resulting local builds may differ byte-for-byte from
other builds of the same source.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -c 'import platform, sys; assert (sys.platform, platform.machine(), sys.version_info[:3]) == ("darwin", "arm64", (3, 12, 13))'

git clone --no-checkout https://github.com/mcroning/LaunchPlane.git
git -C LaunchPlane checkout --detach b71d125eb08074b0992dc849a5f4fc39e45eabc2
git clone --no-checkout https://github.com/mcroning/LCProp.git
git -C LCProp checkout --detach 9d3fa6abe074a298843a800cbc4ece1226bb4ba6

python -m pip install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r desktop-macos-arm64-requirements.txt
python -m pip install --no-deps ./LaunchPlane './LCProp[gui]'
python -m pip check
python -c 'from launchplane.serialization import SCHEMA_VERSION; assert SCHEMA_VERSION == 4, SCHEMA_VERSION'
git -C LCProp rev-parse HEAD
git -C LaunchPlane rev-parse HEAD
python -m pip freeze
```

### Launch the applications

From a Terminal in your macOS desktop session, activate the environment and
launch LC or PR from the working directory.

```bash
python -m lcprop.lc.gui.app
```

```bash
python -m lcprop.pr.gui.app
```

The PR installation also provides the `lcprop-pr` command. For editable
development, follow the [developer quick start](../developer_quickstart.md)
and the [test tiers](../developer_testing.md) in a separate environment.

The exercises below are small starting points. Use the [User Guide](user_guide.md)
for workflow details. The online guide follows public main, so individual
controls may differ in the pinned desktop revision.

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
Record the LCProp commit, experiment file, backend/precision, and result
retention policy with any scientific result.

Continue with the [LCProp User Guide](user_guide.md) before a large or
publication-facing run.

The GUI **Help** menu renders concise topics without an editor. Each topic links
to the [rendered User Guide](https://github.com/mcroning/LCProp/blob/main/docs/user/user_guide.md)
and [documentation index](https://github.com/mcroning/LCProp/blob/main/docs/README.md).
Online pages follow evolving public `main`; pinned installations may differ. Built-in
Help corresponds to your installation.

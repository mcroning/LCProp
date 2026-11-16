# LCProp

LCProp is a Python calculation engine for scalar optical beam propagation in
liquid-crystal (LC) and photorefractive (PR) media. The desktop GUI is the primary
frontend, with optional Colab notebooks and direct Python use. Core calculations
do not require a GUI or LaunchPlane.

## Start here: desktop GUI

The **LC and PR desktop applications are LCProp's primary user interfaces**.
Use them to configure beams, material models, grids and solvers, run calculations,
and inspect results.

- **Install and launch the desktop applications:** [desktop quick start](docs/user/quick_start.md), including the separate LaunchPlane schema-4 companion.
- **Use the desktop controls and workflows:** [user guide](docs/user/user_guide.md).
- **Find supported features and limits:** [implementation status](docs/STATUS.md).

The desktop quick start identifies its source and companion references and their
qualification limits. Use a separate environment; the Colab prerelease checks
below do not qualify a different desktop installation.

## Optional Colab notebooks

Colab provides an alternative browser-based interface with a narrower supported
workflow set than the desktop applications. It does not replace the desktop GUI.

For the CPU-first notebook prerelease, see [LCProp 0.5.0rc3](https://github.com/mcroning/LCProp/releases/tag/v0.5.0rc3) and the [wheel install instructions](https://github.com/mcroning/LCProp/blob/v0.5.0rc3/docs/releases/0.5.0rc3.md). The notebooks download and verify the exact rc3 wheel automatically in a fresh CPU runtime.

- **LC notebook:** [![Open LC notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcroning/LCProp/blob/v0.5.0rc3/notebooks/releases/0.5.0rc3/lcprop_lc_colab.ipynb) · [Download notebook](https://github.com/mcroning/LCProp/releases/download/v0.5.0rc3/lcprop_lc_colab.ipynb)
- **PR notebook:** [![Open PR notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcroning/LCProp/blob/v0.5.0rc3/notebooks/releases/0.5.0rc3/lcprop_pr_colab.ipynb) · [Download notebook](https://github.com/mcroning/LCProp/releases/download/v0.5.0rc3/lcprop_pr_colab.ipynb)

The published rc3 notebooks expose LC static/TD and reduced x-only PR static/TD
on CPU. Full-transverse PR and LC TD completed-result ZIP export are not included.
Ongoing notebook and GPU development is separate from these published artifacts.

The [historical CPU demonstration](https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb)
pins its own historical engine/helper versions and has its own stated limits.

## Engine and development

The engine can also be used without either frontend.

- **Build and install the engine:** [install in a new environment](docs/developer_quickstart.md).
- **Contribute or customize:** [contribution guide](CONTRIBUTING.md) and [extension map](docs/developer_extension_map.md).
- **Explore:** [documentation index](docs/README.md), [architecture](docs/architecture/LCProp_Target_Architecture.md), and [license](LICENSE).

## Which version?

For desktop, use the source pair in the [desktop quick start](docs/user/quick_start.md).
The earlier installed-package qualification reference is commit
`4299ec618810f63ed39431ad1cb374f8f7e6cb93` (distribution label `0.4.0`). Its recorded
installed-package qualification was 907 passed, 24 skipped and two excluded large
checks, plus 30 focused review checks. These are dated, bounded qualifications;
they do not establish every workflow, parameter choice, dependency version or GPU.
Use the commit identity rather than the version label alone. The
[previous detailed README](https://github.com/mcroning/LCProp/blob/4299ec618810f63ed39431ad1cb374f8f7e6cb93/README.md)
is retained at that exact historical revision.

Main includes changes after that qualification. For a wheel built from Git,
record the source commit and wheel SHA-256. The distribution version alone
does not identify which source revision produced a wheel.

## Desktop Beam-editor compatibility

The physical-intent Beam integration requires LaunchPlane **schema 4**;
schema-3 files and the old beam API are intentionally incompatible. There is
no automatic migration. See the [compatibility notice](docs/architecture/beamplane_physical_launch_contract.md#schema-4-compatibility)
before recreating or explicitly re-exporting obsolete definitions. The
calculation engine remains independent of this optional GUI companion.

## Development boundaries

LC and PR own their physics and workflows. Shared propagation accepts prepared
material responses, and shared result records support processing and presentation.
A new material, processing step or frontend should be developed in a fork and a
separate environment. There is no general third-party plugin SDK promised here.

Pull requests propose upstream changes; they do not update anyone's installed
working environment. Maintainer approval, package publication and installation
are separate steps. See [CONTRIBUTING](CONTRIBUTING.md) for the sole-maintainer
review policy and its limits.

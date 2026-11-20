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

The **PR workbench** is available in [LCProp 0.5.0rc4](https://github.com/mcroning/LCProp/releases/tag/v0.5.0rc4). It offers reduced x-only PR Static and time-dependent evolution on CPU, with beam editing, saved results and collapsed advanced controls. See the [rc4 opening and install instructions](https://github.com/mcroning/LCProp/blob/v0.5.0rc4/docs/releases/0.5.0rc4.md).

- **Open PR workbench:** [![Open PR workbench in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcroning/LCProp/blob/v0.5.0rc4/notebooks/pr_workbench_source.ipynb) · [Download notebook](https://github.com/mcroning/LCProp/releases/download/v0.5.0rc4/LCProp-PR-workbench-0.5.0rc4-candidate.ipynb)
- **Required PR setup ZIP:** [LCProp-PR-workbench-setup.zip](https://github.com/mcroning/LCProp/releases/download/v0.5.0rc4/LCProp-PR-workbench-setup.zip). Upload it through Colab's Files sidebar, keep it zipped, then run **Open PR workbench**. If asked, restart the session once and rerun that same cell. The notebook verifies the setup ZIP and its exact wheel; no separate wheel upload or Drive mount is needed.
- **Verify the rc4 downloads:** [SHA256SUMS](https://github.com/mcroning/LCProp/releases/download/v0.5.0rc4/SHA256SUMS). The release includes the wheel, notebook, setup ZIP, candidate notes and checksum manifest.
- **LC notebook remains on rc3:** [![Open LC notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mcroning/LCProp/blob/v0.5.0rc3/notebooks/releases/0.5.0rc3/lcprop_lc_colab.ipynb) · [Download notebook](https://github.com/mcroning/LCProp/releases/download/v0.5.0rc3/lcprop_lc_colab.ipynb) · [rc3 install instructions](https://github.com/mcroning/LCProp/blob/v0.5.0rc3/docs/releases/0.5.0rc3.md). This separate notebook downloads and verifies its pinned rc3 wheel automatically.

The rc4 PR opening flow is CPU-only; full-transverse PR material transport and new GPU qualification are outside its scope. Existing library GPU guards remain unchanged. The LC rc3 notebook offers static/TD on CPU; LC TD completed-result ZIP export is not included. Prior releases remain available. Both the tagged PR source notebook and the downloadable candidate notebook use the exact same opening code and pins; their headings differ. The preserved rc4 notebook still uses preview/candidate wording.

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

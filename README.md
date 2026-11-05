# LCProp

LCProp is a Python calculation engine for scalar optical beam propagation in
liquid-crystal (LC) and photorefractive (PR) media. Desktop and notebook interfaces
sit alongside the engine; core calculations do not require a GUI or LaunchPlane.

## Start here

- **Try a notebook:** [historical CPU demonstration](https://colab.research.google.com/github/mcroning/LCProp/blob/d5b4300e4ccf14d2e89292afcc8f6c8fd378b9e5/notebooks/lc_static_interactive_cpu.ipynb).
  This demonstration pins its own historical engine/helper versions; it does not
  exercise the tested release identified below. Review its stated limits before use.
- **Use the tested engine:** [install in a new environment](docs/developer_quickstart.md).
- **Contribute or customize:** [contribution guide](CONTRIBUTING.md) and
  [extension map](docs/developer_extension_map.md).
- **Explore:** [documentation index](docs/README.md),
  [architecture](docs/architecture/LCProp_Target_Architecture.md), and [license](LICENSE).

## Which version?

The tested reference for this public entry is commit
`4299ec618810f63ed39431ad1cb374f8f7e6cb93` (distribution label `0.4.0`). Its recorded
installed-package qualification was 907 passed, 24 skipped and two excluded large
checks, plus 30 focused review checks. These are dated, bounded qualifications;
they do not establish every workflow, parameter choice, dependency version or GPU.
Use the commit identity rather than the version label alone. The
[previous detailed README](https://github.com/mcroning/LCProp/blob/4299ec618810f63ed39431ad1cb374f8f7e6cb93/README.md)
is retained at that exact historical revision.

Current public-entry documentation may be newer than that reference while keeping
runtime source unchanged. Other historical and experimental branches remain
available; do not assume a branch name identifies a validated scientific result.

## Optional Beam-editor compatibility

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

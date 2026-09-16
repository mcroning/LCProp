# LCProp Release Hygiene and RC Preparation

## Scope

This milestone prepares a truthful release-candidate source state without
changing scientific behavior. It covers installation, test dependencies,
workspace hygiene, third-party attribution, LC Slurm status language, package
qualification, and release-gate documentation.

No LC or PR equation, solver, integrator, request, schema, codec, retention
policy, backend algorithm, propagation rule, or scientific diagnostic changed.

## Installation policy

The headless LCProp core retains Python 3.10 support. The complete GUI and
normal Product-test environment require Python 3.11 or newer because the
separate LaunchPlane Product requires it. LCProp requires LaunchPlane schema 3
or newer. The compatible source revision reviewed with this Product state is
`add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`.

LaunchPlane remains a separately installed public Product. Its current package
version does not uniquely identify schema-3 compatibility, so the Product
documentation points to the public source repository rather than encoding a
Git SHA as a runtime dependency. A separately reviewed LaunchPlane version or
package release remains a formal-package release gate.

LCProp now provides a `test` extra containing pytest and Pillow. A normal full
local Product environment installs LaunchPlane and `LCProp[gui,test]`. CUDA,
cluster, and separately supplied historical-evidence checks remain conditional.

## Workspace and private reference disposition

The tracked `src/lcprop/core/LCProp.code-workspace` file had no runtime or test
consumer and contained a personal interpreter path. It was removed rather than
replaced with another workstation assumption.

The historical `reference/prprop/prprop3d.py` copy had unknown ownership, no
recorded license, and no Product runtime or test dependency. Its exact 61,640
bytes and SHA-256
`0adb5c824459ebaa31adbf9cad541c9ef634ac63f2b83bf42e8c3ee5a62ed28f`
were preserved privately at LCProp-Research commit
`4395a9070ca8330dc9f659e6691101623b73b528` as
`reference_implementations/prprop/prprop3d.py`, classified
`RIGHTS_RESTRICTED_PRIVATE`. Remote verification completed before the Product
candidate removed its copy. Private preservation does not establish public
redistribution rights.

## Publication attribution

Five retained comparison images contain published Figure 6 pixels. The source
article is CC BY 4.0, the existing research reports record the publication and
transformations, and `THIRD_PARTY_NOTICES.md` now provides one Product-level
attribution and an exact affected-file list.

## LC Slurm status

Canonical NumPy static LC Slurm execution is implemented for CPU resource
profiles, but retained real scheduler commissioning evidence does not yet
exist. Current documentation and the GUI now say `commissioning pending`.
TD, soliton, and soliton-existence remote execution remain unsupported. No
cluster job was run and implementation availability was not changed.

## Package and version policy

The source distribution and wheel are built and inspected outside the Product
tree. Installed-artifact qualification verifies imports, packaged PR runtime
calibration/catalog data, entry-point resolution, and representative LC/PR
request construction. Automatic Slurm source deployment remains explicitly a
clean-Git-checkout workflow; a wheel alone is not a deployment source.

The final candidate produced both `lcprop-0.1.0.tar.gz` and
`lcprop-0.1.0-py3-none-any.whl` with `python -m build`. Both artifacts contain
the license and third-party notice, and neither contains the personal workspace
file or the private PRProp reference. The wheel metadata carries the declared
Python floor, optional extras, and `lcprop-pr` entry point. A temporary
non-editable Python 3.12 installation resolved `lcprop` from the installed
wheel, loaded both packaged PR resource sets, resolved the entry point, and
constructed representative LC and PR requests. The focused release tests
passed `13 passed`; the complete Product suite passed `1666 passed, 77
skipped`. Syntax compilation and whitespace checks are part of the final
candidate gate.

The package version remains `0.1.0` in this source-preparation milestone. The
next version change should be a separate explicit and reviewed `0.1.0rc1`
gate after this candidate is integrated and the LaunchPlane distribution gate
is resolved. No tag or release is created here.

The source tree is release-hygiene complete but is not yet a formal RC: the
LaunchPlane distribution identity and the explicit version/tag decision remain
open. LC static Slurm commissioning may remain pending for an RC because the
limitation is now explicit and the uncommissioned path is not presented as
validated.

## Remaining gates

- publish or otherwise version a schema-3-compatible LaunchPlane package;
- perform LC canonical static CPU/NumPy Slurm commissioning, or retain its
  explicit commissioning-pending status for an RC;
- execute the separate version/tag/release decision after candidate review;
- continue historical Product/Research curation independently.

The completed PR GUI/package master plan is now historical and should be
archived through the established Research lifecycle after this milestone is
reviewed, committed, and integrated. The active fanning prompts remain active.

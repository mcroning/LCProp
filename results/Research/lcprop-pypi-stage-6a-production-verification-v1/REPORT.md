# Stage 6A — production distribution verification

**PASS: LCProp 0.3.0 production distribution qualified for the tested installation paths.**
No scientific simulation, parameter sweep, source modification, commit, tag, push,
upload or credential access occurred. No user environment was altered.

## Results

| Gate | Result | Evidence |
|---|---|---|
| Published artifact identity | PASS | Exactly wheel + sdist; PyPI JSON digests and independently downloaded bytes match Stage 5B |
| Fresh wheel installation | PASS | Production index, binary-only LCProp; wheel-install.json/log |
| Independent fresh sdist installation | PASS | Production index, no cached wheel, no-binary=lcprop; sdist-install.json/log |
| Runtime identities | PASS | All 267 qualified hashes match in both installs and after Colab-extra installation |
| Headless core | PASS | Both fresh environments before extras: no Qt, CuPy, LaunchPlane, Matplotlib, IPython or ipywidgets |
| Colab extra | PASS | Production extra resolves; packaged widgets initialize with network/solver calls trapped |
| Deployment compatibility | PASS | resolve_installed_distribution accepts both installations and Colab installation |
| Dependency consistency | PASS | pip check reports no broken requirements in core environments and after extra installation |
| Metadata, licenses and README | PASS | Published JSON matches artifact metadata and committed README; license files exact |
| README destinations | PASS | All 14 unique authored destinations returned anonymous HTTP 200 |
| Live project-page visual inspection | LIMITED | PyPI served a JavaScript challenge; API README checked and rendered locally |

## Provenance and artifacts

Qualified source: `fa8c3b292f2cfd38379870d0c8d085f5ff0c3f78`.
Reference: `../lcprop-pypi-stage-5b-production-release-v1/REPORT.md`.
Anonymous endpoint: https://pypi.org/pypi/lcprop/0.3.0/json

| Artifact | SHA-256 |
|---|---|
| `lcprop-0.3.0-py3-none-any.whl` | `9ab34f4ec244716b3e6f0431e4cac1e0d8a1ebeb17a7064c6422f1ed6b4e3d60` |
| `lcprop-0.3.0.tar.gz` | `6ca4fab1b4d0a5343f9094af39b00a2437cd203da05609d0f10a93935a1eb399` |

PyPI supplies exactly these two artifacts. Their original source binding follows the
Stage 5B clean committed-source build and byte-identical artifact comparison. Pip's
installation reports independently record files.pythonhosted.org URLs and the same
hashes: the sdist path was fetched as .tar.gz and built in isolation, not substituted
with the published wheel. Logs record build steps; the runtime hashes then match.

Fresh environments: `.codex-work/pypi-stage6a/wheel-env` and `sdist-env`, created with
venv and no system site packages. Installation used isolated pip configuration and
explicit `https://pypi.org/simple`. Only LCProp was restricted to wheel or source;
normal dependency wheels were allowed. The Colab extra was subsequently installed
into wheel-env after successful core-only checks. No development checkout is used
for imports. Verification ran with Python -I -B.

## Environment and execution boundary

Python: `3.12.13 | packaged by Anaconda, Inc. | (main, Mar 19 2026, 20:12:32) [Clang 20.1.8 ]`. Platform: `macOS-26.6.2-arm64-arm-64bit`.

| Dependency | Resolved version |
|---|---|
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |
| ipywidgets | 8.1.9 |
| ipython | 9.17.1 |

Full dependency inventories, site-packages paths and deployment digests are in the
three `*-verification.json` files; installation provenance is in `*-install.json`.
Widgets initialize with a launch-plane preview only, an empty completed snapshot,
120-second default budget and correct Configure labels. Network entry points,
interactive execution helper and LocalRunner.run_operation were replaced with
raising guards during initialization. No solver executed. Scientific parity evidence
is retained from Stage 5B/RC qualification; no new numerical/platform accuracy claim.

## Published documentation

Version 0.3.0, BSD-3-Clause, license files LICENSE and THIRD_PARTY_NOTICES.md,
Python >=3.10 and all dependency/extra declarations match the accepted artifact.
Core requires NumPy/SciPy (conditional tomli on older Python); Qt and CuPy remain
optional. Colab adds Matplotlib, ipywidgets and IPython. LaunchPlane remains separate.
The published README exactly matches the source commit, prioritizes ordinary pip
installation, separates developer instructions, links the historical immutable Colab
demonstration and limits scientific claims. No relative documentation destinations.
See `links.json`, `published-README-rendered.html` and `pypi.json`.
The captured `project.html` is a challenge page, not evidence of visual rendering.
A normal-browser visual check remains useful; this does not block artifact/install
qualification. Hosted Colab was not rerun, nor were GUI/GPU/Slurm environments qualified.

## Preservation

All operations were confined to disposable environments/downloads and this new report
directory, with read-only access to source and retained evidence. Existing unrelated
tracked/untracked changes and protected evidence were preserved. No additional
publication action is required to make this already published release available.

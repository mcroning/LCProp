# Stage 3A — LCProp 0.4.0rc1 candidate preparation

**PASS for local candidate preparation; ready for review and an independently authorized immutable release commit. Not yet hosted-Colab qualified or authorized for upload.** Accepted implementation: `896c9360129a1123dc922d4799281d6b2d8f43c1`. No scientific-engine, request-schema, notebook or solver changes. No propagation simulations.

## Candidate inventory

Exactly three tracked release changes:

1. `pyproject.toml`: version 0.3.0 → **0.4.0rc1**; add **ipympl>=0.10.0,<0.11** to `colab` only. Core, GUI, GPU, test dependencies and Python declaration unchanged.
2. `README.md`: candidate two-beam controls, graphical Move/Tilt, shared static fallback, supported short/legacy imports, backend/Colab manager setup, snapshot ownership and qualification limits. Clarifies that the earlier nine-array qualification refers to the 0.3.0 release family, not this new RC.
3. `docs/notebook_package.md`: candidate installation/backend/import guidance and separation from the published notebook.

The existing `notebooks/lc_static_pypi_cpu.ipynb`, its **lcprop[colab]==0.3.0** pin, Git-pinned notebook, and saved `lc_static_local.ipynb` outputs are unchanged. Both import paths are supported in the candidate; ordinary README installation guidance remains version-neutral for published PyPI versions, while the candidate is explicitly not yet published.

This report directory contains the qualification evidence and exact inventories. `source-manifest.json` lists every build-input path/hash; `wheel-inventory.json` and `sdist-inventory.json` list every archive file/hash; `runtime.json` lists all 269 package files. No additional candidate source changes were made.

## Frozen source and provenance

All accepted Stage 2 inventory hashes were verified before editing. Retained logs establish **56 + 5 = 61** passing tests and user-reported local browser acceptance. The runtime files in the candidate were obtained from `git show` at the accepted commit, **not copied from the dirty working tree**. The only overlays are the three release files above. Selected root packaging files, package source, two compatibility examples, notebook-package documentation and the existing selected sdist test were exported according to the existing packaging policy.

Immutable input archive (read-only mode 0444):

`.codex-work/beam-rc1/candidate-source.tar`

SHA-256: `33d7c19ae42233a4fb593a022d12169c6f6461defb5f4c090e6a09499ac49406`

Build source was unpacked from that payload into `.codex-work/beam-rc1/source`; build-generated egg-info/build files are disposable and not source overlays. Source content is identified by the archive hash plus manifest, **not by a new Git commit**. No commit was created. The accepted commit plus approved overlay bytes fully defines this candidate. A future committed-source build must be compared against these runtime and archive member identities; archive-byte reproducibility across different build times has not been claimed.

The unrelated dirty `src/lcprop/pr/published_static.py` and `tests/test_pr_published_static.py` were excluded. Their hashes and saved notebook hashes were checked unchanged at completion. Scientific runtime files, including the PR source in the artifacts, match the accepted commit.

## Artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `.codex-work/beam-rc1/dist/lcprop-0.4.0rc1-py3-none-any.whl` | 760065 bytes | `2cc0ede68970b2d7de08dc385597861ab1706ad7b08fa4fa2169c5539f8e396e` |
| `.codex-work/beam-rc1/dist/lcprop-0.4.0rc1.tar.gz` | 670995 bytes | `7f3367ea5a2940b8cd2256ec9c93ab44bfae2f3907798dda1158c96b1fdf2738` |

Wheel: **276 files**, including **269 runtime files** and distribution metadata/licenses. Sdist: **289 regular files**. Every wheel runtime file equals the frozen source and corresponding sdist member byte-for-byte, including `_beam_position_editor.py`, `_launch_preview.py`, calibration/catalog JSON and PNG resources. Wheel README exclusions remain intact; documentation remains in the sdist. No results, reference data, `.codex-work`, notebooks, ZIPs, wheels or NumPy array outputs are included in the sdist. Only committed accepted package source and approved overlays entered the build.

**`twine check --strict`: PASS for both artifacts.** Metadata reports version 0.4.0rc1, BSD-3-Clause, Python >=3.10 and the expected extras. README markup was validated by strict metadata checking; no direct browser-rendering review was performed this turn. Existing published links were preserved, not repointed to an unavailable new release.

## Build and dependency environment

Local macOS 26.6.2 arm64, Python **3.12.15**. A new disposable build environment used build **1.6.1**, setuptools **84.0.0**, wheel **0.48.0**, Twine **7.0.0**. Both artifacts were built with `python -m build --no-isolation` using those installed tools; the standard build also builds the wheel from the sdist. All environment paths are under `.codex-work/beam-rc1/`. An initial attempt with an older disposable build environment lacked setuptools and failed before producing artifacts; a fresh build environment resolved that tooling issue. No existing user environment was modified.

Fresh independent core, colab and sdist environments used production PyPI dependencies. Actual resolution:

| Dependency | Resolved version |
| --- | --- |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| ipympl | 0.10.0 |
| Matplotlib | 3.11.2 |
| ipywidgets | 8.1.9 |
| IPython | 9.17.1 |
| traitlets | 5.16.1 |
| Pillow | 13.0.0 |
| pytest (test tooling, separately added only to colab test env) | 9.1.1 |

Full resolution URLs/archive identities and versions are retained in `dependency-resolution.json` and the three verification JSON files. Core and sdist environments contain only LCProp, NumPy, SciPy and pip on Python 3.12. The conditional tomli requirement is not selected. ipympl's transitive constraints remain optional; no Qt, CuPy or LaunchPlane package was installed. Pip cache/fontconfig sandbox warnings were nonfatal; Matplotlib completed first-use font initialization. Existing Toolbar deprecation warnings remain.

## Installed-package checks

Checks ran from `/private/tmp` or the separate `.codex-work/beam-rc1/harness` directory, not a source checkout. `lcprop.__file__` resolved to each environment's **site-packages**, verified in the retained JSON. The test harness copies accepted tests/fixtures only and has no `src` tree or repository conftest.

| Check | Result |
| --- | --- |
| Fresh core wheel installation | PASS |
| Independent sdist build/install in second fresh environment | PASS |
| Separate optional colab-extra installation | PASS |
| 269 installed runtime hashes, all three environments | PASS; exact candidate payload |
| Installed distribution resolver | PASS, all three environments |
| `pip check` | PASS, all three environments |
| Core imports with notebook/Qt/CuPy packages absent | PASS |
| Short and legacy InteractiveLC imports | PASS; same class; neither imports notebook libraries merely to expose it |
| Qualified preset request equality with canonical builder | PASS |
| Static controller initialization | PASS, Agg backend and no editor |
| Interactive controller initialization and Beam 2 edit | PASS, real ipympl backend with synthetic standard frontend messages |
| Retained Stage 2 installed regressions | **56 passed**, 46.56 s |
| Real-ipympl installed frame tests | **5 passed**, 22.74 s; five Toolbar warnings |
| Installed package import/request-launch/initialization tests | **3 passed**, 1 propagation case deliberately deselected, 17.33 s |

Thus all **61 Stage 2 tests pass again against the installed candidate**, plus three bounded package checks. Coverage includes canonical coherent/incoherent request and field construction, coordinate/tilt geometry, phase behavior, numeric/graphical equivalence, frozen requests, metadata association, delayed events, failed presentation preserving prior records, and byte-identical snapshot exports. Execution helpers are forbidden or faked in the bounded tests. No scientific propagation was run.

The interactive check is Python/backend qualification, **not a real Colab frontend or browser-painting test**. Stage 2 local browser acceptance is retained user-reported evidence. No new full-repository pass is claimed; the known unrelated historical notebook test failure was not included or altered.

## Retained numerical parity

No new numerical outputs were generated. The nine retained Stage 0.3.0-family package/reference arrays were reloaded and compared for exact shape, dtype and C-order bytes, with recorded hashes reverified (`retained-parity.json`). These remain exact. The accepted Stage 1 BeamPlane evidence separately records exact canonical comparisons for the unchanged single-beam preset and the bounded coincident incoherent pair, with canonical launch/phase checks for coherent/incoherent settings. Stage 2 and the candidate preserve the scientific engine and `lc_static_demo.py` bytes.

This demonstrates preservation of retained evidence and installed request/launch behavior; it is not a fresh nonlinear execution qualification under the newly resolved environment. General separated/coherent multibeam nonlinear behavior and soliton formation remain unqualified claims.

## Reproduction and remaining gates

Build inputs and full logs remain under `.codex-work/beam-rc1/`. Example candidate qualification commands (already executed, not production installation instructions):

```sh
.codex-work/beam-rc1/build-env/bin/python -m build --no-isolation --outdir .codex-work/beam-rc1/dist .codex-work/beam-rc1/source
.codex-work/beam-rc1/build-env/bin/python -m twine check --strict .codex-work/beam-rc1/dist/lcprop-0.4.0rc1-py3-none-any.whl .codex-work/beam-rc1/dist/lcprop-0.4.0rc1.tar.gz
```

1. Review the three-file release diff and this evidence; separately authorize an immutable RC commit. Build from that committed source and verify member/runtime identities before treating any artifacts as publication-final.
2. Prepare the hosted test notebook separately; preserve the published 0.3.0 notebook. Qualify fresh hosted CPU setup with the standard custom widget manager and widget backend, first Beam 2 activation, toggles, Move/Tilt, numeric commits, static fallback, repeated initialization and snapshot/export behavior.
3. Authorize any future bounded scientific reference execution independently. Use cross-platform numerical comparisons, not an unsupported universal byte-equality claim.
4. Separately authorize exact-artifact TestPyPI upload/install qualification, then eventual production release steps. No uploads, tags or pushes occurred here.

**No stop condition triggered:** runtime payload matches accepted source exactly, dependencies remain optional, and installed identities match. Remaining work is release provenance/hosted qualification, not a scientific implementation correction. All candidate changes remain unstaged and uncommitted; unrelated work and protected evidence are preserved.

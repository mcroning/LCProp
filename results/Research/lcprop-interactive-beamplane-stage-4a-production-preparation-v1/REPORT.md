# Stage 4A — Production 0.4.0 preparation

**GO for production-release commit review of this exact candidate.** No commit, tag, push or upload has been performed. This gate is not authorization to publish. Scientific runtime files are unchanged from the hosted-qualified RC1; new validation is limited to packaging, installed imports, rendering and bounded fake-execution regressions.

## Candidate scope and immutable source

Base: `2e195579dc3d76f6477bf6c885fd537043acdf9d`; hosted closure: `d2f7f38b2f3e40cbe05fc6f7e8a3d519362d659b`.

Exactly three release overlays:

- `pyproject.toml`: 0.4.0rc1 → 0.4.0. Parsed configuration comparison confirms this is the only packaging/dependency change.
- `README.md`: current two-beam/Move/Tilt functionality, frozen completed-record ownership, accepted bounded RC1 hosted results, optional dependencies and required Colab restart sequence. Historical public demonstration preserved.
- `docs/notebook_package.md`: release-facing local/hosted setup, manual restart and small version/pip verification, existing-widget Colab presentation, static fallback, reuse and scientific limits. No reference-array or release-qualification machinery is placed in ordinary setup.

Exact overlay hashes: `candidate-overlays.json`. No new production notebook was created: the 0.3.0 pinned notebook and saved local notebook outputs remain byte-identical. The qualified RC1 notebook remains unchanged. Notebook replacement still needs separate review/publication.

The working tree contains known unrelated edits to `src/lcprop/pr/published_static.py` and `tests/test_pr_published_static.py`. They were fingerprinted and preserved, not copied into the candidate. The export was assembled from accepted Git object contents using the same 282-file build-input allowlist as RC1, plus only the three reviewed overlays. Every one of its 269 runtime hashes equals RC1. No additional working-tree runtime source was silently incorporated.

Frozen export: `.codex-work/beam-040/candidate-source.tar`, read-only mode 0444, SHA-256 `57bc92690363e83ff7d24c72cccbd7db4c72f088941769051b24350c861de70e`. This is an uncommitted manifest-defined candidate, not a claim that a new production commit already exists. `source-manifest.json` lists all 282 exact build inputs. Build used an extracted disposable copy of that frozen archive; the archive itself is unchanged.

## Artifacts

Retained directory: `/Users/mcroning/LCProp/.codex-work/beam-040/dist/`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| lcprop-0.4.0-py3-none-any.whl | 760371 | `1bd9e9ba8f7226a4a596221368007d667009a84fe1c90129bcc2ea0296695404` |
| lcprop-0.4.0.tar.gz | 667346 | `74ff25fb7b97197d244690868125a7cd717d6306f2425e8381cb02787ea2944d` |

Wheel: 276 files. Sdist: 289 regular files. Both contain the expected **269 runtime files, all byte-identical to RC1**, including the notebook editor/preview modules and package resources. The sdist also retains its approved docs/examples/test file. Complete member inventories are retained beside this report.

Artifact scanning found no notebook outputs, results/reference/.codex-work payloads, raw scientific arrays, nested release artifacts or private evidence. Bounded common-token/private-key/credential-URL checks found no hits, and decoded non-PNG content contains no /Users/ or /home/ paths. This is bounded screening, not proof against every conceivable secret format. No credential stores were accessed.

## Build and metadata

Existing build environment: Python 3.12.15, macOS arm64; tool versions are recorded in `build-environment.json`. `python -m build --no-isolation` built wheel and sdist from the disposable clean export. No repository source directory or existing build output was used as the package input. The independent sdist installation uses pip's isolated backend build from that exact sdist.

`twine check --strict`: **PASS for both artifacts**. Explicit UTF-8 archive metadata parsing confirms README payload equality; name, Python requirement, requirements/extras and license fields match RC1 except version and the intended README content. License is BSD-3-Clause with LICENSE and THIRD_PARTY_NOTICES.md. Requires-Python remains >=3.10; this is not a claim of new platform commissioning. Project-URL metadata remains unset as in RC1; adding unrelated metadata changes was avoided.

Core requirements remain numpy/scipy and conditional tomli. `colab` remains matplotlib/ipywidgets/IPython plus **ipympl>=0.10.0,<0.11**. GUI and GPU dependencies remain separate extras. No optional graphical dependency leaked into the core installation.

README rendering in the existing Markdown-capable environment produced 25,139 characters of sanitized HTML. Both guides' Python blocks passed syntax checks after excluding IPython magic lines. Fifteen remote Markdown links resolved with HTTP 200; Colab rejects HEAD (405), so its link was checked successfully with GET. These are availability checks, not a new hosted browser acceptance. Historical immutable links remain historical; the new branch-based notebook-guide link currently serves the earlier public guide until the reviewed documentation is published. README explicitly notes this, and the sdist includes the matching release guide. No publication occurred to advance that link during this task.

## Fresh installed-package validation

Separate new environments were created for core wheel, wheel[colab] and sdist. Each installed from the exact retained artifact with production PyPI dependencies, without upgrading existing user environments. All three `pip check` results pass. Compact dependency transaction records and full installed-version inventories are retained. Full pip/build logs remain in `.codex-work/beam-040/`.

Checks ran from /private/tmp or an isolated test harness with no src tree/conftest, not from an importable checkout. All three installed copies pass 269-file hash verification, installed-distribution deployment verification, same-object public/legacy InteractiveLC imports, and canonical default request equality. Core and sdist environments have no ipympl, Matplotlib, ipywidgets, IPython, Qt, LaunchPlane or CuPy. Core/static module imports do not pull those dependencies. The Colab environment passes static fallback and real-ipympl initialization with synthetic frontend messages and propagation explicitly blocked. This is backend validation, not a new Safari/hosted frontend claim.

Installed numerical/widget stack matches the prior local RC1 validation: NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, ipympl 0.10.0, ipywidgets 8.1.9 and IPython 9.17.1. Complete inventories include transitive versions.

## Bounded tests and resolved harness issues

Test source was compared byte-for-byte with the accepted release commit (11 harness test files). No propagation test was executed.

| Selection | Result |
| --- | --- |
| Notebook regression modules, excluding frame-delivery and scientific package comparison | 67 passed, 1 skipped (backend-specific check), 28 warnings, 50.19 s |
| Explicit ipympl backend: deferred transmission plus frame-delivery tests | 5 passed, 5 warnings, 2.85 s |
| Package imports, canonical request/launch parity, widget initialization | 3 passed; 1 propagation comparison deliberately deselected, 24.49 s |

The skipped backend-specific case was exercised successfully in the explicit-ipympl group. This expanded bounded selection covers the retained Stage 2 evidence; the historical 61-test receipt is preserved rather than relabeled as a newly identical test selection. Tests cover Move/Tilt/request parity, static/interactive rendering, stale-event lifecycle, fake-run isolation, metadata and snapshot/export ownership. No new numerical reference was generated.

Two test-harness setup errors were corrected without changing candidate files. A temporary audit script named inspect.py shadowed the standard-library inspect module; renaming it restored installed checks. An initial frame-test invocation omitted backend activation and correctly fell back to static; rerunning with MPLBACKEND=module://ipympl.backend_nbagg passed. Initial failure logs are retained locally. Fontconfig reported unavailable user-cache locations while generating fresh caches; checks still passed. Traitlets/ipympl Toolbar deprecation warnings are the known existing dependency behavior, not a runtime change introduced here.

## Scientific evidence carried forward

RC1 hosted Stage 3K independently verified 9/9 arrays under the original tolerances, 100/100 converged slices in 35.377490939999916 seconds, 6,275 iteration records, exact frozen request, and unchanged arrays/metadata/plots/exports after draft edits. Stage 3J reconciled active NumPy/SciPy origins, and user-observed graphical acceptance covered Beam 2/Move/Tilt/numeric editing. Colab still requires install → manual restart → verification → initialization; restart-free initialization remains unqualified.

These results attach to the retained RC1 evidence and the byte-identical 0.4.0 scientific runtime. This task did not rerun hosted science or claim fresh hosted installation of a version not yet published. Additional beams or exploratory parameter settings remain outside the scientific qualification. Production artifact identity must be reverified after publication.

## Preservation and final gate

Known PR modifications and both protected notebook files were hash-checked unchanged after preparation. Only the three version/documentation overlays and this report/evidence directory are new task changes; temporary files are under `.codex-work/beam-040/`. No engine, request-schema, controller, scientific normalization or dependency specification changes occurred.

**GO for commit review**, contingent on retaining these exact overlay, source-export and artifact identities. Review the manifest-defined candidate and outgoing history before any later commit/publication action. No pushes, tags or uploads are authorized here. Remaining production gates are explicit release commit acceptance, publication authorization and post-publication identity verification; a replacement ordinary 0.4.0 Colab notebook is a separate reviewed deliverable.

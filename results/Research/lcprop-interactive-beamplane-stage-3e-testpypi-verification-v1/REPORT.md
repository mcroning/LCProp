# LCProp 0.4.0rc1 — Stage 3E published artifact verification

**GO for hosted Colab qualification.** Anonymous retrieval of the exact TestPyPI release and independent downloads confirm that the published artifacts are the qualified Stage 3A artifacts, without substitutions. This is permission to proceed to qualification, not evidence that hosted graphics or scientific execution have passed.

Verification date: 2026-10-17. Release source: `2e195579dc3d76f6477bf6c885fd537043acdf9d`.

## Public publication and independent identity checks

[Public release metadata](https://test.pypi.org/pypi/lcprop/0.4.0rc1/json) was retrieved successfully over anonymous HTTPS. It lists exactly two files: one universal Python wheel and one source distribution. Both are unyanked. The prior Stage 3D 404 was historical; publication availability is now verified.

[Release page](https://test.pypi.org/project/lcprop/0.4.0rc1/).

| Public artifact | Bytes | Independently computed SHA-256 | Upload time (UTC) |
| --- | ---: | --- | --- |
| [lcprop-0.4.0rc1-py3-none-any.whl](https://test-files.pythonhosted.org/packages/d1/b1/a786d66e6dec5b890d13c722686ba670d78b0dfd35e31e69c8dfa6fbb401/lcprop-0.4.0rc1-py3-none-any.whl) | 760065 | `2cc0ede68970b2d7de08dc385597861ab1706ad7b08fa4fa2169c5539f8e396e` | 2026-10-17T18:21:01.338404Z |
| [lcprop-0.4.0rc1.tar.gz](https://test-files.pythonhosted.org/packages/99/fe/737018ed8342b920253d5e987cf9347cd24328dc9044038abbd5a829233e/lcprop-0.4.0rc1.tar.gz) | 670995 | `7f3367ea5a2940b8cd2256ec9c93ab44bfae2f3907798dda1158c96b1fdf2738` | 2026-10-17T18:21:02.615596Z |

For each download, the locally computed SHA-256 matches the user-specified qualified identity, the public JSON digest, and the retained artifact in `.codex-work/beam-rc1/dist/`. Direct byte comparisons against both retained artifacts also pass. No artifact was rebuilt or modified.

All **269 runtime files** in the downloaded wheel were independently compared byte for byte against `src/` at the release commit: **PASS**. Stage 3D retains the full source-manifest and wheel/sdist member-inventory correspondence. These artifacts originated from the frozen Stage 3A export subsequently committed as the release source; no new build-from-commit claim is made.

## Metadata checks

Wheel METADATA and sdist PKG-INFO agree on package name/version, Python requirement, dependencies, extras, license-file declarations and project-URL fields. Public dependency metadata matches the wheel exactly. The public long description matches the archive README payload after trimming surrounding whitespace.

- Package: `lcprop`, version `0.4.0rc1`; Python `>=3.10`.
- License expression: `BSD-3-Clause`; license files: `LICENSE`, `THIRD_PARTY_NOTICES.md`.
- Core: `numpy`, `scipy`, and `tomli` only for Python <3.11.
- `colab`: `matplotlib`, `ipywidgets`, `IPython`, `ipympl>=0.10.0,<0.11`.
- `gui`: `PySide6`, `pyqtgraph`, `matplotlib`.
- `gpu`: `cupy-cuda12x`.
- `test`: `pytest>=8`, `Pillow`.

The four extras are optional; ipympl, Qt and CuPy are not core requirements. Public project-URL metadata is unset, matching the qualified package rather than representing a publication discrepancy. No new browser inspection of README rendering was performed; prior strict metadata/rendering checks remain in Stage 3D.

## Prepared Colab notebook

`notebooks/lc_static_pypi_rc1_colab_qualification.ipynb` was inspected without execution. Its version, source commit, wheel filename, wheel SHA-256 and sdist SHA-256 match this publication. It remains output-free.

The notebook checks the exact public inventory and both advertised hashes, independently hashes the downloaded wheel, then installs that local verified wheel with the `colab` extra using production PyPI for dependencies. It checks all 269 installed runtime files, import ownership and dependency consistency. It does not substitute a checkout or an unverified source distribution. This audit downloaded and hashed the sdist independently as well.

Initialization enables Colab's custom widget manager and `%matplotlib widget` before creating the packaged controller; explicit static mode uses a fresh runtime with the inline backend. Repeated initialization reuses the controller. The existing production 0.3.0 notebook was not changed.

## Retained evidence and limits

New raw evidence is retained locally at `.codex-work/beam-stage3e/`: `release.json` and the two independently downloaded artifacts. These are audit downloads, not replacement publication artifacts. The raw metadata SHA-256 is `43ec305e534c859bf79c26fcdfce17baa5c7797c75d951eef4fbe217d2d44d28`.

Stage 3A retains the **61 installed Stage 2 test passes**, three package checks, core/colab/sdist installation evidence, import isolation and numerical-parity evidence. These tests were not rerun during this publication-only audit. Byte-identical publication preserves their artifact identity; it does not qualify Colab's current dependency resolution or browser frontend.

Proceed with the prepared notebook in a fresh hosted CPU runtime. Record installed versions and runtime identity, first Beam 2 activation, repeated toggles, Move/Tilt, numeric edits, canvas stability, controller reuse and static fallback. Distinguish actual browser observations from successful backend initialization. Scientific execution remains exclusively behind Run and requires separately authorized bounded qualification; no propagation was performed here.

**Final decision: GO for hosted Colab qualification; production release approval remains pending qualification.** No installations, simulations, credential access, commits, pushes, tags or uploads occurred. Only this report and anonymous download evidence were written. Unrelated work, saved notebook outputs and protected evidence were preserved.

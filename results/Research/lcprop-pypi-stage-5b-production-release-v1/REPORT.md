# Stage 5B — immutable production 0.3.0 release

## Decision

**GO for production PyPI publication of the exact retained artifacts**, subject to
separate upload authorization and the normal project/version/authentication check.
No technical qualification blocker remains. No upload, push or tag was performed;
production index availability and credentials were not inspected in this task.

Source commit: `fa8c3b292f2cfd38379870d0c8d085f5ff0c3f78`

Message: `chore(release): prepare lcprop 0.3.0`

Parent: `819b93601168a33ebac455993f951cb03de7e007` (RC2).

## Exact committed inventory

- `README.md` — SHA-256 `403857c2cd319ac7b30ef47655207cd93c674c2c8faf062082353ff5d452f54d`
- `pyproject.toml` — SHA-256 `498556478f929a9691cc7b05f47008b7ee79371036178c10b735acff5d18fdff`
- `docs/notebook_package.md` — SHA-256 `4781ef9936848d1710bf7890d8fb16ad89b8237b8600926d6db7a7254758cc1e`
- `tests/test_release_documentation.py` — SHA-256 `6da02658c7894f6408148d5dee69d8ca56fc84ab414a84ce6a0da6c9cbb1a768`

The four pre-commit hashes and exact diff matched Stage 5A. Explicit paths alone
were staged; staged inventory, whitespace and complete patch were reviewed before
commit. No credentials, local artifacts, protected reports or unrelated changes
were included. `staged.patch` preserves that review.

All 267 committed runtime files matched the qualified manifest before committing;
there are no runtime paths in this commit. Comparing parsed pyproject configuration
with RC2 shows only the approved project version changed. APIs, dependencies,
scientific requests and tolerances are unchanged.

## Final artifacts

Built with Python 3.12.13 using `python -m build` from a clean `git archive` export
of the exact commit, not the dirty working tree. Build isolation was enabled, and
the wheel was built through the sdist. See `source-provenance.json` and `build.log`.
The following exact files are retained in this report's `artifacts/` directory:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `lcprop-0.3.0-py3-none-any.whl` | 749946 | `9ab34f4ec244716b3e6f0431e4cac1e0d8a1ebeb17a7064c6422f1ed6b4e3d60` |
| `lcprop-0.3.0.tar.gz` | 655126 | `6ca4fab1b4d0a5343f9094af39b00a2437cd203da05609d0f10a93935a1eb399` |

Both archives were compared against hash-verified Stage 5A candidates. **Every
archive member name and file byte is identical**: 274 wheel files and 287 sdist
files. Complete per-member hashes are in `comparison.json`.

Archive SHA-256 values differ from Stage 5A, as anticipated in its release plan.
Wheel ZIP timestamps differ for all 274 entries. Tar mtimes and associated PAX
headers differ for 318 file/directory entries; compression incorporates those
headers. Inspected modes, ownership, types and file contents do not differ. These
are build-time container changes, not an unexpected payload change. Use the final
hashes above for publication; do not substitute Stage 5A archives or silently rebuild.

## Validation and installed-package checks

- Retained Stage 5A **75 passed, 1 deselected** remains applicable: the four source
  hashes, exact patch and artifact member bytes are unchanged. No redundant solver
  test was run; the deselected test is the full canonical scientific execution.
- Independently reread retained arrays: **9/9 bitwise identical**, with matching
  shapes/dtypes. `retained-parity.json` records source evidence paths and hashes.
  No reference regeneration or simulation occurred.
- Both final artifacts pass **`twine check --strict`** (`twine.txt`). Version 0.3.0,
  README and metadata are identical to accepted Stage 5A member bytes. The Stage 5A
  rendered README and 14 successful public URL checks remain retained there.
- Installed this final wheel non-editably, with no dependency resolution, into two
  existing disposable environments under `.codex-work/`: core-only and Colab-extra.
  This does not change a user environment or claim a new fresh-resolution test.
- Both installations resolve from their own site-packages, report 0.3.0, and verify
  all **267 installed package-file SHA-256 identities** against the qualified runtime
  manifest. `resolve_installed_distribution()` accepts both.
- Core environment has no Matplotlib/IPython/ipywidgets; core operations and shared
  helpers import successfully. Neither environment contains Qt, CuPy or LaunchPlane.
- Colab widgets initialize with network and scientific execution explicitly trapped;
  default budget, empty completed snapshot and both Configure labels pass. No external
  helper download or solver invocation. Exact installed dependencies, Python/platform,
  package locations and deployment digests are in the verification JSON files.
- Sdist contents match the accepted candidate exactly, and the final wheel was built
  from that sdist. This task does not add a separate pip-from-sdist installation test.

## Preservation and remaining gates

All preexisting tracked worktree bytes remain unchanged after commit/build/validation.
The index is empty. The two unrelated tracked modifications remain unstaged:
`src/lcprop/pr/published_static.py`, `tests/test_pr_published_static.py`.
Existing unrelated untracked files and protected evidence were not edited or staged.
Only disposable validation files and this new report/evidence directory were created.
The repository is intentionally not a clean worktree; see `preservation.json`.

No scientific engine changes, simulations, parameter sweeps, cluster operations,
GPU execution, uploads, tags or pushes. Retained qualification supports the same
bounded workflows and environments as RC2, not universal workflow/platform accuracy.

Next gate: authorize publication of these exact two artifact hashes, verify production
project/version availability and authentication without exposing credentials, upload
once, and independently verify the published hashes. No further source change or
rebuild is required for this qualified pair.

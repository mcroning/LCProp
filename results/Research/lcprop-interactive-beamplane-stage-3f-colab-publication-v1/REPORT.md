# Stage 3F — RC1 hosted-Colab notebook publication review

2026-10-17. Candidate approved for publication to `publication/colab-lc-0.4.0rc1` after the checks below. Publication and anonymous retrieval are verified separately after commit; this pre-publication record does not claim they already occurred.

## Candidate

- `notebooks/lc_static_pypi_rc1_colab_qualification.ipynb`
- `notebooks/lc_static_pypi_rc1_colab_qualification.md`
- `results/Research/lcprop-interactive-beamplane-stage-3c-colab-preparation-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-3c-colab-preparation-v1/preparation-identities.json`
- `results/Research/lcprop-interactive-beamplane-stage-3e-testpypi-verification-v1/REPORT.md`
- This Stage 3F report.

The notebook's sole content change since Stage 3C is publication-status prose: the historical 404 is replaced with the verified Stage 3E status. Setup/initialization/checklist code is unchanged. Stage 3C's old notebook hash remains historical evidence. New notebook SHA-256: `1c3bd65bf72fa7988cca5d5d0ceb1af9bea15cf751906dc34e5508c92da66b28`.

## Verification

Exact RC version 0.4.0rc1 and release source `2e195579dc3d76f6477bf6c885fd537043acdf9d` are enforced/recorded. LCProp is installed from the independently hash-verified local wheel downloaded from TestPyPI, not a floating package-index requirement. Dependencies resolve from production PyPI with inherited pip configuration disabled.

Wheel SHA-256: `2cc0ede68970b2d7de08dc385597861ab1706ad7b08fa4fa2169c5539f8e396e`.
Sdist SHA-256: `7f3367ea5a2940b8cd2256ec9c93ab44bfae2f3907798dda1158c96b1fdf2738`.

Both advertised hashes are checked; downloaded wheel bytes and all 269 installed runtime files are checked. Stage 3E independently downloaded and hashed both public artifacts. Version/import ownership and pip consistency checks gate initialization.

Custom widget manager and `%matplotlib widget` precede controller construction. Explicit static fallback uses a fresh runtime with `MODE = "static"`. Repeated initialization reuses the controller. No automatic scientific execution is present; only explicit Run executes science. Hosted visual qualification remains pending.

All code cells compile; execution counts and outputs are empty. Notebook contents have no embedded credentials or private absolute paths. Setup remains collapsed and requires no manual checkout/upload. The existing mock preparation checks were rerun successfully: missing release, metadata mismatch, wheel-byte mismatch, production-only dependency install command and repeated initialization. Network/installer calls were mocked; no installation or simulation occurred.

## Outgoing ancestry audit

Anonymous/public remote branch identities were read before publication. Existing published notebook branch is `89152f65e8565208f44cd5ea9c703d191482e112`. The four additional ancestors are precisely the accepted two-beam, Stage 1B, Stage 2 and RC1 preparation commits: `681f669`, `9151e6b`, `896c936`, `2e19557`. No unrelated ancestor was found beyond the already published history.

These four commits change 63 paths, with 69 distinct changed historical blobs. They contain the expected notebook implementation, tests, release metadata/documentation and validation reports. Bounded credential-pattern screening found no private keys, common PyPI/GitHub/AWS token patterns or credential-bearing HTTP URLs; no changed blob exceeds 1 MB. Prior reports contain reproducibility paths, not credentials; raw private evidence and notebook outputs are excluded. Pattern screening cannot prove absence of every conceivable secret.

The new commit is limited to the six paths above. No production notebook, scientific engine, package metadata or dependencies change in this commit. Existing unrelated tracked PR edits and untracked evidence remain outside the index. Only the dedicated publication branch is to be pushed, without force or tags; the default branch remains unchanged.

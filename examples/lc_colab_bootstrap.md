# Verified LC Colab bootstrap

`lc_colab_bootstrap.py` prepares the interactive example without executing the solver.
It reuses importable preinstalled dependencies, checks loaded versus installed versions,
and never upgrades numerical dependencies in place. Missing direct requirements use
pip `--no-deps`; missing transitive or broken binary dependencies produce diagnostics.
Pip output is captured; failures retain bounded error details. No automatic restart.

Scientific engine pin: `52b00928978a9ec7862a357d38327fd0b9ae857c`.
Interactive helper pin: `6ead2b5746765d865965162d614be8bb79659827`.
The 264-file package manifest is fetched from immutable commit
`8c2e8b1f4bc4477787c2e686c38d4636180f3d75` and SHA-256 verified before use.
Both helper downloads are independently hash-checked. Installed engine bytes must
match the qualified manifest and resolve from site/dist-packages.

`initialize(previous_app)` preserves the existing application and completed snapshot
on a verified rerun. New initialization samples a launch preview using existing
Product APIs; the scientific solver runs only when the user presses Run.
No Qt, LaunchPlane or CuPy requirement.

Publication sequence: commit this bootstrap first, then bind the notebook's bootstrap
revision and SHA-256 to that commit. The notebook integration changes are intentionally
excluded from this bootstrap commit. Hosted qualification follows separate publication;
local tests do not establish fresh hosted installation success.

Independent tests: `tests/test_lc_colab_bootstrap.py` (seven tests). Notebook binding
assertions remain separately in `tests/test_lc_interactive_setup.py` for the subsequent
notebook commit. No assertions were dropped or skipped for test separation.

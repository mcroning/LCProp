# Stage 1D — hosted publication prerequisite and deployment plan

## Outcome: stopped at explicit publication boundary

The current Stage 1C notebook and interactive helper are untracked, confirmed with
`git ls-files notebooks/lc_static_interactive_cpu.ipynb examples/lc_static_interactive.py`.
Neither file is in a local commit, so no immutable Git URL can supply this accepted
candidate. No older published revision was substituted. A genuinely working hosted
link requires publication of these files or equivalent verified release artifacts.
Per the task's stop condition, notebook/setup implementation and installation testing
are deferred; only this report and `source-identities.json` were written.

Local HEAD: `8c2e8b1f4bc4477787c2e686c38d4636180f3d75`.
Branch: `feature/pr-second-order-static`.
Remote: `git@github.com:mcroning/LCProp.git`.
Recorded upstream: `985b80fbbc05616d4fe92f974ef0da24bf84e94c` (not freshly fetched).
Public accessibility and remote publication have not been verified by this audit.

## Smallest deployment design

Use the existing repository and standard non-editable Python packaging. No new API,
frontend framework or physics layer. Pin two identities separately:

1. Engine: `52b00928978a9ec7862a357d38327fd0b9ae857c`, the Stage 5 qualified source.
2. Interactive helper and standalone example: a future published immutable commit H
   containing the accepted bytes recorded in `source-identities.json`.

Prefer installation from the engine's pinned GitHub source archive through pip. This
avoids a user-managed development checkout. Building from that commit produces a
new wheel artifact: do not claim it has the qualified wheel SHA. Compare all installed
scientific package files against the retained 264-file source identity manifest.
If rebuilding cannot reproduce source bytes, instead publish the exact qualified wheel
as a verified release artifact (SHA-256
`1322400b31fac3a28d439f281cc189fb790f1526940dc36af9bf50971fdc5c20`).
The chosen mechanism must pass installation verification before publication readiness.

One explicit setup cell should:

- Require a fresh CPU runtime; reject previously imported conflicting engine/helpers.
- Fetch only immutable engine/helper URLs with bounded downloads and SHA-256 checks.
- Install LCProp non-editably without `gui` or `gpu` extras and install/check the small
  notebook dependencies (NumPy, SciPy, Matplotlib, ipywidgets, IPython).
- Report pip failure clearly and stop before importing/enabling the interface.
- Verify engine revision/artifact identity, every relevant installed package-file hash,
  and site/dist-packages resolution. Version `0.1.0` alone is insufficient.
- Verify both helper hashes, then import from an isolated helper directory (never src).
- Check runtime dependencies and show environment/package/helper provenance.
- Initialize `InteractiveLC` and display its widget. Initialization samples the launch
  using existing APIs but does not execute the scientific solver.

Use the existing completed-snapshot/export mechanisms unchanged. Budget selection
remains execution configuration. Run remains the sole explicit scientific execution
button; opening a link cannot itself bypass Colab's runtime/cell-execution consent.
No Drive mount, private credentials, GPU allocation or local user environment mutation.

## Exact publication sequence (requires separate authorization)

1. Review and commit the accepted helper, standalone-example binding, tests and usage
   documentation as a scoped commit H. Confirm no unrelated files/evidence enter Git.
2. Set the notebook setup cell to literal H and the qualified engine commit, with
   helper and engine-file hashes. Correct the notebook's stale prose saying only the
   original 120-second limit applies; preserve the 120/180/300 control.
3. Validate setup/dependency failure paths and initialization in a disposable environment,
   then commit the self-installing notebook/setup tests as N. Two commits avoid trying
   to embed a commit's own not-yet-known hash into its content.
4. Publish only the reviewed outgoing range with separate push authorization; check
   public read access to both commits. Do not push all pending history implicitly.
5. Supply this immutable link, replacing N with the actual notebook commit:
   `https://colab.research.google.com/github/mcroning/LCProp/blob/<N>/notebooks/lc_static_interactive_cpu.ipynb`
6. Put the resolved badge/link in documentation in a subsequent documentation commit
   if necessary. Do not publish a moving-branch fallback or placeholder as runnable.

First-run instructions then become: open link, select CPU runtime, run the setup cell,
review initialized controls, press Run, explicitly export/download before session expiry.

## Validation required before claiming readiness

No installation or scientific execution performed here. Existing Stage 1C evidence
remains: 21 focused passes, five simulation-dependent tests deselected, retained nine
scientific-array identities exact. Current candidate hashes are recorded separately.

Required bounded deployment tests:

- Execute the actual setup cell against verified local fixtures; then test real pinned
  retrieval/install in a disposable environment outside existing user environments.
- Simulate missing dependencies, absent pip, nonzero installer exit, wrong revision,
  corrupted helper/wheel, missing package files and stale imported modules. Each must
  fail before widget execution is enabled, with no fallback to moving revisions.
- Verify package-file identities and initialization without Qt, LaunchPlane or CuPy;
  assert initialization never calls the solver. Preserve the original qualified request.
- Verify repeat setup either safely reuses the exact verified installation or requests
  a runtime restart; it must not leave a stale helper active.
- Actual hosted Colab: confirm public retrieval, installation, widget rendering, budget
  controls and export/provenance. A scientific hosted rerun is a separate qualification
  step, not required for preparing the deployment logic.

No Product source, candidate, prior evidence, index, commits or remote refs changed.
No commits, pushes, simulations or installations. The task is blocked on publication,
not on a scientific or package-architecture deficiency.

## Self-installing notebook implementation after publication-baseline commit

Publication baseline H is now `d69ef2bd5b93022dc3a4db6cc7f764d066ce078d`.
The notebook has one setup cell containing deployment logic, explicit engine/helper
pins, two helper SHA-256 identities and all 264 installed-package file hashes from
the Stage 5 qualified wheel manifest. No new Product API or scientific helper changes.
The public engine archive is passed to pip for a standard non-editable install with
Matplotlib/ipywidgets; no GUI/GPU extras. This is a source rebuild, not a claim of
wheel-byte identity. Every manifest package file must match before helper imports.
Helper downloads are size-bounded to 1 MB each with a 30-second network timeout.

Setup verifies downloads before installation, provides four progress stages and
raises actionable errors for unavailable immutable revisions, hash mismatch, installer
failure, missing dependencies, conflicting imported engine/helpers and checkout imports.
A verified existing import is reused without reinstalling; the previous application
and completed snapshot are retained on rerun. A first setup with already-imported
scientific dependencies requests a fresh runtime to avoid stale binary imports.
Initialization only constructs controls and the canonical launch preview. Run remains
explicit. No public install is claimed successful yet.

Public unauthenticated HEAD checks on 2026-10-09 returned **404** for both:

- raw helper at H, `examples/lc_static_interactive.py`;
- engine source archive at `52b00928978a9ec7862a357d38327fd0b9ae857c`
  (GitHub redirects to codeload, then 404).

These observations do not distinguish an unpublished commit from repository visibility
restrictions. Public access must be established for both exact pins. No credentials,
branch fallback, substitution, push or installation was attempted. Per the stop
condition, validation uses local download/install fixtures and the already-existing
qualified disposable environment.

Validation: **7 deployment tests passed**, executing the actual notebook setup-cell
functions. Coverage includes successful installer invocation, safe rerun, exact helper
bytes, wrong helper hashes, unavailable revision, missing dependency, installer failure,
and installed-engine manifest mismatch. Real widget initialization and rerun also passed
using local helper transport and the existing installed engine: 264 package files verified,
canonical preset unchanged, no completed run created, and no Qt/LaunchPlane/CuPy imports.
Scientific solver was never invoked. Installer success is mocked, not an end-to-end
installation claim. Logs/scripts remain under `.codex-work/stage1d-setup/`.

Changed files: `notebooks/lc_static_interactive_cpu.ipynb`, its usage `.md`,
`tests/test_lc_interactive_setup.py` (new), and this report. Existing scientific parity
evidence is unchanged. Stage 1C helper hash remains
`492d54cf89093742b1436cee7a7aec9a9539a122815e33811f2a21db7a631b80`;
standalone helper remains
`2ffed40dfc82d47b73bd1f9770c2ef4a12a2918e32bd2c137f6a252ed07f90a3`.

Remaining: review/commit this notebook separately, publish the reviewed revisions,
replace the documented Colab link template with the notebook commit, then verify
fresh hosted installation/widget rendering and exports. No hosted qualification is
claimed. Dependency versions remain environment-reported rather than a cross-platform
lockfile; numerical parity qualification remains scoped to previously tested environments.

## Acceptance before automatic-setup commit

Rechecked exact engine/helper pins and helper bytes against the committed publication
baseline. All 264 embedded package hashes match both the retained qualified manifest
and the Git-object bytes at the pinned engine revision. Seven deployment tests passed
again (0.16 s); real widget initialization and setup rerun passed in the existing
qualified disposable environment using local transport fixtures, without solver work.
Setup reuses the same application object on rerun, preserving its completed snapshot;
it does not call Run or the execution helper. Scientific request and helper code are
unchanged. Installed files resolve from site-packages. Public deployment remains
unqualified and depends on publishing the exact pins and notebook revision.

Only the notebook, usage documentation, deployment tests and this report are approved
for this commit. No installation, simulation, engine edits or push during acceptance.

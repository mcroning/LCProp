# Stage 1C — New-user layout, documentation, configuration, and Help

## Scope and baseline

Baseline: `bcdecf2509af80af16c717d4ac45482368d82751`, including Stage 1A,
Stage 1B, and the portable-workspace release contract. This isolated milestone
implements HA-09, the immediate portions of HA-10, HA-11, and HA-17 from the
[canonical punch list](lcprop_human_acceptance_punch_list_2026-09.md).
The authoritative Product workspace, LaunchPlane, and Research are protected.

## Changes and acceptance evidence

- **HA-09:** Quick Start now obtains both Products, establishes a working
  directory and Python environment, and installs them non-editably for ordinary
  Local use. Developer editable installation is separate. Python requirements
  and the Slurm deployable-checkout limitation are unchanged. README and built-in
  Help link to rendered guides without requiring a Markdown editor. The current
  public Product guides exist on `feature/pr-second-order-static`, not `main`;
  links and acquisition instructions use that branch and explain that published
  documentation may lag local development. Built-in Help ships with the Product.
- **HA-10:** The existing external user-local catalog remains authoritative.
  Its editor displays the configuration location and explicit default checkboxes
  for cluster and resource. Saving another resource no longer silently changes
  the default. Users can make or clear a default; with no explicit default the
  GUI still selects the first available entry without writing a default.
  Startup does not write profiles. The configuration dialog scrolls its form
  while keeping actions accessible. Empty catalogs provide first-profile guidance.
  The guide documents the existing `LCPROP_CLUSTER_CONFIG` override and its
  limited isolation: native file-dialog history and OS preferences are separate.
  Experiment Save/Open retains its empty initial directory.
- **HA-11:** Shared height-for-width wrapping rows replace crowded fixed
  horizontal execution rows. Shared compact application margins leave space for
  Results; PR forms grow fields and wrap long rows. Beam gives LaunchPlane a
  full-width default tab, with optical edge treatment and supported PR Input
  Screen configuration in secondary tabs. LC's disabled Input Screen no longer
  occupies a permanent pane. LaunchPlane internals and beam manipulation are
  unchanged. The existing 1200×760 Results geometry tests remain unchanged.
- **HA-17:** Help starts with scientific meaning and operational choices. It
  explains Gaussian waist/focus and entrance curvature; the periodic FFT domain
  with None/Sponge/Tukey edge treatment; scattering parameters and actual static
  support; Local/Slurm operation; carrier input/output power; linked orthogonal
  slices; and the Stage 1B convergence tables. PR Help does not foreground LC
  restrictions. Ordinary TBC remains distinct from specialized Image Amplification.
  None maps to the existing `periodic` value, including the unchanged stationary
  soliton restriction. User-facing naming is LaunchPlane; internal imports stay
  compatible.

## Scientific non-change and bounded follow-up

No scientific source, equation, optical propagation, boundary mathematics,
scattering default, request/schema, checkpoint/transport version, retention
policy, solver predicate/tolerance/budget, backend semantics, estimator
calibration, or result arrays change. Profile default selection is user-owned
configuration presentation and persistence through the existing catalog schema.
No package requirements or LaunchPlane code are changed.

Dedicated profile/resource duplication and individual resource deletion remain
P2 follow-up. Comprehensive preference isolation and offline packaging of the
full documentation remain outside this milestone. Native macOS interactive
acceptance is still distinct from offscreen Qt checks. No new scientific or
architectural implementation is required by these changes.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**, untouched. Stage 1A
Local-only Continue and Stage 1B result ownership, selected subtab, tables,
execution summaries, and display scaling remain covered by affected regressions.
No Stage 2/3/4 work or fanning work is included.

## Local validation

Validation uses the established Python 3.12 environment, isolated `PYTHONPATH=src`,
`PYTHONDONTWRITEBYTECODE=1`, `QT_QPA_PLATFORM=offscreen`, and pytest
`-p no:cacheprovider`. Logs and Matplotlib cache are outside Product repositories.
Tests exercise local CPU logic and mocks; no cluster, scheduler, SSH, or CUDA
execution is performed.

- Stage 1C focused coverage: 17 cases, included in the affected batch.
- Final affected batch: **152 passed** (11.09 seconds), covering documentation,
  Help, Beam, profile persistence, Results geometry, remote dispatch mocks,
  and Stage 1A/1B regressions.
- Complete Product suite: **1740 passed, 77 skipped in 507.11s (0:08:27)**, with zero failures.
- Python 3.12 in-memory syntax compilation, local Markdown links/anchors,
  whitespace, `git diff --check`, and preservation checks: passed.
- Offscreen PR form and Beam inspection at 1200×760 confirmed readable forms
  and expanded canvas allocation; this does not replace native macOS acceptance.

An initial affected run caught a missing LaunchPlane README link, the old
permanent-Beam-layout assertion, and insufficient Results height at 1200×760.
The link was restored, the Beam assertion now verifies accessible tabs, and
shared compact application spacing preserves Results geometry. A subsequent
run with the unresolved geometry failures crashed in Qt; the corrected affected
batch passed. No failure is accepted as a baseline exception.

Development + Local Validation and development self-review passed; independent
pre-commit review is the next gate. Nothing is staged, committed, integrated, pushed, or archived.
The active Stage 1C prompt is retained only in the isolated worktree outside
the candidate manifest.

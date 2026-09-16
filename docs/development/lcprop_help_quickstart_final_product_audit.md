# LCProp Help, Quick Start, and Final Product Audit

## Scope and baseline

This milestone starts from Product commit
`2477caac14cbfcc844ea4e9ae50098573f8d7cdc` in the isolated worktree
`/private/tmp/lcprop-help-final-audit.coVsxN` on
`codex/pr-gui-help-final-audit`.

The work adds current Product guidance and a small non-scientific Help surface.
It does not change an LC or PR equation, solver, tolerance, request, schema,
codec, launch field, optical propagator, result-retention rule, execution
target, backend default, or commissioning claim.

## Product documentation

`docs/user/quick_start.md` gives one short LC path and one short PR path from
installation through results and persistence. `docs/user/user_guide.md`
describes launch intent, boundaries, LC workflows, the complete eight-cell PR
matrix, execution/backend semantics, Fast/Full retention, MPR, TD products,
progress, persistence, warnings, fanning inputs, and runtime planning.

The documentation deliberately separates:

- physical model and approximation;
- software support as a production choice;
- local validation and composite-workflow validation;
- hardware/backend commissioning.

The PR estimator is described as advisory. Its recommendations do not mutate
the request, and its weakest calibration area—continuation-sensitive nonlinear
full-transverse static execution—is identified explicitly. Publication-facing
guidance recommends high-fidelity choices where justified but requires
problem-specific convergence instead of defining a hidden universal preset.

## In-application Help

Both LC and PR windows now expose the same material-neutral **Help** menu with
nine non-modal topics:

1. Quick Start;
2. Model choices;
3. Beam focusing;
4. Boundary conditions;
5. Fanning and scattering setup;
6. Running on Slurm;
7. Runtime estimates;
8. Fast vs Full;
9. Understanding Results.

The menu is presentation-only. Opening or closing help constructs no request,
starts no worker, and changes no scientific control.

## Canonical PR contract and Research separation

`docs/science/pr_model_contracts.md` provides the minimum current Product
knowledge formerly recoverable only by consulting four CURATED_SHARED
historical reports. It defines the reduced nonlinear equation, both uniform-
reference linearizations, nonlinear full-transverse Profile v1, the periodic
biased current-carrying linearized profile, parameter ownership, null-mode
handling, and approximation language.

The runtime reference module and its original implementation record now point
to this canonical Product contract. No tracked runtime source or Product
documentation depends on the four historical report paths. Those untracked
Product-workspace copies remain untouched; their later Research verification
and disposition require a separate lifecycle operation.

## Final usability audit

### LC

The current LC Product supports local static, TD, soliton, and soliton-sweep
workflows; shared LaunchPane beam intent; focus-defined launches; all three
optical boundaries; result fields/curves/diagnostics; static/TD experiment
persistence; checkpoints; and compatible retained-state initialization.

The audit records rather than hides these bounded follow-ups:

- Slurm is enabled only for canonical LC static propagation;
- the LC GUI does not expose PR's Fast/Full retrieval choice;
- LC GUI-representable runtime/backend/precision options are narrower than the
  headless APIs;
- shared input-screen editing remains disabled for LC requests;
- there is no LC counterpart to the quantitative PR runtime estimator.

These are missing parity or follow-on capabilities, not controls that pretend
to work. They do not invalidate the documented LC workflows and were not
expanded into new development during this milestone.

### PR

All eight model cells are present and mapped through the three independent GUI
axes. Static cells hide material-time controls; exact-modal cells present only
the exact update; nonlinear TD cells present their applicable methods.
Canonical scattering is hidden for reduced static because that request cannot
carry it. Explicit backend intent remains distinct from automatic target-
derived selection.

Fast products retain their established distinction: nearest-zero cuts are
quantitative, while arbitrary MPR slices use a bounded visualization-only
preview. TD movies and scalar curves observe accepted states without changing
the calculation. Structured progress remains best-effort and nonfatal.

### Compatibility and audit correction

The audit found four pre-existing presentation/test defects and reproduced all
five resulting test failures against an extracted pristine baseline before
making a correction:

- one legacy-GUI regression fixture changed a current schema-6 reduced-TD
  payload's version number to 1 while leaving newer `material_response` and
  `scattering` fields present; it now removes every post-v1 field;
- two LC field-selector expectations predated the supported far-field and dB
  products; they now include those current products;
- the even-grid longitudinal-slider test assumed the upper arithmetic center,
  while production deliberately selects the first coordinate nearest zero in
  a physical tie; it now asserts the actual coordinate rule;
- the default-width transverse plot clipped its y-axis artist by about 2.4
  pixels and left too little colorbar-label gutter; the fixed axes rectangles
  reserve one additional percent on the left and move the colorbar two percent
  left without changing any data, coordinates, limits, or field products.

Production codecs, products, and scientific arrays are unchanged.

## Release assessment

No release-blocking documentation link, false GUI control, or missing PR model
cell remains in the reviewed scope. The PR GUI/package is user-complete enough
to return to separately authorized fanning research after this milestone is
reviewed and committed. The LC limitations above are candidates for bounded
future milestones; none should be silently folded into research work.

## Validation

Validation uses the repository Python 3.12 environment with the isolated
worktree `src` directory placed first on `PYTHONPATH`. The system Python 3.9
was rejected after PySide6 aborted during the first attempted collection; no
test body or scientific workflow ran in that invalid attempt.

Recorded validation:

- focused Help, Product-documentation, LC/PR GUI, model-matrix, experiment,
  boundary, result, and estimator tests: `224 passed`;
- isolated reproducer for the five baseline failures: `5 failed`, identically
  on the pristine baseline and pre-correction candidate;
- focused post-correction presentation rerun: `5 passed`;
- complete local Product suite: `1656 passed, 77 skipped`;
- syntax compilation: passed;
- Product-relative Markdown link validation: passed;
- `git diff --check` and supplemental candidate whitespace inspection: passed.

The 77 skips are existing conditional platform/backend tests; no CUDA device,
cluster, remote system, or scheduler was accessed.

# Stage 1D hosted acceptance and Configure label closure

## Evidence scope

The user reports successful automatic installation and widget initialization in hosted
Colab, successful execution with 100/100 slices converged, and successful exploratory
2 mW execution using the selected 180-second wall-time budget. These are accepted
user-reported end-to-end outcomes. No new setup transcript, screenshot or 2 mW result
export was available for independent review during this closure. Exact elapsed time,
2 mW request identity, residual maxima and output-array hashes are therefore unverified.
The 180-second value is the selected budget, not a measured runtime or convergence
criterion. It is checked between slices, so no strict runtime bound is inferred.

The located `Downloads/lc-interactive-download.zip` is the earlier **1 mW** archive,
SHA-256 `175ebed8234bf2edaef3ffd46767ae4742daf2024f81eedb1ddb5d216a90bb0f`.
Its request power and completion summary were rechecked: 100 completed slices and
all_slices_converged true. It must not be relabeled as new 2 mW evidence.

Retained independent scientific evidence is documented in
`results/Research/lcprop-colab-interactive-stage-1-hosted-review-v1/REPORT.md`:
all nine scientific arrays matched the previous hosted Stage 5 reference bitwise;
local comparison satisfied the declared cross-platform criteria, maximum relative
L2 approximately 5.50e-13 for A_final, maximum director difference 3.33e-15 rad.
All 100 slices converged, with maximum final RMS residual 0.004935085357116411
(gate 0.005) and maximum residual 0.019985181428802123 (gate 0.02).
These comparisons apply to the unchanged qualified 1 mW preset and tested environments,
not arbitrary power, voltage, geometry or dependency versions. They were not rerun.

Anonymous publication verification previously established the pinned bootstrap,
manifest, scientific helpers and notebook bytes. Local deployment evidence comprises
seven bootstrap tests plus two notebook integration tests, including compatible
preimported dependencies, 264 engine-file identities, real widget initialization and
same-object snapshot retention on rerun. This closure adds user-reported hosted
usability acceptance, not an independent new numerical-parity qualification.

## Configure presentation correction

The accordion observes `selected_index` and updates immediately:

- expanded: `Configure — collapse to focus on results`;
- collapsed: `Configure — expand to show setup pane`.

The focused regression verifies initial expansion, collapse, re-expansion and a second
collapse, with launch sampling disabled. **1 passed, 13 deselected**. No solver or
installation runs. Inspection confirms only the initialization title/observer plumbing
changed; all other helper function/method ASTs are unchanged. No scientific requests,
engine algorithms, arrays, serialization or snapshot behavior changed.

## Limitations and publication requirements

Execution remains synchronous; responsive browser Stop is not guaranteed. Wall-time
budget exhaustion differs from scientific nonconvergence and is only checked at slice
boundaries. Successful exploratory 2 mW execution does not qualify arbitrary edited
configurations, establish accuracy under refinement, or demonstrate a stationary soliton.
Full independent closure of the 2 mW numerical result would require its retained export
and request/provenance, not a new simulation.

The published bootstrap still pins scientific/presentation helper revision
`d69ef2bd5b93022dc3a4db6cc7f764d066ce078d`. Thus this local label correction will not
appear in the hosted notebook merely by pushing its commit. A separate reviewed change
must pin the new helper revision and SHA-256 in the bootstrap, then pin that bootstrap
revision/hash in the notebook, run deployment tests, and publish with authorization.
Those pin changes are intentionally outside this three-file commit.

Commit inventory: `examples/lc_static_interactive.py`,
`tests/test_lc_static_interactive_presentation.py`, and this closure document only.
Unrelated work and protected evidence preserved. No simulations, installations or push.

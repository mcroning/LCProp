# PR Image-Amplification Composite Execution D1

## Status

Development reference for the Stage D1 local composite-execution milestone.
The implementation is intentionally PR-owned and local-only. It does not add
remote transport, persistence, or broad GUI algorithm selection.

## Architectural Boundary

Image Amplification is a specialized experiment over an ordinary registered
PR operation:

```text
PRImageAmplificationExperimentRequest
    -> deterministic ordinary-request transformation
    -> runner.run_registered("pr", selected base workflow, base request)
    -> common PR image postprocessor
    -> ordinary RunData augmented with image products
```

The orchestrator makes exactly one runner call. It does not register an image
operation that recursively invokes another registered operation. Historical
direct APIs remain available for callers that depend on them.

## Capability Contract

The common postprocessor requires the selected ordinary result to provide:

- prepared input complex channel fields (`A_initial`);
- final complex channel fields (`A_final`);
- runtime grid summary;
- initial and final optical power;
- terminal status.

The ordinary request must provide the grid, beam stack, material, and backend.
Wavelengths, carrier phase gradients, coherence groups, and incident powers
remain in the immutable launch configuration.

The capability table is explicit and small; it is not a plugin framework.

| PR workflow | Request/result | Launch adapter | D1 classification |
|---|---|---|---|
| `pr_timedependent` | `PRRunRequest` / `PRRunResult` | Declarative launch elements | Compatible and validated |
| `pr_static` | `PRStaticRunRequest` / `PRStaticRunResult` | Declarative launch elements | Compatible, validation pending |
| `pr_transverse_static` | `PRTransverseStaticRunRequest` / `PRTransverseStaticRunResult` | Declarative launch elements | Compatible, validation pending |
| `pr_transverse_timedependent` | `PRTransverseRunRequest` / `PRTransverseRunResult` | Prepared-field compatibility adapter | Compatible, validation pending |

The transverse-TD adapter is necessary because that request does not yet
accept declarative launch elements. It prepares `initial_A` deterministically;
the other three adapters preserve `initial_A=None` and allow their ordinary
workflow to construct the launch.

Compatibility means the postprocessor can consume the operation's result. It
does not claim scientific validation for image amplification. Only reduced TD
is exposed by the D1 GUI and classified as validated.

## Immutable Request and Transformation

`PRImageAmplificationExperimentRequest` owns:

- selected `base_workflow_id`;
- the selected algorithm's complete ordinary `base_request`;
- `LaunchConfiguration`;
- pump and signal channel roles.

Solver, transport, boundary, material, grid, backend, and precision settings
are not duplicated in an analysis specification. The deterministic
transformation replaces only the ordinary request's launch input.

For the validated reduced-TD path the transformed request contains the
configuration's beams and `launch_elements`, with `initial_A=None`. This keeps
normalization and passive-screen application in canonical optical launch.

## Common Postprocessor

The common analysis owns:

- carrier-mask construction and carrier isolation;
- output back-propagation;
- zero-response forward propagation and reference back-propagation;
- measured and analytic gain;
- image correlation and normalized RMSE;
- optical-power drift and image-specific power diagnostics;
- augmentation of the ordinary operation's existing `RunData`.

Back-propagation is therefore an experiment-analysis responsibility rather
than a material-workflow responsibility. The runner-created base `RunData` is
reused and augmented; the base result is not converted a second time.

## Progress and Cancellation

Base progress objects are forwarded unchanged through the same callback. The
composite does not synthesize generic propagation percentages. After the base
operation, bounded image-analysis progress identifies carrier isolation,
output back-propagation, zero-response propagation, reference
back-propagation, metric construction, and product augmentation.

The same cancellation token is passed to the base runner. During analysis it
is checked between bounded stages. Cancellation during base execution
preserves the base cancellation. Cancellation during analysis preserves the
successful base result, marks analysis cancelled, and makes the overall
experiment cancelled.

## Status Composition

`PRImageAmplificationCompositeResult` records independently:

- the ordinary `base_runner_result`;
- optional completed image-analysis result;
- `analysis_status` and explanatory message;
- composed overall status.

A base `cancelled`, `failed`, or `not_converged` status is never overwritten by
image success. Failed or cancelled analysis retains the ordinary base result
and ordinary products for diagnosis. Product-augmentation failure is also
classified as analysis failure rather than destroying the base result.

## Historical Compatibility

The following historical functions keep their direct reduced-TD semantics:

- `make_image_amplification_request()`;
- `run_image_amplification()`;
- `run_image_amplification_request()`.

Their normalized launch conventions and return type are unchanged. The new
runner-aware API is additive. For equivalent completed reduced-TD input, local
validation requires bit-for-bit equality of prepared/evolved fields, carrier
products, reconstructions, metrics, workflow diagnostics, and terminal state.

One presentation-provenance difference is intentional: the historical direct
request supplies an already prepared `initial_A`, so its ordinary launch
summary cannot know that the signal screen reduced channel power. The D1
ordinary request carries declarative launch elements and therefore records the
actual post-screen channel power and throughput. All other ordinary summary
values and all specialized image diagnostics remain equal. This is a more
accurate ordinary-operation record, not a numerical or physical difference.

## GUI Scope

The PR GUI now routes Image Amplification through the composite orchestrator,
which selects the existing reduced-TD operation. The ordinary evolution
selector remains visibly constrained to Time Dependent because other
algorithms have architecture-level compatibility but have not yet passed
image-amplification scientific validation. No competing image-specific solver
selector was introduced.

## Local Validation

- focused image-amplification and PR product suites: 56 passed;
- complete PR suite: 406 passed, 57 skipped;
- reduced-TD direct/composite evolution, analysis arrays, metrics, and
  specialized products: bit-for-bit equal;
- syntax compilation: passed;
- scoped `git diff --check`: passed.

The skipped tests retain their existing optional-backend/dependency status; no
cluster or remote execution was used for D1.

## Deferred Work

- scientific image-amplification validation for legacy static and the two
  full-transverse workflows;
- GUI exposure of additional validated base algorithms;
- composite experiment persistence and remote transport;
- generalized composite-operation or plugin frameworks.

# PR Image Amplification Multi-Algorithm Validation — Stage 01

**Date:** 2026-08-27

**Branch:** `feature/pr-second-order-static`

**Implementation base:** `818dad9e221d9dda416b993b0cd16bd88531ec5f`

## Objective

This milestone exposes the ordinary PR algorithm selector inside the
algorithm-agnostic Image Amplification composite experiment and performs the
first specialized validation of `pr_transverse_static` as a second base
algorithm. Image Amplification remains a launch composition plus a common
postprocessor; it does not become a solver or own algorithm-specific execution
paths.

## GUI and Capability Mapping

The existing Evolution-panel workflow selector is reused and relabeled
**Algorithm** while Image Amplification input mode is active. It maps directly
to canonical registered workflow identifiers:

| GUI label | Workflow ID | Specialized status | GUI availability |
|---|---|---|---|
| Reduced TD | `pr_timedependent` | Validated | Selectable |
| Static | `pr_static` | Experimental | Selectable |
| Transverse Static | `pr_transverse_static` | Validated by this milestone | Selectable |
| Transverse TD | `pr_transverse_timedependent` | Experimental | Disabled |

The Transverse TD capability adapter can consume a prepared field, but the
ordinary GUI request builder does not construct the canonical
`PRTransverseRunRequest`. It therefore remains visible but disabled with this
reason rather than being exposed through an incomplete request path.

Experimental selection is nonblocking and visibly identified as specialized
validation pending. Ordinary workflow controls remain authoritative for each
selected algorithm.

## Request and Execution Path

The GUI first constructs the selected ordinary canonical PR request through
`build_pr_request()`. It then wraps that request, the shared
`LaunchConfiguration`, and the pump/signal role indices in
`PRImageAmplificationExperimentRequest`.

For `pr_transverse_static`, the common preparation path:

1. preserves the ordinary material, grid, solver, backend, transport,
   dielectric, boundary, and projection settings;
2. replaces the base launch with the shared beam stack;
3. supplies the image screen once through declarative `launch_elements`;
4. leaves `initial_A=None` so the canonical transverse-static workflow owns
   launch preparation;
5. dispatches `PR_TRANSVERSE_STATIC_OPERATION` through `LocalRunner`; and
6. passes its ordinary final complex field and metadata to the unchanged
   Image Amplification postprocessor.

No transverse-static-specific carrier isolator, back-propagator, gain
calculation, or product adapter was introduced. The composite RunData retains
the ordinary transverse-static fields, convergence diagnostics, replay data,
and provenance and augments them with the common Image Amplification products.

## Bounded Validation Fixture

The reproducible harness is
`scripts/checks/pr_image_amplification_multi_algorithm_validation.py`.

| Parameter | Value |
|---|---:|
| Grid | 24 × 24 × 2 optical intervals |
| Aperture | 40 × 40 µm |
| Interaction length | 10 µm |
| Optical interval | 5 µm |
| Pump/signal powers | 1.0 / 0.2 mW |
| Beam waists | 10 × 10 µm |
| Pump/signal x phase gradients | +0.15 / −0.15 rad/µm |
| Coherence group | `image-validation` for both channels |
| Source | deterministic 8 × 8 diagonal image |
| Image footprint | 8 × 8 µm |
| Dark/background intensity | 0.4 / 0.1 |
| Gain-length product | 0.001 |
| Backend/precision | NumPy / float64 |
| Reduced-TD material evolution | 2 steps, normalized timestep 0.01, semi-implicit |
| Transverse-static outer limit | 8 coupled iterations |

The fixture is deliberately small and weakly coupled. Its purpose is to
validate composition, scientific observables, and status semantics on CPU; it
is not a production image-amplification benchmark.

## Results

The two executions used identical beams, screen elements, image transmission,
incident powers, post-element powers, and carrier mask.

| Metric | Reduced TD | Transverse Static |
|---|---:|---:|
| Base status | `completed` | `converged` |
| Composite status | `completed` | `converged` |
| Analysis status | `completed` | `completed` |
| Measured absolute signal gain | 0.9999528385 | 0.9996805698 |
| Analytic absolute signal gain | 0.9990003802 | 0.9990003802 |
| Image-intensity correlation | 0.9999999991 | 0.9999999942 |
| Normalized image RMSE | 4.5008365 × 10⁻⁵ | 1.2986809 × 10⁻⁴ |
| Relative optical-power drift | −5.83964 × 10⁻¹⁶ | −3.50378 × 10⁻¹⁶ |
| Workflow runtime | 0.00645 s | 0.04213 s |
| Total composite runtime | 0.00691 s | 0.04253 s |

The transverse-static measured gain differs from reduced TD by
−2.7228 × 10⁻⁴ relative. The reconstructed intensity fields differ by
3.3897 × 10⁻⁴ in relative L2 norm. These small observed differences are not
treated as an equality requirement: reduced TD represents a short material-time
evolution, whereas transverse static solves its own zero-flux equilibrium.

Transverse static converged in two coupled iterations. Its final authoritative
equilibrium residual was 2.81823 × 10⁻¹² RMS and 2.33728 × 10⁻¹¹ maximum, and
the independent optical replay reported a field match. Carrier isolation and
the shared back-propagation returned a finite, recognizable reconstruction;
all gain, correlation, NRMSE, and power-accounting metrics were finite and
internally consistent.

## Progress, Cancellation, and Presentation

Base-operation progress is forwarded unchanged. Reduced TD therefore reports
ordinary material-time progress, while transverse static reports accepted
coupled-iteration progress. Both then report the same six Image Amplification
post-processing stages.

A focused cancellation check cancels transverse static after its first
accepted coupled iteration. The ordinary base result returns `cancelled`, the
composite preserves `cancelled`, and image analysis remains `not_run`.
Post-processing cancellation retains the previously established D1 semantics.

The normal GUI completion path accepts the transverse-static composite,
displays **Converged**, installs both ordinary and Image Amplification products,
and reports the authoritative zero-flux residual. Switching among selectable
algorithms leaves the material settings, beam stack, image screen, and
pump/signal roles unchanged.

## Decision and Limitations

`pr_transverse_static` is promoted to **Validated** for the bounded Image
Amplification composite contract. This means the canonical request, ordinary
solver, common postprocessor, products, progress, cancellation, and status
semantics work coherently for the validated CPU fixture. It is not a claim that
reduced TD and static physics should agree for arbitrary material time,
nonlinearity, image scale, or grid resolution.

`pr_static` remains **Experimental** because this milestone verifies its
declarative request adapter but does not perform a specialized scientific
comparison. `pr_transverse_timedependent` also remains **Experimental** and GUI
disabled pending an ordinary GUI request-construction path.

Remote/Slurm execution and experiment persistence for the composite request
remain intentionally deferred. Unsupported paths continue to reject rather
than silently changing request semantics.

No cluster validation is required for this bounded functional and scientific
promotion. A future production-scale comparison may establish performance and
larger-grid scientific behavior without changing the capability contract
validated here.

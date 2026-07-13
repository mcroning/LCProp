# LCProp Development Plan

This document is forward-looking. Current implementation facts belong in
[`STATUS.md`](STATUS.md), and architectural decisions belong in
[`LCProp_Architecture_Blueprint_v1.1.md`](LCProp_Architecture_Blueprint_v1.1.md).

## Near-term implementation order

1. **Commit the grouped-coherence checkpoint.**
   Preserve the tested native `coherence_group` model as the adapter baseline.

2. **Implement and test the thin LCProp-side LaunchPane adapter.**
   Convert enabled `BeamDefinition` objects to `BeamChannel` objects,
   preserve ordering and `coherence_group` strings, apply the default
   `theta_weight` policy, and reject an all-disabled launch clearly.

3. **Embed `LaunchPlaneWidget` in the LCProp Beam tab.**
   Keep LaunchPane independent and route its model through the LCProp adapter.

4. **Synchronize aperture dimensions from `GridPanel`.**
   Keep `GridPanel` authoritative for `x_aperture_um` and
   `y_aperture_um`; pass only those physical dimensions to LaunchPane.

5. **Prove one-beam equivalence with the former `BeamPanel` controls.**
   Compare requests and launch fields for matching wavelength, power, waist,
   position, tilt, and phase inputs.

6. **Test multibeam grouped-coherence requests end to end.**
   Cover enabled-beam filtering, mixed `coherence_group` values, request construction,
   workflow execution, and displayed products.

7. **Add save/reload and workspace persistence.**
   Define versioned request/result manifests and restore editor and result-view
   state without serializing ephemeral runtime objects.

8. **Add the stability workflow.**
   Express stability inputs and outputs as LCProp request/result objects and
   reuse established workflow and product boundaries.

9. **Improve GUI responsiveness and background job handling.**
   Move long runs off the UI thread and define progress, cancellation, error,
   completion, and spinning-wheel behavior.

## Later numerical milestones

- validate and integrate dual-grid methods;
- complete a global z-coupled workflow;
- evaluate additional global, bidirectional, or Newton-style strategies only
  with explicit physics-validation evidence.

Each milestone should leave the full automated suite passing and add focused
tests at the boundary it introduces.

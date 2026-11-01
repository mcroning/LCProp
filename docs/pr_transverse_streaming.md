# Full transverse zero-flux streaming adapter (experimental)

The opt-in model `full_transverse_isotropic_zero_flux_streaming_v1` is available
from `lcprop.pr.transverse.streaming`. It uses isotropic transverse transport
(h_y=m_y=1), zero applied field, periodic boundaries, and scalar optical coupling
through E_x. It is a distinct fourth model, not a tensor-faithful BaTiO3 model.
The existing material residual, Jacobian, Newton–PCG method and optical operator
are reused without modification.

## Public interface

`transverse_request_from_reference(reference, options=..., initial_psi=...)`
copies the original physical grid, beams, material and launch from a
`PRStreamingStaticRequest`. Specify transverse marcher options explicitly; the
one-dimensional solver's tolerances are not silently translated. Apodization
must be disabled and optical substeps must match. This helper copies **physical
inputs only**; bind scattering explicitly in the execution call below.

```python
from lcprop.pr.transverse.streaming import (
    transverse_request_from_reference, TransversePhaseBinding,
    run_pr_transverse_streaming,
)
request = transverse_request_from_reference(reference, options=transverse_options)
result = run_pr_transverse_streaming(
    request,
    launch_peak_intensity_reference=verified_full_domain_channel_peak_sum,
    phase_binding=TransversePhaseBinding(verified_phase_source, reference),
    stop_after_accepted_cells=accepted_prefix,
)
```

The peak is the sum of individual channel peak intensities on the original
full domain, not the coherent total peak or the crop-local peak. Supplying it
requires an explicit launch. Omitting it preserves the marcher's prior local
normalization behavior. The prefix does not shorten L or Nz or rescale gammaL.
No binding means no indexed scattering; Figure 11 callers must supply the
qualified source and its original physical/seed context.

Binding supports existing in-memory, disk-backed, CPU-indexed and GPU-indexed
sources. Source validation checks the required physical prefix before marching;
source-specific hash/runtime checks and physical-event indexing remain intact.
Application occurs exactly once after each accepted complete optical cell.
Failed material trials do not consume phases. Replay starts independently from
the original launch and initial material seed with the same source and indices.

## Acceptance and diagnostics

The adapter gates the **full** equilibrium residual, including derivative-null
components. Existing resolved-mode/gauge projection and even-grid null handling
are unchanged. A projected solution with unresolved full residual fails safely.
The low-level marcher still defaults to its original projected acceptance.

Per-cell diagnostics include full/projected/null residuals, Newton/PCG history,
carrier positivity and conservation, curl, Gauss law, spectral flux/divergence,
and material moments. Flux diagnostics use the existing spectral derivative;
a discrete product-rule defect is not silently treated as another material equation.
The first-pass result also contains power and timing diagnostics. Replay compares
final optical arrays, request identity, material moments and convergence decisions.
Material moments are summary checks, not full pointwise material-array comparisons.
A failed replay preserves the completed first-pass result.

## Memory and qualification limits

The wrapper disables material/residual/optical longitudinal stacks. Plane-sized
material workspaces remain on the selected backend. Final optical/checkpoint
arrays are returned on the host; two passes and scalar histories are retained.
There is no O(Nz*Nx*Ny) storage. Diagnostic FFTs and reductions add overhead;
no H200 performance or memory limit is established by CPU tests.

Separate synthetic H200 qualification must exercise actual XORWOW dispatch,
CuPy Newton–PCG, full/projected/null acceptance, repeated replay, failures and
peak combined workspace usage before any physical-domain experiment. No physical
Figure 11 propagation is authorized by this interface's local qualification.

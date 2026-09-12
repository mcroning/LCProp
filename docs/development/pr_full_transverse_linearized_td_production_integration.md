# Full-Transverse Linearized PR TD Production Integration

## Scope and status

This milestone integrates the committed periodic biased full-transverse
linearized time-dependent material reference into the existing production
`pr_transverse_timedependent` workflow. It is an additive constitutive-response
choice. The established fully nonlinear TD response remains the default and
retains its existing numerical update, optical propagation, source cadence,
products, cancellation boundaries, and workflow identifier.

The baseline is commit `e25a2230054eb0491ac91adc22fd456d8197a414`. This is
ordinary local implementation validation, not a retained authoritative
numerical-evidence run. No clean-checkout evidence package is therefore
required or created.

## Production lifecycle

For each accepted material interval, production execution now performs:

1. a complete optical pass through the last accepted material volume;
2. construction of the frozen transport-intensity source at every z plane;
3. either the unchanged nonlinear IMEX/reference Euler candidate or the exact
   linearized modal candidate selected by `material_response`;
4. atomic acceptance only after the entire candidate volume is complete; and
5. a final optical replay through the last accepted material state.

The linearized branch calls the committed
`solve_pr_biased_linearized_timedependent_reference()` on one transverse plane
at a time. The source is held fixed during each accepted interval and the
reference applies the exact modal exponential, so no material time-stepping
error is added. The leading production volume dimension remains physical z;
each reference call receives one independent `(Nx, Ny)` frozen-intensity
plane. Active backend work is therefore plane-local, while the established
accepted source and material volumes remain retained.

Cancellation is checked before and after every linearized plane solve. A
partially filled candidate is discardable: cancellation returns the preceding
accepted material volume and then performs the usual consistent final optical
replay. Progress remains one event per accepted material interval.

## Configuration and compatibility

`PRTransverseRunRequest.material_response` is appended after every previously
public positional field. Existing positional construction therefore binds
`solver`, backend, runtime state, scattering, and launch elements exactly as
before. The default is

```text
model = nonlinear
reference_intensity = None
```

Linearized execution requires a finite positive explicit `reference_intensity`
representing the uniform total transport intensity (I_0). It is never
inferred from a beam, dark intensity, or background. As in the accepted static
integration, `material.applied_field` remains zero and the periodic biased
current-carrying boundary profile is the sole owner of the transverse applied
mean field. Nonlinear execution continues to require the unbiased Profile v1
boundary.

Both experiment persistence and portable runner transport encode the response
selector. Payloads from before this additive field decode to the nonlinear
default. Results retain the response, reference intensity, applied field,
electrical ensemble, exact reference model identifier, and Experimental status
in their resolved provenance.

## Scientific identity and diagnostics

The production branch does not reproduce or rederive the Fourier operator. It
uses the committed reference from
`linearized_timedependent_reference.py`, which evolves

\[
\partial_\tau\widehat{\delta\psi}
=-\Lambda(\mathbf k)\widehat{\delta\psi}
+S(\mathbf k)\widehat{\delta I}
\]

exactly over the requested interval. Consequently the accepted static kernel,
bias reversal, gauge projection, derivative-null/Nyquist treatment, backend
dtype rules, and long-time limit are inherited from that reviewed reference.

Diagnostics identify `exact_frozen_source_linearized_modal_update`, count
actual plane calls (including a completed call in a subsequently discarded
candidate), record the optical-source cadence and plane-local memory policy,
and mark nonlinear material iterations as not applicable. They do not report
Newton, Krylov, convergence, or nonlinear-TD iteration metadata for the
linearized path.

## GUI, products, and cost guard

The existing material-response selector is now available for full-transverse
TD. Choosing Linearized exposes explicit (I_0) and transverse bias controls,
builds the matching boundary profile, and presents the model as Experimental.
The request summary names the exact frozen-source modal update. Nonlinear TD
continues to present its established controls and status.

Product reconstruction is unchanged. It already uses the resolved boundary
bias when reconstructing `E_x`, so the accepted potential produces the same
field, carrier, active-field, optical, and carrier-power products as the other
full-transverse workflows. The local cost guard retains the same thresholds
and accounts for the fixed-count plane-local FFT work separately from the
nonlinear material step.

## Local validation

Focused tests cover:

- the exact old positional constructor and the nonlinear default;
- explicit-I0 and matching-profile validation;
- exact production/reference identity from zero and arbitrary initial states;
- equilibrium invariance and the long-time static limit;
- positive/negative bias reversal and unbiased weak-response behavior;
- odd/even grids and derivative-null/Nyquist projection;
- NumPy float64 and float32 dtype behavior;
- plane-local calls and cancellation during longitudinal traversal;
- multiple accepted intervals, progress metadata, and final replay;
- a complete optical/material production run with nonzero gain;
- experiment and runner-transport round trips plus legacy payload decoding;
- result/product reconstruction, GUI Experimental presentation, and run cost;
- a conditional CuPy float32 seam without a local GPU claim.

The isolated reference tests, affected TD production/transport/persistence,
GUI/product/cost tests, complete PR suite, syntax compilation, and whitespace
checks are run before self-review. Numerical tolerances in the committed
float64 reference tests are not weakened.

For the deterministic frozen-source production fixture, evolution to
`delta_tau=30` reproduced the static potential, `E_x`, `E_y`, and active field
bitwise; the remaining linearized TD RHS maximum was
`8.371337816534531e-17`. At bias `+/-0.4`, the selected transient K coefficient
was respectively
`-0.27287364047996493 - 0.030363821309825425j` and its exact complex
conjugate, so the bias-driven phase direction reversed with zero conjugation
error. The even-grid joint-null and gauge coefficients were below
`8.7e-18` after inverse/forward transform roundoff.

In the zero-bias weak-modulation fixture, where the source perturbation was
scaled by `1e-5`, nonlinear versus linearized potential differed by
`6.397021411853081e-5` in relative L2 and
`7.837202709462907e-16` maximum absolute value. With zero optical gain in this
material comparison, the source and final optical fields were exactly equal by
construction. A separate two-step production test with nonzero optical gain
exercised the complete optical-source/material/replay lifecycle and finite
product reconstruction. A weak, nearly uniform full optical run with nonzero
gain gave `4.790865841831182e-4` potential relative L2 difference,
`6.076616210232581e-16` maximum potential difference, and
`1.201464136741322e-18` final optical-field relative L2 difference; its
accepted-interval and final-replay source arrays were exactly equal between
the two models.

Final local validation completed with:

- focused reference/integration/TD/transport checks: `67 passed, 2 skipped`;
- complete `tests/test_pr*.py` suite: `672 passed, 74 skipped`;
- syntax compilation of every candidate Python file: passed;
- `git diff --check`: passed.

The skips are existing conditional accelerator/environment tests, including
the new CuPy seam. No CUDA device or cluster was accessed.

## Candidate boundary and exclusions

The intended candidate contains only:

- `src/lcprop/pr/transverse/specs.py`
- `src/lcprop/pr/transverse/workflow.py`
- `src/lcprop/pr/transverse/timedependent_transport_codec.py`
- `src/lcprop/pr/experiment_codec.py`
- `src/lcprop/pr/gui/request_adapter.py`
- `src/lcprop/pr/gui/evolution_panel.py`
- `src/lcprop/pr/gui/main_window.py`
- `src/lcprop/pr/gui/run_cost.py`
- `tests/test_pr_transverse_linearized_timedependent_production.py`
- `tests/test_pr_gui_transverse_timedependent.py`
- `tests/test_pr_gui_transverse_zero_flux.py`
- this development record.

Unrelated dirty-tree content is excluded. This milestone does not change the
committed static or TD Fourier reference, static operator, nonlinear TD
equations or integrators, optical propagation physics, scattering/source
physics, product mathematics, IA mathematics, operation registration, Slurm,
cluster configuration, commissioning evidence, or soliton code. CuPy remains
compatible but uncommissioned for this production TD path.

# Full-Transverse Linearized PR Production Integration

## Status and scope

This milestone integrates the commissioned uniform-reference material
response into the existing `pr_transverse_static` production workflow. The
coupled production path is **Experimental**. The isolated constitutive
operator remains H200-commissioned at source commit `482acef`.

No new workflow identity was added. Reduced x-only physics, nonlinear
full-transverse equations, optical propagation, coherence grouping, Image
Amplification mathematics, and soliton workflows are unchanged.

## Independent model axis

`PRTransverseMaterialResponseSpec` represents the constitutive choice:

```text
model = nonlinear | linearized
reference_intensity = None | explicit positive I0
```

`PRTransverseStaticRunRequest.material_response` defaults to `nonlinear`.
Consequently old constructors, schema-v1/v2 experiment documents, and old
transport payloads retain the established fully nonlinear zero-flux meaning.
Linearized selection is explicit and is supported only by the existing static
full-transverse workflow. The additive request field follows every pre-existing
field, preserving the complete public positional constructor signature.
Internal workflow IDs remain unchanged.

The response spec is separate from evolution and transport. The resulting
axes are therefore:

```text
evolution: static | time dependent
transport: reduced x-only | full transverse
material response: fully nonlinear | linearized
```

Only the `static + full transverse + linearized` new combination is enabled.

## Scientific profile and parameter ownership

Linearized execution selects
`pr_full_transverse_periodic_biased_current_v1`: a periodic x/y bulk cell,
zero-mean perturbation potential, fixed harmonic mean field, and measured
mean current. The electrical boundary carries that new profile identity,
while the unchanged isotropic transport, dielectric, and projection component
profiles are reused. Configuration validation enforces this exact mapping.
The named v1 profile retains `m_y=h_y=1` and `E_active=E_x`.

The request must provide `reference_intensity` explicitly. It is the uniform
complete normalized transport intensity (I_0). The workflow does not infer
it from dark intensity, uniform background, beam peaks, or the instantaneous
spatial mean. This preserves the declared tangent state and records it
unambiguously.

The transverse electrical boundary profile is the sole owner of the
normalized harmonic `applied_field_x`. It maps once to the commissioned
operator and total-field reconstruction. `PRMaterialSpec.applied_field`
continues to belong to reduced x-only A7 physics and must remain zero for
every full-transverse request. Conflicting profile IDs or a nonzero reduced
field are rejected.

## Static workflow integration

The existing optical/material outer iteration is retained:

```text
accepted potential
→ canonical coherent PR-driving intensity
→ commissioned frozen-intensity Fourier response
→ total transverse field and canonical E_active projection
→ unchanged optical propagation
→ refreshed source
→ bounded outer line search and convergence check
```

The material branch calls
`solve_pr_biased_linearized_reference`; it does not duplicate the Fourier
formula. Production invokes it once per independent frozen transverse plane
and immediately retains the completed plane on the host. The CuPy working set
therefore stays `O(Nx Ny)` and never retains an `Nz` material volume on the
device. The planes form an optical-z source collection, not material-time
evolution or retained temporal history.

The optical source remains

\[
I=\sum_g\left|\sum_{c\in g}A_c\right|^2/I_{\rm peak}+I_b.
\]

Thus the response selection does not remove coherent interference. Total
`E_x` is reconstructed as `applied_field_x + delta_E_x`; the imposed mean
field is not included in the periodic potential and is not double-counted.

## Convergence and diagnostics

Each frozen-source material solve is analytic. It has no Newton, PCG, GMRES,
line-search, or material-iteration convergence status. Production convergence
instead measures the outer fixed-point consistency

```text
psi - linearized_response(refreshed_transport_intensity).
```

The existing bounded coupled line search, replay checks, cancellation at
accepted outer boundaries, source/field changes, and optical-power diagnostic
remain active. Linearized results label the authoritative residual
`material_response_consistency`; they do not expose the nonlinear-only TD-RHS
volume or discrete-corrector/Newton diagnostics. Lightweight provenance
records response-call count and the commissioned one-forward/four-inverse FFT
structure per call.

The result dataclass retains its historical `equilibrium_residual_stack`
field for transport compatibility, but the product title and diagnostics name
that volume as material-response consistency for linearized runs.

## Backend and precision

The material branch receives the workflow `BackendSpec` unchanged. Supported
combinations are NumPy float64/complex128, NumPy float32/complex64, CuPy
float64/complex128, and CuPy float32/complex64. Production tests execute both
NumPy precisions. CuPy correctness of the constitutive operator was
commissioned on H200; commissioning of the coupled production workflow is
deferred. No GPU performance claim is made here.

## Provenance and products

Every linearized result records:

- workflow and `transport_model=full_transverse`;
- `material_response` and explicit reference intensity;
- periodic biased current-carrying profile ID;
- fixed-harmonic-mean-field ensemble;
- authoritative transverse applied field;
- requested and resolved backend plus real/complex dtype and precision;
- equation/model identifier and Experimental status;
- computed mean current and mean-intensity perturbation per plane.

Full products preserve optical and reconstructible transverse fields while
omitting the inapplicable TD-RHS residual. Fast retrieval remains unchanged:
it retains compact optical endpoints, far field, diagnostics, and provenance,
without reintroducing three-dimensional material volumes.

## GUI, persistence, and transport

The GUI now uses the transport labels “Reduced x-only PR transport” and “Full
transverse PR transport.” The full-transverse static selection exposes a
separate `Material response` control with `Fully nonlinear` as the default and
`Linearized material response [Experimental]` as the explicit alternative.
Only the supported linearized combination exposes (I_0) and the transverse
applied mean field. Unsupported evolution/transport combinations reset to and
retain nonlinear response.

Experiment request schema v3 persists `material_response`. Decoders continue
to accept schema v1/v2 transverse-static documents and supply the nonlinear
default when the field is absent. The PR-owned transport payload adds the
same field additively; old payloads without it decode as nonlinear. Applied
field and (I_0) round-trip without substitution.

## Image Amplification and soliton seam

The existing full-transverse static Image Amplification capability boundary
can carry the request and consume the common optical endpoint/products. Its
specialized validation status remains Experimental; this milestone does not
promote it or change Image Amplification mathematics. The GUI therefore keeps
ordinary nonlinear transverse-static Image Amplification marked Validated but
shows the linearized material-response selection as Experimental and emits the
existing validation-pending warning.

Future soliton fixed-point machinery can reuse the request-owned material
response spec and the same commissioned frozen-source operator. No soliton
existence, stability, or eigensolver integration is implemented here.

## Time-dependent decision

Linearized `pr_transverse_timedependent` support is deferred. Although the
material tangent equation is derived, a production time integrator, state
semantics, and transient validation boundary have not been commissioned.
Repeated static solves are not used as a surrogate TD evolution.

## Local scientific comparisons

A deterministic 12 x 10, two-plane, broad-beam fixture verifies production
reuse of the commissioned response in float64 and float32, including
potential, both field components, carrier perturbation, dtype, and even-grid
null handling. A self-consistent biased case converged in one accepted outer
iteration, replayed exactly within configured tolerances, and had zero
reported optical-power drift at displayed precision.

At zero bias, the weak-modulation comparison against the unchanged nonlinear
zero-flux workflow produced:

| Quantity | Difference |
| --- | ---: |
| Potential relative L2 | `3.6839247e-4` |
| Potential maximum absolute | `5.1065798e-8` |
| Final optical field relative L2 | `5.3631536e-13` |
| Linearized consistency RMS | `6.2629751e-13` |

Both paths converged. For a deliberately stronger 30 µm-waist modulation,
both again converged but the potential relative L2 difference was `0.263921`
and the maximum-absolute difference was `0.0302147`. This demonstrates that
the selector chooses different constitutive physics outside the perturbative
regime.

## Validation status and deferred commissioning

The coupled production feature is **Experimental** after local validation.
The nonlinear default remains the previously validated production model. A
separate pre-commit review is required before commit, followed by a separately
authorized H200 commissioning of the coupled production path if approved.

Local validation completed with:

- focused linearized/reference, full-transverse static, TD non-enable, marching,
  and GUI tests: `79 passed`;
- affected Image Amplification GUI/status tests: `70 passed`;
- experiment persistence, experiment GUI, remote transport, and Slurm codec
  tests: `166 passed`;
- complete `tests/test_pr*.py` suite: `524 passed, 70 skipped`;
- syntax compilation of every changed Python file: passed;
- `git diff --check`: passed; supplemental no-index checks for both new
  untracked files were also clean.

The skipped tests are the repository's narrowly conditional backend/platform
tests; no CUDA device or cluster was accessed.

## Scientific non-change verdict

No nonlinear constitutive equation, tolerance, Newton/PCG implementation,
optical propagation kernel, or coherence/source equation changed. The existing
`static_workflow.py` necessarily gained the model-axis dispatch and additive
provenance while its nonlinear branch still calls the same zero-flux solver
with the same options. `workflow.py` and `marching_static.py` received only
explicit boundary-profile guards that preserve their former unbiased Profile-v1
acceptance boundary after the electrical profile type was extended. Existing
nonlinear regressions pass without tolerance changes.

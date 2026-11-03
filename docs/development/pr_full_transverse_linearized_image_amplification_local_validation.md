# Local validation of full-transverse linearized PR image amplification

## Scope and baseline

This record covers bounded local Development and Local Validation of Image
Amplification (IA) composed over the production full-transverse static PR
workflow with the linearized material response. The immutable baseline was
commit `977ed384677467fd01ecf962b75d74e2a475ec90` (`Integrate full-transverse
linearized PR response`) on branch `feature/pr-second-order-static`.

No production implementation changed. The permanent candidate consists only
of this record and
`tests/test_pr_transverse_linearized_image_amplification.py`. No CUDA device or
cluster was accessed, and no production, Fourier-response, nonlinear PR,
optical-propagation, IA-analysis, coherence/source, or soliton code changed.

## Architecture verdict

The inspected and executed path is:

```text
pr_image_amplification
  -> ordinary pr_transverse_static operation
  -> material_response=linearized
  -> canonical PRTransverseStaticRunResult
  -> unchanged local carrier isolation
  -> unchanged single full-length inverse linear propagation by -L
  -> unchanged local gain, correlation, and NRMSE analysis
  -> augmentation of the ordinary transverse-static RunData
```

There is no linearized-specific IA solver or postprocessor. A controlled gate
fed identical canonical optical endpoints through nonlinear- and
linearized-labelled base results. The carrier mask, isolated fields,
back-propagated fields, and all specialized IA field products were bitwise
identical. Measured gain, analytic gain, correlation, NRMSE, and relative power
drift were finite and exactly equal. A separate direct calculation reproduced
the stored amplified field with the existing FFT linear kernel evaluated at
exactly `dz=-L`.

## Deterministic paired fixtures

Both paired cases used NumPy float64, a `24 x 16` periodic grid, a
`48 um x 32 um` aperture, `L=10 um`, `dz=5 um`, one optical substep, identical
outer convergence controls, symmetric resolved carrier modes at
`kx=+/-2*pi*3/48 um^-1`, and two mutually coherent channels in the same
`linearized-ia-validation` coherence group. The same checksummed 8-by-8
checkerboard source, signal-channel screen placement, pump/signal roles, and
launch configuration were used within each pair. The material used dark
intensity `0.4`, uniform external background `0.1`, and zero applied bias.
Consequently, the explicit linearization reference was the uniform total
transport intensity `I0=1.5`; it was never inferred by the workflow.

The weak fixture used signal power `1e-8 mW`, pump power `1 mW`, nearly uniform
waists of `1e6 um`, screen transmission levels `0.98` and `1.0`, and
gain-length product `0.01`. The stronger fixture used signal power `0.08 mW`,
waists of `1000 um`, screen transmission levels `0.15` and `1.0`, and
gain-length product `0.16`. Apart from the compatibility-required electrical
profile identity, each nonlinear/linearized pair differed only in its material
response selection and explicit linearized `I0` declaration.

The nonlinear profile cannot represent nonzero periodic bulk bias. Therefore,
the scientific pairs correctly use zero bias rather than manufacturing a false
nonlinear comparison. A separate linearized IA smoke case at `E_app=0.2`
converged and completed local IA analysis.

## Weak-modulation agreement

The following errors compare the nonlinear result with the linearized result.
Relative L2 errors use the nonlinear quantity as denominator.

| Quantity | Relative L2 | Maximum absolute error |
|---|---:|---:|
| potential | `5.6852e-6` | `2.2538e-11` |
| `E_x` | `9.4802e-7` | `3.3256e-11` |
| `E_y` | `7.1368e-4` | `2.8377e-11` |
| active field | `9.4802e-7` | `3.3256e-11` |
| consistency residual | not meaningful at near-zero norm | `1.3707e-8` |
| initial optical field | `0` | `0` |
| final optical field | `2.2180e-13` | `1.6523e-14` |
| output intensity | `4.9830e-14` | `1.2349e-16` |
| far-field intensity | `1.0452e-20` | `2.1581e-16` |
| isolated output signal | `5.5573e-10` | `4.7810e-15` |
| amplified image | `7.5905e-10` | `1.9919e-20` |
| zero-response image | `0` | `0` |

The nonlinear/linearized IA metrics were respectively:

| Metric | Nonlinear | Linearized | Linearized - nonlinear |
|---|---:|---:|---:|
| measured gain | `0.996661661738` | `0.996661661739` | `1.4495e-12` |
| analytic gain | `0.995000148685` | `0.995000148685` | `0` |
| image correlation | `0.999999267860` | `0.999999267762` | `-9.8047e-11` |
| NRMSE | `6.33404534e-6` | `6.33444226e-6` | `3.9692e-10` |
| relative power drift | `0` | `-2.22045e-16` | `-2.22045e-16` |

The carrier masks were bitwise identical, their resolved carrier gradients and
mask populations were identical, and isolated output power agreed with direct
integration of the isolated signal field.

## Stronger-regime divergence

The stronger fixture retained exact initial-field and carrier-mask equality but
showed the expected approximation breakdown:

| Quantity | Difference |
|---|---:|
| potential relative L2 | `0.632762` |
| potential maximum absolute | `7.01818e-3` |
| final optical-field relative L2 | `1.05759e-3` |
| final optical-field maximum absolute | `7.40588e-5` |
| measured-gain difference | `1.50566e-4` |
| correlation difference | `-4.82684e-5` |
| NRMSE difference | `2.59195e-4` |

Both models converged and both IA analyses completed. The nonlinear and
linearized normalized power drifts were each `-2.23511e-16`. This divergence is
the expected loss of accuracy of a first-order material approximation, not a
workflow failure.

## Bias, provenance, transport, and products

The biased linearized smoke case retained `I0=1.5` and `E_app=0.2` in the
request, prepared base request, resolved profile, ordinary RunData summary, and
IA-augmented result. It converged with measured gain `0.996661627809`, image
correlation `0.999999268124`, NRMSE `6.33355e-6`, and power drift
`-2.22045e-16`.

The augmented result identifies the base workflow as `pr_transverse_static`,
the response as `linearized`, the physics profile as
`pr_full_transverse_periodic_biased_current_v1`, the validation status as
`experimental`, the explicit `I0`, `E_app`, NumPy backend, and float64
precision. Same-group coherence survives launch and appears in the canonical
launch summary; source construction therefore uses the authoritative
per-channel coherence groups rather than the obsolete global summary.

The specialized result contains and validates the current canonical fields:
`input_intensity`, `output_intensity`, `far_field_intensity`, `image_source`,
`image_transmission`, `input_signal_intensity`, `output_signal_intensity`,
`amplified_image`, `zero_response_image`, `signal_carrier_mask`, and
`far_field_log_db`. Data are finite, shapes and geometry are consistent, and
coordinates and presentation metadata survive augmentation.

Real transport projections were exercised for both Fast and Full policies.
After decode, all IA arrays and scalar metrics were exactly equal and all
specialized field data, axes, units, kinds, quantities, display extents, and
coordinates matched. Fast retained both optical endpoints, grid and launch
summaries, powers, status, `E_app`, `I0`, response/profile provenance, backend,
and precision while omitting the large three-dimensional material volumes.
This confirms that local IA augmentation does not require those volumes.

## Failure and status behavior

Focused regressions reject missing `I0`, a linearized response paired with the
unbiased nonlinear boundary profile, and attempts to select the linearized
static request under the time-dependent workflow. Failed base results skip IA
analysis; a base result missing its optical endpoint reports analysis failure
while preserving the base result. A non-converged base remains explicitly
non-converged even though the available optical endpoints can still be locally
analyzed. Existing cancellation-before/during-analysis behavior is covered by
the unchanged IA regression suite.

Ordinary nonlinear full-transverse IA remains workflow-level Validated and its
physics and products are unchanged. Coupled linearized IA remains
**Experimental** at the presentation and provenance layers. Local CPU evidence
does not satisfy the deferred remote commissioning gate.

## Representative local timing

One inexpensive local measurement gave:

| Fixture/model | Base workflow | IA reconstruction | Total composite |
|---|---:|---:|---:|
| weak nonlinear | `0.0231 s` | `0.000261 s` | `0.0251 s` |
| weak linearized | `0.0113 s` | `0.000247 s` | `0.0125 s` |
| strong nonlinear | `0.0558 s` | `0.000228 s` | `0.0570 s` |
| strong linearized | `0.0172 s` | `0.000232 s` | `0.0184 s` |
| weak biased linearized | `0.0121 s` | `0.000242 s` | `0.0133 s` |

These are bounded local CPU observations only and are not extrapolated to H200
or production Slurm performance.

## Local validation

The following local commands passed:

- focused linearized IA module: 8 passed;
- combined linearized production, IA GUI/status, transverse production/static,
  product/RunData, persistence, and Fast/Full coverage: 159 passed;
- additional nonlinear IA/readiness/static regressions: 46 passed;
- complete `tests/test_pr*.py` suite: 532 passed, 70 skipped;
- `py_compile` for the new test module;
- `git diff --check`.

The skipped tests are the suite's existing conditional hardware/dependency
coverage; local CuPy absence is not a blocker because this milestone did not
recommission the already commissioned constitutive operator.

## Deferred commissioning gate

After this local-validation milestone is reviewed and committed, the exact next
lifecycle step is a small H200/Slurm production-workflow commissioning: one weak
linearized full-transverse IA run, a matching nonlinear case if practical, Fast
retrieval, local IA augmentation, and runtime/GPU-memory comparison. Until that
evidence exists, the coupled linearized IA status remains Experimental.

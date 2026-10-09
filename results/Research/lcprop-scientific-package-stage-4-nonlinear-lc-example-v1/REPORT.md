# Stage 4 — bounded self-consistent LC static example

**Validated locally with original convergence gates.** No scientific-engine change.
Base installed wheel: Product `52b00928978a9ec7862a357d38327fd0b9ae857c`, version
0.1.0; wheel SHA-256 `1322400b31fac3a28d439f281cc189fb790f1526940dc36af9bf50971fdc5c20`.

## Recovered authority and scope

Exact single-beam request from
`tests/test_static_local_zmarch.py::test_static_local_self_consistent_marches_theta_and_field_along_z`.
The original z-local regression was introduced in commit
`5cb953ee7e4745a5f44062cdd18815f1b167e363`; later diagnostics/normalization and physical
launch updates are retained in the current test. It checks all-slice residual acceptance,
z-dependent director reorientation, optical/material alignment, physical-power
reconstruction, a trusted-scale final y width, and related TD response. It is a
scientific regression fixture, not a fully spatially converged research benchmark.
This task did not rerun its additional TD comparison or two-beam study.

Unlike the deliberately relaxed `1e9` soliton/smoke fixtures discussed in earlier
audits, this test requires unchanged normal RMS 0.005 / maximum 0.02 director gates.
Its existence therefore supports this **nonlinear propagation** demonstration without
inventing a new fixture. It does not provide stationary-soliton authority.

Parameters preserved exactly: 128×128 transverse grid, 75×100 µm aperture,
500 µm propagation, dz=5 µm (100 slices); one normal-incidence 1 mW Gaussian,
10 µm principal radii, x=-20 µm/y=0, wavelength 0.633 µm, coherence group A.
Material defaults ne=1.7/no=1.5, K=7e-12 N, dielectric anisotropy 13;
bias 0.9144 V, theta_bc=0; periodic optical boundary. NumPy float64, existing
optical substeps and all normalization conventions unchanged.

Original solver: local_self_consistent/picard_cn/splitstep/self_consistent;
max_iterations=3 (legacy field), 200 relaxation iterations per coupled pass,
3 coupled passes. RMS/max tolerances remain 0.005/0.02; no optional update tolerance
was introduced. A test extracts the original helper functions from the actual
historical fixture source and requires full request equality.

## Pre-execution resource decision

`pre-execution-budget.json` was written before execution. Three principal float64
128×128×100 volumes total 37.5 MiB; plane workspaces, product conversions, histories,
comparison outputs and plotting add overhead. Planning allowance 1 GiB process RSS,
120 seconds per scientific calculation; no guaranteed runtime was assumed. Worst
configured bound is 60,000 director relaxation iterations, not the observed count.
This is a bounded 100-slice regression rather than a grid/power sweep.

The installed CLI was supervised with a 120-second process cap; each public-runner
call also has a between-slice timeout callback. An initial supervisor attempted
external `ps` RSS sampling, which the sandbox denied before any output directory or
scientific result appeared. The supervisor was corrected to use child-reported
`resource.getrusage` high-water memory instead. No scientific parameters changed.
The 1-GiB estimate was thus checked post-execution, not enforced by live RSS polling.
No resident/bounded/native PR work was involved.

## Implementation and installed execution

New files only:

- `examples/lc_static_nonlinear_cpu.py`
- `examples/lc_static_nonlinear_cpu.md`
- `tests/test_lc_static_nonlinear_example.py`

The standalone example uses existing typed requests, public experiment validation,
analytic launch preflight, LocalRunner/LC_STATIC_OPERATION, RunData and public
`rms_widths`. No new public API or notebook/GUI changes. Requests are separate from
output paths and execution budget. Plotting is noninteractive Matplotlib Agg.

Executed a copied standalone script under `-I` from `/tmp` using the **existing**
disposable Stage 2 non-editable installation. Verified 264 installed package files
against their retained identities before execution; blocked Qt, CuPy and LaunchPlane
imports. Import resolved to
`/private/tmp/lc-static-example-h60l2fc_/venv/lib/python3.12/site-packages/lcprop`.
No installation or environment modification was needed.

Environment: Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, macOS arm64. Matplotlib
reported an unwritable default cache and used a temporary cache; plots succeeded.
This is not fresh-environment or Colab commissioning.

## Completion and convergence results

| Measurement | Self-consistent | Fixed dark-bias comparison |
|---|---:|---:|
| Completed slices | 100/100 | 100/100 |
| Scientific status | all slices converged | convergence not applicable |
| Runtime including product conversion | 14.9647 s | 0.182 s |
| Maximum final slice residual RMS | 0.004935085357 | not applicable |
| Maximum final slice residual maximum | 0.019985181431 | not applicable |
| Final slice x RMS width | 7.121488749 µm | 5.885449242 µm |
| Final slice y RMS width | 6.307060563 µm | 6.013741921 µm |

6275 director relaxation records are preserved with all 100 slice summaries in
`installed-output/summary.json`. Maximum reorientation relative to dark bias is
**0.08002581125 rad**. Every reported scientific array is finite. Nonlinear
normalized optical power is 0.9997691121920113 → 0.9997691121920986, relative drift
about 8.75e-14. Separately reconstructed physical flux diagnostics are
0.9997230027106438 → 0.9949732608568104 mW; they are not the dimensionless norm and
are not silently subjected to its conservation interpretation.

Process MaxRSS: **306,135,040 bytes (~292 MiB)** including plotting/output. These
single-run times and memory are observations, not performance guarantees.

## Fixed-director baseline and optical interpretation

The comparison changes only StaticWorkflowOptions to fixed_theta/none/frozen. It
uses Product's **dark/bias director initialized with initial_theta=None**, not the
illuminated final director. Material, bias, beams, numerical/grid/boundary settings
and solver-limit fields otherwise match. Input complex arrays are exactly identical;
fixed director equals the nonlinear run's initial bias exactly.

The width curves use Product RMS definitions on retained slice-average intensities,
with Product midpoint z coordinates. The last width row is at z=497.5 µm, whereas
the plotted output plane is at z=500 µm. Do not equate those samples.

There is transient focusing: y RMS minimum **2.176572280 µm at z=222.5 µm**, versus
**5.218569677 µm** in the fixed-bias comparison at that slice. Subsequent broadening
leaves both final RMS widths larger than the fixed case. Thus:

- **Successful execution:** yes, all requested slices completed.
- **Self-consistent slice convergence:** yes, under original finite residual gates.
- **Nonlinear reorientation/propagation:** demonstrated; the director differs from
  bias and optical widths differ with the same launch.
- **Nonlinear narrowing:** transient, not monotonic or an endpoint improvement.
- **Stationary soliton/stability:** not established and not claimed.

The preflight warns that the off-axis entrance envelope approaches the periodic
boundary; that warning is retained. No new spatial/time convergence or boundary-
independence study was authorized. The fixture was not moved or retuned to improve
beam narrowing or remove the warning.

## Scientific parity and validation

**4 focused tests passed in 30.86 s**:

1. Exact request equality to recovered existing fixture.
2. Fixed comparison changes only workflow; physical/default/gate preservation.
3. Actual example through public runner versus independent canonical `run_static`:
   exact A_initial, A_final, theta_final, theta_bias, intensity_stack; exact complete
   iteration history and slice summaries; original residual and finite-state gates.
4. Public width postprocessing with a coordinate-coded fixture preserves arrays.

The parity test executes the bounded nonlinear fixture twice in the same source
and numerical environment. Installed source was independently checked against the
same committed bytes. The standalone installed execution additionally exercised
plotting, experiment saving and selected array export. No claim of cross-platform
bitwise equality follows.

Command:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLBACKEND=Agg /Users/mcroning/miniforge3/envs/lcprop/bin/python -B -m pytest -q -p no:cacheprovider tests/test_lc_static_nonlinear_example.py
```

Actual PNG inspected; units and all four panels render. Export manifest independently
verified. Syntax, Python 3.10 grammar, candidate whitespace and git diff --check pass.
No unexpected regression failure; no broad/expensive test suite was run.

## Outputs, API limits and preservation

The example saves original Product experiment JSON for each mode, selected scientific
NPZs/coordinates/widths, full iteration diagnostics, comparison PNG and file hashes.
These are diagnostic exports, not full result transport or restart checkpoints.
Public request validation is distributed between encoder, sampling checks and the
workflow; no new facade was added. `max_iterations` is not the z-local relaxation
budget, so both fields are documented. Cooperative timeout acts at slice boundaries;
it is not an integrator modification or universal hard deadline.

Candidate SHA-256 identities are in `candidate-inventory.json`. All pre-existing
regular-file metadata and all 815 tracked-file hashes were verified unchanged.
Both unrelated tracked modifications, Stage 2/Colab evidence and existing untracked
work are preserved. Index empty. No engine, GUI/notebook, cluster, GPU, commit or push.


## Stage 4 closure — installed-package array comparison (2026-10-09)

**Acceptance recommendation: Stage 4 passes installed-package numerical validation
and is ready for a separately prepared Colab CPU qualification.** This is local
macOS qualification, not a hosted nonlinear LC result or stationary-soliton claim.
No scientific calculation or reference regeneration was performed for this closure.

### Authorities, requests and engine identity

Original retained authority: `installed-output/nonlinear.npz`, `fixed.npz`, both
experiment JSONs and summary from the initial Stage 4 installed-wheel validation.
Comparison target: repository-root `new-nonlinear-lc-run/`, supplied by the user's
`lcprop-new-user` run. Both directories' exact file membership and every manifest
SHA-256 were verified before comparison. The example and tests still match their
accepted candidate inventory. Both saved experiment files are byte-identical across
runs: nonlinear `b620ee2ebd878b89993dbe9ee66645da3aaabe6f793ff94089542ef19ebbbaa2`,
fixed `7aa94ff4e54b94105ad82ccb0a0e946e7031366e18c1b8c93e20ac3d2d570718`.

The reference uses engine 52b00928978a9ec7862a357d38327fd0b9ae857c, wheel
1322400b31fac3a28d439f281cc189fb790f1526940dc36af9bf50971fdc5c20.
The user's installed direct_url receipt identifies wheel
89855d2f22feef3d8b8544ac28131e978f67e42dac28d1ce8505d9f99d38a2ec.
`results/installed-pr-live-td-continuation-review/REPORT.md` binds that review wheel
to base 1f3c9cd plus the seven then-reviewed live-preview overlays, subsequently
committed as 985b80f. **Do not relabel that wheel as built from 52b0092.**
An independent read-only comparison found all **264 installed package files**
byte-identical to the Stage 2/4 committed-engine package inventory. Thus the engine
code is compatible by bytes despite distinct wheel/source-provenance histories.
Version 0.1.0 alone would not have established this.

Installed import location recorded by the run is
`/Users/mcroning/miniforge3/envs/lcprop-new-user/lib/python3.12/site-packages/lcprop/__init__.py`.
Reference environment: Python 3.12.13 / NumPy 2.5.1 / SciPy 1.18.0.
User environment: Python 3.12.14 / NumPy 2.5.3 / SciPy 1.18.1. Both macOS arm64.
There are **no numerical array differences to attribute** to these library versions.
The scientific requests are identical; only environment metadata and measured
runtimes differ in the summaries. Current installed-file verification corroborates
the recorded import/installation; it is not a separately captured run-time file
attestation. No unsupported attribution to BLAS/FFT/library changes is made.

### Complete scientific-array comparison

SHA-256 is over C-order bytes in each stored dtype; shape/dtype and finiteness are
checked independently. All **18/18 arrays are bitwise identical**. Because both
hashes are equal, the last column gives the **reference SHA-256 = installed SHA-256**.
Separate hash fields, differing-element counts and thresholds are preserved in
`installed-parity-review/comparison.json`.

| Array | Dtype / shape | Bitwise | Max abs | RMS abs | Relative L2 | Max relative | Reference SHA-256 = installed SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| nonlinear/A_initial | complex128 [1, 128, 128] | Yes | 0 | 0 | 0 | 0 | `d0512443687a3d556ce18790f8dbb669c57a41b8980d0c6595c7fec951b3fba3` |
| nonlinear/A_final | complex128 [1, 128, 128] | Yes | 0 | 0 | 0 | 0 | `599b279c096952da2650041d428791c4748810134c0e8e6ae126a8d4cd9fbbaa` |
| nonlinear/theta | float64 [100, 128, 128] | Yes | 0 | 0 | 0 | 0 | `b7cf30880d188fcba003c479ea290463080fbd0e854a285660a2c374ce232fb9` |
| nonlinear/theta_bias | float64 [128, 128] | Yes | 0 | 0 | 0 | 0 | `fd4005b4df100022e08758acdbdda85ff7e0a2bb576034b06a1ef6453bdab446` |
| nonlinear/intensity_stack | float64 [100, 128, 128] | Yes | 0 | 0 | 0 | 0 | `75547c769c6e275845776a9ebc49effbb7b7c85b2f5f7056c28640d9fe559690` |
| nonlinear/widths_um | float64 [100, 2] | Yes | 0 | 0 | 0 | 0 | `4072720ade767552d46b0698f5de6149e34c0e576df08fa2e0ee5433909a15f5` |
| nonlinear/x_um | float64 [128] | Yes | 0 | 0 | 0 | 0 | `61fd04d91f7ed8fa5bcb1b047c3748a5edf957eb61bee7d1e9199d1adfc5e064` |
| nonlinear/y_um | float64 [128] | Yes | 0 | 0 | 0 | 0 | `b460fd98873c22484e76cf22dc5bb35d6faf80ea86bed899c9c9bc4383856780` |
| nonlinear/z_um | float64 [100] | Yes | 0 | 0 | 0 | 0 | `7d3fd1ec319efad4fa839adeaea318e9853b12a37a979199e067a7d8ac8552f4` |
| fixed/A_initial | complex128 [1, 128, 128] | Yes | 0 | 0 | 0 | 0 | `d0512443687a3d556ce18790f8dbb669c57a41b8980d0c6595c7fec951b3fba3` |
| fixed/A_final | complex128 [1, 128, 128] | Yes | 0 | 0 | 0 | 0 | `8dbce0a85426739c9a9a913384e16a7dbe394c1e71f202a8e210fa2e7c03f67c` |
| fixed/theta | float64 [128, 128] | Yes | 0 | 0 | 0 | 0 | `fd4005b4df100022e08758acdbdda85ff7e0a2bb576034b06a1ef6453bdab446` |
| fixed/theta_bias | float64 [128, 128] | Yes | 0 | 0 | 0 | 0 | `fd4005b4df100022e08758acdbdda85ff7e0a2bb576034b06a1ef6453bdab446` |
| fixed/intensity_stack | float64 [100, 128, 128] | Yes | 0 | 0 | 0 | 0 | `ff3f278e884b46e1dab3c6cd59af484de50cf50b2221b4b88e823eda9ef98110` |
| fixed/widths_um | float64 [100, 2] | Yes | 0 | 0 | 0 | 0 | `70a07f401e27320aa1d6fe748f6bd806a34210d1b99acb42a5c7373913e6e91b` |
| fixed/x_um | float64 [128] | Yes | 0 | 0 | 0 | 0 | `61fd04d91f7ed8fa5bcb1b047c3748a5edf957eb61bee7d1e9199d1adfc5e064` |
| fixed/y_um | float64 [128] | Yes | 0 | 0 | 0 | 0 | `b460fd98873c22484e76cf22dc5bb35d6faf80ea86bed899c9c9bc4383856780` |
| fixed/z_um | float64 [100] | Yes | 0 | 0 | 0 | 0 | `7d3fd1ec319efad4fa839adeaea318e9853b12a37a979199e067a7d8ac8552f4` |

Maximum relative difference is measured only at reference elements with magnitude
>= `max(1e-300, 1e-12 * max(abs(reference)))`. This is a denominator reporting rule,
not a new acceptance tolerance. Every absolute/relative difference is zero; no
numerical tolerance was needed. No phase alignment, rounding or rescaling was used.

### Diagnostics and accepted propagation

All nonlinear and fixed scientific summary entries match exactly, including all
**6275 relaxation records**, **100 slice summaries**, termination/convergence flags,
residuals, normalized/physical powers and warnings. Only the elapsed-time fields
differ within those result summaries. The nonlinear run completes 100/100 slices;
every final slice passes the original RMS <=0.005 / max <=0.02 gates. Recorded
maxima remain **0.0049350853571153985 / 0.019985181430640875**.

The saved RMS-width curves were additionally recomputed from retained intensity
arrays and coordinates only; they match saved width arrays exactly. Maximum director
reorientation was likewise checked from retained theta/bias and equals
**0.0800258112453327 rad** in both summaries. Endpoint-slice widths remain nonlinear
(7.1214887487, 6.3070605625) µm and fixed (5.8854492421, 6.0137419214) µm.
Normalized initial/final powers remain 0.9997691121920113 / 0.9997691121920986.
Physical mW diagnostics are separately identical; they are not substituted for the
normalized norm. Fixed-baseline self-consistent convergence remains not applicable.
The user's nonlinear runtime is approximately **13.620 s**, versus reference
14.965 s; no causal performance inference follows from these two observations.

### Figure ownership and artifact coherence

Both original manifests bind each PNG together with its experiment, summary and
NPZ files. The unchanged example source generates them from the same completed
result objects; the matching requests, arrays, histories and timestamps provide
consistent artifact provenance. The PNG file bytes differ, but decoded 1300×1040
pixels are **exactly identical**. PNG metadata records Matplotlib **3.11.0** in the
reference and **3.11.2** in the user image. Both images were inspected; panels,
labels and curves agree. This is evidence of a rendering metadata difference,
not a scientific difference or a stale plotted dataset. No figures were regenerated.
Manifests are integrity bindings, not cryptographic proof of process causality;
there is no contradictory or stale-artifact evidence.

### Limits and preservation

The authority is the original retained **installed-example output**, with its exact
canonical-workflow parity established by the Stage 4 focused tests. The independent
canonical test run itself did not save a second full reference archive. No such
missing archive was silently recreated. The example's selected exports contain
all 18 arrays compared here, but do not retain every internal solver array (for
example, the complete midpoint torque-source volume). This closure covers retained
arrays and full scalar iteration/slice diagnostics, not unexported internals.

Original archives, user-run files, source, tests and scientific evidence were not
changed. Only this requested report appendix and new installed-parity-review files
were written. No simulations, installation, source modification, commit or push.

**Stage 4 accepted for the demonstrated local self-consistent LC propagation example;
ready for Colab CPU qualification preparation. No stationary-soliton claim.**

# PR Coupled-Static GPU Commissioning Report

**Commissioning record:** 1

**Date:** 2026-08-06

**Branch:** `feature/pr-second-order-static`

**Run status:** Passed after environment-only remediation

## Milestone Summary

- First successful scheduler-assigned GPU commissioning of the PR coupled-static workflow.
- Commissioned the backend-native cyclic-tridiagonal solver, batched fixed-intensity PR static solve, and complete coupled optical/material static workflow with CuPy.
- Verified the GPU implementations against their NumPy or dense CPU references at the tolerances encoded in the reviewed tests.
- Verified optical-power conservation within a relative tolerance of `2e-13` for the coupled workflow case.
- Preserved the original blocked run and its import-path diagnosis as part of the commissioning history.
- Established GPU operational readiness for bounded PR static scientific validation; it did not establish production-scale static image-amplification results.

## Objective

This commissioning calculation tested the new PR coupled-static numerical milestone on a scheduler-assigned GPU. Its purpose was to establish that the production cyclic solver, fixed-intensity material solve, and complete slice-local coupled static workflow execute through CuPy without NumPy backend fallback and agree with their CPU references at the reviewed tolerances.

This was a numerical commissioning exercise, not a research benchmark. The request deliberately ran three small deterministic regression tests rather than a production-scale optical calculation. Passing therefore establishes that the implemented GPU paths work for the commissioned cases; it does not establish spatial convergence, robustness in difficult nonlinear regimes, or quantitative image-amplification science.

## Software and Reproduction Identity

| Item | Recorded value |
|---|---|
| Git checkout | `243c2f62501829d03170fb3f05c2b1c266b3d6a5` |
| Repository | `/cluster/tufts/cglab/mcroning/LCProp` |
| LCProp import path | `/cluster/tufts/cglab/mcroning/LCProp/src/lcprop/__init__.py` |
| Python executable | `/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python` |
| Slurm script | `/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop_pr_static_gpu_validation.sbatch` |
| Successful script SHA-256 | `34f794321bd6eeebdbf8d0e7858d8cb2401e6b378a88d3dd7c5942b94c85ca34` |
| CUDA module | `cuda/12.9.0` |
| Persistent run directory | `/cluster/tufts/cglab/mcroning/lcprop_runs/pr-static-gpu-2221987` |
| Local retrieval directory | `/private/tmp/lcprop-pr-static-gpu-2221987` |

The exact submission command was:

```bash
ssh -o BatchMode=yes mcroning@login.pax.tufts.edu \
  'sbatch --parsable /cluster/tufts/cglab/mcroning/lcprop_runs/lcprop_pr_static_gpu_validation.sbatch'
```

The scheduler script verified the Git SHA and repository cleanliness before invoking pytest. It also recorded the resolved LCProp import path, `PYTHONPATH`, CUDA module, compiler version, GPU identity, package versions, stdout and stderr locations, and process exit status.

The compact [metrics and provenance summary](assets/pr_coupled_static_gpu_commissioning_2026-08-06/metrics.json) and [raw successful-run provenance](assets/pr_coupled_static_gpu_commissioning_2026-08-06/provenance.txt) are preserved with this report.

## Original Blocked Run and Remediation

The first approved attempt, Slurm job `2221964`, reached `pax007`, acquired an NVIDIA A100 GPU, loaded CUDA successfully, and verified the correct detached Git SHA. It nevertheless failed during pytest collection with exit code `4:0` after 11 seconds. JUnit recorded three collection errors and no executed tests.

The Python environment imported an older LCProp installation from:

```text
/cluster/tufts/cglab/mcroning/condaenv/prenv/lib/python3.10/site-packages/lcprop/__init__.py
```

That installed copy predated `lcprop.pr.cyclic`, `lcprop.pr.static_workflow`, and `solve_pr_static_intensity_batched`. The checked-out tests therefore referred to modules that were present at Git SHA `243c2f6` but absent from the imported package. This was an execution-environment path error, not a numerical test failure.

The remediation changed only the Slurm environment setup:

```bash
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
```

The script also recorded the resolved `PYTHONPATH` in provenance. Production code, tests, numerical tolerances, requested resources, CUDA module, Python environment, and Git checkout were unchanged. A non-submitting import probe confirmed that LCProp and the new PR modules resolved from the detached checkout before the second submission. The blocked script had SHA-256 `c1702d12ec70fb5c7911ba6b10e5e755afe1a496a4cf21319a0dccdfa92cb351`; the remediated successful script had SHA-256 `34f794321bd6eeebdbf8d0e7858d8cb2401e6b378a88d3dd7c5942b94c85ca34`.

## Cluster Configuration

| Item | Value |
|---|---:|
| Slurm job ID | `2221987` |
| Scheduler result | `COMPLETED`, exit code `0:0` |
| Compute node | `pax007` |
| Partition / QOS / account | `gpu` / `normal` / `default` |
| Nodes / tasks | 1 / 1 |
| CPUs per task | 2 |
| Requested host memory | 8 GiB |
| Requested wall time | 5 minutes |
| Requested GPUs | 1 |
| Scheduler elapsed time | 24 seconds |
| GPU | NVIDIA A100 80GB PCIe |
| CUDA-visible device | 0 |
| NVIDIA driver | 575.57.08 |
| CUDA module | `cuda/12.9.0` |
| CUDA compiler | 12.9.41 |
| CuPy | 13.6.0 |
| Python | 3.10.4 |
| NumPy / SciPy | 2.1.0 / 1.15.2 |
| Pytest | 8.4.2 |

CuPy reported one visible device and identified it as the scheduler-assigned NVIDIA A100. The coupled workflow test explicitly required `backend=cupy` and `is_gpu=true`; those assertions passed. The exact checkout's `src` directory was first on `PYTHONPATH`, and provenance confirmed that `lcprop.__file__` resolved there rather than to `site-packages`.

## Commissioned Tests

The successful job executed exactly these tests:

1. `tests/test_pr_cyclic.py::test_batched_cyclic_solver_is_cupy_compatible_when_available`
2. `tests/test_pr_static.py::test_batched_static_material_solve_stays_on_cupy_when_available`
3. `tests/test_pr_static_workflow.py::test_cupy_static_workflow_agrees_with_numpy_when_available`

The scheduler script audited the JUnit result and failed unless it contained exactly three tests, zero failures, zero errors, and zero skips. The recorded result was:

```text
3 passed in 13.69s
junit_tests=3 failures=0 errors=0 skipped=0
```

### Numerical Acceptance Criteria

| Component | Compared quantities | Relative tolerance | Absolute tolerance | Result |
|---|---|---:|---:|---|
| Cyclic solver | CuPy cyclic result versus dense CPU reference | `2e-13` | `2e-13` | Passed |
| Fixed-intensity static solve | Batched CuPy `E` versus dense NumPy `E` | `2e-12` | `2e-13` | Passed |
| Coupled static workflow | `E_final`, `A_final`, and `source_intensity_stack` versus an identical NumPy request | `2e-11` | `2e-12` | Passed |
| Optical-power conservation | `power_final` versus `power_initial` using `pytest.approx` | `2e-13` | Pytest default | Passed |

The tests asserted these limits directly. They did not emit the measured elementwise differences or relative power drift, so the permanent claim is intentionally tolerance-bounded: each comparison and the power-conservation check passed at its declared tolerance. No more precise numerical value is inferred from the pass result.

## What Was Validated

This commissioning run validated, for the tested deterministic `float64` cases:

- CuPy acquisition of a real scheduler-assigned GPU;
- import and execution from the exact detached LCProp checkout;
- device-native execution of the cyclic-tridiagonal solver;
- the cyclic solver's agreement with a dense CPU reference;
- device-native execution of the batched fixed-intensity PR static solve;
- fixed-intensity CuPy/NumPy agreement;
- complete coupled-static optical/material workflow execution with CuPy;
- coupled-workflow agreement with an identical NumPy request for final material state, final optical field, and refreshed source-intensity stack;
- reported CuPy backend identity with no accepted NumPy fallback;
- optical-power conservation within the declared relative tolerance;
- capture of scheduler, CUDA, package, import-path, JUnit, stdout, stderr, and exit-code evidence.

## What Was Not Yet Validated

This commissioning run did not validate:

- production-scale static image-amplification science;
- convergence with transverse or longitudinal grid refinement;
- finite-aperture or boundary sensitivity;
- difficult nonlinear regimes, multiple-solution behavior, or broad convergence basins;
- continuation or checkpoint workflows for the coupled-static solver;
- execution through generic workers, products, or persistence dispatch;
- GUI construction, launch, visualization, or result handling;
- GPU performance scaling or a CPU/GPU performance comparison;
- measured GPU memory high-water behavior;
- exact CPU/GPU error magnitudes below the asserted thresholds.

The result should therefore be read as operational and numerical-path commissioning, not as validation of a particular static PR experiment.

## Runtime, Memory, and Output Status

| Measurement | Result |
|---|---:|
| Scheduler elapsed time | 24 s |
| Pytest time | 13.688 s |
| Cyclic solver test | 9.835 s |
| Fixed-intensity solve test | 1.331 s |
| Coupled workflow test | 1.970 s |
| Batch MaxRSS | 139,620 KiB |
| Scheduler exit status | `COMPLETED`, `0:0` |
| Recorded process exit code | 0 |
| Standard output | Test summary only; expected and nonempty |
| Standard error | Empty |

These timings include first-use GPU and kernel initialization effects and are not performance benchmarks. The scheduler accounting captured host resident memory, not GPU peak allocation.

## Data Preservation

The following compact artifacts are committed with this report:

- `metrics.json`, SHA-256 `cb129ef1fe541c7b6c5c7102464f03d61727eb47c074245b0792ee82cff04c54`
- `provenance.txt`, SHA-256 `f0e3af62ec9d59c32f5070566cea88f13f45ea91789bd2edaa0b81995aa89235`

The complete small successful-run evidence remains remotely at:

```text
/cluster/tufts/cglab/mcroning/lcprop_runs/pr-static-gpu-2221987
/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-static-gpu-2221987.stdout
/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-pr-static-gpu-2221987.stderr
```

A retrieved working copy was stored at `/private/tmp/lcprop-pr-static-gpu-2221987` when this report was prepared. That local temporary path is not a durable archive. The original blocked-run evidence remains at `/cluster/tufts/cglab/mcroning/lcprop_runs/pr-static-gpu-2221964` and `/private/tmp/lcprop-pr-static-gpu-2221964`.

No raw optical or material arrays were produced or committed. The JUnit XML, stdout, empty stderr, and exit-code files are omitted from Git because their compact substantive content is represented in `metrics.json` and `provenance.txt` and remains in the recorded run directories.

## Conclusion

The PR coupled-static GPU numerical milestone passed commissioning on an NVIDIA A100. At exact Git SHA `243c2f62501829d03170fb3f05c2b1c266b3d6a5`, the cyclic solver, batched fixed-intensity material solve, and complete coupled optical/material static workflow executed through CuPy and passed their CPU-reference and power-conservation assertions. The original failed attempt was traced to a stale installed-package import path and corrected solely by making the exact checkout authoritative through `PYTHONPATH`.

This result removes GPU backend execution as an immediate blocker for PR static research. It does not yet establish production-scale scientific behavior or numerical convergence for a static image-amplification problem.

## Scientific Significance

This milestone establishes the first GPU-validated implementation of the coupled PR static workflow. It demonstrates that the backend-native cyclic solver, batched fixed-intensity Newton material solve, and coupled optical/material static iteration execute correctly on production GPU hardware and agree with CPU reference calculations within the reviewed tolerances.

This commissioning validates the numerical infrastructure required for coupled-static PR studies. It does not by itself validate static image amplification, convergence of production calculations, or long-time physical behavior.

## Recommended Next Scientific Step

Use the commissioned coupled-static workflow to perform one bounded static image-amplification pilot using the same image, beam geometry, material parameters, reconstruction pipeline, and evaluation metrics as the previously commissioned transient image-amplification study.

The objective is to compare the static solution directly with the long-time transient solution before beginning broader parameter studies or production-scale calculations. Record the coupled residuals, replay consistency, material state, optical field, source intensity, signal gain, reconstruction metrics, and total power. Where practical, report actual NumPy/CuPy differences rather than only assertion thresholds.

Hold the image, physical model, beam geometry, grid, aperture, and evaluation method fixed for this comparison. Only after the static and sufficiently relaxed transient solutions are understood should grid refinement, aperture studies, production-scale calculations, or difficult nonlinear regimes begin.

---

End of commissioning record.

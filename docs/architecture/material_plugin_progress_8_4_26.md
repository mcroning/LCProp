# Material Plugin Progress — August 4, 2026

**Prepared:** 2026-08-05

**Branch:** `feature/pr-gui-vertical-slice`

**Current implementation commit:** `b45ff56` — Add standalone photorefractive GUI

**Review purpose:** Present the August 4 material-composition work for independent architectural review by ChatGPT.

## Executive Summary

On August 4, LCProp advanced from a material-neutral optical propagation seam to a complete in-tree photorefractive application vertical slice. The work added explicit material/workflow operation composition, PR continuation and disk checkpoints, shared material-aware checkpoint dispatch, a headless execution integration test, and a standalone PR GUI. The existing LC entry points and user interface were preserved.

The result is not an external plugin framework. It is evidence that an independent material package can now own its request, physical state, workflow, optical-response preparation, continuation, persistence payload, product conversion, and GUI composition while reusing LCProp's optical engine, execution services, checkpoint dispatch, beam editor, and result workspace. The design remains deliberately simple: dataclasses, callables, explicit identifiers, and explicit registration. It does not use inheritance, dynamic discovery, a universal material-state class, or a public third-party plugin API.

The central architecture remains:

```text
Beam definitions
        │
        ▼
Generic optical launch
        │
        ▼
Optical fields A
        │
        ├──────────────┐
        ▼              ▼
 LC source          PR source
        │              │
        ▼              ▼
 LC state theta     PR state E
        │              │
        ▼              ▼
 LC response      PR response
        └──────┬───────┘
               ▼
advance_prepared_response()
```

The August 4 work extended this physical boundary upward into execution, persistence, products, continuation, and a material-specific application. It did not generalize or reorganize the LC physics packages.

## Starting Point and Constraints

Before this work, the following had already been established:

- `advance_prepared_response()` accepted a material-prepared shared `(Nx, Ny)` or per-channel `(Nch, Nx, Ny)` complex half-step response.
- LC and PR both used the same optical propagation engine.
- PR owned the normalized space-charge state `E`, its hopping-model evolution, and the conversion from `E` to an optical response.
- Headless PR two-beam coupling had been validated with periodic plane waves and finite Gaussian beams.
- The repository architecture review concluded that the optical layer was material-neutral, while execution, persistence, and GUI composition remained incompletely exercised by a second material.

The approved implementation constraints were:

- preserve existing LC entry points and behavior;
- use simple dataclasses, callables, and explicit registration;
- do not introduce inheritance or dynamic plugin discovery;
- add tests at each boundary;
- do not move packages or generalize LC algorithms;
- stop before an external plugin API or universal material model.

## Work Completed

### 1. Explicit material/workflow operation composition

The execution layer now accepts an explicitly registered `WorkflowOperation`. Each operation supplies:

- a stable `material_id`;
- a stable `workflow_id`;
- a concrete workflow callable;
- a concrete result-to-`RunData` adapter.

PR exposes `PR_TIMEDEPENDENT_OPERATION` with the identity `pr/pr_timedependent`. `LocalRunner` can execute a supplied operation directly or look it up by its exact registered key. It returns both the material-owned physical result and the shared presentation product.

Legacy LC methods such as `run_static()` and `run_timedependent()` remain compatibility entry points. There is no global discovery mechanism and no material-name conditional in the generic registered-operation path.

### 2. PR checkpoint continuation

PR now has a material-owned continuation contract based on `PRTimeDependentCheckpoint`. The checkpoint preserves the physical state needed for an exact continuation:

- the original launch field `A0`;
- the initial PR state `E_initial`;
- the current PR state `E_current`;
- the request and completed material-step/time metadata.

Continuation validates compatibility, resumes from the preserved `E_current`, and advances by an explicitly requested number of additional material steps. It does not reduce the checkpoint to an optical phase screen or refractive-index perturbation; the true PR state remains authoritative.

### 3. PR-owned disk codec and shared checkpoint dispatch

PR owns its versioned checkpoint serialization and payload interpretation. The shared persistence layer owns only composition and dispatch through `CheckpointCodec` and `CheckpointCodecRegistry`.

The registry dispatches saves by checkpoint type and identified loads by `(material_id, workflow_id)`. It rejects duplicate identities, duplicate checkpoint types, unknown material/workflow identities, mismatched loaded types, and ambiguous dispatch.

Existing LC static and time-dependent checkpoint functions and disk formats were retained. Explicit legacy aliases map historical material-less workflow metadata to LC. New material formats do not receive that fallback.

This provides a model for future material-owned codecs without first moving or rewriting the existing LC codecs.

### 4. Headless end-to-end composition proof

An integration test exercises the complete PR path through shared local execution:

```text
PRRunRequest
    -> registered PR operation
    -> LocalRunner
    -> PR workflow
    -> prepared optical response
    -> shared optical propagation
    -> PRRunResult
    -> PR product adapter
    -> RunData
```

The test verifies identity, physical result preservation, shared products, execution status, progress, cancellation behavior, and deterministic operation registration without changing LC dispatch.

### 5. Standalone PR GUI

A separate `LCProp PR` application was implemented rather than adding a material selector to the LC window. This reflects the product expectation that most users will work primarily with either LC or PR physics and should see terminology and controls appropriate to that material.

The PR window provides:

- **PR Material**, **Beam**, **Grid**, **Evolution**, and **Results** tabs;
- PR-specific normalized material fields and solver controls;
- shared beam editing with PR-owned defaults;
- shared `RunData` result presentation;
- background execution through `WorkflowWorker`;
- progress reporting, cooperative stop, and safe application shutdown;
- checkpoint save, load, compatibility checking, and continuation;
- request summaries and preflight sampling, aperture, and timestep warnings.

The GUI constructs the exact registered `pr/pr_timedependent` operation through `LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))`. It does not call an alternative GUI-only physics path.

The launch controls continue to expose transverse phase gradients in `rad/µm`, not geometric angles. The inverse beam adapter used during checkpoint loading preserves channel order, name, wavelength, power, waists, centers, phase gradients, phase, and coherence group. The LC compatibility field `theta_weight` is not exposed as a PR control.

The application can be started with:

```bash
python -m lcprop.pr.gui.app
```

After installing the project entry point, it can also be started with:

```bash
lcprop-pr
```

## Implemented Call Paths

### Fresh interactive run

```text
PR GUI panels
    -> build_pr_request()
    -> PRRunRequest
    -> WorkflowWorker
    -> PRMainWindow._run_registered()
    -> LocalRunner.run_registered("pr", "pr_timedependent", ...)
    -> PR_TIMEDEPENDENT_OPERATION
    -> run_pr_timedependent()
    -> advance_prepared_response() for optical advancement
    -> PRRunResult
    -> pr_result_to_run_data()
    -> shared result workspace
```

### Checkpoint save and load

```text
PRTimeDependentCheckpoint
    -> save_run_checkpoint()
    -> CheckpointCodecRegistry
    -> PR checkpoint codec
    -> versioned PR metadata and arrays

checkpoint directory
    -> load_run_checkpoint()
    -> material/workflow identity dispatch
    -> PR checkpoint codec
    -> validated PRTimeDependentCheckpoint
```

### Interactive continuation

The GUI hydrates editable controls from the checkpoint request, but `A0`, `E_initial`, and `E_current` remain owned by the checkpoint. The current GUI material-step count is interpreted as the number of additional steps. Compatibility validation ignores only the original `Nt`; incompatible edits disable continuation and display the reason without discarding the loaded checkpoint.

## Validation Evidence

The following results were recorded during the August 4 development and review cycle:

| Scope | Result |
|---|---:|
| GUI and request-adapter focus | 38 passed |
| PR workflow and composition focus | 61 passed, 1 skipped |
| Checkpoint and persistence focus | 61 passed |
| Broad GUI and checkpoint focus | 94 passed |
| Final relevant PR GUI suite | 96 passed in 30.53 s |
| Real offscreen Qt window startup/shutdown | Passed |

An earlier repository-wide run recorded 326 passed, 2 skipped, and 5 failures. Four failures were multiprocessing restrictions in the sandbox and passed when rerun outside it. The remaining isolated failure was the pre-existing/current LC off-axis bias-coupling test, `tests/test_offaxis_bias_coupling.py::test_offaxis_beam_receives_bias_and_updated_self_consistent_theta`; none of the PR GUI changes entered that LC static workflow path. This was not established by a clean-checkout baseline comparison, so it should be described as unrelated by code path rather than conclusively classified as a historical baseline failure.

No LC workflow, LC GUI implementation, LC checkpoint format, or module under `src/lcprop/algorithms/` was changed by the standalone PR GUI commit.

## Interactive Physics Checks

### Two-beam coupling and transverse resolution

The first GUI trial used a `128 x 128` grid over a `200 µm x 200 µm` periodic aperture with phase gradients `kx = ±1.6 rad/µm`. The coherent grating had only about 1.26 samples per period. The GUI correctly warned that the grating was inadequately sampled, and the displayed transfer was not physically useful.

The case was repeated with `Nx = 1024`, `Ny = 128`, the same aperture, a `1000 µm` interaction length, `dz = 10 µm`, gain-length product 3, refractive index 2.4, two equal `20 µm` waist beams launched from `x = -35 µm` and `x = +35 µm`, and phase gradients `+1.6 rad/µm` and `-1.6 rad/µm`. Forty material steps at normalized `dt = 0.03` produced no preflight warnings and showed clear directional two-beam energy transfer. This is an interactive confirmation of the already validated headless coupling behavior and demonstrates why PR-specific sampling checks are necessary.

The current undersampling check is a warning rather than a hard rejection. Whether physically unusable coherent-grating resolutions should be rejected is an open product decision.

### Longitudinal selection in the result workspace

The 2-D view labeled `Final PR Space-Charge Field` initially displays the middle longitudinal slice. Here, “Final” refers to final material time, not to the output `z` plane. Selecting a position in an `x-z` or `y-z` view changes the longitudinal slice shown in the 2-D field view.

The data are not being sampled incorrectly at the workflow level, but the initial midpoint selection and label can make the field appear to be an on-axis midplane result. This is a presentation ambiguity worth correcting in a later, narrowly scoped GUI polish step.

### Exploratory screening-soliton observation

A preliminary single-beam run was made with normalized applied field 5, uniform background 0.5, dark intensity 0.01, a `10 µm` waist, a `4 mm` interaction length, and normalized material time 3.6. The beam path did not yet show the monotonic shift expected from the paper's mature screening-soliton example.

This is not treated as a failure of the material-composition milestone. Screening solitons have not yet been implemented as a validated benchmark, and the trial had several confounding factors: early material time relative to the paper's later results, a small periodic aperture with an explicit boundary warning, possible physical-to-normalized field and background differences, and differences in numerical kernel and boundary treatment. The observation should be retained for the future soliton validation plan without diverting the current architecture work.

## Commit and Branch Record

All six commits below were made on August 4, 2026, in one ancestry chain:

| Commit | Branch milestone | Change |
|---|---|---|
| `5a44bce` | `arch/pr-response-spike` | Add material operation composition seam |
| `21a51dd` | `feature/pr-checkpoint-continuation` ancestry | Add PR checkpoint continuation |
| `31d5366` | `feature/pr-checkpoint-continuation` | Add PR checkpoint disk codec |
| `ca615d7` | `feature/material-persistence-composition` | Add material checkpoint codec composition |
| `c8d1d51` | `feature/pr-headless-integration` | Add PR headless composition integration test |
| `b45ff56` | `feature/pr-gui-vertical-slice` | Add standalone photorefractive GUI |

The final branch was pushed to `origin/feature/pr-gui-vertical-slice`. Commit `b45ff56` changed 16 files with 2,473 insertions and 2 deletions. Its principal additions were the PR GUI package, request and inverse launch adapters, PR aperture analysis, tests, the launcher entry point, and the permanent GUI design record at `docs/architecture/pr_gui_vertical_slice_design.md`.

## Architectural Assessment

The work supports the following conclusions:

1. **The optical plugin boundary is real.** LC and PR produce different physical states and responses but use the same field launch and propagation engine.
2. **A second material can use shared local execution.** Material/workflow identity and ordinary callables are enough for the present in-tree use case.
3. **Physical state can remain material-owned through continuation and persistence.** Shared dispatch does not need to understand `theta` or `E` payloads.
4. **Generic presentation products are useful without replacing physical results.** `PRRunResult` remains authoritative while `RunData` supplies selected fields and diagnostics to common views.
5. **Separate material applications can reuse GUI infrastructure.** A universal physics panel or material selector is not required.
6. **PR is a proof and design model, not a template that LC must immediately imitate.** LC package reorganization should occur only when a concrete migration benefit justifies it.

LCProp therefore has a credible in-tree material-composition architecture, but it does not yet have—and does not yet need—a third-party plugin framework. External discovery, versioned plugin contracts, and broad package reorganization would be premature.

## Known Limitations and Deferred Work

- The PR material integrator is still the explicit-Euler reference method; the stiff intensity-weighted diffusion term motivates a future improved integrator.
- The PR material derivative uses periodic centered finite differences in `x`; there is no apodization.
- Sampling and boundary problems are currently reported as preflight warnings rather than always rejected.
- The GUI does not yet expose the headless modal projections and analytic gain diagnostics used by the two-beam benchmark.
- The default selected longitudinal slice for 2-D PR state fields is not obvious from the field label.
- Screening solitons, image amplification, scattering, and fanning remain unvalidated application milestones.
- There is no public plugin API, dynamic discovery, universal material state, or universal material GUI contract.
- Existing LC codecs have not been reorganized into an LC-owned codec module; such a move can be considered later as compatibility work, not as a prerequisite for adding materials.

## Recommended Next Step

The architecture should pause at this proof point long enough for review. The next physics milestone should be a bounded PR validation case rather than additional plugin framework work. Image amplification is the strongest candidate because it exercises a realistic signal/background interaction, spatially varying gain, PR material evolution, and experimentally meaningful output diagnostics without requiring the noise model needed for fanning or the more delicate existence criteria needed for screening solitons.

Small GUI presentation corrections—especially making the selected `z` plane explicit and deciding whether severe grating undersampling should be rejected—can be handled independently and should not trigger architecture refactoring.

## Questions for Independent Review

1. Is the current explicit in-tree operation and codec composition an appropriate stopping point before any external plugin API is considered?
2. Does the default checkpoint registry's import of both LC and PR codecs create an undesirable dependency direction, or is it acceptable application composition at the current scale?
3. Should severe coherent-grating undersampling become a validation error while less severe aperture and resolution issues remain warnings?
4. Should selected longitudinal-plane semantics be added to generic `RunData` or remain state owned by the result workspace?
5. Is PR continuation appropriately kept in the PR GUI composition, or is there now enough repeated behavior to justify a small shared continuation service?
6. Are there hidden LC assumptions in the completed execution, persistence, product, or GUI paths that should be addressed before starting image amplification?
7. Does the evidence justify postponing LC package reorganization and a possible future `src/lcprop/lc/products.py` until a concrete LC migration is scheduled?

## Review Scope Requested

The requested review is architectural and diagnostic. It should evaluate dependency direction, ownership, compatibility risk, and the adequacy of the validation evidence. It should not propose a broad rewrite, a universal material hierarchy, dynamic plugin discovery, or changes to the validated optical seam unless a concrete defect is identified.

---

End of progress report.

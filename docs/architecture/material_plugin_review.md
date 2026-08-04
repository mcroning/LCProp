# Material Plugin Architecture Review

## Executive Summary

LCProp now contains the essential technical seam required to support more than one nonlinear material model, but it does not yet contain a complete material-plugin architecture. The distinction is important. The photorefractive (PR) implementation demonstrates that a second material can reuse the beam model, runtime grid, backend selection, launch construction, diffraction kernel, multichannel field representation, and split-step optical advancement without changing the liquid-crystal (LC) implementation. This is strong evidence that the optical core has reached a genuinely material-neutral boundary.

The key boundary is a prepared multiplicative optical response. A material model converts its own state into a shared `(Nx, Ny)` or per-channel `(Nch, Nx, Ny)` complex half-step screen, and `advance_prepared_response()` applies that screen around a common linear optical step. LC code derives the screen from director angle and refractive-index physics. PR code derives it from the normalized space-charge field `E`. Neither material model has to own FFT propagation. This is the most important prerequisite for a plugin architecture, and it is already implemented and validated by coherent two-beam coupling and finite Gaussian crossing benchmarks.

The rest of the application is less general. The standard request and result types describe LC material, bias, director initialization, and theta convergence. The shared workflow package constructs LC bias fields and director solvers. The runner protocol retains static and time-dependent LC compatibility methods, but local execution now also accepts explicitly registered material operations composed from a run callable and a result-to-`RunData` adapter. Shared persistence composes LC and PR checkpoint codecs through material/workflow descriptors while retaining explicit aliases for material-less legacy LC metadata. Product conversion combines a useful generic data model with LC-specific result adapters, while PR now owns its independent product adapter. The GUI has reusable workers, plotting views, workspace concepts, and beam/grid controls, but its experiment composition, physics panel, solver choices, run dispatch, continuation, and result handling remain LC-oriented.

Consequently, substantial restructuring of the numerical or optical engine is not required. Moderate, incremental restructuring is required at the application-composition boundary if PR and future materials are to participate in the same runners, persistence, diagnostics, and GUI. The codebase is naturally evolving toward an **in-tree material plugin model**, in which each material package owns its state, physics, solver options, workflow, optical-response preparation, validation, and material-specific presentation. It is not yet ready for independently installed or dynamically discovered third-party plugins.

The implemented composition proof confirms that the smallest practical boundary is not a universal `MaterialState` hierarchy or a plugin framework. LC and PR now expose explicit in-tree workflow operations composed from material identity, workflow identity, a run callable, and a result-to-`RunData` adapter. Existing LC entry points remain compatibility methods. Persistence functions and material-specific GUI panel composition remain deliberately outside this boundary.

A formal external plugin mechanism should therefore be postponed. The headless second-material path now covers execution, cancellation, progress reporting, and products without material-name conditionals in generic dispatch. Persistence and—if product requirements justify it—the shared GUI shell remain unproven boundaries. LCProp has a successful material-neutral propagation core and an implemented in-tree application-composition seam, but not yet a complete material-neutral application architecture.

## Background

LCProp began as an LC propagation application. Its central material state is the director angle `theta`; its material parameters describe LC elastic and optical properties; its bias model constructs a Freedericksz director profile; and its static and time-dependent workflows couple optical intensity to director relaxation. Requests, results, diagnostics, persistence, and much of the GUI were built around that model.

The material-neutral optical-response seam separated this LC state evolution from optical advancement. Instead of requiring the propagation engine to interpret a director field, a caller may prepare the material response as a complex half-step screen and pass it to `advance_prepared_response()`. The propagation engine retains responsibility for multichannel field advancement, diffraction, response-screen shape validation, and Strang ordering. LC-owned code retains responsibility for converting `theta` to the appropriate optical response.

The PR vertical slice then added a genuinely different nonlinear material. Its preserved material state is the normalized space-charge field `E`, evolved by a PR hopping-model equation. PR-owned code maps `E` to a refractive-index or phase response and uses the same prepared-response seam. The implementation has its own request and result dataclasses, material and solver specifications, source construction, evolution, workflow, and validation helpers.

Finally, the two-beam work validated more than package structure. It exercised coherent multichannel launch, kernel-derived beam-crossing geometry, Fourier-mode power measurement, analytic plane-wave gain, material-time convergence, and finite Gaussian crossing. It therefore established that the common optical engine can support behavior that is not a renamed LC calculation.

The addition of a second material raises the plugin question because the lower numerical layers already compose cleanly while the upper application layers do not. The issue is no longer whether optics can be shared. It is where material ownership should end, how a material workflow should enter shared execution and presentation services, and how much generalization is justified by the two implementations now present.

## Current Architecture

The implemented architecture has three distinguishable layers, although the package layout does not enforce them uniformly.

### Generic optics and numerical services

The optical field is represented as a channel stack shaped `(Nch, Nx, Ny)`. `BeamChannel` and `BeamStack` describe launch wavelength, power, waist, center, transverse phase gradients, phase, and coherence grouping. Launch construction creates Gaussian channel fields, while coherence-aware intensity routines combine channels according to their groups. The backend module selects NumPy or CuPy and establishes real and complex precision. `GridSpec`, `RuntimeGrid`, and `make_grid()` construct physical and Fourier coordinates used by both LC and PR.

The generic optical path consists of `linear_kernel()`, response-screen application, and `advance_prepared_response()`. The response screen may be spatially shared by all channels or specified independently per channel. Its shape is checked explicitly. Optical substep selection and kernel construction are also independent of a particular material equation, although their inputs include a caller-supplied estimate of maximum refractive-index change.

Generic execution primitives include cancellation and progress records. The product layer also contains generic geometry, field, curve, diagnostic, collection, and `RunData` containers. GUI workers can execute arbitrary callables, and most view widgets consume product metadata rather than material state directly.

### LC implementation

The `lc` package owns the bias calculation, LC coupling formulas, and conversion from director state to optical response. The theta solvers in `algorithms` implement LC director evolution. The standard workflows assemble LC material properties, bias, launch fields, theta relaxation, and optical advancement into static, time-dependent, soliton, and sweep operations.

LC concepts also remain in nominally shared locations. `LCMaterial`, `BiasSpec`, LC-derived quantities, LC request types, and theta-bearing result types live under `core`. `RuntimeGrid` includes normalized `du` and `dv` coordinates defined for the LC director equation. The launch model carries `theta_weight`. Persistence, standard runner methods, result conversion, diagnostics, and GUI experiment composition are built around the LC workflow family.

### PR implementation

The `pr` package is an additive, headless material implementation. It owns `PRMaterialSpec`, `PRSolverOptions`, `PRRunRequest`, `PRRunResult`, normalized source construction, spatial derivatives, timestep validation, hopping-model evolution, the `E`-to-response mapping, and its time-dependent workflow. It also owns crossing geometry, aperture checks, plane-wave modal projections, analytic gain comparison, propagation traces, and finite Gaussian coupling benchmarks.

The PR workflow directly uses the shared backend, grid, beam launch, power calculation, diffraction kernel, and `advance_prepared_response()`. It supports shared cancellation and progress records, owns a checkpoint and continuation contract with a material-owned disk codec, owns a `PRRunResult`-to-`RunData` adapter, and exposes an explicit operation that can be registered with `LocalRunner`. Its codec participates in shared persistence dispatch without exposing PR payload semantics to that layer. It does not use the standard LC workflows or GUI. This establishes a shared headless application path while preserving a material-owned physical workflow.

### Generic GUI components

The GUI contains reusable infrastructure, but it is not a material-neutral application as a whole. `WorkflowWorker` is callable-based and uses common cancellation and progress facilities. The workspace and plotting views largely consume `RunData`, `FieldData`, and metadata. Beam and grid controls represent concepts useful to both materials, subject to material-specific validation and defaults.

By contrast, the physics panel constructs `LCMaterial` and `BiasSpec`; the solver panel offers director-oriented strategies; and the main window owns LC-specific request construction, run dispatch, continuation, theta live state, checkpoint behavior, and result summaries. These are correctly useful for LC but are not currently selected through a material composition boundary.

## Analysis

The classifications below describe current behavior, not desired names or future package locations. “Mostly generic” means that a reusable core exists but is mixed with material assumptions or material-specific adapters.

| Directory | Classification | Basis |
|---|---|---|
| `src/lcprop/core/` | Mostly generic | Backend, beam-stack concepts, grid geometry, and execution records are reusable. The same directory also defines LC material and bias specifications, LC-derived quantities, theta-bearing requests and results, LC-normalized grid coordinates, and `BeamChannel.theta_weight`. |
| `src/lcprop/optics/` | Mostly generic | Launch, coherence-aware multichannel intensity, FFT kernels, prepared response screens, substeps, and propagation are reusable. LC compatibility entry points that accept `theta`, theta-weighted intensity handling, and theta metadata remain alongside the generic seam. |
| `src/lcprop/lc/` | Material-specific | Bias, LC coupling, director-to-index conversion, and LC optical-response preparation belong to the LC model. This is an appropriate material boundary. |
| `src/lcprop/pr/` | Material-specific | PR material/solver specifications, state evolution, source intensity, optical-response conversion, workflow, checkpoint codec, geometry, and coupling validation belong to the PR model. This is also an appropriate material boundary. |
| `src/lcprop/workflows/` | Material-specific | Runtime construction and all public workflows are director-centric. They import theta solvers, build LC bias and refractive-index quantities, initialize theta, and expose LC static, time-dependent, soliton, and sweep behavior. |
| `src/lcprop/algorithms/` | Mostly generic | Thomas solvers and portions of the Fourier-y and black-box loop machinery are general numerical infrastructure. `theta_cn.py`, `theta_picard.py`, and `theta_cn_zcoupled.py` are explicitly LC director algorithms. `static_relax.py` and `td_zmarch.py` accept callbacks and avoid material imports, but their state names, shapes, and contracts are theta-oriented. |
| `src/lcprop/runners/` | Mostly generic | `WorkflowOperation`, `RunnerResult`, opt-in registration, and local execution are material-neutral. The runner protocol and `LocalRunner` retain enumerated LC methods as compatibility entry points, while registered LC and PR operations use the shared path. |
| `src/lcprop/persistence/` | Mostly generic | The descriptor, registry, save dispatch, identified load dispatch, and explicit legacy aliases are material-neutral composition infrastructure. The static and time-dependent payload implementations in this directory remain LC-specific compatibility codecs and preserve their historical material-less formats. PR payload encoding remains under `src/lcprop/pr/`. |
| `src/lcprop/products/` | Mostly generic | Geometry, fields, curves, diagnostics, collections, and `RunData` are material-neutral containers. LC result and live-state adapters remain here, while PR owns a separate adapter under `src/lcprop/pr/`. Diagnostic functions still mix generic power/centroid/width metrics with theta metrics and a theta static residual. Static torque balance is LC-specific. |
| `src/lcprop/gui/` | Mostly generic | Worker, workspace, retained presentation concepts, and many views are reusable. Main-window composition, the physics and solver panels, workflow selection, request construction, continuation, checkpoint handling, and several result paths assume LC and theta. |
| `src/lcprop/adapters/` | Mostly generic | The LaunchPlane adapter preserves generic beam properties and phase gradients, but its public conversion includes the LC-specific `theta_weight` field. |
| `tests/` | Mostly generic | Tests cover shared optics, backends, grids, products, and GUI infrastructure as well as separate LC and PR behavior. Material-operation tests now exercise descriptor validation, explicit registration, legacy compatibility, LC and PR execution, cancellation, progress, and product conversion through one shared path. |

### Optics

Optics is the strongest reusable subsystem. The propagation engine operates on complex fields and prepared responses, not on `theta` or `E`, when called through the new seam. Multichannel screen shapes are explicit, and the split-step ordering is owned by the optical engine. PR's use of this path is direct evidence of reuse.

The remaining LC assumptions are compatibility features rather than a fundamental limitation of the kernel. `advance_slice()` and related helpers still convert theta-oriented inputs; launch results expose theta weights; and some intensity helpers describe director driving. These mixed responsibilities can confuse ownership, but they do not prevent a new material from using the lower-level interface.

### Workflows

The standard workflows are implicitly director-centric. `workflows/runtime.py` imports the theta algorithms and LC bias/response calculations, constructs theta initial conditions, creates Picard relaxation and z-coupled theta steps, and exposes optics callbacks in terms of theta slices. Static, time-dependent, soliton, and sweep modules build on those assumptions.

PR demonstrates the current extension method: create a separate material-owned workflow that calls the common optical layer, implements the shared execution lifecycle, and supplies a material-owned product adapter. LC and PR workflow implementations still coexist rather than sharing a material solver contract, while their operation descriptors provide a common headless application contract. Persistence and GUI paths remain outside that composition.

### Execution and runners

`core/execution.py` is generic. Cancellation and progress reporting do not require an LC state. PR now uses those records directly and observes cancellation only at completed material-time boundaries. The runner protocol still enumerates `run_static`, `continue_static`, and `run_timedependent` compatibility methods with LC-defined request and checkpoint expectations.

The shared dispatch problem has been addressed additively. `WorkflowOperation` contains a material identifier, workflow identifier, run callable, and product adapter. `LocalRunner` accepts explicit operation registration and returns both the concrete physical result and shared `RunData`. LC and PR packages compose their own operation descriptors. There is no global registry, discovery mechanism, inheritance hierarchy, or change to legacy LC method behavior.

### Requests and results

There is no single material-neutral run request or result. The types in `core/requests.py` are LC application contracts despite their location: they require `LCMaterial`, `BiasSpec`, theta solver settings, and optional initial theta. The types in `core/results.py` store theta fields, bias fields, theta residuals, and LC workflow records.

PR correctly preserves its independent state by defining `PRRunRequest` and `PRRunResult` in `lcprop.pr.specs`. `E_initial` and `E_final` remain first-class state. This separation is preferable to forcing unlike state into a weak universal record. Shared operation dispatch and `RunData` presentation now compose those concrete types without creating a universal union of LC and PR fields.

### Persistence

Shared disk persistence uses a small `CheckpointCodec` descriptor containing material ID, workflow ID, checkpoint type, and save/load callables. `CheckpointCodecRegistry` rejects duplicate identities and checkpoint types, dispatches saves by checkpoint type, and dispatches identified loads by material/workflow key. Exact save-type matches take precedence, while an unambiguous subclass remains compatible with the previous LC dispatch behavior. The registry verifies that a loader returns the checkpoint type declared by its descriptor.

PR owns a versioned disk codec for its validated `E` checkpoint and exact continuation path. It reuses the three-file run-directory convention while explicitly identifying the `pr` material, `pr_timedependent` workflow, and PR-owned schema in both JSON documents. Its array payload preserves `E_initial`, `E_current`, and `A0`. The shared registry delegates to this codec without interpreting the payload.

The existing LC static and time-dependent functions are registered through descriptors without changing their APIs or disk schemas. Because historical LC provenance contains a workflow but no material identifier, the default registry has explicit `static -> lc/static` and `timedependent -> lc/timedependent` legacy aliases. New material codecs do not inherit that fallback. Unknown and cross-material identities are rejected before payload decoding.

### Diagnostics and products

Power, centroid, RMS width, and intensity metrics are generic. `Geometry`, `FieldData`, `CurveData`, `DiagnosticData`, and `RunData` provide a useful neutral presentation model. These are suitable shared outputs for material workflows.

The central conversion layer remains LC-specific. `from_static_result()`, `from_timedependent_result()`, live-state adapters, soliton adapters, and the top-level conversion dispatch know the existing LC result family. PR now owns `pr_result_to_run_data()`, which represents only fields and diagnostics present in `PRRunResult`; benchmark-only modal, gain, and crossing data are not inferred. The diagnostics module still places theta metrics and theta residual evaluation next to generic optical metrics. PR coupling diagnostics remain material-owned and do not yet have separate shared-product adapters.

### GUI

The GUI already separates some reusable presentation from LC experiment setup. A material plugin could plausibly reuse workers, beam/grid controls, the workspace, fields, curves, and generic diagnostics. It should not be expected to reuse LC voltage, boundary-angle, director-solver, or theta-convergence controls.

The present main window is the limiting point because it directly assembles the LC panels and branches among known LC workflows and result forms. Adding PR today would require either more PR-specific branches in the main window or a new PR-specific window. The former would accumulate conditionals; the latter would duplicate shared navigation and presentation. A small material GUI composition boundary is justified when PR receives an interactive workflow, but a full GUI plugin API is not yet supported by implementation evidence.

### Algorithms

The algorithms directory should not be treated as a plugin boundary. Its Thomas solvers are genuinely generic numerical infrastructure. `fft_y.py` implements reusable periodic-y operations but documents and serves the theta solvers. The CN, Picard, and z-coupled modules are correctly explicit theta/director algorithms and should remain so. `static_relax.py` and `td_zmarch.py` are callback-driven orchestration kernels with few direct physics dependencies, but their public vocabulary and state contracts remain theta-based.

Material plugins should own or call the algorithms appropriate to their state. There is no benefit in renaming director solvers to appear generic. Any future extraction of orchestration should be based on demonstrated common control flow between LC and PR, not on superficial similarity between `theta` and `E` arrays.

## Existing Abstractions

### Abstractions that already support multiple materials

- `BackendSpec`, backend selection, array conversion, and precision handling support NumPy and CuPy without knowledge of material physics.

- `GridSpec`, most of `RuntimeGrid`, and physical/Fourier coordinate construction are shared by LC and PR.

- `BeamChannel` and `BeamStack` provide common optical launch data, transverse phase gradients, coherence groups, and channel validation.

- Launch construction and coherence-aware multichannel intensity provide a shared optical input representation.

- `linear_kernel()` and the shared field-stack convention support material-independent diffraction.

- Prepared optical responses are a concrete interoperability format: a material supplies either one complex spatial screen or one screen per channel.

- `advance_prepared_response()` supplies the material-neutral optical advancement contract and preserves Strang response/diffraction ordering.

- Optical substep planning accepts material estimates as data rather than implementing a particular material law.

- `CancellationToken` and `RunProgress` provide workflow-independent execution signals.

- `Geometry`, `FieldData`, `CurveData`, `DiagnosticData`, and `RunData` provide material-neutral presentation primitives.

- Callable GUI workers and metadata-driven plotting views can display results without interpreting a material PDE.

### Abstractions that still assume LC

- `BeamChannel.theta_weight` and launch-result theta weights place a director-driving parameter in the otherwise generic beam model.

- `RuntimeGrid.du` and `RuntimeGrid.dv` encode the LC director normalization, and `from_cell()` names LC cell geometry.

- `LCMaterial`, `BiasSpec`, LC-derived quantities, and theta-oriented request/result classes are located in `core`, implying broader generality than their contracts provide.

- Optics retains theta-oriented compatibility functions alongside the prepared-response API.

- Shared workflow runtime construction is specifically an LC director runtime.

- The runner protocol retains LC-specific compatibility methods even though the registered-operation path is material-neutral.

- Persistence dispatches by two LC workflow names and concrete checkpoint types.

- Product conversion and several diagnostics interpret theta-specific result fields.

- GUI experiment composition assumes LC physics, director solvers, theta continuation, and LC checkpoint behavior.

These assumptions are not all defects. Several are valid LC responsibilities whose current package location or integration path makes them appear shared. The architectural task is to make ownership explicit at composition boundaries, not to erase material vocabulary.

## Candidate Plugin Boundary

The smallest practical plugin boundary is an explicit, in-tree composition boundary around a material workflow. It should build on the interfaces already exercised by LC and PR rather than introduce a new inheritance hierarchy.

### Shared platform responsibilities

The shared platform should continue to own:

- backend and precision selection;
- physical and Fourier grid construction;
- beam/channel definitions that are truly optical;
- launch-field construction and coherence grouping;
- diffraction kernels and optical substep mechanics;
- validation and application of prepared response screens;
- cancellation, progress, and runner transport;
- generic result products and plotting views;
- a common persistence envelope and provenance conventions, if persistence is requested by a material workflow;
- application navigation and shared beam, grid, execution, and results UI where those concepts are genuinely common.

### Material plugin responsibilities

An LC or PR material package should own:

- material and solver specifications;
- the preserved material state and its initial condition;
- material PDEs, boundary conditions, stability rules, and numerical evolution;
- the definition and normalization of optical driving intensity;
- conversion from material state to a prepared optical half-step response;
- material-specific convergence and continuation rules;
- workflow assembly and material-specific validation;
- material diagnostics and benchmark logic;
- conversion of its result into shared `RunData` products;
- serialization of its request, state, and continuation data when persistence is supported.

No universal `MaterialState` base class is warranted. `theta` and `E` have different dimensions, boundary conditions, time semantics, and validation requirements. Their concrete dataclasses and arrays should remain material-owned. The useful shared contract is behavioral and compositional: a workflow can run with standard execution signals, can advance optics using a prepared response, and can expose products through a shared result adapter.

### Material GUI responsibilities

A material GUI should own controls and explanations for its material parameters, state initialization, solver choices, convergence criteria, validation warnings, and material-specific diagnostics. The LC GUI should continue to speak explicitly about voltage, director angle, theta solvers, and LC boundaries. A future PR GUI should speak explicitly about applied field, background or dark intensity, normalized material time, PR stability, and PR benchmarks.

The shared GUI should own the application shell, material/workflow selection, reusable beam and grid controls, execution status, run history, and generic field/curve presentation. Material panels should be composed into that shell rather than forced into a lowest-common-denominator physics form.

### Minimal registration data

The current code supports a small explicit descriptor or mapping composed of ordinary callables and metadata. At minimum, an executable material operation needs:

- a stable material/workflow identifier;
- a workflow callable accepting its own request and standard cancellation/progress hooks;
- a callable that converts its result to `RunData`;
- optional save/load callables for material-owned persistence;
- optional GUI factories only when an interactive workflow exists.

This need not be an abstract base class, dynamic entry-point system, or independently versioned plugin SDK. An explicit in-tree registry would be sufficient to remove material-name branching from shared dispatch while preserving direct functions and dataclasses.

## Migration Strategy

Migration should proceed by making the existing second material use one shared service at a time. Each step should preserve direct LC and PR headless entry points until the shared route has equivalent tests.

| Order | Change | Change type | Risk | Recommendation |
|---:|---|---|---|---|
| 1 | Define and document stable material/workflow identifiers and the ownership of request, state, result, and prepared response. | Architecture cleanup | Low | Do now. This clarifies boundaries without moving code or changing physics. |
| 2 | Add a PR result-to-`RunData` adapter using the existing generic product containers, and test LC and PR product conversion independently. | Additive API | Low | Do next. It validates the presentation boundary without requiring a GUI. |
| 3 | Allow the local execution layer to invoke an explicitly registered workflow callable and result adapter, while retaining existing `LocalRunner` methods as compatibility entry points. | API change | Medium | Do after product adaptation. Exercise cancellation, progress, errors, and reproducibility with both materials. |
| 4 | Separate generic optical diagnostics from theta-specific diagnostics by ownership and dispatch, without changing their mathematics. | Architecture cleanup | Low to medium | Do when PR products are integrated. Avoid a universal diagnostics interface until required. |
| 5 | Introduce shared persistence composition using material/workflow identity, with LC- and PR-owned payload codecs. | API and persistence change | Medium | Implemented additively. PR supplies identified metadata; existing LC formats participate through unchanged codecs and explicit legacy aliases. |
| 6 | Compose material-specific setup panels into the shared GUI shell and route execution through the workflow registration path. | GUI change | High | Postpone until a minimal PR GUI is a product requirement. First stabilize headless execution and products. |
| 7 | Relocate LC-only specifications, derived quantities, request/result models, and compatibility helpers from nominally generic packages. | Package reorganization | High | Postpone. Use compatibility imports and deprecation policy only after the target boundaries have been exercised. Package movement is not required to prove the architecture. |
| 8 | Remove or replace LC fields embedded in shared models, including `theta_weight` and LC-normalized grid coordinates. | API change | High | Postpone until material-owned adapters can preserve all existing LC semantics and saved-run compatibility. |
| 9 | Add dynamic discovery or support independently installed plugins. | Framework and packaging change | High | Do not do now. Reassess only after at least two materials use the same end-to-end application contract. |

### Implementation status as of 2026-08-04

Migration steps 1 through 3 and the composition portion of step 5 have been implemented for the headless path. Stable LC and PR material/workflow identifiers are carried by explicit operation and checkpoint-codec descriptors. PR owns its result-to-`RunData` adapter and supports shared cancellation and progress reporting. `LocalRunner` can execute explicitly registered LC and PR operations and returns both the concrete material result and its presentation product. Shared checkpoint dispatch accepts both materials while existing LC runner and direct persistence functions remain unchanged compatibility entry points.

The implementation intentionally leaves step 4 unchanged. PR has a material-owned checkpoint preserving `E_initial`, accepted `E_current`, and `A0`, with exact cumulative continuation and a PR-owned versioned disk codec. Shared persistence now composes that codec with adapters around the unchanged LC codecs, and legacy LC loading is explicit rather than inferred as a generic material rule. No diagnostics have been moved, no GUI consumes the operation registry, and no package reorganization or external discovery mechanism has been introduced.

### Architecture cleanup

Architecture cleanup should first clarify which callable owns each conversion and diagnostic. It can be additive: PR gains a product adapter; shared dispatch accepts registered operations; LC-specific diagnostics remain explicit. This improves boundaries without changing import paths or public data structures prematurely.

### Package reorganization

Package reorganization is not the first dependency. Moving `LCMaterial`, `BiasSpec`, LC requests/results, or director helpers would create widespread import and persistence churn while contributing little new evidence about plugin viability. Such moves should follow a stable composition boundary and include compatibility shims and saved-data migration plans.

### API changes

The most valuable API change is workflow dispatch by explicit operation metadata and callables rather than by an ever-growing runner method list. Product and persistence adapters can follow the same composition pattern. Changes to `BeamChannel`, `RuntimeGrid`, or existing LC request/result types are more disruptive and should wait until their replacement semantics are proven.

### GUI changes

The GUI should be the last major integration step. Headless PR workflows and shared `RunData` should first establish the material boundary. A GUI composition mechanism can then be sized to real PR controls instead of guessed in advance. Shared controls should remain shared only when validation, units, defaults, and meaning are genuinely common.

## Risks

### Premature generalization

The largest technical risk is replacing clear LC and PR code with abstractions that represent neither material well. Director relaxation and the PR hopping model do not share enough state semantics to justify a universal solver or material-state class. Generalization should target application composition and optical interoperability, not the material equations.

### Duplicated GUI behavior

Independent material panels can duplicate layout, validation messaging, run controls, and help text. Conversely, one universal physics panel would produce ambiguous terminology and conditional complexity. The practical balance is a shared shell and presentation layer with material-owned setup panels and small reusable control groups.

### Testing matrix growth

Each material adds CPU/GPU, precision, workflow, persistence, cancellation, product, and GUI combinations. A plugin architecture also needs contract tests for screen shapes, optical power behavior, progress, result adaptation, schema identity, and error reporting. Material benchmarks must remain material-owned; a generic contract suite cannot replace analytic and experimental validation.

### Persistence compatibility

Plugin-owned state and schema evolution can make old runs unreadable if identity and versioning are weak. LC theta checkpoints and future PR `E` checkpoints require different payloads and continuation rules. Introducing a generic envelope without migration tests could obscure rather than solve compatibility problems.

### Maintenance and ownership

Shared-platform changes can break every material, while a material-local change may affect only one benchmark family. The project will need explicit ownership rules and tests at the prepared-response, workflow-dispatch, product, and persistence boundaries. A plugin contract also becomes documentation that must be versioned and maintained.

### User experience fragmentation

Materials have different parameter vocabularies, stability limits, boundary assumptions, and meaningful workflows. Separate panels and diagnostics can make the application feel like unrelated tools unless navigation, units, run lifecycle, products, provenance, and documentation remain consistent. Hiding those differences would be worse: it could permit physically invalid configurations.

### Lowest-common-denominator results

A generic result type containing every possible material field would become sparse and ambiguous. Conversely, exposing only optical intensity would discard essential state such as `theta` or `E`. The existing `RunData` product model offers a safer boundary: concrete material results remain rich, while selected fields, curves, and diagnostics are adapted for shared presentation.

### Dynamic-plugin complexity

External discovery introduces version negotiation, dependency isolation, backend compatibility, security, and support questions that the current in-tree implementations do not answer. Treating internal packages as plugins first provides most architectural benefits without committing to that operational burden.

## Conclusions

A material-plugin architecture is justified at the level of **explicit in-tree composition**. LCProp has two fundamentally different material models, and PR has validated that they can share the same optical propagation engine through prepared response screens. Preserving separate material state, equations, workflows, and benchmarks while sharing optics and application services is now a concrete need rather than speculative extensibility.

LCProp should not implement a formal external plugin framework now. The current architecture has a mature optical boundary but incomplete execution, product, persistence, and GUI boundaries. Dynamic discovery, universal state interfaces, broad package moves, and generalized material solvers would create compatibility and maintenance costs before the necessary contracts have been exercised.

LC and PR now use shared headless execution, product, and persistence-composition paths without payload-specific branching in generic dispatch. PR's concrete checkpoint and continuation requirement is validated in memory and through a material-identified, PR-owned disk codec. Existing LC disk formats remain readable through explicit compatibility aliases. A material-composed interactive path, when a concrete PR GUI is required, should precede promotion of these internal contracts into a public third-party plugin API.

The present architecture is therefore best described as **material-neutral at the optical layer, package-separated at the material layer, and explicitly plugin-composed for headless execution, products, and persistence**. GUI composition remains material-specific. The next work should validate additional boundaries only when concrete PR requirements justify them, rather than redesign the validated numerical core.

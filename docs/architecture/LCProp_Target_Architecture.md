# LCProp Peer-Material Target Architecture

**Status:** Accepted architecture decision record

**Decision date:** 2026-08-11

**Scope:** LCProp package ownership, dependency direction, and migration policy

> This document defines the target architecture for LCProp. Subsequent
> architecture, development, commissioning, and research tasks should conform
> to it unless superseded by a later explicitly approved architecture revision.

## 1. Purpose

LCProp began as a liquid-crystal propagation application. The addition of an
independent photorefractive implementation demonstrated that its optical
engine, execution seams, presentation records, and selected GUI components can
support fundamentally different nonlinear materials. This ADR fixes the
architectural interpretation of that result and defines a conservative path
from the present repository to stable peer-material ownership.

The purpose is not to create a plugin framework. It is to make ownership and
dependency direction explicit, preserve validated physics, and prevent future
migration work from drifting into opportunistic redesign.

## 2. Core Decision

`lcprop.lc` and `lcprop.pr` are peer material/nonlinearity packages. Shared
packages provide optical, numerical, execution, presentation, and reusable GUI
infrastructure. Each material package owns its physical state, equations,
source construction, solvers, requests, results, workflows, product adapters,
persistence codecs, and material-specific application controls.

No `materials/` parent package, material base class, dynamic plugin discovery,
or universal request/result hierarchy is required.

The existing flat peer structure under `lcprop/lc/` and `lcprop/pr/` is an
acceptable target. Deeper package nesting should be introduced only when the
number or cohesion of real modules demonstrates a need.

## 3. System Model

```text
                         application composition root
                                      |
                         registered WorkflowOperation
                                      |
             +------------------------+------------------------+
             |                                                 |
      LC-owned request                                  PR-owned request
             |                                                 |
       LC workflow                                        PR workflow
             |                                                 |
             +--------> shared optical channel fields <--------+
             |                    |                            |
       LC source builder    shared propagation          PR source builder
             |                    |                            |
       LC material solve <-> prepared optical response <-> PR material solve
             |                                                 |
       LC-owned result                                  PR-owned result
             |                                                 |
       LC product adapter                               PR product adapter
             +------------------------+------------------------+
                                      |
                               shared RunData
                                      |
                        shared views / export infrastructure
```

LC and PR may depend on shared optics and genuinely generic numerics. Reusable
shared infrastructure must not depend on LC, PR, or another material's
physics.

## 4. Dependency-Direction Rules

The normal dependency direction is:

```text
material applications -> material packages -> shared infrastructure
```

The following rules are normative:

1. Material packages may import shared infrastructure.
2. Reusable shared infrastructure must not import material physics.
3. One material package must not depend on another material package.
4. Explicit application composition roots may import material-owned
   operations, codecs, product adapters, and GUIs solely to assemble an
   application.
5. Compatibility shims may temporarily import canonical material modules to
   preserve historical import paths. Such shims are migration artifacts, not
   shared ownership.
6. Shared records must not inspect material result internals. Material-owned
   adapters perform that conversion.

Examples of valid composition roots include default runner registration,
persistence registry assembly, GUI application startup, and workflow-operation
registration. The exception does not authorize material physics inside the
runner, registry, or reusable GUI implementation.

Canonical ownership must be objectively verifiable through module location,
`__module__` provenance, legacy/canonical object identity, and
dependency-direction tests.

## 5. Ownership Matrix

| Concern | Shared platform | Material package |
|---|---|---|
| Optical channel specification | Wavelength, power, waist, position, transverse phase gradient, phase, coherence, future polarization | Interpretation of those channels as a material source |
| Launch and propagation | Channel fields, optical powers, wavelengths, coherence, propagation kernels | Material response preparation and coupling sequence |
| Grid | Physical coordinates, apertures, spacing, backend arrays | Material-normalized coordinates and material boundary interpretation |
| Numerical methods | Only demonstrably equation-independent algorithms | PDEs, residuals, operator assembly, closures, material boundary conditions, physics-specific solver choices |
| Requests and results | Genuinely shared composition records only | Workflow-specific request and result types |
| Execution | `WorkflowOperation`, runner protocol, progress and cancellation infrastructure | Registered workflow callables and product adapters |
| Presentation | `RunData`, field, curve, and diagnostic records; reusable views | Conversion from material result and material-specific diagnostics |
| Persistence | Registry/dispatch and reusable storage utilities | Checkpoint state, fingerprints, schemas, and codecs |
| GUI | Reusable optical launch, physical grid, worker, progress, view, workspace, and export components | Material application and physics/workflow controls |

## 6. Shared Optical Contract

Shared beam and launch models contain only optical field specifications and
optical metadata.

Permitted shared optical properties include:

- wavelength and future spectral or temporal description;
- optical power;
- transverse shape and waist;
- launch position;
- phase and transverse phase gradients;
- coherence grouping or future mutual-coherence information;
- future polarization, vector-mode, or propagation-direction information when
  required by a concrete optical model.

Shared beam and launch models must not contain:

- `theta_weight`;
- beam ratio as a material-coupling concept;
- material coupling coefficients;
- constitutive parameters;
- nonlinear weighting factors;
- absorption, heating, photoconductive, or director-driving efficiencies;
- any other material-specific metadata.

Every material package constructs its own driving source from propagated
optical channels. Shared optics may provide material-neutral operations such
as per-channel intensity, coherent group fields, total optical intensity, and
physical channel powers. It must not choose a material source formula.

`theta_weight` and `LaunchResult.theta_weights` are scheduled for complete
removal. They must not be relocated, renamed, or generalized into a universal
per-channel nonlinear weight. Compatibility handling must not silently discard
a supported historical nonunit value; it must explicitly translate the value
into a temporary LC-owned compatibility representation or reject it clearly.

## 7. Material-to-Optics Contract

The validated contract today is a material-prepared multiplicative complex
response consumed by `advance_prepared_response()` for split-step propagation.
That seam is material-neutral for the supported multiplicative response and is
not a declaration that all future constitutive responses are local phase
screens.

If a concrete future model requires a nonlocal or operator-valued optical
response, it should introduce an explicit propagation adapter justified by
that model. LCProp must not create a speculative universal response hierarchy
in anticipation of unknown requirements.

The split-step angular-spectrum implementation is one propagation backend, not
the architecture itself. Alternative scalar BPM, vector, coupled-mode, or
nonlocal propagation methods remain future decisions.

## 8. Material Source Construction

Source construction is material-owned because the same channel fields may
drive different equations through different observables. A material may use:

- coherent or incoherent intensity;
- selected interference terms;
- dark or uniform background intensity;
- scattering contributions;
- polarization-dependent observables;
- deposited rather than propagated power;
- material-dependent spectral or electro-optic coupling.

LC source construction owns the conversion from channel fields to the director
driving intensity. PR source construction owns dark/background intensity,
coherent interference, scattering, and other PR-driving quantities. There is
no universal source formula.

## 9. Numerical-Layer Contract

Generic numerics are defined by equation independence, not by directory
location.

Potentially shared numerical infrastructure includes backend-independent
array helpers, FFT differentiation, elliptic inversion, tridiagonal and sparse
linear algebra, interpolation, residual norms, line-search machinery, and
time-integration infrastructure only when their interfaces contain no material
state or equation assumptions.

Material packages own equations, state variables, residual definitions,
operator assembly, nonlinear closures, boundary conditions, and solver choices
tied to material physics. A reusable Newton or line-search kernel does not make
the PR residual generic; a reusable tridiagonal kernel does not make an LC
director equation generic.

The current modules `theta_cn.py`, `theta_picard.py`,
`theta_cn_zcoupled.py`, `static_relax.py`, and `td_zmarch.py` are semantically
LC-owned even though they are physically located under `lcprop.algorithms`.
Their theta/director terminology is correct. Mechanical relocation, if
approved, must be a separate behavior-preserving task; they must not be renamed
into generic material solvers.

No universal `MaterialSolver` abstraction is required.

## 10. Request and Result Model

Requests and results are material-owned and workflow-specific. Every concrete
request and result has exactly one material owner and one workflow meaning.
Exact class names are not normative, and package qualification may provide
sufficient material identity.

Separate static, time-dependent, continuation, soliton, sweep, streaming, or
research request/result types are appropriate when their semantics differ.
Workflow-specific PR types are not themselves an architectural problem. A
migration gap exists only where ownership is inconsistent, naming obscures
semantics, or compatibility types remain in a shared-looking namespace.

Naming symmetry follows semantic equivalence rather than aesthetics.
`PRMaterialSpec` should not be renamed to `PRMaterial` merely to resemble
`LCMaterial`; first establish that the two names describe the same conceptual
role. Composition is preferred over inheritance, and LCProp must not introduce
a giant universal `RunRequest` or universal material result dataclass.

## 11. Presentation Model

Shared presentation records include `RunData`, geometry, field, curve, and
diagnostic records and their collections. These records describe how data may
be presented without encoding material state.

Each material package owns conversion from its concrete result to shared
presentation data. Material-specific diagnostics, including theta/director
diagnostics and PR space-charge diagnostics, remain material-owned. Shared GUI
views and export code consume presentation records rather than material result
internals.

## 12. Persistence Model

Persistence serves three distinct purposes and must not collapse them into one
universal schema:

1. **Checkpoint and restart:** material-owned state, compatibility validation,
   fingerprint, schema, and codec; shared registry and dispatch.
2. **Scientific result archive:** material-specific state, provenance, and
   scientifically necessary arrays and metadata.
3. **Presentation and export:** figures, curves, movies, tables, and selected
   derived data produced through shared export infrastructure from
   material-provided presentation records.

Checkpoint compatibility is a material concern. Shared persistence composition
may register codecs but must not interpret theta, space-charge fields, or other
material states.

## 13. GUI Model

The target is shared GUI components plus material-owned applications, not a
monolithic automatically generated material GUI.

Shared components may own optical launch controls, physical grid controls,
workers, progress, reusable field/image/curve views, results workspaces, and
generic export shells. Material applications own physics, solver, continuation,
diagnostic, and workflow controls. The current standalone PR GUI validates this
model. The existing top-level GUI remains predominantly the LC application and
requires an ownership cleanup rather than a framework rewrite.

Most users may work entirely within one material application. A global material
selector is not required.

## 14. Execution and Composition

The existing callable/dataclass `WorkflowOperation` seam is the canonical
execution-composition mechanism:

```text
material-owned request
        -> registered WorkflowOperation
        -> material-owned workflow
        -> material-owned result
        -> material-owned product adapter
        -> shared RunData
```

Shared runners dispatch by registered material and workflow identifiers. They
must not branch on LC or PR physics. Explicit application composition roots may
register built-in operations and codecs. Dynamic discovery, Python entry-point
plugins, and an SDK are not required.

## 15. Current Repository Mapping

The ownership mapping below is checked against public baseline
`cb17b6be8c0621a57e13e0a1410f7d7201f5b424` (2026-11-03). The original August
migration plan is retained in section 18 as historical context, not pending work.

| Concern | Canonical location | Historical compatibility location |
|---|---|---|
| LC material/context specifications | `lcprop.lc.specs` | `lcprop.core.context` |
| LC requests | `lcprop.lc.requests` | `lcprop.core.requests`, selected workflow modules |
| LC results | `lcprop.lc.results` | `lcprop.core.results`, selected workflow modules |
| LC product adapters | `lcprop.lc.products` | `lcprop.products.data_model` |
| LC diagnostics | `lcprop.lc.diagnostics` | `lcprop.products.diagnostics` |
| LC torque-balance products | `lcprop.lc.static_torque_balance` | `lcprop.products.static_torque_balance` |
| LC workflows and runtime composition | `lcprop.lc.workflows.*` | `lcprop.workflows.*` |
| LC operation registration | `lcprop.lc.operations` | composed by runners/applications |
| LC algorithms | `lcprop.lc.theta_cn`, `theta_cn_zcoupled`, `theta_picard`, `static_relax`, `td_zmarch` | matching `lcprop.algorithms.*` module aliases |
| LC checkpoint implementation | `lcprop.lc.persistence.static`, `timedependent` | `lcprop.persistence.static`, `timedependent` module aliases |
| LC application | `lcprop.lc.gui.*` | selected `lcprop.gui.*` launchers/module aliases; reusable GUI framework remains shared |

PR already owns its material specifications, source construction, evolution,
static and time-dependent workflows, prepared optical response, product
adapter, operation registration, checkpoints/codecs, standalone GUI, and
validated reference implementation under `lcprop.pr`. Its workflow-specific
types are legitimate; remaining naming or ownership inconsistencies should be
reviewed individually rather than normalized mechanically.

Shared infrastructure currently includes beam/grid/backend records, optical
launch and propagation, `WorkflowOperation`, runner protocols, presentation
records, reusable views/workspace components, and persistence composition.

The original ADR described a then-uncommitted Stage 1-4 implementation. At the
baseline above, the canonical implementations and aliases are committed; that
old pre-commit gate is not a current instruction. This source inspection does
not assert complete scientific, platform, or remote-workflow qualification.

## 16. Compatibility-Shim Policy

Compatibility shims preserve legacy imports while canonical ownership moves.
They must:

- re-export or alias the canonical object rather than duplicate behavior;
- preserve object identity where feasible;
- remain free of physics implementations;
- be covered by legacy/canonical identity and import-order tests;
- be documented as transitional exceptions to dependency direction;
- not become the import path used by new material-owned implementation code.

Shims may be retained intentionally for public API stability. Retirement must
be a controlled compatibility decision after canonical paths are proven; it is
not a prerequisite for architectural completion.

## 17. Implemented ownership and retained compatibility

The original migration-gap list is preserved in the historical version linked
below. At the recorded baseline:

- Shared [BeamChannel](../../src/lcprop/core/beams.py),
  [launch records](../../src/lcprop/optics/launch.py) and
  [split-step propagation](../../src/lcprop/optics/splitstep.py) have no
  `theta_weight`/`theta_weights` source-coupling field. LC constructs its source
  in [lc.source](../../src/lcprop/lc/source.py) and response in
  [lc.propagation](../../src/lcprop/lc/propagation.py).
- [RuntimeGrid](../../src/lcprop/core/grid.py) stores physical spacing; LC's
  normalized spacing is owned by [lc.normalization](../../src/lcprop/lc/normalization.py).
  The lazy legacy `from_cell` export remains an explicit compatibility exception.
- LC algorithms and checkpoint implementations live in the canonical locations
  in section 15. Historical modules alias those implementations; they are not
  alternate places to add physics.
- [LocalRunner](../../src/lcprop/runners/local.py) dispatches explicit registered
  operations. Its convenience methods such as `run_static` still import LC
  operations for compatibility; removal is not required or authorized here.
- The LC application lives under `lcprop.lc.gui`; shared views/workspace and
  selected legacy LC launchers/aliases coexist under `lcprop.gui`.

Identity/provenance obligations are represented by
[LC facade tests](../../tests/test_lc_ownership_facades.py),
[algorithm ownership tests](../../tests/test_lc_algorithm_ownership.py) and
[GUI ownership tests](../../tests/test_gui_ownership.py). Their presence is not a
claim of a fresh test run. Compatibility exports in core, products, workflows,
persistence and GUI remain deliberate exceptions, not generic material APIs.

PR has both production integration and isolated reference paths. Do not apply
the original assertion that all transverse work is reference-only to current
requests; use the [PR model contracts](../science/pr_model_contracts.md) and
workflow-specific support evidence. This update does not claim every migration
or commissioning task is complete. Future alias retirement, new shared
abstractions, or automatic remote environment provisioning needs its own scope;
no current source relocation follows from this ADR.

## 18. Historical ordered migration plan

The plan below records the original sequencing, including work subsequently
implemented. It is retained for provenance, not an executable backlog. Use
sections 15 and 17 for the checked implementation status. The
[original gap list and plan](https://github.com/mcroning/LCProp/blob/cb17b6be8c0621a57e13e0a1410f7d7201f5b424/docs/architecture/LCProp_Target_Architecture.md#17-present-migration-gaps)
remain available at an immutable revision.

Migration proceeds conservatively, one seam per task whenever practical:

### Stabilization gate

1. Approve and commit this ADR as a documentation-only architecture decision.
2. Run a bounded pre-commit review of the Stage 1-4 ownership implementation.
3. Commit only the reviewed Stage 1-4 architecture files and tests.
4. Use that implementation SHA as the baseline before Stage 5.

### Post-stabilization stages

5. Inventory historical nonunit `theta_weight` usage and define explicit
   compatibility behavior.
6. Establish LC-owned source construction, migrate LC workflows to it, and
   completely remove `theta_weight` from shared beams, launch, optics, and
   adapters without introducing an equivalent generic nonlinear weight.
7. Separate LC compatibility helpers from canonical shared split-step optics.
8. Move LC-normalized grid quantities out of shared `RuntimeGrid`.
9. Mechanically relocate semantically LC-owned algorithms while preserving
   their names, signatures, and behavior; separately extract only numerics
   proven equation-independent by LC and PR or another concrete use.
10. Complete LC persistence implementation ownership while retaining shared
    codec registry/dispatch.
11. Remove LC-specific compatibility paths from `LocalRunner` in favor of
    registered operations.
12. Separate the LC application from reusable GUI components without changing
    the standalone PR GUI or rewriting the GUI framework.
13. Audit PR naming and placement for semantic clarity only.
14. Review compatibility shims and intentionally retain or retire each through
    a controlled compatibility pass.

Each stage requires its own scope, validation gate, and approval. The order may
change only through an explicitly approved architecture revision or when a
measured dependency makes a local reordering necessary.

## 19. Anti-Distraction Rules

> During a migration stage, unrelated architecture improvements, physics
> extensions, solver upgrades, GUI redesigns, performance tuning, and cleanup
> opportunities must be recorded as deferred items rather than implemented
> unless they are necessary to complete the current stage.

Additionally:

- change one architectural seam per migration task whenever practical;
- preserve behavior through compatibility shims before removing legacy paths;
- do not combine scientific changes with ownership moves;
- do not generalize from one material unless LC and PR, or a concrete future
  material, justify the abstraction;
- prefer composition over inheritance;
- prefer explicit material-owned types over universal base classes;
- preserve validated physics while reorganizing ownership;
- do not weaken assertions to conceal migration regressions;
- pre-existing regressions do not become scope for unrelated migration;
- record new ideas as deferred unless they block the approved target;
- use mechanical relocation before abstraction.

## 20. Explicitly Deferred Features

The following are outside the current architecture migration:

- vector optical propagation and full Maxwell solvers;
- generalized material plugin discovery or dynamic entry-point registration;
- universal material, solver, request, result, or state base classes;
- arbitrary material factory systems;
- GUI framework rewrite or automatic single-window material GUI generation;
- PR physics expansion and full PR tensor optical response;
- new nonlinear materials;
- new propagator families without a concrete validated requirement;
- package rename from LCProp to NLOProp.

## 21. Definition of Architectural Completion

The migration is complete when:

- LC and PR are peer canonical material packages;
- shared beam, launch, grid, and optics contain no LC- or PR-specific physics
  metadata;
- shared optics does not import LC or PR physics except through explicitly
  documented compatibility shims outside canonical shared implementations;
- shared numerics contains only equation-independent algorithms;
- material packages own material equations, requests, results, solvers,
  workflows, products, persistence codecs, and GUIs;
- shared execution runs material-owned operations without material-specific
  physics branches;
- shared presentation consumes material adapters rather than material result
  internals;
- canonical ownership is verified through location, provenance, identity, and
  dependency tests;
- compatibility shims are documented and either intentionally retained or
  removed in a controlled pass;
- LC and PR validation remain at their established baselines except for
  explicitly documented pre-existing failures.

Architectural completion does not require implementing deferred propagators,
materials, physics, or plugin infrastructure.

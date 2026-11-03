# Material Composition Architecture — Design Rationale

**Status:** Current architectural rationale

**Canonical contract:**
[`LCProp_Target_Architecture.md`](LCProp_Target_Architecture.md)

## Purpose

This document records why LCProp uses explicit, in-tree peer material packages
rather than a conventional plugin framework. It preserves the durable design
decisions and tradeoffs that emerged while adding photorefractive physics to
an initially liquid-crystal application.

The canonical architecture decision record remains authoritative. This
document explains the reasoning behind that structure rather than defining a
second contract.

## Decision

LCProp uses **material composition**, not dynamic plugins.

Each supported material is an explicit package that owns its physical state,
equations, source construction, solver controls, workflows, result types,
persistence payloads, diagnostics, and material-specific application. Shared
services accept small concrete compositions: prepared optical responses,
workflow operations, result adapters, and checkpoint codecs.

The architecture deliberately does not introduce:

- dynamic discovery or independently installed plugins;
- a universal `MaterialState` or `MaterialSolver` base class;
- an arbitrary material factory;
- one condition-heavy GUI with a global material selector;
- a speculative response hierarchy for optical interactions not yet present.

## Why the optical boundary comes first

The most important shared interface is
`advance_prepared_response()`. A material package converts its state into a
shared or per-channel multiplicative complex response for one optical
half-step. The optical engine validates the response shape and applies it in
the established split-step ordering around a material-neutral linear hop.

This separates two responsibilities cleanly:

- the material owns the constitutive mapping from its state to the optical
  response;
- the optical engine owns launch fields, diffraction, channel propagation,
  and response application.

LC maps director state to an optical response. PR maps normalized space-charge
state `E` to an optical response. The shared propagator does not need to know
which material produced the screen.

This boundary is intentionally narrower than a claim that every future
material can be represented by a local phase screen. A future model requiring
a nonlocal or operator-valued interaction should add a justified propagation
adapter rather than distort the existing interface.

## Material ownership

Material ownership is semantic, not merely a directory convention.

### Shared platform responsibilities

- beam and coherence representation;
- physical runtime grids and Fourier coordinates;
- NumPy/CuPy backend selection and precision;
- optical launch and propagation;
- cancellation and progress records;
- explicit workflow-operation dispatch;
- generic field, curve, diagnostic, and `RunData` presentation records;
- checkpoint-codec registration and dispatch;
- reusable GUI workers, workspace, editors, and result views.

### Material package responsibilities

- physical parameters and normalization;
- material state and initial conditions;
- optical driving-source construction;
- residuals, evolution equations, and solvers;
- convergence and continuation policy;
- conversion from material state to optical response;
- workflow-specific requests and results;
- material-state diagnostics and result adaptation;
- checkpoint payload encoding and validation;
- material-specific GUI controls and application assembly.

Workflow-specific request and result types are expected. Their specificity is
not architectural leakage when they live with their material owner and enter
shared services through explicit composition.

## Why composition is preferred to inheritance

LC director dynamics and PR space-charge dynamics do not share a useful state
shape, residual, convergence policy, or time interpretation. A common base
class would either expose almost no behavior or encode accidental assumptions
from one material.

LCProp instead composes direct dataclasses and callables:

- `WorkflowOperation` associates material and workflow identifiers with a run
  callable and a result-to-`RunData` adapter.
- `CheckpointCodec` associates material and workflow identifiers with a
  checkpoint type and save/load callables.
- prepared optical responses carry the material-to-optics interaction without
  exposing the material state.

These explicit registrations are inspectable, deterministic, and sufficient
for the in-tree materials. Dynamic discovery would add packaging, security,
versioning, compatibility, and support obligations without improving the
current scientific workflows.

## Execution and products

Shared local execution dispatches registered material operations without
branching on material names. LC compatibility methods remain available for
historical callers, but canonical LC operations and the PR operation use the
same material-neutral execution path.

The presentation model follows the same pattern. `RunData`, `FieldData`,
`CurveData`, and `DiagnosticData` are generic records. LC and PR each own the
adapter that selects faithfully representable products from their richer
workflow results. `RunData` is a presentation contract, not a replacement for
the material-owned scientific result.

## Persistence

Persistence is shared at the dispatch boundary and material-owned at the
payload boundary. The shared registry selects a codec by explicit material and
workflow identity. Each material codec owns request reconstruction, state
arrays, validation, fingerprints, and schema evolution.

Historical material-less LC checkpoint metadata is handled by explicit legacy
aliases. New formats must record material identity; compatibility is not a
reason to make LC payload semantics generic.

## GUI direction

Most users work primarily with one material. Separate LC and PR applications
therefore provide a clearer user experience than a universal material
selector. Shared GUI infrastructure remains appropriate for concepts that are
actually common:

- beam editing and grid controls;
- background workers, cancellation, and progress;
- workspace and run history;
- generic field, longitudinal, curve, and diagnostic views.

The material applications own their physics panels, solver controls, request
construction, continuation semantics, and material-specific interpretation.
This avoids conditionals that could mix LC and PR state while still reusing
stable presentation and execution services.

## Compatibility strategy

The ownership migration preserved historical imports and entry points through
thin compatibility shims. A shim must resolve to the canonical implementation
object; it must not become a second implementation or a location for new
behavior. Ownership is verified through module location, module provenance,
legacy/canonical object identity, and dependency-direction tests.

This policy allowed mechanical relocation before abstraction and reduced the
risk of changing numerical behavior during package cleanup.

## Decisions intentionally deferred

- external plugin discovery and a third-party SDK;
- repository or top-level package renaming;
- vector optical propagation;
- universal material requests, results, state, solvers, or GUIs;
- generalized serialization beyond concrete codec needs;
- extraction of orchestration abstractions without two proven consumers;
- redesign of validated LC or PR physics for architectural symmetry.

These may be reconsidered only when a concrete new requirement demonstrates
that the present compositions are insufficient.

## Architectural outcome

Adding PR demonstrated that LCProp can support fundamentally different
nonlinear materials through one optical and execution platform. The resulting
architecture is broader than its historical name but remains deliberately
simple: peer material packages, shared numerical and application services, and
explicit composition at each boundary.

That is the stable design. Future material work should first attempt to use the
existing prepared-response, operation, product, persistence, and GUI-framework
interfaces. New abstraction should follow evidence from a real material, not
precede it.

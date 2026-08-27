# General Workflow Input Screens — Stage B3

## Scope and status

Stage B3 makes the shared B1 intensity-raster launch plan an ordinary PR
workflow input. It does not change the specialized PR Image Amplification
experiment, material equations, propagation kernels, persistence schemas, or
remote transport schemas.

The supported workflow boundary is:

- reduced PR time-dependent (`pr_timedependent`);
- reduced coupled PR static (`pr_static`);
- canonical full-transverse zero-flux static (`pr_transverse_static`).

The full-transverse time-dependent research API and other specialized PR
research workflows are not part of this GUI-facing milestone.

## Shared launch configuration

`lcprop.optics.launch_configuration.LaunchConfiguration` is an immutable,
material-neutral composition of a canonical `BeamStack` and an ordered tuple
of `ChannelLaunchElements`. Construction validates the enabled-channel index
space, element types, duplicate assignments, and the underlying beam stack.
The GUI editor produces this object; scientific workflows receive only its
immutable request data and never depend on widget state.

Each supported PR request owns:

```text
beams: BeamStack
launch_elements: tuple[ChannelLaunchElements, ...] = ()
```

The empty default preserves the established launch path.

## Launch and power semantics

The shared `build_launch()` sequence is unchanged:

1. construct and normalize incident channels from their requested physical
   powers;
2. apply each ordered passive screen to its assigned canonical channel;
3. do not renormalize the post-screen field;
4. record incident powers, post-element powers, per-channel throughput, and
   post-element total power in `LaunchResult`;
5. pass the transformed field to the ordinary material workflow.

An intensity image on a beam is therefore a launch optical element. It does
not turn an ordinary static calculation into an Image Amplification
experiment and does not add carrier-recovery or image-quality diagnostics.

## Any-beam and mapping rules

Assignments use the canonical enabled-channel index after LaunchPlane has
removed disabled beam definitions. No beam position has special image
semantics: the first, second, or another enabled channel may be screened, and
multiple channels may carry independent ordered screens. Reordering or
deleting beams causes the editor to rebuild its canonical assignment mapping;
a stale out-of-range request assignment is rejected before propagation.

## Explicit initial fields

`initial_A` means an already-prepared runtime launch. A public request that
combines `initial_A` with nonempty `launch_elements` is rejected because the
requested precedence would otherwise be ambiguous and could apply a screen
twice. Internal accepted-state TD continuation retains the original launch
plan for provenance and power reporting while using the checkpoint's already
prepared `A0` exactly once.

## GUI behavior

The ordinary PR Beam tab now enables the shared input-screen editor. The Run
path snapshots `BeamPanel.launch_configuration()` and dispatches the same
registered PR operation used by a no-screen request.

The specialized Image Amplification input mode remains Stage A: its Beam tab
is disabled and its existing controls and request construction are unchanged.
The LC Beam editor remains disabled because LC request, initialization, and
persistence ownership require a later bounded integration stage. No host
silently ignores a configured launch plan.

## Persistence and remote execution

Stage B3 does not redesign persistence or transport. Until a later schema
milestone, nonempty launch plans are rejected explicitly by:

- PR experiment request persistence;
- PR time-dependent disk-checkpoint persistence;
- canonical PR transverse-static remote transport.

Existing empty-plan files and payloads retain their schema and behavior.
Restoring an existing empty-plan experiment clears any transient editor
assignments, preventing prior unsaved GUI state from contaminating the loaded
request.
In-memory TD checkpoint continuation remains supported.

## Declarative screen capability

The B1/B2 declarative plan currently represents ordered passive intensity
raster screens. The lower-level B1 passive-field transform also supports
general complex and pure-phase arrays, but those arrays are not a
`ChannelLaunchElements` request type and are therefore not claimed as a B3
headless workflow feature.

## Validation

Focused validation covers:

- immutable shared launch configuration and stale-index rejection;
- empty-plan compatibility;
- independently prepared versus workflow `A_initial` bitwise identity;
- first-, second-, and third-channel assignments and two screened channels;
- unaffected-channel identity and post-element power accounting;
- reduced TD, reduced static, and transverse-static consumption;
- explicit `initial_A` conflict rejection;
- in-memory continuation without double application;
- GUI request construction and registered-operation dispatch;
- specialized Image Amplification Beam-tab isolation;
- explicit persistence and remote-transport rejection;
- retained B1 screen physics and B2 editor mapping/preview behavior.

## Deferred work

- specialized Image Amplification rewiring and removal of its Stage A input
  duplication;
- LC request/workflow consumption;
- full-transverse TD and research-workflow adoption;
- experiment/checkpoint/remote schema support for raster sources;
- declarative phase-image screens.

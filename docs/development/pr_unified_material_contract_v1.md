# Unified PR material contracts — M1

These records define a future unified nonlinear material interface. They do not
implement material physics, spatial operators, optical projection, a workflow,
a codec, continuation, GUI controls, or TD. Importing `lcprop.pr.unified` imports
no backend, registers nothing, and executes no calculation. Import the defining
`specs` and `state` modules explicitly. Existing PR dispatch remains unchanged.

## Identity and units

| Contract | Identity |
|---|---|
| Material | `pr_unified_nonlinear_hopping_v1` |
| Spatial family | `periodic_compatible_fitted_flux_v1` |
| State | `pr_unified_q_psi_b_state_v1` |
| Intensity | `pr_local_total_transport_intensity_v1` |
| Normalization | `pr_channel_peak_reference_v1` |
| Diagnostics | `pr_unified_material_diagnostics_v1` |
| Units | `pr_normalized_material_units_v1` |
| Float32 state policy | `state32_linear64_bernoulli64_v1` |
| Float64 state policy | `state64_linear64_bernoulli64_v1` |

Unknown identities fail validation. These are contract identities, not registered
workflow IDs or persistence schema support. The original commissioned workflows
and the unstaged solver-rescue candidate are unaffected.

## Spatial metadata and layout

`PRUnifiedSpatialSpec` uses explicit `active_axes`, `active_shape`,
`normalized_lengths`, `batch_axes`, and `batch_shape`. Only periodic isotropic
uniform-grid metadata is supported. Active sizes are integers at least two;
normalized lengths are finite positive host real scalars. Batch size is a positive
integer. Booleans are not numeric sizes/parameters. No z axis is accepted.

| Problem | Active axes / shape | Batch axes / shape | q, psi, I layout | b layout | Future gauge/neutrality reduction |
|---|---|---|---|---|---|
| Material line | x / (Nx,) | none | (Nx,) | (1,) | x |
| Independent reduced columns | x / (Nx,) | y / (Ny,) | (Nx,Ny) | (Ny,1) | x separately in each column |
| Connected transverse plane | x,y / (Nx,Ny) | none | (Nx,Ny) | (2,) | x and y together |

The equal array shapes of a reduced column batch and a connected plane do not
make them the same problem. `reduction_axes` expresses this distinction without
performing a reduction. No plane-wide harmonic scalar replaces the independent
column harmonic fields. A named closure's target is shared metadata applied to
each independent active domain; it does not average columns into one circuit.
Arbitrary batching, y-only transport, longitudinal volumes, anisotropy,
nonperiodic electrodes and nonuniform-grid descriptions are rejected.

## Electrical experiment

`PRElectricalClosureSpec(identity, dimension, target, ...)` detaches small target
metadata into immutable tuples. Dimension must match the number of active axes,
not the number of stored array axes. All targets are finite normalized host
scalars. Named profiles are:

- `periodic_unbiased_zero_flux_v1`: zero target in one or two dimensions.
- `periodic_fixed_mean_field_v1`: target is prescribed harmonic field b.
- `periodic_prescribed_mean_current_v1`: target is prescribed mean conduction
  current at Static equilibrium; b is not implicitly fixed.
- `periodic_fixed_x_field_zero_y_current_v1`: two dimensions only, target
  `(b_x, 0)` prescribes x field and **zero mean y current**, not zero b_y.
- `periodic_a7_reservoir_current_1d_v1`: one dimension only; records
  `reservoir_field = E_app`, `background_intensity = I_b`, and the derived
  current target `(E_app * I_b,)`. `PRElectricalClosureSpec.a7(...)` computes
  this scalar metadata. Direct construction must supply exactly consistent
  metadata. The reservoir field is not the solved internal harmonic field.

I_b means configured total dark plus uniform background in normalized units;
zero background is permitted metadata, but does not guarantee positive transport
intensity. A7 negative reservoir fields are allowed; background is finite and
nonnegative and the derived current must be finite. Reservoir parameters on
other closures are rejected. There are no arbitrary U/V matrices, load/circuit
parameters, time dynamics, or silently inferred electrical experiments.

## Explicit numerical policy

`PRMaterialPrecisionSpec` requires a consistent identity and all dtype fields.
Float32 state/output selects float64 linear solves and Bernoulli evaluation;
float64 state/output retains float64 for both. Contradictory overrides fail.

The intended future mixed solver policy is assembled A32/r32 → A64/r64 → solve
on the selected backend → one cast of the correction to float32 for unchanged
state updates. M1 declares this policy but performs no casting, solve or backend
resolution. It makes no CPU/CuPy parity or native qualification claim itself.

## Borrowed state and intensity

`PRUnifiedMaterialState` stores q (log carrier), periodic psi, harmonic b,
spatial/closure/precision specs, and explicit resolved backend name (`numpy` or
`cupy`; no `auto`). It does not store canonical E. Deriving exp(q), fields or flux
is out of scope. `validate_structure()` verifies named identities, metadata,
array type, shape, native-endian real dtype and, for CuPy, common device ID.
It uses metadata from the already loaded backend; it does not import/allocate a
GPU backend or cause host conversion. Lists, mixed backends, complex arrays,
wrong precision/layout and cross-device states are rejected.

`PRTransportIntensity` borrows a total-I array with the same spatial/precision
metadata. It records the positive launch channel-peak reference, nonnegative
dark and uniform background, their finite total, and normalization identity.
The intended convention is optical/reference + dark + uniform. Reference
normalization is neither I_dark nor a fixed transport denominator. Structural
validation does not recompute that formula or inspect positivity/finiteness of
array values.

Constructors for state/intensity merely store references. Call
`validate_structure()` explicitly before using their layout. It deliberately
accepts structurally valid buffers containing NaN, negative intensity, or an
incorrect potential mean: physical acceptance belongs to a future solver. No
implicit copies, scans, reductions, normalization, regauging or array allocation
occur. Small Python metadata objects may be created during validation.

The caller owns runtime buffers, keeps them alive, and must not mutate them while
a consumer borrows a record. **Frozen dataclasses do not make NumPy/CuPy buffers
immutable.** A later owner mutation is visible through the record; records are
not snapshots or transactional checkpoints. Runtime ownership, detached trial
updates, physical checks and acceptance will be established in M2. State and
intensity record equality is object identity, avoiding elementwise array
comparisons. No M1 operation marks caller buffers read-only or changes ownership.

## Diagnostics are observations, not a certificate

`PRMaterialDiagnostics` contains immutable tuples of named finite host scalar
observations and nonnegative scalar limits. Names must be nonempty and unique
within each collection. Signed observations (e.g. harmonic field/current) are
allowed. Status is `not_evaluated` for empty metadata or `reported` for supplied
metadata. Unknown versions/status values fail; `accepted` is not a status.
M1 neither computes observations nor compares them with limits and exposes no
`passed` flag. A reported above-limit residual is valid metadata, not an accepted
physical state. Nonfinite observations must be handled by a future failure-evidence
contract; M1 does not implement that runtime error schema.

## Review boundary

Only three new modules, two focused test files and this document are introduced.
Tests use tiny synthetic arrays/metadata, including a simulated device metadata
type (not native CUDA certification). They cover closure/layout/precision identity,
borrowing, absence of array operations during validation, device consistency and
an isolated namespace import. No solver/operator, scientific acceptance test,
workflow registry, codec, checkpoint, resource estimator, GUI or TD code changes.
No existing Product file or pre-existing unstaged research candidate is modified.

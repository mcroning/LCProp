# Accepted-state physical TD continuation

`pr_transverse_accepted_psi_physical_segments_v1` supports local NumPy nonlinear
full-transverse TD with the existing `spectral_imex_euler` update and
`frozen_material_published_optical_first_v1` optical coupling. Canonical state is
accepted periodic ψ with layout `periodic_spectral_psi_zxy_v1`, not carrier,
field reconstructions or viewer products. Both scientific precisions retain
their exact dtype. Cross-backend continuation and reduced TD illumination
interventions are deferred; the older reduced same-request restart is unchanged.

`checkpoint_from_result(request, result)` requires retained accepted ψ, physical
validity and matching complete request provenance produced by the current
workflow. Fast products and historical results without that binding cannot be
promoted to checkpoints. A checkpoint owns an immutable-byte-backed array and
validates its state/provenance hashes on use. Frozen records alone do not make
arbitrary Python metadata immutable. Save/load uses a separate versioned,
hash-checked checkpoint file plus an NPY state (no pickle). Existing archive
schemas and segment-local result time retain their historical meanings.

`continue_transverse_td(request, checkpoint)` copies the accepted state without
regauging, rebuilding or casting it. It constructs the new entrance illumination
from beam settings, never the old optical endpoint. The first new step executes
the established complete optical source march against this state before its
material update. The post-step observational march remains observational.

Immutable: grid, apertures, z discretization, topology, transport/dielectric/
projection/model, integrator, scientific precision/backend, optical substeps,
normalization identity, wavelength/index, material scales, gain and canonical
scattering configuration. Mutable: physical beam powers/positions/angles/phases,
coherence and portable launch configuration, physical dark/uniform irradiances,
additional Nt and Δτ. Numerical timestep changes do not transform prior state
or rescale prior time. Equal-cadence time blocks are coalesced for segmentation-
independent floating-point accounting; cumulative τ and segment-local τ are
both retained. No laboratory-seconds conversion exists.

Only `pr_integral_total_illumination_mean_irradiance_v1` is accepted. Each new
segment independently derives Pref and mean reference irradiance from its own
physical illumination. Zero writing power makes an exact zero optical launch;
positive physical dark/background yields uniform transport intensity one.
Zero total physical illumination is rejected. There is no intensity floor.
The zero optical products use an explicit zero-reference presentation path,
without changing positive-power or historical presentation arithmetic.

Completed and cancelled results can supply accepted checkpoints. Failure of a
continuation or registered eligible fresh run raises `ContinuationFailure` with
its last accepted checkpoint and original exception cause. Candidate failures
never advance state or time. A pre-launch continuation failure retains the old
accepted segment's reference. The GUI receives accepted failure evidence before
the worker traceback, so Save/Continue uses the last accepted state.

The existing GUI Continue button opens a bounded segment editor. State-defining
parameters are inherited and locked; beam power (zero disables), position,
angles/phase, dark/background mW/cm², Nt and Δτ are editable. Inspect shows
source identity, cumulative time, changes and normalization. Save/Load Checkpoint
supports this new checkpoint explicitly; it does not reinterpret old reduced
checkpoints. Common Results includes segment lineage/reference metadata and
cumulative-time curves while preserving segment-local movie times and field
viewers. GUI execution is local only. Array-backed nonportable launch elements
remain outside this continuation contract; ordinary execution is unchanged.

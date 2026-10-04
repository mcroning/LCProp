# Unified nonlinear Static GUI integration (M6)

Fresh ordinary Static calculations with Fully nonlinear response select
`pr_static_unified_published_optical_first_v1`. Reduced x-only means independent
y batches on the optical plane; Full x-y means one connected material plane.
Local-I remains the distinct paper Eq. (5) response on the commissioned published
workflow. Tangent and all TD selections retain their existing identities.

Saved legacy requests are never converted implicitly. Representable legacy
requests restore their old workflow; unsupported midpoint/symmetric requests
retain the existing explicit loading rejection. A loaded legacy nonlinear Static
request can be replaced explicitly using **Create fresh unified nonlinear Static
request**. That action starts a new unbiased electrical experiment; it is not a
checkpoint or a state conversion. No continuation is provided.

Electrical conditions are explicit: unbiased zero flux, prescribed internal mean
field, prescribed normalized mean current, reduced A7 reservoir/current, and
full-x-y fixed-x-field/open-y-current. A7 retains E_app and I_b and imposes
J_ext = E_app I_b; b is solved. The open-y condition fixes mean J_y = 0, not b_y.
Dark plus uniform background has the unchanged transport meaning. The old
applied-field widget is disabled for unified requests; closure controls own these
quantities. No arbitrary circuit conditions are offered.

The adapter prepares the launch with the existing Product launch primitive on the
explicit execution backend. Its fresh launch-definition schema is
`pr_unified_static_fresh_launch_v1`; the prepared request remains the unchanged
M5 `UnifiedStaticRequest`. The registered material-neutral operation delegates
once to M5 `run_unified_products`. Local and Slurm use the same registration.
Prepared requests and canonical q/psi/b results use the dedicated M5 codecs;
the transport result envelope carries the exact M5 archive. Fresh experiment
files store the launch definition, not a legacy E-only or continuation state.

No material, projection, optical stepping, scattering or precision implementation
is duplicated. The projection is fixed to
`adjacent_face_arithmetic_to_optical_node_v1`. Float32 means state/output float32,
with float64 sensitive linear solves and Bernoulli evaluation; float64 uses
float64 throughout those components. Solver gates remain certified constants,
not the old coupled-pass/backtracking controls.

Interactive retains output intensity and boundary intensity cuts, together with
M5's mandatory accepted endpoint/material state. Analysis can select exact
far-field intensity and complex endpoint. Selecting a unified Analysis product
switches the retrieval policy to Analysis. Full additionally selects far field;
no policy constructs full longitudinal scientific volumes.

Results display **Boundary Intensity** and, when requested, **Far Field Intensity**
through the normal image selector. The display receives the M5 arrays and its
x/y or s_x/s_y coordinates directly. It performs no Fourier transform, optical
replay, new material solve, peak normalization or axis fabrication. Field titles
and result status identify the actual accepted reached z for completed, failed
or cancelled results. Pre-launch failures expose no fabricated optical image.
The complex endpoint remains available in the result archive.

M5 planning is used without allocating a launch for preflight. Array estimates
exclude archive metadata/allocator overhead and are not measured peak bounds.
Full x-y requests above 12,288 active nodes fail on every execution target,
including H200. This is an algorithmic reference-solver limit; reduced large
optical planes are not subject to that connected-2D ceiling. No large-2D
feasibility or M6 native GUI certification is claimed.

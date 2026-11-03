# Published optical-first reduced PR Static

Fresh ordinary reduced Static requests use `pr_static_published_optical_first_v1`
with arithmetic `per_cell_full_linear_arriving_material_full_phase_scattering_v1`.
The ordering reference is `reference/prprop/photonics-12-00113-v3/prprop3d.py`,
its Static branch in the longitudinal loop near line 1200.

Each requested cell performs exactly one full optical hop, constructs transport
intensity from that arriving field, solves the selected material equation, applies
one full material phase, then one canonical V2 scattering increment. It does not
average entrance/exit sources, split the material phase, perform optical/material
fixed-point iterations or backtracking, or require an independent replay.

Both Local-intensity field-linear Eq. (5) and nonlinear reduced hopping use the
same optical driver. Their existing material equations and solvers are unchanged.
Nonlinear material Newton iterations and material line search remain internal to
the fixed-intensity solve. Convergence, finite residuals and positive nonlinear
carrier density gate acceptance; failure retains the preceding accepted boundary.
No scientific retry occurs. Material controls are distinct from retired optical
coupling controls. GUI float32/float64 material defaults retain the existing
reduced Static tolerance values; explicitly edited settings remain explicit.

Optical samples include the accepted launch at z=0 and every accepted boundary.
Material/source/residual samples use separate explicit `material_z_um` values at
right endpoints h,2h,...,L. They are not centers. Products publish only accepted
cells; cancelled or failed candidates contribute no samples. Completed selected
Analysis results can contain exact final far-field intensity and complex endpoint.
Presentation normalization and bounded cut/preview construction reuse the
committed streaming collector; they never enter scientific arithmetic.

This milestone retains the streaming implementation's periodic optical-boundary
scope and exact integer-cell domain qualification. Canonical scattering uses the
requested physical cell, seed and normalization. It does not import historical
windows, derivative symbols, launch conventions, or legacy scattering.

The old `pr_static` midpoint request and symmetric
`pr_static_local_intensity_planes_v1` remain independently registered and retain
their original codecs and arithmetic. Loading them into the fresh GUI fails
explicitly rather than migrating their science. Their headless dispatch remains
available. Image-amplification experiments and continuation are not introduced
for the new identity. TD and transverse workflows retain their existing dispatch.

Scientific workspaces remain transverse; retained cuts/previews scale as
O(Ncells (Nx+Ny)). No longitudinal optical/material/replay volume is allocated by
the new workflow. Runtime/memory estimates are uncalibrated planning only.
Native NumPy/CuPy parity, nonlinear validity, exact product/codec behavior,
transfer accounting and H200 resource measurement remain commissioning work.

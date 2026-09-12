"""Backend-native nonlinear and linearized full-transverse PR dynamics."""

from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Callable

import numpy as np

from lcprop.core.backend import asnumpy, get_backend, scalar_float
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.launch_configuration import reject_prepared_launch_conflict
from lcprop.optics.screens import validate_channel_launch_elements
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.scattering import canonical_scattering_provenance
from lcprop.pr.transverse.diagnostics import state_diagnostics
from lcprop.pr.transverse.linearized_reference import (
    FIXED_MEAN_FIELD_ENSEMBLE,
    PRBiasedLinearizedReferenceSpec,
)
from lcprop.pr.transverse.linearized_timedependent_reference import (
    PR_PERIODIC_BIASED_LINEARIZED_TIMEDEPENDENT_REFERENCE_V1,
    linearized_timedependent_rhs,
    solve_pr_biased_linearized_timedependent_reference,
)
from lcprop.pr.transverse.projection import project_active_field
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_FULL_TRANSVERSE_PROFILE_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    PR_TRANSVERSE_IMEX_EULER,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseRunRequest,
    PRTransverseRunResult,
)
from lcprop.pr.transverse.transport import (
    explicit_euler_step,
    imex_euler_step,
    state_from_potential,
)
from lcprop.pr.workflow import (
    _apply_canonical_scattering_after_slice,
    _canonical_scattering_phase_stack,
    _validate_canonical_scattering_for_grid,
    advance_pr_slice_with_midpoint_source,
)


ProgressCallback = Callable[[RunProgress], None]


class _CancellationRequested(Exception):
    def __init__(self, stage: str):
        super().__init__(stage)
        self.stage = stage


def _check_cancel(token: CancellationToken | None, stage: str) -> None:
    if token is not None and token.is_cancelled():
        raise _CancellationRequested(stage)


def _validate_request(request: PRTransverseRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    if request.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR:
        if request.boundary.profile_id != PR_FULL_TRANSVERSE_PROFILE_V1:
            raise ValueError(
                "time-dependent Profile v1 does not support the periodic biased "
                "electrical profile"
            )
        if float(request.material.applied_field) != 0.0:
            raise ValueError("Profile v1 requires material.applied_field=0")
    request.material_response.validate_configuration(
        material_applied_field=request.material.applied_field,
        transport=request.transport,
        dielectric=request.dielectric,
        boundary=request.boundary,
        projection=request.projection,
    )
    request.solver.validate()
    request.backend.validate()
    if request.scattering is not None:
        request.scattering.validate()
    validate_channel_launch_elements(
        request.launch_elements,
        n_channels=len(request.beams.channels),
    )
    reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
    if request.backend.backend not in ("numpy", "cupy"):
        raise ValueError(
            "Profile v1 requires explicit backend='numpy' or backend='cupy'"
        )


def _linearized_reference_spec(
    request: PRTransverseRunRequest,
    *,
    dx_normalized: float,
    dy_normalized: float,
) -> PRBiasedLinearizedReferenceSpec:
    reference_intensity = request.material_response.reference_intensity
    if reference_intensity is None:  # guarded by request validation
        raise ValueError("linearized material response requires reference_intensity")
    return PRBiasedLinearizedReferenceSpec(
        reference_intensity=float(reference_intensity),
        applied_field=float(request.boundary.applied_field_x),
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        m_y=float(request.transport.m_y),
        h_y=float(request.dielectric.h_y),
    )


def _linearized_material_step(
    psi,
    source,
    *,
    request: PRTransverseRunRequest,
    spec: PRBiasedLinearizedReferenceSpec,
    xp,
    cancellation_token: CancellationToken | None,
    material_response_calls: list[int],
):
    """Advance independent planes with the accepted exact modal propagator."""

    candidate = xp.empty_like(psi)
    for plane_index in range(source.shape[0]):
        _check_cancel(cancellation_token, "linearized_material_plane")
        response = solve_pr_biased_linearized_timedependent_reference(
            source[plane_index],
            time_normalized=float(request.solver.dt_normalized),
            spec=spec,
            initial_delta_psi=psi[plane_index],
            backend=request.backend,
        )
        candidate[plane_index] = response.delta_psi
        material_response_calls[0] += 1
        _check_cancel(cancellation_token, "linearized_material_plane")
    return candidate


def _initial_fields(request, *, launch, grid, complex_dtype, real_dtype):
    xp = grid.xp
    if request.initial_A is None:
        A0 = launch.A0.copy()
    else:
        A0 = xp.asarray(request.initial_A, dtype=complex_dtype).copy()
        if A0.shape != launch.A0.shape:
            raise ValueError(f"initial_A shape {A0.shape} does not match {launch.A0.shape}")
        if not bool(asnumpy(xp.all(xp.isfinite(A0)))):
            raise ValueError("initial_A must contain only finite values")
    expected = (grid.Nz, grid.Nx, grid.Ny)
    if request.initial_psi is None:
        psi = xp.zeros(expected, dtype=real_dtype)
    else:
        psi = xp.asarray(request.initial_psi, dtype=real_dtype).copy()
        if psi.shape != expected:
            raise ValueError(f"initial_psi shape {psi.shape} does not match {expected}")
        if not bool(asnumpy(xp.all(xp.isfinite(psi)))):
            raise ValueError("initial_psi must contain only finite values")
    psi -= xp.mean(psi, axis=(-2, -1), keepdims=True)
    return A0, psi


def _optical_pass(
    A0,
    psi,
    *,
    request,
    grid,
    kernel,
    peak_reference,
    wavelength_um,
    dx_normalized,
    dy_normalized,
    cancellation_token=None,
    scattering_phase_stack=None,
):
    xp = grid.xp
    A = A0.copy()
    source = xp.empty(psi.shape, dtype=grid.real_dtype)
    intensity_before = None
    state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        applied_field_x=request.boundary.applied_field_x,
        xp=xp,
    )
    active = project_active_field(
        state.E_x, state.E_y, profile=request.projection, xp=xp
    )
    for z_index in range(grid.Nz):
        _check_cancel(cancellation_token, "material_source_optical_z_march")
        A, source[z_index], intensity_before = advance_pr_slice_with_midpoint_source(
            A,
            active[z_index],
            kernel=kernel,
            optical_substeps=request.solver.optical_substeps,
            dz_um=grid.dz_um,
            wavelength_um=wavelength_um,
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            peak_intensity_reference=peak_reference,
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=xp,
            _intensity_before=intensity_before,
            _return_exit_intensity=True,
        )
        _apply_canonical_scattering_after_slice(
            A,
            scattering=request.scattering,
            z_index=z_index,
            grid=grid,
            z_length_um=request.grid.z_length_um,
            xp=xp,
            phase=(
                None
                if scattering_phase_stack is None
                else scattering_phase_stack[z_index]
            ),
        )
    return A, source


def run_pr_transverse_timedependent(
    request: PRTransverseRunRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PRTransverseRunResult:
    """Run the full-transverse PR workflow on the requested backend.

    A material candidate becomes physical only after its complete source pass
    and finite Euler update. Cancellation during either discardable stage
    returns the preceding accepted ψ state, followed by a consistent replay.
    """

    started = perf_counter()
    _validate_request(request)
    backend = get_backend(request.backend)
    xp = backend.xp
    real_dtype = backend.real_dtype
    complex_dtype = backend.complex_dtype
    grid = make_grid(request.grid, xp=xp, real_dtype=real_dtype)
    _validate_canonical_scattering_for_grid(
        request.scattering,
        grid=grid,
        z_length_um=request.grid.z_length_um,
    )
    launch = build_launch(
        request.beams,
        grid,
        complex_dtype=complex_dtype,
        launch_elements=request.launch_elements,
    )
    A0, psi = _initial_fields(
        request,
        launch=launch,
        grid=grid,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
    )
    psi_initial = psi.copy()
    wavelengths = tuple(float(ch.wavelength_um) for ch in request.beams.channels)
    if any(value != wavelengths[0] for value in wavelengths[1:]):
        raise ValueError("transverse PR workflow requires one shared wavelength")
    wavelength_um = wavelengths[0]
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / int(request.solver.optical_substeps),
        wavelength=wavelength_um,
        n_ref=request.material.refractive_index,
        xp=xp,
    )
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    scattering_phase_stack = _canonical_scattering_phase_stack(
        request.scattering,
        grid=grid,
        z_length_um=request.grid.z_length_um,
        xp=xp,
    )
    k0 = request.material.characteristic_wavenumber_per_um
    dx_normalized = k0 * grid.dx_um
    dy_normalized = k0 * grid.dy_um
    initial_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        applied_field_x=request.boundary.applied_field_x,
        xp=xp,
    )
    initial_carrier = xp.sum(initial_state.carrier_density, axis=(-2, -1))
    max_carrier_drift = xp.asarray(0.0, dtype=real_dtype)
    completed_steps = 0
    cancelled = False
    cancellation_stage = None
    material_response_calls = [0]
    linearized_spec = (
        _linearized_reference_spec(
            request,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
        )
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
        else None
    )

    for step_index in range(int(request.solver.Nt)):
        try:
            _check_cancel(cancellation_token, "material_step_boundary")
            _, source = _optical_pass(
                A0,
                psi,
                request=request,
                grid=grid,
                kernel=kernel,
                peak_reference=peak_reference,
                wavelength_um=wavelength_um,
                dx_normalized=dx_normalized,
                dy_normalized=dy_normalized,
                cancellation_token=cancellation_token,
                scattering_phase_stack=scattering_phase_stack,
            )
            if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
                candidate = _linearized_material_step(
                    psi,
                    source,
                    request=request,
                    spec=linearized_spec,
                    xp=xp,
                    cancellation_token=cancellation_token,
                    material_response_calls=material_response_calls,
                )
            else:
                if request.solver.integrator == PR_TRANSVERSE_IMEX_EULER:
                    step_function = imex_euler_step
                elif request.solver.integrator == PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE:
                    step_function = explicit_euler_step
                else:  # guarded by PRTransverseSolverOptions.validate()
                    raise ValueError(
                        f"unknown transverse PR integrator: {request.solver.integrator}"
                    )
                candidate = step_function(
                    psi,
                    source,
                    dt_normalized=request.solver.dt_normalized,
                    dx_normalized=dx_normalized,
                    dy_normalized=dy_normalized,
                    m_y=request.transport.m_y,
                    h_y=request.dielectric.h_y,
                    applied_field_x=request.boundary.applied_field_x,
                    xp=xp,
                )
            _check_cancel(cancellation_token, "after_material_candidate")
        except _CancellationRequested as exc:
            cancelled = True
            cancellation_stage = exc.stage
            break

        psi = candidate
        completed_steps = step_index + 1
        accepted_state = state_from_potential(
            psi,
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            h_y=request.dielectric.h_y,
            applied_field_x=request.boundary.applied_field_x,
            xp=xp,
        )
        current_carrier = xp.sum(accepted_state.carrier_density, axis=(-2, -1))
        max_carrier_drift = xp.maximum(
            max_carrier_drift,
            xp.max(
                xp.abs(current_carrier - initial_carrier)
                / xp.abs(initial_carrier)
            ),
        )
        if progress_callback is not None:
            material_time = completed_steps * float(request.solver.dt_normalized)
            progress_callback(
                RunProgress(
                    workflow=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
                    status="running",
                    completed_units=completed_steps,
                    total_units=int(request.solver.Nt),
                    current_coordinate=material_time,
                    coordinate_name="material_time",
                    coordinate_unit="normalized",
                    elapsed_wall_time=perf_counter() - started,
                    latest_field_state={
                        "psi_current": np.asarray(asnumpy(psi)).copy()
                    },
                    message="transverse PR material-time step accepted",
                    diagnostics={
                        "material_response": request.material_response.model,
                        "material_update": (
                            "exact_frozen_source_modal"
                            if request.material_response.model
                            == PR_MATERIAL_RESPONSE_LINEARIZED
                            else request.solver.integrator
                        ),
                    },
                    completed_step=completed_steps,
                    total_steps=int(request.solver.Nt),
                    current_time=material_time,
                    segment_completed_steps=completed_steps,
                    segment_total_steps=int(request.solver.Nt),
                    cumulative_completed_steps=completed_steps,
                    segment_elapsed_time=material_time,
                    cumulative_time=material_time,
                )
            )

    # A complete replay through the last accepted state keeps products physical.
    A_final, final_source = _optical_pass(
        A0,
        psi,
        request=request,
        grid=grid,
        kernel=kernel,
        peak_reference=peak_reference,
        wavelength_um=wavelength_um,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        scattering_phase_stack=scattering_phase_stack,
    )
    final_state = state_from_potential(
        psi,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        applied_field_x=request.boundary.applied_field_x,
        xp=xp,
    )
    diagnostics = state_diagnostics(
        final_state,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=request.dielectric.h_y,
        xp=xp,
    )
    power_initial = normalized_power(A0, grid)
    power_final = normalized_power(A_final, grid)
    diagnostics.update(
        {
            "carrier_relative_drift_max": scalar_float(max_carrier_drift),
            "optical_power_relative_drift": (power_final - power_initial) / power_initial,
            "finite_optical_state": bool(asnumpy(xp.all(xp.isfinite(A_final)))),
            "cancellation_observed_stage": cancellation_stage,
            "integrator_policy": (
                "exact_frozen_source_linearized_modal_update"
                if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
                else (
                    "production_first_order_spectral_imex"
                    if request.solver.integrator == PR_TRANSVERSE_IMEX_EULER
                    else "transparent_reference_not_production_default"
                )
            ),
            "complete_final_optical_replay": True,
            "material_response": request.material_response.model,
            "material_response_validation": (
                "experimental"
                if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
                else "validated"
            ),
            "material_response_calls": material_response_calls[0],
            "accepted_material_steps": completed_steps,
            "material_time_normalized": (
                completed_steps * float(request.solver.dt_normalized)
            ),
        }
    )
    if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
        rhs_sum_squares = 0.0
        rhs_max = 0.0
        rhs_cells = 0
        for plane_index in range(final_source.shape[0]):
            rhs_plane = linearized_timedependent_rhs(
                psi[plane_index],
                final_source[plane_index],
                spec=linearized_spec,
                backend=request.backend,
            )
            rhs_sum_squares += scalar_float(xp.sum(rhs_plane * rhs_plane))
            rhs_max = max(rhs_max, scalar_float(xp.max(xp.abs(rhs_plane))))
            rhs_cells += int(rhs_plane.size)
        diagnostics.update({
            "frozen_intensity_material_equation": (
                "exact_analytic_fourier_transient"
            ),
            "source_cadence": "one_complete_optical_pass_per_material_interval",
            "nonlinear_material_iterations": "not_applicable",
            "linearized_rhs_rms": (rhs_sum_squares / rhs_cells) ** 0.5,
            "linearized_rhs_max": rhs_max,
            "memory_policy": {
                "backend_working_set": "one_transverse_material_plane",
                "retained_material_state": "full_longitudinal_volume",
                "fft_batch_axes": "transverse_only",
            },
        })
    scattering_provenance = None
    if request.scattering is not None:
        scattering_provenance = canonical_scattering_provenance(
            request.scattering,
            z_length_um=request.grid.z_length_um,
            Nx=grid.Nx,
            Ny=grid.Ny,
            x_aperture_um=request.grid.x_aperture_um,
            y_aperture_um=request.grid.y_aperture_um,
            real_dtype=grid.real_dtype,
            xp=xp,
        )
        diagnostics["canonical_scattering"] = scattering_provenance
    physics_profile_id = (
        PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
        if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
        else PR_FULL_TRANSVERSE_PROFILE_V1
    )
    resolved_profile = {
        "physics_profile_id": physics_profile_id,
        "workflow": PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        "transport_model": "full_transverse",
        "material_response": asdict(request.material_response),
        "grid_request": asdict(request.grid),
        "beam_request": asdict(request.beams),
        "material": asdict(request.material),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "characteristic_wavenumber_per_um": k0,
        "dx_normalized": dx_normalized,
        "dy_normalized": dy_normalized,
    }
    if request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED:
        resolved_profile.update({
            "linearized_model_id": (
                PR_PERIODIC_BIASED_LINEARIZED_TIMEDEPENDENT_REFERENCE_V1
            ),
            "electrical_ensemble": FIXED_MEAN_FIELD_ENSEMBLE,
            "reference_intensity": float(
                request.material_response.reference_intensity
            ),
            "applied_field": float(request.boundary.applied_field_x),
            "validation_status": "experimental",
        })
    if scattering_provenance is not None:
        resolved_profile["canonical_scattering"] = scattering_provenance
    return PRTransverseRunResult(
        A_initial=np.asarray(asnumpy(A0)).copy(),
        A_final=np.asarray(asnumpy(A_final)).copy(),
        psi_initial=np.asarray(asnumpy(psi_initial)).copy(),
        psi_final=np.asarray(asnumpy(psi)).copy(),
        power_initial=power_initial,
        power_final=power_final,
        completed_steps=completed_steps,
        time_normalized=completed_steps * float(request.solver.dt_normalized),
        grid_summary=grid.summary(),
        launch_summary=launch.summary(),
        backend_summary=backend.summary(),
        resolved_profile=resolved_profile,
        status="cancelled" if cancelled else "completed",
        requested_steps=int(request.solver.Nt),
        diagnostics=diagnostics,
    )


__all__ = ["run_pr_transverse_timedependent"]

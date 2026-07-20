from dataclasses import replace

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.optics.splitstep import advance_slice, linear_kernel
from lcprop.optics.substeps import (
    build_optical_substep_kernel,
    resolve_optical_substeps,
)
from lcprop.workflows import run_static, run_timedependent
from lcprop.workflows.runtime import build_runtime_components
from lcprop.workflows.soliton import SolitonRequest, run_soliton


def _runtime(*, enabled=True, max_substeps=16):
    return RuntimeOptions(
        optical_substeps_enabled=enabled,
        optical_dn_max_est=0.02,
        optical_max_phase_per_substep_rad=0.30,
        optical_max_substeps=max_substeps,
    )


def _request(*, strategy="fixed_theta", runtime=None, beams=None):
    workflow = (
        StaticWorkflowOptions()
        if strategy == "fixed_theta"
        else StaticWorkflowOptions(
            strategy="local_self_consistent",
            theta_solver="picard_cn",
            optics_solver="splitstep",
            coupling="self_consistent",
        )
    )
    return StaticRunRequest(
        grid=GridSpec(
            Nx=12,
            Ny=10,
            dz_um=5.0,
            x_aperture_um=30.0,
            y_aperture_um=25.0,
            z_length_um=5.0,
        ),
        material=LCMaterial(
            ne=1.7,
            no=1.5,
            K=7.0e-12,
            delta_epsilon=13.0,
        ),
        bias=BiasSpec(V_bias=0.0, theta_bc=np.pi / 4.0),
        beams=beams
        or BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    power_mW=0.05,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                ),
            )
        ),
        solver=StaticSolverOptions(
            workflow=workflow,
            static_max_relax_iterations=1,
            static_max_coupled_passes=1,
            record_iteration_history=False,
        ),
        output=OutputOptions(save_slices=False, save_full=False),
        runtime=runtime or _runtime(),
    )


def _sample_field_and_theta():
    rng = np.random.default_rng(123)
    A = (
        rng.normal(size=(1, 9, 8))
        + 1j * rng.normal(size=(1, 9, 8))
    ).astype(np.complex128)
    theta = rng.uniform(0.2, 1.0, size=(9, 8))
    fxy2 = (
        np.fft.fftfreq(9, d=0.6)[:, None] ** 2
        + np.fft.fftfreq(8, d=0.7)[None, :] ** 2
    )
    return A, theta, fxy2


def test_disabled_nsub_one_reproduces_legacy_propagation_exactly():
    A, theta, fxy2 = _sample_field_and_theta()
    full_kernel = linear_kernel(
        fxy2,
        dz=5.0,
        wavelength=0.633,
        n_ref=1.5,
        xp=np,
    )
    plan, disabled_kernel = build_optical_substep_kernel(
        fxy2,
        dz_um=5.0,
        propagation_wavelength_um=0.633,
        active_wavelengths_um=(0.633,),
        n_ref=1.5,
        enabled=False,
        dn_max_est=0.02,
        max_phase_per_substep_rad=0.30,
        max_substeps=16,
        xp=np,
    )
    np.testing.assert_array_equal(disabled_kernel, full_kernel)
    assert plan.Nsub == 1

    legacy = advance_slice(
        A.copy(),
        theta,
        kernel=full_kernel,
        dz=5.0,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
        xp=np,
    )
    resolved = advance_slice(
        A.copy(),
        theta,
        kernel=disabled_kernel,
        dz=5.0,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
        Nsub=plan.Nsub,
        xp=np,
    )
    np.testing.assert_array_equal(resolved, legacy)

    request = _request(runtime=_runtime(enabled=False))
    production = run_static(request)
    components = build_runtime_components(request)
    legacy_workflow_field = advance_slice(
        components.launch.A0.copy(),
        components.bias.theta_2d,
        kernel=linear_kernel(
            components.grid.fxy2_um,
            dz=components.grid.dz_um,
            wavelength=components.wavelength_um,
            n_ref=components.n_ref,
            xp=components.grid.xp,
        ),
        dz=components.grid.dz_um,
        wavelength=components.wavelength_um,
        n_ref=components.n_ref,
        ne=request.material.ne,
        no=request.material.no,
        xp=components.grid.xp,
    )
    np.testing.assert_array_equal(production.A_final, legacy_workflow_field)


def test_nominal_substeps_match_explicit_fine_grid_fixed_theta():
    A, theta, fxy2 = _sample_field_and_theta()
    plan, kernel_sub = build_optical_substep_kernel(
        fxy2,
        dz_um=20.0,
        propagation_wavelength_um=0.633,
        active_wavelengths_um=(0.633,),
        n_ref=1.5,
        enabled=True,
        dn_max_est=0.02,
        max_phase_per_substep_rad=0.30,
        max_substeps=16,
        xp=np,
    )
    nominal = advance_slice(
        A.copy(),
        theta,
        kernel=kernel_sub,
        dz=20.0,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
        Nsub=plan.Nsub,
        xp=np,
    )
    explicit = A.copy()
    for _ in range(plan.Nsub):
        advance_slice(
            explicit,
            theta,
            kernel=kernel_sub,
            dz=plan.dz_sub_um,
            wavelength=0.633,
            n_ref=1.5,
            ne=1.7,
            no=1.5,
            Nsub=1,
            xp=np,
        )
    np.testing.assert_array_equal(nominal, explicit)

    bad_full_kernel = linear_kernel(
        fxy2,
        dz=20.0,
        wavelength=0.633,
        n_ref=1.5,
        xp=np,
    )
    repeated_full_dz = advance_slice(
        A.copy(),
        theta,
        kernel=bad_full_kernel,
        dz=20.0,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
        Nsub=plan.Nsub,
        xp=np,
    )
    assert not np.allclose(nominal, repeated_full_dz, rtol=1e-10, atol=1e-12)


def test_static_td_and_soliton_resolve_same_plan_and_kernel_spacing():
    request = _request()
    expected = resolve_optical_substeps(
        dz_um=request.grid.dz_um,
        wavelengths_um=(
            channel.wavelength_um for channel in request.beams.channels
        ),
        enabled=request.runtime.optical_substeps_enabled,
        dn_max_est=request.runtime.optical_dn_max_est,
        max_phase_per_substep_rad=(
            request.runtime.optical_max_phase_per_substep_rad
        ),
        max_substeps=request.runtime.optical_max_substeps,
    )

    fixed = run_static(request)
    local = run_static(
        replace(
            request,
            solver=_request(strategy="local_self_consistent").solver,
        )
    )
    td = run_timedependent(
        TimeDependentRunRequest(
            grid=request.grid,
            material=request.material,
            bias=request.bias,
            beams=request.beams,
            solver=TimeDependentSolverOptions(Nt=0),
            output=request.output,
            runtime=request.runtime,
        )
    )
    soliton = run_soliton(
        SolitonRequest(
            base=request,
            max_outer=1,
            theta_steps_per_outer=1,
            field_mix=0.25,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )

    summaries = (
        fixed.provenance,
        local.provenance,
        td.provenance,
        soliton.metrics,
    )
    for summary in summaries:
        assert summary["optical_Nsub"] == expected.Nsub
        assert summary["optical_dz_sub_um"] == expected.dz_sub_um
        assert summary["optical_phi_est_rad"] == expected.phi_est_rad
        assert (
            summary["optical_substep_cap_reached"]
            == expected.cap_reached
        )

    components = build_runtime_components(request)
    expected_kernel = linear_kernel(
        components.grid.fxy2_um,
        dz=expected.dz_sub_um,
        wavelength=components.wavelength_um,
        n_ref=components.n_ref,
        xp=components.grid.xp,
    )
    np.testing.assert_array_equal(components.kernel, expected_kernel)


def test_multichannel_uses_shortest_active_wavelength():
    plan = resolve_optical_substeps(
        dz_um=10.0,
        wavelengths_um=(1.064, 0.532, 0.633),
        enabled=True,
        dn_max_est=0.02,
        max_phase_per_substep_rad=0.30,
        max_substeps=64,
    )
    restrictive = resolve_optical_substeps(
        dz_um=10.0,
        wavelengths_um=(0.532,),
        enabled=True,
        dn_max_est=0.02,
        max_phase_per_substep_rad=0.30,
        max_substeps=64,
    )
    assert plan.shortest_wavelength_um == 0.532
    assert plan.Nsub == restrictive.Nsub
    assert plan.phi_est_rad == restrictive.phi_est_rad

    beams = BeamStack(
        channels=(
            BeamChannel(
                name="long",
                wavelength_um=1.064,
                power_mW=0.025,
                waist_x_um=3.0,
                waist_y_um=3.0,
            ),
            BeamChannel(
                name="short",
                wavelength_um=0.532,
                power_mW=0.025,
                waist_x_um=3.0,
                waist_y_um=3.0,
            ),
        )
    )
    components = build_runtime_components(
        _request(beams=beams)
    )
    integrated = resolve_optical_substeps(
        dz_um=components.grid.dz_um,
        wavelengths_um=(1.064, 0.532),
        enabled=components.request.runtime.optical_substeps_enabled,
        dn_max_est=components.request.runtime.optical_dn_max_est,
        max_phase_per_substep_rad=(
            components.request.runtime.optical_max_phase_per_substep_rad
        ),
        max_substeps=components.request.runtime.optical_max_substeps,
    )
    assert components.optical_substeps.Nsub == integrated.Nsub
    assert components.optical_substeps.shortest_wavelength_um == 0.532


def test_cap_reached_is_explicit_in_plan_and_workflow_diagnostics():
    runtime = _runtime(enabled=True, max_substeps=2)
    request = _request(runtime=runtime)
    plan = resolve_optical_substeps(
        dz_um=request.grid.dz_um,
        wavelengths_um=(0.633,),
        enabled=True,
        dn_max_est=runtime.optical_dn_max_est,
        max_phase_per_substep_rad=(
            runtime.optical_max_phase_per_substep_rad
        ),
        max_substeps=runtime.optical_max_substeps,
    )
    assert plan.Nsub == 2
    assert plan.cap_reached is True

    result = run_static(request)
    assert result.provenance["optical_substep_cap_reached"] is True
    assert "optical substep cap reached" in result.warnings

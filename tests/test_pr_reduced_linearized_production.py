from dataclasses import replace
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

import lcprop.persistence  # Complete codec registration before direct imports.
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.experiment_codec import (
    decode_pr_static_request,
    encode_pr_static_request,
)
from lcprop.pr.gui.run_cost import classify_pr_run_cost
from lcprop.pr.products import pr_static_result_to_run_data
from lcprop.pr.reduced_linearized import (
    PRReducedLinearizedSpec,
    reduced_linearized_residual,
    reduced_linearized_symbol,
    solve_pr_reduced_linearized_intensity,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static_transport_codec import (
    decode_pr_static_transport_request,
    decode_pr_static_transport_result,
    encode_pr_static_transport_request,
    encode_pr_static_transport_result,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    run_pr_static,
)
from lcprop.pr.transverse.linearized_reference import (
    PRBiasedLinearizedReferenceSpec,
    solve_pr_biased_linearized_reference,
)
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.transport.result_policy import FAST_RESULT_POLICY


def _relative_l2(left, right) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.linalg.norm(left - right) / np.linalg.norm(right))


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _base_request(
    *,
    nx=24,
    ny=12,
    background=20.0,
    applied_field=0.4,
    gain_length_product=0.04,
    y_variation=True,
    linearized=False,
):
    grid = GridSpec(
        Nx=nx,
        Ny=ny,
        x_aperture_um=48.0,
        y_aperture_um=36.0,
        dz_um=4.0,
        z_length_um=8.0,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=18.0,
                waist_y_um=15.0,
            ),
        )
    )
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    intensity = 1.0 + 0.12 * np.cos(2.0 * math.pi * 2.0 * x / nx)
    if y_variation:
        intensity = intensity * (
            1.0 + 0.10 * np.cos(2.0 * math.pi * y / ny)
        )
    else:
        intensity = np.repeat(intensity, ny, axis=1)
    initial_A = np.sqrt(intensity)[None].astype(np.complex128)
    return PRStaticRunRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=background,
            applied_field=applied_field,
            gain_length_product=gain_length_product,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRStaticWorkflowOptions(
            max_coupled_passes=12,
            residual_rms_tolerance=1e-8,
            residual_max_tolerance=1e-7,
            optical_substeps=1,
        ),
        backend=BackendSpec(
            backend="numpy", precision="float64", verbose=False
        ),
        initial_A=initial_A,
        material_response=PRTransverseMaterialResponseSpec(
            model=(
                PR_MATERIAL_RESPONSE_LINEARIZED
                if linearized
                else PR_MATERIAL_RESPONSE_NONLINEAR
            ),
            reference_intensity=(background + 1.0 if linearized else None),
        ),
    )


def _full_linearized_request(reduced: PRStaticRunRequest):
    return PRTransverseStaticRunRequest(
        grid=reduced.grid,
        beams=reduced.beams,
        material=replace(reduced.material, applied_field=0.0),
        solver=PRTransverseStaticWorkflowOptions(
            max_coupled_iterations=12,
            equilibrium_rms_tolerance=1e-8,
            equilibrium_max_tolerance=1e-7,
            replay_rtol=1e-11,
            replay_atol=1e-12,
        ),
        backend=reduced.backend,
        initial_A=reduced.initial_A,
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=0.0,
        ),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=reduced.material_response.reference_intensity,
        ),
    )


def test_reduced_linearized_symbol_is_direct_centered_difference_linearization():
    nx = 32
    mode = 3
    spec = PRReducedLinearizedSpec(
        reference_intensity=0.8,
        applied_field=-0.6,
        background_intensity=0.8,
        dx_normalized=0.17,
    )
    symbol = reduced_linearized_symbol(nx, spec=spec)
    phase = 2.0 * math.pi * mode / nx
    k1 = math.sin(phase) / spec.dx_normalized
    k2_squared = 4.0 * math.sin(0.5 * phase) ** 2 / spec.dx_normalized**2
    expected = (1j * k1 - spec.equilibrium_field) / (
        spec.reference_intensity
        * (1.0 + k2_squared + 1j * spec.equilibrium_field * k1)
    )

    assert symbol.k1.dtype == np.float64
    assert symbol.response_kernel.dtype == np.complex128
    assert symbol.response_kernel[mode] == pytest.approx(expected, rel=2e-16)


def test_even_grid_nyquist_uses_exact_centered_first_derivative_null():
    nx, ny = 32, 4
    spec = PRReducedLinearizedSpec(1.0, 0.5, 1.0, 0.2)
    symbol = reduced_linearized_symbol(nx, spec=spec)
    perturbation = 0.01 * (-1.0) ** np.arange(nx)[:, None]
    intensity = np.repeat(1.0 + perturbation, ny, axis=1)
    result = solve_pr_reduced_linearized_intensity(intensity, spec=spec)
    expected_response = -spec.equilibrium_field / (
        spec.reference_intensity * (1.0 + 4.0 / spec.dx_normalized**2)
    )

    assert symbol.k1[nx // 2] == 0.0
    np.testing.assert_allclose(
        result.delta_E,
        expected_response * np.repeat(perturbation, ny, axis=1),
        rtol=0.0,
        atol=3e-18,
    )
    assert np.max(np.abs(result.residual)) < 2e-14


@pytest.mark.parametrize(
    ("precision", "real_dtype", "complex_dtype", "atol"),
    [
        ("float64", np.float64, np.complex128, 2e-13),
        ("float32", np.float32, np.complex64, 3e-5),
    ],
)
def test_reduced_linearized_multimode_solve_closes_residual_and_preserves_dtype(
    precision, real_dtype, complex_dtype, atol
):
    nx, ny = 31, 7
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    intensity = (
        1.2
        + 0.03 * np.cos(2.0 * math.pi * 2.0 * x / nx)
        + 0.02 * np.sin(2.0 * math.pi * 5.0 * x / nx)
        * np.cos(2.0 * math.pi * y / ny)
    ).astype(real_dtype)
    spec = PRReducedLinearizedSpec(1.2, 0.7, 0.9, 0.21)
    result = solve_pr_reduced_linearized_intensity(
        intensity,
        spec=spec,
        backend=BackendSpec("numpy", precision, False),
    )

    assert result.E.dtype == real_dtype
    assert result.delta_E.dtype == real_dtype
    assert result.residual.dtype == real_dtype
    assert result.response_kernel.dtype == complex_dtype
    assert np.max(np.abs(result.residual)) < atol


def test_float32_reduced_linearized_matches_float64_and_bias_mapping():
    nx, ny = 33, 5
    x = np.arange(nx)[:, None]
    intensity64 = np.repeat(
        0.95 + 0.04 * np.cos(2.0 * math.pi * 4.0 * x / nx), ny, axis=1
    )
    spec = PRReducedLinearizedSpec(0.95, -0.5, 0.76, 0.13)
    result64 = solve_pr_reduced_linearized_intensity(intensity64, spec=spec)
    result32 = solve_pr_reduced_linearized_intensity(
        intensity64.astype(np.float32),
        spec=spec,
        backend=BackendSpec("numpy", "float32", False),
    )

    assert spec.equilibrium_field == pytest.approx(-0.4)
    np.testing.assert_allclose(result32.E, result64.E, rtol=2e-6, atol=2e-7)
    uniform = np.full((nx, ny), 1.1, dtype=np.float64)
    uniform_result = solve_pr_reduced_linearized_intensity(uniform, spec=spec)
    expected = spec.equilibrium_field * (
        1.0 - (1.1 - spec.reference_intensity) / spec.reference_intensity
    )
    np.testing.assert_allclose(uniform_result.E, expected, rtol=0.0, atol=2e-16)


def test_numpy_float32_reduced_linearized_production_converges_with_own_defaults():
    request64 = _base_request(linearized=True)
    request64 = replace(
        request64,
        solver=PRStaticWorkflowOptions(max_coupled_passes=12),
    )
    request32 = replace(
        request64,
        backend=BackendSpec("numpy", "float32", False),
    )
    result64 = run_pr_static(request64)
    result32 = run_pr_static(request32)

    assert result64.converged and result32.converged
    assert result32.A_final.dtype == np.complex64
    assert result32.E_final.dtype == np.float32
    np.testing.assert_allclose(
        result32.E_final, result64.E_final, rtol=2e-5, atol=2e-6
    )
    assert result32.tolerance_provenance[
        "coupled_residual_rms_tolerance"
    ]["value"] == 5e-6


def test_reduced_linearized_requires_explicit_positive_reference_intensity():
    with pytest.raises(ValueError, match="reference_intensity"):
        PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED
        ).validate()
    with pytest.raises(ValueError, match="reference_intensity"):
        PRReducedLinearizedSpec(0.0, 0.0, 0.2, 0.1).validate()


def test_reduced_request_defaults_nonlinear_and_explicit_default_is_unchanged():
    default = _base_request()
    explicit = replace(
        default,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_NONLINEAR
        ),
    )
    default_result = run_pr_static(default)
    explicit_result = run_pr_static(explicit)

    assert default.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR
    np.testing.assert_array_equal(default_result.E_final, explicit_result.E_final)
    np.testing.assert_array_equal(default_result.A_final, explicit_result.A_final)
    assert default_result.material_response_summary["model"] == "nonlinear"


def test_old_reduced_static_positional_request_keeps_every_existing_binding():
    expected = _base_request()
    positional = PRStaticRunRequest(
        expected.grid,
        expected.beams,
        expected.material,
        expected.solver,
        expected.backend,
        expected.launch_elements,
        expected.initial_A,
        expected.initial_E,
    )

    assert positional.solver is expected.solver
    assert positional.backend is expected.backend
    assert positional.initial_A is expected.initial_A
    assert positional.initial_E is expected.initial_E
    assert positional.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR


def test_reduced_linearized_persistence_and_transport_round_trip():
    request = _base_request(linearized=True)
    persisted = replace(request, initial_A=None)
    encoded = encode_pr_static_request(persisted)
    assert encoded["material_response"] == {
        "model": "linearized",
        "reference_intensity": 21.0,
    }
    assert decode_pr_static_request(encoded) == persisted

    legacy = encode_pr_static_request(persisted)
    legacy.pop("material_response")
    assert (
        decode_pr_static_request(legacy).material_response.model
        == PR_MATERIAL_RESPONSE_NONLINEAR
    )

    portable = encode_pr_static_transport_request(request)
    decoded = decode_pr_static_transport_request(
        portable.payload.metadata, portable.payload.arrays
    )
    assert decoded.material_response == request.material_response
    np.testing.assert_array_equal(decoded.initial_A, request.initial_A)

    result = run_pr_static(request)
    encoded_result = encode_pr_static_transport_result(result)
    decoded_result = decode_pr_static_transport_result(
        encoded_result.payload.metadata, encoded_result.payload.arrays
    )
    assert decoded_result.material_response_summary == (
        result.material_response_summary
    )
    fast = encode_pr_static_transport_result(
        result, result_policy=FAST_RESULT_POLICY
    )
    fast_result = decode_pr_static_transport_result(
        fast.payload.metadata, fast.payload.arrays
    )
    run_data = pr_static_result_to_run_data(fast_result)
    assert run_data.diagnostics["summary"].values[
        "material_response"
    ] == result.material_response_summary


def test_reduced_linearized_gui_selection_and_experimental_status(app):
    from lcprop.pr.gui.evolution_panel import PREvolutionPanel
    from lcprop.pr.gui.main_window import PRMainWindow

    panel = PREvolutionPanel()
    panel.set_workflow_id("pr_static")
    assert not panel.material_response.isHidden()
    linearized_index = panel.material_response.findData(
        PR_MATERIAL_RESPONSE_LINEARIZED
    )
    panel.material_response.setCurrentIndex(linearized_index)
    panel.reference_intensity.setValue(3.25)

    assert panel.material_response.isEnabled()
    assert not panel.reference_intensity.isHidden()
    assert panel.transverse_applied_field.isHidden()
    assert "Experimental" in panel.material_response.currentText()
    assert panel.transverse_material_response() == PRTransverseMaterialResponseSpec(
        model=PR_MATERIAL_RESPONSE_LINEARIZED,
        reference_intensity=3.25,
    )
    panel.set_image_amplification_mode(True)
    assert panel.image_amplification_validation_status() == (
        "compatible_validation_pending"
    )
    assert "Experimental" in panel.algorithm_status.text()

    window = PRMainWindow()
    window.evolution_panel.set_workflow_id("pr_static")
    response_index = window.evolution_panel.material_response.findData(
        PR_MATERIAL_RESPONSE_LINEARIZED
    )
    window.evolution_panel.material_response.setCurrentIndex(response_index)
    window.evolution_panel.reference_intensity.setValue(3.25)
    request = window.build_request()
    summary = window.describe_request(request)
    assert request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
    assert request.material_response.reference_intensity == 3.25
    assert "Reduced x-only PR transport" in summary
    assert "Linearized material response [Experimental]" in summary
    window.close()


def test_reduced_linearized_cost_is_lower_than_reduced_nonlinear():
    nonlinear = _base_request()
    linearized = replace(
        nonlinear,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=21.0,
        ),
    )
    nonlinear_cost = classify_pr_run_cost(nonlinear, execution_target="local")
    linearized_cost = classify_pr_run_cost(linearized, execution_target="local")

    assert linearized_cost.work_score < nonlinear_cost.work_score
    assert "Linearized" in linearized_cost.model_label
    assert "one-dimensional FFT" in linearized_cost.rationale


def test_pre_cancelled_reduced_linearized_run_stops_at_outer_safe_boundary():
    token = CancellationToken()
    token.cancel()
    result = run_pr_static(
        _base_request(linearized=True), cancellation_token=token
    )

    assert result.status == "cancelled"
    assert result.completed_slices == 0
    assert result.material_response_summary["material_response_calls"] == 0


def test_weak_and_strong_reduced_nonlinear_vs_linearized_production_comparison():
    weak_nonlinear = run_pr_static(_base_request(background=20.0))
    weak_linearized = run_pr_static(
        _base_request(background=20.0, linearized=True)
    )
    strong_nonlinear = run_pr_static(
        _base_request(background=0.2, gain_length_product=0.08)
    )
    strong_linearized = run_pr_static(
        _base_request(
            background=0.2,
            gain_length_product=0.08,
            linearized=True,
        )
    )

    assert weak_nonlinear.converged and weak_linearized.converged
    assert strong_nonlinear.converged and strong_linearized.converged
    weak_material = _relative_l2(
        weak_linearized.E_final, weak_nonlinear.E_final
    )
    weak_optical = _relative_l2(
        weak_linearized.A_final, weak_nonlinear.A_final
    )
    strong_material = _relative_l2(
        strong_linearized.E_final, strong_nonlinear.E_final
    )
    strong_optical = _relative_l2(
        strong_linearized.A_final, strong_nonlinear.A_final
    )
    assert weak_material < 4e-3
    assert weak_optical < 2e-4
    assert strong_material > 5.0 * weak_material
    assert strong_optical > 5.0 * weak_optical
    assert weak_linearized.material_response_summary["solver"] == (
        "analytic_centered_difference_fourier"
    )
    assert weak_linearized.tolerance_provenance["material_solver"] == {
        "source": "not_applicable_direct_linearized_solve"
    }
    assert abs(
        weak_linearized.power_final / weak_linearized.power_initial - 1.0
    ) < 5e-14
    assert abs(
        strong_linearized.power_final / strong_linearized.power_initial - 1.0
    ) < 5e-14


def test_y_independent_reduced_limit_has_expected_discretization_difference():
    nx, ny = 257, 7
    dx = 2.0 * math.pi / nx
    x = np.arange(nx)[:, None] * dx
    intensity = np.repeat(0.8 + 1e-4 * np.cos(3.0 * x), ny, axis=1)
    reduced = solve_pr_reduced_linearized_intensity(
        intensity,
        spec=PRReducedLinearizedSpec(0.8, -0.6, 0.8, dx),
    )
    full = solve_pr_biased_linearized_reference(
        intensity,
        spec=PRBiasedLinearizedReferenceSpec(
            reference_intensity=0.8,
            applied_field=-0.6,
            dx_normalized=dx,
            dy_normalized=1.0,
            m_y=1.0,
            h_y=1.0,
        ),
    )

    error = _relative_l2(reduced.delta_E, full.delta_E_x)
    assert error < 8e-4
    assert np.max(np.abs(full.delta_E_y)) < 1e-18


def test_y_independent_reduced_full_difference_converges_at_second_order():
    domain_length = 9.6
    mode = 2
    reference = 100.0
    amplitude = 0.01
    errors = []

    for nx in (48, 96, 192, 384):
        dx = domain_length / nx
        phase = 2.0 * math.pi * mode / nx
        x = np.arange(nx)[:, None]
        intensity = np.repeat(
            reference + amplitude * np.cos(phase * x), 3, axis=1
        )
        reduced = solve_pr_reduced_linearized_intensity(
            intensity,
            spec=PRReducedLinearizedSpec(reference, 0.0, reference, dx),
        )
        full = solve_pr_biased_linearized_reference(
            intensity,
            spec=PRBiasedLinearizedReferenceSpec(
                reference_intensity=reference,
                applied_field=0.0,
                dx_normalized=dx,
                dy_normalized=1.0,
            ),
        )
        error = _relative_l2(reduced.delta_E, full.delta_E_x)
        errors.append(error)

        k = phase / dx
        k1 = math.sin(phase) / dx
        k2_squared = 4.0 * math.sin(0.5 * phase) ** 2 / dx**2
        reduced_response = k1 / (1.0 + k2_squared)
        spectral_response = k / (1.0 + k**2)
        predicted = abs(reduced_response - spectral_response) / abs(
            spectral_response
        )
        assert error == pytest.approx(predicted, rel=2e-10, abs=2e-13)

    np.testing.assert_allclose(
        errors,
        (
            7.81373645361322e-3,
            1.95394797728831e-3,
            4.88518886302556e-4,
            1.22131711320854e-4,
        ),
        rtol=2e-10,
        atol=2e-13,
    )
    ratios = np.asarray(errors[:-1]) / np.asarray(errors[1:])
    assert np.all((ratios > 3.99) & (ratios < 4.01))


def test_y_independent_production_comparison_enforces_i0_equals_ib():
    reduced_request = _base_request(
        nx=48,
        ny=9,
        background=100.0,
        applied_field=0.0,
        gain_length_product=0.02,
        y_variation=False,
        linearized=True,
    )
    reduced_request = replace(
        reduced_request,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=reduced_request.material.background_intensity,
        ),
    )
    assert (
        reduced_request.material_response.reference_intensity
        == reduced_request.material.background_intensity
    )
    full_request = _full_linearized_request(reduced_request)
    reduced = run_pr_static(reduced_request)
    full = run_pr_transverse_static(full_request)
    dx_normalized = (
        full_request.material.characteristic_wavenumber_per_um
        * full_request.grid.x_aperture_um
        / full_request.grid.Nx
    )
    dy_normalized = (
        full_request.material.characteristic_wavenumber_per_um
        * full_request.grid.y_aperture_um
        / full_request.grid.Ny
    )
    full_state = state_from_potential(
        full.psi_final,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        applied_field_x=0.0,
    )

    assert reduced.converged and full.converged
    assert _relative_l2(reduced.E_final, full_state.E_x) < 0.025
    assert _relative_l2(reduced.A_final, full.A_final) < 2e-5


def test_genuine_2d_reduced_vs_full_linearized_changes_material_and_optics():
    reduced_request = _base_request(
        background=20.0,
        applied_field=0.0,
        y_variation=True,
        linearized=True,
    )
    full_request = _full_linearized_request(reduced_request)
    reduced = run_pr_static(reduced_request)
    full = run_pr_transverse_static(full_request)

    assert reduced.converged and full.converged
    dx_normalized = (
        full_request.material.characteristic_wavenumber_per_um
        * full_request.grid.x_aperture_um
        / full_request.grid.Nx
    )
    dy_normalized = (
        full_request.material.characteristic_wavenumber_per_um
        * full_request.grid.y_aperture_um
        / full_request.grid.Ny
    )
    full_state = state_from_potential(
        full.psi_final,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        applied_field_x=full_request.boundary.applied_field_x,
    )
    material_error = _relative_l2(reduced.E_final, full_state.E_x)
    optical_error = _relative_l2(reduced.A_final, full.A_final)
    intensity_error = _relative_l2(
        np.sum(np.abs(reduced.A_final) ** 2, axis=0),
        np.sum(np.abs(full.A_final) ** 2, axis=0),
    )
    far_field_error = _relative_l2(
        np.fft.fft2(reduced.A_final, axes=(-2, -1)),
        np.fft.fft2(full.A_final, axes=(-2, -1)),
    )
    assert material_error > 1e-5
    assert optical_error > 1e-8
    assert intensity_error > 1e-10
    assert far_field_error > 1e-8


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_reduced_linearized_cupy_parity_when_gpu_is_available(precision):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
        cp.zeros(1)
    except Exception as exc:
        pytest.skip(f"CuPy GPU unavailable: {exc}")
    real_dtype = np.float32 if precision == "float32" else np.float64
    intensity = np.full((17, 5), 1.0, dtype=real_dtype)
    intensity += (
        0.02 * np.cos(2.0 * math.pi * np.arange(17)[:, None] / 17)
    ).astype(real_dtype)
    spec = PRReducedLinearizedSpec(1.0, 0.3, 0.8, 0.2)
    expected = solve_pr_reduced_linearized_intensity(
        intensity,
        spec=spec,
        backend=BackendSpec("numpy", precision, False),
    )
    actual = solve_pr_reduced_linearized_intensity(
        cp.asarray(intensity),
        spec=spec,
        backend=BackendSpec("cupy", precision, False),
    )
    tolerance = 2e-6 if precision == "float32" else 2e-13
    np.testing.assert_allclose(
        cp.asnumpy(actual.E), expected.E, rtol=tolerance, atol=tolerance / 10
    )

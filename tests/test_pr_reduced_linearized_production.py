"""Retained uniform-reference research oracles; no fixed-I0 production dispatch."""
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
            BeamChannel(wavelength_um=0.633, w1_um=18.0, w2_um=15.0),
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

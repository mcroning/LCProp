from __future__ import annotations

import math

import numpy as np
import pytest

from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.transverse_reference import (
    BATIO3_CLAMPED_EPSILON_A,
    BATIO3_CLAMPED_EPSILON_C,
    RotatedUniaxialDielectric,
    TransverseReferenceOptions,
    active_field_from_components,
    potential_rhs,
    run_transverse_reference,
    state_from_potential,
)


@pytest.mark.parametrize(
    ("angle_deg", "expected_epsilon_xx", "expected_h_y"),
    [
        (
            0.0,
            BATIO3_CLAMPED_EPSILON_C,
            BATIO3_CLAMPED_EPSILON_A / BATIO3_CLAMPED_EPSILON_C,
        ),
        (
            45.0,
            (BATIO3_CLAMPED_EPSILON_A + BATIO3_CLAMPED_EPSILON_C) / 2.0,
            BATIO3_CLAMPED_EPSILON_A
            / ((BATIO3_CLAMPED_EPSILON_A + BATIO3_CLAMPED_EPSILON_C) / 2.0),
        ),
        (90.0, BATIO3_CLAMPED_EPSILON_A, 1.0),
    ],
)
def test_rotated_barium_titanate_dielectric_limits(
    angle_deg,
    expected_epsilon_xx,
    expected_h_y,
):
    dielectric = RotatedUniaxialDielectric(
        c_axis_xz_angle_deg=angle_deg,
    )

    assert dielectric.epsilon_xx == pytest.approx(expected_epsilon_xx)
    assert dielectric.epsilon_yy == pytest.approx(BATIO3_CLAMPED_EPSILON_A)
    assert dielectric.h_y == pytest.approx(expected_h_y)


def test_crystal_axis_constructor_exposes_derived_dielectric_and_preserves_default():
    isotropic = TransverseReferenceOptions(1.0, 1.0, 0.01, 1)
    anisotropic = TransverseReferenceOptions.from_barium_titanate_c_axis(
        dx_normalized=1.0,
        dy_normalized=1.0,
        dt_normalized=0.01,
        steps=1,
        c_axis_xz_angle_deg=45.0,
    )

    assert isotropic.h_y == 1.0
    assert isotropic.dielectric is None
    assert anisotropic.dielectric is not None
    assert anisotropic.h_y == pytest.approx(2200.0 / 1128.0)
    assert anisotropic.dielectric.epsilon_xx == pytest.approx(1128.0)


def _spectral_x(field: np.ndarray, dx: float) -> np.ndarray:
    kx = 2.0 * math.pi * np.fft.fftfreq(field.shape[0], d=dx)[:, None]
    return np.fft.ifft(1j * kx * np.fft.fft(field, axis=0), axis=0).real


def _spectral_xx(field: np.ndarray, dx: float) -> np.ndarray:
    kx = 2.0 * math.pi * np.fft.fftfreq(field.shape[0], d=dx)[:, None]
    return np.fft.ifft(-(kx * kx) * np.fft.fft(field, axis=0), axis=0).real


def _y_uniform_reduction_error(Nx: int) -> tuple[float, float]:
    length = 2.0 * math.pi
    dx = length / Nx
    x = np.arange(Nx) * dx
    psi_x = 0.03 * np.cos(x) + 0.01 * np.cos(2.0 * x)
    intensity_x = 1.0 + 0.2 * np.cos(x) + 0.05 * np.cos(3.0 * x)
    psi = np.repeat(psi_x[:, None], 5, axis=1)
    intensity = np.repeat(intensity_x[:, None], 5, axis=1)
    options = TransverseReferenceOptions(
        dx_normalized=dx,
        dy_normalized=1.0,
        dt_normalized=1.0e-4,
        steps=1,
    )

    state = state_from_potential(psi, options=options)
    psi_tau = potential_rhs(psi, intensity, options=options)
    transverse_E_tau = -_spectral_x(psi_tau, dx)
    spectral_a7 = (
        -(state.E_x * intensity - _spectral_x(intensity, dx))
        * (1.0 + _spectral_x(state.E_x, dx))
        + intensity * _spectral_xx(state.E_x, dx)
    )
    exact_relative = np.linalg.norm(transverse_E_tau - spectral_a7) / np.linalg.norm(
        spectral_a7
    )

    production_a7 = hopping_rhs(
        state.E_x,
        intensity,
        applied_field=0.0,
        background_intensity=0.0,
        dx_normalized=dx,
        xp=np,
    )
    production_relative = np.linalg.norm(production_a7 - transverse_E_tau) / np.linalg.norm(
        transverse_E_tau
    )
    return float(exact_relative), float(production_relative)


def _localized_case():
    Nx, Ny = 48, 40
    dx = dy = 0.5
    x = (np.arange(Nx) - Nx / 2.0) * dx
    y = (np.arange(Ny) - Ny / 2.0) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    intensity = 0.1 + np.exp(-((X - 1.2) ** 2 / 10.0 + (Y + 1.7) ** 2 / 3.0))
    options = TransverseReferenceOptions(
        dx_normalized=dx,
        dy_normalized=dy,
        dt_normalized=0.002,
        steps=50,
    )
    result = run_transverse_reference(
        np.zeros_like(intensity),
        intensity,
        options=options,
    )
    return intensity, options, result


def test_y_uniform_transverse_equation_reduces_to_a7_and_centered_path_converges():
    exact_errors = []
    production_errors = []
    for Nx in (32, 64, 128, 256):
        exact, production = _y_uniform_reduction_error(Nx)
        exact_errors.append(exact)
        production_errors.append(production)

    assert max(exact_errors) < 1.0e-11
    assert all(
        fine < coarse / 3.8
        for coarse, fine in zip(production_errors, production_errors[1:])
    )
    assert production_errors[-1] < 5.0e-4


def test_periodic_transverse_reference_conserves_carrier_number():
    _, _, result = _localized_case()

    assert result.diagnostics.carrier_relative_drift < 5.0e-15
    assert np.allclose(
        result.diagnostics.carrier_integrals,
        result.diagnostics.carrier_integrals[0],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_transverse_reference_satisfies_curl_and_gauss_closure():
    _, _, result = _localized_case()

    assert result.diagnostics.curl_rms < 1.0e-15
    assert result.diagnostics.curl_max < 1.0e-14
    assert result.diagnostics.gauss_rms < 1.0e-15
    assert result.diagnostics.gauss_max < 1.0e-14


def test_localized_forcing_generates_genuine_y_transport_not_independent_a7_lines():
    intensity, options, result = _localized_case()
    final = result.final_state

    reduced_E = np.zeros_like(intensity)
    for _ in range(options.steps):
        reduced_E += options.dt_normalized * hopping_rhs(
            reduced_E,
            intensity,
            applied_field=0.0,
            background_intensity=0.1,
            dx_normalized=options.dx_normalized,
            xp=np,
        )

    E_y_rms = float(np.sqrt(np.mean(final.E_y * final.E_y)))
    carrier_rms = float(
        np.sqrt(np.mean((final.carrier_density - 1.0) ** 2))
    )
    independent_line_difference = float(
        np.linalg.norm(final.E_x - reduced_E) / np.linalg.norm(final.E_x)
    )
    assert E_y_rms > 5.0e-3
    assert carrier_rms > 5.0e-3
    assert independent_line_difference > 1.0e-2


def test_rotated_dielectric_changes_localized_transverse_electrostatics():
    intensity, isotropic_options, isotropic = _localized_case()
    anisotropic_options = TransverseReferenceOptions.from_barium_titanate_c_axis(
        dx_normalized=isotropic_options.dx_normalized,
        dy_normalized=isotropic_options.dy_normalized,
        dt_normalized=isotropic_options.dt_normalized,
        steps=isotropic_options.steps,
        c_axis_xz_angle_deg=45.0,
        m_y=isotropic_options.m_y,
        applied_field_x=isotropic_options.applied_field_x,
    )
    anisotropic = run_transverse_reference(
        np.zeros_like(intensity),
        intensity,
        options=anisotropic_options,
    )

    relative_E_x = np.linalg.norm(
        anisotropic.final_state.E_x - isotropic.final_state.E_x
    ) / np.linalg.norm(isotropic.final_state.E_x)
    relative_E_y = np.linalg.norm(
        anisotropic.final_state.E_y - isotropic.final_state.E_y
    ) / np.linalg.norm(isotropic.final_state.E_y)

    assert relative_E_x > 1.0e-3
    assert relative_E_y > 1.0e-3


def test_active_field_projection_is_explicit_and_defaults_to_x():
    E_x = np.arange(15, dtype=float).reshape(3, 5)
    E_y = np.flip(E_x, axis=1)

    assert np.array_equal(active_field_from_components(E_x, E_y), E_x)
    assert np.allclose(
        active_field_from_components(E_x, E_y, x_weight=0.25, y_weight=-0.5),
        0.25 * E_x - 0.5 * E_y,
    )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        (np.zeros((2, 4)), "both sizes at least 3"),
        (np.zeros((4, 4, 1)), r"shape \(Nx, Ny\)"),
    ],
)
def test_reference_rejects_invalid_state_shapes(field, match):
    options = TransverseReferenceOptions(1.0, 1.0, 0.01, 1)
    with pytest.raises(ValueError, match=match):
        state_from_potential(field, options=options)

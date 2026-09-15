from __future__ import annotations

import math

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.pr.evolution import hopping_rhs
from lcprop.pr.reduced_linearized import (
    PRReducedLinearizedSpec,
    solve_pr_reduced_linearized_intensity,
)
from lcprop.pr.reduced_linearized_timedependent import (
    reduced_linearized_timedependent_coefficients,
    reduced_linearized_timedependent_rhs,
    solve_pr_reduced_linearized_timedependent,
)


def _backend(precision: str = "float64", backend: str = "numpy") -> BackendSpec:
    return BackendSpec(backend=backend, precision=precision, verbose=False)


def _spec(nx: int, *, bias: float = 0.7) -> PRReducedLinearizedSpec:
    return PRReducedLinearizedSpec(
        reference_intensity=1.4,
        applied_field=bias,
        background_intensity=0.8,
        dx_normalized=2.0 * math.pi / nx,
    )


def _fields(nx: int, ny: int, dtype):
    x = np.arange(nx, dtype=np.float64)[:, None] * 2.0 * math.pi / nx
    y = np.arange(ny, dtype=np.float64)[None, :] * 2.0 * math.pi / ny
    intensity = 1.4 + 0.08 * np.cos(2.0 * x) + 0.05 * np.sin(3.0 * x + y)
    delta_E = 0.03 * np.sin(x) - 0.02 * np.cos(2.0 * x - y)
    return intensity.astype(dtype), delta_E.astype(dtype)


def test_discrete_modal_coefficients_and_static_limit_are_exact():
    nx, ny = 31, 7
    spec = _spec(nx)
    intensity, delta_E = _fields(nx, ny, np.float64)
    state = delta_E + spec.equilibrium_field
    decay, source = reduced_linearized_timedependent_coefficients(nx, spec=spec)
    phase = 2.0 * math.pi * np.fft.fftfreq(nx)
    k1 = np.sin(phase) / spec.dx_normalized
    k2_squared = 4.0 * np.sin(phase / 2.0) ** 2 / spec.dx_normalized**2
    expected_decay = spec.reference_intensity * (
        1.0 + k2_squared + 1j * spec.equilibrium_field * k1
    )
    expected_source = -spec.equilibrium_field + 1j * k1
    np.testing.assert_allclose(decay, expected_decay, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(source, expected_source, rtol=2e-15, atol=2e-15)
    assert np.all(decay.real > 0.0)

    static = solve_pr_reduced_linearized_intensity(intensity, spec=spec)
    equilibrium = solve_pr_reduced_linearized_timedependent(
        static.E, intensity, dt_normalized=0.37, spec=spec
    )
    np.testing.assert_allclose(equilibrium.E, static.E, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(
        equilibrium.equilibrium_E, static.E, rtol=0.0, atol=2e-16
    )
    np.testing.assert_allclose(
        source / decay, static.response_kernel, rtol=3e-16, atol=0.0
    )
    assert np.max(np.abs(equilibrium.rhs)) < 2e-14

    direct = solve_pr_reduced_linearized_timedependent(
        state, intensity, dt_normalized=0.29, spec=spec
    )
    expected_hat = np.fft.fft(static.delta_E, axis=-2) + (
        np.fft.fft(delta_E, axis=-2) - np.fft.fft(static.delta_E, axis=-2)
    ) * np.exp(-decay[:, None] * 0.29)
    np.testing.assert_allclose(
        np.fft.fft(direct.delta_E, axis=-2),
        expected_hat,
        rtol=3e-14,
        atol=2e-14,
    )


def test_zero_arbitrary_and_equilibrium_initial_states_follow_exact_update():
    nx, ny = 27, 4
    spec = _spec(nx, bias=0.5)
    uniform = np.full((nx, ny), spec.reference_intensity)
    zero = solve_pr_reduced_linearized_timedependent(
        np.zeros_like(uniform), uniform, dt_normalized=0.23, spec=spec
    )
    expected_zero = spec.equilibrium_field * (1.0 - np.exp(-1.4 * 0.23))
    np.testing.assert_allclose(zero.E, expected_zero, rtol=0.0, atol=2e-16)

    intensity, delta_E = _fields(nx, ny, np.float64)
    arbitrary = solve_pr_reduced_linearized_timedependent(
        delta_E + spec.equilibrium_field,
        intensity,
        dt_normalized=0.17,
        spec=spec,
    )
    assert not np.array_equal(arbitrary.E, delta_E + spec.equilibrium_field)
    equilibrium = solve_pr_reduced_linearized_intensity(intensity, spec=spec)
    stationary = solve_pr_reduced_linearized_timedependent(
        equilibrium.E, intensity, dt_normalized=0.17, spec=spec
    )
    np.testing.assert_allclose(stationary.E, equilibrium.E, rtol=0.0, atol=2e-16)


def test_nonlinear_rhs_is_an_independent_second_order_taylor_oracle():
    nx, ny = 47, 5
    spec = _spec(nx, bias=0.55)
    intensity, delta_E = _fields(nx, ny, np.float64)
    delta_I = intensity - spec.reference_intensity
    baseline_E = np.full((nx, ny), spec.equilibrium_field)
    baseline_I = np.full((nx, ny), spec.reference_intensity)
    F0 = hopping_rhs(
        baseline_E,
        baseline_I,
        applied_field=spec.applied_field,
        background_intensity=spec.background_intensity,
        dx_normalized=spec.dx_normalized,
        xp=np,
    )
    linear = reduced_linearized_timedependent_rhs(
        baseline_E + delta_E,
        baseline_I + delta_I,
        spec=spec,
    )
    epsilons = np.asarray([2e-2, 1e-2, 5e-3, 2.5e-3])
    errors = []
    for epsilon in epsilons:
        nonlinear = hopping_rhs(
            baseline_E + epsilon * delta_E,
            baseline_I + epsilon * delta_I,
            applied_field=spec.applied_field,
            background_intensity=spec.background_intensity,
            dx_normalized=spec.dx_normalized,
            xp=np,
        )
        errors.append(np.linalg.norm(nonlinear - F0 - epsilon * linear))
    order = np.polyfit(np.log(epsilons), np.log(errors), 1)[0]
    assert order == pytest.approx(2.0, abs=0.03)


def test_bias_reversal_conjugates_cosine_mode_and_reverses_drift():
    nx, ny = 33, 4
    plus_spec = _spec(nx, bias=0.75)
    minus_spec = _spec(nx, bias=-0.75)
    x = np.arange(nx)[:, None] * 2.0 * math.pi / nx
    intensity = np.broadcast_to(1.4 + 0.06 * np.cos(3.0 * x), (nx, ny)).copy()
    plus_decay, plus_source = reduced_linearized_timedependent_coefficients(
        nx, spec=plus_spec
    )
    minus_decay, minus_source = reduced_linearized_timedependent_coefficients(
        nx, spec=minus_spec
    )
    np.testing.assert_array_equal(minus_decay, plus_decay.conjugate())
    np.testing.assert_array_equal(minus_source, -plus_source.conjugate())
    assert plus_decay[3].imag == pytest.approx(-minus_decay[3].imag)
    assert plus_decay[3].real == pytest.approx(minus_decay[3].real)
    plus = solve_pr_reduced_linearized_timedependent(
        np.full_like(intensity, plus_spec.equilibrium_field),
        intensity,
        dt_normalized=0.31,
        spec=plus_spec,
    )
    minus = solve_pr_reduced_linearized_timedependent(
        np.full_like(intensity, minus_spec.equilibrium_field),
        intensity,
        dt_normalized=0.31,
        spec=minus_spec,
    )
    plus_hat = np.fft.fft(plus.delta_E, axis=-2)
    minus_hat = np.fft.fft(minus.delta_E, axis=-2)
    np.testing.assert_allclose(
        minus_hat[3], -plus_hat[3].conjugate(), rtol=2e-14, atol=2e-16
    )
    reflected_plus = np.concatenate((plus.E[:1], plus.E[:0:-1]), axis=0)
    np.testing.assert_allclose(minus.E, -reflected_plus, rtol=0.0, atol=3e-16)


def test_unbiased_resolved_modes_have_real_decay_and_derivative_source():
    nx, ny = 35, 3
    spec = _spec(nx, bias=0.0)
    decay, source = reduced_linearized_timedependent_coefficients(nx, spec=spec)
    np.testing.assert_array_equal(decay.imag, np.zeros(nx))
    np.testing.assert_array_equal(source.real, np.zeros(nx))
    x = np.arange(nx)[:, None] * 2.0 * math.pi / nx
    intensity = np.broadcast_to(1.4 + 0.05 * np.cos(4.0 * x), (nx, ny)).copy()
    result = solve_pr_reduced_linearized_timedependent(
        np.zeros_like(intensity), intensity, dt_normalized=0.25, spec=spec
    )
    coefficient = np.fft.fft(result.E, axis=-2)[4, 0]
    assert abs(coefficient.real) < 2e-15
    assert coefficient.imag != 0.0


@pytest.mark.parametrize("nx", [31, 32])
def test_constant_and_centered_difference_nyquist_modes_are_finite(nx):
    ny = 3
    spec = _spec(nx, bias=0.6)
    decay, source = reduced_linearized_timedependent_coefficients(nx, spec=spec)
    assert decay[0] == pytest.approx(spec.reference_intensity)
    assert source[0] == pytest.approx(-spec.equilibrium_field)
    assert np.isfinite(decay).all()
    assert np.isfinite(source).all()
    if nx % 2 == 0:
        nyquist = nx // 2
        assert source[nyquist].imag == 0.0
        assert decay[nyquist].imag == 0.0
        assert decay[nyquist].real > decay[0].real
        alternating = (-1.0) ** np.arange(nx)[:, None]
        intensity = np.broadcast_to(1.4 + 0.04 * alternating, (nx, ny)).copy()
        biased = solve_pr_reduced_linearized_timedependent(
            np.full_like(intensity, spec.equilibrium_field),
            intensity,
            dt_normalized=0.2,
            spec=spec,
        )
        assert np.isfinite(biased.E).all()
        unbiased_spec = _spec(nx, bias=0.0)
        unbiased = solve_pr_reduced_linearized_timedependent(
            np.zeros_like(intensity),
            intensity,
            dt_normalized=0.2,
            spec=unbiased_spec,
        )
        np.testing.assert_array_equal(unbiased.E, np.zeros_like(intensity))


@pytest.mark.parametrize(
    ("precision", "dtype", "complex_dtype", "tolerance"),
    [
        ("float64", np.float64, np.complex128, 3e-14),
        ("float32", np.float32, np.complex64, 3e-6),
    ],
)
def test_numpy_precision_and_batch_independence(
    precision, dtype, complex_dtype, tolerance
):
    nx, ny = 29, 5
    spec = _spec(nx, bias=-0.45)
    intensity, delta_E = _fields(nx, ny, dtype)
    states = np.stack(
        [delta_E + spec.equilibrium_field, 0.5 * delta_E + spec.equilibrium_field]
    ).astype(dtype)
    sources = np.stack([intensity, intensity + dtype(0.01)]).astype(dtype)
    result = solve_pr_reduced_linearized_timedependent(
        states,
        sources,
        dt_normalized=0.24,
        spec=spec,
        backend=_backend(precision),
    )
    assert result.E.dtype == dtype
    assert result.delta_E.dtype == dtype
    assert result.rhs.dtype == dtype
    assert result.decay_rate.dtype == complex_dtype
    assert result.source_coefficient.dtype == complex_dtype
    for batch in range(2):
        single = solve_pr_reduced_linearized_timedependent(
            states[batch],
            sources[batch],
            dt_normalized=0.24,
            spec=spec,
            backend=_backend(precision),
        )
        np.testing.assert_allclose(
            result.E[batch], single.E, rtol=0.0, atol=tolerance
        )


def test_float32_resolved_multimode_result_agrees_with_float64():
    nx, ny = 37, 6
    spec = _spec(nx, bias=0.65)
    intensity64, delta_E64 = _fields(nx, ny, np.float64)
    result64 = solve_pr_reduced_linearized_timedependent(
        delta_E64 + spec.equilibrium_field,
        intensity64,
        dt_normalized=0.33,
        spec=spec,
        backend=_backend("float64"),
    )
    result32 = solve_pr_reduced_linearized_timedependent(
        (delta_E64 + spec.equilibrium_field).astype(np.float32),
        intensity64.astype(np.float32),
        dt_normalized=0.33,
        spec=spec,
        backend=_backend("float32"),
    )
    assert result32.E.dtype == np.float32
    assert result32.decay_rate.dtype == np.complex64
    np.testing.assert_allclose(result32.E, result64.E, rtol=3e-6, atol=3e-7)


def test_cupy_float64_is_conditional_and_uses_the_real_backend():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("CuPy GPU is unavailable")
    nx, ny = 17, 3
    spec = _spec(nx)
    intensity, delta_E = _fields(nx, ny, np.float64)
    result = solve_pr_reduced_linearized_timedependent(
        cp.asarray(delta_E + spec.equilibrium_field),
        cp.asarray(intensity),
        dt_normalized=0.1,
        spec=spec,
        backend=_backend("float64", "cupy"),
    )
    assert isinstance(result.E, cp.ndarray)
    assert result.resolved_backend == "cupy"
    assert result.is_gpu is True

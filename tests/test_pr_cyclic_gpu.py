import numpy as np
import pytest

from lcprop.pr.cyclic import solve_cyclic_tridiagonal_rows
from lcprop.pr.cyclic_gpu import solve_cyclic_tridiagonal_rows_gpu
from lcprop.pr.evolution import solve_periodic_variable_diffusion
from lcprop.pr.evolution import semi_implicit_trapezoidal_step


def _cupy_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
        cp.cuda.Device().compute_capability
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")
    return cp


def _residual(lower, diagonal, upper, solution, rhs):
    return (
        diagonal * solution
        + lower * np.roll(solution, 1, axis=-2)
        + upper * np.roll(solution, -1, axis=-2)
        - rhs
    )


@pytest.mark.parametrize("shape", ((3, 2), (19, 4), (2, 31, 5)))
@pytest.mark.parametrize(
    ("dtype", "rtol", "residual_tolerance"),
    ((np.float32, 3e-5, 2e-5), (np.float64, 4e-13, 2e-13)),
)
def test_gpu_cyclic_matches_numpy_variable_coefficient_reference(
    shape, dtype, rtol, residual_tolerance
):
    cp = _cupy_device()
    rng = np.random.default_rng(20260817 + shape[-2])
    lower = rng.uniform(-0.25, -0.01, size=shape).astype(dtype)
    upper = rng.uniform(-0.25, -0.01, size=shape).astype(dtype)
    diagonal = (1.5 + np.abs(lower) + np.abs(upper)).astype(dtype)
    rhs = rng.normal(size=shape).astype(dtype)
    expected = solve_cyclic_tridiagonal_rows(
        lower, diagonal, upper, rhs, xp=np
    )

    actual = cp.asnumpy(
        solve_cyclic_tridiagonal_rows_gpu(
            cp.asarray(lower),
            cp.asarray(diagonal),
            cp.asarray(upper),
            cp.asarray(rhs),
        )
    )
    residual = _residual(lower, diagonal, upper, actual, rhs)

    assert actual.dtype == dtype
    assert np.allclose(actual, expected, rtol=rtol, atol=rtol)
    assert np.sqrt(np.mean(residual * residual)) < residual_tolerance
    assert np.max(np.abs(residual)) < 4.0 * residual_tolerance


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
@pytest.mark.parametrize("intensity_kind", ("smooth", "sharp"))
def test_gpu_pr_diffusion_dispatch_matches_numpy(dtype, intensity_kind):
    cp = _cupy_device()
    rng = np.random.default_rng(818)
    shape = (3, 32, 7)
    rhs = rng.normal(scale=0.2, size=shape).astype(dtype)
    if intensity_kind == "smooth":
        x = np.linspace(0.0, 2.0 * np.pi, shape[-2], endpoint=False)
        intensity = np.broadcast_to(
            (0.4 + 1.6 * np.sin(x) ** 2)[None, :, None], shape
        ).copy().astype(dtype)
    else:
        intensity = np.full(shape, 0.01, dtype=dtype)
        intensity[:, 9:15, :] = 4.0

    expected = solve_periodic_variable_diffusion(
        rhs,
        intensity,
        alpha=0.17,
        dx_normalized=0.31,
        xp=np,
    )
    actual = solve_periodic_variable_diffusion(
        cp.asarray(rhs),
        cp.asarray(intensity),
        alpha=0.17,
        dx_normalized=0.31,
        xp=cp,
    )
    tolerance = 4e-5 if dtype == np.float32 else 5e-13
    assert np.allclose(
        cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance
    )


def test_gpu_cyclic_rejects_nonfinite_and_singular_inputs():
    cp = _cupy_device()
    shape = (8, 3)
    lower = cp.full(shape, -0.1, dtype=cp.float64)
    diagonal = cp.full(shape, 1.2, dtype=cp.float64)
    upper = cp.full(shape, -0.1, dtype=cp.float64)
    rhs = cp.ones(shape, dtype=cp.float64)

    bad_lower = lower.copy()
    bad_lower[2, 1] = cp.nan
    with pytest.raises(ValueError, match="lower must contain only finite"):
        solve_cyclic_tridiagonal_rows_gpu(
            bad_lower, diagonal, upper, rhs
        )

    zero_diagonal = cp.zeros_like(diagonal)
    with pytest.raises(np.linalg.LinAlgError, match="leading diagonal"):
        solve_cyclic_tridiagonal_rows_gpu(
            lower, zero_diagonal, upper, rhs
        )

    elimination_lower = cp.ones(shape, dtype=cp.float64)
    elimination_upper = cp.ones(shape, dtype=cp.float64)
    elimination_diagonal = cp.ones(shape, dtype=cp.float64)
    elimination_diagonal[1] = 0.5
    with pytest.raises(np.linalg.LinAlgError, match="zero or near-zero pivot"):
        solve_cyclic_tridiagonal_rows_gpu(
            elimination_lower,
            elimination_diagonal,
            elimination_upper,
            rhs,
        )

    # The periodic discrete Laplacian has a constant nullspace. Its Thomas
    # pivots are valid, but the final Sherman--Morrison denominator is zero.
    laplacian_lower = cp.full(shape, -1.0, dtype=cp.float64)
    laplacian_diagonal = cp.full(shape, 2.0, dtype=cp.float64)
    with pytest.raises(np.linalg.LinAlgError, match="zero or near-zero pivot"):
        solve_cyclic_tridiagonal_rows_gpu(
            laplacian_lower,
            laplacian_diagonal,
            laplacian_lower,
            rhs,
        )


def test_gpu_cyclic_requires_direct_c_contiguous_layout():
    cp = _cupy_device()
    value = cp.ones((3, 8, 4), dtype=cp.float32)[:, :, ::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        solve_cyclic_tridiagonal_rows_gpu(value, value, value, value)


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_gpu_semi_implicit_step_matches_numpy_reference(dtype):
    cp = _cupy_device()
    rng = np.random.default_rng(119)
    shape = (4, 24, 6)
    state = rng.normal(scale=0.04, size=shape).astype(dtype)
    base_intensity = (0.15 + rng.random(shape)).astype(dtype)

    def numpy_source(candidate):
        return base_intensity + np.asarray(0.03, dtype=dtype) * np.tanh(candidate)

    def cupy_source(candidate):
        return cp.asarray(base_intensity) + cp.asarray(
            0.03, dtype=dtype
        ) * cp.tanh(candidate)

    expected = semi_implicit_trapezoidal_step(
        state,
        numpy_source,
        dt_normalized=0.025,
        applied_field=0.4,
        background_intensity=0.1,
        dx_normalized=0.28,
        xp=np,
    )
    actual = semi_implicit_trapezoidal_step(
        cp.asarray(state),
        cupy_source,
        dt_normalized=0.025,
        applied_field=0.4,
        background_intensity=0.1,
        dx_normalized=0.28,
        xp=cp,
    )
    tolerance = 7e-5 if dtype == np.float32 else 8e-13
    assert np.allclose(
        cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance
    )

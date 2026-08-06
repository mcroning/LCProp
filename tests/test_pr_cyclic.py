import numpy as np
import pytest

from lcprop.pr.cyclic import solve_cyclic_tridiagonal_rows
from lcprop.pr.static import (
    batched_newton_direction,
    fixed_intensity_jacobian_rows,
)


def _dense_solve(lower, diagonal, upper, rhs):
    Nx = rhs.shape[-2]
    moved_shape = np.moveaxis(rhs, -2, -1).shape
    lower_rows = np.moveaxis(lower, -2, -1).reshape((-1, Nx))
    diagonal_rows = np.moveaxis(diagonal, -2, -1).reshape((-1, Nx))
    upper_rows = np.moveaxis(upper, -2, -1).reshape((-1, Nx))
    rhs_rows = np.moveaxis(rhs, -2, -1).reshape((-1, Nx))
    solution_rows = np.empty_like(rhs_rows)
    indices = np.arange(Nx)
    for row_index in range(rhs_rows.shape[0]):
        matrix = np.zeros((Nx, Nx), dtype=rhs.dtype)
        matrix[indices, indices] = diagonal_rows[row_index]
        matrix[indices, (indices - 1) % Nx] = lower_rows[row_index]
        matrix[indices, (indices + 1) % Nx] = upper_rows[row_index]
        solution_rows[row_index] = np.linalg.solve(
            matrix, rhs_rows[row_index]
        )
    return np.moveaxis(solution_rows.reshape(moved_shape), -1, -2)


@pytest.mark.parametrize("shape", ((19, 4), (2, 19, 4)))
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    ((np.float64, 2e-13), (np.float32, 2e-6)),
)
def test_batched_cyclic_solver_matches_dense_reference(shape, dtype, tolerance):
    rng = np.random.default_rng(20260806)
    lower = rng.uniform(-0.2, 0.2, size=shape).astype(dtype)
    upper = rng.uniform(-0.2, 0.2, size=shape).astype(dtype)
    diagonal = (
        2.0 + np.abs(lower) + np.abs(upper)
    ).astype(dtype)
    rhs = rng.normal(size=shape).astype(dtype)
    originals = tuple(array.copy() for array in (lower, diagonal, upper, rhs))

    actual = solve_cyclic_tridiagonal_rows(
        lower, diagonal, upper, rhs, xp=np
    )
    expected = _dense_solve(lower, diagonal, upper, rhs)

    assert actual.dtype == rhs.dtype
    assert np.allclose(actual, expected, rtol=tolerance, atol=tolerance)
    for array, original in zip((lower, diagonal, upper, rhs), originals):
        assert np.array_equal(array, original)


def test_batched_pr_newton_direction_matches_existing_dense_reference():
    rng = np.random.default_rng(17)
    E = rng.normal(scale=0.03, size=(2, 24, 3))
    intensity = 0.8 + rng.random(E.shape)
    residual = rng.normal(scale=0.1, size=E.shape)
    lower, diagonal, upper = fixed_intensity_jacobian_rows(
        E,
        intensity,
        dx_normalized=0.35,
    )

    actual = batched_newton_direction(
        residual,
        lower,
        diagonal,
        upper,
        xp=np,
    )
    expected = _dense_solve(lower, diagonal, upper, -residual)

    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 3e-15


def test_batched_cyclic_solver_rejects_accidental_shapes_and_singular_pivot():
    arrays = [np.ones((8, 3), dtype=np.float64) for _ in range(4)]
    lower, diagonal, upper, rhs = arrays
    with pytest.raises(ValueError, match="identical shapes"):
        solve_cyclic_tridiagonal_rows(
            lower[:-1], diagonal, upper, rhs, xp=np
        )
    with pytest.raises(ValueError, match="Nx >= 3"):
        solve_cyclic_tridiagonal_rows(
            lower[:2], diagonal[:2], upper[:2], rhs[:2], xp=np
        )

    diagonal.fill(0.0)
    with pytest.raises(np.linalg.LinAlgError, match="leading diagonal"):
        solve_cyclic_tridiagonal_rows(
            lower, diagonal, upper, rhs, xp=np
        )


def test_batched_cyclic_solver_is_cupy_compatible_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    rng = np.random.default_rng(29)
    shape = (2, 32, 5)
    lower_np = rng.uniform(-0.1, 0.1, size=shape).astype(np.float64)
    upper_np = rng.uniform(-0.1, 0.1, size=shape).astype(np.float64)
    diagonal_np = 2.0 + np.abs(lower_np) + np.abs(upper_np)
    rhs_np = rng.normal(size=shape)
    expected = _dense_solve(
        lower_np, diagonal_np, upper_np, rhs_np
    )

    actual = solve_cyclic_tridiagonal_rows(
        cp.asarray(lower_np),
        cp.asarray(diagonal_np),
        cp.asarray(upper_np),
        cp.asarray(rhs_np),
        xp=cp,
    )

    assert isinstance(actual, cp.ndarray)
    assert np.allclose(cp.asnumpy(actual), expected, rtol=2e-13, atol=2e-13)


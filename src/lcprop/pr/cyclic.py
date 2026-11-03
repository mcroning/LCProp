"""Batched periodic tridiagonal linear algebra for PR solvers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _array_has_true(value, *, xp: Any) -> bool:
    reduced = xp.any(value)
    return bool(reduced.item() if hasattr(reduced, "item") else reduced)


def _validate_inputs(lower, diagonal, upper, rhs, *, xp: Any) -> None:
    if not (
        lower.shape == diagonal.shape == upper.shape == rhs.shape
    ):
        raise ValueError(
            "lower, diagonal, upper, and rhs must have identical shapes"
        )
    if rhs.ndim not in (2, 3):
        raise ValueError(
            "cyclic systems must have shape (Nx, Ny) or (Nz, Nx, Ny)"
        )
    if int(rhs.shape[-2]) < 3:
        raise ValueError("cyclic solve requires Nx >= 3")
    arrays = {
        "lower": lower,
        "diagonal": diagonal,
        "upper": upper,
        "rhs": rhs,
    }
    for name, array in arrays.items():
        if getattr(array.dtype, "kind", None) != "f":
            raise TypeError(f"{name} must have a real floating-point dtype")
        if array.dtype != rhs.dtype:
            raise TypeError("all cyclic-system arrays must have the same dtype")
        if _array_has_true(~xp.isfinite(array), xp=xp):
            raise ValueError(f"{name} must contain only finite values")


def _pivot_floor(lower, diagonal, upper, *, xp: Any):
    scale = xp.maximum(
        xp.asarray(1.0, dtype=diagonal.dtype),
        xp.maximum(
            xp.max(xp.abs(lower), axis=1),
            xp.maximum(
                xp.max(xp.abs(diagonal), axis=1),
                xp.max(xp.abs(upper), axis=1),
            ),
        ),
    )
    return xp.asarray(
        32.0 * np.finfo(diagonal.dtype).eps,
        dtype=diagonal.dtype,
    ) * scale


def _solve_tridiagonal_rows(
    lower,
    diagonal,
    upper,
    rhs,
    *,
    pivot_floor,
    xp: Any,
):
    """Solve independent nonperiodic tridiagonal systems without pivoting."""

    _, n = rhs.shape
    cprime = xp.empty_like(rhs)
    dprime = xp.empty_like(rhs)

    denominator = diagonal[:, 0]
    invalid_pivot = xp.abs(denominator) <= pivot_floor
    cprime[:, 0] = upper[:, 0] / denominator
    dprime[:, 0] = rhs[:, 0] / denominator

    for index in range(1, n):
        denominator = (
            diagonal[:, index]
            - lower[:, index] * cprime[:, index - 1]
        )
        invalid_pivot = xp.logical_or(
            invalid_pivot,
            xp.abs(denominator) <= pivot_floor,
        )
        if index < n - 1:
            cprime[:, index] = upper[:, index] / denominator
        else:
            cprime[:, index] = 0.0
        dprime[:, index] = (
            rhs[:, index]
            - lower[:, index] * dprime[:, index - 1]
        ) / denominator

    solution = xp.empty_like(rhs)
    solution[:, -1] = dprime[:, -1]
    for index in range(n - 2, -1, -1):
        solution[:, index] = (
            dprime[:, index]
            - cprime[:, index] * solution[:, index + 1]
        )
    return solution, invalid_pivot


def solve_cyclic_tridiagonal_rows(
    lower,
    diagonal,
    upper,
    rhs,
    *,
    xp: Any,
):
    """Solve periodic tridiagonal systems along the x axis ``-2``.

    All arrays must have exactly the same ``(Nx, Ny)`` or ``(Nz, Nx, Ny)``
    shape and floating-point dtype. At x index ``j``, ``lower[j]`` multiplies
    the unknown at ``j-1`` and ``upper[j]`` multiplies the unknown at ``j+1``
    with periodic indexing. The solve is batched over every remaining z/y
    row and uses a Sherman–Morrison reduction with two Thomas solves.

    The inputs are not modified. The algorithm does not perform pivoting and
    raises ``LinAlgError`` when its elimination pivots are numerically zero.
    """

    _validate_inputs(lower, diagonal, upper, rhs, xp=xp)
    Nx = int(rhs.shape[-2])
    moved_shape = xp.moveaxis(rhs, -2, -1).shape
    lower_rows = xp.moveaxis(lower, -2, -1).reshape((-1, Nx))
    diagonal_rows = xp.moveaxis(diagonal, -2, -1).reshape((-1, Nx))
    upper_rows = xp.moveaxis(upper, -2, -1).reshape((-1, Nx))
    rhs_rows = xp.moveaxis(rhs, -2, -1).reshape((-1, Nx))
    floor = _pivot_floor(lower_rows, diagonal_rows, upper_rows, xp=xp)

    corner_upper_right = lower_rows[:, 0]
    corner_lower_left = upper_rows[:, -1]
    gamma = -diagonal_rows[:, 0]
    if _array_has_true(xp.abs(gamma) <= floor, xp=xp):
        raise np.linalg.LinAlgError(
            "cyclic solve requires a nonzero leading diagonal"
        )
    modified_diagonal = diagonal_rows.copy()
    modified_diagonal[:, 0] -= gamma
    modified_diagonal[:, -1] -= (
        corner_upper_right * corner_lower_left / gamma
    )

    primary, primary_invalid = _solve_tridiagonal_rows(
        lower_rows,
        modified_diagonal,
        upper_rows,
        rhs_rows,
        pivot_floor=floor,
        xp=xp,
    )
    correction_rhs = xp.zeros_like(rhs_rows)
    correction_rhs[:, 0] = gamma
    correction_rhs[:, -1] = corner_lower_left
    correction, correction_invalid = _solve_tridiagonal_rows(
        lower_rows,
        modified_diagonal,
        upper_rows,
        correction_rhs,
        pivot_floor=floor,
        xp=xp,
    )
    factor_denominator = (
        1.0
        + correction[:, 0]
        + corner_upper_right * correction[:, -1] / gamma
    )
    correction_denominator_invalid = xp.abs(factor_denominator) <= floor
    factor = (
        primary[:, 0]
        + corner_upper_right * primary[:, -1] / gamma
    ) / factor_denominator
    solution_rows = primary - factor[:, None] * correction
    solution = xp.moveaxis(
        solution_rows.reshape(moved_shape),
        -1,
        -2,
    )
    invalid = xp.logical_or(primary_invalid, correction_invalid)
    invalid = xp.logical_or(invalid, correction_denominator_invalid)
    if _array_has_true(invalid, xp=xp):
        raise np.linalg.LinAlgError(
            "cyclic solve encountered a zero or near-zero pivot"
        )
    if _array_has_true(~xp.isfinite(solution), xp=xp):
        raise np.linalg.LinAlgError("cyclic solve produced nonfinite values")
    return solution.astype(rhs.dtype, copy=False)


__all__ = ["solve_cyclic_tridiagonal_rows"]

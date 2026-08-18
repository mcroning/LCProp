"""CuPy kernel for PR variable-coefficient cyclic tridiagonal systems."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


_CUDA_TEMPLATE = r"""
extern "C" __global__
void KERNEL_NAME(
    const SCALAR *lower,
    const SCALAR *diagonal,
    const SCALAR *upper,
    const SCALAR *rhs,
    SCALAR *cprime,
    SCALAR *dprime,
    SCALAR *solution,
    int *status,
    const int rows,
    const int nx,
    const int ny,
    const SCALAR epsilon)
{
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= rows) return;

    const int z = row / ny;
    const int y = row - z * ny;
    const long long base = ((long long)z * nx * ny) + y;
    #define IDX(j) (base + ((long long)(j) * ny))

    SCALAR scale = (SCALAR)1;
    for (int j = 0; j < nx; ++j) {
        const SCALAR value = lower[IDX(j)];
        if (!isfinite(value)) { status[row] = 1; return; }
        scale = fmax(scale, fabs(value));
    }
    for (int j = 0; j < nx; ++j) {
        const SCALAR value = diagonal[IDX(j)];
        if (!isfinite(value)) { status[row] = 2; return; }
        scale = fmax(scale, fabs(value));
    }
    for (int j = 0; j < nx; ++j) {
        const SCALAR value = upper[IDX(j)];
        if (!isfinite(value)) { status[row] = 3; return; }
        scale = fmax(scale, fabs(value));
    }
    for (int j = 0; j < nx; ++j) {
        const SCALAR value = rhs[IDX(j)];
        if (!isfinite(value)) { status[row] = 4; return; }
    }

    const SCALAR floor = (SCALAR)32 * epsilon * scale;
    const SCALAR corner_upper_right = lower[IDX(0)];
    const SCALAR corner_lower_left = upper[IDX(nx - 1)];
    const SCALAR gamma = -diagonal[IDX(0)];
    if (fabs(gamma) <= floor) { status[row] = 5; return; }

    // Primary nonperiodic Thomas solve with the cyclic diagonal modification.
    SCALAR denominator = diagonal[IDX(0)] - gamma;
    if (fabs(denominator) <= floor || !isfinite(denominator)) {
        status[row] = 6; return;
    }
    cprime[IDX(0)] = upper[IDX(0)] / denominator;
    dprime[IDX(0)] = rhs[IDX(0)] / denominator;
    for (int j = 1; j < nx; ++j) {
        SCALAR modified_diagonal = diagonal[IDX(j)];
        if (j == nx - 1) {
            modified_diagonal -= corner_upper_right * corner_lower_left / gamma;
        }
        denominator = modified_diagonal
            - lower[IDX(j)] * cprime[IDX(j - 1)];
        if (fabs(denominator) <= floor || !isfinite(denominator)) {
            status[row] = 6; return;
        }
        cprime[IDX(j)] = j < nx - 1
            ? upper[IDX(j)] / denominator : (SCALAR)0;
        dprime[IDX(j)] = (
            rhs[IDX(j)] - lower[IDX(j)] * dprime[IDX(j - 1)]
        ) / denominator;
    }
    solution[IDX(nx - 1)] = dprime[IDX(nx - 1)];
    for (int j = nx - 2; j >= 0; --j) {
        solution[IDX(j)] = dprime[IDX(j)]
            - cprime[IDX(j)] * solution[IDX(j + 1)];
    }

    // Correction solve. Reuse cprime/dprime; dprime becomes the correction.
    denominator = diagonal[IDX(0)] - gamma;
    cprime[IDX(0)] = upper[IDX(0)] / denominator;
    dprime[IDX(0)] = gamma / denominator;
    for (int j = 1; j < nx; ++j) {
        SCALAR modified_diagonal = diagonal[IDX(j)];
        if (j == nx - 1) {
            modified_diagonal -= corner_upper_right * corner_lower_left / gamma;
        }
        denominator = modified_diagonal
            - lower[IDX(j)] * cprime[IDX(j - 1)];
        if (fabs(denominator) <= floor || !isfinite(denominator)) {
            status[row] = 6; return;
        }
        cprime[IDX(j)] = j < nx - 1
            ? upper[IDX(j)] / denominator : (SCALAR)0;
        const SCALAR correction_rhs = j == nx - 1
            ? corner_lower_left : (SCALAR)0;
        dprime[IDX(j)] = (
            correction_rhs - lower[IDX(j)] * dprime[IDX(j - 1)]
        ) / denominator;
    }
    for (int j = nx - 2; j >= 0; --j) {
        dprime[IDX(j)] -= cprime[IDX(j)] * dprime[IDX(j + 1)];
    }

    const SCALAR factor_denominator = (SCALAR)1 + dprime[IDX(0)]
        + corner_upper_right * dprime[IDX(nx - 1)] / gamma;
    if (fabs(factor_denominator) <= floor || !isfinite(factor_denominator)) {
        status[row] = 7; return;
    }
    const SCALAR factor = (
        solution[IDX(0)]
        + corner_upper_right * solution[IDX(nx - 1)] / gamma
    ) / factor_denominator;
    for (int j = 0; j < nx; ++j) {
        solution[IDX(j)] -= factor * dprime[IDX(j)];
        if (!isfinite(solution[IDX(j)])) { status[row] = 8; return; }
    }

    #undef IDX
}
"""


@lru_cache(maxsize=2)
def _kernel(dtype_name: str):
    import cupy as cp

    if dtype_name == "float32":
        scalar = "float"
        name = "pr_cyclic_variable_float32"
    elif dtype_name == "float64":
        scalar = "double"
        name = "pr_cyclic_variable_float64"
    else:  # guarded by the public validator
        raise TypeError(dtype_name)
    source = _CUDA_TEMPLATE.replace("SCALAR", scalar).replace(
        "KERNEL_NAME", name
    )
    return cp.RawKernel(source, name)


def _require_c_contiguous(array: Any, *, name: str) -> None:
    if not bool(array.flags.c_contiguous):
        raise ValueError(
            f"{name} must be C-contiguous for the direct-layout GPU solve"
        )


def solve_cyclic_tridiagonal_rows_gpu(
    lower,
    diagonal,
    upper,
    rhs,
    *,
    threads_per_block: int = 128,
):
    """Solve real variable-coefficient cyclic x rows with a CuPy RawKernel.

    Arrays have shape ``(Nx, Ny)`` or ``(Nz, Nx, Ny)``. The kernel indexes
    the production C-contiguous layout directly, so it does not transpose or
    materialize ``(Nz*Ny, Nx)`` coefficient volumes. Each CUDA thread solves
    one complete x row using cyclic Sherman--Morrison with two Thomas sweeps.
    Inputs are not modified.
    """

    import cupy as cp

    arrays = {
        "lower": lower,
        "diagonal": diagonal,
        "upper": upper,
        "rhs": rhs,
    }
    if not all(isinstance(value, cp.ndarray) for value in arrays.values()):
        raise TypeError("GPU cyclic solver requires CuPy arrays")
    if not (lower.shape == diagonal.shape == upper.shape == rhs.shape):
        raise ValueError(
            "lower, diagonal, upper, and rhs must have identical shapes"
        )
    if rhs.ndim not in (2, 3):
        raise ValueError(
            "cyclic systems must have shape (Nx, Ny) or (Nz, Nx, Ny)"
        )
    nx = int(rhs.shape[-2])
    ny = int(rhs.shape[-1])
    if nx < 3:
        raise ValueError("cyclic solve requires Nx >= 3")
    if ny < 1:
        raise ValueError("cyclic solve requires Ny >= 1")
    if rhs.dtype not in (cp.float32, cp.float64):
        raise TypeError("rhs must have dtype float32 or float64")
    for name, array in arrays.items():
        if array.dtype != rhs.dtype:
            raise TypeError("all cyclic-system arrays must have the same dtype")
        _require_c_contiguous(array, name=name)

    block = int(threads_per_block)
    if block < 1 or block > 1024:
        raise ValueError("threads_per_block must be between 1 and 1024")
    rows = int(np.prod(rhs.shape[:-2], dtype=np.int64)) * ny
    cprime = cp.empty_like(rhs)
    dprime = cp.empty_like(rhs)
    solution = cp.empty_like(rhs)
    status = cp.zeros(rows, dtype=cp.int32)
    epsilon = rhs.dtype.type(np.finfo(rhs.dtype).eps)
    kernel = _kernel(str(rhs.dtype))
    kernel(
        ((rows + block - 1) // block,),
        (block,),
        (
            lower,
            diagonal,
            upper,
            rhs,
            cprime,
            dprime,
            solution,
            status,
            np.int32(rows),
            np.int32(nx),
            np.int32(ny),
            epsilon,
        ),
    )

    # One synchronization after all rows replaces per-x Python synchronizations.
    status_host = cp.asnumpy(status)
    if np.any(status_host == 1):
        raise ValueError("lower must contain only finite values")
    if np.any(status_host == 2):
        raise ValueError("diagonal must contain only finite values")
    if np.any(status_host == 3):
        raise ValueError("upper must contain only finite values")
    if np.any(status_host == 4):
        raise ValueError("rhs must contain only finite values")
    if np.any(status_host == 5):
        raise np.linalg.LinAlgError(
            "cyclic solve requires a nonzero leading diagonal"
        )
    if np.any((status_host == 6) | (status_host == 7)):
        raise np.linalg.LinAlgError(
            "cyclic solve encountered a zero or near-zero pivot"
        )
    if np.any(status_host == 8):
        raise np.linalg.LinAlgError("cyclic solve produced nonfinite values")
    return solution


__all__ = ["solve_cyclic_tridiagonal_rows_gpu"]

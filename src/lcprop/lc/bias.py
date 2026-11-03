"""Build equilibrium dark LC director bias fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import scipy.special as spspec

from lcprop.lc.specs import BiasSpec
from lcprop.core.grid import RuntimeGrid
from lcprop.lc.normalization import make_lc_spatial_normalization

Array = Any
EPS0 = 8.8541878128e-12


def compute_b_from_voltage(
    V_bias: float,
    *,
    K: float,
    delta_epsilon: float,
) -> float:
    """Return b = delta_epsilon eps0 V_bias^2 / (8K)."""
    return float(delta_epsilon) * EPS0 * float(V_bias) ** 2 / (8.0 * float(K))


def compute_freedericksz_voltage(
    *,
    K: float,
    delta_epsilon: float,
) -> float:
    """Return one-constant Freedericksz voltage."""
    return math.pi * math.sqrt(float(K) / (EPS0 * float(delta_epsilon)))


def resolved_b(material, bias) -> float:
    """Return b_override if present, otherwise compute b from voltage."""
    material.validate()
    bias.validate()
    if bias.b_override is not None:
        return float(bias.b_override)
    return compute_b_from_voltage(
        bias.V_bias,
        K=material.K,
        delta_epsilon=material.delta_epsilon,
    )


@dataclass(frozen=True)
class BiasResult:
    """Prepared director bias fields."""

    theta_2d: Array
    theta_stack: Array
    theta_bc: float
    theta_clamp: tuple[float, float]
    b: float

    def summary(self) -> dict[str, float | tuple[float, float]]:
        return {
            "theta_bc": float(self.theta_bc),
            "theta_clamp": self.theta_clamp,
            "b": float(self.b),
            "theta_center": float(_to_numpy(self.theta_2d)[_to_numpy(self.theta_2d).shape[0] // 2, 0]),
            "theta_min": float(_to_numpy(self.theta_2d).min()),
            "theta_max": float(_to_numpy(self.theta_2d).max()),
        }


def _to_numpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def _bisect_root(f, lo: float = 1e-12, hi: float = 1.0 - 1e-12, *, max_iter: int = 100) -> float:
    flo = float(f(lo))
    fhi = float(f(hi))
    if flo * fhi > 0:
        raise RuntimeError(f"Could not bracket elliptic modulus root: f(lo)={flo}, f(hi)={fhi}")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = float(f(mid))
        if flo * fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        if abs(hi - lo) < 1e-14:
            break

    return 0.5 * (lo + hi)


def theta0_from_b_zero_bc(b: float) -> float:
    """Return center angle for the exact zero-boundary dark solution."""
    b_c = math.pi**2 / 8.0
    if float(b) <= b_c:
        return 0.0

    two_b = 2.0 * float(b)

    def f(m: float) -> float:
        Km = spspec.ellipk(m)
        return float(Km * Km - two_b)

    m = _bisect_root(f)
    return float(math.asin(math.sqrt(m)))


def theta0_from_b_dirichlet_bc(b: float, theta_bc: float) -> float:
    """Return center angle for the exact dark solution with nonzero Dirichlet boundary."""
    b = float(b)
    theta_bc = float(theta_bc)

    if abs(theta_bc) < 1e-14:
        return theta0_from_b_zero_bc(b)

    if b <= 0.0:
        return theta_bc

    two_b = 2.0 * b
    sqrt2b = math.sqrt(two_b)
    sbc = math.sin(theta_bc)

    def f(m: float) -> float:
        _, cn, dn, _ = spspec.ellipj(sqrt2b, float(m))
        cd = cn / dn
        return float(math.sqrt(m) * cd - sbc)

    m = _bisect_root(f)
    return float(math.asin(math.sqrt(m)))


def b_from_theta0_zero_bc(theta0: float) -> float:
    """Inverse of theta0_from_b_zero_bc for zero boundary angle."""
    s = math.sin(float(theta0))
    m = min(max(s * s, 1e-12), 1.0 - 1e-12)
    Km = spspec.ellipk(m)
    return float(0.5 * Km * Km)


def theta_bias_1d_exact(
    grid: RuntimeGrid,
    bias: BiasSpec,
    material,
    *,
    eps_clip: float = 1e-12,
) -> Array:
    """Return the exact 1-D dark-bias director profile on the x grid.

    Solves theta_xx + b sin(2 theta) = 0 with Dirichlet values theta_bc
    at both x boundaries, using the Jacobi elliptic cd construction.
    """

    bias.validate()
    material.validate()

    xp = grid.xp
    dtype = grid.real_dtype

    b = resolved_b(material, bias)
    theta_bc = float(bias.theta_bc)

    # Dimensionless LC transverse coordinate u in [-1, 1].
    normalization = make_lc_spatial_normalization(grid)
    u_np = _to_numpy(normalization.u).astype(np.float64, copy=False)

    if abs(theta_bc) < 1e-14:
        theta_np = _theta_bias_1d_exact_zero_bc_np(u_np, b, eps_clip=eps_clip)
    else:
        theta_np = _theta_bias_1d_exact_dirichlet_np(
            u_np,
            b,
            theta_bc=theta_bc,
            eps_clip=eps_clip,
        )

    theta_np = np.clip(theta_np, float(bias.theta_min), float(bias.theta_max))
    theta_np[0] = theta_bc
    theta_np[-1] = theta_bc

    return xp.asarray(theta_np, dtype=dtype)


def _theta_bias_1d_exact_zero_bc_np(u: np.ndarray, b: float, *, eps_clip: float) -> np.ndarray:
    b = float(b)
    b_c = math.pi**2 / 8.0

    if b <= b_c:
        return np.zeros_like(u, dtype=np.float64)

    two_b = 2.0 * b

    def f(m: float) -> float:
        Km = spspec.ellipk(m)
        return float(Km * Km - two_b)

    m = _bisect_root(f)
    theta0 = math.asin(math.sqrt(m))

    arg = math.sqrt(two_b) * u
    _, cn, dn, _ = spspec.ellipj(arg, float(m))
    cd = cn / dn

    s = math.sin(theta0) * cd
    s = np.clip(s, -1.0 + eps_clip, 1.0 - eps_clip)
    theta = np.arcsin(s)
    theta[0] = 0.0
    theta[-1] = 0.0
    return theta


def _theta_bias_1d_exact_dirichlet_np(
    u: np.ndarray,
    b: float,
    *,
    theta_bc: float,
    eps_clip: float,
) -> np.ndarray:
    b = float(b)
    theta_bc = float(theta_bc)
    two_b = 2.0 * b
    sbc = math.sin(theta_bc)
    sqrt2b = math.sqrt(max(two_b, 0.0))

    if b <= 0.0:
        return np.full_like(u, theta_bc, dtype=np.float64)

    def f(m: float) -> float:
        _, cn, dn, _ = spspec.ellipj(sqrt2b, float(m))
        cd = cn / dn
        return float(math.sqrt(m) * cd - sbc)

    m = _bisect_root(f)
    theta0 = math.asin(math.sqrt(m))

    arg = sqrt2b * u
    _, cn, dn, _ = spspec.ellipj(arg, float(m))
    cd = cn / dn

    s = math.sin(theta0) * cd
    s = np.clip(s, -1.0 + eps_clip, 1.0 - eps_clip)
    theta = np.arcsin(s)

    theta[0] = theta_bc
    theta[-1] = theta_bc
    return theta


def build_exact_bias_2d(bias: BiasSpec, grid: RuntimeGrid, material) -> Array:
    """Return exact dark-bias theta(x,y), tiled uniformly in y."""
    xp = grid.xp
    theta_1d = theta_bias_1d_exact(grid, bias, material)
    theta_2d = xp.repeat(theta_1d[:, None], int(grid.Ny), axis=1).astype(
        grid.real_dtype,
        copy=False,
    )
    theta_2d[0, :] = float(bias.theta_bc)
    theta_2d[-1, :] = float(bias.theta_bc)
    return theta_2d


def stack_theta(theta_2d: Array, grid: RuntimeGrid) -> Array:
    """Repeat a 2-D theta field into a z-stack."""
    xp = grid.xp
    return xp.repeat(theta_2d[None, :, :], int(grid.Nz), axis=0).astype(
        grid.real_dtype,
        copy=False,
    )


def build_bias(bias: BiasSpec, grid: RuntimeGrid, material) -> BiasResult:
    """Build exact dark-bias 2-D and z-stack theta fields."""
    b = resolved_b(material, bias)
    theta_2d = build_exact_bias_2d(bias, grid, material)
    theta_stack = stack_theta(theta_2d, grid)

    return BiasResult(
        theta_2d=theta_2d,
        theta_stack=theta_stack,
        theta_bc=float(bias.theta_bc),
        theta_clamp=(float(bias.theta_min), float(bias.theta_max)),
        b=float(b),
    )


__all__ = [
    "Array",
    "BiasResult",
    "EPS0",
    "compute_b_from_voltage",
    "compute_freedericksz_voltage",
    "resolved_b",
    "theta0_from_b_zero_bc",
    "theta0_from_b_dirichlet_bc",
    "b_from_theta0_zero_bc",
    "theta_bias_1d_exact",
    "build_exact_bias_2d",
    "stack_theta",
    "build_bias",
]

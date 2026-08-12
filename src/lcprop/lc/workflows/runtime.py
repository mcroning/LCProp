"""Shared workflow builders for LCProp.

This module constructs reusable runtime components from RunRequest objects.

It is intentionally between core objects and workflows:

- core objects describe the experiment
- builders construct numerical components
- workflows orchestrate algorithms

No long-running workflow loops should live here.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Any

from lcprop.lc.requests import StaticRunRequest
from lcprop.core.grid import RuntimeGrid, make_grid
from lcprop.core.derived import resolved_b
from lcprop.lc.coupling import resolved_bi
from lcprop.lc.bias import BiasResult, build_bias
from lcprop.optics.launch import LaunchResult, build_launch
from lcprop.optics.splitstep import (
    advance_slice_with_midintensity,
    weighted_theta_intensity,
)
from lcprop.optics.substeps import (
    OpticalSubstepPlan,
    build_optical_substep_kernel,
)
from lcprop.algorithms.theta_cn import prepare_cn_operator
from lcprop.algorithms.theta_picard import cn_trapezoid_picard_step
from lcprop.algorithms.theta_cn_zcoupled import cn_trapezoid_picard_step_zcoupled


@dataclass(frozen=True)
class CNOperator:
    """Prepared x/y CN operator."""

    s: Any
    off: Any
    diag: Any
    lam_y: Any


@dataclass(frozen=True)
class RuntimeComponents:
    """Reusable components built from a run request."""

    request: StaticRunRequest
    grid: RuntimeGrid
    bias: BiasResult
    launch: LaunchResult

    b: float
    bi: float

    wavelength_um: float
    n_ref: float
    coherent: bool
    coherence_groups: tuple[str, ...]
    physical_total_power_mW: float
    power_fractions: Any

    kernel: Any
    optical_substeps: OpticalSubstepPlan
    cn: CNOperator


def initial_A_field(components: RuntimeComponents):
    """Return the optical initial condition for workflows using runtime components."""
    A0 = components.launch.A0
    initial_A = getattr(components.request, "initial_A", None)
    if initial_A is None:
        return A0.copy()
    A = components.grid.xp.asarray(initial_A, dtype=A0.dtype).copy()
    if A.shape != A0.shape:
        raise ValueError(f"initial_A shape {A.shape} does not match launch field shape {A0.shape}")
    return A


def initial_theta_field(components: RuntimeComponents):
    """Return the theta initial condition for workflows using runtime components."""
    theta0 = components.bias.theta_2d
    initial_theta = getattr(components.request, "initial_theta", None)
    if initial_theta is None:
        return theta0.copy()
    theta = components.grid.xp.asarray(initial_theta, dtype=theta0.dtype).copy()
    if theta.shape != theta0.shape:
        raise ValueError(f"initial_theta shape {theta.shape} does not match bias field shape {theta0.shape}")
    return theta


def build_runtime_components(
    request: StaticRunRequest,
    *,
    theta_dt: float = 0.01,
    mobility: float = 1.0,
) -> RuntimeComponents:
    """Build common runtime components for static/TD-style workflows."""

    request.grid.validate()
    request.material.validate()
    request.bias.validate()
    request.beams.validate()
    request.runtime.validate()

    grid = make_grid(request.grid, real_dtype=np.float64 if request.runtime.precision == "float64" else np.float32)
    bias = build_bias(request.bias, grid, request.material)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128 if request.runtime.precision == "float64" else np.complex64)

    wavelength_um = float(request.beams.channels[0].wavelength_um)
    n_ref = float(request.material.no)
    coherence_groups = launch.coherence_groups
    coherent = len(set(coherence_groups)) == 1

    optical_substeps, kernel = build_optical_substep_kernel(
        grid.fxy2_um,
        dz_um=grid.dz_um,
        propagation_wavelength_um=wavelength_um,
        active_wavelengths_um=(
            channel.wavelength_um for channel in request.beams.channels
        ),
        n_ref=n_ref,
        enabled=request.runtime.optical_substeps_enabled,
        dn_max_est=request.runtime.optical_dn_max_est,
        max_phase_per_substep_rad=(
            request.runtime.optical_max_phase_per_substep_rad
        ),
        max_substeps=request.runtime.optical_max_substeps,
        xp=grid.xp,
    )

    b = resolved_b(request.material, request.bias)
    bi = resolved_bi(request.grid, request.material, request.beams)

    s, off, diag, lam_y = prepare_cn_operator(
        dt=theta_dt,
        mobility=mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=grid.xp,
        dtype=grid.real_dtype,
    )

    return RuntimeComponents(
        request=request,
        grid=grid,
        bias=bias,
        launch=launch,
        b=b,
        bi=bi,
        wavelength_um=wavelength_um,
        n_ref=n_ref,
        coherent=coherent,
        coherence_groups=coherence_groups,
        physical_total_power_mW=launch.physical_total_power_mW,
        power_fractions=launch.power_fractions,
        kernel=kernel,
        optical_substeps=optical_substeps,
        cn=CNOperator(s=s, off=off, diag=diag, lam_y=lam_y),
    )


def initial_theta_intensity(components: RuntimeComponents):
    """Return effective theta-driving intensity for the launch field."""

    return weighted_theta_intensity(
        initial_A_field(components),
        components.launch.theta_weights,
        coherent=components.coherent,
        coherence_groups=components.coherence_groups,
        xp=components.grid.xp,
    )


def make_picard_theta_relax(
    components: RuntimeComponents,
    *,
    theta_dt: float = 0.01,
    mobility: float = 1.0,
    max_iter: int = 4,
    tol_update: float = 1e-6,
):
    """Return theta_relax(theta, intensity, outer) callback for static relax."""

    request = components.request
    grid = components.grid
    cn = components.cn
    xp = grid.xp

    def theta_relax(theta, intensity, outer):
        return cn_trapezoid_picard_step(
            theta,
            intensity,
            intensity,
            b=components.b,
            bi=components.bi,
            dt=theta_dt,
            mobility=mobility,
            dx=grid.du,
            dy=grid.dv,
            s=cn.s,
            off=cn.off,
            diag=cn.diag,
            max_iter=max_iter,
            tol_update=tol_update,
            clamp=(request.bias.theta_min, request.bias.theta_max),
            xp=xp,
        )

    return theta_relax


def make_global_uniform_theta_iteration(components: RuntimeComponents):
    """Return the legacy one-theta-for-all-z optics iteration.

    This helper intentionally restarts from the entrance field and applies one
    uniform 2-D theta field through every z slice. It is retained for explicit
    global iteration experiments; it does not implement local self-consistency.
    """

    grid = components.grid
    xp = grid.xp

    def optics_update(A_unused, theta, outer):
        A = initial_A_field(components)
        I_mid = initial_theta_intensity(components)

        for _ in range(grid.Nz):
            A, _, _, I_mid = advance_slice_with_midintensity(
                A,
                theta,
                kernel=components.kernel,
                dz=grid.dz_um,
                wavelength=components.wavelength_um,
                n_ref=components.n_ref,
                ne=components.request.material.ne,
                no=components.request.material.no,
                coherent=components.coherent,
                coherence_groups=components.coherence_groups,
                theta_weights=components.launch.theta_weights,
                Nsub=components.optical_substeps.Nsub,
                xp=xp,
            )

        return A, I_mid

    return optics_update


def make_td_optics_step(components: RuntimeComponents):
    """Return optics_step(A, theta_k, k) callback for TD z-march."""

    grid = components.grid
    xp = grid.xp

    def optics_step(A, theta_k, k):
        A, _, _, I_mid = advance_slice_with_midintensity(
            A,
            theta_k,
            kernel=components.kernel,
            dz=grid.dz_um,
            wavelength=components.wavelength_um,
            n_ref=components.n_ref,
            ne=components.request.material.ne,
            no=components.request.material.no,
            coherent=components.coherent,
            coherence_groups=components.coherence_groups,
            theta_weights=components.launch.theta_weights,
            Nsub=components.optical_substeps.Nsub,
            xp=xp,
        )
        return A, I_mid

    return optics_step


def make_zcoupled_theta_step(
    components: RuntimeComponents,
    *,
    theta_dt: float = 7.5e-4,
    mobility: float = 1.0,
    gamma_z: float = 0.0,
    max_iter: int = 4,
    tol_update: float = 1e-6,
):
    """Return theta_step(theta_k, I_mid, theta_prev, theta_next, k) for TD."""

    request = components.request
    grid = components.grid
    cn = components.cn
    xp = grid.xp

    # z coordinate in the theta PDE is optional; gamma_z=0 recovers the
    # uncoupled slice update.
    inv_dz2 = 1.0 / (float(grid.dz_um) * float(grid.dz_um))

    def theta_step(theta_k, I_mid, theta_prev, theta_next, k):
        return cn_trapezoid_picard_step_zcoupled(
            theta_k,
            I_mid,
            dt=theta_dt,
            b=components.b,
            bi=components.bi,
            theta_prev=theta_prev,
            theta_next=theta_next,
            gamma_z=gamma_z,
            mobility=mobility,
            dx=grid.du,
            dy=grid.dv,
            s=cn.s,
            off=cn.off,
            diag=cn.diag,
            inv_dz2=inv_dz2,
            max_iter=max_iter,
            tol_update=tol_update,
            clamp=(request.bias.theta_min, request.bias.theta_max),
            xp=xp,
        )

    return theta_step

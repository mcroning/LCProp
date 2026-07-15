"""Dz-independent transverse eigensoliton polishing solver.

This module leaves ``soliton.py`` unchanged. It reuses the verified optical
operator, parity, normalization, theta-relaxation, and diagnostics already
implemented there, and adds only a direct transverse eigenvalue solve.
"""

from __future__ import annotations

import time as _time
from dataclasses import replace

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from lcprop.core.backend import asnumpy
from lcprop.optics.splitstep import total_intensity
from lcprop.workflows.runtime import (
    build_runtime_components,
    make_zcoupled_theta_step,
)
from lcprop.workflows.soliton import (
    SolitonRequest,
    SolitonResult,
    _apply_mode_seed,
    _canonical_mode,
    _field_difference,
    _normalize_power,
    _prepare_initial_A,
    _prepare_initial_theta,
    _project_field_parity,
    _project_mode_parity,
    _project_theta_parity,
    _sym_x_even,
    _sym_x_odd,
    _sym_y_even,
    _sym_y_odd,
    _target_power,
    apply_optical_eigen_operator,
    cn_trapezoid_picard_step,
    intensity_metrics,
    make_beta_symbol,
    rayleigh_beta,
    residual_theta_static,
    solve_const_offdiag_batched,
)


def _project_field_parity_numpy(field: np.ndarray, mode: str) -> np.ndarray:
    mode = _canonical_mode(mode)
    if mode == "custom":
        return field
    if mode == "00":
        return _sym_y_even(_sym_x_even(field))
    if mode == "10":
        return _sym_y_even(_sym_x_odd(field))
    if mode == "01":
        return _sym_y_odd(_sym_x_even(field))
    if mode == "11":
        return _sym_y_odd(_sym_x_odd(field))
    return field


def _transverse_eigenpair(
    A_guess,
    theta,
    beta_symbol,
    *,
    grid,
    mode: str,
    ne: float,
    no: float,
    n_ref: float,
    wavelength_um: float,
    eig_tol: float,
    eig_maxiter: int,
):
    """Solve H(theta) A = beta A using the verified LCProp optical operator."""
    xp = grid.xp
    shape = tuple(int(v) for v in theta.shape)
    size = int(np.prod(shape))

    def matvec(vector):
        field_np = np.asarray(vector, dtype=np.complex128).reshape(shape)
        field_np = _project_field_parity_numpy(field_np, mode)
        field = xp.asarray(field_np, dtype=xp.complex128)[None, :, :]
        h_field = apply_optical_eigen_operator(
            field,
            theta,
            beta_symbol,
            grid=grid,
            ne=float(ne),
            no=float(no),
            n_ref=float(n_ref),
            wavelength_um=float(wavelength_um),
        )
        h_field = _project_field_parity(h_field, mode)
        return np.asarray(asnumpy(h_field[0]), dtype=np.complex128).ravel()

    operator = LinearOperator(
        shape=(size, size),
        matvec=matvec,
        dtype=np.complex128,
    )

    v0 = np.asarray(asnumpy(A_guess), dtype=np.complex128)
    if v0.ndim == 3:
        if v0.shape[0] != 1:
            raise ValueError(
                "The transverse eigensoliton solver currently supports one optical channel"
            )
        v0 = v0[0]
    v0 = _project_field_parity_numpy(v0, mode).ravel()
    norm = np.linalg.norm(v0)
    if norm == 0.0:
        raise ValueError("Initial field is zero after parity projection")
    v0 /= norm

    _, eigenvectors = eigsh(
        operator,
        k=1,
        which="LA",
        v0=v0,
        tol=float(eig_tol),
        maxiter=int(eig_maxiter),
    )

    field_np = eigenvectors[:, 0].reshape(shape)
    field_np = _project_field_parity_numpy(field_np, mode)
    field = xp.asarray(field_np, dtype=xp.complex128)[None, :, :]

    beta = float(
        asnumpy(
            rayleigh_beta(
                field,
                theta,
                beta_symbol,
                grid=grid,
                ne=float(ne),
                no=float(no),
                n_ref=float(n_ref),
                wavelength_um=float(wavelength_um),
            )
        )
    )

    h_field_np = matvec(field_np.ravel()).reshape(shape)
    optical_residual = float(
        np.linalg.norm(h_field_np - beta * field_np)
        / (np.linalg.norm(field_np) + 1e-300)
    )
    return field_np, beta, optical_residual


def _request_value(request: SolitonRequest, name: str, default):
    return getattr(request, name, default)


def run_soliton(request: SolitonRequest) -> SolitonResult:
    """Solve the coupled stationary transverse LC/optical problem."""
    request.validate()

    eig_tol = float(_request_value(request, "eig_tol", 1e-9))
    eig_maxiter = int(_request_value(request, "eig_maxiter", 500))
    if eig_tol <= 0.0:
        raise ValueError("eig_tol must be > 0")
    if eig_maxiter < 1:
        raise ValueError("eig_maxiter must be >= 1")

    runtime = build_runtime_components(request.base, theta_dt=7.5e-4, mobility=1.0)
    grid = runtime.grid
    xp = grid.xp

    if len(request.base.beams.channels) != 1:
        raise ValueError(
            "The transverse eigensoliton solver currently supports exactly one optical channel"
        )

    physical_power_mW = _target_power(request.base.beams)
    target_power = 1.0
    coherent = runtime.coherent

    A = _prepare_initial_A(
        request.initial_A,
        runtime.launch,
        grid=grid,
        target_power=target_power,
        coherent=coherent,
        xp=xp,
    )
    if request.initial_A is None:
        A = _apply_mode_seed(
            A,
            grid=grid,
            beams=request.base.beams,
            mode=request.mode,
            xp=xp,
        )
        A = _normalize_power(
            A,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            xp=xp,
        )

    theta = _prepare_initial_theta(request.initial_theta, runtime.bias, xp=xp)
    A, theta = _project_mode_parity(
        A,
        theta,
        request.mode,
        grid=grid,
        target_power=target_power,
        coherent=coherent,
        theta_clamp=runtime.bias.theta_clamp,
        theta_bc=request.base.bias.theta_bc,
        xp=xp,
    )

    theta_ref = np.asarray(asnumpy(runtime.bias.theta_2d), dtype=np.float64)
    ne = float(request.base.material.ne)
    no = float(request.base.material.no)
    n_ref = float(
        np.mean(
            (ne * no)
            / np.sqrt((ne * np.cos(theta_ref)) ** 2 + (no * np.sin(theta_ref)) ** 2)
        )
    )
    wavelength_um = float(runtime.wavelength_um)
    beta_symbol, _ = make_beta_symbol(
        grid,
        wavelength_um=wavelength_um,
        n_ref=n_ref,
        subtract_carrier=True,
    )
    cn = runtime.cn

    def relax_theta(theta_in, intensity):
        theta_out = theta_in
        for _ in range(int(request.theta_steps_per_outer)):
            theta_new = cn_trapezoid_picard_step(
                theta_out,
                intensity,
                intensity,
                b=runtime.b,
                bi=runtime.bi,
                dt=7.5e-4,
                mobility=1.0,
                dx=grid.du,
                dy=grid.dv,
                s=cn.s,
                off=cn.off,
                diag=cn.diag,
                max_iter=4,
                tol_update=1e-6,
                clamp=runtime.bias.theta_clamp,
                tridiag_solver=solve_const_offdiag_batched,
                xp=xp,
            )
            theta_out = (
                (1.0 - float(request.theta_mix)) * theta_out
                + float(request.theta_mix) * theta_new
            )
            theta_out[0, :] = request.base.bias.theta_bc
            theta_out[-1, :] = request.base.bias.theta_bc
        return theta_out

    history: list[dict] = []
    converged = False
    beta = float("nan")
    optical_residual = float("inf")
    t0 = _time.perf_counter()

    for outer in range(int(request.max_outer)):
        A_prev = A.copy()
        theta_prev = theta.copy()

        intensity = total_intensity(A, coherent=coherent, xp=xp)
        theta = relax_theta(theta, intensity)
        theta = _project_theta_parity(theta, request.mode)
        theta = xp.clip(
            theta,
            float(runtime.bias.theta_clamp[0]),
            float(runtime.bias.theta_clamp[1]),
        )
        theta[0, :] = request.base.bias.theta_bc
        theta[-1, :] = request.base.bias.theta_bc

        A_np, beta, optical_residual = _transverse_eigenpair(
            A,
            theta,
            beta_symbol,
            grid=grid,
            mode=request.mode,
            ne=ne,
            no=no,
            n_ref=n_ref,
            wavelength_um=wavelength_um,
            eig_tol=eig_tol,
            eig_maxiter=eig_maxiter,
        )
        A_candidate = xp.asarray(A_np, dtype=A.dtype)[None, :, :]
        A_candidate = _normalize_power(
            A_candidate,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            xp=xp,
        )

        dxdy = float(grid.dx_um) * float(grid.dy_um)
        overlap = xp.sum(xp.conj(A) * A_candidate) * dxdy
        A_candidate = A_candidate * xp.exp(-1j * xp.angle(overlap))
        A = (
            (1.0 - float(request.field_mix)) * A
            + float(request.field_mix) * A_candidate
        )
        A = _project_field_parity(A, request.mode)
        A = _normalize_power(
            A,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            xp=xp,
        )

        field_rel, overlap_abs = _field_difference(A_prev, A, grid=grid, xp=xp)
        dtheta = theta - theta_prev
        dtheta_rms = float(asnumpy(xp.sqrt(xp.mean(dtheta * dtheta))))
        dtheta_max = float(asnumpy(xp.max(xp.abs(dtheta))))

        intensity = total_intensity(A, coherent=coherent, xp=xp)
        residual = residual_theta_static(
            theta,
            intensity,
            b=runtime.b,
            bi=runtime.bi,
            dx=grid.du,
            dy=grid.dv,
            theta_bc=request.base.bias.theta_bc,
            xp=xp,
        )

        row = {
            "outer": int(outer + 1),
            "field_rel": field_rel,
            "overlap_abs": overlap_abs,
            "dtheta_rms": dtheta_rms,
            "dtheta_max": dtheta_max,
            "beta": beta,
            "optical_residual": optical_residual,
            **residual,
        }
        row.update(intensity_metrics(intensity, grid))
        row["theta_max"] = float(asnumpy(xp.max(theta)))
        history.append(row)

        if (
            field_rel < float(request.tol_field)
            and dtheta_rms < float(request.tol_theta)
            and optical_residual < float(request.tol_field)
            and residual["residual_rms"] < float(request.tol_residual_rms)
            and residual["residual_max"] < float(request.tol_residual_max)
        ):
            converged = True
            break

    # Alternate the exact discrete TD theta fixed-point solve with an exact
    # transverse optical eigensolve so the returned (A, theta) pair is jointly
    # self-consistent for the production TD operators.
    theta_step = make_zcoupled_theta_step(
        runtime,
        theta_dt=7.5e-4,
        mobility=1.0,
        gamma_z=0.0,
        max_iter=4,
        tol_update=1e-6,
    )
    final_theta_polish_steps = 0
    final_theta_update_rms = float("inf")
    final_coupled_polish_cycles = 0

    for final_coupled_polish_cycles in range(1, 6):
        intensity = total_intensity(A, coherent=coherent, xp=xp)

        for _ in range(10000):
            theta_new = theta_step(
                theta,
                intensity,
                theta,
                theta,
                0,
            )
            theta_new = _project_theta_parity(theta_new, request.mode)
            theta_new = xp.clip(
                theta_new,
                float(runtime.bias.theta_clamp[0]),
                float(runtime.bias.theta_clamp[1]),
            )
            theta_new[0, :] = request.base.bias.theta_bc
            theta_new[-1, :] = request.base.bias.theta_bc

            dtheta = theta_new - theta
            final_theta_update_rms = float(
                asnumpy(xp.sqrt(xp.mean(dtheta * dtheta)))
            )
            theta = theta_new
            final_theta_polish_steps += 1
            if final_theta_update_rms < 1e-9:
                break

        A_np, beta, optical_residual = _transverse_eigenpair(
            A,
            theta,
            beta_symbol,
            grid=grid,
            mode=request.mode,
            ne=ne,
            no=no,
            n_ref=n_ref,
            wavelength_um=wavelength_um,
            eig_tol=eig_tol,
            eig_maxiter=eig_maxiter,
        )
        A_exact = xp.asarray(A_np, dtype=A.dtype)[None, :, :]
        A_exact = _normalize_power(
            A_exact,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            xp=xp,
        )
        dxdy = float(grid.dx_um) * float(grid.dy_um)
        overlap = xp.sum(xp.conj(A) * A_exact) * dxdy
        A_exact = A_exact * xp.exp(-1j * xp.angle(overlap))
        A = _project_field_parity(A_exact, request.mode)
        A = _normalize_power(
            A,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            xp=xp,
        )

        # Re-evaluate the TD theta residual using the newly updated eigenmode.
        intensity = total_intensity(A, coherent=coherent, xp=xp)
        theta_probe = theta_step(
            theta,
            intensity,
            theta,
            theta,
            0,
        )
        theta_probe = _project_theta_parity(theta_probe, request.mode)
        theta_probe = xp.clip(
            theta_probe,
            float(runtime.bias.theta_clamp[0]),
            float(runtime.bias.theta_clamp[1]),
        )
        theta_probe[0, :] = request.base.bias.theta_bc
        theta_probe[-1, :] = request.base.bias.theta_bc
        dtheta_probe = theta_probe - theta
        final_theta_update_rms = float(
            asnumpy(xp.sqrt(xp.mean(dtheta_probe * dtheta_probe)))
        )
        if final_theta_update_rms < 1e-9:
            break

    elapsed = _time.perf_counter() - t0
    intensity = total_intensity(A, coherent=coherent, xp=xp)
    metrics = intensity_metrics(intensity, grid)
    metrics.update(
        residual_theta_static(
            theta,
            intensity,
            b=runtime.b,
            bi=runtime.bi,
            dx=grid.du,
            dy=grid.dv,
            theta_bc=request.base.bias.theta_bc,
            xp=xp,
        )
    )
    if history:
        metrics.update(history[-1])

    metrics.update(
        {
            "Nx": grid.Nx,
            "Ny": grid.Ny,
            "Nz": grid.Nz,
            "outer_steps": len(history),
            "theta_steps_per_outer": int(request.theta_steps_per_outer),
            "field_mix": float(request.field_mix),
            "theta_mix": float(request.theta_mix),
            "converged": bool(converged),
            "convergence_status": "converged" if converged else "max_outer_reached",
            "target_power_mW": float(physical_power_mW),
            "physical_power_mW": float(physical_power_mW),
            "normalized_field_integral_target": float(target_power),
            "mode": _canonical_mode(request.mode),
            "solver": "transverse_eigen",
            "b": float(runtime.b),
            "bi": float(runtime.bi),
            "n_ref": float(n_ref),
            "elapsed_s": float(elapsed),
            "theta_max": float(asnumpy(xp.max(theta))),
            "used_initial_A": bool(request.initial_A is not None),
            "used_initial_theta": bool(request.initial_theta is not None),
            "optical_residual": float(optical_residual),
            "final_theta_polish_steps": int(final_theta_polish_steps),
            "final_theta_update_rms": float(final_theta_update_rms),
            "final_coupled_polish_cycles": int(final_coupled_polish_cycles),
        }
    )

    return SolitonResult(
        metrics=metrics,
        mode=request.mode,
        samples=history,
        A=A,
        theta=theta,
        intensity=intensity,
        history=history,
        converged=converged,
    )


def polish_soliton(
    request: SolitonRequest,
    seed: SolitonResult,
    *,
    max_outer: int | None = None,
    field_mix: float | None = None,
    theta_mix: float | None = None,
) -> SolitonResult:
    """Polish an existing soliton with the dz-independent transverse solver."""
    updates = {
        "initial_A": seed.A,
        "initial_theta": seed.theta,
    }
    if max_outer is not None:
        updates["max_outer"] = int(max_outer)
    if field_mix is not None:
        updates["field_mix"] = float(field_mix)
    if theta_mix is not None:
        updates["theta_mix"] = float(theta_mix)

    return run_soliton(replace(request, **updates))


__all__ = ["SolitonRequest", "SolitonResult", "run_soliton", "polish_soliton"]

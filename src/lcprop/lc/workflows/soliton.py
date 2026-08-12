"""Clean LCProp soliton workflow."""

from __future__ import annotations

from typing import Any
import math
import time as _time

from lcprop.lc.requests import SolitonRequest
from lcprop.lc.results import SolitonResult
from lcprop.core.backend import asnumpy, synchronize
from lcprop.core.derived import compute_neff
from lcprop.lc.propagation import advance_slice
from lcprop.optics.splitstep import total_intensity
from lcprop.optics.substeps import build_optical_substep_kernel
from lcprop.lc.theta_picard import cn_trapezoid_picard_step
from lcprop.algorithms.thomas import solve_const_offdiag_batched
from lcprop.lc.diagnostics import residual_theta_static
from lcprop.products.diagnostics import intensity_metrics
from lcprop.lc.workflows.runtime import build_runtime_components
from lcprop.core.execution import RunProgress


def _target_power(beams) -> float:
    return float(sum(float(ch.power_mW) for ch in beams.channels))


def _normalize_power(
    A,
    *,
    target_power: float,
    grid,
    coherent: bool,
    xp,
    coherence_groups=None,
):
    """Normalize the sum of per-channel integrals, not grouped interference.

    The normalized field integral is independent of coherent cross terms;
    physical total power remains in ``runtime.bi``.
    """
    p = xp.sum(xp.abs(A) ** 2) * float(grid.dx_um) * float(grid.dy_um)
    scale = xp.sqrt(float(target_power) / (p + xp.asarray(1e-300, dtype=p.dtype)))
    return A * scale


def _prepare_initial_A(
    initial_A,
    launch,
    *,
    grid,
    target_power: float,
    coherent: bool,
    xp,
    coherence_groups=None,
):
    if initial_A is None:
        A = launch.A0.copy()
    else:
        A = xp.asarray(initial_A, dtype=launch.A0.dtype).copy()
        if A.shape != launch.A0.shape:
            raise ValueError(f"initial_A shape {A.shape} does not match {launch.A0.shape}")
    return _normalize_power(
        A,
        target_power=target_power,
        grid=grid,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )


def _prepare_initial_theta(initial_theta, bias, *, xp):
    if initial_theta is None:
        theta = bias.theta_2d.copy()
    else:
        theta = xp.asarray(initial_theta, dtype=bias.theta_2d.dtype).copy()
        if theta.ndim == 3:
            theta = theta[int(theta.shape[0]) // 2].copy()
        if theta.shape != bias.theta_2d.shape:
            raise ValueError(f"initial_theta shape {theta.shape} does not match {bias.theta_2d.shape}")
    return theta


def _canonical_mode(mode: str) -> str:
    mode = str(mode).upper()
    aliases = {
        "TEM00": "00",
        "TEM10": "10",
        "TEM01": "01",
        "TEM11": "11",
        "CUSTOM": "custom",
    }
    return aliases.get(mode, mode)


def _mode_seed_factor(grid, beams, mode: str, *, xp):
    """Return the Hermite-like seed factor for the requested transverse mode."""
    mode = _canonical_mode(mode)
    if mode in ("00", "custom"):
        return None

    if not beams.channels:
        return None

    ch0 = beams.channels[0]
    wx = max(1e-300, float(getattr(ch0, "waist_x_um", 1.0)))
    wy = max(1e-300, float(getattr(ch0, "waist_y_um", wx)))

    X = grid.x_um[:, None]
    Y = grid.y_um[None, :]

    if mode == "10":
        return X / wx
    if mode == "01":
        return Y / wy
    if mode == "11":
        return (X / wx) * (Y / wy)
    return None


def _apply_mode_seed(A, *, grid, beams, mode: str, xp):
    """Apply the legacy Hermite-like nodal seed to an initial optical field."""
    factor = _mode_seed_factor(grid, beams, mode, xp=xp)
    if factor is None:
        return A
    if A.ndim == 2:
        return A * factor
    return A * factor[None, :, :]


def _sym_x_even(F):
    return 0.5 * (F + F[..., ::-1, :])


def _sym_x_odd(F):
    return 0.5 * (F - F[..., ::-1, :])


def _sym_y_even(F):
    return 0.5 * (F + F[..., :, ::-1])


def _sym_y_odd(F):
    return 0.5 * (F - F[..., :, ::-1])


def _project_field_parity(A, mode: str):
    """Project optical field onto the requested x/y parity family."""
    mode = _canonical_mode(mode)
    if mode == "custom":
        return A
    if mode == "00":
        return _sym_y_even(_sym_x_even(A))
    if mode == "10":
        return _sym_y_even(_sym_x_odd(A))
    if mode == "01":
        return _sym_y_odd(_sym_x_even(A))
    if mode == "11":
        return _sym_y_odd(_sym_x_odd(A))
    return A


def _project_theta_parity(theta, mode: str):
    """Project director field onto the even symmetry implied by intensity."""
    mode = _canonical_mode(mode)
    if mode == "custom":
        return theta
    # For scalar LC response, theta is driven by intensity and remains even
    # across every axis in which the optical field has a fixed parity.
    return _sym_y_even(_sym_x_even(theta))


def _project_mode_parity(
    A,
    theta,
    mode: str,
    *,
    grid,
    target_power: float,
    coherent: bool,
    theta_clamp,
    theta_bc: float,
    xp,
    coherence_groups=None,
):
    """Project A/theta onto the selected mode family and restore constraints."""
    mode = _canonical_mode(mode)
    if mode == "custom":
        return A, theta

    A = _project_field_parity(A, mode)
    A = _normalize_power(
        A,
        target_power=target_power,
        grid=grid,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )

    theta = _project_theta_parity(theta, mode)
    theta = xp.clip(theta, float(theta_clamp[0]), float(theta_clamp[1]))
    theta[0, :] = float(theta_bc)
    theta[-1, :] = float(theta_bc)
    return A, theta


def _field_difference(A, B, *, grid, xp) -> tuple[float, float]:
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    ov = xp.sum(xp.conj(A) * B) * dxdy
    pA = xp.sum(xp.abs(A) ** 2) * dxdy
    pB = xp.sum(xp.abs(B) ** 2) * dxdy
    ovn = ov / xp.sqrt(pA * pB + 1e-300)
    B_aligned = B * xp.exp(-1j * xp.angle(ovn))
    rel = xp.sqrt(xp.sum(xp.abs(B_aligned - A) ** 2) * dxdy / (pA + 1e-300))
    return float(asnumpy(rel)), float(asnumpy(xp.abs(ovn)))


def make_beta_symbol(grid, *, wavelength_um: float, n_ref: float, subtract_carrier: bool = True):
    xp = grid.xp
    q0 = 2.0 * math.pi * float(n_ref) / float(wavelength_um)
    arg = 1.0 - (float(wavelength_um) / float(n_ref)) ** 2 * grid.fxy2_um
    mask = arg > 0.0

    q = xp.zeros_like(grid.fxy2_um, dtype=grid.real_dtype)
    q[mask] = (q0 * xp.sqrt(arg[mask])).astype(grid.real_dtype, copy=False)

    if subtract_carrier:
        q = q - xp.asarray(q0, dtype=q.dtype)

    return q, mask


def optical_potential(theta, *, ne: float, no: float, n_ref: float, wavelength_um: float, xp):
    neff = compute_neff(theta, ne=ne, no=no, xp=xp)
    k0 = 2.0 * math.pi / float(wavelength_um)
    return k0 * (neff - float(n_ref))


def apply_optical_eigen_operator(A, theta, beta_symbol, *, grid, ne: float, no: float, n_ref: float, wavelength_um: float):
    xp = grid.xp
    Ahat = xp.fft.fft2(A)
    HA = xp.fft.ifft2(beta_symbol * Ahat)
    HA = HA + optical_potential(
        theta,
        ne=ne,
        no=no,
        n_ref=n_ref,
        wavelength_um=wavelength_um,
        xp=xp,
    ) * A
    return HA


def rayleigh_beta(A, theta, beta_symbol, *, grid, ne: float, no: float, n_ref: float, wavelength_um: float) -> float:
    xp = grid.xp
    area = float(grid.dx_um) * float(grid.dy_um)
    HA = apply_optical_eigen_operator(
        A,
        theta,
        beta_symbol,
        grid=grid,
        ne=ne,
        no=no,
        n_ref=n_ref,
        wavelength_um=wavelength_um,
    )
    num = xp.vdot(A.ravel(), HA.ravel()) * area
    den = xp.vdot(A.ravel(), A.ravel()) * area
    return float(asnumpy(xp.real(num / den)))


def run_soliton(
    request: SolitonRequest,
    *,
    cancellation_token=None,
    progress_callback=None,
) -> SolitonResult:
    request.validate()

    runtime = build_runtime_components(request.base, theta_dt=7.5e-4, mobility=1.0)
    grid = runtime.grid
    xp = grid.xp

    physical_power_mW = _target_power(request.base.beams)
    target_power = 1.0
    coherent = runtime.coherent
    coherence_groups = runtime.coherence_groups

    A = _prepare_initial_A(
        request.initial_A,
        runtime.launch,
        grid=grid,
        target_power=target_power,
        coherent=coherent,
        coherence_groups=coherence_groups,
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
            coherence_groups=coherence_groups,
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
        coherence_groups=coherence_groups,
        theta_clamp=runtime.bias.theta_clamp,
        theta_bc=request.base.bias.theta_bc,
        xp=xp,
    )

    n_ref = float(asnumpy(compute_neff(
        theta,
        ne=request.base.material.ne,
        no=request.base.material.no,
        xp=xp,
    )).mean())

    n_ref_beta = float(asnumpy(compute_neff(
        runtime.bias.theta_2d,
        ne=request.base.material.ne,
        no=request.base.material.no,
        xp=xp,
    )).mean())

    wavelength_um = runtime.wavelength_um

    optical_substeps, h_sub = build_optical_substep_kernel(
        grid.fxy2_um,
        dz_um=grid.dz_um,
        propagation_wavelength_um=wavelength_um,
        active_wavelengths_um=(
            channel.wavelength_um
            for channel in request.base.beams.channels
        ),
        n_ref=n_ref,
        enabled=request.base.runtime.optical_substeps_enabled,
        dn_max_est=request.base.runtime.optical_dn_max_est,
        max_phase_per_substep_rad=(
            request.base.runtime.optical_max_phase_per_substep_rad
        ),
        max_substeps=request.base.runtime.optical_max_substeps,
        xp=xp,
    )
    Nsub = optical_substeps.Nsub

    beta_symbol, _ = make_beta_symbol(
        grid,
        wavelength_um=wavelength_um,
        n_ref=n_ref_beta,
        subtract_carrier=True,
    )

    cn = runtime.cn

    def relax_theta(theta_in, intensity):
        out = theta_in
        for _ in range(int(request.theta_steps_per_outer)):
            out_new = cn_trapezoid_picard_step(
                out,
                intensity,
                intensity,
                b=runtime.b,
                bi=runtime.bi,
                dt=7.5e-4,
                mobility=1.0,
                dx=runtime.normalization.du,
                dy=runtime.normalization.dv,
                s=cn.s,
                off=cn.off,
                diag=cn.diag,
                max_iter=4,
                tol_update=1e-6,
                clamp=runtime.bias.theta_clamp,
                tridiag_solver=solve_const_offdiag_batched,
                xp=xp,
            )
            out = (1.0 - float(request.theta_mix)) * out + float(request.theta_mix) * out_new
            out[0, :] = request.base.bias.theta_bc
            out[-1, :] = request.base.bias.theta_bc
        return out

    def propagate(A_in, theta_in):
        A_out = A_in.copy()
        for _k in range(int(grid.Nz)):
            advance_slice(
                A_out,
                theta_in,
                kernel=h_sub,
                dz=float(grid.dz_um),
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=request.base.material.ne,
                no=request.base.material.no,
                Nsub=Nsub,
                xp=xp,
            )
        return A_out

    history: list[dict] = []
    converged = False
    stopped = False

    synchronize(xp)
    t0 = _time.perf_counter()

    intensity = total_intensity(A, coherent=coherent, coherence_groups=coherence_groups, xp=xp)

    for outer in range(int(request.max_outer)):
        if cancellation_token is not None and cancellation_token.is_cancelled():
            stopped = True
            break
        A_prev = A.copy()
        theta_prev = theta.copy()

        intensity = total_intensity(A, coherent=coherent, coherence_groups=coherence_groups, xp=xp)
        theta = relax_theta(theta, intensity)

        A, theta = _project_mode_parity(
            A,
            theta,
            request.mode,
            grid=grid,
            target_power=target_power,
            coherent=coherent,
            coherence_groups=coherence_groups,
            theta_clamp=runtime.bias.theta_clamp,
            theta_bc=request.base.bias.theta_bc,
            xp=xp,
        )

        A_end = propagate(A, theta)
        A_end = _normalize_power(
            A_end,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            coherence_groups=coherence_groups,
            xp=xp,
        )

        dxdy = float(grid.dx_um) * float(grid.dy_um)
        ov = xp.sum(xp.conj(A) * A_end) * dxdy
        phase = xp.angle(ov)
        A_candidate = A_end * xp.exp(-1j * phase)

        A = (1.0 - float(request.field_mix)) * A + float(request.field_mix) * A_candidate
        A = _normalize_power(
            A,
            target_power=target_power,
            grid=grid,
            coherent=coherent,
            coherence_groups=coherence_groups,
            xp=xp,
        )

        A, theta = _project_mode_parity(
            A,
            theta,
            request.mode,
            grid=grid,
            target_power=target_power,
            coherent=coherent,
            coherence_groups=coherence_groups,
            theta_clamp=runtime.bias.theta_clamp,
            theta_bc=request.base.bias.theta_bc,
            xp=xp,
        )

        field_rel, overlap_abs = _field_difference(A_prev, A, grid=grid, xp=xp)

        dtheta = theta - theta_prev
        dtheta_rms = float(asnumpy(xp.sqrt(xp.mean(dtheta * dtheta))))
        dtheta_max = float(asnumpy(xp.max(xp.abs(dtheta))))

        intensity = total_intensity(A, coherent=coherent, coherence_groups=coherence_groups, xp=xp)

        resid = residual_theta_static(
            theta,
            intensity,
            b=runtime.b,
            bi=runtime.bi,
            dx=runtime.normalization.du,
            dy=runtime.normalization.dv,
            theta_bc=request.base.bias.theta_bc,
            xp=xp,
        )

        beta = rayleigh_beta(
            A,
            theta,
            beta_symbol,
            grid=grid,
            ne=request.base.material.ne,
            no=request.base.material.no,
            n_ref=n_ref_beta,
            wavelength_um=wavelength_um,
        )

        row = {
            "outer": int(outer + 1),
            "field_rel": field_rel,
            "overlap_abs": overlap_abs,
            "dtheta_rms": dtheta_rms,
            "dtheta_max": dtheta_max,
            "beta": beta,
            **resid,
            **optical_substeps.diagnostics(),
        }
        row.update(intensity_metrics(intensity, grid))
        row["theta_max"] = float(asnumpy(xp.max(theta)))
        history.append(row)

        if progress_callback is not None:
            partial = SolitonResult(
                metrics={**row, "convergence_status": "running"},
                mode=request.mode,
                samples=list(history),
                A=asnumpy(A).copy(),
                theta=asnumpy(theta).copy(),
                intensity=asnumpy(intensity).copy(),
                history=list(history),
                converged=False,
                status="running",
                completed_iterations=len(history),
                total_iterations=int(request.max_outer),
                request=request,
                usable_as_initial_condition=True,
            )
            progress_callback(
                RunProgress(
                    workflow="soliton",
                    status="running",
                    completed_units=len(history),
                    total_units=int(request.max_outer),
                    current_coordinate=float(len(history)),
                    coordinate_name="iteration",
                    coordinate_unit="",
                    elapsed_wall_time=_time.perf_counter() - t0,
                    latest_field_state=partial,
                    checkpoint_available=True,
                    diagnostics=dict(row),
                )
            )

        if (
            field_rel < float(request.tol_field)
            and dtheta_rms < float(request.tol_theta)
            and resid["residual_rms"] < float(request.tol_residual_rms)
            and resid["residual_max"] < float(request.tol_residual_max)
        ):
            converged = True
            break

    synchronize(xp)
    elapsed = _time.perf_counter() - t0

    intensity = total_intensity(A, coherent=coherent, coherence_groups=coherence_groups, xp=xp)
    metrics = intensity_metrics(intensity, grid)
    metrics.update(residual_theta_static(
        theta,
        intensity,
        b=runtime.b,
        bi=runtime.bi,
        dx=runtime.normalization.du,
        dy=runtime.normalization.dv,
        theta_bc=request.base.bias.theta_bc,
        xp=xp,
    ))

    if history:
        for key in ("field_rel", "overlap_abs", "dtheta_rms", "dtheta_max", "beta"):
            if key in history[-1]:
                metrics[key] = history[-1][key]

    metrics.update({
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Nz": grid.Nz,
        "outer_steps": len(history),
        "theta_steps_per_outer": int(request.theta_steps_per_outer),
        "field_mix": float(request.field_mix),
        "theta_mix": float(request.theta_mix),
        "converged": bool(converged),
        "convergence_status": (
            "stopped" if stopped else ("converged" if converged else "max_outer_reached")
        ),
        "target_power_mW": float(physical_power_mW),
        "physical_power_mW": float(physical_power_mW),
        "normalized_field_integral_target": float(target_power),
        "mode": _canonical_mode(request.mode),
        "b": float(runtime.b),
        "bi": float(runtime.bi),
        "n_ref": float(n_ref),
        "Nsub": int(Nsub),
        **optical_substeps.diagnostics(),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(theta))),
        "used_initial_A": bool(request.initial_A is not None),
        "used_initial_theta": bool(request.initial_theta is not None),
    })

    return SolitonResult(
        metrics=metrics,
        mode=request.mode,
        samples=history,
        A=A,
        theta=theta,
        intensity=intensity,
        history=history,
        converged=converged,
        status="stopped" if stopped else "completed",
        completed_iterations=len(history),
        total_iterations=int(request.max_outer),
        request=request,
        usable_as_initial_condition=bool(history),
    )


__all__ = [
    "SolitonRequest",
    "SolitonResult",
    "run_soliton",
]

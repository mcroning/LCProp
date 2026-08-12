"""Static director torque-balance data preparation and Matplotlib rendering.

This module operates on a completed z-local static result. It does not run or
modify a solver, persist torque volumes, or depend on Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lcprop.lc.theta_cn import (
    laplacian_dirichletx_periody,
    static_director_residual,
    theta_drive,
)
from lcprop.core.backend import asnumpy


@dataclass(frozen=True)
class StaticTorqueBalanceData:
    """Fixed-x y-z cuts of the accepted static director balance."""

    x_index: int
    x_um: float
    y_um: np.ndarray
    z_um: np.ndarray
    elastic: np.ndarray
    drive: np.ndarray
    residual: np.ndarray
    relative_residual: np.ndarray
    relative_mask: np.ndarray
    torque_limit: float
    residual_limit: float
    relative_limit: float
    metrics: dict[str, float | int | str]


def _coordinates_from_grid_summary(
    grid_summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    nx = int(grid_summary["Nx"])
    ny = int(grid_summary["Ny"])
    dx_um = float(grid_summary["dx_um"])
    dy_um = float(grid_summary["dy_um"])
    x_um = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * dx_um
    y_um = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * dy_um
    return x_um, y_um


def _finite_abs_max(values: np.ndarray, *, name: str) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError(f"{name} contains no finite values")
    return float(np.max(finite))


def build_static_torque_balance_data(
    result,
    *,
    x_index: int,
) -> StaticTorqueBalanceData:
    """Build fixed-x torque-balance cuts from a z-local static result.

    The source arrays must both have shape ``(Nz, Nx, Ny)``:
    ``result.theta_final`` and ``result.theta_intensity_stack``. The latter is
    the accepted, theta-weighted optical midpoint intensity used by the final
    director residual at each z slice.
    """

    theta = np.asarray(asnumpy(result.theta_final))
    intensity_source = getattr(result, "theta_intensity_stack", None)
    if intensity_source is None:
        raise ValueError(
            "static torque balance requires theta_intensity_stack from a "
            "z-local self-consistent result"
        )
    intensity = np.asarray(asnumpy(intensity_source))

    if theta.ndim != 3:
        raise ValueError(
            f"theta_final must have shape (Nz, Nx, Ny), got {theta.shape}"
        )
    if intensity.shape != theta.shape:
        raise ValueError(
            "theta_intensity_stack must match theta_final shape; "
            f"got {intensity.shape} and {theta.shape}"
        )

    nz, nx, ny = theta.shape
    grid_summary = result.grid_summary
    if (int(grid_summary["Nx"]), int(grid_summary["Ny"])) != (nx, ny):
        raise ValueError("grid_summary transverse shape does not match theta_final")
    if int(grid_summary.get("Nz", nz)) != nz:
        raise ValueError("grid_summary Nz does not match theta_final")

    ix = int(x_index)
    if ix <= 0 or ix >= nx - 1:
        raise ValueError(
            f"x_index must select an interior director row in [1, {nx - 2}]"
        )

    x_um, y_um = _coordinates_from_grid_summary(grid_summary)
    dz_um = float(grid_summary["dz_um"])
    z_mid_um = (np.arange(nz, dtype=float) + 0.5) * dz_um

    b = float(result.bias_summary["b"])
    coupling_summary = getattr(result, "coupling_summary", {})
    if "bi_um2" not in coupling_summary:
        raise ValueError("static result does not contain coupling_summary['bi_um2']")
    bi = float(coupling_summary["bi_um2"])
    du = float(grid_summary["du"])
    dv = float(grid_summary["dv"])

    elastic_cut = np.empty((nz, ny), dtype=np.float64)
    drive_cut = np.empty((nz, ny), dtype=np.float64)
    residual_cut = np.empty((nz, ny), dtype=np.float64)

    # Each full 2-D term is temporary. Only the requested x row is retained.
    for iz in range(nz):
        theta_slice = theta[iz].astype(np.float64, copy=False)
        intensity_slice = intensity[iz].astype(np.float64, copy=False)
        elastic = laplacian_dirichletx_periody(theta_slice, du, dv)
        drive = theta_drive(theta_slice, intensity_slice, b=b, bi=bi)
        residual = static_director_residual(
            theta_slice,
            intensity_slice,
            b=b,
            bi=bi,
            dx=du,
            dy=dv,
        )
        elastic_cut[iz] = elastic[ix, :]
        drive_cut[iz] = drive[ix, :]
        residual_cut[iz] = residual[ix, :]

    expected_shape = (nz, ny)
    if elastic_cut.shape != expected_shape:
        raise AssertionError("elastic fixed-x cut shape is not (Nz, Ny)")
    if drive_cut.shape != expected_shape:
        raise AssertionError("drive fixed-x cut shape is not (Nz, Ny)")
    if residual_cut.shape != expected_shape:
        raise AssertionError("residual fixed-x cut shape is not (Nz, Ny)")
    if len(z_mid_um) != nz or len(y_um) != ny:
        raise AssertionError("physical coordinates do not match fixed-x cut shape")

    torque_limit = max(
        _finite_abs_max(elastic_cut, name="elastic torque"),
        _finite_abs_max(drive_cut, name="drive torque"),
    )
    residual_limit = _finite_abs_max(residual_cut, name="static residual")
    denominator = np.abs(elastic_cut) + np.abs(drive_cut)
    source_dtype = np.result_type(theta.dtype, intensity.dtype)
    if not np.issubdtype(source_dtype, np.floating):
        source_dtype = np.dtype(np.float64)
    floor = max(
        100.0 * np.finfo(source_dtype).eps * torque_limit,
        1.0e-12 * torque_limit,
    )
    relative_mask = denominator > floor
    relative = np.full_like(residual_cut, np.nan, dtype=float)
    relative[relative_mask] = (
        residual_cut[relative_mask] / denominator[relative_mask]
    )
    finite_relative = np.abs(relative[np.isfinite(relative)])
    if finite_relative.size:
        relative_limit = float(np.percentile(finite_relative, 99.0))
    else:
        relative_limit = 1.0
    if not np.isfinite(relative_limit) or relative_limit == 0.0:
        relative_limit = 1.0

    cancellation_error = float(
        np.max(np.abs((elastic_cut + drive_cut) - residual_cut))
    )
    iz_max, iy_max = np.unravel_index(
        np.nanargmax(np.abs(residual_cut)),
        residual_cut.shape,
    )
    slice_summaries = tuple(getattr(result, "slice_summaries", ()) or ())
    converged_slices = sum(bool(item.converged) for item in slice_summaries)
    total_slices = len(slice_summaries) if slice_summaries else nz

    metrics: dict[str, float | int | str] = {
        "selected_x_index": ix,
        "selected_x_um": float(x_um[ix]),
        "volume_shape": str(theta.shape),
        "cut_shape": str(expected_shape),
        "z_sampling": "accepted-slice midpoint",
        "b": b,
        "bi_um2": bi,
        "relative_denominator_floor": float(floor),
        "max_abs_elastic": _finite_abs_max(elastic_cut, name="elastic torque"),
        "max_abs_drive": _finite_abs_max(drive_cut, name="drive torque"),
        "max_abs_residual": residual_limit,
        "residual_rms": float(np.sqrt(np.nanmean(residual_cut**2))),
        "max_displayed_abs_relative_residual": relative_limit,
        "converged_slices": converged_slices,
        "total_slices": total_slices,
        "cancellation_error": cancellation_error,
        "residual_argmax_z_index": int(iz_max),
        "residual_argmax_z_um": float(z_mid_um[iz_max]),
        "residual_argmax_y_index": int(iy_max),
        "residual_argmax_y_um": float(y_um[iy_max]),
        "residual_argmax_value": float(residual_cut[iz_max, iy_max]),
    }

    return StaticTorqueBalanceData(
        x_index=ix,
        x_um=float(x_um[ix]),
        y_um=y_um,
        z_um=z_mid_um,
        elastic=elastic_cut,
        drive=drive_cut,
        residual=residual_cut,
        relative_residual=relative,
        relative_mask=relative_mask,
        torque_limit=torque_limit,
        residual_limit=residual_limit,
        relative_limit=relative_limit,
        metrics=metrics,
    )


def _summary_text(data: StaticTorqueBalanceData) -> str:
    metrics = data.metrics
    return (
        f"x index {data.x_index}; x = {data.x_um:.6g} µm; "
        f"volume {metrics['volume_shape']}; cut {metrics['cut_shape']}; "
        "z: accepted-slice midpoint\n"
        f"max |T_elastic| = {metrics['max_abs_elastic']:.6g}; "
        f"max |T_drive| = {metrics['max_abs_drive']:.6g}; "
        f"max |R| = {metrics['max_abs_residual']:.6g}; "
        f"RMS R = {metrics['residual_rms']:.6g}; "
        f"displayed |R_rel| ≤ {data.relative_limit:.6g}\n"
        f"converged {metrics['converged_slices']}/{metrics['total_slices']}; "
        f"floor = {metrics['relative_denominator_floor']:.3e}; "
        f"cancellation error = {metrics['cancellation_error']:.3e}; "
        f"max |R| at (iz={metrics['residual_argmax_z_index']}, "
        f"z={metrics['residual_argmax_z_um']:.6g} µm, "
        f"iy={metrics['residual_argmax_y_index']}, "
        f"y={metrics['residual_argmax_y_um']:.6g} µm)"
    )


def plot_static_torque_balance(
    data: StaticTorqueBalanceData,
    *,
    figure=None,
):
    """Render the four-panel static torque-balance figure.

    Returns ``(figure, axes)`` where ``axes`` has shape ``(2, 2)``.
    Matplotlib is imported lazily so unrelated LCProp imports remain safe.
    """

    import matplotlib.pyplot as plt

    if figure is None:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8),
            constrained_layout=True,
        )
    else:
        fig = figure
        fig.clear()
        fig.set_constrained_layout(True)
        axes = fig.subplots(2, 2)

    extent = [
        float(data.y_um[0]),
        float(data.y_um[-1]),
        float(data.z_um[0]),
        float(data.z_um[-1]),
    ]
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="#eeeeee")
    panels = (
        (axes[0, 0], data.elastic, "Transverse elastic torque", data.torque_limit, "dimensionless torque"),
        (axes[0, 1], data.drive, "Optical and bias drive torque", data.torque_limit, "dimensionless torque"),
        (axes[1, 0], data.residual, "Static residual", data.residual_limit, "residual"),
        (axes[1, 1], data.relative_residual, "Relative residual", data.relative_limit, "relative residual"),
    )

    for ax, values, title, limit, colorbar_label in panels:
        display_limit = float(limit)
        if not np.isfinite(display_limit) or display_limit <= 0.0:
            display_limit = 1.0
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=-display_limit,
            vmax=display_limit,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("y (µm)")
        ax.set_ylabel("z (µm)")
        cbar = fig.colorbar(image, ax=ax, pad=0.02)
        cbar.set_label(colorbar_label)

    fig.suptitle(
        f"LC static torque balance at x = {data.x_um:.6g} µm\n"
        "accepted-slice midpoint sampling"
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.13, 1.0, 0.88))
    fig.text(
        0.5,
        0.015,
        _summary_text(data),
        ha="center",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    return fig, axes


__all__ = [
    "StaticTorqueBalanceData",
    "build_static_torque_balance_data",
    "plot_static_torque_balance",
]

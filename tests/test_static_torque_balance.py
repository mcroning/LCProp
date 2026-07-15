from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lcprop.algorithms.theta_cn import laplacian_dirichletx_periody
from lcprop.products.static_torque_balance import (
    build_static_torque_balance_data,
    plot_static_torque_balance,
)


def _synthetic_result(*, nz: int = 4, zero: bool = False):
    nx, ny = 5, 7
    du, dv = 0.5, 0.75
    b, bi = 2.0, 4.0
    theta = np.zeros((nz, nx, ny), dtype=np.float64)
    intensity = np.zeros_like(theta)

    if not zero:
        base_x = np.asarray([0.0, 0.24, 0.35, 0.19, 0.0])[:, None]
        y_phase = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)[None, :]
        x_scale = np.asarray([0.0, 0.8, 1.0, 1.25, 0.0])[:, None]
        for iz in range(nz):
            theta[iz] = base_x * (1.0 + 0.03 * iz) * (
                1.0 + 0.04 * np.cos(y_phase)
            )
            elastic = laplacian_dirichletx_periody(theta[iz], du, dv)
            target_residual = (
                1.0e-4
                * (iz + 1)
                * x_scale
                * np.sin(y_phase + 0.2 * iz)
            )
            sin_two_theta = np.sin(2.0 * theta[iz, 1:-1])
            intensity[iz, 1:-1] = (
                (target_residual[1:-1] - elastic[1:-1])
                / (bi * sin_two_theta)
                - b / bi
            )

    summaries = tuple(SimpleNamespace(converged=(iz != nz - 1)) for iz in range(nz))
    return SimpleNamespace(
        theta_final=theta,
        theta_intensity_stack=intensity,
        grid_summary={
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "dx_um": 2.0,
            "dy_um": 3.0,
            "dz_um": 5.0,
            "du": du,
            "dv": dv,
            "x_aperture_um": 10.0,
            "y_aperture_um": 21.0,
            "z_length_um": 5.0 * nz,
            "real_dtype": "float64",
        },
        bias_summary={"b": b},
        coupling_summary={"bi_um2": bi},
        slice_summaries=summaries,
    )


def test_fixed_x_cut_is_z_x_y_indexing_without_transpose():
    result = _synthetic_result()
    data = build_static_torque_balance_data(result, x_index=2)
    expected = np.stack(
        [
            laplacian_dirichletx_periody(theta_slice, 0.5, 0.75)[2, :]
            for theta_slice in result.theta_final
        ]
    )

    assert data.elastic.shape == (4, 7)
    assert data.drive.shape == (4, 7)
    assert data.residual.shape == (4, 7)
    assert np.array_equal(data.elastic, expected)
    assert data.elastic[3, 5] == expected[3, 5]


def test_changing_x_index_changes_expected_x_row():
    result = _synthetic_result()
    left = build_static_torque_balance_data(result, x_index=1)
    right = build_static_torque_balance_data(result, x_index=3)

    assert left.x_um < right.x_um
    assert not np.array_equal(left.elastic, right.elastic)
    assert not np.array_equal(left.drive, right.drive)


def test_midpoint_z_coordinates_use_physical_microns():
    result = _synthetic_result(nz=100)
    data = build_static_torque_balance_data(result, x_index=2)

    assert len(data.z_um) == 100
    assert data.z_um[0] == 2.5
    assert data.z_um[-1] == 497.5


def test_relative_residual_masks_negligible_torque():
    data = build_static_torque_balance_data(
        _synthetic_result(nz=3, zero=True),
        x_index=2,
    )

    assert not np.any(data.relative_mask)
    assert np.isnan(data.relative_residual).all()
    assert data.relative_limit == 1.0


def test_balance_cancellation_and_argmax_coordinate_mapping():
    result = _synthetic_result()
    data = build_static_torque_balance_data(result, x_index=2)
    metrics = data.metrics

    assert metrics["cancellation_error"] < 1.0e-14
    assert data.residual_limit < data.torque_limit
    assert np.isfinite(data.relative_residual[data.relative_mask]).all()
    iz = int(metrics["residual_argmax_z_index"])
    iy = int(metrics["residual_argmax_y_index"])
    assert metrics["residual_argmax_z_um"] == data.z_um[iz]
    assert metrics["residual_argmax_y_um"] == data.y_um[iy]
    assert metrics["residual_argmax_value"] == data.residual[iz, iy]
    assert metrics["converged_slices"] == 3
    assert metrics["total_slices"] == 4


def test_plot_clims_axes_and_source_immutability():
    result = _synthetic_result()
    theta_before = result.theta_final.copy()
    intensity_before = result.theta_intensity_stack.copy()
    data = build_static_torque_balance_data(result, x_index=2)

    fig, axes = plot_static_torque_balance(data)
    upper_clims = (
        axes[0, 0].images[0].get_clim(),
        axes[0, 1].images[0].get_clim(),
    )
    residual_clim = axes[1, 0].images[0].get_clim()

    assert upper_clims[0] == upper_clims[1]
    assert upper_clims[0] == (-data.torque_limit, data.torque_limit)
    assert residual_clim == (-data.residual_limit, data.residual_limit)
    assert residual_clim != upper_clims[0]
    assert axes[0, 0].images[0].get_extent() == [
        data.y_um[0],
        data.y_um[-1],
        data.z_um[0],
        data.z_um[-1],
    ]
    assert [ax.get_title() for ax in axes.ravel()] == [
        "Transverse elastic torque",
        "Optical and bias drive torque",
        "Static residual",
        "Relative residual",
    ]
    assert all(ax.get_xlabel() == "y (µm)" for ax in axes.ravel())
    assert all(ax.get_ylabel() == "z (µm)" for ax in axes.ravel())
    assert np.array_equal(result.theta_final, theta_before)
    assert np.array_equal(result.theta_intensity_stack, intensity_before)
    plt.close(fig)

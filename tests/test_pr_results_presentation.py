from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.gui.views.image_pane import ImagePane
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.products import pr_transverse_static_result_to_run_data
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def static_products():
    request = PRTransverseStaticRunRequest(
        grid=GridSpec(
            Nx=16,
            Ny=12,
            x_aperture_um=32.0,
            y_aperture_um=24.0,
            dz_um=4.0,
            z_length_um=12.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633,
            waist_x_um=7.0,
            waist_y_um=5.0,
            tilt_x_rad_per_um=0.15,
            tilt_y_rad_per_um=-0.08,
            coherence_group="presentation",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            gain_length_product=1.0e-3,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=5),
        backend=BackendSpec(backend="numpy", precision="float64"),
    )
    result = run_pr_transverse_static(request)
    assert result.converged
    return result, pr_transverse_static_result_to_run_data(result)


def test_direction_cosine_spectrum_has_physical_axes_and_power_normalization():
    nx, ny = 16, 12
    dx_um, dy_um = 2.0, 3.0
    wavelength_um, refractive_index = 0.633, 2.4
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    A = np.empty((2, nx, ny), dtype=np.complex128)
    A[0] = np.exp(2j * np.pi * (2 * x / nx - y / ny))
    A[1] = 0.5 * np.exp(2j * np.pi * (-3 * x / nx + 2 * y / ny))

    spectrum = direction_cosine_spectrum(
        A,
        dx_um=dx_um,
        dy_um=dy_um,
        wavelength_um=wavelength_um,
        refractive_index=refractive_index,
        coherence_groups=("one", "two"),
    )

    expected_sx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_um)) * (
        wavelength_um / refractive_index
    )
    expected_sy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy_um)) * (
        wavelength_um / refractive_index
    )
    np.testing.assert_array_equal(spectrum.s_x, expected_sx)
    np.testing.assert_array_equal(spectrum.s_y, expected_sy)
    dsx = spectrum.s_x[1] - spectrum.s_x[0]
    dsy = spectrum.s_y[1] - spectrum.s_y[0]
    far_power = np.sum(spectrum.intensity) * dsx * dsy
    near_power = np.sum(np.abs(A) ** 2) * dx_um * dy_um
    assert far_power == pytest.approx(near_power, rel=2e-15)


def test_static_product_has_compact_longitudinal_and_selected_material_views(
    static_products,
):
    result, run_data = static_products
    fields = run_data.fields
    nx = result.grid_summary["Nx"]
    ny = result.grid_summary["Ny"]
    nz = result.grid_summary["Nz"]
    ix = int(run_data.diagnostics["presentation"].values["selected_x_index"])
    iy = int(run_data.diagnostics["presentation"].values["selected_y_index"])
    iz = int(run_data.diagnostics["presentation"].values["selected_z_index"])
    dz_um = float(result.grid_summary["dz_um"])

    assert run_data.geometry.z[0] == pytest.approx(0.5 * dz_um)
    assert run_data.geometry.z[iz] == pytest.approx((iz + 0.5) * dz_um)
    assert run_data.geometry.z[-1] == pytest.approx((nz - 0.5) * dz_um)
    assert (
        run_data.diagnostics["presentation"].values["selected_z_um"]
        == pytest.approx((iz + 0.5) * dz_um)
    )

    assert fields["optical_intensity_xz"].axes == ("z", "x")
    assert fields["optical_intensity_xz"].data.shape == (nz, nx)
    assert fields["optical_intensity_yz"].axes == ("z", "y")
    assert fields["optical_intensity_yz"].data.shape == (nz, ny)
    peak_reference = np.sum(
        np.max(np.abs(result.A_initial) ** 2, axis=(-2, -1))
    )
    expected_xz = (result.source_intensity_stack[:, :, iy] - 0.5) * peak_reference
    expected_yz = (result.source_intensity_stack[:, ix, :] - 0.5) * peak_reference
    np.testing.assert_allclose(fields["optical_intensity_xz"].data, expected_xz)
    np.testing.assert_allclose(fields["optical_intensity_yz"].data, expected_yz)

    np.testing.assert_array_equal(fields["E_x_xz"].data, fields["E_x"].data[:, :, iy])
    np.testing.assert_array_equal(fields["E_y_yz"].data, fields["E_y"].data[:, ix, :])
    for selected, volume in (
        ("selected_psi_plane", "psi"),
        ("selected_P_plane", "P"),
        ("selected_E_x_plane", "E_x"),
        ("selected_E_y_plane", "E_y"),
    ):
        field = fields[selected]
        assert field.source_volume_key == volume
        assert field.data.shape == (nx, ny)
        assert not field.data.flags.writeable
        np.testing.assert_array_equal(field.data, fields[volume].data[iz])


def test_far_field_products_share_axes_and_mask_without_mutation(static_products):
    result, run_data = static_products
    fields = run_data.fields
    linear = fields["far_field_intensity"]
    unmasked_before = np.asarray(linear.data).copy()
    log_db = fields["far_field_log_db"]
    masked = fields["far_field_carrier_masked"]

    assert linear.axes == log_db.axes == masked.axes == ("s_x", "s_y")
    np.testing.assert_array_equal(
        linear.coordinates["s_x"], log_db.coordinates["s_x"]
    )
    np.testing.assert_array_equal(
        linear.coordinates["s_y"], masked.coordinates["s_y"]
    )
    assert np.max(log_db.data) == pytest.approx(0.0, abs=1e-12)
    assert np.min(log_db.data) >= -120.0
    excluded = np.asarray(masked.data) == 0.0
    assert np.any(excluded)
    assert np.any(unmasked_before[excluded] > 0.0)
    np.testing.assert_array_equal(linear.data, unmasked_before)
    assert not np.shares_memory(linear.data, masked.data)

    dsx = linear.coordinates["s_x"][1] - linear.coordinates["s_x"][0]
    dsy = linear.coordinates["s_y"][1] - linear.coordinates["s_y"][0]
    presented_power = np.sum(linear.data) * dsx * dsy
    assert presented_power == pytest.approx(result.power_final, rel=2e-14)
    presentation = run_data.diagnostics["presentation"].values
    assert presentation["carrier_mask_role"] == "diagnostic_presentation_only"
    assert presentation["sparse_presentation_contract"] is True
    carrier = presentation["carrier_mask_regions"][0]
    k_medium = 2.0 * np.pi * 2.4 / 0.633
    assert carrier["center_s_x"] == pytest.approx(0.15 / k_medium)
    assert carrier["center_s_y"] == pytest.approx(-0.08 / k_medium)


def test_input_output_profiles_and_physical_image_extents(app, static_products):
    _result, run_data = static_products
    x_curve = run_data.curves["input_output_x_profile"]
    y_curve = run_data.curves["input_output_y_profile"]
    assert x_curve.y.shape == (len(run_data.geometry.x), 2)
    assert y_curve.y.shape == (len(run_data.geometry.y), 2)
    assert x_curve.series_labels == y_curve.series_labels == ("Input", "Output")

    pane = ImagePane()
    pane.set_run_data(run_data)
    index = pane.field_selector.findData("far_field_intensity")
    pane.field_selector.setCurrentIndex(index)
    field = run_data.fields["far_field_intensity"]
    assert tuple(pane.image_view.image.get_extent()) == pytest.approx((
        field.coordinates["s_x"][0],
        field.coordinates["s_x"][-1],
        field.coordinates["s_y"][0],
        field.coordinates["s_y"][-1],
    ))

    selected_positions = []
    pane.positionSelected.connect(
        lambda ix, iy: selected_positions.append((ix, iy))
    )
    pane._position_selected(2, 3)
    assert selected_positions == []

    index = pane.field_selector.findData("optical_intensity_xz")
    pane.field_selector.setCurrentIndex(index)
    assert tuple(pane.image_view.image.get_extent()) == pytest.approx((
        run_data.geometry.z[0],
        run_data.geometry.z[-1],
        run_data.geometry.x[0],
        run_data.geometry.x[-1],
    ))

    index = pane.field_selector.findData("selected_psi_plane")
    pane.field_selector.setCurrentIndex(index)
    pane._position_selected(2, 3)
    assert selected_positions == [(2, 3)]
    pane.close()

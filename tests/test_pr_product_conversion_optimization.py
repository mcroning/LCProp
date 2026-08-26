from __future__ import annotations

import numpy as np
import pytest

import lcprop.pr.transverse.products as product_module
from lcprop.pr.transverse.products import pr_transverse_static_result_to_run_data
from lcprop.pr.transverse.specs import PR_FULL_TRANSVERSE_PROFILE_V1
from lcprop.pr.transverse.static_workflow import PRTransverseStaticRunResult
from lcprop.pr.transverse.transport import state_from_potential


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_chunked_product_state_reconstruction_is_bitwise_identical(
    monkeypatch, dtype
):
    rng = np.random.default_rng(1407)
    potential = (0.03 * rng.standard_normal((7, 17, 19))).astype(dtype)
    keywords = {
        "dx_normalized": 0.31,
        "dy_normalized": 0.27,
        "h_y": 1.4,
        "applied_field_x": 0.2,
    }
    expected = state_from_potential(potential, xp=np, **keywords)

    monkeypatch.setattr(
        product_module,
        "_STATE_RECONSTRUCTION_CHUNK_BYTES",
        2 * potential.shape[-2] * potential.shape[-1] * potential.itemsize,
    )
    monkeypatch.setattr(product_module, "_STATE_RECONSTRUCTION_MIN_PLANE_CELLS", 1)
    actual = product_module._state_from_potential_for_products(
        potential, **keywords
    )

    for expected_array, actual_array in zip(
        (
            expected.psi,
            expected.carrier_density,
            expected.E_x,
            expected.E_y,
        ),
        (
            actual.psi,
            actual.carrier_density,
            actual.E_x,
            actual.E_y,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_array, expected_array)
        assert actual_array.dtype == expected_array.dtype


def test_mixed_uniform_chunk_preserves_whole_volume_uniform_shortcut(monkeypatch):
    potential = np.zeros((5, 12, 10), dtype=np.float64)
    x = np.arange(12)[:, None]
    y = np.arange(10)[None, :]
    potential[2:] = 0.01 * np.cos(2.0 * np.pi * (x / 12 + y / 10))
    keywords = {
        "dx_normalized": 0.4,
        "dy_normalized": 0.5,
        "h_y": 1.0,
        "applied_field_x": 0.0,
    }
    expected = state_from_potential(potential, xp=np, **keywords)
    monkeypatch.setattr(
        product_module,
        "_STATE_RECONSTRUCTION_CHUNK_BYTES",
        2 * potential.shape[-2] * potential.shape[-1] * potential.itemsize,
    )
    monkeypatch.setattr(product_module, "_STATE_RECONSTRUCTION_MIN_PLANE_CELLS", 1)

    actual = product_module._state_from_potential_for_products(
        potential, **keywords
    )

    for expected_array, actual_array in zip(
        (
            expected.psi,
            expected.carrier_density,
            expected.E_x,
            expected.E_y,
        ),
        (
            actual.psi,
            actual.carrier_density,
            actual.E_x,
            actual.E_y,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_small_product_volume_keeps_direct_reconstruction(monkeypatch):
    potential = np.zeros((5, 32, 24), dtype=np.float32)
    calls = []
    original = product_module.state_from_potential

    def counted(value, **keywords):
        calls.append(np.asarray(value).shape)
        return original(value, **keywords)

    monkeypatch.setattr(product_module, "state_from_potential", counted)
    product_module._state_from_potential_for_products(
        potential,
        dx_normalized=0.4,
        dy_normalized=0.5,
        h_y=1.0,
        applied_field_x=0.0,
    )

    assert calls == [potential.shape]


def test_large_float32_product_volume_uses_two_plane_chunks(monkeypatch):
    x = np.arange(512, dtype=np.float32)[:, None]
    y = np.arange(512, dtype=np.float32)[None, :]
    plane = np.cos(2.0 * np.pi * (x / 512 + y / 512))
    potential = np.stack((plane, 1.1 * plane, 1.2 * plane)).astype(np.float32)
    calls = []
    original = product_module.state_from_potential

    def counted(value, **keywords):
        calls.append(np.asarray(value).shape)
        return original(value, **keywords)

    monkeypatch.setattr(product_module, "state_from_potential", counted)
    product_module._state_from_potential_for_products(
        potential,
        dx_normalized=0.4,
        dy_normalized=0.5,
        h_y=1.0,
        applied_field_x=0.0,
    )

    assert calls == [(2, 512, 512), (1, 512, 512)]


def _synthetic_static_result() -> PRTransverseStaticRunResult:
    nz, nx, ny = 7, 18, 16
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    z = np.arange(nz)[:, None, None]
    potential = (
        0.01
        * np.cos(2.0 * np.pi * (2.0 * x[None, :, :] / nx + y[None, :, :] / ny))
        * (1.0 + 0.03 * z)
    ).astype(np.float32)
    phase = 2.0 * np.pi * (2.0 * x / nx - y / ny)
    initial = np.exp(1j * phase)[None, :, :].astype(np.complex64)
    final = (0.97 * np.exp(1j * (phase + 0.04)))[None, :, :].astype(np.complex64)
    source = np.broadcast_to(
        (1.3 + 0.1 * np.cos(2.0 * np.pi * x / nx))[None, :, :],
        (nz, nx, ny),
    ).astype(np.float32, copy=True)
    zeros = np.zeros_like(potential)
    profile = {
        "physics_profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
        "workflow": "pr_transverse_static",
        "dx_normalized": 0.4,
        "dy_normalized": 0.4,
        "material": {
            "dark_intensity": 0.2,
            "uniform_background_intensity": 0.1,
            "refractive_index": 2.4,
        },
        "beam_request": {
            "channels": (
                {
                    "wavelength_um": 0.633,
                    "tilt_x_rad_per_um": 0.2,
                    "tilt_y_rad_per_um": -0.1,
                    "waist_x_um": 8.0,
                    "waist_y_um": 7.0,
                },
            ),
        },
        "dielectric": {"h_y": 1.0},
        "boundary": {"applied_field_x": 0.0},
        "projection": {
            "profile_id": PR_FULL_TRANSVERSE_PROFILE_V1,
            "g_x": 1.0,
            "g_y": 0.0,
        },
    }
    return PRTransverseStaticRunResult(
        A_initial=initial,
        A_final=final,
        psi_initial=np.zeros_like(potential),
        psi_final=potential,
        source_intensity_stack=source,
        equilibrium_residual_stack=zeros,
        td_rhs_residual_stack=zeros.copy(),
        power_initial=float(np.sum(np.abs(initial) ** 2)),
        power_final=float(np.sum(np.abs(final) ** 2)),
        converged=True,
        completed_coupled_iterations=2,
        iteration_records=(),
        material_iteration_records=(),
        discrete_iteration_records=(),
        grid_summary={
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "dx_um": 2.0,
            "dy_um": 2.0,
            "dz_um": 4.0,
        },
        launch_summary={"coherence_groups": ("signal",)},
        backend_summary={"backend": "numpy", "precision": "float32"},
        resolved_profile=profile,
        replay_diagnostics={"complete_independent_replay": True},
        diagnostics={"equilibrium_residual_rms": 0.0},
        timing={"total_seconds": 1.0},
        status="converged",
    )


def test_static_adapter_products_are_unchanged_by_chunking(monkeypatch):
    result = _synthetic_static_result()
    original_helper = product_module._state_from_potential_for_products

    def direct(potential, **keywords):
        return state_from_potential(potential, xp=np, **keywords)

    monkeypatch.setattr(product_module, "_state_from_potential_for_products", direct)
    expected = pr_transverse_static_result_to_run_data(result)
    monkeypatch.setattr(
        product_module, "_state_from_potential_for_products", original_helper
    )
    monkeypatch.setattr(
        product_module,
        "_STATE_RECONSTRUCTION_CHUNK_BYTES",
        2 * result.psi_final.shape[-2] * result.psi_final.shape[-1]
        * result.psi_final.itemsize,
    )
    monkeypatch.setattr(product_module, "_STATE_RECONSTRUCTION_MIN_PLANE_CELLS", 1)
    actual = pr_transverse_static_result_to_run_data(result)

    assert tuple(actual.fields.keys()) == tuple(expected.fields.keys())
    for key in expected.fields.keys():
        expected_field = expected.fields[key]
        actual_field = actual.fields[key]
        assert actual_field.axes == expected_field.axes
        assert actual_field.kind == expected_field.kind
        assert actual_field.quantity == expected_field.quantity
        assert actual_field.value_unit == expected_field.value_unit
        assert actual_field.source_volume_key == expected_field.source_volume_key
        np.testing.assert_array_equal(actual_field.data, expected_field.data)
        assert actual_field.data.dtype == expected_field.data.dtype
        assert actual_field.coordinates.keys() == expected_field.coordinates.keys()
        for axis in expected_field.coordinates:
            np.testing.assert_array_equal(
                actual_field.coordinates[axis], expected_field.coordinates[axis]
            )

    assert tuple(actual.curves.keys()) == tuple(expected.curves.keys())
    for key in expected.curves.keys():
        np.testing.assert_array_equal(actual.curves[key].x, expected.curves[key].x)
        np.testing.assert_array_equal(actual.curves[key].y, expected.curves[key].y)
    assert tuple(actual.diagnostics.keys()) == tuple(expected.diagnostics.keys())
    for key in expected.diagnostics.keys():
        assert actual.diagnostics[key].values == expected.diagnostics[key].values

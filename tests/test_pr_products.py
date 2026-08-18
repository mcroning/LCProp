from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.products import PR_TIMEDEPENDENT_WORKFLOW, pr_result_to_run_data
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
)
from lcprop.pr.workflow import run_pr_timedependent


def _synthetic_pr_result() -> PRRunResult:
    nx, ny, nz = 4, 3, 2
    A_initial = np.ones((2, nx, ny), dtype=np.complex128)
    A_final = A_initial.copy()
    A_final[1] = -0.5
    E_initial = np.arange(nz * nx * ny, dtype=float).reshape(nz, nx, ny)
    E_final = E_initial + 0.25
    source = np.full((nz, nx, ny), 1.3)
    return PRRunResult(
        A_initial=A_initial,
        A_final=A_final,
        E_initial=E_initial,
        E_final=E_final,
        source_intensity_stack=source,
        power_initial=1.0,
        power_final=0.999999999,
        completed_steps=7,
        time_normalized=0.35,
        grid_summary={
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "dx_um": 2.0,
            "dy_um": 4.0,
            "dz_um": 5.0,
            "x_aperture_um": 8.0,
            "y_aperture_um": 12.0,
            "z_length_um": 10.0,
        },
        launch_summary={
            "Nch": 2,
            "coherence": "coherent",
            "coherence_groups": ["shared", "shared"],
            "physical_total_power_mW": 2.0,
        },
        diagnostics={
            "backend": {"backend": "numpy", "precision": "float64"},
            "conservative_dt_limit": 0.1,
        },
    )


def test_pr_result_to_run_data_preserves_fields_coordinates_and_semantics():
    result = _synthetic_pr_result()

    data = pr_result_to_run_data(result)

    assert data.workflow == PR_TIMEDEPENDENT_WORKFLOW
    np.testing.assert_array_equal(data.geometry.x, [-3.0, -1.0, 1.0, 3.0])
    np.testing.assert_array_equal(data.geometry.y, [-4.0, 0.0, 4.0])
    np.testing.assert_array_equal(data.geometry.z, [0.0, 5.0])
    assert list(data.fields.keys()) == [
        "input_intensity",
        "output_intensity",
        "initial_E",
        "final_E",
        "initial_E_stack",
        "final_E_stack",
        "pr_driving_intensity_stack",
    ]

    # Both channels share one coherence group: fields interfere before their
    # intensity is evaluated.
    np.testing.assert_allclose(data.fields["input_intensity"].data, 4.0)
    np.testing.assert_allclose(data.fields["output_intensity"].data, 0.25)
    np.testing.assert_array_equal(
        data.fields["initial_E_stack"].data,
        result.E_initial,
    )
    np.testing.assert_array_equal(
        data.fields["final_E_stack"].data,
        result.E_final,
    )
    np.testing.assert_array_equal(
        data.fields["pr_driving_intensity_stack"].data,
        result.source_intensity_stack,
    )
    np.testing.assert_array_equal(
        data.fields["initial_E"].data,
        result.E_initial[1],
    )
    np.testing.assert_array_equal(
        data.fields["final_E"].data,
        result.E_final[1],
    )

    assert data.fields["input_intensity"].value_unit == "1/µm²"
    assert data.fields["final_E_stack"].axes == ("z", "x", "y")
    assert data.fields["final_E_stack"].quantity == (
        "normalized_space_charge_field"
    )
    driving = data.fields["pr_driving_intensity_stack"]
    assert driving.quantity == "normalized_pr_driving_intensity"
    assert driving.value_unit == "1"
    assert list(data.curves.keys()) == []

    summary = data.diagnostics["summary"].values
    assert summary["material"] == "photorefractive"
    assert summary["normalized_field_integral_initial"] == 1.0
    assert summary["normalized_field_integral_final"] == 0.999999999
    assert summary["completed_material_steps"] == 7
    assert summary["requested_material_steps"] == 0
    assert summary["material_time_normalized"] == 0.35
    assert summary["status"] == "completed"
    assert data.diagnostics["pr_workflow"].values == result.diagnostics


def test_pr_product_conversion_is_deterministic_and_detached_from_result():
    result = _synthetic_pr_result()

    first = pr_result_to_run_data(result)
    second = pr_result_to_run_data(result)

    assert list(first.fields.keys()) == list(second.fields.keys())
    for key in first.fields:
        np.testing.assert_array_equal(
            first.fields[key].data,
            second.fields[key].data,
        )

    first.fields["initial_E_stack"].data[0, 0, 0] = -999.0
    first.diagnostics["summary"].values["grid"]["Nx"] = -1
    first.diagnostics["pr_workflow"].values["backend"]["backend"] = "changed"

    assert result.E_initial[0, 0, 0] == 0.0
    assert result.grid_summary["Nx"] == 4
    assert result.diagnostics["backend"]["backend"] == "numpy"
    assert second.fields["initial_E_stack"].data[0, 0, 0] == 0.0


def test_pr_result_to_run_data_rejects_inconsistent_volume_shape():
    result = _synthetic_pr_result()
    malformed = replace(
        result,
        E_final=np.zeros((1, 4, 3)),
    )

    with pytest.raises(ValueError, match="E_final must have shape"):
        pr_result_to_run_data(malformed)


def test_small_real_pr_run_converts_to_run_data():
    request = PRRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=6,
            x_aperture_um=80.0,
            y_aperture_um=60.0,
            dz_um=5.0,
            z_length_um=10.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    waist_x_um=20.0,
                    waist_y_um=20.0,
                    coherence_group="pr-smoke",
                ),
            ),
        ),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            applied_field=0.5,
            gain_length_product=0.1,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRSolverOptions(
            Nt=1,
            dt_normalized=0.01,
            optical_substeps=1,
        ),
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
    )

    result = run_pr_timedependent(request)
    data = pr_result_to_run_data(result)

    assert data.workflow == PR_TIMEDEPENDENT_WORKFLOW
    assert data.fields["input_intensity"].data.shape == (8, 6)
    assert data.fields["output_intensity"].data.shape == (8, 6)
    assert data.fields["initial_E_stack"].data.shape == (2, 8, 6)
    assert data.fields["final_E_stack"].data.shape == (2, 8, 6)
    assert data.fields["pr_driving_intensity_stack"].data.shape == (2, 8, 6)
    assert data.diagnostics["summary"].values[
        "completed_material_steps"
    ] == 1
    assert data.diagnostics["summary"].values[
        "requested_material_steps"
    ] == 1
    assert data.diagnostics["summary"].values["status"] == "completed"
    assert data.diagnostics["pr_workflow"].values["backend"][
        "backend"
    ] == "numpy"
    assert data.fields["output_intensity"].display_name == (
        "Output Plane Intensity"
    )
    assert data.fields["pr_driving_intensity_stack"].display_name == (
        "Final PR-Driving Intensity"
    )


def test_pr_result_to_run_data_rejects_other_result_types():
    with pytest.raises(TypeError, match="PRRunResult"):
        pr_result_to_run_data(object())


def test_td_volume_products_share_readonly_authoritative_result_memory():
    grid = GridSpec(
        Nx=8,
        Ny=6,
        x_aperture_um=80.0,
        y_aperture_um=60.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    request = PRRunRequest(
        grid=grid,
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    waist_x_um=20.0,
                    waist_y_um=20.0,
                ),
            ),
        ),
        solver=PRSolverOptions(Nt=1, dt_normalized=0.01),
        backend=BackendSpec(
            backend="numpy",
            precision="float64",
            verbose=False,
        ),
    )
    token = CancellationToken()
    token.cancel()
    result = run_pr_timedependent(request, cancellation_token=token)
    data = pr_result_to_run_data(result)

    for key, authoritative in (
        ("initial_E_stack", result.E_initial),
        ("final_E_stack", result.E_final),
        ("pr_driving_intensity_stack", result.source_intensity_stack),
    ):
        presented = data.fields[key].data
        assert presented is not authoritative
        assert np.shares_memory(presented, authoritative)
        assert not presented.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            presented.flat[0] = -999.0

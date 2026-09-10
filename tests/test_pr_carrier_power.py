import math
import os
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.gui.workspace import Workspace
from lcprop.pr.carrier_power import (
    carrier_power_diagnostic,
    nearest_carrier_partition,
)
from lcprop.pr.image_amplification import (
    isolate_signal_carrier,
    signal_carrier_mask,
)
from lcprop.pr.products import pr_result_to_run_data, pr_static_result_to_run_data
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.static_transport_codec import (
    decode_pr_static_transport_result,
    encode_pr_static_transport_result,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    run_pr_static,
)
from lcprop.pr.timedependent_transport_codec import (
    decode_pr_timedependent_transport_result,
    encode_pr_timedependent_transport_result,
)
from lcprop.pr.transverse.products import (
    pr_transverse_static_result_to_run_data,
)
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseBoundaryProfile,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    run_pr_transverse_static,
)
from lcprop.pr.transverse.transport_codec import (
    decode_pr_transverse_static_transport_result,
    encode_pr_transverse_static_transport_result,
)
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.transport.result_policy import FAST_RESULT_POLICY


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _mode_field(shape, index, power, *, dx_um, dy_um, dtype=np.complex128):
    spectrum = np.zeros(shape, dtype=np.complex128)
    spectrum[index] = math.sqrt(power * np.prod(shape) / (dx_um * dy_um))
    return np.fft.ifft2(spectrum).astype(dtype)


def _carrier_center(shape, index, *, dx_um, dy_um):
    return (
        float(2.0 * math.pi * np.fft.fftfreq(shape[0], d=dx_um)[index[0]]),
        float(2.0 * math.pi * np.fft.fftfreq(shape[1], d=dy_um)[index[1]]),
    )


def _threshold_fixture_diagnostic(
    signal_input_power,
    *,
    signal_output_power,
    dtype=np.complex128,
):
    shape = (16, 12)
    dx_um, dy_um = 0.5, 0.75
    indices = ((2, 1), (14, 11))
    centers = tuple(
        _carrier_center(shape, index, dx_um=dx_um, dy_um=dy_um)
        for index in indices
    )
    initial = np.stack((
        _mode_field(
            shape, indices[0], 1.0, dx_um=dx_um, dy_um=dy_um, dtype=dtype
        ),
        _mode_field(
            shape,
            indices[1],
            signal_input_power,
            dx_um=dx_um,
            dy_um=dy_um,
            dtype=dtype,
        ),
    ))
    final = np.stack((
        initial[0],
        _mode_field(
            shape,
            indices[1],
            signal_output_power,
            dx_um=dx_um,
            dy_um=dy_um,
            dtype=dtype,
        ),
    ))
    return carrier_power_diagnostic(
        initial,
        final,
        dx_um=dx_um,
        dy_um=dy_um,
        coherence_groups=("shared", "shared"),
        carrier_channels=(
            {
                "name": "Pump",
                "kx_rad_per_um": centers[0][0],
                "ky_rad_per_um": centers[0][1],
            },
            {
                "name": "Signal",
                "kx_rad_per_um": centers[1][0],
                "ky_rad_per_um": centers[1][1],
            },
        ),
    )


@pytest.mark.parametrize(
    "centers",
    [
        ((-1.0, 0.0), (1.0, 0.0)),
        ((0.0, -1.0), (0.0, 1.0)),
        ((-1.0, -1.0), (1.0, 1.0)),
        ((-0.7, 1.3), (0.7, -1.3)),
    ],
)
def test_partition_follows_perpendicular_bisector_for_arbitrary_orientation(
    centers,
):
    partition = nearest_carrier_partition(
        (18, 16),
        dx_um=0.4,
        dy_um=0.7,
        centers_k_rad_per_um=centers,
    )
    kx = partition.kx_rad_per_um[:, None]
    ky = partition.ky_rad_per_um[None, :]
    first_distance = (kx - centers[0][0]) ** 2 + (ky - centers[0][1]) ** 2
    second_distance = (kx - centers[1][0]) ** 2 + (ky - centers[1][1]) ** 2

    np.testing.assert_array_equal(
        partition.weights[0][first_distance < second_distance], 1.0
    )
    np.testing.assert_array_equal(
        partition.weights[0][first_distance > second_distance], 0.0
    )
    np.testing.assert_array_equal(
        partition.weights[0][first_distance == second_distance], 0.5
    )
    np.testing.assert_array_equal(np.sum(partition.weights, axis=0), 1.0)


def test_synthetic_transfer_recovers_power_gain_delta_and_balance():
    shape = (16, 12)
    dx_um, dy_um = 0.5, 0.75
    indices = ((2, 1), (14, 11))
    centers = tuple(
        _carrier_center(shape, index, dx_um=dx_um, dy_um=dy_um)
        for index in indices
    )
    initial = np.stack((
        _mode_field(shape, indices[0], 0.75, dx_um=dx_um, dy_um=dy_um),
        _mode_field(shape, indices[1], 0.25, dx_um=dx_um, dy_um=dy_um),
    ))
    final = np.stack((
        _mode_field(shape, indices[0], 0.5, dx_um=dx_um, dy_um=dy_um)
        + _mode_field(shape, indices[1], 0.5, dx_um=dx_um, dy_um=dy_um),
        np.zeros(shape, dtype=np.complex128),
    ))
    diagnostic = carrier_power_diagnostic(
        initial,
        final,
        dx_um=dx_um,
        dy_um=dy_um,
        coherence_groups=("shared", "shared"),
        carrier_channels=(
            {
                "name": "Pump",
                "kx_rad_per_um": centers[0][0],
                "ky_rad_per_um": centers[0][1],
            },
            {
                "name": "Signal",
                "kx_rad_per_um": centers[1][0],
                "ky_rad_per_um": centers[1][1],
            },
        ),
        physical_total_power_mW=4.0,
    )

    assert diagnostic["status"] == "ok"
    np.testing.assert_allclose(diagnostic["carrier_power_input"], (0.75, 0.25))
    np.testing.assert_allclose(diagnostic["carrier_power_output"], (0.5, 0.5))
    np.testing.assert_allclose(diagnostic["carrier_gain"], (2.0 / 3.0, 2.0))
    np.testing.assert_allclose(diagnostic["carrier_delta_power"], (-0.25, 0.25))
    np.testing.assert_allclose(diagnostic["carrier_power_input_mW"], (3.0, 1.0))
    assert abs(diagnostic["carrier_power_balance_error"]) < 2e-15
    assert diagnostic["carrier_separation_quality"] == pytest.approx(1.0)
    assert diagnostic["carrier_gain_status"] == ["available", "available"]
    assert diagnostic["carrier_gain_unavailable_reason"] == [None, None]


def test_exactly_zero_input_carrier_power_omits_unstable_gain():
    diagnostic = _threshold_fixture_diagnostic(
        0.0, signal_output_power=0.25
    )

    assert diagnostic["status"] == "insufficient_input_carrier_power"
    assert diagnostic["carrier_gain"][1] is None
    assert diagnostic["carrier_gain_status"][1] == (
        "insufficient_input_carrier_power"
    )
    assert diagnostic["carrier_gain_unavailable_reason"][1] is not None
    assert diagnostic["carrier_power_input"][1] <= diagnostic[
        "carrier_input_power_threshold"
    ]
    assert diagnostic["carrier_power_output"][1] == pytest.approx(0.25)
    assert diagnostic["carrier_delta_power"][1] == pytest.approx(0.25)


def test_tiny_positive_input_carrier_power_below_relative_threshold():
    diagnostic = _threshold_fixture_diagnostic(
        0.5e-12, signal_output_power=0.25
    )

    assert diagnostic["carrier_input_power_relative_threshold"] == pytest.approx(
        1e-12
    )
    assert diagnostic["status"] == "insufficient_input_carrier_power"
    assert diagnostic["carrier_gain"][1] is None
    assert diagnostic["carrier_gain_status"][1] == (
        "insufficient_input_carrier_power"
    )
    assert diagnostic["carrier_power_input"][1] > 0.0
    assert diagnostic["carrier_power_input"][1] <= diagnostic[
        "carrier_input_power_threshold"
    ]


def test_input_carrier_power_just_above_relative_threshold_reports_gain():
    diagnostic = _threshold_fixture_diagnostic(
        2e-12, signal_output_power=4e-12
    )

    assert diagnostic["status"] == "ok"
    assert diagnostic["carrier_power_input"][1] > diagnostic[
        "carrier_input_power_threshold"
    ]
    assert diagnostic["carrier_gain"][1] == pytest.approx(2.0)
    assert diagnostic["carrier_gain_status"][1] == "available"
    assert diagnostic["carrier_gain_unavailable_reason"][1] is None


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype", "tolerance"),
    [
        (np.float64, np.complex128, 2e-13),
        (np.float32, np.complex64, 2e-6),
    ],
)
def test_parseval_partition_matches_coherent_spatial_power(
    real_dtype, complex_dtype, tolerance
):
    shape = (17, 14)
    dx_um, dy_um = 0.6, 0.8
    rng = np.random.default_rng(20260910)
    initial = (
        rng.normal(size=(2, *shape)) + 1j * rng.normal(size=(2, *shape))
    ).astype(complex_dtype)
    coherent_power = (
        np.sum(np.abs(np.sum(initial, axis=0)) ** 2, dtype=np.float64)
        * dx_um
        * dy_um
    )
    initial = (initial / math.sqrt(coherent_power)).astype(complex_dtype)
    final = (0.9 * initial).astype(complex_dtype)
    diagnostic = carrier_power_diagnostic(
        initial,
        final,
        dx_um=dx_um,
        dy_um=dy_um,
        coherence_groups=("shared", "shared"),
        carrier_channels=(
            {"name": "A", "kx_rad_per_um": -2.0, "ky_rad_per_um": 0.4},
            {"name": "B", "kx_rad_per_um": 2.0, "ky_rad_per_um": -0.4},
        ),
    )

    assert abs(diagnostic["parseval_partition_error_input"]) < tolerance
    assert abs(diagnostic["parseval_partition_error_output"]) < tolerance
    assert abs(diagnostic["carrier_power_balance_error"]) < tolerance
    assert diagnostic["carrier_input_power_relative_threshold"] == pytest.approx(
        max(1e-12, 64.0 * np.finfo(real_dtype).eps)
    )


def test_different_groups_and_nearly_coincident_carriers_are_explicit():
    fields = np.ones((2, 8, 8), dtype=np.complex128)
    channels = (
        {"name": "A", "kx_rad_per_um": 0.0, "ky_rad_per_um": 0.0},
        {"name": "B", "kx_rad_per_um": 0.0, "ky_rad_per_um": 0.0},
    )
    separate = carrier_power_diagnostic(
        fields,
        fields,
        dx_um=1.0,
        dy_um=1.0,
        coherence_groups=("a", "b"),
        carrier_channels=channels,
    )
    coincident = carrier_power_diagnostic(
        fields,
        fields,
        dx_um=1.0,
        dy_um=1.0,
        coherence_groups=("same", "same"),
        carrier_channels=channels,
    )

    assert separate["status"] == "not_applicable"
    assert "different coherence groups" in separate["reason"]
    assert coincident["status"] == "insufficient_separation"
    assert coincident["carrier_gain"] == [None, None]
    assert coincident["warning"] is not None


def test_ia_signal_mask_reuses_legacy_second_carrier_tie_policy():
    grid = make_grid(GridSpec(
        Nx=16,
        Ny=12,
        x_aperture_um=16.0,
        y_aperture_um=12.0,
        z_length_um=1.0,
    ))
    pump_kx = 2.0 * math.pi * 2.0 / 16.0
    signal_kx = -pump_kx
    expected = nearest_carrier_partition(
        (grid.Nx, grid.Ny),
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
        centers_k_rad_per_um=((pump_kx, 0.0), (signal_kx, 0.0)),
        tie_policy="second",
    ).weights[1].astype(bool)
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    pump = _mode_field(
        (grid.Nx, grid.Ny),
        (2, 0),
        0.75,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
    )
    signal = _mode_field(
        (grid.Nx, grid.Ny),
        (grid.Nx - 2, 0),
        0.25,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
    )

    np.testing.assert_array_equal(mask, expected)
    coherent = pump + signal
    recovered = isolate_signal_carrier(coherent, mask)
    np.testing.assert_allclose(recovered, signal, rtol=0.0, atol=2e-15)
    diagnostic = carrier_power_diagnostic(
        np.stack((pump, signal)),
        np.stack((pump, signal)),
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
        coherence_groups=("shared", "shared"),
        carrier_channels=(
            {"name": "Pump", "kx_rad_per_um": pump_kx, "ky_rad_per_um": 0.0},
            {"name": "Signal", "kx_rad_per_um": signal_kx, "ky_rad_per_um": 0.0},
        ),
    )
    ia_signal_power = float(
        np.sum(np.abs(recovered) ** 2) * grid.dx_um * grid.dy_um
    )
    assert diagnostic["carrier_power_input"][1] == pytest.approx(
        ia_signal_power, abs=2e-15
    )


@pytest.fixture(scope="module")
def three_model_results():
    grid = GridSpec(
        Nx=32,
        Ny=24,
        x_aperture_um=64.0,
        y_aperture_um=48.0,
        dz_um=4.0,
        z_length_um=4.0,
    )
    kx = 2.0 * math.pi * 3.0 / grid.x_aperture_um
    ky = 2.0 * math.pi * 2.0 / grid.y_aperture_um
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="Pump",
                power_mW=3.0,
                waist_x_um=20.0,
                waist_y_um=18.0,
                tilt_x_rad_per_um=kx,
                tilt_y_rad_per_um=ky,
            ),
            BeamChannel(
                name="Signal",
                power_mW=1.0,
                waist_x_um=18.0,
                waist_y_um=16.0,
                tilt_x_rad_per_um=-kx,
                tilt_y_rad_per_um=-ky,
            ),
        ),
        coherence="coherent",
    )
    material = PRMaterialSpec(
        dark_intensity=20.0,
        applied_field=0.0,
        gain_length_product=0.02,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.2,
    )
    common = dict(
        grid=grid,
        beams=beams,
        material=material,
        backend=BackendSpec("numpy", "float64", False),
    )
    nonlinear = run_pr_static(PRStaticRunRequest(
        **common,
        solver=PRStaticWorkflowOptions(max_coupled_passes=12),
    ))
    linearized_request = PRStaticRunRequest(
        **common,
        solver=PRStaticWorkflowOptions(max_coupled_passes=12),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=21.0,
        ),
    )
    linearized = run_pr_static(linearized_request)
    transverse = run_pr_transverse_static(PRTransverseStaticRunRequest(
        **common,
        boundary=PRTransverseBoundaryProfile(
            profile_id=PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
            applied_field_x=0.0,
        ),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=12),
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=21.0,
        ),
    ))
    return nonlinear, linearized_request, linearized, transverse


def test_three_model_products_report_finite_carrier_powers(three_model_results):
    nonlinear, _request, linearized, transverse = three_model_results
    diagnostics = (
        pr_static_result_to_run_data(nonlinear).diagnostics["carrier_power"].values,
        pr_static_result_to_run_data(linearized).diagnostics["carrier_power"].values,
        pr_transverse_static_result_to_run_data(transverse).diagnostics[
            "carrier_power"
        ].values,
    )

    for diagnostic in diagnostics:
        assert diagnostic["status"] == "ok"
        assert diagnostic["carrier_separation_quality"] > 0.99
        assert np.all(np.isfinite(diagnostic["carrier_power_input"]))
        assert np.all(np.isfinite(diagnostic["carrier_power_output"]))
        assert np.all(np.isfinite(diagnostic["carrier_gain"]))
        assert abs(diagnostic["carrier_power_balance_error"]) < 2e-13

    dxdy = (
        float(nonlinear.grid_summary["dx_um"])
        * float(nonlinear.grid_summary["dy_um"])
    )
    lineage_input = np.sum(
        np.abs(nonlinear.A_initial) ** 2, axis=(-2, -1)
    ) * dxdy
    lineage_output = np.sum(
        np.abs(nonlinear.A_final) ** 2, axis=(-2, -1)
    ) * dxdy
    np.testing.assert_allclose(lineage_output, lineage_input, rtol=0.0, atol=2e-14)
    assert np.max(np.abs(diagnostics[0]["carrier_delta_power"])) > 1e-5


def test_fast_and_full_transport_have_identical_carrier_diagnostics(
    three_model_results,
):
    _nonlinear, _request, linearized, transverse = three_model_results
    full_payload = encode_pr_static_transport_result(linearized)
    fast_payload = encode_pr_static_transport_result(
        linearized, result_policy=FAST_RESULT_POLICY
    )
    full = decode_pr_static_transport_result(
        full_payload.payload.metadata, full_payload.payload.arrays
    )
    fast = decode_pr_static_transport_result(
        fast_payload.payload.metadata, fast_payload.payload.arrays
    )

    full_diagnostic = pr_static_result_to_run_data(full).diagnostics[
        "carrier_power"
    ].values
    fast_diagnostic = pr_static_result_to_run_data(fast).diagnostics[
        "carrier_power"
    ].values
    assert fast_diagnostic == full_diagnostic

    transverse_full_payload = encode_pr_transverse_static_transport_result(
        transverse
    )
    transverse_fast_payload = encode_pr_transverse_static_transport_result(
        transverse, result_policy=FAST_RESULT_POLICY
    )
    transverse_full = decode_pr_transverse_static_transport_result(
        transverse_full_payload.payload.metadata,
        transverse_full_payload.payload.arrays,
    )
    transverse_fast = decode_pr_transverse_static_transport_result(
        transverse_fast_payload.payload.metadata,
        transverse_fast_payload.payload.arrays,
    )
    transverse_full_diagnostic = pr_transverse_static_result_to_run_data(
        transverse_full
    ).diagnostics["carrier_power"].values
    transverse_fast_diagnostic = pr_transverse_static_result_to_run_data(
        transverse_fast
    ).diagnostics["carrier_power"].values
    assert transverse_fast_diagnostic == transverse_full_diagnostic


def test_reduced_td_fast_and_full_carrier_diagnostics_match(
    three_model_results,
):
    _nonlinear, static_request, _linearized, _transverse = three_model_results
    result = run_pr_timedependent(PRRunRequest(
        grid=static_request.grid,
        beams=static_request.beams,
        material=static_request.material,
        solver=PRSolverOptions(Nt=0, dt_normalized=1e-4),
        backend=static_request.backend,
    ))
    full_payload = encode_pr_timedependent_transport_result(result)
    fast_payload = encode_pr_timedependent_transport_result(
        result, result_policy=FAST_RESULT_POLICY
    )
    full = decode_pr_timedependent_transport_result(
        full_payload.payload.metadata, full_payload.payload.arrays
    )
    fast = decode_pr_timedependent_transport_result(
        fast_payload.payload.metadata, fast_payload.payload.arrays
    )

    full_values = pr_result_to_run_data(full).diagnostics["carrier_power"].values
    fast_values = pr_result_to_run_data(fast).diagnostics["carrier_power"].values
    assert full_values["status"] == "ok"
    assert fast_values == full_values


def test_gui_formats_compact_carrier_power_table(app, three_model_results):
    del app
    _nonlinear, _request, linearized, _transverse = three_model_results
    run_data = pr_static_result_to_run_data(linearized)
    workspace = Workspace()
    workspace.set_run_data(run_data)
    text = workspace.diagnostics_view.toPlainText()

    assert (
        "Carrier | Input Power (normalized) | Output Power (normalized) | "
        "Gain | Delta Power (normalized)"
    ) in text
    assert "Pump |" in text
    assert "Signal |" in text
    assert "carrier_separation_quality:" in text
    assert "carrier_power_balance_error:" in text
    workspace.close()


def test_pr_launch_summary_retains_compact_carrier_geometry(three_model_results):
    _nonlinear, request, linearized, _transverse = three_model_results
    assert linearized.launch_summary["carrier_channels"] == [
        {
            "name": channel.name,
            "kx_rad_per_um": channel.tilt_x_rad_per_um,
            "ky_rad_per_um": channel.tilt_y_rad_per_um,
        }
        for channel in request.beams.channels
    ]

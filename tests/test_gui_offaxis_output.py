from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.gui.main_window import LCPropMainWindow
from lcprop.lc.propagation import advance_slice
from lcprop.optics.splitstep import total_intensity
from lcprop.workflows.runtime import build_runtime_components
import lcprop.workflows.static as static_workflow


def _centroid_x(A, x_um) -> float:
    intensity = np.sum(np.abs(np.asarray(A)) ** 2, axis=0)
    return float(np.sum(intensity * np.asarray(x_um)[:, None]) / np.sum(intensity))


def test_exact_gui_offaxis_run_plots_current_final_intensity(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow()
    window.grid_panel.Nx.setValue(128)
    window.grid_panel.Ny.setValue(128)
    window.grid_panel.dz_um.setValue(5.0)
    window.grid_panel.z_length_um.setValue(500.0)
    window.grid_panel.x_aperture_um.setValue(75.0)
    window.grid_panel.y_aperture_um.setValue(100.0)
    window.beam_panel.set_beam_stack_definition(
        BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="off-axis",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    x_um=-20.0,
                    y_um=0.0,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                    coherence_group="laser_A",
                ),
            )
        )
    )

    request = window.build_request()
    channel = request.beams.channels[0]
    assert (channel.x0_um, channel.y0_um) == (-20.0, 0.0)
    assert (channel.waist_x_um, channel.waist_y_um) == (3.0, 3.0)
    assert request.grid.Nx == request.grid.Ny == 128
    assert request.grid.x_aperture_um == 75.0
    assert request.grid.y_aperture_um == 100.0
    assert request.grid.z_length_um == 500.0
    assert request.grid.dz_um == 5.0
    assert request.solver.workflow.strategy == "local_self_consistent"

    components = build_runtime_components(request)
    A0 = np.asarray(components.launch.A0).copy()
    trial_centroids: list[float] = []
    original_advance = static_workflow.advance_slice_with_midpoint_source

    def observed_advance(A, theta, **kwargs):
        result = original_advance(A, theta, **kwargs)
        trial_centroids.append(_centroid_x(result[0], components.grid.x_um))
        return result

    monkeypatch.setattr(
        static_workflow,
        "advance_slice_with_midpoint_source",
        observed_advance,
    )

    captured = {}
    original_runner = window.runner.run_static

    def capture_runner(run_request, **kwargs):
        captured["request"] = run_request
        captured["runner_result"] = original_runner(run_request, **kwargs)
        return captured["runner_result"]

    monkeypatch.setattr(window.runner, "run_static", capture_runner)
    window.run_static_clicked()
    deadline = time.monotonic() + 240.0
    while window._background_running:
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background static run")
        time.sleep(0.002)
    app.processEvents()

    assert captured["request"] == request
    result = captured["runner_result"].result
    assert len(trial_centroids) == sum(
        1 + summary.optical_passes for summary in result.slice_summaries
    )
    accepted_centroids = []
    call_index = 0
    for summary in result.slice_summaries:
        call_index += 1 + summary.optical_passes
        accepted_centroids.append(trial_centroids[call_index - 1])
    final_trajectory = np.asarray([-20.0, *accepted_centroids])
    assert np.isclose(final_trajectory[0], -20.0, atol=1.0e-10)
    assert np.isclose(
        final_trajectory[-1],
        _centroid_x(result.A_final, components.grid.x_um),
    )

    runner_intensity = np.asarray(
        total_intensity(
            result.A_final,
            coherence_groups=tuple(result.launch_summary["coherence_groups"]),
        )
    )
    initial_intensity = np.sum(np.abs(A0) ** 2, axis=0)
    image_pane = window.results_panel.workspace.image_pane
    run_data = image_pane._run_data

    output_index = image_pane.field_selector.findData("final_intensity")
    image_pane.field_selector.setCurrentIndex(output_index)
    assert image_pane.field_selector.currentData() == "final_intensity"
    assert image_pane.field_selector.currentText() == "Output Plane Intensity"
    plotted_field = run_data.fields["final_intensity"]
    assert plotted_field.axes == ("x", "y")
    assert np.array_equal(np.asarray(plotted_field.data), runner_intensity)
    assert not np.allclose(np.asarray(plotted_field.data), initial_intensity)
    assert np.array_equal(
        np.asarray(run_data.fields["delta_theta_stack"].data),
        np.asarray(result.theta_final) - np.asarray(result.theta_bias)[None],
    )

    # Matplotlib consumes image[row, column] = image[y, x]. ImageView owns this
    # intentional display-only transpose; the FieldData remains data[x, y].
    artist_array = np.asarray(image_pane.image_view.image.get_array())
    assert np.array_equal(artist_array, runner_intensity.T)

    peak_ix, peak_iy = np.unravel_index(
        int(np.argmax(runner_intensity)),
        runner_intensity.shape,
    )
    assert np.isfinite(float(components.grid.x_um[peak_ix]))
    assert 0 <= peak_iy < components.grid.Ny

    A_frozen = A0.copy()
    frozen_centroids = [_centroid_x(A_frozen, components.grid.x_um)]
    for _ in range(components.grid.Nz):
        advance_slice(
            A_frozen,
            components.bias.theta_2d,
            kernel=components.kernel,
            dz=components.grid.dz_um,
            wavelength=components.wavelength_um,
            n_ref=components.n_ref,
            ne=request.material.ne,
            no=request.material.no,
            xp=components.grid.xp,
        )
        frozen_centroids.append(
            _centroid_x(A_frozen, components.grid.x_um)
        )

    assert not np.isclose(
        frozen_centroids[-1],
        final_trajectory[-1],
        atol=0.001,
    )
    window.close()

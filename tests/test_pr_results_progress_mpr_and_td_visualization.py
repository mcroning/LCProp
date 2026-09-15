from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QDesktopServices

from lcprop.core.execution import RunProgress
from lcprop.gui.workspace import Workspace
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.products import _td_scalar_curves
from lcprop.pr.visualization import (
    FAST_MPR_TARGET_BYTES,
    block_average_2d,
    encode_td_preview_movie,
    make_fast_intensity_preview,
    td_movie_frame_indices,
)
from lcprop.pr.static_transport_codec import PR_STATIC_TRANSPORT_CODEC
from lcprop.pr.timedependent_transport_codec import PR_TIMEDEPENDENT_TRANSPORT_CODEC
from lcprop.pr.transverse.timedependent_transport_codec import (
    PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC,
)
from lcprop.pr.transverse.transport_codec import (
    PR_TRANSVERSE_STATIC_TRANSPORT_CODEC,
)
from lcprop.products.data_model import (
    ArtifactData,
    FieldCollection,
    Geometry,
    RunData,
    make_field,
)
from lcprop.transport.executor import _write_progress


def _grid(*, nz=7, nx=8, ny=6):
    return {
        "Nz": nz,
        "Nx": nx,
        "Ny": ny,
        "dz_um": 2.5,
        "dx_um": 1.0,
        "dy_um": 2.0,
    }


def test_preview_uses_block_means_actual_z_planes_and_bounded_float32():
    source = np.arange(7 * 8 * 6, dtype=float).reshape(7, 8, 6)
    preview = make_fast_intensity_preview(
        source,
        grid_summary=_grid(),
        peak_intensity_reference=2.0,
        background_intensity=0.0,
    )
    assert preview.intensity.dtype == np.float32
    assert preview.intensity.nbytes <= FAST_MPR_TARGET_BYTES
    assert preview.metadata["visualization_only"] is True
    assert preview.metadata["retained_z_indices"] == list(range(7))
    np.testing.assert_array_equal(
        preview.metadata["preview_coordinates_um"]["z"],
        np.arange(7) * 2.5,
    )
    np.testing.assert_array_equal(preview.intensity, source * 2.0)


def test_visualization_result_codec_versions_advance_and_accept_predecessors():
    expected = (
        (PR_STATIC_TRANSPORT_CODEC, 3, 2),
        (PR_TIMEDEPENDENT_TRANSPORT_CODEC, 4, 3),
        (PR_TRANSVERSE_STATIC_TRANSPORT_CODEC, 4, 3),
        (PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC, 3, 2),
    )
    for codec, current, previous in expected:
        assert codec.result_codec_version == current
        assert previous in codec.compatible_result_codec_versions
        assert current not in codec.compatible_result_codec_versions


def test_transverse_block_averaging_is_not_decimation():
    source = np.arange(16, dtype=float).reshape(4, 4)
    reduced, x_edges, y_edges = block_average_2d(
        source, max_x=2, max_y=2
    )
    np.testing.assert_array_equal(x_edges, [0, 2, 4])
    np.testing.assert_array_equal(y_edges, [0, 2, 4])
    np.testing.assert_allclose(reduced, [[2.5, 4.5], [10.5, 12.5]])


def test_large_fast_preview_saturates_but_never_exceeds_size_budget():
    grid = _grid(nz=150, nx=200, ny=160)
    source = np.ones((150, 200, 160), dtype=np.float32)
    preview = make_fast_intensity_preview(
        source,
        grid_summary=grid,
        peak_intensity_reference=1.0,
        background_intensity=0.0,
    )
    assert preview.intensity.shape == (113, 96, 96)
    assert preview.intensity.nbytes == 4_165_632
    assert preview.intensity.nbytes <= FAST_MPR_TARGET_BYTES


def test_td_movie_selection_and_encoding_failure_are_bounded_and_nonfatal():
    indices = td_movie_frame_indices(1000)
    assert indices[0] == 0
    assert indices[-1] == 1000
    assert indices.size <= 36
    movie = encode_td_preview_movie(
        [np.ones((4, 6), dtype=np.float32)],
        frame_indices=[0],
        material_times=[0.0],
        original_grid=_grid(),
        original_cadence=0.1,
        ffmpeg_path="/bin/false",
    )
    assert movie.data is None
    assert movie.warning is not None
    assert movie.metadata["status"] == "encoding_failed"
    assert movie.metadata["fixed_color_limits"] == [0.0, 1.0]


def test_td_curves_suppress_unretained_and_model_inapplicable_quantities():
    assert len(_td_scalar_curves(SimpleNamespace(td_scalar_history=()))) == 0
    curves = _td_scalar_curves(SimpleNamespace(td_scalar_history=(
        {
            "material_time_normalized": 0.1,
            "material_state_change_rms": 2e-3,
        },
    )))
    assert tuple(curves) == ("material_state_change_rms",)


def test_atomic_progress_artifact_is_compact_and_json_safe(tmp_path):
    progress = RunProgress(
        workflow="pr_static",
        status="running",
        completed_units=2,
        total_units=10,
        current_coordinate=4.0,
        coordinate_name="z",
        coordinate_unit="um",
        elapsed_wall_time=0.5,
        latest_field_state={"large": np.ones((4, 4))},
        diagnostics={"phase": "replay"},
        message="optical replay",
    )
    assert _write_progress(tmp_path, progress) is True
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload == {
        "schema_version": 1,
        "workflow": "pr_static",
        "status": "running",
        "phase": "replay",
        "message": "optical replay",
        "completed_units": 2,
        "total_units": 10,
        "current_coordinate": 4.0,
        "coordinate_name": "z",
        "coordinate_unit": "um",
        "elapsed_wall_time": 0.5,
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_mpr_links_xyz_indices_and_uses_field_coordinates():
    QApplication.instance() or QApplication([])
    volume = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    coordinates = {
        "z": np.array([1.0, 3.0, 8.0, 13.0]),
        "x": np.array([-4.0, -1.0, 0.5, 2.0, 5.0]),
        "y": np.array([-8.0, -2.0, -0.2, 0.7, 4.0, 9.0]),
    }
    fields = FieldCollection()
    fields.add("preview", make_field(
        "preview", "Downsampled Fast preview", volume,
        ("z", "x", "y"), "intensity_preview",
        {"z": "um", "x": "um", "y": "um"}, "longitudinal",
        coordinates=coordinates,
    ))
    fields.add("preview_xy", make_field(
        "preview_xy", "Downsampled Fast preview x-y", volume[2],
        ("x", "y"), "intensity_preview", {"x": "um", "y": "um"},
        source_volume_key="preview", coordinates=coordinates,
        initially_selected=True,
    ))
    run_data = RunData(
        workflow="pr_timedependent",
        geometry=Geometry(
            x=np.linspace(-10, 10, 50),
            y=np.linspace(-20, 20, 60),
            z=np.linspace(0, 20, 40),
        ),
        fields=fields,
    )
    workspace = Workspace()
    workspace.set_run_data(run_data)
    pane = workspace.longitudinal_pane
    assert (pane._ix, pane._iy, pane._iz) == (2, 2, 2)
    pane.set_cut_indices(4, 5)
    pane.z_plane_slider.setValue(3)
    np.testing.assert_array_equal(pane.xz_view._field.data, volume[:, :, 5])
    np.testing.assert_array_equal(pane.yz_view._field.data, volume[:, 4, :])
    np.testing.assert_array_equal(
        workspace.image_pane.image_view._field.data, volume[3]
    )
    assert workspace.image_pane._z_index == 3
    assert workspace.image_pane.image_view._crosshair_index == (4, 5)
    assert workspace.image_pane.image_view._display_coordinates_to_index(
        4.8, 8.8
    ) == (4, 5)
    assert "x=5" in pane.position_label.text()
    assert "y=9" in pane.position_label.text()
    assert "z=13" in pane.position_label.text()


def test_normal_console_progress_is_bounded_to_phase_and_ten_percent_buckets():
    app = QApplication.instance() or QApplication([])
    window = PRMainWindow()
    emitted = [
        window._progress_console_due("static", index, 100, "solve")
        for index in range(1, 101)
    ]
    assert sum(emitted) == 11
    assert window._progress_console_due("static", 1, 100, "replay") is True
    window.close()
    assert app is not None


def test_workspace_exposes_clean_open_action_for_td_mp4(monkeypatch):
    QApplication.instance() or QApplication([])
    opened = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        lambda url: opened.append(url) or True,
    )
    run_data = RunData(
        workflow="pr_timedependent",
        artifacts={
            "td_preview_movie": ArtifactData(
                key="td_preview_movie",
                display_name="Downsampled TD Preview (MP4)",
                data=np.frombuffer(b"mp4-test", dtype=np.uint8),
                media_type="video/mp4",
                filename="preview.mp4",
                metadata={"visualization_only": True},
            )
        },
    )
    workspace = Workspace()
    workspace.set_run_data(run_data)
    assert workspace.open_td_preview.isVisible() is False
    assert workspace.open_td_preview.isHidden() is False
    workspace.open_td_preview.click()
    assert len(opened) == 1
    assert opened[0].toLocalFile().endswith("preview.mp4")

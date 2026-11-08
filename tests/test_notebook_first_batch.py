"""Synthetic data and fake runners only. No propagation or launch sampling."""
import asyncio
from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from lcprop.notebook import InteractiveBatch
from lcprop.notebook import batch_requests as requests
from lcprop.notebook.batch_files import archive_completed, unpack_completed
from lcprop.notebook.batch_results import draw_plane, retained_plane
from lcprop.notebook.batch_session import BatchSession, Record, identity
from lcprop.products.data_model import FieldCollection, Geometry, RunData, make_field
from tests.test_pr_products import _synthetic_pr_result


def validate(route, request):
    good = SimpleNamespace(require_valid=lambda: None, warnings=(),
                           aperture=SimpleNamespace(warnings=()))
    return requests.validate(route, request, sampling=lambda *a, **kw: good,
                             pr_preflight=lambda r: good)


def products():
    coords = Geometry(x=np.arange(4.), y=np.arange(3.), z=np.arange(2.))
    field = make_field("volume", "Synthetic volume", np.arange(24.).reshape(2, 4, 3),
                       ("z", "x", "y"), "intensity")
    return RunData("static", coords, FieldCollection([("volume", field)]))


def preview(*_):
    return make_field("input", "Synthetic preview", np.ones((4, 3)), ("x", "y"),
                      "intensity", {"x": "um", "y": "um"},
                      coordinates={"x": np.arange(4.), "y": np.arange(3.)})


@pytest.fixture(autouse=True)
def prohibit_science(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real scientific execution is prohibited in these tests")
    monkeypatch.setattr("lcprop.runners.local.LocalRunner.run_operation", forbidden)
    monkeypatch.setattr(requests, "build_launch", forbidden)


@pytest.mark.parametrize("route", ["lc", "pr"])
def test_exact_request_codec_roundtrip_preserves_beam_intent(tmp_path, route):
    session = BatchSession(validator=validate)
    request = requests.preset(route)
    beam = replace(request.beams.channels[0], theta_ext_rad=-0.013,
                   phi_rad=0.3, phase_rad=-0.8, coherence_group="original")
    second = replace(beam, name="other", x0_um=20., coherence_group="other-group")
    request = replace(request, beams=requests.beam_stack(request.beams, (beam, second), None))
    assert session.set_draft(route, request, session.generation)
    path = tmp_path / "draft.lcprop.json"
    session.save_draft(path)
    saved = path.read_bytes()
    session.load_draft(path)
    assert validate(route, session.draft)[0] == validate(route, request)[0]
    assert session.draft.beams.channels[0].theta_ext_rad == -0.013
    assert session.draft.beams.channels[0].phase_rad == -0.8
    assert session.draft.beams.coherence_groups == ("original", "other-group")
    assert requests.beam_stack(request.beams, (beam, second), "coherent").coherence_groups == (
        "original", "original")
    assert path.read_bytes() == saved


def test_unsupported_draft_is_atomic(tmp_path):
    from lcprop.persistence import write_experiment_file

    session = BatchSession(validator=validate)
    original = session.draft
    request = replace(original, grid=replace(original.grid, Nx=64))
    path = tmp_path / "unsupported.json"
    write_experiment_file(path, request, material_id="lc", workflow_id="static",
                          registry=session.experiments)
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="grid"):
        session.load_draft(path)
    assert session.draft is original and path.read_bytes() == saved
    write_experiment_file(path, original, material_id="lc", workflow_id="static",
                          registry=session.experiments, presentation_payload={"unknown": {"keep": 1}})
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="Presentation"):
        session.load_draft(path)
    assert session.draft is original and path.read_bytes() == saved
    request = requests.preset("pr")
    with pytest.raises(ValueError, match="1–3"):
        validate("pr", replace(request, solver=replace(request.solver, Nt=4)))


def test_preview_plumbing_uses_canonical_samples(monkeypatch):
    calls = []
    grid = SimpleNamespace(x_um=np.arange(4.), y_um=np.arange(3.))
    monkeypatch.setattr(requests, "qualify_launch_sampling", lambda *a, **kw:
                        SimpleNamespace(require_valid=lambda: None))

    def fake_grid(spec, **kwargs):
        calls.append((spec.Nx, spec.Ny))
        return grid

    def fake_launch(beams, passed_grid, *, context, complex_dtype):
        assert complex_dtype == np.complex128
        assert grid is passed_grid is context.grid
        assert context.interaction_length_um == 500.
        return SimpleNamespace(A0=np.ones((1, 4, 3), dtype=complex),
                               coherence_groups=beams.coherence_groups)

    monkeypatch.setattr(requests, "make_grid", fake_grid)
    monkeypatch.setattr(requests, "build_launch", fake_launch)
    request = requests.preset("lc")
    field = requests.preview("lc", request)
    assert calls == [(128, 128)]
    np.testing.assert_equal(field.data, np.ones((4, 3)))
    assert field.axes == ("x", "y") and field.coordinates["x"] is grid.x_um


def test_worker_stop_heartbeat_and_stale_click_ownership(monkeypatch):
    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    worker_entered, acknowledged = Event(), Event()
    calls, ui_threads = [], []
    current = get_ident()

    class Fake:
        def run_operation(self, operation, request, *, cancellation_token, progress_callback):
            calls.append((operation.key, request.bias.V_bias, get_ident()))
            if len(calls) == 1:
                return SimpleNamespace(result=SimpleNamespace(status="completed"),
                                       run_data=products(), message="done")
            for unit in range(30):
                progress_callback(SimpleNamespace(completed_units=unit, total_units=100,
                                                   message="fake progress"))
            worker_entered.set()
            for _ in range(400):
                if cancellation_token.is_cancelled():
                    break
                Event().wait(.005)
            assert cancellation_token.is_cancelled()
            acknowledged.set()
            progress_callback(SimpleNamespace(completed_units=1, total_units=2, message="partial"))
            return SimpleNamespace(result=SimpleNamespace(status="stopped"),
                                   run_data=products(), message="stopped")

    async def exercise():
        app = InteractiveBatch(runner=Fake(), validator=validate, preview_factory=preview)
        session = app.session
        original_notify = session.notify
        session.notify = lambda: (ui_threads.append(get_ident()), original_notify())
        old_run = app.run_button
        old_run.click()
        old_run.click()
        await asyncio.wait_for(session.task, 3)
        first = session.completed
        assert first is not None and len(calls) == 1
        old_run.click()
        assert not session.busy
        generation = session.generation
        assert session.start(generation)
        assert not session.start(generation)
        assert not session.set_draft("pr", requests.preset("pr"), generation)
        heartbeats = 0
        while not worker_entered.is_set() or heartbeats < 3 or session.progress is None:
            heartbeats += 1
            await asyncio.sleep(.01)
        assert session.busy and not acknowledged.is_set()
        assert session.progress == (29, 100, "fake progress")
        app.stop_button.click()
        assert session.status == "stopping" and session.busy
        await asyncio.wait_for(session.task, 3)
        assert acknowledged.is_set() and session.completed is first
        assert session.partial.run_id != first.run_id and session.status == "stopped"
        assert all(thread != current for _, _, thread in calls)
        assert set(ui_threads) == {current}
        await app.dispose()

    asyncio.run(exercise())


def test_failure_and_disposal_keep_last_completed():
    class Fake:
        def run_operation(self, operation, request, *, cancellation_token, progress_callback):
            if request.bias.V_bias == 0:
                raise ValueError("fake failure")
            return SimpleNamespace(result=SimpleNamespace(status="completed"),
                                   run_data=products(), message="done")

    async def exercise():
        session = BatchSession(runner=Fake(), validator=validate)
        assert session.start(session.generation)
        await session.task
        first = session.completed
        request = replace(session.draft, bias=replace(session.draft.bias, V_bias=0.))
        session.set_draft("lc", request, session.generation)
        session.start(session.generation)
        await session.task
        assert session.status == "failed" and session.completed is first
        session.set_draft("lc", requests.preset("lc"), session.generation)
        session.start(session.generation)
        await session.dispose()
        assert session.completed is first and not session.busy and session.disposed
        assert not session.start(session.generation)

    asyncio.run(exercise())


def test_retained_cuts_and_aspect():
    from matplotlib.figure import Figure

    data = products()
    field = data.fields["volume"]
    xy = retained_plane(field, data.geometry, ("x", "y"), 1)
    xz = retained_plane(field, data.geometry, ("z", "x"), 2)
    yz = retained_plane(field, data.geometry, ("z", "y"), 3)
    np.testing.assert_equal(xy.values, field.data[1].T)
    np.testing.assert_equal(xz.values, field.data[:, :, 2].T)
    np.testing.assert_equal(yz.values, field.data[:, 3, :].T)
    field = replace(field, units={"x": "mm", "z": "um"})
    xz = retained_plane(field, data.geometry, ("z", "x"), 2)
    fig = Figure()
    physical, square = fig.subplots(1, 2)
    draw_plane(physical, xz, False)
    draw_plane(square, xz, True)
    assert physical.get_aspect() == pytest.approx(1000.)
    assert square.get_box_aspect() == 1
    assert physical.get_xlim() == square.get_xlim()
    assert physical.get_ylim() == square.get_ylim()
    with pytest.raises(ValueError, match="No retained"):
        retained_plane(preview(), None, ("z", "x"))


def test_completed_codec_export_reopen_survives_draft_edit(tmp_path):
    session = BatchSession(validator=validate)
    route = "pr"
    request = requests.preset(route)
    operation, experiment, _ = requests.ROUTES[route]
    from lcprop.pr.checkpoint import PRTimeDependentCheckpoint

    result = _synthetic_pr_result()
    request = replace(request,
                      material=requests.PRMaterialSpec(),
                      grid=replace(request.grid, Nx=4, Ny=3, x_aperture_um=8.,
                                   y_aperture_um=12., dz_um=5., z_length_um=10.),
                      beams=replace(request.beams, channels=(
                          request.beams.channels[0],
                          replace(request.beams.channels[0], name="beam-2"),
                      )))
    checkpoint = PRTimeDependentCheckpoint(
        request, result.E_initial, result.E_final, result.A_initial, 3, 3, .003,
        result.grid_summary, "float64", "complex128")
    result = replace(result, completed_steps=3, requested_steps=3,
                     time_normalized=.003, checkpoint=checkpoint)
    encoded, digest = identity(experiment.encode_request(request))
    session.completed = Record("synthetic-pr", route, request, encoded, digest,
                               result, operation.to_run_data(result))
    first = session.export_completed(tmp_path / "first")
    editable = requests.preset("pr")
    session.set_draft("pr", replace(editable, solver=replace(editable.solver, Nt=1)),
                      session.generation)
    second = session.export_completed(tmp_path / "second")
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    zipped = archive_completed(first)
    restored = unpack_completed(zipped.read_bytes(), tmp_path)
    session.reopen(restored)
    assert session.completed.digest == digest
    np.testing.assert_equal(session.completed.result.E_final, result.E_final)
    assert session.draft.solver.Nt == 1


def test_reject_traversal_in_completed_upload(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../bad", "bad")
    with pytest.raises(ValueError, match="Unexpected"):
        unpack_completed(path.read_bytes(), tmp_path)


def test_lc_completed_transport_and_tamper_rejection(tmp_path):
    from lcprop.lc.results import StaticRunResult

    session = BatchSession(validator=validate)
    request = requests.preset("lc")
    request = replace(request, grid=replace(request.grid, Nx=4, Ny=3,
                      x_aperture_um=8., y_aperture_um=12., dz_um=5., z_length_um=10.))
    operation, experiment, _ = requests.ROUTES["lc"]
    a = np.ones((1, 4, 3), dtype=np.complex128)
    theta = np.zeros((4, 3))
    grid = {"Nx": 4, "Ny": 3, "Nz": 2, "dx_um": 2., "dy_um": 4., "dz_um": 5.,
            "x_aperture_um": 8., "y_aperture_um": 12., "z_length_um": 10.}
    result = StaticRunResult(a, theta, theta, 1., 1., grid, {}, {}, 2,
                             "synthetic", A_initial=a, intensity_stack=np.ones((2, 4, 3)),
                             theta_intensity_stack=np.zeros((2, 4, 3)))
    encoded, digest = identity(experiment.encode_request(request))
    session.completed = Record("synthetic-lc", "lc", request, encoded, digest,
                               result, operation.to_run_data(result))
    first = session.export_completed(tmp_path / "first")
    session.reopen(unpack_completed(archive_completed(first).read_bytes(), tmp_path))
    good = session.completed
    assert good.digest == digest
    assert "intensity_stack" in good.products.fields
    second = session.export_completed(tmp_path / "second")
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    (second / "output/result.json").write_text("{}")
    with pytest.raises(ValueError):
        session.reopen(second)
    assert session.completed is good


def test_widget_displays_synthetic_preview_without_run(monkeypatch):
    from lcprop.notebook import interactive_batch

    displays = []
    monkeypatch.setattr(interactive_batch, "display", displays.append)
    app = InteractiveBatch(validator=validate, preview_factory=preview)
    assert app.session.status == "idle"
    assert app.field.value == "input"
    assert displays
    app.cut.value = "x–z"
    assert app.position.disabled
    app.cut.value = "x–y"
    app.aspect.value = "Square"
    assert displays[-1].axes[0].get_box_aspect() == 1
    asyncio.run(app.dispose())


def test_pr_worker_dispatch_freezes_admitted_request():
    from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION

    calls = []

    class Fake:
        def run_operation(self, operation, request, *, cancellation_token, progress_callback):
            calls.append((operation, request))
            return SimpleNamespace(result=SimpleNamespace(status="completed"),
                                   run_data=products(), message="done")

    async def exercise():
        session = BatchSession(runner=Fake(), validator=validate)
        draft = requests.preset("pr")
        session.set_draft("pr", draft, session.generation)
        session.start(session.generation)
        await session.task
        assert calls == [(PR_TIMEDEPENDENT_OPERATION, draft)]
        admitted = session.completed
        assert admitted.request == draft
        session.set_draft("pr", replace(draft, solver=replace(draft.solver, Nt=1)),
                          session.generation)
        assert admitted.request.solver.Nt == 3
        assert admitted.digest == identity(validate("pr", draft)[0])[1]

    asyncio.run(exercise())


def test_notebook_display_cell_reuses_controller(monkeypatch):
    import json
    from pathlib import Path
    import sys

    import IPython
    import IPython.display
    import lcprop.notebook

    instances, views, setup = [], [], []

    def new_controller():
        controller = SimpleNamespace(widget=object(), session=SimpleNamespace(disposed=False))
        instances.append(controller)
        return controller

    monkeypatch.setattr(lcprop.notebook, "InteractiveBatch", new_controller)
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(
        run_line_magic=lambda *args: setup.append(args)))
    monkeypatch.setattr(IPython.display, "display", views.append)
    monkeypatch.setattr(IPython.display, "clear_output", lambda **_: None)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        output=SimpleNamespace(enable_custom_widget_manager=lambda: setup.append("widgets"))))
    notebook = Path(__file__).resolve().parents[1] / "notebooks/colab_first_batch.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {}
    for _ in range(2):
        exec("".join(cells[-1]["source"]), namespace)
    assert len(instances) == 1
    assert views == [instances[0].widget] * 2
    assert setup == ["widgets", ("matplotlib", "widget")]

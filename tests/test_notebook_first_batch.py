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
    with pytest.raises(ValueError, match="presentation fields: unknown"):
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


@pytest.mark.parametrize(("filename", "controller_name"), [
    ("colab_first_batch.ipynb", "InteractiveBatch"),
    ("lcprop_lc_colab.ipynb", "LCNotebook"),
    ("lcprop_pr_colab.ipynb", "PRNotebook"),
])
def test_notebook_display_cell_reuses_controller(monkeypatch, filename, controller_name):
    import json
    from pathlib import Path
    import sys

    import IPython
    import IPython.display
    import lcprop.notebook

    instances, views, setup = [], [], []

    def new_controller():
        controller = SimpleNamespace(widget=object(), session=SimpleNamespace(disposed=False))
        controller.display_in_colab = lambda: views.append(controller.widget)
        instances.append(controller)
        return controller

    monkeypatch.setattr(lcprop.notebook, controller_name, new_controller)
    monkeypatch.setattr(IPython, "get_ipython", lambda: SimpleNamespace(
        run_line_magic=lambda *args: setup.append(args)))
    monkeypatch.setattr(IPython.display, "display", views.append)
    monkeypatch.setattr(IPython.display, "clear_output", lambda **_: None)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        output=SimpleNamespace(enable_custom_widget_manager=lambda: setup.append("widgets"))))
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / filename
    cells = json.loads(notebook.read_text())["cells"]
    namespace = {}
    for _ in range(2):
        exec("".join(cells[-1]["source"]), namespace)
    assert len(instances) == 1
    assert views == [instances[0].widget] * 2
    assert setup == ["widgets", ("matplotlib", "widget")]


def desktop_fixture(session, path, route, *, remote=False):
    pytest.importorskip("launchplane.serialization")
    from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop, beam_stack_to_launchplane
    from lcprop.persistence import write_experiment_file
    from lcprop.persistence.execution_intent import ExecutionIntent
    from launchplane.serialization import beam_stack_to_dict

    request = requests.preset(route)
    a = replace(request.beams.channels[0], theta_ext_rad=-.021, phi_rad=.4,
                phase_rad=-.7, coherence_group="first")
    b = replace(a, name="second", x0_um=19., theta_ext_rad=.01,
                phase_rad=.6, coherence_group="second")
    stack = beam_stack_to_launchplane(replace(request.beams, channels=(a, b)))
    disabled = replace(stack.beams[0], name="beam-2", enabled=False,
                       phase_rad=-.4, x_um=24.)
    stack = replace(stack, beams=(stack.beams[0], disabled, stack.beams[1]))
    request = replace(request, beams=beam_stack_definition_to_lcprop(stack))
    presentation = {"beam_editor": {"provider": "launchplane", "schema_version": 1,
                                   "beam_stack": beam_stack_to_dict(stack)}}
    intent = None
    if route == "pr":
        intent = ExecutionIntent("slurm" if remote else "local",
                                 "cluster" if remote else None,
                                 "profile" if remote else None,
                                 "numpy", "float64", "fast")
    operation = requests.ROUTES[route][0]
    write_experiment_file(path, request, material_id=operation.material_id,
                          workflow_id=operation.workflow_id, registry=session.experiments,
                          presentation_payload=presentation, execution_intent=intent)
    return request, presentation, intent


@pytest.mark.parametrize("route", ["lc", "pr"])
def test_normal_desktop_draft_roundtrip_and_beam_edits(tmp_path, route):
    import sys

    from lcprop.persistence import read_experiment_file
    beam_stack_from_dict = pytest.importorskip("launchplane.serialization").beam_stack_from_dict
    from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop

    session = BatchSession(validator=validate)
    original = tmp_path / "desktop.lcprop.json"
    request, presentation, intent = desktop_fixture(session, original, route)
    original_bytes = original.read_bytes()
    session.load_draft(original)
    untouched = tmp_path / "untouched.lcprop.json"
    session.save_draft(untouched)
    assert untouched.read_bytes() == original_bytes
    session.load_draft(untouched)
    assert session.draft == request

    first, second = request.beams.channels
    beams = requests.beam_stack(request.beams, (
        replace(first, theta_ext_rad=-.017, phase_rad=-1.2, power_mW=.8),
        replace(second, x0_um=23.),
    ), "coherent")
    assert session.set_draft(route, replace(request, beams=beams), session.generation)
    edited = tmp_path / "edited.lcprop.json"
    session.save_draft(edited)
    loaded = read_experiment_file(edited, registry=session.experiments)
    assert loaded.request == session.draft
    assert loaded.execution_intent == intent
    assert loaded.request.beams.coherence_groups == ("first", "first")
    assert loaded.request.beams.channels[0].theta_ext_rad == -.017
    assert loaded.request.beams.channels[0].phase_rad == -1.2
    assert [ch.name for ch in loaded.request.beams.channels] == [first.name, second.name]
    stack = beam_stack_from_dict(loaded.presentation_payload["beam_editor"]["beam_stack"])
    assert beam_stack_definition_to_lcprop(stack) == loaded.request.beams
    assert loaded.presentation_payload["beam_editor"]["beam_stack"]["beams"][1] == (
        presentation["beam_editor"]["beam_stack"]["beams"][1])
    session.load_draft(edited)
    assert session.draft == loaded.request and original.read_bytes() == original_bytes
    assert not any(name.startswith(("PySide6", "PyQt6")) for name in sys.modules)


def test_remote_desktop_intent_rejected_without_replacing_metadata(tmp_path):
    session = BatchSession(validator=validate)
    local = tmp_path / "local.json"
    remote = tmp_path / "remote.json"
    desktop_fixture(session, local, "pr")
    desktop_fixture(session, remote, "pr", remote=True)
    session.load_draft(local)
    draft, presentation, intent, generation = (
        session.draft, session.presentation, session.execution_intent, session.generation)
    original = remote.read_bytes()
    with pytest.raises(ValueError, match="Slurm execution intent"):
        session.load_draft(remote)
    assert session.draft is draft and session.presentation is presentation
    assert session.execution_intent is intent and session.generation == generation
    assert remote.read_bytes() == original


def test_add_beam_avoids_disabled_desktop_name(tmp_path, monkeypatch):

    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    app = InteractiveBatch(validator=validate, preview_factory=preview)
    path = tmp_path / "desktop.json"
    desktop_fixture(app.session, path, "lc")
    app.session.load_draft(path)
    app._sync()

    app.beam_plane.beam.value = app.session.draft.beams.channels[1].name
    app.beam_plane.delete.click()
    saved = tmp_path / "one-beam.json"
    app.session.save_draft(saved)
    app.session.load_draft(saved)
    app._sync()
    app.beam_plane.add.click()
    assert [beam.name for beam in app.session.draft.beams.channels] == [
        "off-axis", "beam-3"]
    assert app.session.presentation["beam_editor"]["beam_stack"]["beams"][1]["enabled"] is False
    edited = tmp_path / "two-beams.json"
    app.session.save_draft(edited)
    app.session.load_draft(edited)
    assert app.session.draft.beams.channels[1].name == "beam-3"
    asyncio.run(app.dispose())


@pytest.mark.parametrize("value", [
    {"draft.json": {"metadata": {"name": "draft.json"}, "content": b"{}"}},
    ({"name": "draft.json", "content": memoryview(b"{}")},),
])
def test_upload_normalizes_both_widget_formats(value):
    from lcprop.notebook.batch_files import selected_upload

    assert selected_upload(value) == ("draft.json", b"{}")


def test_refresh_recovers_terminal_controls(monkeypatch):
    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    app = InteractiveBatch(validator=validate, preview_factory=preview)
    app.status.value = "Stopping"
    app.session.status = "stopped"
    app.session.generation += 1
    sent = []
    monkeypatch.setattr(app.status, "send_state", lambda *args, **kwargs: sent.append(args))
    app.refresh_button.click()
    assert app.status.value.startswith("<b>stopped</b>")
    assert not app.run_button.disabled
    assert app.stop_button.disabled
    assert ("value",) in sent
    asyncio.run(app.dispose())


def test_widget_backend_reuses_canvas_across_redraws(monkeypatch):
    import ipywidgets as W
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from lcprop.notebook import interactive_batch

    class Canvas(FigureCanvasAgg, W.Box):
        def __init__(self, figure):
            W.Box.__init__(self)
            FigureCanvasAgg.__init__(self, figure)

    figures = []

    def make_figure(**kwargs):
        figure = Figure(**kwargs)
        Canvas(figure)
        figures.append(figure)
        return figure

    monkeypatch.setattr(interactive_batch.plt, "figure", make_figure)
    app = InteractiveBatch(validator=validate, preview_factory=preview)
    canvas, = app.canvas_host.children
    app.aspect.value = "Square"
    app.cut.value = "x–z"
    assert app.canvas_host.layout.display == "none"
    app.cut.value = "x–y"
    assert app.canvas_host.children == (canvas,)
    assert app.canvas_host.layout.display == ""
    assert len(figures) == 1 and len(figures[0].axes) == 2
    asyncio.run(app.dispose())


def test_zip_selection_survives_status_refresh(monkeypatch):
    import ipywidgets as W
    from traitlets import Any, Bool
    from lcprop.notebook import interactive_batch

    class LegacyUpload(W.Box):
        value = Any()
        disabled = Bool(False)

    monkeypatch.setattr(W, "FileUpload", lambda **_: LegacyUpload())
    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    app = InteractiveBatch(validator=validate, preview_factory=preview)
    upload = app.upload
    upload.value = {"completed.zip": {"content": b"synthetic ZIP bytes"}}
    calls = []

    def unpack(data, directory):
        assert data == b"synthetic ZIP bytes"
        return directory / "decoded"

    monkeypatch.setattr(interactive_batch, "unpack_completed", unpack)
    monkeypatch.setattr(app.session, "reopen", calls.append)
    app.session.generation += 1
    app.refresh_button.click()
    assert app.upload is upload and app.files.children[0] is upload
    assert "completed.zip" in app.upload_receipt.value
    app.load_button.click()
    assert calls == [app.directory / "decoded"]
    assert "Reopened" in app.notice.value
    asyncio.run(app.dispose())


def test_native_colab_picker_reuses_validated_reopen(monkeypatch):
    import sys
    from pathlib import Path
    from lcprop.notebook import interactive_batch

    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    app = InteractiveBatch(validator=validate, preview_factory=preview)
    received = []

    def upload(*, target_dir):
        path = Path(target_dir) / "completed.zip"
        path.write_bytes(b"test ZIP")
        return {str(path): path.read_bytes()}

    def unpack(data, directory):
        assert data == b"test ZIP"
        return directory / "decoded"

    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(
        files=SimpleNamespace(upload=upload)))
    monkeypatch.setattr(interactive_batch, "unpack_completed", unpack)
    monkeypatch.setattr(app.session, "reopen", received.append)
    app.open_from_colab()
    assert received == [app.directory / "decoded"]
    assert app.notice.value == "Reopened <b>completed.zip</b>."
    asyncio.run(app.dispose())


@pytest.mark.parametrize("route", ["lc", "pr"])
def test_beam_plane_objects_and_native_roundtrip(tmp_path, monkeypatch, route):
    from lcprop.notebook import LCNotebook, PRNotebook
    from lcprop.notebook.batch_beams import objects

    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    app = (LCNotebook if route == "lc" else PRNotebook)(
        validator=validate, preview_factory=preview)
    panel = app.beam_plane
    try:
        first = panel.beam.value
        assert panel.enabled.disabled and panel.delete.disabled
        panel.add.click()
        second = panel.beam.value
        assert second != first and len(app.session.draft.beams.channels) == 2
        panel.fields["x0_um"].value = 15.0
        panel.fields["theta_ext_rad"].value = -0.03
        panel.fields["phase_rad"].value = -0.8
        assert app.session.draft.beams.channels[1].x0_um == 15.0
        panel.duplicate.click()
        third = panel.beam.value
        assert third not in {first, second} and not panel.enabled.value
        assert panel.fields["phase_rad"].value == -0.8
        panel.beam.value = second
        panel.enabled.value = False
        panel.beam.value = third
        panel.enabled.value = True
        assert [ch.name for ch in app.session.draft.beams.channels] == [first, third]
        saved = tmp_path / "draft.json"
        app.session.save_draft(saved)
        snapshot = objects(app.session.draft.beams, app.session.presentation)
        app.session.load_draft(saved)
        app._refresh_preview()
        app._sync()
        assert objects(app.session.draft.beams, app.session.presentation) == snapshot
        panel.beam.value = second
        assert not panel.enabled.value
        assert panel.fields["theta_ext_rad"].value == -0.03
        panel.delete.click()
        assert second not in [name for _, name in panel.beam.options]

        # A busy or stale editor cannot change the admitted draft.
        draft = app.session.draft
        app.session.active_id = "synthetic-admitted"
        try:
            panel.sync()
            assert all(widget.disabled for widget in panel.inputs)
            panel.fields["x0_um"].value = 7.0
            panel.add.click()
            assert app.session.draft is draft
        finally:
            app.session.active_id = None
        panel.sync()
        assert not panel.add.disabled
    finally:
        asyncio.run(app.dispose())


@pytest.mark.parametrize("route", ["lc", "pr"])
def test_beam_plane_canvas_gestures_and_stale_drag(monkeypatch, route):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from lcprop.notebook import LCNotebook, PRNotebook
    from lcprop.notebook import batch_beam_plane
    from lcprop.notebook._launch_preview import ray_endpoint

    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)

    # Supply an actual Agg canvas for transforms and artists. Its display slot
    # is replaced by a widget without changing the canvas used by the editor.
    import ipywidgets as W

    class Canvas(FigureCanvasAgg, W.HTML):
        def __init__(self, figure):
            W.HTML.__init__(self)
            FigureCanvasAgg.__init__(self, figure)

    def canvas():
        figure = Figure()
        return figure, Canvas(figure)

    monkeypatch.setattr(batch_beam_plane, "widget_canvas", canvas)
    samples = []

    def sample(route, request):
        samples.append(request.beams)
        field = preview()
        return replace(field, data=np.full_like(field.data, request.beams.channels[0].x0_um))

    app = (LCNotebook if route == "lc" else PRNotebook)(
        validator=validate, preview_factory=sample)
    panel = app.beam_plane

    def event(x, y):
        px, py = panel.ax.transData.transform((x, y))
        return SimpleNamespace(inaxes=panel.ax, x=px, y=py, xdata=x, ydata=y, button=1)

    try:
        figure = panel.figure
        first = panel.beam.value
        panel.duplicate.click()
        channel = app.session.draft.beams.channels[1]
        start = event(channel.x0_um, channel.y0_um)
        panel.press(start)
        assert panel.overlap.layout.display == ""
        panel.overlap.value = first
        assert panel.beam.value == first
        panel.press(start)
        end = event(channel.x0_um + 12, channel.y0_um + 4)
        old = app.session.draft
        panel.motion(end)
        assert app.session.draft is old and panel.drag_marker is not None
        panel.release(end)
        assert panel.fields["x0_um"].value == channel.x0_um + 12
        assert app.session.draft.beams.channels[1] == channel
        assert samples[-1] == app.session.draft.beams and panel.figure is figure
        np.testing.assert_equal(panel.ax.collections[0].get_array(),
                                np.full((3, 4), channel.x0_um + 12))

        panel.mode.value = "Tilt"
        assert panel.extent.value == "Fit"
        channel = app.session.draft.beams.channels[0]
        x, y = channel.x0_um, channel.y0_um
        panel.press(event(x, y))
        panel.release(event(x - 2, y))
        tilted = app.session.draft.beams.channels[0]
        assert tilted.theta_ext_rad < 0
        index = requests.refractive_index(route, app.session.draft)
        np.testing.assert_allclose(ray_endpoint(tilted, app.session.draft, index), (x - 2, y))

        if route == "pr":
            circle, = panel.ax.patches
            assert 9 < circle.radius < 10
            panel.press(event(x - 2, y))
            panel.release(event(x + circle.radius * 1.05, y))
            assert app.session.draft.beams.channels[0] == tilted
            assert "outside the reachable exit region" in panel.preview_status.value

        panel.mode.value = "Move"
        panel.extent.value = "Full aperture"
        panel.press(event(x, y))
        app.session.generation += 1
        panel.release(event(x + 4, y))
        assert app.session.draft.beams.channels[0] == tilted
        full = panel.ax.get_xlim()
        panel.extent.value = "Fit"
        assert panel.ax.get_xlim() != full
        panel.extent.value = "Full aperture"
        assert panel.ax.get_xlim() == full
        panel.coherence.value = "coherent"
        assert len(set(app.session.draft.beams.coherence_groups)) == 1
        panel.coherence.value = "incoherent"
        assert len(set(app.session.draft.beams.coherence_groups)) == 2
    finally:
        asyncio.run(app.dispose())
        assert panel.gesture is None


def test_material_notebooks_and_progress_lifecycle(monkeypatch):
    from lcprop.notebook import LCNotebook, PRNotebook

    monkeypatch.setattr(InteractiveBatch, "_draw", lambda self, *_: None)
    advance, finish = Event(), Event()
    callbacks = []

    class Fake:
        def run_operation(self, operation, request, *, cancellation_token, progress_callback):
            callbacks.append(progress_callback)
            if len(callbacks) == 1:
                progress_callback(SimpleNamespace(completed_units=None, total_units=None,
                                                   message="Preparing"))
                assert advance.wait(3)
                progress_callback(SimpleNamespace(completed_units=1, total_units=3,
                                                   message="Material step"))
                assert finish.wait(3)
                return SimpleNamespace(result=SimpleNamespace(status="completed"),
                                       run_data=products(), message="done")
            for _ in range(600):
                if cancellation_token.is_cancelled():
                    break
                Event().wait(.005)
            assert cancellation_token.is_cancelled()
            return SimpleNamespace(result=SimpleNamespace(status="stopped"),
                                   run_data=products(), message="stopped")

    async def wait_until(predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(.01)

        await asyncio.wait_for(poll(), 3)

    async def exercise():
        runner = Fake()
        lc = LCNotebook(runner=runner, preview_factory=preview)
        pr = PRNotebook(runner=runner, preview_factory=preview)
        try:
            assert (lc.session.route, pr.session.route) == ("lc", "pr")
            with pytest.raises(ValueError, match="only accepts LC"):
                lc.session.set_draft("pr", requests.preset("pr"), lc.session.generation)
            pr.run_button.click()
            await wait_until(lambda: "Preparing <progress>" in pr.progress_view.value)
            elapsed = pr.session.elapsed
            await wait_until(lambda: pr.session.elapsed > elapsed)
            assert lc.progress_view.value == "CPU · No admitted run"
            advance.set()
            await wait_until(lambda: "Material step" in pr.progress_view.value)
            assert "1/3" in pr.progress_view.value
            assert pr.session.active_id in pr.progress_view.value
            finish.set()
            await asyncio.wait_for(pr.session.task, 3)
            first = pr.session.completed
            assert first is not None
            pr.run_button.click()
            await wait_until(lambda: len(callbacks) == 2)
            callbacks[0](SimpleNamespace(completed_units=2, total_units=3, message="Stale"))
            pr.stop_button.click()
            assert "Stop requested" in pr.progress_view.value
            await asyncio.wait_for(pr.session.task, 3)
            assert "Stop acknowledged" in pr.progress_view.value
            assert "Stale" not in pr.progress_view.value
            assert pr.session.completed is first
            pr.run_button.click()
            await wait_until(lambda: len(callbacks) == 3)
            await asyncio.wait_for(pr.dispose(), 3)
            assert pr.session.task.done() and not pr.session.busy
        finally:
            advance.set()
            finish.set()
            await lc.dispose()
            if not pr.session.disposed:
                await pr.dispose()

    asyncio.run(exercise())

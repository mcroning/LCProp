"""Local notebook contract checks. No real workflow or GPU is executed."""
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from lcprop.notebook import batch_device, batch_requests, batch_session
from lcprop.notebook.batch_files import archive_completed, save_verified_copy, unpack_completed
from lcprop.notebook.batch_resources import MIB, RESERVE, RESULT_LIMIT, WORK_LIMIT, estimate, require_work
from lcprop.notebook.batch_results import retained_plane
from lcprop.notebook.batch_session import BatchSession
from lcprop.products.data_model import FieldData, Geometry


def forbidden(*args, **kwargs):
    pytest.fail("Real solver or device probe called")


@pytest.fixture(autouse=True)
def guard_real_execution(monkeypatch):
    monkeypatch.setattr(batch_session.LocalRunner, "run_operation", forbidden)
    monkeypatch.setattr(batch_device, "probe_cupy", forbidden)
    for key, (operation, experiment, transport) in tuple(batch_requests.ROUTES.items()):
        monkeypatch.setitem(batch_requests.ROUTES, key,
                            (replace(operation, run=forbidden), experiment, transport))


@pytest.fixture
def session():
    return BatchSession(memory_available=lambda: 4096 * MIB)


@pytest.mark.parametrize("route,key,shape", [
    ("lc", ("lc", "static"), (128, 128, 100)),
    ("lc_td", ("lc", "timedependent"), (128, 128, 100)),
    ("pr_static", ("pr", "pr_static"), (64, 48, 1000)),
    ("pr", ("pr", "pr_timedependent"), (64, 48, 1000)),
])
def test_defaults_codec_and_fake_dispatch(tmp_path, session, route, key, shape):
    session.switch(route)
    request = session.draft
    grid = request.grid
    assert (grid.Nx, grid.Ny, round(grid.z_length_um / grid.dz_um)) == shape
    charge = estimate(route, request)
    assert charge.retained <= RESULT_LIMIT and charge.working <= WORK_LIMIT
    assert session.preflight(route, request)[2] == charge
    request = replace(request, grid=replace(grid, Nx=grid.Nx + 2))
    assert session.set_draft(route, request, session.generation)
    path = session.save_draft(tmp_path / "draft.json")
    session.load_draft(path)
    assert session.draft == request
    if route == "lc_td":
        assert request.initial_theta is None and request.initial_A is None
        assert request.solver.Nt == 20
    calls = []

    def dispatch(operation, snapshot, **kwargs):
        calls.append((operation.key, snapshot))
        kwargs["progress_callback"](SimpleNamespace(
            stage="fake step", completed_units=1, total_units=1))
        return SimpleNamespace(
            result=SimpleNamespace(status="completed", backend_summary={
                "backend": "numpy", "real_dtype": "float64", "complex_dtype": "complex128",
            }, diagnostics={"backend": {
                "backend": "numpy", "real_dtype": "float64", "complex_dtype": "complex128",
            }}), run_data=None, message="fake complete",
        )

    session.runner = SimpleNamespace(run_operation=dispatch)

    async def exercise():
        assert session.start(session.generation)
        await session.task
    asyncio.run(exercise())
    assert session.status == "completed", session.message
    assert session.completed.route == route
    assert session.progress == (1, 1, "fake step")
    assert calls[0][0] == key and calls[0][1].grid == request.grid
    assert session.completed.device.resolved == "numpy"
    if route == "lc_td":
        with pytest.raises(ValueError, match="LC time-dependent completed ZIP"):
            session.export_completed(tmp_path / "unsupported")


def test_route_state_copy_and_atomic_rejection(tmp_path, session):
    session.draft = replace(session.draft, grid=replace(session.draft.grid, Ny=130))
    original = session.draft
    old_generation = session.generation
    session.presentation = {"marker": 1}
    session.switch("lc_td")
    assert not session.set_draft("lc_td", session.draft, old_generation)
    solver = session.draft.solver
    session.copy_grid("lc")
    assert session.draft.grid == original.grid and session.draft.solver == solver
    session.switch("lc")
    assert session.draft == original and session.presentation == {"marker": 1}
    session.draft = replace(original, grid=replace(original.grid, Nx=1))

    async def reject():
        with pytest.raises(ValueError, match="integers"):
            session.start(session.generation)
        session.draft = replace(original, grid=replace(original.grid, Nx=4096))
        with pytest.raises(ValueError, match="256 MiB"):
            session.start(session.generation)
    asyncio.run(reject())
    assert not session.busy


def test_budget_boundaries():
    require_work(WORK_LIMIT, 0, WORK_LIMIT + RESERVE)
    with pytest.raises(ValueError, match="budget"):
        require_work(WORK_LIMIT, 1, WORK_LIMIT + RESERVE)
    with pytest.raises(ValueError, match="budget"):
        require_work(1, 0, RESERVE)
    with pytest.raises(ValueError, match="unknown"):
        require_work(1, 0, None)


def test_stop_ownership(session):
    entered, release = Event(), Event()

    def dispatch(operation, request, **kwargs):
        entered.set()
        assert release.wait(3)
        assert kwargs["cancellation_token"].is_cancelled()
        return SimpleNamespace(result=SimpleNamespace(status="completed"), run_data=None,
                               message="fake late completion")
    session.runner = SimpleNamespace(run_operation=dispatch)

    async def exercise():
        assert session.start(session.generation)
        assert await asyncio.to_thread(entered.wait, 3)
        with pytest.raises(RuntimeError, match="running"):
            session.switch("lc_td")
        assert not session.start(session.generation)
        session.stop()
        assert session.status == "stopping"
        release.set()
        await session.task
        assert session.status == "stopped"
        assert session.completed is None and session.partial is not None
        session.switch("lc_td")
        assert session.partial is None
        session.switch("lc")
        assert session.partial is not None
    asyncio.run(exercise())


def test_synthetic_far_field_and_fixed_cut():
    coordinates = {"s_x": [-.2, .2], "s_y": [-.3, 0, .3]}
    field = FieldData("far", "Far field", np.ones((2, 3)), ("s_x", "s_y"), "far",
                      {"s_x": "1", "s_y": "1"}, coordinates=coordinates)
    assert retained_plane(field, None, field.axes).values.shape == (3, 2)
    cut = FieldData("xz", "Fixed", np.ones((2, 3)), ("z", "x"), "intensity",
                    coordinates={"y_cut_um": 1.5})
    geometry = Geometry(x=[0, 1, 2], z=[1, 2])
    assert "fixed y = 1.5" in retained_plane(cut, geometry, ("z", "x")).caption
    with pytest.raises(ValueError, match="No retained"):
        retained_plane(cut, geometry, ("x", "y"))


def test_verified_drive_copy(tmp_path, monkeypatch):
    from lcprop.notebook import batch_files

    source = tmp_path / "artifact.json"
    source.write_bytes(b"saved request")
    drive = tmp_path / "drive"
    drive.mkdir()
    target = save_verified_copy(source, drive)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        save_verified_copy(source, drive)
    target.unlink()
    monkeypatch.setattr(batch_files.shutil, "copyfileobj",
                        lambda inp, out: out.write(b"incorrect"))
    with pytest.raises(OSError, match="checksum"):
        save_verified_copy(source, drive)
    assert not target.exists() and source.read_bytes() == b"saved request"


def test_legacy_pr_zip(tmp_path, session):
    fixture = os.environ.get("LCPROP_NOTEBOOK_PR_ZIP")
    receipt = os.environ.get("LCPROP_NOTEBOOK_PR_RECEIPT")
    if not fixture or not receipt:
        pytest.skip("Provide the retained PR ZIP and receipt")
    source = Path(fixture).read_bytes()
    expected = json.loads(Path(receipt).read_text())
    assert hashlib.sha256(source).hexdigest() == expected["source_zip_sha256"]
    directory = unpack_completed(source, tmp_path)
    session.memory_available = lambda: 1300 * MIB
    session.reopen(directory, upload_bytes=len(source))
    result_bytes, archive_bytes = session.resident_storage()
    assert result_bytes <= RESULT_LIMIT < result_bytes + archive_bytes
    assert archive_bytes == sum(len(data) for _, data in session.completed.package_files)
    accepted = session.completed
    session.memory_available = lambda: 800 * MIB
    with pytest.raises(ValueError, match="Working-memory budget"):
        session.reopen(directory)
    assert session.completed is accepted
    session.memory_available = lambda: 4096 * MIB
    assert session.route == "pr" and session.completed.result.completed_steps == 3
    result = archive_completed(session.export_completed(tmp_path / "export"))
    with ZipFile(result) as archive:
        members = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    assert len(members) == 9
    assert members == {name: value["source_sha256"] for name, value in expected["members"].items()}


@pytest.mark.parametrize("material,alternate", [("lc", "lc_td"), ("pr", "pr_static")])
def test_grid_solver_and_workflow_widgets(monkeypatch, material, alternate):
    from lcprop.notebook.interactive_batch import LCNotebook, PRNotebook

    cls = LCNotebook if material == "lc" else PRNotebook
    monkeypatch.setattr(cls, "_draw", lambda self, *_: None)
    app = cls(preview_factory=lambda *_: None, memory_available=lambda: 4096 * MIB)
    original = app.session.draft
    app.grid_inputs["Nx"].value += 2
    app.grid_inputs["z_length_um"].value += .1
    assert app.session.draft.grid.Nx == original.grid.Nx + 2
    assert app.session.draft.grid.z_length_um == original.grid.z_length_um + .1
    app.workflow_control.value = alternate
    assert app.session.route == alternate, app.notice.value
    key = "Nt" if material == "lc" else "max_coupled_passes"
    app.solver_inputs[key].value += 1
    assert getattr(app.session.draft.solver, key) == getattr(batch_requests.preset(alternate).solver, key) + 1
    app.workflow_control.value = material
    assert app.session.draft.grid.Nx == original.grid.Nx + 2
    app.grid_inputs["Nx"].value = 1
    assert app.run_button.disabled and app._preview is None
    asyncio.run(app.dispose())


def test_pr_static_completed_codec(tmp_path, session):
    from tests.test_pr_static_products import _synthetic_static_result

    session.switch("pr_static")
    draft = session.draft
    session.draft = replace(draft, grid=replace(draft.grid, Nx=4, Ny=3,
        x_aperture_um=8., y_aperture_um=12., z_length_um=10., dz_um=5.))
    result = replace(_synthetic_static_result(), backend_summary={
        "backend": "numpy", "real_dtype": "float64", "complex_dtype": "complex128",
    })

    def dispatch(operation, request, **kwargs):
        return SimpleNamespace(result=result, run_data=operation.to_run_data(result),
                               message="synthetic static")
    session.runner = SimpleNamespace(run_operation=dispatch)

    async def exercise():
        assert session.start(session.generation)
        await session.task

    asyncio.run(exercise())
    assert session.status == "completed", session.message
    assert session.completed.result.status == "converged"
    directory = session.export_completed(tmp_path / "pr-static")
    archive = archive_completed(directory)
    session.reopen(unpack_completed(archive.read_bytes(), tmp_path))
    assert session.completed.result.status == "converged"
    assert "final_E_stack" in session.completed.products.fields


def test_elapsed_budget_preserves_completed(monkeypatch, session):
    entered = Event()
    now = [0.]
    monkeypatch.setattr(batch_session, "monotonic", lambda: now[0])

    def dispatch(operation, request, **kwargs):
        if entered.is_set():
            token = kwargs["cancellation_token"]
            for _ in range(400):
                if token.is_cancelled():
                    break
                Event().wait(.01)
            assert token.is_cancelled()
        entered.set()
        return SimpleNamespace(result=SimpleNamespace(status="completed"), run_data=None,
                               message="fake complete")

    async def exercise():
        session.runner = SimpleNamespace(run_operation=dispatch)
        assert session.start(session.generation)
        await session.task
        accepted = session.completed
        assert session.start(session.generation)
        await asyncio.sleep(.02)
        now[0] = 121.
        await asyncio.wait_for(session.task, 5.)
        assert session.status == "stopped"
        assert session.completed is accepted
    asyncio.run(exercise())


def test_gpu_admission_does_not_probe_other_routes():
    from lcprop.notebook.batch_device import DeviceUnavailable, resolve_device

    for route in ("pr_static", "pr"):
        draft = batch_requests.preset(route)
        draft = replace(draft, grid=replace(draft.grid, Nx=draft.grid.Nx + 2),
                        backend=replace(draft.backend, backend="cupy"))
        with pytest.raises(DeviceUnavailable, match="qualified reduced PR TD envelope"):
            resolve_device(route, draft, forbidden)


def test_aggregate_nested_limit_before_decode(tmp_path, monkeypatch, session):
    from zipfile import ZIP_DEFLATED
    from lcprop.notebook import batch_files

    package = tmp_path / "package"
    with ZipFile(tmp_path / "input.zip", "w") as outer:
        for group, name in (("request", "request_arrays"), ("output", "result_arrays")):
            folder = package / group
            folder.mkdir(parents=True)
            path = folder / f"{name}.npz"
            with ZipFile(path, "w", ZIP_DEFLATED) as inner:
                inner.writestr("data.npy", b"x" * 6000)
            outer.write(path, path.relative_to(package))
    monkeypatch.setattr(batch_files, "MAX_UPLOAD_BYTES", 10000)
    monkeypatch.setattr(batch_session, "read_request_package", forbidden)
    for action in (lambda: archive_completed(package),
                   lambda: unpack_completed((tmp_path / "input.zip").read_bytes(), tmp_path),
                   lambda: session.reopen(package)):
        with pytest.raises(ValueError, match="Combined expanded"):
            action()
    assert not list(tmp_path.glob("reopen-*"))


def test_drive_interruption_and_promotion_collision(tmp_path, monkeypatch):
    from lcprop.notebook import batch_files

    source = tmp_path / "result.zip"
    source.write_bytes(b"complete artifact")
    drive = tmp_path / "drive"
    drive.mkdir()
    target = drive / source.name
    original_copy = batch_files.shutil.copyfileobj

    def interrupt(inp, out):
        out.write(b"partial")
        assert not target.exists()
        raise OSError("mock interrupted transfer")

    monkeypatch.setattr(batch_files.shutil, "copyfileobj", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        save_verified_copy(source, drive)
    assert not list(drive.iterdir())

    def collide(inp, out):
        original_copy(inp, out)
        target.write_bytes(b"another writer")

    monkeypatch.setattr(batch_files.shutil, "copyfileobj", collide)
    with pytest.raises(FileExistsError):
        save_verified_copy(source, drive)
    assert list(drive.iterdir()) == [target]
    assert target.read_bytes() == b"another writer"
    assert source.read_bytes() == b"complete artifact"


@pytest.mark.parametrize("outcome", ["success", "collision", "interrupted", "corrupt"])
def test_drive_exclusive_create_fallback(tmp_path, monkeypatch, outcome):
    from lcprop.notebook import batch_files

    source = tmp_path / "result.zip"
    source.write_bytes(b"verified artifact")
    drive = tmp_path / "drive"
    drive.mkdir()
    target = drive / source.name
    original_copy = batch_files.shutil.copyfileobj

    def unsupported(staged, destination):
        if outcome == "collision":
            destination.write_bytes(b"another writer")
        raise batch_files._AtomicPromotionUnsupported()

    def copy(inp, out):
        if Path(out.name) == target:
            if outcome == "interrupted":
                out.write(b"partial")
                raise OSError("interrupted")
            if outcome == "corrupt":
                out.write(b"bad bytes")
                return
        original_copy(inp, out)

    monkeypatch.setattr(batch_files, "_promote_verified_copy", unsupported)
    monkeypatch.setattr(batch_files.shutil, "copyfileobj", copy)
    if outcome == "success":
        assert save_verified_copy(source, drive) == target
        assert target.read_bytes() == source.read_bytes()
    elif outcome == "collision":
        with pytest.raises(FileExistsError):
            save_verified_copy(source, drive)
        assert target.read_bytes() == b"another writer"
    else:
        with pytest.raises(OSError, match="interrupted|checksum"):
            save_verified_copy(source, drive)
        assert not target.exists()
    assert not list(drive.glob(".lcprop-copy-*"))
    assert source.read_bytes() == b"verified artifact"

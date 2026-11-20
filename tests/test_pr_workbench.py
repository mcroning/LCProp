"""Fake dispatch and synthetic saved data only; real execution is forbidden."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest

from lcprop.notebook import LCNotebook, PRNotebook, PRWorkbench, open_pr_workbench
from lcprop.notebook import batch_requests as requests
from lcprop.notebook import pr_workbench_preview as preview_module
from lcprop.notebook.batch_files import archive_completed
from lcprop.notebook.batch_resources import RESULT_LIMIT, WORK_LIMIT, RESERVE, MIB, Charge
from lcprop.notebook.batch_session import Record, identity
from tests.test_notebook_first_batch import validate, preview
from tests.test_pr_static_products import _synthetic_static_result


@pytest.fixture(autouse=True)
def prohibit_execution(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Real execution/launch forbidden in workbench tests")
    monkeypatch.setattr("lcprop.runners.local.LocalRunner.run_operation", forbidden)
    monkeypatch.setattr(requests, "build_launch", forbidden)
    monkeypatch.setattr(preview_module, "build_launch", forbidden)
    monkeypatch.setattr("lcprop.notebook.batch_session.available_memory", lambda: 4 * 1024**3)
    monkeypatch.setitem(sys.modules, "cupy", None)


@pytest.fixture
def app(tmp_path):
    panel = PRWorkbench(files_root=tmp_path, validator=validate, preview_factory=preview)
    yield panel
    panel.session.notify = lambda: None


def hashes(path):
    with zipfile.ZipFile(path) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}


def test_routes_controls_public_limits_and_no_private_policy(app):
    assert app.session.route == "pr" and app.device_control.value == "numpy"
    assert {v for _, v in app.workflow_control.options} == {"pr_static", "pr"}
    assert app.advanced.children[1].layout.display == "none"
    assert (RESULT_LIMIT, WORK_LIMIT, RESERVE) == (256*MIB, 2560*MIB, 512*MIB)
    assert type(app.session).__name__ == "BatchSession"
    before = deepcopy(app.session.draft)
    app.workflow_control.value = "pr_static"
    assert app.session.route == "pr_static"
    app.grid_inputs["Nx"].value = 80
    static = deepcopy(app.session.draft)
    app.workflow_control.value = "pr"
    assert app.session.draft == before
    app.workflow_control.value = "pr_static"
    assert app.session.draft == static
    for field in app.beam_plane.fields.values():
        assert not field.disabled


def test_synthetic_example_and_exact_nine_member_roundtrip(app, tmp_path):
    app.open_example()
    record = app.session.completed
    assert record.run_id == "synthetic-pr-demo" and app.source.value == "Completed"
    assert "Synthetic example" in app.scientific_status.value
    assert "TD evolution" in app.scientific_status.value
    app.field.value = next(key for key, field in record.products.fields.items()
                           if field.data.ndim == 3)
    for plane in ("x–y", "x–z", "y–z"):
        app.cut.value = plane
        for aspect in ("Physical", "Square"):
            app.aspect.value = aspect
            app._draw()
            assert not app.plot_message.value
    archive = archive_completed(app.session.export_completed(tmp_path/'export'))
    resource = resources.files("lcprop.notebook.assets").joinpath("synthetic_pr.zip")
    with resources.as_file(resource) as source:
        assert len(hashes(source)) == 9
        assert hashes(source) == hashes(archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith('.json'):
                assert b'/Users/' not in z.read(name) and b'mcroning' not in z.read(name)


def test_opening_refresh_and_redisplay_preserve_unsaved_state(app, monkeypatch):
    app.open_example()
    app.grid_inputs['Nx'].value = 80
    draft = deepcopy(app.session.draft); result = app.session.completed
    invalid = app.solver_inputs['Nt']
    invalid.value = 5
    draft = deepcopy(app.session.draft)
    namespace = {'lcprop_pr_workbench': app}
    monkeypatch.setattr(app, 'display_in_colab', lambda: None)
    for _ in range(3):
        assert open_pr_workbench(namespace) is app
        app.refresh_button.click()
        assert app.session.draft == draft and app.session.completed is result
        assert app.solver_inputs['Nt'].value == 5
    callbacks = app.aspect._trait_notifiers['value']['change']
    assert [c for c in callbacks if getattr(c, '__self__', None) is app] == [app._draw]


def test_invalid_text_is_not_reset_by_refresh(app):
    app.workflow_control.value = 'pr_static'
    control = app.solver_inputs['residual_rms_tolerance']
    control.value = 'invalid'
    before = deepcopy(app.session.draft)
    app._refresh_status()
    assert control.value == 'invalid' and app.run_button.disabled
    assert app.session.draft == before


def test_independent_image_cannot_enable_invalid_scientific_grid(app):
    def strict(route, request):
        if request.grid.Nx < 200:
            raise ValueError('Scientific sampling invalid')
        return validate(route, request)
    app.session.validator = strict
    app.grid_inputs['Nx'].value = 64
    app.beam_plane.fields['x0_um'].value = 1.0
    assert app._preview is not None and app.run_button.disabled
    assert 'Editing preview only' in app.beam_plane.preview_status.value
    with pytest.raises(ValueError, match='sampling invalid'):
        app.session.preflight(app.session.route, app.session.draft)
    app.grid_inputs['Nx'].value = 256
    # Resource checks also remain active; use a short synthetic grid.
    app.grid_inputs['z_length_um'].value = 10
    assert not app.run_button.disabled


def test_preview_sampling_factory_is_bounded_and_request_preserving(monkeypatch):
    request = requests.preset('pr')
    request = replace(request, grid=replace(request.grid, Nx=16, Ny=8))
    original = deepcopy(request)
    def qualify(beams, grid, *a, **k):
        return SimpleNamespace(errors=(() if grid.Nx >= 256 else ('Beam 1 (demo), x: Nyquist',)),
                               warnings=(), require_valid=lambda: None)
    monkeypatch.setattr(preview_module, 'qualify_launch_sampling', qualify)
    assert preview_module.editing_grid('pr', request).Nx == 256
    assert request == original
    monkeypatch.setattr(preview_module, 'MAX_PREVIEW_SAMPLES', 100)
    with pytest.raises(ValueError, match='display budget'):
        preview_module.editing_grid('pr', request)
    with pytest.raises(ValueError, match='reduced'):
        preview_module.editing_grid('pr_xy_static', request)


def test_preview_uses_canonical_helpers_with_fake_launch(monkeypatch):
    request = requests.preset('pr');spec = request.grid
    grid = SimpleNamespace(x_um=np.arange(4.), y_um=np.arange(3.))
    calls=[]
    monkeypatch.setattr(preview_module, 'editing_grid', lambda *a: spec)
    monkeypatch.setattr(preview_module, 'make_grid', lambda *a, **k: grid)
    def fake_launch(beams, runtime, *, context, complex_dtype):
        calls.append((runtime is grid, context.grid is grid, complex_dtype))
        return SimpleNamespace(A0=np.ones((1,4,3),dtype=complex),coherence_groups=beams.coherence_groups)
    monkeypatch.setattr(preview_module, 'build_launch', fake_launch)
    field=preview_module.editing_preview('pr',request)
    assert calls == [(True,True,np.complex128)]
    np.testing.assert_array_equal(field.data,np.ones((4,3)))
    assert field.axes==('x','y') and 'editing preview' in field.display_name


def test_unavailable_longitudinal_draft_does_not_erase_workbench(app, monkeypatch):
    children=app.widget.children
    monkeypatch.setattr(app.plot,'clear_output',lambda *a,**k:pytest.fail('parent output cleared'))
    app.cut.value='x–z';app._draw()
    assert 'XY data only' in app.plot_message.value
    assert app.widget.children == children
    app.workflow_control.value='pr_static'
    assert app.widget.children == children


def test_files_success_failure_and_size_guards(app,tmp_path):
    app.open_example();prior=app.session.completed
    path=app.session.save_draft(tmp_path/'beams.lcprop.json')
    app.scan_files();assert str(path) in dict(app.saved_file.options).values()
    app.grid_inputs['Nx'].value=80
    app.open_path(path)
    assert app.session.completed is prior
    bad=tmp_path/'bad.zip';bad.write_bytes(b'bad')
    with pytest.raises(zipfile.BadZipFile):app.open_path(bad)
    assert app.session.completed is prior
    huge=tmp_path/'huge.zip'
    with huge.open('wb') as f:f.truncate(128*MIB+1)
    with pytest.raises(ValueError,match='128 MiB'):app.open_path(huge)
    assert app.session.completed is prior
    with pytest.raises(ValueError,match='256 MiB'):
        app.session.require_admission(Charge(257*MIB,1,1))


def test_static_and_partial_statuses_and_record_replacement(app):
    app.workflow_control.value='pr_static';s=app.session
    request=replace(s.draft,grid=replace(s.draft.grid,Nx=4,Ny=3,x_aperture_um=8.,y_aperture_um=12.,dz_um=5.,z_length_um=10.))
    op,codec,_=requests.ROUTES['pr_static'];encoded,sha=identity(codec.encode_request(request))
    for status in ('not_converged','converged'):
        result=_synthetic_static_result(status=status)
        s.completed=Record('static-'+status,'pr_static',request,encoded,sha,result,op.to_run_data(result))
        s._notify()
        assert ('Static converged' if status=='converged' else 'Static not converged') in app.scientific_status.value
        assert s.completed.run_id in app.scientific_status.value
        assert 'input_preview' not in dict(app.field.options).values()
    s.partial=s.completed;s._notify();app.source.value='Stopped / partial'
    assert 'Stopped / partial' in app.scientific_status.value


def test_fake_stop_error_and_prior_result_preservation(app):
    app.open_example();s=app.session;prior=s.completed
    class Fake:
        def run_operation(self,op,request,*,cancellation_token,progress_callback,**kw):
            while not cancellation_token.is_cancelled():
                progress_callback(SimpleNamespace(completed_units=1,total_units=3,stage='Fake only',message=''))
                time.sleep(.01)
            return SimpleNamespace(result=SimpleNamespace(status='cancelled',diagnostics={'backend':{'backend':'numpy','real_dtype':'float64','complex_dtype':'complex128'}}),run_data=prior.products,message='Fake Stop acknowledged')
    async def check():
        s.runner=Fake();app.run_button.click()
        for _ in range(100):
            await asyncio.sleep(.01)
            if s.progress:break
        assert s.busy and s.progress and app.workflow_control.disabled
        assert 'Another run is active' in app.run_availability.value
        app.stop_button.click();await asyncio.wait_for(s.task,3)
        assert s.status=='stopped' and s.completed is prior and s.partial is not None
        class Failure:
            def run_operation(self,*a,**k):raise ValueError('Fake failure')
        s.runner=Failure();app.run_button.click();await asyncio.wait_for(s.task,3)
        assert s.status=='failed' and s.completed is prior
    asyncio.run(check())


def test_lazy_core_and_notebook_import_without_optional_dependencies():
    code='''
import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'matplotlib','ipywidgets','IPython','cupy','PySide6','launchplane'}:raise AssertionError(name)
sys.meta_path.insert(0,Block())
import lcprop
import lcprop.notebook
assert 'PRWorkbench' in lcprop.notebook.__all__
assert 'lcprop.notebook.pr_workbench' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_existing_lc_and_pr_views_retain_default_composition():
    for cls in (LCNotebook,PRNotebook):
        p=cls(validator=validate,preview_factory=preview)
        assert tuple(p.editor_sections.values())==p.editor.children
        assert len(p.editor.children)==8
        assert p.beam_plane_type is not PRWorkbench.beam_plane_type


def test_export_limit_and_cpu_device_guard_are_unchanged(app, tmp_path):
    app.open_example();prior=app.session.completed
    app.session.completed=replace(prior,export_bytes=128*MIB+1)
    app.session._notify()
    assert app.export_button.disabled and '128 MiB' in app.export_notice.value
    with pytest.raises(ValueError,match='128 MiB'):
        app.session.export_completed(tmp_path/'oversize')
    assert not (tmp_path/'oversize').exists()
    app.session.completed=prior;app.session._notify()
    app.session.gpu_probe=lambda *a:pytest.fail('CPU must not probe CUDA')
    app.device_control.value='numpy'
    app.session.preflight(app.session.route,app.session.draft)
    assert not app.run_button.disabled
    app.workflow_control.value='pr_static'
    app.device_control.value='cupy'
    assert app.run_button.disabled
    app.device_control.value='numpy'
    assert not app.run_button.disabled


def test_disposed_or_foreign_namespace_is_not_silently_replaced(app):
    namespace={'lcprop_pr_workbench':object()}
    with pytest.raises(RuntimeError,match='Save your work'):
        open_pr_workbench(namespace,hosted=False)
    namespace={'lcprop_pr_workbench':app}
    asyncio.run(app.dispose())
    with pytest.raises(RuntimeError,match='Save needed files'):
        open_pr_workbench(namespace,hosted=False)
    assert namespace['lcprop_pr_workbench'] is app


def test_source_notebook_is_clean_and_has_one_noninstalling_open_cell():
    import ast
    import json
    notebook = json.loads((Path(__file__).parents[1] / "notebooks" /
                           "pr_workbench_source.ipynb").read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 1 and cells[0]["outputs"] == []
    code = "".join(cells[0]["source"])
    ast.parse(code)
    assert "_open_pr_notebook(globals())" in code
    assert "drive.mount" not in code
    assert "PIN_SETUP_SHA256" not in code and "PIN_WHEEL_SHA256" not in code
    assert "run_operation" not in code


def test_workbench_figure_fits_narrow_panel_and_default_pr_size_unchanged(app):
    import matplotlib.pyplot as plt
    small = app._new_result_figure()
    regular = PRNotebook._new_result_figure(app)
    try:
        assert tuple(small.get_size_inches()) == (3.6, 3.2)
        assert tuple(regular.get_size_inches()) == (6., 4.)
    finally:
        plt.close(small)
        plt.close(regular)


def test_result_limit_explained_next_to_run_without_summary_dump(app):
    app.workflow_control.value = 'pr_static'
    app.beam_plane.duplicate.click()
    for key, value in {'Nx':256, 'Ny':64, 'z_length_um':1000., 'dz_um':5.}.items():
        app.grid_inputs[key].value = value
    draft = deepcopy(app.session.draft)
    assert app.run_button.disabled
    assert '265.0 MiB exceeds the 256 MiB per-result limit' in app.run_availability.value
    assert 'Review Grid settings' in app.run_button.tooltip
    assert app.run_availability in app.editor_sections['buttons'].children
    assert 'Working ' not in app.editor.children[2].value
    assert app.session.draft == draft
    app.grid_inputs['Nx'].value = 128
    assert not app.run_button.disabled
    assert 'Inputs admitted' in app.run_availability.value
    assert '265.0' not in app.run_availability.value


def test_invalid_text_unknown_memory_and_device_each_explain_run(app):
    app.workflow_control.value = 'pr_static'
    text = app.solver_inputs['residual_rms_tolerance']
    text.value = 'invalid'
    assert app.run_button.disabled and 'invalid' in app.run_availability.value
    text.value = ''
    app.session.memory_available = lambda: None
    app._update_editor_summary()
    assert app.run_button.disabled and 'memory availability is unknown' in app.run_availability.value
    app.session.memory_available = lambda: 4 * 1024**3
    app.device_control.value = 'cupy'
    assert app.run_button.disabled and 'Select CPU for PR Static' in app.run_availability.value
    app.device_control.value = 'numpy'
    assert not app.run_button.disabled and 'Inputs admitted' in app.run_availability.value


def test_busy_run_explanation_preserves_limits_and_result(app):
    app.open_example()
    result = app.session.completed
    app.session.status = 'running'
    app.session.active_id = 'fake-run'
    app._update_editor_summary()
    assert app.run_button.disabled and 'Another run is active' in app.run_availability.value
    assert app.session.completed is result
    app.session.status = 'idle'
    app.session.active_id = None
    app._update_editor_summary()
    assert not app.run_button.disabled
    assert (RESULT_LIMIT, WORK_LIMIT, RESERVE) == (256*MIB, 2560*MIB, 512*MIB)


def test_uploaded_json_names_are_listed_and_open_without_result_loss(app, tmp_path):
    app.open_example()
    prior = app.session.completed
    saved = app.session.save_draft(tmp_path / 'original.lcprop.json')
    draft = deepcopy(app.session.draft)
    names = ['beam-study.json', 'BEAMS.JSON', 'upper.LCPROP.JSON']
    for name in names:
        (tmp_path / name).write_bytes(saved.read_bytes())
    (tmp_path / 'notes.txt').write_text('not an experiment')
    app.scan_files()
    options = dict(app.saved_file.options)
    assert all(str(tmp_path / name) in options.values() for name in names)
    assert str(tmp_path / 'notes.txt') not in options.values()
    for name in names:
        app.saved_file.value = str(tmp_path / name)
        app.scan_files()
        assert app.saved_file.value == str(tmp_path / name)
        app.grid_inputs['Nx'].value = 80
        app.open_button.click()
        assert app.session.draft == draft
        assert app.session.completed is prior
        assert 'Loaded' in app.notice.value

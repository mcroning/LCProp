"""Stage 2C isolation using synthetic results only; no propagation."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from lcprop.notebook import interactive_lc as ui
from tests.test_interactive_lc_dragging import app, drag, event
from tests.test_interactive_lc_multibeam_presentation import snapshot


def record(request):
    return ui.CompletedRun(json.dumps(ui.encode_lc_static_request(request), sort_keys=True),
                           snapshot('single').arrays_npz,
                           json.dumps({'iteration_history': [{'residual': .001}],
                                       'request_x': request.beams.channels[0].x0_um}))


def test_capture_busy_guards_publication_and_new_run(app, monkeypatch, tmp_path):
    calls=[]
    canvas=app._position_editor.canvas
    original=app.request()
    def fake(request, progress, *, time_limit):
        calls.append(request)
        assert request == original and request is not app.base
        encoded=ui.encode_lc_static_request(request)
        app.controls['x'].value=9
        assert app.controls['x'].value == original.beams.channels[0].x0_um
        app.load('/must-not-open')
        app._restore()
        app._run()
        drag(app._position_editor,(-20,0),(-10,2))
        progress(SimpleNamespace(completed_units=100,total_units=100,elapsed_wall_time=1))
        assert ui.encode_lc_static_request(request)==encoded
        return record(request)
    monkeypatch.setattr(ui,'execute',fake)
    old=app.run_button
    old.click()
    assert len(calls)==1 and app.completed is not None, app.status.value
    assert app._position_editor.canvas is canvas
    saved=app.completed
    view=(app.result_output.outputs,app.completed_summary.value,app.executed_parameters.value)
    assert 'View executed parameters' in view[2]
    digest=hashlib.sha256(saved.request_json.encode()).hexdigest()
    assert digest in view[1] and digest in view[2]
    ui.export_completed(saved,tmp_path/'before')
    old.click(); app._run(old)
    assert len(calls)==1
    drag(app._position_editor,(-20,0),(-10,2))
    assert app.controls['x'].value==-10
    assert app.completed is saved
    assert view==(app.result_output.outputs,app.completed_summary.value,app.executed_parameters.value)
    ui.export_completed(app.completed,tmp_path/'after')
    for name in ('arrays.npz','summary.json','experiment.lcprop.json','visualization.json','fields.png','sha256.json'):
        assert (tmp_path/'before'/name).read_bytes()==(tmp_path/'after'/name).read_bytes()
    def next_run(request, *a, **k):
        calls.append(request)
        return record(request)
    monkeypatch.setattr(ui,'execute',next_run)
    app.run_button.click()
    assert len(calls)==2 and calls[-1].beams.channels[0].x0_um==-10
    assert app.completed.request.beams.channels[0].x0_um==-10
    assert saved.request.beams.channels[0].x0_um==-20


@pytest.mark.parametrize('failure',['execute','plot','png','mismatch'])
def test_preparation_failure_keeps_entire_record(app,monkeypatch,failure):
    saved=record(app.request());app.completed=saved
    app.completed_summary.value='previous';app.executed_parameters.value='previous parameters'
    app.result_output.outputs=({'output_type':'display_data','data':{'text/plain':'previous image'},'metadata':{}},)
    before=(app.completed,app.completed_summary.value,app.executed_parameters.value,app.result_output.outputs)
    def fake(request,*a,**k):
        if failure=='execute':raise TimeoutError('synthetic')
        if failure=='mismatch':return record(ui.request_for_controls(x=1))
        return record(request)
    monkeypatch.setattr(ui,'execute',fake)
    if failure=='plot':
        monkeypatch.setattr(ui,'plot_completed',lambda *a: (_ for _ in ()).throw(ValueError('plot')))
    if failure=='png':
        from matplotlib.figure import Figure
        monkeypatch.setattr(Figure,'savefig',lambda *a,**k: (_ for _ in ()).throw(ValueError('png')))
    old=app.run_button
    old.click()
    assert before==(app.completed,app.completed_summary.value,app.executed_parameters.value,app.result_output.outputs)
    assert 'Previous completed result retained' in app.status.value
    assert app.run_button is not old and not app.busy


def test_load_reset_delayed_draft_and_disposal(app,tmp_path,monkeypatch):
    from lcprop.persistence import save_experiment
    e=app._position_editor
    saved=record(app.request());app.completed=saved
    request=ui.request_for_controls(x=-17)
    path=tmp_path/'request.json'
    save_experiment(request,path,material_id='lc',workflow_id='static')
    e.press(event(e,-20,0));app.load(path)
    e.release(event(e,-10,2))
    assert app.request()==request and app._position_editor is e
    drag(e,(-17,0),(-9,2))
    assert app.controls['x'].value==-9 and app.completed is saved
    app._restore()
    assert app.request()==ui.build_request() and app._position_editor is e
    old=app.run_button
    app.close()
    old.click(); app._run(); app.load('/not-read'); app._restore()
    drag(e,(-20,0),(-8,3))
    assert app.request()==ui.build_request() and app.completed is saved
    assert not e.active


def test_invalid_draft_keeps_canvas_and_cannot_run(app,monkeypatch):
    e=app._position_editor
    app.controls['w1'].value=-1
    assert app._position_editor is e and app.preview_host.children==(e.canvas,)
    assert 'stale' in app.editor_status.value
    app.run_button.click()
    assert app.completed is None
    app.controls['w1'].value=10
    assert app._position_editor is e and 'validated' in app.status.value


def test_execute_detaches_request_and_serializes_result_metadata(monkeypatch):
    import numpy as np
    original=ui.build_request()
    arrays=snapshot('single').arrays()
    captured=[]
    def run(self, operation, request, *, progress_callback):
        assert request is not original and request == original
        captured.append(request)
        progress_callback(SimpleNamespace(completed_units=100,total_units=100))
        result=SimpleNamespace(request=request,status='completed',all_slices_converged=True,
            A_initial=np.ones((3,4),complex),A_final=np.ones((3,4),complex)*2,
            theta_final=arrays['theta'],theta_bias=arrays['theta_bias'],
            intensity_stack=arrays['intensity_stack'])
        return SimpleNamespace(result=result,run_data=SimpleNamespace(geometry=SimpleNamespace(
            x=arrays['x_um'],y=arrays['y_um'],z=arrays['z_um'])))
    monkeypatch.setattr(ui.LocalRunner,'run_operation',run)
    monkeypatch.setattr(ui,'widths',lambda c:arrays['widths_um'])
    monkeypatch.setattr(ui,'diagnostics',lambda c,t,w:{'iteration_history':[{'residual':.001}], 'warnings':w})
    monkeypatch.setattr(ui,'environment',lambda:{'test':'synthetic runner'})
    snap=ui.execute(original)
    assert snap.request==captured[0]
    assert snap.summary['request_sha256']==hashlib.sha256(snap.request_json.encode()).hexdigest()
    assert snap.summary['wall_time_budget_seconds']==120
    assert snap.summary['iteration_history']==[{'residual':.001}]
    old=snap.arrays_npz
    arrays['theta'][:]=999
    decoded=snap.arrays();decoded['theta'][:]=-999
    assert snap.arrays_npz==old and snap.summary['environment']=={'test':'synthetic runner'}


def test_publication_trait_failure_rolls_back(app,monkeypatch):
    saved=record(app.request());app.completed=saved
    app.completed_summary.value='old summary'
    app.executed_parameters.value='old parameters'
    before=(app.completed,app.completed_summary.value,app.executed_parameters.value,app.result_output.outputs)
    fired=[]
    def reject_once(change):
        if not fired:
            fired.append(True)
            raise RuntimeError('synthetic publication failure')
    app.executed_parameters.observe(reject_once,names='value')
    monkeypatch.setattr(ui,'execute',lambda request,*a,**k:record(request))
    app.run_button.click()
    assert fired
    assert before==(app.completed,app.completed_summary.value,app.executed_parameters.value,app.result_output.outputs)
    assert 'Previous completed result retained' in app.status.value

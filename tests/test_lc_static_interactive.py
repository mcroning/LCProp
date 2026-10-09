"""Bounded presentation/request tests; one shared canonical comparison pair."""
from dataclasses import replace, FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import lc_static_interactive as ui
from lcprop.lc.workflows.static import run_static
from lcprop.persistence import load_experiment, save_experiment
from lcprop.optics.launch import build_launch, OpticalLaunchContext
from lcprop.core.grid import make_grid


@pytest.fixture(scope='module')
def completed():
    progress=[]
    snapshot=ui.execute(ui.build_request(),progress.append)
    # Independent canonical workflow call in the same installed environment.
    result=run_static(ui.build_request())
    return snapshot,result,progress


def test_request_and_launch_exact():
    r=ui.request_for_controls()
    assert r==ui.build_request()
    g=make_grid(r.grid,real_dtype=np.float64)
    canonical=build_launch(r.beams,g,complex_dtype=np.complex128,
        context=OpticalLaunchContext(grid=g,n_ref=r.material.no,interaction_length_um=r.grid.z_length_um))
    _,launch,intensity=ui.launch_preview(r)
    np.testing.assert_array_equal(launch.A0,canonical.A0)
    np.testing.assert_array_equal(intensity,ui.total_intensity(canonical.A0,coherence_groups=canonical.coherence_groups))
    assert ui.request_for_controls(x=-18.,w1=11.,power=.8).beams.channels[0].power_mW==.8


def test_scientific_parity_and_progress(completed):
    snapshot,result,progress=completed
    a=snapshot.arrays()
    for key,attr in [('A_initial','A_initial'),('A_final','A_final'),('theta','theta_final'),
                     ('theta_bias','theta_bias'),('intensity_stack','intensity_stack')]:
        np.testing.assert_array_equal(a[key],getattr(result,attr))
    from dataclasses import asdict
    assert snapshot.summary['iteration_history']==[asdict(x) for x in result.iteration_records]
    assert snapshot.summary['slice_summaries']==[asdict(x) for x in result.slice_summaries]
    assert [p.completed_units for p in progress]==list(range(1,101))
    assert result.all_slices_converged and result.completed_slices==100
    assert snapshot.summary['residual_rms_max']<=.005 and snapshot.summary['residual_max']<=.02


def test_snapshot_immutable_and_export(completed,tmp_path):
    snap,_,_=completed
    before=hashlib.sha256(snap.arrays_npz).hexdigest()
    with pytest.raises(FrozenInstanceError): snap.request_json='{}'
    a=snap.arrays();a['theta'][:]=0
    metadata=snap.summary;metadata['status']='corrupt'
    assert hashlib.sha256(snap.arrays_npz).hexdigest()==before
    assert snap.summary['status']=='completed'
    ui.export_completed(snap,tmp_path/'one')
    ui.export_completed(snap,tmp_path/'two')
    assert load_experiment(tmp_path/'one/experiment.lcprop.json').request==snap.request
    assert (tmp_path/'one/arrays.npz').read_bytes()==snap.arrays_npz
    assert (tmp_path/'one/fields.png').read_bytes()==(tmp_path/'two/fields.png').read_bytes()
    for n,h in json.loads((tmp_path/'one/sha256.json').read_text()).items():
        assert hashlib.sha256((tmp_path/'one'/n).read_bytes()).hexdigest()==h
    with pytest.raises(FileExistsError): ui.export_completed(snap,tmp_path/'one')


def test_ui_stale_order_failure_and_repeated_run(completed,tmp_path,monkeypatch):
    snap,_,_=completed
    calls=[]
    def success(request,progress,*,time_limit=120.):
        calls.append(request)
        return snap
    monkeypatch.setattr(ui,'execute',success)
    form=ui.InteractiveLC()
    assert calls==[] and form.completed is None
    form.run_button.click()  # Real widget event path, only scientific call substituted here.
    assert calls==[ui.build_request()] and form.completed is snap
    form.controls['x'].value=-18.
    assert len(calls)==1 and form.completed is snap
    form.path.value=str(tmp_path/'export');form.save_button.click()
    assert load_experiment(tmp_path/'export/experiment.lcprop.json').request==snap.request
    form.run_button.click()
    assert len(calls)==2 and calls[-1].beams.channels[0].x0_um==-18.
    def failure(*args,**kwargs): raise ValueError('diagnostic rejection')
    monkeypatch.setattr(ui,'execute',failure)
    form.run_button.click()
    assert form.completed is snap and 'failed' in form.status.value
    form.controls['power'].value=-1
    assert 'Invalid draft' in form.status.value and form.completed is snap
    assert not form.busy and not form.run_button.disabled


def test_no_save_before_run_and_load(tmp_path):
    form=ui.InteractiveLC()
    form.path.value=str(tmp_path/'missing');form.save_button.click()
    assert not (tmp_path/'missing').exists() and 'rejected' in form.status.value
    r=ui.request_for_controls(x=-17.1234567890123,w2=9.876543210987)
    save_experiment(r,tmp_path/'request.lcprop.json',material_id='lc',workflow_id='static')
    form.load(tmp_path/'request.lcprop.json')
    assert form.request()==r and form.completed is None
    with pytest.raises(ValueError):ui.require_bounded_request(replace(r,grid=replace(r.grid,Nx=256)))
    with pytest.raises(ValueError):ui.launch_preview(ui.request_for_controls(power=0))


def test_headless_imports():
    assert not any(k.split('.')[0] in ('PySide6','launchplane','cupy') for k in sys.modules)


def test_notebook_missing_dependency_is_clear(monkeypatch):
    notebook=json.loads((ROOT/'notebooks/lc_static_interactive_cpu.ipynb').read_text())
    cell=next(c for c in notebook['cells'] if c['cell_type']=='code')
    real=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name:None if name=='ipywidgets' else real(name))
    with pytest.raises(RuntimeError,match='Missing dependencies: ipywidgets'):
        exec(''.join(cell['source']),{})


def test_unsupported_desktop_request_is_not_silently_changed(tmp_path):
    form=ui.InteractiveLC()
    previous=form.request()
    invalid=replace(ui.build_request(),grid=replace(ui.build_request().grid,Nx=256))
    p=tmp_path/'large.lcprop.json'
    save_experiment(invalid,p,material_id='lc',workflow_id='static')
    before=p.read_bytes()
    with pytest.raises(ValueError,match='qualified grid'):
        form.load(p)
    assert form.request()==previous and p.read_bytes()==before


def test_voltage_fresh_bias_initialization(monkeypatch):
    import lcprop.lc.workflows.static as workflow
    original=workflow.build_bias
    seen=[]
    class StopBeforePropagation(Exception): pass
    def inspect(bias,grid,material):
        result=original(bias,grid,material)
        seen.append((bias.V_bias,result.theta_2d.copy()))
        raise StopBeforePropagation
    monkeypatch.setattr(workflow,'build_bias',inspect)
    for voltage in (.9144,1.0):
        request=ui.request_for_controls(V_bias=voltage)
        assert request.initial_theta is None and request.initial_A is None
        assert request.bias.V_bias==voltage
        with pytest.raises(StopBeforePropagation):ui.execute(request)
    assert [x[0] for x in seen]==[.9144,1.0]
    assert not np.array_equal(seen[0][1],seen[1][1])
    with pytest.raises(ValueError):ui.request_for_controls(V_bias=-1)
    with pytest.raises(ValueError):ui.request_for_controls(V_bias=float('nan'))


def test_offcenter_stored_slices_axes_and_missing_data():
    import io
    request=ui.request_for_controls(x=-18,y=3,V_bias=1.)
    x=np.array([-30.,-20.,-10.,0.]);y=np.array([-8.,-4.,0.,4.,8.]);z=np.array([2.5,7.5,12.5])
    volume=np.arange(60,dtype=float).reshape(3,4,5)
    a=dict(x_um=x,y_um=y,z_um=z,intensity_stack=volume,theta=1000+volume,theta_bias=np.zeros((4,5)),widths_um=np.ones((3,2)))
    b=io.BytesIO();np.savez(b,**a)
    snapshot=ui.CompletedRun(json.dumps(ui.encode_lc_static_request(request)),b.getvalue(),'{}')
    arrays,meta,slices=ui.longitudinal_slices(snapshot)
    assert meta['x_index']==1 and meta['y_index']==3
    assert meta['sampled_x_um']==-20 and meta['sampled_y_um']==4
    for got,expected in zip(slices,(volume[:,:,3],volume[:,1,:],(1000+volume)[:,:,3],(1000+volume)[:,1,:])):
        np.testing.assert_array_equal(got,expected)
    fig=ui.plot_completed(snapshot)
    axes=[ax for ax in fig.axes if ax.get_title()]
    for ax,values in zip(axes[:4],slices):
        np.testing.assert_array_equal(np.asarray(ax.collections[0].get_array()).reshape(values.T.shape),values.T)
        assert '$z$' in ax.get_xlabel()
    assert 'y=4.00' in axes[0].get_title() and 'x=-20.00' in axes[1].get_title()
    import matplotlib.pyplot as plt
    plt.close(fig)
    del arrays['theta']
    with pytest.raises(ValueError,match='theta volume unavailable'):ui.slice_selection(request,arrays)


def test_layout_warnings_restore_and_voltage_roundtrip(tmp_path,monkeypatch):
    monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('Editing must not run science'))
    form=ui.InteractiveLC()
    assert form.control_grid.layout.flex_flow=='row wrap'
    assert all(v.layout.width=='100%' and not v.description for v in form.controls.values())
    assert form.status.layout.width=='100%' and form.widget.layout.max_width=='100%'
    sampling=ui.qualify_launch_sampling(form.request().beams,form.request().grid,form.request().material.no,
                                       boundary=form.request().optical_boundary)
    assert sampling.warnings and 'Launch advisory:' in form.advisories.value
    form.controls['V_bias'].value=1.
    assert 'Exploratory' in form.status.value and form.request().bias.V_bias==1.
    p=tmp_path/'voltage.lcprop.json'
    save_experiment(form.request(),p,material_id='lc',workflow_id='static')
    form.restore_button.click()
    assert form.request()==ui.build_request() and 'Qualified preset.' in form.status.value
    form.load(p);assert form.request().bias.V_bias==1. and form.controls['V_bias'].value==1.


def test_completed_voltage_and_slices_are_snapshot_owned(completed,tmp_path):
    snap,_,_=completed
    form=ui.InteractiveLC();form.completed=snap
    form.controls['x'].value=-17.;form.controls['y'].value=3.;form.controls['V_bias'].value=1.
    ui.export_completed(form.completed,tmp_path/'snapshot')
    meta=json.loads((tmp_path/'snapshot/visualization.json').read_text())
    assert meta['V_bias']==.9144 and meta['requested_x_um']==-20 and meta['requested_y_um']==0
    assert load_experiment(tmp_path/'snapshot/experiment.lcprop.json').request==snap.request
    assert snap.summary['warnings']
    assert snap.summary['visualization']==meta


def test_display_size_caps_preserve_figure(monkeypatch):
    import matplotlib.pyplot as plt
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda obj:None)
    fig,ax=plt.subplots(figsize=(7,5))
    data=np.arange(12).reshape(3,4)
    ax.imshow(data)
    before=fig.get_size_inches().copy()
    preview=ui.display_figure(fig,width_px=375)
    results=ui.display_figure(fig)
    assert preview.layout.width=='375px' and results.layout.width=='900px'
    for widget in (preview,results):
        assert widget.layout.max_width=='100%' and widget.layout.height=='auto'
    assert preview.value==results.value  # CSS sizing alone; identical rendered pixels.
    np.testing.assert_array_equal(fig.get_size_inches(),before)
    np.testing.assert_array_equal(ax.images[0].get_array(),data)
    plt.close(fig)

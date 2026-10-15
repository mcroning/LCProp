"""Bounded native two-beam controller qualification; no alternative physics."""
from dataclasses import replace, asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lcprop.notebook import interactive_lc as ui
from lcprop.core.beams import BeamChannel, BeamStack, LEGACY_COHERENCE_GROUP
from lcprop.core.grid import make_grid
from lcprop.optics.launch import OpticalLaunchContext, build_launch, normalized_power
from lcprop.optics.splitstep import total_intensity
from lcprop.persistence import save_experiment, load_experiment


def two_request(*, coherence='incoherent', phase=0.):
    first = ui.build_request().beams.channels[0]
    second = replace(first, name='second', power_mW=.5, phase_rad=phase, coherence_group='B')
    return ui.request_for_controls(power=.5, second_beam=second, coherence=coherence)


def test_default_and_canonical_two_request():
    preset = ui.build_request()
    assert ui.request_for_controls() == preset
    second = replace(preset.beams.channels[0], name='second', power_mW=.5, coherence_group='B')
    expected = replace(preset, beams=BeamStack(channels=(replace(preset.beams.channels[0], power_mW=.5),second)))
    assert two_request() == expected
    assert ui.encode_lc_static_request(two_request()) == ui.encode_lc_static_request(expected)
    assert ui.request_for_controls(base=expected, power=.5) == expected
    assert len(ui.request_for_controls(base=expected, second_beam=None).beams.channels) == 1


@pytest.mark.parametrize('coherence', ['coherent','incoherent'])
def test_geometry_and_launch_parity(coherence):
    r = two_request(coherence=coherence, phase=.7)
    first,second = r.beams.channels
    r = replace(r, beams=replace(r.beams, channels=(
        replace(first, x0_um=-5., y0_um=3., theta_ext_rad=.01, phi_rad=.3, psi_rad=.5, w1_um=9.,w2_um=12.),
        replace(second, x0_um=5.,y0_um=-2.,theta_ext_rad=-.01,phi_rad=.7,psi_rad=.8,w1_um=11.,w2_um=8.))))
    grid = make_grid(r.grid, real_dtype=np.float64)
    expected = build_launch(r.beams, grid, complex_dtype=np.complex128,
        context=OpticalLaunchContext(grid=grid,n_ref=r.material.no,interaction_length_um=r.grid.z_length_um))
    _,launch,intensity = ui.launch_preview(r)
    np.testing.assert_array_equal(launch.A0, expected.A0)
    np.testing.assert_array_equal(intensity,total_intensity(expected.A0,coherence_groups=expected.coherence_groups))
    assert launch.coherence_groups == r.beams.coherence_groups
    assert normalized_power(launch.A0,grid) == normalized_power(expected.A0,grid)


def test_phase_interference_without_renormalization():
    _,constructive,I0 = ui.launch_preview(two_request(coherence='coherent'))
    _,destructive,Ipi = ui.launch_preview(two_request(coherence='coherent',phase=np.pi))
    _,_,incoherent = ui.launch_preview(two_request())
    _,_,incoherent_pi = ui.launch_preview(two_request(phase=np.pi))
    np.testing.assert_allclose(I0,2*incoherent,rtol=2e-15,atol=1e-18)
    assert np.max(Ipi) < np.max(I0)*1e-28
    np.testing.assert_allclose(incoherent,incoherent_pi,rtol=2e-15,atol=1e-18)
    np.testing.assert_array_equal(abs(constructive.A0),abs(destructive.A0))
    # Stack word is deliberately contradictory: explicit groups still win.
    r=two_request(coherence='coherent')
    _,_,override=ui.launch_preview(replace(r,beams=replace(r.beams,coherence='incoherent')))
    np.testing.assert_array_equal(override,I0)


@pytest.mark.parametrize('mutation,match', [
    (lambda r:replace(r,beams=replace(r.beams,channels=r.beams.channels+(r.beams.channels[0],))), 'one or two'),
    (lambda r:replace(r,beams=replace(r.beams,channels=(r.beams.channels[0],replace(r.beams.channels[1],wavelength_um=.7)))), 'wavelength'),
    (lambda r:replace(r,beams=replace(r.beams,channels=(r.beams.channels[0],replace(r.beams.channels[1],coherence_group=LEGACY_COHERENCE_GROUP)))), 'migration'),
    (lambda r:replace(r,beams=replace(r.beams,channels=tuple(replace(b,power_mW=0) for b in r.beams.channels))), 'power'),
    (lambda r:replace(r,beams=replace(r.beams,channels=(replace(r.beams.channels[0],theta_ext_rad=1.),r.beams.channels[1]))), 'sampling|Nyquist|nyquist'),
])
def test_validation_is_canonical(mutation,match):
    with pytest.raises(ValueError,match=match):ui.launch_preview(mutation(two_request()))


def test_zero_power_is_not_removal():
    r=two_request()
    r=replace(r,beams=replace(r.beams,channels=(r.beams.channels[0],replace(r.beams.channels[1],power_mW=0))))
    _,launch,_=ui.launch_preview(r)
    assert launch.A0.shape[0]==2 and np.count_nonzero(launch.A0[1])==0


@pytest.fixture
def quiet_form(monkeypatch):
    # No render work is needed to test control-to-request mapping.
    monkeypatch.setattr(ui.InteractiveLC,'_changed',lambda *a,**k:None)
    monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('Edits must not execute'))
    return ui.InteractiveLC()


def test_controls_geometry_relative_phase_and_restore(quiet_form):
    form=quiet_form
    assert form.request()==ui.build_request()
    form.second_enabled.value=True
    form.beam2_controls['x0_um'].value=7
    form.beam2_controls['y0_um'].value=-3
    for name,value in dict(theta_ext_rad=.01,phi_rad=.2,psi_rad=.4,wavelength_um=.633).items():
        form.beam2_controls[name].value=value
    form.beam1_controls['phase_rad'].value=.25
    form.beam2_controls['phase_rad'].value=.75
    form.coherence.value='coherent'
    r=form.request();b=r.beams.channels[1]
    assert b.phase_rad==1. and b.x0_um==7 and b.y0_um==-3
    assert b.theta_ext_rad==.01 and b.psi_rad==.4
    assert r.beams.coherence_groups[0]==r.beams.coherence_groups[1]
    form.coherence.value='incoherent'
    assert len(set(form.request().beams.coherence_groups))==2
    form.restore_button.click()
    assert form.request()==ui.build_request() and not form.second_enabled.value


@pytest.mark.parametrize('legacy', [False,True])
def test_exact_native_roundtrip_and_relative_phase_precision(quiet_form,tmp_path,legacy):
    r=two_request(coherence='coherent',phase=.123456789012345)
    first,second=r.beams.channels
    r=replace(r,beams=replace(r.beams,channels=(replace(first,phase_rad=1e4,
        coherence_group=LEGACY_COHERENCE_GROUP if legacy else 'custom-laser'),
        replace(second,coherence_group=LEGACY_COHERENCE_GROUP if legacy else 'custom-laser'))))
    path=tmp_path/'experiment.lcprop.json'
    save_experiment(r,path,material_id='lc',workflow_id='static')
    before=path.read_bytes()
    quiet_form.load(path)
    assert quiet_form.request()==r==load_experiment(path).request
    assert path.read_bytes()==before
    assert quiet_form.request().beams.channels[1].phase_rad==second.phase_rad
    quiet_form.beam2_controls['phase_rad'].value=.5
    assert quiet_form.request().beams.channels[1].phase_rad==10000.5


def synthetic_snapshot(request):
    a=dict(x_um=np.array([-20.,0.,20.]), y_um=np.array([-1.,0.,1.]), z_um=np.array([2.5,7.5]),
        intensity_stack=np.ones((2,3,3)),theta=np.ones((2,3,3)),theta_bias=np.zeros((3,3)),widths_um=np.ones((2,2)))
    b=io.BytesIO();np.savez(b,**a)
    return ui.CompletedRun(json.dumps(ui.encode_lc_static_request(request)),b.getvalue(),'{}')


def test_snapshot_ownership_and_export_after_multibeam_edits(quiet_form,monkeypatch,tmp_path):
    import matplotlib.pyplot as plt
    form=quiet_form;snapshot=synthetic_snapshot(two_request())
    form.completed=snapshot
    form.result_output.outputs=({'output_type':'display_data','data':{'text/plain':'prior plot'},'metadata':{}},)
    previous=form.result_output.outputs
    form.second_enabled.value=True
    form.beam2_controls['power_mW'].value=0
    form.controls['x'].value=4
    def timeout(*args,**kwargs): raise TimeoutError('Time budget exhausted; convergence not assessed.')
    monkeypatch.setattr(ui,'execute',timeout)
    form.run_button.click()
    assert form.completed is snapshot and form.result_output.outputs==previous
    assert all(not c.disabled for c in form._draft_controls())
    ui.export_completed(form.completed,tmp_path/'saved')
    assert load_experiment(tmp_path/'saved/experiment.lcprop.json').request==snapshot.request
    assert (tmp_path/'saved/arrays.npz').read_bytes()==snapshot.arrays_npz
    fig=ui.plot_completed(snapshot)
    assert 'Combined RMS width' in fig.axes[-1].get_ylabel()
    assert 'Beam 1' in fig._suptitle.get_text()
    plt.close(fig)
    assert ui.slice_selection(snapshot.request,snapshot.arrays())['requested_x_um']==-20.
    form.restore_button.click();assert form.completed is snapshot


def test_editor_metadata_validation_not_bypassed(quiet_form,monkeypatch,tmp_path):
    from lcprop.persistence.experiments import ExperimentPresentationError
    before=quiet_form.request()
    snapshot=synthetic_snapshot(before);quiet_form.completed=snapshot
    def rejected(path):raise ExperimentPresentationError('invalid LaunchPlane presentation metadata: dependency unavailable')
    monkeypatch.setattr(ui,'load_experiment',rejected)
    quiet_form.load_path.value=str(tmp_path/'desktop.lcprop.json')
    quiet_form.load_button.click()
    assert 'Qt editor metadata' in quiet_form.status.value
    assert quiet_form.request()==before and quiet_form.completed is snapshot


@pytest.mark.parametrize('count', [1,2])
def test_bounded_execution_exact_canonical_arrays(count,tmp_path):
    from lcprop.lc.workflows.static import run_static
    from lcprop.products.diagnostics import rms_widths
    request=ui.build_request() if count==1 else two_request()
    progress=[]
    snapshot=ui.execute(request,progress.append,time_limit=120.)
    result=run_static(request)
    grid=make_grid(request.grid,real_dtype=np.float64)
    expected=dict(A_initial=result.A_initial,A_final=result.A_final,theta=result.theta_final,
        theta_bias=result.theta_bias,intensity_stack=result.intensity_stack,
        widths_um=np.asarray([rms_widths(p,grid) for p in result.intensity_stack]),
        x_um=grid.x_um,y_um=grid.y_um,z_um=(np.arange(100)+.5)*request.grid.dz_um)
    arrays=snapshot.arrays()
    for name,value in expected.items():
        np.testing.assert_array_equal(arrays[name],value,err_msg=name)
        assert arrays[name].dtype==value.dtype
        assert arrays[name].tobytes(order='C')==value.tobytes(order='C'), name
    assert snapshot.summary['slice_summaries']==[asdict(s) for s in result.slice_summaries]
    assert snapshot.summary['iteration_history']==[asdict(s) for s in result.iteration_records]
    assert result.all_slices_converged and result.completed_slices==100
    assert [p.completed_units for p in progress]==list(range(1,101))
    metadata={'beams':count,'request':ui.encode_lc_static_request(request),'summary':snapshot.summary,
        'arrays':{k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes(order='C')).hexdigest()) for k,v in arrays.items()}}
    (tmp_path/'parity.json').write_text(json.dumps(metadata,indent=2))


def test_optional_import_isolation_subprocess():
    code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,name,*args):
        if name.split('.')[0] in {'PySide6','launchplane','cupy','ipywidgets','matplotlib','IPython'}:
            raise AssertionError(name)
sys.meta_path.insert(0,Block())
from lcprop.notebook import InteractiveLC
from lcprop.notebook.interactive_lc import request_for_controls
assert len(request_for_controls().beams.channels)==1
'''
    subprocess.run([sys.executable,'-B','-c',code],check=True)


def test_live_widget_events_only_run_on_button(monkeypatch,tmp_path):
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda *a,**k:None)
    calls=[]
    def completed(request,progress,*,time_limit):
        calls.append(request)
        return synthetic_snapshot(request)
    monkeypatch.setattr(ui,'execute',completed)
    form=ui.InteractiveLC()
    assert form.beam2_form.layout.display=='none'
    form.second_enabled.value=True
    assert form.beam2_form.layout.display=='flex'
    form.coherence.value='coherent'
    form.beam2_controls['phase_rad'].value=.4
    assert not calls and form.completed is None
    assert 'Effective coherence groups:' in form.advisories.value
    assert 'Exploratory' in form.status.value
    form.run_button.click()
    assert len(calls)==1 and len(calls[0].beams.channels)==2
    snapshot=form.completed
    form.beam1_controls['wavelength_um'].value=.7
    assert 'Invalid draft' in form.status.value  # Never silently change Beam 2.
    assert form.beam2_controls['wavelength_um'].value==.633
    form.beam2_controls['wavelength_um'].value=.7
    assert 'Draft launch validated' in form.status.value
    assert form.completed is snapshot and len(calls)==1
    form.path.value=str(tmp_path/'saved');form.save_button.click()
    assert load_experiment(tmp_path/'saved/experiment.lcprop.json').request==calls[0]
    form.second_enabled.value=False
    assert len(form.request().beams.channels)==1 and form.completed is snapshot


def test_actual_loader_rejects_editor_payload_without_dependency(quiet_form,tmp_path,monkeypatch):
    import builtins
    request=two_request()
    path=tmp_path/'desktop.lcprop.json'
    save_experiment(request,path,material_id='lc',workflow_id='static')
    doc=json.loads(path.read_text())
    doc['presentation_payload']={'beam_editor':{'provider':'launchplane','schema_version':1,'beam_stack':{}}}
    path.write_text(json.dumps(doc))
    before=path.read_bytes();prior=quiet_form.request()
    real=builtins.__import__
    def no_launchplane(name,*args,**kwargs):
        if name.startswith('launchplane'):raise ImportError('LaunchPlane unavailable in this headless environment')
        return real(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_launchplane)
    from lcprop.persistence.experiments import ExperimentPresentationError
    with pytest.raises(ExperimentPresentationError,match='LaunchPlane unavailable'):
        quiet_form.load(path)
    assert path.read_bytes()==before and quiet_form.request()==prior


@pytest.mark.parametrize('second_group', ['A','B'])
def test_desktop_adapter_request_identity(second_group):
    model=pytest.importorskip('launchplane.model')
    from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
    definition=model.BeamStackDefinition(beams=(
        model.BeamDefinition(name='off-axis',x_um=-4.,y_um=2.,w1_um=9.,w2_um=12.,power_mW=.6,
                             coherence_group='A',theta_ext_rad=.01,phi_rad=.2,psi_rad=.3,phase_rad=.4),
        model.BeamDefinition(name='second',x_um=5.,y_um=-3.,w1_um=11.,w2_um=8.,power_mW=.4,
                             coherence_group=second_group,theta_ext_rad=-.01,phi_rad=.5,psi_rad=.6,phase_rad=.7)))
    canonical=replace(ui.build_request(),beams=beam_stack_definition_to_lcprop(definition))
    actual=ui.request_for_controls(x=-4.,y=2.,w1=9.,w2=12.,power=.6,
        beam1_options=dict(theta_ext_rad=.01,phi_rad=.2,psi_rad=.3,phase_rad=.4),
        second_beam=BeamChannel(name='second',x0_um=5.,y0_um=-3.,w1_um=11.,w2_um=8.,power_mW=.4,
            coherence_group=second_group,theta_ext_rad=-.01,phi_rad=.5,psi_rad=.6,phase_rad=.7))
    assert ui.encode_lc_static_request(actual)==ui.encode_lc_static_request(canonical)
    assert 'PySide6' not in sys.modules

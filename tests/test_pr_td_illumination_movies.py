from dataclasses import replace
import numpy as np
import pytest
from tests.test_pr_reduced_physical_continuation import request
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.pr.reduced_continuation import describe_continuation,continue_physical_pr
from lcprop.pr.persistence import save_pr_checkpoint,load_pr_checkpoint
from lcprop.pr.trajectory import pack_frames,unpack_frames,movie_artifacts


@pytest.mark.parametrize('change',['same','power','off','position','angle','uniform'])
def test_reduced_interventions(change,tmp_path):
    r=request();first=run_pr_timedependent(replace(r,solver=replace(r.solver,Nt=2)))
    new=replace(r,solver=replace(r.solver,Nt=2))
    c=r.beams.channels[0]
    if change in ('power','off','position','angle'):
        kw={'power':{'power_mW':.25},'off':{'power_mW':0.},'position':{'x0_um':1.},'angle':{'theta_ext_rad':.0001}}[change]
        new=replace(new,beams=replace(r.beams,channels=(replace(c,**kw),)))
    if change=='uniform':new=replace(new,material=replace(r.material,uniform_irradiance_W_cm2=.7))
    descriptor=describe_continuation(new,first.checkpoint)
    boundary=continue_physical_pr(new,first.checkpoint,0)
    np.testing.assert_array_equal(boundary.E_final,first.E_final)
    result=continue_physical_pr(new,first.checkpoint)
    assert result.completed_steps==4 and result.time_normalized==.04
    assert result.diagnostics['source_normalization']==descriptor['new_reference']
    if change=='same':
        full=run_pr_timedependent(r)
        np.testing.assert_array_equal(result.E_final,full.E_final)
        np.testing.assert_array_equal(result.A_final,full.A_final)
    if change=='off':assert not np.any(result.A_final)
    save_pr_checkpoint(result.checkpoint,tmp_path);cp=load_pr_checkpoint(tmp_path)
    assert cp.segment_lineage==result.checkpoint.segment_lineage
    a=continue_physical_pr(new,cp,1);b=continue_physical_pr(new,result.checkpoint,1)
    np.testing.assert_array_equal(a.E_final,b.E_final)


def test_locked_cadence_and_legacy_and_prepared_array():
    r=request();cp=run_pr_timedependent(r).checkpoint
    for bad in (replace(r,solver=replace(r.solver,dt_normalized=.02)),replace(r,initial_A=cp.A0)):
        with pytest.raises(ValueError):describe_continuation(bad,cp)


@pytest.mark.parametrize('transverse',[False,True])
def test_movie_observational_codec(transverse):
    if transverse:
        from tests.test_pr_transverse_continuation import request as req
        from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent as run
        from lcprop.pr.transverse.timedependent_transport_codec import encode_pr_transverse_timedependent_transport_result as encode,decode_pr_transverse_timedependent_transport_result as decode
        r=req();state='psi_final'
    else:
        from lcprop.pr.timedependent_transport_codec import encode_pr_timedependent_transport_result as encode,decode_pr_timedependent_transport_result as decode
        r=request();run=run_pr_timedependent;state='E_final'
    off=run(r);on=run(r,progress_callback=lambda _:None)
    np.testing.assert_array_equal(getattr(off,state),getattr(on,state));np.testing.assert_array_equal(off.A_final,on.A_final)
    np.testing.assert_array_equal(off.source_intensity_stack,on.source_intensity_stack)
    original=movie_artifacts(on)['td_trajectory']
    material_movie=movie_artifacts(on)['td_trajectory_material']
    from lcprop.pr.visualization import downsample_td_movie_frame
    np.testing.assert_array_equal(material_movie.data[-1],downsample_td_movie_frame(getattr(on,state)[getattr(on,state).shape[0]//2]))
    assert np.all(np.diff(original.metadata['times'])>0)
    for policy in ('fast','full'):
        encoded=encode(on,result_policy=policy);decoded=decode(encoded.payload.metadata,encoded.payload.arrays)
        saved=movie_artifacts(decoded)['td_trajectory']
        np.testing.assert_array_equal(saved.data,original.data)
        assert saved.metadata==original.metadata
        np.testing.assert_array_equal(movie_artifacts(decoded)['td_trajectory_material'].data,material_movie.data)
    assert not movie_artifacts(off)


def test_bounded_payload_rejects_nonfinite_or_oversize():
    for shape in ((36,128,128),(36,128,64)):
        frames=np.ones(shape,np.float32);np.testing.assert_array_equal(unpack_frames(pack_frames(frames)),frames)
        assert frames.nbytes<=2359296
    with pytest.raises(ValueError):pack_frames(np.zeros((37,128,128)))
    with pytest.raises(ValueError):pack_frames(np.full((1,2,2),np.nan))


def test_movie_ui_irregular_scale_and_controls():
    from PySide6.QtWidgets import QApplication
    from lcprop.gui.workspace import Workspace
    from lcprop.products.data_model import RunData,Geometry,ArtifactData
    from lcprop.gui.views.display_scale import scale_key
    app=QApplication.instance() or QApplication([]);w=Workspace();p=w.trajectory_player
    a=ArtifactData('td_trajectory','optical',np.array([[[0.,1.],[2.,3.]],[[0.,10.],[20.,30.]],[[0.,2.],[4.,6.]]]),
        'application/x-lcprop-trajectory','x',{'times':[.1,.13,.8],'segment_start':.1,'value_unit':'1/um^2',
        'coordinates':{'x':[-1.,1.],'y':[-1.,1.]},'availability':'Optical xy only.'})
    data=RunData(workflow='pr_timedependent',geometry=Geometry(),artifacts={'td_trajectory':a})
    p.set_run_data(data);p.show_frame(0);p.scale.setCurrentIndex(1)
    key=scale_key(w.image_pane.image_view._field) if hasattr(w.image_pane,'image_view') else next(iter(w.display_scales.settings))
    assert w.display_scales.settings[key][1]==(0.,30.)
    p.scale.setCurrentIndex(0);assert w.display_scales.settings[key][0]=='auto'
    w.display_scales.configure(key,'fixed',(1.,9.));p.scale.setCurrentIndex(1)
    assert w.display_scales.settings[key]==('fixed',(1.,9.))
    p.step(1);assert 'τ=0.13' in p.label.text();p.toggle();assert p.timer.isActive();p.toggle();assert not p.timer.isActive()
    p.slider.setValue(2);p.step(1);assert p.slider.value()==2 and not p.timer.isActive()
    p.loop.setChecked(True);p.step(1);assert p.slider.value()==0
    p.restore_endpoint();p.set_run_data(replace(data,artifacts={}));assert not p.play.isEnabled() and 'unavailable' in p.label.text()
    w.close()


def test_reduced_editor_and_dark_products(tmp_path):
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.transverse_continuation import TransverseContinuationDialog
    from lcprop.pr.products import pr_result_to_run_data
    from lcprop.pr.timedependent_transport_codec import encode_pr_timedependent_transport_result as encode,decode_pr_timedependent_transport_result as decode
    app=QApplication.instance() or QApplication([])
    r=request();cp=run_pr_timedependent(r).checkpoint
    dialog=TransverseContinuationDialog(cp,describe=describe_continuation,reduced=True)
    assert not dialog.dt.isEnabled()
    dialog.beams.item(0,0).setText('0');new=dialog.build_request()
    result=continue_physical_pr(new,cp,1,progress_callback=lambda _:None)
    for policy in ('fast','full'):
        payload=encode(result,result_policy=policy)
        restored=decode(payload.payload.metadata,payload.payload.arrays)
        data=pr_result_to_run_data(restored)
        movie=data.artifacts['td_trajectory']
        assert not np.any(movie.data)
        assert movie.metadata['segment_start']==cp.time_normalized
    dialog.close()


def test_transverse_continued_movie_times():
    from tests.test_pr_transverse_continuation import request as req
    from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
    from lcprop.pr.transverse.continuation import checkpoint_from_result,continue_transverse_td
    r=req();first=run_pr_transverse_timedependent(r)
    cp=checkpoint_from_result(r,first)
    result=continue_transverse_td(r,cp,progress_callback=lambda _:None)
    movie=movie_artifacts(result)['td_trajectory']
    assert movie.metadata['segment_start']==cp.time_normalized
    assert min(movie.metadata['times'])>=cp.time_normalized
    assert max(movie.metadata['times'])==cp.time_normalized+result.time_normalized


def test_lineage_rejection_and_dark_drive():
    from lcprop.pr.checkpoint import validate_pr_checkpoint
    from lcprop.pr.evolution import periodic_derivatives_x
    r=request();cp=run_pr_timedependent(r).checkpoint
    dark=replace(r,beams=replace(r.beams,channels=(replace(r.beams.channels[0],power_mW=0.),)))
    result=continue_physical_pr(dark,cp,1)
    np.testing.assert_array_equal(result.source_intensity_stack,np.ones_like(result.source_intensity_stack))
    dx=r.material.characteristic_wavenumber_per_um*r.grid.x_aperture_um/r.grid.Nx
    assert np.min(1+periodic_derivatives_x(result.E_final,dx_normalized=dx,xp=np)[0])>0
    bad=dict(result.checkpoint.segment_lineage[0],identity='unknown')
    with pytest.raises(ValueError,match='unknown reduced'):
        validate_pr_checkpoint(replace(result.checkpoint,segment_lineage=(bad,)))
    zero=replace(dark,material=replace(dark.material,dark_irradiance_W_cm2=0.,uniform_irradiance_W_cm2=0.))
    with pytest.raises(ValueError):describe_continuation(zero,cp)


def test_gui_reduced_continue_uses_editor_request(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.pr.gui.transverse_continuation import TransverseContinuationDialog
    app=QApplication.instance() or QApplication([])
    w=PRMainWindow();w.evolution_panel.set_workflow_id('pr_timedependent')
    r=request();w.last_checkpoint=run_pr_timedependent(r).checkpoint
    def accept(dialog):
        dialog.beams.item(0,0).setText('0.25')
        dialog.request=dialog.build_request()
        return 1
    monkeypatch.setattr(TransverseContinuationDialog,'exec',accept)
    captured=[]
    monkeypatch.setattr(w,'_start_background',lambda r,**kw:captured.append((r,kw)))
    w._refresh_checkpoint_controls();assert w.continue_button.isEnabled()
    w.continue_clicked()
    assert len(captured)==1 and captured[0][0].beams.channels[0].power_mW==.25
    assert 'old_reference' in captured[0][1]['summary']
    w.close()

import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
from time import monotonic
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from lcprop.gui.live_optical_preview import LiveOpticalPreview
from lcprop.pr.transverse.live_preview import OpticalPassPreview

@pytest.fixture
def app():return QApplication.instance() or QApplication([])


def frame(run='r',step=0,magnitude=1.):
    return OpticalPassPreview(run,'segment-a',step+1,step,step,step*.01,
        magnitude*np.arange(12.).reshape(3,4),np.arange(4.),np.arange(1.,4.),-.5,(3,4,6),False)


def test_fixed_limits_manual_and_old_run(app):
    v=LiveOpticalPreview();v.begin('r');f=frame();before=f.intensity_xz.tobytes()
    v.show_frame(f);limits=v.view.image.get_clim()
    v.show_frame(frame(step=1,magnitude=3.));assert v.view.image.get_clim()==limits
    v.scales.configure(v.key,'fixed',(0,100));assert v.view.image.get_clim()==(0,100)
    v.show_frame(frame(run='old',step=100));assert v.frame.observed_step==1
    v.finish('failed');assert 'last complete preview' in v.label.text()
    assert 'cumulative τ=0.01' in v.label.text()
    assert before==f.intensity_xz.tobytes()
    v.begin('new');v.show_frame(frame(run='new',magnitude=0.));assert v.view.image.get_clim()==(0,1)
    v.close();v.deleteLater()
    from PySide6.QtCore import QCoreApplication,QEvent
    QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)


def test_real_gui_worker_completed_owner_and_preview(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    from tests.test_pr_transverse_continuation import request
    from tests.test_synchronized_fields import fixture
    from lcprop.pr.transverse.continuation import run_continuable_transverse_td
    from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
    from lcprop.runners.base import RunnerResult
    from lcprop.pr.transverse.specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    from lcprop.pr.specs import PR_MATERIAL_ID
    w=PRMainWindow();ws=w.results_panel.workspace;old=fixture();ws.set_run_data(old,state='Completed old run')
    old_owner=ws._displayed_request
    seen=[]
    def runner(r,**kwargs):
        assert 'optical_preview' in kwargs
        out=run_continuable_transverse_td(r,**kwargs)
        return RunnerResult(kind=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,result=out,
            run_data=pr_transverse_result_to_run_data(out),message='completed',material_id=PR_MATERIAL_ID)
    w._start_background(request(2),summary='New run',runner_callable=runner,run_label='Running')
    original=w._live_view.set_latest_progress
    def latest(text):
        w._poll_live_preview()
        seen.append((ws._displayed_request,ws._display_state,
                     w._live_view.frame.segment_step,w.last_progress.completed_units))
        original(text)
    w._live_view.set_latest_progress=latest
    end=monotonic()+20
    while w._background_running and monotonic()<end:
        app.processEvents()
        from PySide6.QtTest import QTest
        QTest.qWait(5)
    assert not w._background_running and w.run_status=='completed'
    assert seen and all(owner==old_owner and state=='Completed old run' and step<=units-1
                        for owner,state,step,units in seen)
    assert w._live_view.frame.final_replay and w._live_view.frame.observed_step==2
    assert w._live_mailbox.closed
    assert 'last complete preview' in w._live_view.label.text()
    from functools import partial
    from lcprop.pr.gui.main_window import _continue_transverse_operation
    cp=w.last_checkpoint;old_frame=w._live_view.frame;limits=w._live_view.view.image.get_clim()
    assert w._live_view.checkpoint_identity==cp.identity
    w._start_background(request(2),summary='Continue',
        runner_callable=partial(_continue_transverse_operation,checkpoint=cp),
        run_label='Continuing',execution_runner=w.local_runner,preview_checkpoint=cp)
    assert w._live_view.frame is old_frame
    assert 'awaiting first optical observation from Segment 2' in w._live_view.label.text()
    end=monotonic()+20
    while w._background_running and monotonic()<end:
        app.processEvents();QTest.qWait(5)
    assert not w._background_running and w.run_status=='completed'
    assert w._live_view.frame.segment_number==2
    assert w._live_view.frame.run_id!=old_frame.run_id
    assert w._live_view.view.image.get_clim()==limits
    w.close();w.deleteLater()
    from PySide6.QtCore import QCoreApplication,QEvent
    QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)


@pytest.mark.parametrize('status',['failed','cancelled'])
def test_previous_segment_waiting_replacement_and_independent_clear(app,status):
    v=LiveOpticalPreview();v.begin('r',segment_number=1)
    old=replace(frame(step=4),segment_number=1,total_steps=10)
    v.show_frame(old);v.scales.configure(v.key,'fixed',(0.,42.));v.bind_checkpoint('checkpoint')
    v.finish('completed');v.begin('next',segment_number=2,preserve=True)
    assert v.frame is old and v.view.image.get_clim()==(0.,42.)
    assert 'Previous segment — awaiting first optical observation from Segment 2' in v.label.text()
    assert 'Segment 1, step 4/10' in v.label.text()
    assert 'segment-a' not in v.label.text() and 'segment-a' in v.label.toolTip()
    v.finish(status);assert v.frame is old and 'previous segment retained' in v.label.text()
    v.begin('next2',segment_number=2,preserve=True)
    v.show_frame(replace(old,pass_id=999));assert v.frame is old
    fresh=replace(old,run_id='next2',segment_id='segment-b',segment_number=2,pass_id=1,segment_step=0)
    v.show_frame(fresh);assert v.frame is fresh
    assert fresh.cumulative_time==old.cumulative_time
    assert v.view.image.get_clim()==(0.,42.)
    assert 'Previous segment' not in v.label.text() and 'Segment 2, step 0/10' in v.label.text()
    v.show_frame(replace(old,pass_id=1000));assert v.frame is fresh
    v.begin('independent',segment_number=1)
    assert v.frame is None and v.initial_limits is None and v.checkpoint_identity is None
    v.show_frame(replace(fresh,run_id='independent',segment_number=None))
    assert 'lineage unavailable' in v.label.text()
    v.close();v.deleteLater()

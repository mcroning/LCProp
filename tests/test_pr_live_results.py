"""Accepted-state ownership and bounded Local reduced-PR presentation."""
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.gui.workspace import Workspace
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.live_results import (
    PRLivePreviewPolicy, PRLiveSnapshot, reduced_pr_live_snapshot,
    reduced_pr_live_to_run_data,
)
from lcprop.pr.specs import (
    PR_TIMEDEPENDENT_WORKFLOW, PR_EULER_INTEGRATOR, PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_EXACT_MODAL_INTEGRATOR,
)
import lcprop.pr.workflow as workflow
import lcprop.pr.live_results as live
from lcprop.transport.executor import _write_progress
from tests.test_pr_execution import _request, _assert_same_physical_result
from tests.test_stage1b_transparency import image_data


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def close_widget(widget):
    # Explicitly dispose test-owned Qt/Matplotlib objects before later modules
    # run real worker threads and process deferred GUI events.
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def collect_test_objects_on_gui_thread():
    yield
    # Matplotlib/Qt wrappers can form Python cycles. Collect this module's test
    # objects here, rather than leaving collection to a subsequent worker thread.
    import gc
    gc.collect()


@pytest.fixture
def no_movie_encoder(monkeypatch):
    # Encoding is unrelated to ephemeral previews; preserve frame collection.
    monkeypatch.setattr(workflow, 'encode_td_preview_movie',
                        lambda *a, **k: SimpleNamespace(data=None, metadata={}, warning=None))


def request_for(integrator, steps=3):
    r = _request(steps=steps)
    r = replace(r, solver=replace(r.solver, integrator=integrator),
                initial_E=np.broadcast_to(
                    .01*np.sin(np.arange(8)[None, :, None]), (2, 8, 6)).copy())
    if integrator == PR_EXACT_MODAL_INTEGRATOR:
        r = replace(r, material_response=replace(r.material_response, model='linearized', reference_intensity=1.))
    return r


@pytest.mark.parametrize('integrator', [PR_EULER_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR, PR_EXACT_MODAL_INTEGRATOR])
def test_preview_equivalence_and_identical_optical_call_order(monkeypatch, no_movie_encoder, integrator):
    r = request_for(integrator)
    original = workflow._optical_pass
    calls = []
    def observe(A, E, **kwargs):
        calls.append((kwargs.get('cancellation_stage'), E.copy()))
        return original(A, E, **kwargs)
    monkeypatch.setattr(workflow, '_optical_pass', observe)
    old_progress = []
    old = workflow.run_pr_timedependent(r, progress_callback=old_progress.append)
    old_calls = calls.copy(); calls.clear()
    new_progress = []
    new = workflow.run_pr_timedependent(r, progress_callback=new_progress.append,
                                      live_preview_policy=PRLivePreviewPolicy(interval_seconds=0))
    _assert_same_physical_result(new, old)
    assert new.td_scalar_history == old.td_scalar_history
    assert new.checkpoint.request == old.checkpoint.request
    assert len(calls) == len(old_calls)
    for (stage, E), (old_stage, old_E) in zip(calls, old_calls):
        assert stage == old_stage
        np.testing.assert_array_equal(E, old_E)
    for p, old_p in zip(new_progress, old_progress):
        snap = p.latest_field_state
        assert isinstance(snap, PRLiveSnapshot)
        assert snap.completed_steps == p.completed_units
        assert snap.time_normalized == p.current_coordinate
        state = old_p.latest_field_state
        np.testing.assert_array_equal(snap.material_plane, state['E_current'][1])
        np.testing.assert_array_equal(snap.output_intensity,
                                      abs(state['A_current'][0])**2)
        assert snap.scalar_values == new.td_scalar_history[p.completed_units-1]
        assert not snap.material_plane.flags.writeable
    # Policy has no effect on callback-free scientific execution either.
    _assert_same_physical_result(
        workflow.run_pr_timedependent(r, live_preview_policy=PRLivePreviewPolicy()),
        workflow.run_pr_timedependent(r))


@pytest.mark.parametrize('stage', ['candidate', 'incomplete_replay', 'after_replay'])
def test_cancelled_candidate_never_published(monkeypatch, no_movie_encoder, stage):
    token = CancellationToken(); progress = []
    if stage == 'candidate':
        old = workflow.euler_step
        def step(*a, **k):
            result = old(*a, **k); token.cancel(); return result
        monkeypatch.setattr(workflow, 'euler_step', step)
    else:
        old = workflow._optical_pass
        def optical(*a, **k):
            if k.get('cancellation_stage') == 'progress_optical_z_march' and stage == 'incomplete_replay':
                token.cancel()
            result = old(*a, **k)
            if k.get('cancellation_stage') == 'progress_optical_z_march':
                token.cancel()
            return result
        monkeypatch.setattr(workflow, '_optical_pass', optical)
    result = workflow.run_pr_timedependent(_request(), cancellation_token=token,
        progress_callback=progress.append, live_preview_policy=PRLivePreviewPolicy())
    assert result.completed_steps == 0 and result.status == 'cancelled'
    assert not progress
    np.testing.assert_array_equal(result.E_final, result.E_initial)


def test_throttle_first_final_stop_and_continue_identity(no_movie_encoder):
    policy = PRLivePreviewPolicy(interval_seconds=1e9)
    progress = []
    complete = workflow.run_pr_timedependent(_request(steps=4),
        progress_callback=progress.append, live_preview_policy=policy)
    assert [p.completed_units for p in progress if p.latest_field_state is not None] == [1, 4]
    token = CancellationToken(); progress = []
    def observe(p):
        progress.append(p)
        if p.completed_units == 2: token.cancel()
    stopped = workflow.run_pr_timedependent(_request(steps=4), cancellation_token=token,
        progress_callback=observe, live_preview_policy=policy)
    snapshots = [p.latest_field_state for p in progress if p.latest_field_state is not None]
    assert [s.completed_steps for s in snapshots] == [1, 2]
    np.testing.assert_array_equal(snapshots[-1].material_plane, stopped.E_final[1])
    resumed_progress = []
    resumed = workflow.continue_pr_timedependent(_request(steps=4), stopped.checkpoint, 2,
        progress_callback=resumed_progress.append, live_preview_policy=policy)
    _assert_same_physical_result(resumed, complete)
    assert [p.completed_units for p in resumed_progress] == [3, 4]
    assert [p.segment_completed_steps for p in resumed_progress] == [1, 2]
    assert [p.current_time for p in resumed_progress] == [.03, .04]


def snapshot_fixture(monkeypatch=None):
    shape = (17, 19, 15)
    E = np.arange(np.prod(shape), dtype=float).reshape(shape)
    A = np.ones((1, 19, 15), dtype=complex)
    source = 2 + E / E.max()
    grid = SimpleNamespace(xp=np, Nz=17, Nx=19, Ny=15, dx_um=2., dy_um=3., dz_um=4.)
    policy = PRLivePreviewPolicy(max_x=7, max_y=5, max_z=9)
    if monkeypatch is not None:
        def transfer(value):
            assert np.ndim(value) < 3
            assert np.size(value) <= 9*7
            return np.asarray(value)
        monkeypatch.setattr(live, 'asnumpy', transfer)
    snap = reduced_pr_live_snapshot(E=E, A=A, source=source, grid=grid,
        groups=('g',), peak_reference=2., background=2., policy=policy,
        completed_steps=2, segment_completed_steps=2, requested_steps=4,
        time_normalized=.02, scalar_values={'material_state_change_rms': .5},
        material_response='nonlinear')
    return snap, policy, E, source


def test_backend_sampling_before_host_transfer_and_byte_bound(monkeypatch):
    snap, policy, E, source = snapshot_fixture(monkeypatch)
    assert snap.array_bytes == policy.maximum_array_bytes
    assert PRLivePreviewPolicy().maximum_array_bytes == 790528
    for field in fields(snap):
        value = getattr(snap, field.name)
        if isinstance(value, np.ndarray):
            assert value.ndim <= 2 and not value.flags.writeable
            assert not np.shares_memory(value, E)
            assert not np.shares_memory(value, source)
    z = np.rint(np.linspace(0, 16, 9)).astype(int)
    x = np.rint(np.linspace(0, 18, 7)).astype(int)
    np.testing.assert_array_equal(snap.optical_xz, (source[z[:, None], x[None, :], 7]-2)*2)
    saved = snap.material_plane.copy(); E[:] = -99
    np.testing.assert_array_equal(snap.material_plane, saved)


def test_products_selection_scaling_ownership_and_failure(app, monkeypatch):
    snap, _, _, _ = snapshot_fixture()
    data = reduced_pr_live_to_run_data(snap)
    w = Workspace(); w.begin_request('old'); w.set_run_data(image_data())
    w.begin_request('new')
    assert w.result_ownership.text().startswith('Previous result')
    w.tabs.setCurrentWidget(w.console)
    w.set_run_data(data, state='Current accepted state')
    assert w.result_ownership.text() == 'Current accepted state'
    assert w.tabs.currentWidget() is w.console
    xy, long = w.image_pane, w.longitudinal_pane
    assert xy.field_selector.count() == 2
    xy.field_selector.setCurrentIndex(xy.field_selector.findData('live_material_plane'))
    long.z_plane_slider.setValue(1)
    xy.scale_controls.lower.setText('0'); xy.scale_controls.upper.setText('100')
    xy.scale_controls.apply_limits()
    old_long = long.xz_view.image.get_clim()
    assert xy.scales is not long.scales
    w.set_run_data(reduced_pr_live_to_run_data(replace(snap, completed_steps=3)),
                   state='Current accepted state')
    assert xy.field_selector.currentData() == 'live_material_plane'
    assert long.z_plane_slider.value() == 1
    assert 'fixed cuts' in long.z_plane_label.text()
    assert str(int(snap.z_um[1])) in long.z_plane_label.text()
    assert long.xz_view.image.get_clim() == long.yz_view.image.get_clim() == old_long
    assert xy.image_view.image.get_clim() == (0., 100.)
    w.finish_attempt('State at stop/cancellation')
    assert w.result_ownership.text() == 'State at stop/cancellation'
    def fail(*a): raise ValueError('render failure')
    monkeypatch.setattr(w.curve_pane, 'set_run_data', fail)
    with pytest.raises(ValueError): w.set_run_data(data, state='Current accepted state')
    assert w.result_ownership.text() == 'No displayed result — result update failed'
    assert w.tabs.currentWidget() is w.console and w.image_pane.isHidden()
    close_widget(w)


def test_same_volume_selection_coordinates_survive_update(app):
    w = Workspace(); w.set_run_data(image_data())
    p = w.longitudinal_pane
    p.z_plane_slider.setValue(0); p.set_cut_indices(1, 2)
    w.set_run_data(image_data(2.))
    assert (p._ix, p._iy, p._iz) == (1, 2, 0)
    close_widget(w)


def test_gui_dispatch_is_local_reduced_only_and_throttle_time_truthful(app):
    w = PRMainWindow(); calls = []
    w.runner = SimpleNamespace(run_registered=lambda *a, **k: calls.append(k))
    w._run_registered(_request())
    assert isinstance(calls[-1]['live_preview_policy'], PRLivePreviewPolicy)
    w.slurm_runner = w.runner
    w._remote_runner_kwargs = lambda: {}
    w._run_registered(_request())
    assert 'live_preview_policy' not in calls[-1]
    snap, _, _, _ = snapshot_fixture()
    p = RunProgress(PR_TIMEDEPENDENT_WORKFLOW, 'running', 2, 4, .02,
                    'material_time', 'normalized', 1., latest_field_state=snap)
    ws = w.results_panel.workspace
    ws.begin_request('run'); ws.tabs.setCurrentWidget(ws.console)
    w._on_progress(p)
    time_text = ws.image_pane.td_time_label.text()
    w._on_progress(replace(p, completed_units=3, current_coordinate=.03, latest_field_state=None))
    assert w.status_label.text() == 'Step 3/4'
    assert ws.image_pane.td_time_label.text() == time_text
    assert ws.tabs.currentWidget() is ws.console
    ws.finish_attempt('State at failure')
    assert ws.result_ownership.text() == 'State at failure'
    close_widget(w)


def test_remote_telemetry_omits_preview_arrays(tmp_path):
    snap, _, _, _ = snapshot_fixture()
    p = RunProgress(PR_TIMEDEPENDENT_WORKFLOW, 'running', 2, 4, .02,
                    'material_time', 'normalized', 1., latest_field_state=snap)
    assert _write_progress(tmp_path, p)
    import json
    payload = json.loads((tmp_path/'progress.json').read_text())
    assert set(payload) == {'schema_version', 'workflow', 'status', 'phase', 'message',
        'completed_units', 'total_units', 'current_coordinate', 'coordinate_name',
        'coordinate_unit', 'elapsed_wall_time'}
    assert all(not isinstance(value, (list, dict)) for value in payload.values())


def test_pre_cancelled_run_and_zero_steps_do_not_invent_accepted_updates(no_movie_encoder):
    token = CancellationToken(); token.cancel(); events = []
    result = workflow.run_pr_timedependent(_request(), cancellation_token=token,
        progress_callback=events.append, live_preview_policy=PRLivePreviewPolicy())
    assert result.completed_steps == 0 and not events
    result = workflow.run_pr_timedependent(_request(steps=0),
        progress_callback=events.append, live_preview_policy=PRLivePreviewPolicy())
    assert result.completed_steps == 0 and not events


def test_preview_does_not_transfer_full_volume_in_workflow(monkeypatch, no_movie_encoder):
    original = workflow.asnumpy
    # Full-volume transfers remain legitimate at final-result construction.
    # While a progress snapshot is built, only the snapshot builder may transfer
    # its sampled planes/cuts, never the legacy full E/source arrays.
    original_builder = workflow.reduced_pr_live_snapshot
    built = []
    def builder(**kwargs):
        assert kwargs['E'].ndim == kwargs['source'].ndim == 3
        result = original_builder(**kwargs); built.append(result); return result
    monkeypatch.setattr(workflow, 'reduced_pr_live_snapshot', builder)
    in_loop = True
    def transfer(value):
        if in_loop and getattr(value, 'ndim', 0) == 3:
            pytest.fail('legacy full-volume host transfer during accepted progress')
        return original(value)
    # The initial state validation may inspect arrays, but initialization does
    # not call asnumpy on complete volumes. Disable the spy before final output.
    monkeypatch.setattr(workflow, 'asnumpy', transfer)
    def observe(p):
        nonlocal in_loop
        if p.completed_units == p.total_units: in_loop = False
    workflow.run_pr_timedependent(_request(steps=2), progress_callback=observe,
        live_preview_policy=PRLivePreviewPolicy(interval_seconds=0))
    assert len(built) == 2


def test_coherent_intensity_is_computed_after_backend_sampling(monkeypatch):
    original = live.total_intensity; observed = []
    def intensity(A, **kwargs):
        observed.append(A.shape)
        return original(A, **kwargs)
    monkeypatch.setattr(live, 'total_intensity', intensity)
    snapshot_fixture()
    assert observed == [(1, 7, 5)]


def test_local_continue_adapter_requests_bounded_preview(monkeypatch, no_movie_encoder):
    import lcprop.pr.gui.main_window as gui
    result = workflow.run_pr_timedependent(_request(steps=1))
    seen = {}
    def continuation(*a, **kwargs):
        seen.update(kwargs); return result
    monkeypatch.setattr(gui, 'continue_pr_timedependent', continuation)
    gui._continue_pr_operation(_request(), checkpoint=result.checkpoint, additional_steps=1)
    assert isinstance(seen['live_preview_policy'], PRLivePreviewPolicy)


def test_full_transverse_local_dispatch_does_not_enable_reduced_preview(app):
    from tests.test_pr_transverse_production import _request as transverse_request
    w = PRMainWindow(); seen = []
    w.runner = SimpleNamespace(run_registered=lambda *a, **k: seen.append(k))
    w._run_registered(transverse_request())
    assert 'live_preview_policy' not in seen[-1]
    close_widget(w)


def test_generic_cuts_locked_manual_auto_and_new_run(app):
    snap, _, _, _ = snapshot_fixture()
    w = Workspace(); w.set_run_data(reduced_pr_live_to_run_data(snap), state='Current accepted state')
    pane = w.longitudinal_pane; c = pane.scale_controls
    c.lower.setText('0'); c.upper.setText('3'); c.apply_limits()
    c.mode.setCurrentIndex(c.mode.findData('locked'))
    newer = replace(snap, optical_xz=snap.optical_xz*10, optical_yz=snap.optical_yz*10)
    w.set_run_data(reduced_pr_live_to_run_data(newer), state='Current accepted state')
    assert pane.xz_view.image.get_clim() == pane.yz_view.image.get_clim() == (0., 3.)
    w.reset_field_color_scales(); w.set_run_data(reduced_pr_live_to_run_data(newer), state='Current accepted state')
    assert pane.xz_view.image.get_clim() == pane.yz_view.image.get_clim()
    assert pane.xz_view.image.get_clim() != (0., 3.)
    c.mode.setCurrentIndex(c.mode.findData('auto'))
    w.set_run_data(reduced_pr_live_to_run_data(snap), state='Current accepted state')
    assert pane.xz_view.image.get_clim() == pane.yz_view.image.get_clim()
    close_widget(w)


def test_semi_implicit_predictor_is_discarded_with_live_preview(monkeypatch, no_movie_encoder):
    import lcprop.pr.evolution as evolution
    token = CancellationToken(); events = []
    old = evolution.solve_periodic_variable_diffusion
    def solve(*a, **k):
        candidate = old(*a, **k); token.cancel(); return candidate
    monkeypatch.setattr(evolution, 'solve_periodic_variable_diffusion', solve)
    result = workflow.run_pr_timedependent(request_for(PR_SEMI_IMPLICIT_INTEGRATOR),
        cancellation_token=token, progress_callback=events.append,
        live_preview_policy=PRLivePreviewPolicy())
    assert not events and result.completed_steps == 0
    assert result.diagnostics['cancellation_observed_stage'] == 'semi_implicit_predictor_corrector'
    np.testing.assert_array_equal(result.E_initial, result.E_final)


def test_failed_replay_cannot_publish_trial_as_accepted(monkeypatch, no_movie_encoder):
    old = workflow._optical_pass; events = []; replays = 0
    def optical(*a, **k):
        nonlocal replays
        if k.get('cancellation_stage') == 'progress_optical_z_march':
            replays += 1
            if replays == 2: raise RuntimeError('trial replay failed')
        return old(*a, **k)
    monkeypatch.setattr(workflow, '_optical_pass', optical)
    with pytest.raises(RuntimeError, match='trial replay failed'):
        workflow.run_pr_timedependent(_request(), progress_callback=events.append,
                                     live_preview_policy=PRLivePreviewPolicy(interval_seconds=0))
    assert [p.completed_units for p in events] == [1]
    assert events[0].latest_field_state.completed_steps == 1


def test_sample_provenance_keeps_reduced_model_validation_status():
    snap, _, _, _ = snapshot_fixture()
    data = reduced_pr_live_to_run_data(replace(snap, material_response='linearized'))
    summary = data.diagnostics['live_accepted_state'].values
    assert summary['material_response'] == 'linearized'
    assert summary['material_response_validation'] == 'locally_validated'
    assert summary['visualization_only'] is True
    assert not data.curves and not data.artifacts


def test_real_worker_run_and_continue_render_intermediate_accepted_state(app, monkeypatch, no_movie_encoder):
    """Exercise native dispatch/signals/slot/adapter/views without replacing them."""
    from PySide6.QtCore import QThread
    from tests.test_pr_gui_main_window import _wait_for

    w = PRMainWindow()
    w.grid_panel.Nx.setValue(8)
    w.grid_panel.Ny.setValue(8)
    w.grid_panel.dz_um.setValue(10.)
    w.grid_panel.z_length_um.setValue(20.)
    w.evolution_panel.Nt.setValue(10)
    ws = w.results_panel.workspace
    gui_thread = QThread.currentThread()
    deliveries, displays, runner_threads = [], [], []
    original_data = ws.set_run_data
    original_time = ws.set_td_time_indicator
    original_runner = w.local_runner.run_registered

    def run_registered(*args, **kwargs):
        runner_threads.append(QThread.currentThread())
        return original_runner(*args, **kwargs)

    def set_data(data, *, state=None):
        original_data(data, state=state)
        if state == 'Current accepted state':
            deliveries.append((data, ws._attempt, ws._displayed_attempt))

    def set_time(text):
        original_time(text)
        progress = w.last_progress
        if progress is None or not isinstance(progress.latest_field_state, PRLiveSnapshot):
            return
        # Observe after the real slot has rendered fields AND set their time.
        # Store evidence here; assertions run outside Qt's exception boundary.
        xy, longitudinal = ws.image_pane, ws.longitudinal_pane
        displays.append(dict(
            snapshot=progress.latest_field_state, text=text,
            ownership=ws.result_ownership.text(),
            xy_key=xy.field_selector.currentData(),
            xy_count=xy.field_selector.count(),
            xy_size=np.size(xy.image_view.image.get_array()),
            cut_key=longitudinal.field_selector.currentData(),
            cut_count=longitudinal.field_selector.count(),
            xz_size=np.size(longitudinal.xz_view.image.get_array()),
            yz_size=np.size(longitudinal.yz_view.image.get_array()),
            thread=QThread.currentThread(),
        ))

    monkeypatch.setattr(w.local_runner, 'run_registered', run_registered)
    monkeypatch.setattr(ws, 'set_run_data', set_data)
    monkeypatch.setattr(ws, 'set_td_time_indicator', set_time)
    try:
        w.run_clicked()
        _wait_for(app, lambda: not w._background_running)
        assert w.last_checkpoint is not None
        first_run = list(displays)
        displays.clear()
        w.continue_clicked()
        _wait_for(app, lambda: not w._background_running)
        assert runner_threads and all(t != gui_thread for t in runner_threads)
        for observed, first_step, total in ((first_run, 1, 10), (displays, 11, 20)):
            assert observed
            first = observed[0]
            s = first['snapshot']
            assert s.completed_steps == first_step < total
            assert s.segment_completed_steps == 1
            assert s.requested_steps == total
            assert s.time_normalized == pytest.approx(first_step * .001)
            assert first['ownership'] == 'Current accepted state'
            assert first['xy_count'] >= 1 and first['xy_size'] > 0
            assert first['xy_key'] == 'live_output_intensity'
            assert first['cut_count'] >= 1 and first['xz_size'] > 0 and first['yz_size'] > 0
            assert first['cut_key'] == '__paired_cuts__:live_optical_xz'
            assert first['text'] == (
                f'PR material time: {s.time_normalized:.6g} normalized; '
                f'step {first_step}/{total}')
            assert first['thread'] == gui_thread
            assert observed[-1]['snapshot'].completed_steps == total
        assert deliveries
        assert all(len(data.fields) == 4 and attempt == owner
                   for data, attempt, owner in deliveries)
        assert len({attempt for _, attempt, _ in deliveries}) == 2
    finally:
        assert w.shutdown_background_run()
        close_widget(w)

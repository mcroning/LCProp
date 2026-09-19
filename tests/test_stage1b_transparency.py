"""Stage 1B presentation regressions using retained/synthetic state, not solves."""
from dataclasses import replace
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.gui.workspace import Workspace
from lcprop.gui.request_transparency import execution_summary
from lcprop.gui.views.image_pane import display_limits
from lcprop.gui.views.display_scale import scale_key
from lcprop.lc.requests import SolitonRequest
from lcprop.lc.results import SolitonResult, SolitonSweepMember, ParameterSweepResult
from lcprop.lc.products import from_parameter_sweep_result, from_soliton_result
from lcprop.lc.soliton_presentation import qualification
from lcprop.products.data_model import FieldData, FieldCollection, RunData, CurveData, CurveCollection
from tests.test_all_workflows import make_base_static_request


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def soliton(**changes):
    request = SolitonRequest(base=make_base_static_request())
    last = dict(outer=100, beta=2., residual_rms=1e-5, residual_max=1e-4,
                field_rel=0.001414, overlap_abs=0.999999, dtheta_rms=1e-7)
    result = SolitonResult(request=request, history=[last], metrics={
        **last, "convergence_status": "max_outer_reached", "sx_um": 1., "sy_um": 2.,
    }, completed_iterations=100, total_iterations=100,
        A=np.ones((1, 4, 4), complex), theta=np.zeros((4, 4)), intensity=np.ones((4, 4)))
    return replace(result, **changes)


def sweep_result(member=None):
    result = soliton() if member is None else member
    samples = [{"i": 0, "requested_power_mW": 1., **result.metrics, "converged": result.converged}]
    return ParameterSweepResult(samples=samples, results=[result], members=[
        SolitonSweepMember(0, 1., result=result, status="completed", converged=result.converged),
        SolitonSweepMember(1, 2., status="failed", error_text="worker failure"),
        SolitonSweepMember(2, 3., status="cancelled"),
    ])


def test_gui_defaults_and_exact_failed_gate_not_overlap():
    result = soliton()
    req = result.request
    assert (req.max_outer, req.field_mix, req.tol_field, req.tol_theta,
            req.tol_residual_rms, req.tol_residual_max) == (100, .5, 1e-4, 1e-5, 5e-3, 5e-2)
    summary, gates = qualification(result)
    assert summary['termination_reason'] == 'max_outer_reached'
    assert summary['iteration_budget'] == summary['completed_iterations'] == 100
    assert 'did not converge' in summary['solver_status']
    assert [r['key'] for r in gates if r['qualification'] == 'Fail'] == ['field_rel']
    assert gates[0]['value'] == result.history[-1]['field_rel']
    assert gates[0]['tolerance'] == req.tol_field


@pytest.mark.parametrize('key,tol', [('field_rel','tol_field'), ('dtheta_rms','tol_theta'),
                                     ('residual_rms','tol_residual_rms'), ('residual_max','tol_residual_max')])
def test_strict_gate_uses_exact_outer_value_and_custom_tolerance(key, tol):
    result = soliton()
    result.request = replace(result.request, **{tol: .0123456789})
    result.history[-1][key] = .0123456789
    result.metrics[key] = 0.  # Final display metrics must not override qualification.
    _, gates = qualification(result)
    row = next(r for r in gates if r['key'] == key)
    assert row['qualification'] == 'Fail'
    assert row['value'] == row['tolerance'] == .0123456789


def test_transverse_qualification_uses_outer_optical_residual_not_polish():
    result = soliton()
    result.history[-1]['optical_residual'] = .01
    result.metrics.update(solver='transverse_eigen', optical_residual=1e-12)
    summary, gates = qualification(result)
    assert len(gates) == 5 and gates[-1]['value'] == .01
    assert gates[-1]['qualification'] == 'Fail'
    assert 'does not update' in summary['polishing_note']


def test_no_completed_iteration_does_not_fabricate_gate_values():
    result = soliton(history=[], status='stopped', completed_iterations=0,
                     metrics={'convergence_status':'stopped'})
    summary, gates = qualification(result)
    assert summary['solver_status'] == 'Stopped / cancelled'
    assert all(g['qualification'] == 'Unavailable' for g in gates)


def test_samples_inspectable_and_distinct_worker_states(app):
    source = sweep_result()
    run = from_parameter_sweep_result(source)
    rows = run.diagnostics['members'].values['rows']
    assert 'did not converge' in rows[0]['solver_status']
    assert rows[1]['solver_status'] == 'Execution failure'
    assert rows[2]['solver_status'] == 'Stopped / cancelled'
    assert rows[0]['field_rel_qualification_value'] == source.results[0].history[-1]['field_rel']
    assert rows[0]['iteration_budget'] == 100
    note = run.diagnostics['convergence_gates'].values['interpretation']
    assert 'previously converged' in note and 'must still pass' in note and 'nonexistence' in note
    workspace = Workspace()
    workspace.set_run_data(run)
    pane = workspace.table_pane
    pane.selector.setCurrentIndex(pane.selector.findData('members'))
    assert pane.table.rowCount() == 3
    values = [pane.table.item(0,j).text() for j in range(pane.table.columnCount())]
    assert repr(source.results[0].history[-1]['field_rel']) in values
    workspace.tabs.setCurrentWidget(pane)
    workspace.set_run_data(RunData('empty'))
    assert workspace.tabs.currentWidget() is pane and pane.table.rowCount() == 0
    workspace.close()


def test_table_cleared_after_partial_render_failure(app, monkeypatch):
    workspace = Workspace();run = from_parameter_sweep_result(sweep_result())
    workspace.set_run_data(run)
    def fail(*a):raise RuntimeError('render failed')
    monkeypatch.setattr(workspace.curve_pane, 'set_run_data', fail)
    with pytest.raises(RuntimeError):workspace.set_run_data(run)
    assert workspace.table_pane.table.rowCount() == 0
    assert 'update failed' in workspace.result_ownership.text()
    workspace.close()


def test_curves_tolerances_overlap_and_width_labels(app):
    run = from_parameter_sweep_result(sweep_result())
    workspace = Workspace();workspace.set_run_data(run);pane = workspace.curve_pane
    pane.curve_selector.setCurrentIndex(pane.curve_selector.findData('field_rel'))
    assert 'Strict tolerance' in pane.tolerance_label.text()
    assert len(pane.curve_view.ax.lines) == 2
    assert pane.curve_view.ax.lines[-1].get_ydata()[0] == 1e-4
    pane.curve_selector.setCurrentIndex(pane.curve_selector.findData('overlap_abs'))
    assert not pane.tolerance_label.text()
    assert not pane.curve_view.ax.yaxis.get_major_formatter().get_useOffset()
    assert len(pane.curve_view.ax.lines) == 1
    widths = run.curves['transverse_rms_widths']
    assert widths.display_name == 'Transverse RMS widths' and widths.series_labels == ('x','y')
    assert run.curves['converged'].display_name == 'Solver converged within configured limits'
    workspace.close()


def test_single_soliton_gates_are_available():
    run = from_soliton_result(soliton())
    assert len(run.diagnostics['convergence_gates'].values['rows']) == 4


def image_data(scale=1.):
    volume = np.arange(24., dtype=float).reshape(2,3,4) * scale
    fields = FieldCollection([
        ('volume', FieldData('volume','Intensity volume',volume,('z','x','y'),'intensity',quantity='I')),
        ('plane', FieldData('plane','Intensity slice',volume[1],('x','y'),'intensity',quantity='I',source_volume_key='volume')),
        ('theta', FieldData('theta','Director',np.ones((3,4)),('x','y'),'theta',quantity='theta')),
    ])
    return RunData('timedependent',fields=fields,curves=CurveCollection([
        ('power',CurveData('power','Power',np.arange(2),np.array([1.,2.]),'t','P'))]))


def test_auto_fixed_return_auto_and_invalid_ranges(app):
    workspace = Workspace();run = image_data();workspace.set_run_data(run)
    pane = workspace.image_pane; controls = pane.scale_controls
    selected = pane._field_at_selected_z(run.fields['plane'])
    assert pane.image_view.image.get_clim() == display_limits(selected)
    original = {k:f.data.copy() for k,f in run.fields.items()}
    curve = run.curves['power'].y.copy()
    controls.lower.setText('2');controls.upper.setText('10');controls.apply_limits()
    assert pane.image_view.image.get_clim() == (2.,10.)
    assert workspace.longitudinal_pane.xz_view.image.get_clim() == (0.,23.)
    assert workspace.longitudinal_pane.yz_view.image.get_clim() == (0.,23.)
    for lo,hi in [('nan','4'),('5','2'),('a','5'),('inf','10'),('3','3')]:
        controls.lower.setText(lo);controls.upper.setText(hi);controls.apply_limits()
        assert 'Invalid' in controls.message.text()
        assert pane.image_view.image.get_clim() == (2.,10.)
    controls.mode.setCurrentIndex(controls.mode.findData('auto'))
    assert pane.image_view.image.get_clim() == display_limits(selected)
    for k,f in run.fields.items():np.testing.assert_array_equal(f.data,original[k])
    np.testing.assert_array_equal(run.curves['power'].y,curve)
    workspace.close()


def test_locked_frames_share_mapping_unlocked_rescale_and_quantity_isolation(app, tmp_path):
    workspace = Workspace();first = image_data();workspace.set_run_data(first)
    pane = workspace.image_pane;controls = pane.scale_controls
    limits = pane.image_view.image.get_clim()
    controls.mode.setCurrentIndex(controls.mode.findData('locked'))
    color_before = pane.image_view.image.cmap(pane.image_view.image.norm(5.))
    second = image_data(100.);workspace.set_run_data(second)
    assert pane.image_view.image.get_clim() == limits
    assert workspace.longitudinal_pane.xz_view.image.get_clim() == (0.,2300.)
    assert pane.image_view.image.cmap(pane.image_view.image.norm(5.)) == color_before
    pane.field_selector.setCurrentIndex(pane.field_selector.findData('theta'))
    assert pane.scale_controls.mode.currentData() == 'auto'
    assert pane.image_view.image.get_clim() != limits
    pane.field_selector.setCurrentIndex(pane.field_selector.findData('plane'))
    controls.mode.setCurrentIndex(controls.mode.findData('auto'))
    assert pane.image_view.image.get_clim()[1] > limits[1]
    # Scientific serialization is still the unmodified second frame.
    target = tmp_path/'scientific.npz'
    np.savez(target, volume=second.fields['volume'].data, power=second.curves['power'].y)
    with np.load(target) as saved:
        np.testing.assert_array_equal(saved['volume'],np.arange(24.).reshape(2,3,4)*100.)
        np.testing.assert_array_equal(saved['power'],[1.,2.])
    workspace.close()


@pytest.mark.parametrize('backend',['numpy','cupy','auto'])
@pytest.mark.parametrize('remote',[False,True])
def test_execution_summary_truthful_and_no_backend_mutation(backend,remote):
    profile = SimpleNamespace(name='H200',gpus=1)
    runner = SimpleNamespace(config=SimpleNamespace(cluster_profile='cluster',default_resource_profile='H200',resource_profiles=[profile]))
    request = SimpleNamespace(backend=SimpleNamespace(backend=backend,precision='float64'))
    window = SimpleNamespace(runner=runner if remote else object(),slurm_runner=runner,_explicit_slurm_runner=runner,
        remote_execution_controls=SimpleNamespace(runner_kwargs=lambda:{}),
        result_policy_selector=SimpleNamespace(currentData=lambda:'fast'))
    text = execution_summary(window,request)
    assert ('Execution target: Slurm' if remote else 'Execution target: Local') in text
    assert f'Requested backend: {backend}' in text and 'Precision: float64' in text
    assert 'Resolved backend: unresolved' in text
    assert ('will not use the allocated GPU' in text) == (remote and backend=='numpy')
    if remote:assert 'Resource: H200' in text and 'Retrieval policy: fast' in text
    assert request.backend.backend == backend


def test_planning_separates_configured_execution_without_changing_estimate(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.pr.runtime_estimator import estimate_pr_resources, format_pr_resource_estimate
    window = PRMainWindow()
    request = window._request_for_local_cost(window.build_request())
    expected = format_pr_resource_estimate(estimate_pr_resources(request))
    window._estimate_current_resources()
    text = window.resource_estimator_panel.output.toPlainText()
    assert text.startswith('Configured execution:') and 'Execution target: Local' in text
    assert text.endswith(expected) and 'Comparison estimates (not the selected execution plan)' in text
    window.result_policy_selector.setCurrentIndex(1)
    assert 'stale' in window.resource_estimator_panel.output.toPlainText()
    window.close()


def test_actual_gui_existence_request_uses_soliton_defaults(app):
    from lcprop.lc.gui.main_window import LCPropMainWindow
    from lcprop.lc.requests import ParameterSweepRequest
    window = LCPropMainWindow()
    request = window.build_soliton_existence_request()
    assert isinstance(request, ParameterSweepRequest)
    assert isinstance(request.base, SolitonRequest)
    defaults = SolitonRequest(base=request.base.base)
    for name in ('max_outer','field_mix','tol_field','tol_theta','tol_residual_rms','tol_residual_max'):
        assert getattr(request.base,name) == getattr(defaults,name)
    window.close()


@pytest.mark.parametrize('value',[0., float('nan')])
def test_lock_handles_constant_and_nonfinite_longitudinal_fields(app,value):
    workspace = Workspace();data = image_data()
    data.fields['volume'].data.fill(value)
    workspace.set_run_data(data)
    controls = workspace.longitudinal_pane.scale_controls
    controls.mode.setCurrentIndex(controls.mode.findData('locked'))
    limits = workspace.longitudinal_pane.xz_view.image.get_clim()
    assert np.all(np.isfinite(limits)) and limits[0] < limits[1]
    workspace.set_run_data(image_data(10.))
    assert workspace.longitudinal_pane.xz_view.image.get_clim() == limits
    workspace.close()


def test_converged_status_and_capability_aware_missing_request():
    result = soliton(converged=True, metrics={'convergence_status':'converged'})
    summary, _ = qualification(result)
    assert summary['solver_status'] == 'Solver converged within configured limits'
    _, rows = qualification(replace(result, request=None))
    assert all(r['tolerance'] is None and r['qualification']=='Unavailable' for r in rows)


def test_progress_does_not_overwrite_pending_manual_limit_edits(app):
    workspace = Workspace()
    workspace.set_run_data(image_data())
    controls = workspace.image_pane.scale_controls
    controls.lower.setText('3.5')
    controls.upper.setText('8.5')
    # QLineEdit marks user edits as modified; setText alone is programmatic.
    controls.lower.setModified(True)
    controls.upper.setModified(True)
    workspace.set_run_data(image_data(20.))
    assert controls.lower.text() == '3.5' and controls.upper.text() == '8.5'
    controls.apply_limits()
    assert workspace.image_pane.image_view.image.get_clim() == (3.5, 8.5)
    controls.mode.setCurrentIndex(controls.mode.findData('auto'))
    assert not controls.lower.isModified() and not controls.upper.isModified()
    workspace.close()


def test_legacy_generic_sweep_retained_results_still_expose_gates():
    source = sweep_result()
    source.members = []
    run = from_parameter_sweep_result(source)
    gates = run.diagnostics['convergence_gates'].values['rows']
    assert len(gates) == 4
    assert gates[0]['value'] == source.results[0].history[-1]['field_rel']

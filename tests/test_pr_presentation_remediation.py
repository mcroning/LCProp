"""GUI-only ownership and presentation checks; no workflow execution."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFormLayout

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_startup_frame_and_label_hover(app):
    from lcprop.optics.physical_launch import transverse_frame
    from lcprop.pr.gui.beam_panel import pr_default_beam_stack_definition, make_pr_beam_panel
    beam=pr_default_beam_stack_definition().beams[0]
    assert beam.theta_ext_rad == beam.phi_rad == beam.psi_rad == 0
    np.testing.assert_array_equal(transverse_frame(0,0,0), [[1,0,0],[0,1,0]])
    p=make_pr_beam_panel(x_aperture_um=200,y_aperture_um=200)
    w=p.launch_plane_widget
    for control in (w.w1_spin,w.w2_spin,w.psi_spin):
        labels=[f.labelForField(control) for f in w.findChildren(QFormLayout)]
        assert any(label is not None and label.toolTip()==control.toolTip() for label in labels)
    assert 'aligns with x' in w.w1_spin.toolTip()
    assert '\n' in w.enabled_checkbox.toolTip()
    p.close()


def test_save_folder_formats():
    from lcprop.pr.gui.presentation_guidance import checkpoint_folder_hint
    assert 'checkpoint.npz, request.json and provenance.json' in checkpoint_folder_hint()
    assert 'accepted-psi.npy and transverse-continuation.json' in checkpoint_folder_hint(transverse=True)


def test_current_inspection_and_history_are_separate(app, monkeypatch):
    from lcprop.pr.gui.main_window import PRMainWindow
    w=PRMainWindow();ws=w.results_panel.workspace
    ws._displayed_request='HISTORICAL SCIENTIFIC REQUEST';ws._displayed_attempt=1
    ws._display_state='Completed';ws._attempt=1
    w.result_policy_selector.setCurrentIndex((w.result_policy_selector.currentIndex()+1)%w.result_policy_selector.count())
    assert 'configuration changed' in ws.request_summary.toPlainText()
    assert ws._displayed_request=='HISTORICAL SCIENTIFIC REQUEST'
    monkeypatch.setattr(w,'build_request',lambda: (_ for _ in ()).throw(ValueError('missing explicit irradiance')))
    w.preview_request_clicked()
    assert 'missing explicit irradiance' in ws.operation_status.text()
    assert 'request rejected; no new result' in ws.request_summary.toPlainText()
    assert 'Previous result' in ws.result_ownership.text()
    assert ws._displayed_request=='HISTORICAL SCIENTIFIC REQUEST'
    monkeypatch.setattr(w,'build_request',lambda: object())
    monkeypatch.setattr(w,'_validate_execution_request',lambda r: None)
    monkeypatch.setattr(w,'describe_request',lambda r: 'CURRENT EXECUTION SELECTION')
    w.preview_request_clicked()
    assert 'CURRENT EXECUTION SELECTION' in ws.request_summary.toPlainText()
    assert ws._displayed_request=='HISTORICAL SCIENTIFIC REQUEST'
    w.close()


def test_loaded_source_and_checkpoint_identity(app,monkeypatch,tmp_path):
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.pr.transverse import continuation
    cp=continuation.TransverseTDCheckpoint(None,None,{'status':'cancelled',
        'cumulative_completed_steps':12,'cumulative_time':.5},'detail-hash')
    (tmp_path/'transverse-continuation.json').write_text('{}')
    monkeypatch.setattr(continuation,'load_checkpoint',lambda path: cp)
    w=PRMainWindow();w.load_checkpoint_from(tmp_path)
    assert w.last_checkpoint is cp
    text=w.results_panel.workspace.request_summary.toPlainText()
    assert str(tmp_path.resolve()) in text and 'Run PR uses the current controls' in text
    assert 'detail-hash' in text and 'detail-hash' not in w.checkpoint_owner_label.text()
    assert str(tmp_path.resolve()) in w.checkpoint_source_label.text()
    w._mark_request_inspection_stale()
    assert str(tmp_path.resolve()) in w.checkpoint_source_label.text()
    assert not w.checkpoint_source_label.isHidden()
    assert 'accepted-psi.npy' in w.save_checkpoint_button.toolTip()
    # A failed load cannot relabel the still-retained checkpoint.
    monkeypatch.setattr(continuation,'load_checkpoint',lambda p: (_ for _ in ()).throw(ValueError('bad checkpoint')))
    with pytest.raises(ValueError,match='bad checkpoint'):
        w.load_checkpoint_from(tmp_path)
    assert w.last_checkpoint is cp and str(tmp_path.resolve()) in w.checkpoint_source_label.text()
    w.last_checkpoint=None;w._refresh_checkpoint_controls()
    assert w.checkpoint_source_label.isHidden() and w._loaded_checkpoint_source is None
    w.close()


def test_editor_details_optional_request_unchanged(app):
    from tests.test_pr_transverse_continuation import request
    from lcprop.pr.gui.transverse_continuation import TransverseContinuationDialog
    r=request();cp=SimpleNamespace(request=r,record={'lineage':[]},time_normalized=.1,completed_steps=2)
    describe=lambda r,c: {'identity':'retained-detail-hash'}
    d=TransverseContinuationDialog(cp,describe=describe)
    assert d.preview.isHidden() and 'retained-detail-hash' in d.preview.toPlainText()
    before=d.build_request()
    d.details_button.setChecked(True)
    assert not d.preview.isHidden() and d.build_request()==before
    d.dark.setText('bad');d.inspect_request()
    assert 'Invalid continuation' in d.inspection_status.text()
    d.close()


def test_explicit_startup_inputs_without_changing_defaults(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    w=PRMainWindow()
    assert w.material_panel.dark_irradiance.text()==''
    assert w.material_panel.uniform_irradiance.text()==''
    with pytest.raises(ValueError,match='irradiance'):
        w.build_request()
    w.material_panel.dark_irradiance.setText('10')
    w.material_panel.uniform_irradiance.setText('0')
    r=w.build_request()
    assert r.material.dark_irradiance_W_cm2 == .01
    assert r.material.uniform_irradiance_W_cm2 == 0
    w._validate_execution_request(r)
    w.close()


def test_inspection_reads_current_local_slurm_selection(app, monkeypatch):
    from lcprop.pr.gui.main_window import PRMainWindow
    w=PRMainWindow()
    w.material_panel.dark_irradiance.setText('10');w.material_panel.uniform_irradiance.setText('0')
    request=w.build_request()
    # Exercise request/summary generation, never remote validation or execution.
    monkeypatch.setattr(w,'_validate_execution_request',lambda r: None)
    w.preview_request_clicked()
    assert 'Execution target: Local' in w.results_panel.workspace.request_summary.toPlainText()
    w.slurm_runner=w._explicit_slurm_runner=SimpleNamespace(name='test Slurm',config=SimpleNamespace(
        cluster_profile='fixture',resource_profiles=(),default_resource_profile='h200'))
    w.execution_target_selector.setCurrentIndex(w.execution_target_selector.findData('slurm'))
    assert 'configuration changed' in w.results_panel.workspace.request_summary.toPlainText()
    from dataclasses import replace
    remote=w.build_request()
    assert replace(remote,backend=request.backend)==request
    assert remote.backend.backend == 'cupy'  # Existing execution-context behavior.
    w.preview_request_clicked()
    assert 'Execution target: Slurm' in w.results_panel.workspace.request_summary.toPlainText()
    w.execution_target_selector.setCurrentIndex(w.execution_target_selector.findData('local'))
    w.preview_request_clicked()
    assert 'Execution target: Local' in w.results_panel.workspace.request_summary.toPlainText()
    assert w.build_request()==request
    w.close()


@pytest.mark.parametrize('control', ['backend', 'precision'])
def test_execution_details_invalidate_preview_without_rewriting_history(app, control):
    from lcprop.pr.gui.main_window import PRMainWindow
    w=PRMainWindow();ws=w.results_panel.workspace
    ws._displayed_request='immutable completed request'
    ws.set_request_summary('old validated inspection')
    selector=getattr(w.evolution_panel,control)
    selector.setCurrentIndex((selector.currentIndex()+1)%selector.count())
    assert 'configuration changed' in ws.request_summary.toPlainText()
    assert ws._displayed_request=='immutable completed request'
    assert selector.currentIndex()==1
    w.close()

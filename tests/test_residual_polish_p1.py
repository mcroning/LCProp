"""P1 presentation regressions: retained products and offscreen widgets only."""
import gc
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QFocusEvent
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from lcprop.gui.number_format import format_number
from lcprop.gui.numeric_widgets import CompactDoubleSpinBox
from lcprop.gui.convergence_summary import member_explanations
from lcprop.gui.workspace import Workspace
from lcprop.gui.views.display_scale import DisplayScales, DisplayScaleControls
from lcprop.gui.help import HELP_TOPICS
from lcprop.products.data_model import RunData, DiagnosticData, DiagnosticCollection
from lcprop.lc.products import from_parameter_sweep_result
from tests.test_stage1b_transparency import soliton, sweep_result, image_data


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('value,quantity,expected', [
    (.005000000000000001, 'ordinary', '0.005'),
    (2.4, 'ordinary', '2.4'), (6.4e22, 'ordinary', '6.4e22'),
    (.42840199999, 'ordinary', '0.428402'),
    (9.99714e-5, 'ordinary', '9.99714e-5'),
    (12.3456789123, 'coordinate', '12.3457'),
    (.005000000000000001, 'tolerance', '0.005'),
    (1. - 1e-14, 'overlap_abs', '0.99999999999999'),
])
def test_quantity_formatting_is_display_only(value, quantity, expected):
    payload = {'value': value}
    before = json.dumps(payload)
    assert format_number(value, quantity=quantity) == expected
    assert json.dumps(payload) == before
    assert float(format_number(value, exact=True)) == value


@pytest.mark.parametrize('value,decimals', [(2.4,9), (6.4e22,9), (.123456789012,12), (9.99714e-5,12)])
def test_material_spin_display_keeps_exact_value_and_edit_precision(app, value, decimals):
    w = CompactDoubleSpinBox(); w.setRange(0, 1e30);w.setDecimals(decimals);w.setValue(value)
    original = w.value();changes=[];w.valueChanged.connect(changes.append)
    assert w.text() == format_number(original)
    w.interpretText()
    assert w.value() == original
    w.focusInEvent(QFocusEvent(QEvent.FocusIn))
    w.focusOutEvent(QFocusEvent(QEvent.FocusOut))
    assert w.value() == original and changes == []
    w.lineEdit().setText('0.123456789');w.interpretText()
    assert w.value() == .123456789
    w.close()


def test_compact_limits_apply_without_edit_keeps_exact_range(app):
    scales=DisplayScales();c=DisplayScaleControls(scales)
    limits=(.123456789012345, .987654321012345)
    c.show_scale('field', limits)
    assert c.lower.text() == '0.123457'
    assert str(limits[0]) in c.lower.toolTip()
    c.apply_limits()
    assert scales.settings['field'] == ('fixed', limits)
    c.lower.setText('.22222222222222');c.lower.setModified(True)
    c.apply_limits()
    assert scales.settings['field'][1] == (.22222222222222,limits[1])
    c.close()


def test_lifecycle_labels_keep_internal_identity_and_previous_result(app):
    w=Workspace();w.set_request_summary('standalone preview')
    assert w.request_summary.toPlainText() == 'standalone preview'
    w.begin_request('Run');w.set_request_summary('first config');w.set_run_data(image_data())
    w.tabs.setCurrentWidget(w.console)
    assert w.result_ownership.text()=='Completed result'
    assert w._displayed_attempt==1 and 'Request 1' in w.request_summary.toPlainText()
    w.begin_request('Run');w.set_request_summary('second config')
    assert w.result_ownership.text().startswith('Previous result')
    assert 'first config' in w.request_summary.toPlainText()
    assert w._displayed_attempt==1 and w._attempt==2
    w.set_run_data(RunData('empty'))
    assert w.result_ownership.text()=='Completed result' and w._displayed_attempt==2
    assert w.tabs.currentWidget() is w.console and w.table_pane.table.rowCount()==0
    assert w.table_pane.empty_label.isHidden() is False
    w.close()


def test_existing_console_boundary_is_bold_timestamped_and_preserves_history(app):
    w=Workspace();w.tabs.setCurrentWidget(w.console)
    w.operation_boundary('Run <special>');w.append_console('warning one')
    w.operation_boundary('Continue');w.append_console('progress two')
    text=w.console.toPlainText()
    assert text.index('Run <special>') < text.index('warning one') < text.index('Continue') < text.index('progress two')
    assert text.count('Run <special>')==1 and text.count('Continue')==1
    assert '[' in text and 'font-weight:700' in w.console.toHtml()
    assert w.tabs.currentWidget() is w.console
    w.close()


def test_progress_wording_preserves_explicit_capability_and_previous_ownership(app):
    w=Workspace();w.begin_request('Run')
    w.set_run_data(RunData('static'),state='Current accepted state')
    assert 'not available yet' in w.longitudinal_pane.no_data_label.text()
    w.begin_request('Run')
    assert 'Previous result' in w.result_ownership.text()
    w.set_run_data(RunData('static'))
    assert 'retained in this result' in w.longitudinal_pane.no_data_label.text()
    w.set_run_data(RunData('soliton'),state='Current progress state')
    assert 'transverse fields only' in w.longitudinal_pane.no_data_label.text()
    explicit=RunData('soliton',longitudinal_message='Stationary soliton has no longitudinal evolution.')
    w.set_run_data(explicit,state='Current progress state')
    assert w.longitudinal_pane.no_data_label.text()==explicit.longitudinal_message
    w.close()


def test_empty_tables_explained_and_populated_tables_have_exact_tooltips(app):
    w=Workspace();run=RunData('timedependent')
    w.set_run_data(run);w.tabs.setCurrentWidget(w.table_pane)
    assert not w.table_pane.empty_label.isHidden() and w.table_pane.selector.isHidden()
    raw=.005000000000000001
    values={'rows':[{'beta':raw,'overlap_abs':1.-1e-14}]}
    richer=RunData('example',diagnostics=DiagnosticCollection([('table',DiagnosticData('table','Samples',values))]))
    w.set_run_data(richer);t=w.table_pane.table
    assert t.item(0,0).text()=='0.005' and t.item(0,0).toolTip()==repr(raw)
    assert t.item(0,1).text()=='0.99999999999999'
    assert values['rows'][0]['beta']==raw and w.tabs.currentWidget() is w.table_pane
    w.set_run_data(run);assert t.rowCount()==0 and not w.table_pane.empty_label.isHidden()
    w.close()


def test_summary_uses_retained_gates_and_separates_execution(app):
    result=soliton();result.history[-1]['field_rel']=.000517
    result.metrics['field_rel']=0.  # Returned/post-polish metric is not qualification.
    run=from_parameter_sweep_result(sweep_result(result));before=repr(run.diagnostics['convergence_gates'].values)
    explanations=member_explanations(run);first=explanations[0][2]
    assert '1 mW' in explanations[0][1]
    assert 'Execution status: completed' in first
    assert 'within configured limits: No' in first
    assert 'Reached the iteration limit' in first and '100 / 100' in first
    assert 'Field relative change = 0.000517; required < 0.0001 (failed)' in first
    assert 'Other configured convergence gates passed' in first
    assert 'physical nonexistence or instability' in first
    assert 'Execution status: failed' in explanations[1][2] and 'worker failure' in explanations[1][2]
    assert 'unavailable' in explanations[1][2]
    assert repr(run.diagnostics['convergence_gates'].values)==before


def test_summary_unknown_evidence_and_strict_equality(app):
    result=soliton();result.history[-1]['field_rel']=result.request.tol_field
    first=member_explanations(from_parameter_sweep_result(sweep_result(result)))[0][2]
    assert '0.0001; required < 0.0001 (failed)' in first
    result.history=[]
    first=member_explanations(from_parameter_sweep_result(sweep_result(result)))[0][2]
    assert 'unavailable' in first and 'Other configured' not in first


def test_summary_selected_member_and_partial_render_failure(app,monkeypatch):
    w=Workspace();run=from_parameter_sweep_result(sweep_result())
    w.set_run_data(run);w.convergence_selector.setCurrentIndex(1);w.tabs.setCurrentWidget(w.console)
    w.set_run_data(run)
    assert w.convergence_selector.currentData()==1 and w.tabs.currentWidget() is w.console
    monkeypatch.setattr(w,'_format_diagnostics',lambda run:(_ for _ in ()).throw(RuntimeError('render failure')))
    with pytest.raises(RuntimeError):w.set_run_data(run)
    assert w.convergence_box.isHidden() and not w.convergence_text.toPlainText()
    assert 'update failed' in w.result_ownership.text() and w._displayed_attempt is None
    assert w.table_pane.table.rowCount()==0 and w.tabs.currentWidget() is w.console
    w.close()


@pytest.mark.parametrize('width',[1000,1200])
def test_coordinates_and_selector_are_separate_scale_entries_accessible(app,width):
    w=Workspace();w.resize(width,700);w.set_run_data(image_data());w.show();app.processEvents()
    pane=w.longitudinal_pane
    selector_y=pane.field_selector.mapTo(pane,QPoint(0,pane.field_selector.height())).y()
    coordinate_y=pane.position_label.mapTo(pane,QPoint()).y()
    assert selector_y <= coordinate_y
    for c in (pane.scale_controls,w.image_pane.scale_controls):
        gc.collect()
        flow = c.layout().itemAt(0).layout()
        assert flow.count() == 4
        assert all(flow.itemAt(i).widget().parentWidget() is c for i in range(4))
        assert flow.heightForWidth(c.width()) > 0
        assert c.lower.width()>=100 and c.upper.width()>=100
        assert c.lower.placeholderText()=='Minimum' and c.upper.placeholderText()=='Maximum'
        assert c.mode.currentData()=='auto'
    w.close()


def test_pr_labels_and_spin_presentation_do_not_change_request(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.lc.gui.panels.experiment_panel import ExperimentPanel
    w=PRMainWindow();before=w.build_request()
    assert any('Material time steps this run'==x.text() for x in w.findChildren(QLabel))
    assert 'additional' in w.evolution_panel.Nt.toolTip()
    for spin in w.findChildren(CompactDoubleSpinBox):spin.interpretText()
    assert w.build_request()==before
    panel=ExperimentPanel();assert all('LocalRunner' not in x.text() for x in panel.findChildren(QLabel))
    panel.close();w.close()


def test_launchplane_selector_full_text_and_original_guidance_accessible(app):
    from lcprop.gui.panels.beam_panel import BeamPanel
    w=BeamPanel();selectors=w.launch_plane_widget.findChildren(QComboBox)
    laser = w.launch_plane_widget.laser_combo
    assert 'same Laser name interfere coherently' in laser.toolTip()
    assert 'different Laser names are mutually incoherent' in laser.toolTip()
    assert not any(x.findData('focused_gaussian') >= 0 for x in selectors)
    assert 'Signed external tilt' in w.launch_plane_widget.theta_spin.toolTip()
    assert 'fixed phi' in w.launch_plane_widget.theta_spin.toolTip()
    assert 'grazing incidence is unsupported' in w.launch_plane_widget.theta_spin.toolTip()
    assert not any(x.findData('transverse_wavevector') >= 0 for x in selectors)
    assert w.minimumSizeHint().width()<1200
    w.close()


def test_help_is_progressive_and_retains_scientific_hold():
    topics={t.key:t.markdown for t in HELP_TOPICS}
    for key in ['results','slurm','fanning_scattering']:assert topics[key].count('### ')>=3
    assert 'SCIENTIFIC HOLD / unestablished' in topics['fanning_scattering']
    assert 'local_source' not in topics['slurm'] and 'clean deployable' in topics['slurm']
    assert all('**'+x+'**' in topics['results'] for x in ['Fields','Curves','Samples / Tables','Diagnostics','Request','Console'])


def test_member_summary_does_not_crowd_fields_and_details_retain_ownership(app):
    from PySide6.QtCore import QRect
    from lcprop.lc.gui.main_window import LCPropMainWindow
    window=LCPropMainWindow();window.resize(1200,760)
    window.tabs.setCurrentWidget(window.results_panel)
    w=window.results_panel.workspace
    w.begin_request('Run');w.set_run_data(from_parameter_sweep_result(sweep_result()))
    window.show();app.processEvents();app.processEvents()
    for child in (w.convergence_box,w.image_pane.image_view):
        assert w.contentsRect().contains(QRect(child.mapTo(w,QPoint()),child.size()))
    assert 'Field relative change failed' in w.convergence_selector.currentText()
    selected=w.tabs.currentIndex();w.convergence_details.click();app.processEvents()
    assert w.convergence_dialog.isVisible() and w.tabs.currentIndex()==selected
    assert 'required <' in w.convergence_text.toPlainText()
    assert w.convergence_ownership.text()=='Completed result'
    w.begin_request('Run')
    assert w.convergence_ownership.text().startswith('Previous result')
    w.set_run_data(RunData('empty'))
    assert not w.convergence_dialog.isVisible() and not w.convergence_text.toPlainText()
    window.close()

"""Presentation must describe existing policies without changing their inputs."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import asdict
import copy
import pytest
from PySide6.QtWidgets import QApplication, QFormLayout
from lcprop.pr.gui.presentation_guidance import unified_support_guidance, TD_ELECTRICAL_GUIDANCE
from lcprop.pr.gui.unified_controls import UnifiedClosurePanel
from lcprop.pr.unified.specs import UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, A7_CURRENT, OPEN_TRANSVERSE


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_support_uses_policy_not_a_second_table(monkeypatch):
    from lcprop.pr.unified import solver_specs
    calls = []
    def sentinel(backend, precision, closure):
        calls.append((backend, precision.identity, closure.identity))
        return 777
    monkeypatch.setattr(solver_specs, 'scalable_axis_limit', sentinel)
    text = unified_support_guidance()
    assert len(calls) == 16
    assert text.count('≤777') == 4
    assert '512' not in text
    assert 'independent y columns' in text and '384×32' in text
    assert 'direct/reference' in text and 'resource' in text


def test_current_guidance():
    text = unified_support_guidance()
    for value in ('2048', '1024', '96', '256'):
        assert value in text
    assert '≤512' not in text


@pytest.mark.parametrize('dimension,kind', [(1,UNBIASED),(1,FIXED_FIELD),(1,PRESCRIBED_CURRENT),
    (1,A7_CURRENT),(2,UNBIASED),(2,FIXED_FIELD),(2,PRESCRIBED_CURRENT),(2,OPEN_TRANSVERSE)])
def test_closure_presentation_preserves_values(app, dimension, kind):
    panel = UnifiedClosurePanel()
    panel.set_dimension(dimension)
    panel.condition.setCurrentIndex(panel.condition.findData(kind))
    panel.x.setValue(.123456789012);panel.y.setValue(.345678901234)
    before = asdict(panel.closure(.1))
    panel.refresh()
    assert asdict(panel.closure(.1)) == before
    assert panel.x.minimumWidth() >= panel.x.fontMetrics().horizontalAdvance('-123456789.123456789012')
    label = panel.form.labelForField(panel.x).text()
    if kind in (FIXED_FIELD, OPEN_TRANSVERSE):
        assert 'Mean internal field' in label and 'dimensionless' in label
        assert 'E_s b' in panel.x.toolTip() and '−∇ψ' in panel.x.toolTip()
    elif kind == PRESCRIBED_CURRENT:
        assert 'current' in label and 'Mean internal field' not in label
    elif kind == A7_CURRENT:
        assert 'Applied bias' in label and 'need not equal' in panel.x.toolTip()
    panel.close()


def test_beam_presentation_preserves_intent(app):
    from lcprop.gui.panels.beam_panel import BeamPanel
    from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
    panel = BeamPanel()
    w = panel.launch_plane_widget
    before = panel.beam_stack_definition
    w.w1_spin.setValue(4.5);w.w2_spin.setValue(7.25);w.psi_spin.setValue(23.)
    stack = panel.beam_stack_definition
    request = beam_stack_definition_to_lcprop(stack)
    assert request.channels[0].w1_um == 4.5 and request.channels[0].w2_um == 7.25
    for i, editor in enumerate((w.w1_spin,w.w2_spin),1):
        labels=[f.labelForField(editor) for f in w.findChildren(QFormLayout)]
        assert any(label is not None and label.text()==f'Principal radius {i}' for label in labels)
        assert '1/e²' in editor.toolTip() and 'beam-normal' in editor.toolTip()
    assert 'does not rotate an applied image' in w.psi_spin.toolTip()
    assert 'zero' in w.enabled_checkbox.toolTip() and 'compatibility' in w.enabled_checkbox.toolTip()
    assert panel.beam_stack_definition == stack
    assert before.beams[0].enabled
    panel.close()


def test_checkpoint_owner_preserves_existing_static_selection_eligibility(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.pr.transverse.continuation import TransverseTDCheckpoint
    # Structural presentation fixture: no solver or checkpoint validation invoked.
    checkpoint = TransverseTDCheckpoint(None, None,
        {'status':'cancelled','cumulative_completed_steps':12,'cumulative_time':.5}, 'fixture')
    w=PRMainWindow()
    from lcprop.pr.unified.integration import WORKFLOW_ID
    w.evolution_panel.set_workflow_id(WORKFLOW_ID)
    w.last_checkpoint=checkpoint
    record=copy.deepcopy(checkpoint.record)
    w._refresh_checkpoint_controls()
    assert w.last_checkpoint is checkpoint and checkpoint.record==record
    assert w.continue_button.isEnabled() and w.save_checkpoint_button.isEnabled()
    assert w.continue_button.text()=='Continue Full-transverse TD'
    text=w.checkpoint_owner_label.text()
    assert 'cancelled' in text and '12' in text and '0.5' in text
    assert 'cumulative steps' in text
    assert 'not the current Run PR' in w.checkpoint_owner_label.toolTip()
    w._background_running=True;w._refresh_checkpoint_controls()
    assert not w.continue_button.isEnabled() and not w.save_checkpoint_button.isEnabled()
    w._background_running=False;w.last_checkpoint=None;w._refresh_checkpoint_controls()
    assert not w.continue_button.isEnabled()
    assert 'Static has no commissioned' in w.checkpoint_owner_label.toolTip()
    assert 'Fast previews' in w.checkpoint_owner_label.toolTip()
    w.close()


def test_td_explanation_is_informational(app):
    from lcprop.pr.gui.evolution_panel import PREvolutionPanel
    from lcprop.pr.transverse.specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    panel=PREvolutionPanel();panel.set_workflow_id(PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW)
    assert TD_ELECTRICAL_GUIDANCE in panel.execution_guidance.text()
    assert 'b_x = b_y = 0' in panel.execution_guidance.text()
    assert 'not TD controls' in panel.execution_guidance.text()
    assert panel.unified_closure.isHidden()
    panel.close()

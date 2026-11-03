"""Presentation checks using accepted metadata; no scientific execution."""
import pytest
from PySide6.QtWidgets import QApplication
from lcprop.core.execution import RunProgress
from lcprop.pr.transverse.specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW

@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])

@pytest.mark.parametrize('workflow,coordinate,elapsed', [
    ('pr_timedependent', 12.5, 2.5),
    (PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW, 2.5, 0.),
])
def test_cumulative_time_uses_accepted_metadata(workflow, coordinate, elapsed):
    from lcprop.pr.gui.presentation_guidance import accepted_td_time_text
    p=RunProgress(workflow=workflow,status='running',completed_units=5,total_units=10,
        current_coordinate=coordinate,coordinate_name='material_time',coordinate_unit='normalized',
        elapsed_wall_time=9999.,segment_start_time=10.,cumulative_time=12.5,
        segment_elapsed_time=elapsed)
    text=accepted_td_time_text(p)
    assert text.startswith('PR cumulative material time τ = 12.5')
    assert 'segment τ = 2.5' in text and 'start τ = 10' in text
    assert '9999' not in text


def test_folder_and_multiline_beam_help(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    w=PRMainWindow()
    assert w.save_checkpoint_button.text()=='Save Checkpoint Folder…'
    from lcprop.pr.gui.beam_panel import make_pr_beam_panel
    p=make_pr_beam_panel(x_aperture_um=200,y_aperture_um=200)
    b=p.launch_plane_widget
    for spin in (b.w1_spin,b.w2_spin):
        assert spin.toolTip().count('\n')>=3
        assert '1/e field radius = 1/e² intensity radius' in spin.toolTip()
        assert 'zero roll' in spin.toolTip()
    assert '\n' in b.psi_spin.toolTip() and 'does not rotate an applied image' in b.psi_spin.toolTip()
    p.close();w.close()


def test_waiting_results_are_not_final_unavailability(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.products.data_model import RunData, Geometry
    w=PRMainWindow(); ws=w.results_panel.workspace
    ws.invalidate_products()
    assert 'not available yet' in ws.longitudinal_pane.no_data_label.text()
    for state in ('Completed result','State at stop/cancellation','Failed result'):
        ws.set_run_data(RunData(workflow='pr_timedependent',geometry=Geometry()),state=state)
        assert 'No longitudinal fields are retained' in ws.longitudinal_pane.no_data_label.text()
    w.close()


def test_current_tooltip_is_dimensionless_model_current(app):
    from lcprop.pr.gui.unified_controls import UnifiedClosurePanel
    from lcprop.pr.unified.specs import PRESCRIBED_CURRENT
    p=UnifiedClosurePanel();p.set_dimension(2)
    p.condition.setCurrentIndex(p.condition.findData(PRESCRIBED_CURRENT))
    before=p.closure(1.)
    text=p.x.toolTip()
    assert 'J = n I E − ∇(n I)' in text and 'spatial mean' in text
    assert 'dark/background' in text and 'not an SI current' in text
    assert p.closure(1.)==before
    p.close()


def test_transverse_progress_shows_cumulative_time_in_status(app):
    from lcprop.pr.gui.main_window import PRMainWindow
    w=PRMainWindow()
    p=RunProgress(workflow=PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,status='running',
        completed_units=2,total_units=5,current_coordinate=.5,coordinate_name='material_time',
        coordinate_unit='normalized',elapsed_wall_time=123.,segment_start_time=2.,cumulative_time=2.5)
    w._on_progress(p)
    assert 'cumulative material time τ = 2.5' in w.status_label.text()
    assert 'segment τ = 0.5' in w.status_label.text()
    w.close()


def test_solver_help_distinguishes_physics(app):
    from lcprop.pr.gui.evolution_panel import PREvolutionPanel
    p=PREvolutionPanel()
    assert 'same physical model' in p.unified_solver.toolTip()
    assert 'bounded validation' in p.unified_solver.toolTip()
    p.close()


def test_tangent_help_does_not_call_model_legacy(app):
    from lcprop.pr.gui.evolution_panel import PREvolutionPanel
    from lcprop.pr.transverse.specs import PR_MATERIAL_RESPONSE_LINEARIZED
    p=PREvolutionPanel()
    p.transport_model.setCurrentIndex(p.transport_model.findData('full_transverse'))
    p.material_response.setCurrentIndex(p.material_response.findData(PR_MATERIAL_RESPONSE_LINEARIZED))
    text=p.material_response.toolTip()
    assert 'linearized material model' in text and 'reference intensity I0' in text
    assert 'legacy' not in text.lower()
    p.close()

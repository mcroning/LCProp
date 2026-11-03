"""New-session initialization and saved-request isolation."""
import json
from dataclasses import replace
import pytest
from PySide6.QtWidgets import QApplication
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.request_adapter import apply_pr_request
from lcprop.pr.experiment_codec import encode_pr_timedependent_request, decode_pr_timedependent_request
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V1, PR_CANONICAL_SCATTERING_V2

@pytest.fixture
def window():
    app=QApplication.instance() or QApplication([])
    w=PRMainWindow();yield w;w.close()


def test_fresh_and_scattering_enabled_request(window):
    r=window.build_request();e=window.evolution_panel
    assert r.material.dark_irradiance_W_cm2==.01
    assert r.material.uniform_irradiance_W_cm2==0.
    assert r.scattering is None
    assert (e.scattering_epsilon.value(),e.scattering_correlation_um.value(),e.scattering_seed.value())==(.02,.4,0)
    assert e.scattering_algorithm.currentData()==PR_CANONICAL_SCATTERING_V1
    e.scattering_enabled.setChecked(True);s=window.build_request().scattering
    assert (s.epsilon,s.transverse_correlation_um,s.realization_seed)==(.02,.4,0)
    assert s.algorithm_version==PR_CANONICAL_SCATTERING_V1


@pytest.mark.parametrize('physical',[False,True])
@pytest.mark.parametrize('algorithm',[None,PR_CANONICAL_SCATTERING_V1,PR_CANONICAL_SCATTERING_V2])
def test_saved_request_exact_roundtrip(window,physical,algorithm,record_property):
    r=window.build_request()
    material=(replace(r.material,dark_irradiance_W_cm2=0.,uniform_irradiance_W_cm2=.035)
              if physical else PRMaterialSpec(dark_intensity=.023,uniform_background_intensity=.13))
    s=None if algorithm is None else PRCanonicalScatteringSpec(.007,.8,37,2.,algorithm)
    r=replace(r,material=material,scattering=s,solver=replace(r.solver,Nt=17,dt_normalized=.0007))
    encoded=encode_pr_timedependent_request(r)
    restored=decode_pr_timedependent_request(json.loads(json.dumps(encoded)))
    apply_pr_request(restored,material_panel=window.material_panel,beam_panel=window.beam_panel,
        grid_panel=window.grid_panel,evolution_panel=window.evolution_panel)
    assert encode_pr_timedependent_request(window.build_request())==encoded
    record_property('saved_scientific_payload',json.dumps(encoded,sort_keys=True))


@pytest.mark.parametrize('field',['dark_irradiance','uniform_irradiance'])
def test_cleared_required_irradiance_rejected(window,field):
    getattr(window.material_panel,field).clear()
    with pytest.raises(ValueError,match='irradiance'):window.build_request()
    window.preview_request_clicked()
    assert 'irradiance' in window.results_panel.workspace.operation_status.text()


def test_incompatible_mode_stays_explicit(window):
    from lcprop.pr.static_workflow import PR_STATIC_WORKFLOW
    from lcprop.pr.illumination import INTEGRAL_NORMALIZATION
    e=window.evolution_panel;e.legacy_static=True
    e.workflow.setCurrentIndex(e.workflow.findData(PR_STATIC_WORKFLOW))
    # Directly select the historical model; do not migrate illumination.
    from lcprop.pr.gui.request_adapter import build_pr_request
    original=e.workflow_id;e.workflow_id=lambda: PR_STATIC_WORKFLOW
    try:
        with pytest.raises(ValueError,match='choose explicit legacy mode'):window.build_request()
    finally:e.workflow_id=original
    assert window.material_panel.normalization_mode.currentData()==INTEGRAL_NORMALIZATION


def test_api_defaults_not_redefined():
    from lcprop.pr.specs import PRSolverOptions
    assert PRMaterialSpec().dark_irradiance_W_cm2 is None
    assert PRMaterialSpec().normalization_identity=='pr_channel_peak_reference_v1'
    assert PRSolverOptions().dt_normalized==.001


def test_workflow_specific_startup_timestep_and_loaded_values(window):
    from lcprop.pr.specs import PRSolverOptions, PR_SEMI_IMPLICIT_INTEGRATOR
    e=window.evolution_panel
    assert window.build_request().solver.dt_normalized==.01
    e.transport_model.setCurrentIndex(e.transport_model.findData('full_transverse'))
    assert e.dt_normalized.value()==.001
    e.transport_model.setCurrentIndex(e.transport_model.findData('reduced_x'))
    assert e.dt_normalized.value()==.01
    e.set_solver(PRSolverOptions(Nt=7,dt_normalized=.01,integrator=PR_SEMI_IMPLICIT_INTEGRATOR))
    e.transport_model.setCurrentIndex(e.transport_model.findData('full_transverse'))
    assert e.dt_normalized.value()==.01  # Loaded value is not a startup default.
    e.dt_normalized.setValue(.0004)
    e.transport_model.setCurrentIndex(e.transport_model.findData('reduced_x'))
    assert e.dt_normalized.value()==.0004

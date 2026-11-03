"""Reduced physical restart preserves E and rebuilds its physical launch."""
from dataclasses import replace
import numpy as np
import pytest
from lcprop.pr.workflow import run_pr_timedependent, continue_pr_timedependent
from lcprop.pr.persistence import save_pr_checkpoint, load_pr_checkpoint
from lcprop.pr.specs import PR_SEMI_IMPLICIT_INTEGRATOR
from tests.test_pr_execution import _request
from tests.test_pr_physical_illumination import physical
from tests.test_pr_checkpoint import _assert_same_cumulative_physics


def request(precision='float64'):
    r = _request(steps=4)
    return replace(r, initial_A=None, material=physical(r.material),
                   backend=replace(r.backend, precision=precision),
                   solver=replace(r.solver, integrator=PR_SEMI_IMPLICIT_INTEGRATOR))


@pytest.mark.parametrize('precision', ['float64', 'float32'])
def test_physical_restart_exact_and_persisted(precision, tmp_path):
    r = request(precision)
    first = run_pr_timedependent(replace(r, solver=replace(r.solver, Nt=2)))
    save_pr_checkpoint(first.checkpoint, tmp_path)
    cp = load_pr_checkpoint(tmp_path)
    continuous = run_pr_timedependent(r)
    for checkpoint in (first.checkpoint, cp):
        resumed = continue_pr_timedependent(r, checkpoint, 2)
        _assert_same_cumulative_physics(resumed, continuous)
        assert resumed.diagnostics['source_normalization'] == continuous.diagnostics['source_normalization'] == first.diagnostics['source_normalization']
        assert resumed.checkpoint.request.beams == r.beams
        assert resumed.checkpoint.request.material == r.material
        np.testing.assert_array_equal(checkpoint.E_current, first.E_final)


def test_bare_prepared_launch_still_rejected():
    r = request()
    with pytest.raises(ValueError, match='prepared initial_A requires an explicit physical scale'):
        run_pr_timedependent(replace(r, initial_A=np.ones((1, 8, 6), complex)))


@pytest.mark.parametrize('field', ['dark_irradiance_W_cm2','uniform_irradiance_W_cm2'])
def test_preflight_requires_explicit_irradiances(field):
    from lcprop.pr.gui.request_adapter import validate_pr_gui_request
    r = request()
    with pytest.raises(ValueError, match=field.split("_")[0] + " irradiance"):
        validate_pr_gui_request(replace(r, material=replace(r.material, **{field:None})))
    validate_pr_gui_request(replace(r, material=physical(r.material, dark=0., uniform=0.)))


def test_gui_invalid_request_is_bounded_without_worker(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui import main_window as module
    app = QApplication.instance() or QApplication([])
    window = module.PRMainWindow()
    r = request()
    invalid = replace(r, material=replace(r.material, dark_irradiance_W_cm2=None))
    monkeypatch.setattr(window, 'build_request', lambda: invalid)
    messages=[]
    monkeypatch.setattr(module, 'report_failure', lambda owner,text: messages.append(text))
    window.run_clicked()
    assert window.status_label.text()=='Invalid request'
    assert len(messages)==1 and 'dark irradiance' in messages[0]
    assert 'Traceback' not in messages[0]
    assert not window._background_running
    window.close()


def test_continuation_cannot_bypass_prepared_array_guard_or_legacy_identity():
    r = request()
    first = run_pr_timedependent(replace(r, solver=replace(r.solver, Nt=1)))
    with pytest.raises(ValueError, match='prepared initial_A requires an explicit physical scale'):
        continue_pr_timedependent(replace(r, initial_A=first.A_initial), first.checkpoint, 1)
    legacy = _request(steps=1)
    old = run_pr_timedependent(legacy)
    with pytest.raises(ValueError, match='incompatible material'):
        continue_pr_timedependent(replace(legacy, initial_A=None, material=physical(legacy.material)), old.checkpoint, 1)

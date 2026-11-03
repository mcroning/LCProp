"""Sampling must see canonical pending edits before real GUI dispatch."""
import math

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from launchplane.model import BeamDefinition, BeamStackDefinition
from lcprop.gui.request_transparency import inspect_request
from lcprop.optics.boundaries import TransverseBoundarySpec
from tests.test_preview_state_lifecycle import host, snapshot
from tests.test_pending_beam_editor_sync import type_pending


@pytest.mark.parametrize('nx',[128,256])
def test_pending_theta_direct_run_and_inspection_block_new_carrier(host,monkeypatch,nx):
    app,w=host;e=w.beam_panel.launch_plane_widget
    w.material_panel.refractive_index.setValue(2.4)
    w.grid_panel.Nx.setValue(nx);w.grid_panel.Ny.setValue(256)
    w.beam_panel.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(w1_um=20,w2_um=20),)))
    w._validate_execution_request(w.build_request())
    requests=[]
    monkeypatch.setattr(w,'_start_background',lambda request,**kw:requests.append(request))
    w.run_button.setFocusPolicy(Qt.NoFocus)
    type_pending(app,e,e.theta_spin,'30')
    QTest.mouseClick(w.run_button,Qt.LeftButton)
    assert not requests
    assert e.beam_stack.beams[0].theta_ext_rad==math.pi/6
    snapshot(w)  # analytic footprint survives even a coarse preview grid
    status=w.results_panel.workspace.operation_status.text()
    assert 'Grid' in status and 'minimum Nx=316' in status
    inspect_request(w,w.build_request)
    assert 'Failed' in w.results_panel.workspace.operation_status.text()
    assert not requests
    w.grid_panel.Nx.setValue(512)
    inspect_request(w,w.build_request)
    assert 'Request validated; not executed' in w.results_panel.workspace.operation_status.text()
    QTest.mouseClick(w.run_button,Qt.LeftButton)
    assert len(requests)==1


@pytest.mark.parametrize('mode,word',[('periodic','wrap'),('sponge','attenuated')])
def test_boundary_warning_does_not_block_valid_run(host,monkeypatch,mode,word):
    app,w=host
    w.material_panel.refractive_index.setValue(2.4)
    w.grid_panel.Nx.setValue(512);w.grid_panel.Ny.setValue(256)
    w.grid_panel.z_length_um.setValue(1000)
    w.beam_panel.set_optical_boundary(TransverseBoundarySpec(mode=mode))
    w.beam_panel.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(w1_um=20,w2_um=20,theta_ext_rad=math.pi/6),)))
    request=w.build_request()
    assert word in w.describe_request(request)
    requests=[]
    monkeypatch.setattr(w,'_start_background',lambda request,**kw:requests.append(request))
    monkeypatch.setattr(w,'_local_run_cost_guard',lambda request:True)
    QTest.mouseClick(w.run_button,Qt.LeftButton)
    assert len(requests)==1


def test_lc_shared_preflight(host):
    from lcprop.lc.gui.main_window import LCPropMainWindow
    from PySide6.QtCore import QCoreApplication,QEvent
    app,_=host;w=LCPropMainWindow()
    try:
        w.grid_panel.x_aperture_um.setValue(200)
        w.grid_panel.Nx.setValue(128)
        w.beam_panel.set_beam_stack_definition(BeamStackDefinition(beams=(
            BeamDefinition(theta_ext_rad=math.pi/6),)))
        with pytest.raises(ValueError,match='minimum Nx=316'):
            w._validate_execution_request(w.build_request())
    finally:
        w.close();w.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)

"""Native edit/focus/Run ordering through shared Product request ownership."""
import math

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition,BeamStackDefinition
from tests.test_preview_state_lifecycle import host, snapshot, same
from tests.test_pr_gui_main_window import _wait_for


def type_pending(app,widget,spin,text):
    widget.inspector_scroll.ensureWidgetVisible(spin)
    spin.setFocus();app.processEvents()
    spin.selectAll();QTest.keyClicks(spin,text)
    assert spin.hasFocus() or spin.lineEdit().hasFocus()


def circle_to_thirty(before,after):
    np.testing.assert_allclose(np.linalg.eigvalsh(before[2][0][2]),[1/400,1/400],atol=1e-16)
    # Independent external incidence projection: radius 20/cos(30), not a fit.
    xy=after[2][0][3]
    np.testing.assert_allclose((xy[:,0]/(20/math.cos(math.pi/6)))**2+(xy[:,1]/20)**2,1,atol=2e-14)


@pytest.mark.parametrize('retain_editor_focus',[False,True])
def test_real_theta_typing_direct_run_synchronizes_before_worker(host,monkeypatch,retain_editor_focus):
    app,w=host;p=w.beam_panel;e=p.launch_plane_widget
    w.grid_panel.Nx.setValue(512);w.grid_panel.Ny.setValue(64)
    w.grid_panel.dz_um.setValue(10);w.grid_panel.z_length_um.setValue(20)
    w.material_panel.gain_length_product.setValue(0)
    p.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(w1_um=20,w2_um=20),)))
    before=snapshot(w);events=[];at_dispatch=[]
    e.beamStackChanged.connect(events.append)
    original=w._start_background
    def start(request,**kwargs):
        at_dispatch.append((request,snapshot(w)))
        original(request,**kwargs)
    monkeypatch.setattr(w,'_start_background',start)
    if retain_editor_focus:
        # Native push buttons may not take focus. Still click the real control.
        w.run_button.setFocusPolicy(Qt.NoFocus)
    type_pending(app,e,e.theta_spin,'30')
    assert '30' in e.theta_spin.text()
    same(before,snapshot(w));assert not events
    QTest.mouseClick(w.run_button,Qt.LeftButton)
    assert len(at_dispatch)==1
    request,committed=at_dispatch[0]
    assert request.beams.channels[0].theta_ext_rad==math.pi/6
    circle_to_thirty(before,committed)
    _wait_for(app,lambda:not w._background_running,timeout=30)
    assert w.run_status=='completed'
    assert len(events)==1
    same(committed,snapshot(w))
    w.tabs.setCurrentWidget(p);app.processEvents();same(committed,snapshot(w))
    result=w.last_result
    from lcprop.core.grid import make_grid
    grid=make_grid(request.grid)
    def centroid(a):
        intensity=np.abs(a[0])**2
        return np.array([(intensity*grid.x_um[:,None]).sum(),
                         (intensity*grid.y_um[None,:]).sum()])/intensity.sum()
    expected=20*.5/math.sqrt(2**2-.5**2)
    np.testing.assert_allclose(centroid(result.A_final)-centroid(result.A_initial),[expected,0],atol=.005,rtol=0)


@pytest.mark.parametrize('key',[Qt.Key_Tab,Qt.Key_Return])
def test_real_tab_return_converge_on_same_committed_preview(host,key):
    app,w=host;p=w.beam_panel;e=p.launch_plane_widget
    p.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(w1_um=20,w2_um=20),)))
    before=snapshot(w);events=[];e.beamStackChanged.connect(events.append)
    type_pending(app,e,e.theta_spin,'30');same(before,snapshot(w))
    QTest.keyClick(e.theta_spin,key);app.processEvents()
    assert e.beam_stack.beams[0].theta_ext_rad==math.pi/6
    if key==Qt.Key_Tab:assert e.phi_spin.hasFocus() or e.phi_spin.lineEdit().hasFocus()
    circle_to_thirty(before,snapshot(w));assert len(events)==1
    assert w.build_request().beams.channels[0].theta_ext_rad==math.pi/6
    assert len(events)==1


@pytest.mark.parametrize('control,text,field,value',[
    ('phi_spin','35','phi_rad',math.radians(35)),
    ('w1_spin','25','w1_um',25),('w2_spin','15','w2_um',15),
    ('x_spin','7','x0_um',7),('y_spin','-8','y0_um',-8),
    ('psi_spin','40','psi_rad',math.radians(40)),
    ('n_ext_spin','1.2','n_ext',1.2),('power_spin','0.5','power_mW',.5),
    ('wavelength_spin','0.7','wavelength_um',.7),('phase_spin','0.4','phase_rad',.4)])
def test_direct_run_flushes_all_physical_edit_categories(host,monkeypatch,control,text,field,value):
    app,w=host;p=w.beam_panel;e=p.launch_plane_widget
    p.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(w1_um=20,w2_um=10,theta_ext_rad=.1),)))
    requests=[]
    # Dispatch boundary only; actual worker execution is covered above.
    def dispatch(request,**kwargs):
        requests.append(request)
        snapshot(w)  # Every rendered contour must match the committed model.
    monkeypatch.setattr(w,'_start_background',dispatch)
    w.run_button.setFocusPolicy(Qt.NoFocus)
    type_pending(app,e,getattr(e,control),text)
    QTest.mouseClick(w.run_button,Qt.LeftButton)
    assert len(requests)==1
    assert getattr(requests[0].beams.channels[0],field)==pytest.approx(value)
    model_field={'x0_um':'x_um','y0_um':'y_um'}.get(field,field)
    assert getattr(e.beam_stack.beams[0],model_field)==pytest.approx(value)


@pytest.mark.parametrize('invalid',['-','90','100'])
def test_invalid_theta_cannot_be_fixed_up_before_run_validation(host,monkeypatch,invalid):
    app,w=host;p=w.beam_panel;e=p.launch_plane_widget
    p.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(w1_um=20,w2_um=20),)))
    original=e.beam_stack;calls=[]
    monkeypatch.setattr(w,'_start_background',lambda *a,**k:calls.append(a))
    type_pending(app,e,e.theta_spin,'-')
    if invalid!='-':
        # Qt prevents typing a fully out-of-range value; also guard text supplied
        # programmatically (e.g. accessibility) without a silent native fixup.
        e.theta_spin.lineEdit().setText(invalid)
    QTest.mouseClick(w.run_button,Qt.LeftButton);app.processEvents()
    assert not calls and not w._background_running
    assert e.beam_stack==original
    assert invalid in e.theta_spin.text()
    assert w.status_label.text()=='Invalid request'


def test_lc_uses_same_pending_edit_boundary_and_rejects_invalid_text():
    from lcprop.lc.gui.main_window import LCPropMainWindow
    app=QApplication.instance() or QApplication([]);w=LCPropMainWindow()
    try:
        w.show();w.tabs.setCurrentWidget(w.beam_panel);app.processEvents()
        e=w.beam_panel.launch_plane_widget
        type_pending(app,e,e.w1_spin,'25')
        assert w.build_request().beams.channels[0].w1_um==25
        assert e.beam_stack.beams[0].w1_um==25
        type_pending(app,e,e.theta_spin,'-')
        with pytest.raises(ValueError,match='Invalid pending beam edit'):
            w.build_request()
    finally:
        w.close();w.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()


@pytest.mark.parametrize('bad,other,other_text,repair', [
    ('theta_spin', 'phi_spin', '35', '30'),
    ('w1_spin', 'x_spin', '7', '25'),
    ('x_spin', 'psi_spin', '40', '8'),
    ('n_ext_spin', 'w2_spin', '15', '1.2'),
])
def test_invalid_edit_survives_other_commit_and_run_then_recovers(host, monkeypatch, bad, other, other_text, repair):
    app, w = host
    e = w.beam_panel.launch_plane_widget
    invalid = getattr(e, bad)
    calls = []
    monkeypatch.setattr(w, '_start_background', lambda request, **kwargs: calls.append(request))
    type_pending(app, e, invalid, '-')
    if invalid.minimum() >= 0:
        invalid.selectAll(); QTest.keyClick(invalid, Qt.Key_Backspace)
    assert not invalid.hasAcceptableInput()
    unresolved = invalid.text()
    QTest.keyClick(invalid, Qt.Key_Tab)
    type_pending(app, e, getattr(e, other), other_text)
    QTest.keyClick(getattr(e, other), Qt.Key_Return)
    app.processEvents()
    assert invalid.text() == unresolved
    assert getattr(e, other).value() == float(other_text)
    snapshot(w)
    QTest.mouseClick(w.run_button, Qt.LeftButton)
    assert not calls and w.status_label.text() == 'Invalid request'
    assert invalid.text() == unresolved
    assert invalid.accessibleName() in w.results_panel.workspace.operation_status.text()
    w.tabs.setCurrentWidget(w.beam_panel)
    type_pending(app, e, invalid, repair)
    QTest.mouseClick(w.run_button, Qt.LeftButton)
    assert len(calls) == 1
    snapshot(w)


def test_unselected_invalid_beam_blocks_run_after_host_rebuild_and_preview_refresh(host, monkeypatch):
    app, w = host
    p = w.beam_panel
    e = p.launch_plane_widget
    p.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(name='first', w1_um=20, w2_um=20),
        BeamDefinition(name='second', x_um=20, w1_um=20, w2_um=20))))
    type_pending(app, e, e.theta_spin, '-')
    unresolved = e.theta_spin.text()
    QTest.keyClick(e.theta_spin, Qt.Key_Tab)
    e.object_list.setCurrentRow(1)
    type_pending(app, e, e.phi_spin, '35')
    QTest.keyClick(e.phi_spin, Qt.Key_Return)
    for refresh in (lambda: p.set_aperture(180, 180), p._boundary_changed,
                    e.fit_button.click, e.full_aperture_button.click,
                    lambda: w.resize(1000, 720)):
        refresh(); app.processEvents()
    e.object_list.setCurrentRow(0)
    assert e.theta_spin.text() == unresolved
    assert e.beam_stack.beams[0].theta_ext_rad == 0
    assert e.beam_stack.beams[1].phi_rad == math.radians(35)
    snapshot(w)
    e.object_list.setCurrentRow(1)
    calls = []
    monkeypatch.setattr(w, '_start_background', lambda request, **kwargs: calls.append(request))
    QTest.mouseClick(w.run_button, Qt.LeftButton)
    assert not calls and w.status_label.text() == 'Invalid request'
    assert e.object_list.currentRow() == 0
    assert e.theta_spin.text() == unresolved
    assert 'External theta' in w.results_panel.workspace.operation_status.text()
    w.tabs.setCurrentWidget(p)
    type_pending(app, e, e.theta_spin, '30')
    QTest.mouseClick(w.run_button, Qt.LeftButton)
    assert len(calls) == 1
    assert calls[0].beams.channels[0].theta_ext_rad == math.pi/6
    assert calls[0].beams.channels[1].phi_rad == math.radians(35)
    snapshot(w)


def test_invalid_center_survives_aperture_range_refresh(host, monkeypatch):
    app, w = host
    p = w.beam_panel
    e = p.launch_plane_widget
    type_pending(app, e, e.x_spin, '-')
    unresolved = e.x_spin.text()
    p.set_aperture(180, 180)
    app.processEvents()
    assert e.x_spin.text() == unresolved
    calls = []
    monkeypatch.setattr(w, '_start_background', lambda request, **kwargs: calls.append(request))
    QTest.mouseClick(w.run_button, Qt.LeftButton)
    assert not calls and w.status_label.text() == 'Invalid request'

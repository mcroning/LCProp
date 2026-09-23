"""Independent signed physical launch and production host presentation checks."""
from dataclasses import replace
import math
import numpy as np
import pytest
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.splitstep import scalar_angular_spectrum_kernel, hop_linear


@pytest.mark.parametrize('theta,phi', [(20,0),(-20,0),(20,90),(-20,90),(-17,137)])
def test_launch_to_scalar_propagation_sign(theta, phi):
    theta, phi = map(math.radians, (theta, phi))
    n, wavelength, length = 2., .633, 20.
    grid = make_grid(GridSpec(Nx=384, Ny=384, x_aperture_um=120.,
                             y_aperture_um=120., z_length_um=length, dz_um=length))
    beam = BeamChannel(theta_ext_rad=theta, phi_rad=phi, w1_um=12., w2_um=12.)
    launch = build_launch(BeamStack(channels=(beam,)), grid,
                         context=OpticalLaunchContext(grid,n,length))
    f = np.fft.fftfreq(grid.Nx, d=grid.dx_um)
    kernel = scalar_angular_spectrum_kernel(f[:,None]**2+f[None,:]**2,
        dz=length, wavelength=wavelength, n_ref=n)
    final = hop_linear(launch.A0, kernel)
    def centroid(a):
        i = abs(a[0])**2
        return np.array([(i*grid.x_um[:,None]).sum(),(i*grid.y_um[None,:]).sum()])/i.sum()
    # Independent external intent -> tangential k -> positive internal kz.
    kx = 2*math.pi/wavelength*math.sin(theta)*math.cos(phi)
    ky = 2*math.pi/wavelength*math.sin(theta)*math.sin(phi)
    kz = math.sqrt((2*math.pi*n/wavelength)**2-kx*kx-ky*ky)
    expected = length*np.array([kx,ky])/kz
    np.testing.assert_allclose(centroid(final)-centroid(launch.A0), expected, atol=0.001, rtol=0.001)


@pytest.fixture
def host():
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.main_window import PRMainWindow
    app = QApplication.instance() or QApplication([])
    window = PRMainWindow()
    window.resize(1200,900)
    window.show(); app.processEvents()
    try:
        yield app, window
    finally:
        window.close(); window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete); app.processEvents()


@pytest.mark.parametrize('theta,phi', [(20,0),(-20,0),(20,90),(-20,90),(-17,137)])
def test_real_host_ray_tracks_signed_intent_material_and_length(host,theta,phi):
    from launchplane.model import BeamDefinition, BeamStackDefinition
    app, window = host
    panel = window.beam_panel
    window.grid_panel.z_length_um.setValue(20)
    window.material_panel.refractive_index.setValue(2)
    panel.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(
        name='test',x_um=3,y_um=-4,theta_ext_rad=math.radians(theta),phi_rad=math.radians(phi),w1_um=12,w2_um=12),)))
    widget = panel.launch_plane_widget
    def expected(index,length):
        sx = math.sin(math.radians(theta))*math.cos(math.radians(phi))
        sy = math.sin(math.radians(theta))*math.sin(math.radians(phi))
        return length*np.array([sx,sy])/math.sqrt(index**2-sx*sx-sy*sy)
    def ray():
        p = widget.scene.beam_items[0].tilt_tip_offset()
        return [p.x(),-p.y()]
    np.testing.assert_allclose(ray(),expected(2,20),atol=1e-12)
    before = widget._resolved_items[0].path()
    external_k = widget.beam_stack.beams[0].transverse_wavevector_rad_per_um
    window.material_panel.refractive_index.setValue(2.4)
    np.testing.assert_allclose(ray(),expected(2.4,20),atol=1e-12)
    assert widget._resolved_items[0].path() == before
    assert widget.beam_stack.beams[0].transverse_wavevector_rad_per_um == external_k
    window.grid_panel.z_length_um.setValue(40)
    np.testing.assert_allclose(ray(),expected(2.4,40),atol=1e-12)
    assert widget.beam_stack.beams[0].theta_ext_rad == math.radians(theta)
    assert widget.canvas_title.text() == 'Input face'
    assert widget.scene.beam_items[0]._label.text() == '1'


def test_aperture_controls_edit_real_grid_and_roundtrip_signed_request(host):
    from launchplane.model import BeamDefinition, BeamStackDefinition
    from lcprop.pr.experiment_codec import encode_pr_timedependent_request, decode_pr_timedependent_request
    app, window = host
    panel = window.beam_panel
    panel.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(theta_ext_rad=-.2),)))
    assert panel.x_aperture_control.keyboardTracking() == window.grid_panel.x_aperture_um.keyboardTracking()
    panel.x_aperture_control.lineEdit().setText("160 µm")
    panel.y_aperture_control.setValue(120)
    assert window.grid_panel.x_aperture_um.value() == 160
    assert window.grid_panel.y_aperture_um.value() == 120
    assert panel.launch_plane_definition.x_aperture_um == 160
    assert panel.launch_plane_definition.y_aperture_um == 120
    window.grid_panel.x_aperture_um.setValue(180)
    assert panel.x_aperture_control.value() == 180
    request = window.build_request()
    assert request.grid.x_aperture_um == 180
    assert request.grid.y_aperture_um == 120
    assert request.beams.channels[0].theta_ext_rad == -.2
    # Codec's schema already stores the signed float verbatim.
    payload = encode_pr_timedependent_request(request)
    restored = decode_pr_timedependent_request(payload)
    assert restored.beams.channels[0].theta_ext_rad == -.2
    assert restored.grid == request.grid


def test_disabled_beam_does_not_shift_host_ray_indices(host):
    from launchplane.model import BeamDefinition, BeamStackDefinition
    app, window = host
    panel = window.beam_panel
    window.grid_panel.z_length_um.setValue(20)
    panel.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(name='disabled',enabled=False,theta_ext_rad=.2),
        BeamDefinition(name='negative',theta_ext_rad=-.2),
        BeamDefinition(name='positive',theta_ext_rad=.2))))
    a,b,c = panel.launch_plane_widget.scene.beam_items
    assert a._resolved_tip is None
    assert b.tilt_tip_offset().x() < 0 < c.tilt_tip_offset().x()
    assert len(panel.launch_plane_widget._resolved_items) == 2


@pytest.mark.parametrize('theta,phi', [(20,0),(-20,0),(20,90),(-20,90),(-17,137)])
def test_ordinary_pr_workflow_centroid_matches_physical_ray(theta,phi):
    from lcprop.pr.static_workflow import PRStaticRunRequest, run_pr_static
    from lcprop.pr.specs import PRMaterialSpec
    from lcprop.core.backend import BackendSpec
    theta,phi = map(math.radians,(theta,phi))
    length,n = 20.,2.
    request = PRStaticRunRequest(
        grid=GridSpec(Nx=256,Ny=256,x_aperture_um=100.,y_aperture_um=100.,
                      z_length_um=length,dz_um=length),
        beams=BeamStack(channels=(BeamChannel(theta_ext_rad=theta,phi_rad=phi,w1_um=12,w2_um=12),)),
        material=PRMaterialSpec(refractive_index=n,gain_length_product=0.,applied_field=0.),
        backend=BackendSpec(backend='numpy',precision='float64',verbose=False))
    result=run_pr_static(request)
    grid=make_grid(request.grid)
    def centroid(a):
        i=abs(a[0])**2
        return np.array([(i*grid.x_um[:,None]).sum(),(i*grid.y_um[None,:]).sum()])/i.sum()
    sx,sy=math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi)
    expected=length*np.array([sx,sy])/math.sqrt(n*n-sx*sx-sy*sy)
    np.testing.assert_allclose(centroid(result.A_final)-centroid(result.A_initial),expected,atol=.001,rtol=.001)
    assert result.converged


def test_lc_aperture_and_context_use_same_grid():
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from lcprop.lc.gui.main_window import LCPropMainWindow
    from launchplane.model import BeamDefinition, BeamStackDefinition
    app=QApplication.instance() or QApplication([])
    window=LCPropMainWindow()
    try:
        panel=window.beam_panel
        panel.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(theta_ext_rad=-.2),)))
        window.grid_panel.z_length_um.setValue(30)
        window.physics_panel.no.setValue(1.7)
        panel.x_aperture_control.setValue(180)
        panel.y_aperture_control.setValue(130)
        request=window.build_request()
        assert request.grid.x_aperture_um==180 and request.grid.y_aperture_um==130
        expected=30*math.sin(-.2)/math.sqrt(1.7**2-math.sin(-.2)**2)
        assert panel.launch_plane_widget.scene.beam_items[0].tilt_tip_offset().x()==pytest.approx(expected)
    finally:
        window.close();window.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()


def test_real_viewport_reciprocal_crossing_through_product_inverse_and_propagation(host):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtTest import QTest
    from launchplane.model import BeamDefinition, BeamStackDefinition
    from lcprop.adapters.launchplane import beam_definition_to_channel
    app,window=host
    window.tabs.setCurrentWidget(window.beam_panel)
    window.grid_panel.z_length_um.setValue(100)
    window.material_panel.refractive_index.setValue(1.5)
    panel=window.beam_panel
    panel.x_aperture_control.setValue(140);panel.y_aperture_control.setValue(140)
    panel.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(name='A',x_um=-10,y_um=-4,w1_um=8,w2_um=8),
        BeamDefinition(name='B',x_um=10,y_um=4,w1_um=8,w2_um=8))))
    widget=panel.launch_plane_widget
    widget.view.fit_aperture();app.processEvents()
    initial=widget.beam_stack
    for index in (0,1):
        item=widget.scene.beam_items[index]
        other=widget.scene.beam_items[1-index]
        start=widget.view.mapFromScene(item._ray_head.scenePos())
        end=widget.view.mapFromScene(other._center_marker.scenePos())
        QTest.mousePress(widget.view.viewport(),Qt.LeftButton,pos=start)
        assert widget.scene.mouseGrabberItem() is item._ray_head
        for fraction in (.25,.5,.75,1.):
            position=start+QPoint(round((end.x()-start.x())*fraction),round((end.y()-start.y())*fraction))
            QTest.mouseMove(widget.view.viewport(),position);app.processEvents()
            assert widget.scene.mouseGrabberItem() is item._ray_head
        QTest.mouseRelease(widget.view.viewport(),Qt.LeftButton,pos=end)
        app.processEvents()
        # Pixel quantization, including the press offset, bounds manual placement.
        tolerance=1.1/widget.view.transform().m11()
        assert item._ray_head.scenePos().x() == pytest.approx(other.pos().x(),abs=tolerance)
        assert item._ray_head.scenePos().y() == pytest.approx(other.pos().y(),abs=tolerance)
        assert widget.theta_spin.value() == pytest.approx(math.degrees(widget.beam_stack.beams[index].theta_ext_rad),abs=1e-6)
    for index,beam in enumerate(widget.beam_stack.beams):
        assert replace(beam,theta_ext_rad=0,phi_rad=0) == initial.beams[index]
        # Independently propagate each UI-edited narrow beam. No material response,
        # no ray-coordinate helper supplies the measured centroid.
        grid=make_grid(GridSpec(Nx=512,Ny=512,x_aperture_um=140,y_aperture_um=140,z_length_um=100,dz_um=100))
        launch=build_launch(BeamStack(channels=(beam_definition_to_channel(beam),)),grid,
                           context=OpticalLaunchContext(grid,1.5,100))
        f=np.fft.fftfreq(grid.Nx,d=grid.dx_um)
        kernel=scalar_angular_spectrum_kernel(f[:,None]**2+f[None,:]**2,dz=100,wavelength=.633,n_ref=1.5)
        final=hop_linear(launch.A0,kernel)
        intensity=abs(final[0])**2
        centroid=np.array([(intensity*grid.x_um[:,None]).sum(),(intensity*grid.y_um[None,:]).sum()])/intensity.sum()
        tip=widget.scene.beam_items[index]._ray_head.scenePos()
        np.testing.assert_allclose(centroid,[tip.x(),-tip.y()],atol=.006,rtol=0)
        opposite=initial.beams[1-index]
        np.testing.assert_allclose(centroid,[opposite.x_um,opposite.y_um],atol=tolerance+.006,rtol=0)
    # A real host rejects physically unreachable motion without altering intent.
    before=widget.beam_stack
    item=widget.scene.beam_items[1]
    start=widget.view.mapFromScene(item._ray_head.scenePos())
    end=widget.view.mapFromScene(QPointF(2000,0))
    QTest.mousePress(widget.view.viewport(),Qt.LeftButton,pos=start)
    QTest.mouseMove(widget.view.viewport(),end)
    QTest.mouseRelease(widget.view.viewport(),Qt.LeftButton,pos=end)
    assert widget.beam_stack == before
    assert 'unreachable' in widget.execution_status.text()

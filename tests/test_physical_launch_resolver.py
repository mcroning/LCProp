"""Independent analytic checks of external physical intent and interface flux."""
from dataclasses import replace
import math

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import OpticalLaunchContext, build_launch, channel_power_integrals
from lcprop.optics.physical_launch import resolve_beam_geometry


def launch(beam, *, aperture=80., n=2., dtype=np.complex128):
    grid = make_grid(GridSpec(Nx=512, Ny=384, x_aperture_um=aperture,
                             y_aperture_um=aperture, z_length_um=1.), real_dtype=np.float64)
    result = build_launch(BeamStack(channels=(beam,)), grid, complex_dtype=dtype,
                         context=OpticalLaunchContext(grid, n, 1.))
    return grid, result


@pytest.mark.parametrize('phi', [0., .47, 2.3])
def test_snell_footprint_internal_radii_and_external_grating(phi):
    theta, n_ext, n_int, w = .6, 1., 2.2, 4.
    beam = BeamChannel(theta_ext_rad=theta, phi_rad=phi, n_ext=n_ext, w1_um=w, w2_um=w)
    g = resolve_beam_geometry(beam, n_int)
    assert n_int*math.sin(g.theta_internal) == pytest.approx(n_ext*math.sin(theta))
    k0 = 2*math.pi/beam.wavelength_um
    assert [g.kx, g.ky] == pytest.approx([k0*math.sin(theta)*math.cos(phi), k0*math.sin(theta)*math.sin(phi)])
    direction = np.array([math.cos(phi), math.sin(phi)])
    perpendicular = np.array([-math.sin(phi), math.cos(phi)])
    assert direction@g.interface_quadratic@direction == pytest.approx((math.cos(theta)/w)**2)
    assert perpendicular@g.interface_quadratic@perpendicular == pytest.approx(1/w**2)
    radii = sorted(1/np.sqrt(np.linalg.eigvalsh(g.internal_quadratic)))
    assert radii == pytest.approx(sorted([w, w*math.cos(g.theta_internal)/math.cos(theta)]))
    opposite = resolve_beam_geometry(replace(beam, phi_rad=phi+math.pi), n_int)
    period = 2*math.pi/math.hypot(g.kx-opposite.kx, g.ky-opposite.ky)
    assert period == pytest.approx(beam.wavelength_um/(2*n_ext*math.sin(theta)))
    assert math.hypot(g.kx/g.kz_internal, g.ky/g.kz_internal) == pytest.approx(math.tan(g.theta_internal))


@pytest.mark.parametrize('theta', [0., .5, 1.0])
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_analytic_flux_normalization_without_sampled_renormalization(theta, dtype):
    grid, result = launch(BeamChannel(power_mW=3., theta_ext_rad=theta, w1_um=4., w2_um=5.), dtype=dtype)
    cosine = math.sqrt(1-(math.sin(theta)/2)**2)
    assert result.A0.dtype == dtype
    norm = channel_power_integrals(result.A0, grid)[0]
    assert cosine*norm == pytest.approx(1., rel=2e-6)
    assert result.power_metadata['captured_pre_screen_mW'] == pytest.approx([3.], rel=2e-6)
    assert norm == pytest.approx(1/cosine, rel=2e-6)


def test_half_aperture_is_lost_and_not_renormalized():
    grid, result = launch(BeamChannel(x0_um=40., w1_um=4., w2_um=4.))
    assert result.power_metadata['capture_fraction'][0] == pytest.approx(.5, abs=1e-12)
    assert channel_power_integrals(result.A0, grid)[0] == pytest.approx(.5, abs=1e-12)
    assert result.power_metadata['capture_warnings'][0]


def test_roll_circular_invariance_and_elliptical_rotation():
    b = BeamChannel(theta_ext_rad=.4, phi_rad=.7, w1_um=4., w2_um=4.)
    a = resolve_beam_geometry(b, 2.)
    r = resolve_beam_geometry(replace(b, psi_rad=.83), 2.)
    np.testing.assert_allclose(a.interface_quadratic, r.interface_quadratic, atol=1e-16)
    b = replace(b, theta_ext_rad=0, w2_um=8., psi_rad=math.pi/2)
    np.testing.assert_allclose(resolve_beam_geometry(b, 2.).interface_quadratic,
                               np.diag([1/64, 1/16]), atol=1e-16)


def test_phase_reference_and_no_double_carrier():
    b = BeamChannel(theta_ext_rad=.2, phi_rad=.4, x0_um=1., y0_um=-2., phase_rad=.7)
    grid, result = launch(b)
    g = result.resolved_geometry[0]
    x, y = grid.x_um[:, None]-b.x0_um, grid.y_um[None, :]-b.y0_um
    expected = np.exp(1j*(g.kx*x+g.ky*y+b.phase_rad))
    mask = np.abs(result.A0[0]) > 1e-8
    np.testing.assert_allclose((result.A0[0]/np.maximum(np.abs(result.A0[0]), 1e-300))[mask], expected[mask], atol=1e-14)
    _, rotated = launch(replace(b, phase_rad=1.2))
    np.testing.assert_allclose(abs(result.A0), abs(rotated.A0), atol=1e-16)


def test_invalid_mode_and_obsolete_constructor_rejected():
    with pytest.raises(ValueError, match='no forward'):
        resolve_beam_geometry(BeamChannel(n_ext=2., theta_ext_rad=1.), 1.)
    with pytest.raises(TypeError):
        BeamChannel(tilt_x_rad_per_um=.1)
    with pytest.raises(ValueError, match='focused launch is deferred'):
        BeamChannel(profile='focused_gaussian').validate()


def test_carrier_units_gate_preserves_partition_and_endpoint_ratios():
    from lcprop.pr.carrier_power import carrier_power_diagnostic_from_summary
    grid = make_grid(GridSpec(Nx=128, Ny=48, x_aperture_um=64., y_aperture_um=48.))
    x, y = grid.x_um[:, None], grid.y_um[None, :]
    envelope = np.exp(-(x/9)**2-(y/7)**2)
    q = 2*np.pi*8/64
    initial = np.stack([envelope*np.exp(1j*q*x), envelope*np.exp(-1j*q*x)])
    final = initial*np.array([.8, 1.2])[:, None, None]
    summary = {'coherence_groups': ['g', 'g'], 'physical_total_power_mW': 4.,
               'carrier_channels': [{'name': 'a', 'kx_rad_per_um': q, 'ky_rad_per_um': 0.},
                                    {'name': 'b', 'kx_rad_per_um': -q, 'ky_rad_per_um': 0.}]}
    legacy = carrier_power_diagnostic_from_summary(initial, final, grid_summary=grid.summary(), launch_summary=summary)
    physical = carrier_power_diagnostic_from_summary(initial, final, grid_summary=grid.summary(),
        launch_summary={**summary, 'field_normalization': 'physical_irradiance_carrier_cosine_v1',
                        'power_normalization': {'power_scale_mW': 4.}})
    for key in ('carrier_power_input', 'carrier_power_output', 'carrier_gain',
                'carrier_partition_method', 'carrier_separation_quality',
                'carrier_power_balance_error', 'parseval_partition_error_input'):
        assert physical[key] == legacy[key]
    assert physical['carrier_power_input_mW'] is None
    assert physical['physical_power_unit'] is None
    assert physical['physical_power_availability'] == 'unavailable'
    assert 'angular flux weighting' in physical['physical_power_unavailable_reason']
    assert physical['power_normalization'] == {'power_scale_mW': 4.}


def test_arbitrary_elliptical_roll_uses_full_external_frame():
    b = BeamChannel(theta_ext_rad=.7, phi_rad=.8, psi_rad=.4, w1_um=3., w2_um=8.)
    g = resolve_beam_geometry(b, 1.7)
    c, s = math.cos(.7), math.sin(.7)
    u, v = math.cos(.8), math.sin(.8)
    e1 = np.array([1-(1-c)*u*u, -(1-c)*u*v, -s*u])
    e2 = np.array([-(1-c)*u*v, 1-(1-c)*v*v, -s*v])
    a = math.cos(.4)*e1+math.sin(.4)*e2
    d = -math.sin(.4)*e1+math.cos(.4)*e2
    for x, y in [(2., -3.), (-1., 4.), (5., 7.)]:
        point = np.array([x, y, 0.])
        expected_exponent = (a@point/3)**2+(d@point/8)**2
        assert np.array([x,y])@g.interface_quadratic@np.array([x,y]) == pytest.approx(expected_exponent)
    # The internal form maps back to the same face envelope; radii are derived.
    ti = math.asin(math.sin(.7)/1.7)
    ci = math.cos(ti)
    mint = np.array([[1-(1-ci)*u*u, -(1-ci)*u*v],
                     [-(1-ci)*u*v, 1-(1-ci)*v*v]])
    np.testing.assert_allclose(mint.T@g.internal_quadratic@mint, g.interface_quadratic, atol=1e-16)


def test_oblique_screen_overlap_and_coherent_groups_are_not_renormalized():
    from lcprop.optics.screens import RasterSource, ScreenPlacement, IntensityRasterScreen, ChannelLaunchElements, prepare_intensity_raster_screen
    b = BeamChannel(theta_ext_rad=.35, phi_rad=.5, w1_um=4., w2_um=6., coherence_group='same')
    grid, one = launch(b)
    screen = IntensityRasterScreen(RasterSource.from_array(np.array([[0., .3], [.8, 1.]])),
        ScreenPlacement(center_x_um=1., center_y_um=-2., width_um=10., height_um=14., boundary_policy='reject'))
    t = prepare_intensity_raster_screen(screen, grid)
    result = build_launch(BeamStack(channels=(b,)), grid, complex_dtype=np.complex128,
        context=OpticalLaunchContext(grid, 2., 1.), launch_elements=(ChannelLaunchElements(0, (screen,)),))
    np.testing.assert_allclose(result.A0[0], one.A0[0]*np.sqrt(t), atol=1e-16)
    ci = math.sqrt(1-(math.sin(.35)/2)**2)
    expected = ci*np.sum(t*abs(one.A0[0])**2)*grid.dx_um*grid.dy_um
    assert result.post_element_physical_powers_mW[0] == pytest.approx(expected)
    assert expected < one.power_metadata['captured_pre_screen_mW'][0]
    together = build_launch(BeamStack(channels=(b, replace(b, name='second'))), grid,
        complex_dtype=np.complex128, context=OpticalLaunchContext(grid, 2., 1.))
    one_flux = one.power_metadata['coherent_group_scalar_flux']['same']['scalar_axial_current_mW']
    combined_flux = together.power_metadata['coherent_group_scalar_flux']['same']['scalar_axial_current_mW']
    assert combined_flux == pytest.approx(4*one_flux, rel=1e-13)


def test_aliased_carrier_cannot_receive_physical_power_qualification():
    grid = make_grid(GridSpec(Nx=16, Ny=16, x_aperture_um=80., y_aperture_um=80.))
    b = BeamChannel(theta_ext_rad=.8, w1_um=10., w2_um=10.)
    # An unresolved carrier is now rejected before a misleading field is built.
    with pytest.raises(ValueError, match="Optical launch sampling invalid"):
        build_launch(BeamStack(channels=(b,)), grid,
                     context=OpticalLaunchContext(grid, 2., 1.))

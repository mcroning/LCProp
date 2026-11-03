"""Independent analytic sampling and execution-boundary regressions."""
import math
from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.sampling import qualify_launch_sampling, strict_minimum_samples
from lcprop.optics.boundaries import TransverseBoundarySpec


def beam(**kwargs):
    return BeamChannel(w1_um=20, w2_um=20, theta_ext_rad=math.pi/6, **kwargs)


def grid(nx=512, ny=256, length=1000):
    return GridSpec(Nx=nx, Ny=ny, x_aperture_um=200, y_aperture_um=200,
                    z_length_um=length, dz_um=10)


@pytest.mark.parametrize('nx,valid', [(128,False),(256,False),(512,True)])
@pytest.mark.parametrize('sign', [-1,1])
def test_native_carrier(nx,valid,sign):
    b=replace(beam(),theta_ext_rad=sign*math.pi/6)
    r=qualify_launch_sampling(BeamStack((b,)),grid(nx),2.4)
    assert bool(r.errors) != valid
    if not valid:
        assert len(r.errors)==1 and 'x:' in r.errors[0]
        assert 'minimum Nx=316' in r.errors[0]
        for token in ('4.96302157','rad/µm','Lx=200','dx=','Grid'):
            if token=='Grid':
                with pytest.raises(ValueError,match=token):r.require_valid()
            else: assert token in r.errors[0]
    else:
        r.require_valid()
        assert not any('spectral' in w for w in r.warnings)


@pytest.mark.parametrize('phi,nx,ny,axes',[(math.pi/2,256,128,('y',)),
                                        (math.pi/4,128,128,('x','y'))])
def test_axes(phi,nx,ny,axes):
    r=qualify_launch_sampling(BeamStack((beam(phi_rad=phi),)),grid(nx,ny),2.4)
    assert len(r.errors)==len(axes)
    for axis in axes:assert any(f', {axis}:' in e for e in r.errors)


def test_multiple_coherent_beams():
    stack=BeamStack((beam(name='first'),beam(name='second',phi_rad=math.pi/2)),coherence='coherent')
    r=qualify_launch_sampling(stack,grid(128,128),2.4)
    assert len(r.errors)==2 and 'first' in r.errors[0] and 'second' in r.errors[1]


def test_strict_integer_minimum_and_nyquist_equality():
    limit=math.pi*128/200
    for k in (limit,-limit):
        assert strict_minimum_samples(k,200)==129
    assert strict_minimum_samples(math.nextafter(limit,0),200)==128
    assert strict_minimum_samples(math.nextafter(limit,math.inf),200)==129
    # Choose parameters giving an exactly representable resolved k=1.
    b=BeamChannel(wavelength_um=2*math.pi*math.sin(.3),theta_ext_rad=.3)
    g=GridSpec(Nx=128,Ny=128,x_aperture_um=128*math.pi)
    r=qualify_launch_sampling(BeamStack((b,)),g,2.4)
    assert r.errors


def test_covariance_roll_and_oblique_projection():
    # At theta=0, 90-degree roll exchanges spectral marginal widths.
    b=BeamChannel(w1_um=.4,w2_um=20)
    g=grid(128,128,10)
    r=qualify_launch_sampling(BeamStack((b,)),g,2.4)
    assert not r.errors
    assert any(', x: Gaussian' in w for w in r.warnings)
    assert not any(', y: Gaussian' in w for w in r.warnings)
    rolled=qualify_launch_sampling(BeamStack((replace(b,psi_rad=math.pi/2),)),g,2.4)
    assert any(', y: Gaussian' in w for w in rolled.warnings)
    assert not any(', x: Gaussian' in w for w in rolled.warnings)
    # Independent 45-degree rolled covariance at normal incidence: diagonal average.
    r=qualify_launch_sampling(BeamStack((replace(b,psi_rad=math.pi/4),)),g,2.4)
    expected=4*math.sqrt((1/.4**2+1/20**2)/2)
    assert sum(f'{expected:.9g}' in w for w in r.warnings)==2
    # Circular oblique footprint: spectrum narrows along the incidence direction.
    from lcprop.optics.physical_launch import resolve_beam_geometry
    b=beam(phi_rad=.7)
    q=resolve_beam_geometry(b,2.4).interface_quadratic
    u=np.array([math.cos(.7),math.sin(.7)])
    np.testing.assert_allclose(q,(np.eye(2)-math.sin(math.pi/6)**2*np.outer(u,u))/400,atol=1e-17)


@pytest.mark.parametrize('mode,word',[('periodic','wrap'),('sponge','attenuated'),('tukey','attenuated')])
def test_boundary_advisory_independent_of_sampling(mode,word):
    spec=TransverseBoundarySpec(mode=mode)
    r=qualify_launch_sampling(BeamStack((beam(),)),grid(),2.4,boundary=spec)
    r.require_valid()
    assert any(word in w for w in r.warnings)
    bad=qualify_launch_sampling(BeamStack((beam(),)),grid(128),2.4,boundary=spec)
    assert bad.errors and any(word in w for w in bad.warnings)
    safe=qualify_launch_sampling(BeamStack((replace(beam(),theta_ext_rad=0),)),grid(length=10),2.4,boundary=spec)
    assert not safe.warnings


def test_builder_cannot_bypass_gate_and_valid_carrier_is_unchanged():
    stack=BeamStack((beam(),))
    coarse=make_grid(grid(128),real_dtype=np.float64)
    with pytest.raises(ValueError,match='minimum Nx=316'):
        build_launch(stack,coarse,context=OpticalLaunchContext(coarse,2.4,1000))
    fine=make_grid(grid(),real_dtype=np.float64)
    launch=build_launch(stack,fine,complex_dtype=np.complex128,
                        context=OpticalLaunchContext(fine,2.4,1000))
    # The actual sampled phase gradient must match conserved tangential k.
    field=launch.A0[0]
    phase=np.angle(field[257,128]/field[256,128])/fine.dx_um
    assert phase==pytest.approx(2*math.pi/.633*.5,abs=2e-14)


def test_disabled_beam_removed_before_gate():
    from launchplane.model import BeamDefinition,BeamStackDefinition
    from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
    stack=beam_stack_definition_to_lcprop(BeamStackDefinition(beams=(
        BeamDefinition(),BeamDefinition(name='disabled',theta_ext_rad=math.pi/6,enabled=False))))
    qualify_launch_sampling(stack,grid(128),2.4).require_valid()


def test_unequal_rolled_oblique_bandwidth_independent_projection():
    theta,phi,roll=.4,.7,.8
    w1,w2=.5,1.3
    b=BeamChannel(w1_um=w1,w2_um=w2,theta_ext_rad=theta,phi_rad=phi,psi_rad=roll)
    c,s=math.cos(theta),math.sin(phi);cp=math.cos(phi)
    # Explicit laboratory projections of the two zero-roll transverse vectors.
    e1=np.array([c+(1-c)*s*s,-(1-c)*s*cp])
    e2=np.array([-(1-c)*s*cp,c+(1-c)*cp*cp])
    a=math.cos(roll)*e1+math.sin(roll)*e2
    d=-math.sin(roll)*e1+math.cos(roll)*e2
    covariance=np.outer(a,a)/w1**2+np.outer(d,d)/w2**2
    report=qualify_launch_sampling(BeamStack((b,)),grid(256,256,10),2.4)
    k=2*math.pi/.633*math.sin(theta)*np.array([cp,s])
    for j,axis in enumerate(('x','y')):
        extent=abs(k[j])+4*math.sqrt(covariance[j,j])
        assert any(f', {axis}: Gaussian' in line and f'{extent:.9g}' in line
                   for line in report.warnings)


def test_valid_sampling_boundary_operator_still_wraps_or_attenuates():
    from lcprop.optics.boundaries import transverse_boundary_mask
    g=make_grid(grid(length=10),real_dtype=np.float64)
    stack=BeamStack((beam(),))
    for mode in ('periodic','sponge'):
        boundary=TransverseBoundarySpec(mode=mode)
        qualify_launch_sampling(stack,g,2.4,boundary=boundary).require_valid()
        mask=transverse_boundary_mask(g,boundary,propagation_distance_um=10)
        if mode=='periodic':
            assert mask is None  # no absorber; existing FFT periodicity is retained
        else:
            assert mask[0,0]<1 and mask[g.Nx//2,g.Ny//2]==1


def test_physical_cutoff_is_separate_from_discrete_sampling():
    with pytest.raises(ValueError,match='no forward propagating transmitted mode'):
        qualify_launch_sampling(BeamStack((beam(),)),grid(),.4)
    # Same physical angle transmits into n=2.4 but a coarse grid is unresolved.
    assert qualify_launch_sampling(BeamStack((beam(),)),grid(128),2.4).errors


@pytest.mark.parametrize('sign',[-1,1])
def test_signed_near_nyquist_request_gate(sign):
    b=BeamChannel(wavelength_um=2*math.pi*math.sin(.3),theta_ext_rad=sign*.3)
    n=128
    equal=n*math.pi
    for width,invalid in [(equal,True),(equal*(1+1e-12),True),(equal*(1-1e-12),False)]:
        g=GridSpec(Nx=n,Ny=n,x_aperture_um=width)
        assert bool(qualify_launch_sampling(BeamStack((b,)),g,2.4).errors)==invalid


@pytest.mark.parametrize('n',[7,31,127,255,327])
def test_minimum_uses_same_nyquist_evaluation_as_grid_gate(n):
    # Equivalent real formulas can differ by one ULP at non-power-of-two counts.
    limit=math.pi/(200/n)
    assert strict_minimum_samples(limit,200)==n+1
    assert strict_minimum_samples(math.nextafter(limit,0),200)==n

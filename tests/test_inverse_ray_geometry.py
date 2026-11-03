"""Independent analytic interface inversion and signed-branch qualification."""
import math
from dataclasses import replace
import pytest
from lcprop.core.beams import BeamChannel
from lcprop.optics.physical_launch import external_direction_for_exit, resolve_beam_geometry


@pytest.mark.parametrize('theta,phi,n', [(.3,0,1.5),(-.3,0,1.5),(.7,1.2,2.4),
    (-.6,3.13,2.4),(.4,3.15,1.7),(-.4,6.27,1.7),(.4,.01,1.7),(-.7,4.8,3.2)])
def test_inverse_independent_analytic_roundtrip(theta,phi,n):
    channel=BeamChannel(theta_ext_rad=theta,phi_rad=phi,n_ext=1.2,x0_um=3,y0_um=-4)
    length=173.
    # Oracle uses unit tangential direction and Snell, not the forward helper.
    tx=channel.n_ext*math.sin(theta)*math.cos(phi)
    ty=channel.n_ext*math.sin(theta)*math.sin(phi)
    denominator=math.sqrt(n*n-tx*tx-ty*ty)
    exit_xy=(3+length*tx/denominator,-4+length*ty/denominator)
    recovered=external_direction_for_exit(channel,n,length,*exit_xy)
    assert recovered == pytest.approx((theta,phi),abs=2e-14)
    geometry=resolve_beam_geometry(replace(channel,theta_ext_rad=recovered[0],phi_rad=recovered[1]),n)
    assert (3+length*geometry.kx/geometry.kz_internal,-4+length*geometry.ky/geometry.kz_internal) == pytest.approx(exit_xy)


def test_zero_crossing_keeps_meridian_and_changes_signed_theta():
    channel=BeamChannel(theta_ext_rad=.1,phi_rad=.73)
    for distance in (2.,.1,0.,-.1,-2.):
        theta,phi=external_direction_for_exit(channel,1.5,100,
            distance*math.cos(.73),distance*math.sin(.73))
        assert phi == pytest.approx(.73)
        assert theta*distance >= 0
        channel=replace(channel,theta_ext_rad=theta,phi_rad=phi)
    theta,phi=external_direction_for_exit(channel,1.5,100,0,0)
    assert math.copysign(1,theta) == -1 and phi == pytest.approx(.73)


@pytest.mark.parametrize('sign', [-1,1])
def test_azimuth_wrap_and_tie_preserve_signed_intent(sign):
    channel=BeamChannel(theta_ext_rad=sign*.2,phi_rad=2*math.pi-.01)
    theta,phi=external_direction_for_exit(channel,1.5,100,sign*2,sign*.02)
    assert theta*sign > 0 and phi == pytest.approx(math.atan2(.02,2))
    channel=replace(channel,theta_ext_rad=sign*.2,phi_rad=0)
    theta,phi=external_direction_for_exit(channel,1.5,100,0,2)
    assert theta*sign > 0


@pytest.mark.parametrize('n,length,x,y', [(2,100,100,0),(2,0,0,0),(0,100,1,1),
    (2,100,float('nan'),0),(2,float('inf'),1,1)])
def test_unreachable_invalid_exit_rejected(n,length,x,y):
    with pytest.raises(ValueError):
        external_direction_for_exit(BeamChannel(),n,length,x,y)

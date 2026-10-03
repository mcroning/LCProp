"""Frozen M4 design oracles, independent of the implemented reconstruction."""
from dataclasses import replace
import numpy as np
import pytest

from lcprop.pr.unified.projection import (
    PROJECTION_ID, electric_field_face, electric_field_optical_node,
)
from lcprop.pr.unified.specs import (
    PRUnifiedSpatialSpec, PRElectricalClosureSpec, PRMaterialPrecisionSpec,
    UNBIASED, MIXED_PRECISION,
)
from lcprop.pr.unified.state import PRUnifiedMaterialState
from lcprop.pr.optical_response import delta_n_from_E


def state(p, *, lengths=None, batch=False, b=None, backend='numpy'):
    dim = 1 if batch else p.ndim
    spatial = PRUnifiedSpatialSpec(p.shape[:dim], lengths or (2*np.pi,)*dim,
        active_axes=('x',) if dim == 1 else ('x','y'),
        batch_shape=p.shape[1:] if batch else (), batch_axes=('y',) if batch else ())
    xp = np if backend == 'numpy' else __import__('cupy')
    precision = PRMaterialPrecisionSpec() if p.dtype.name == 'float64' else PRMaterialPrecisionSpec(
        MIXED_PRECISION, 'float32', output_dtype='float32')
    b = xp.zeros(spatial.harmonic_shape, dtype=p.dtype) if b is None else b
    return PRUnifiedMaterialState(xp.zeros_like(p),p,b,spatial,
        PRElectricalClosureSpec(UNBIASED,dim,(0.,)*dim),precision,backend)


def phase(E):
    xp = np if isinstance(E,np.ndarray) else __import__('cupy')
    return xp.exp(1j*(2*np.pi/.633)*50*delta_n_from_E(E,
        gain_length_product=10.,interaction_length_um=4000.,wavelength_um=.633))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('m',[0,1,4,12,24,31,32])
def test_frozen_fourier_sign_amplitude(dtype,m):
    N=64;h=2*np.pi/N;x=np.arange(N)*h
    p=(.01*np.cos(m*x)).astype(dtype)
    if m==32: p=(.01*(-1.)**np.arange(N)).astype(dtype)
    s=state(p);before=[v.tobytes() for v in (s.q,s.psi,s.b)]
    E=electric_field_optical_node(s)
    expected=.01*np.sin(m*x)*np.sin(m*h)/h
    if m in (0,32):
        expected=np.zeros(N)
        assert np.count_nonzero(E)==0
    np.testing.assert_allclose(E,expected,rtol=0,atol=4e-7 if dtype==np.float32 else 1e-14)
    assert E.dtype==dtype and not np.shares_memory(E,p)
    assert before==[v.tobytes() for v in (s.q,s.psi,s.b)]


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('kind',['sinusoid','localized'])
def test_frozen_field_and_phase_refinement(dtype,kind):
    errors=[]
    for N in (32,64,128,256):
        x=np.arange(N)*2*np.pi/N
        p=.01*np.sin(x) if kind=='sinusoid' else .05*np.exp(2*(np.cos(x)-1))
        expected=-.01*np.cos(x) if kind=='sinusoid' else .1*np.sin(x)*np.exp(2*(np.cos(x)-1))
        E=electric_field_optical_node(state(p.astype(dtype)))
        screen=phase(E)
        assert screen.dtype==(np.complex64 if dtype==np.float32 else np.complex128)
        errors.append([np.sqrt(np.mean(abs(E-expected)**2)),
                       np.sqrt(np.mean(abs(screen-np.exp(-.25j*expected))**2))])
    rates=np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.all((rates>1.85)&(rates<2.15))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_constant_harmonic_and_batch_independence(dtype):
    b=np.array([[.125],[.25],[-.5]],dtype=dtype)
    p=np.full((32,3),.125,dtype=dtype)
    s=state(p,batch=True,b=b)
    E=electric_field_optical_node(s)
    np.testing.assert_array_equal(E,np.broadcast_to(b[:,0],p.shape))
    np.testing.assert_array_equal(phase(E),phase(np.broadcast_to(b[:,0],p.shape)))
    with pytest.raises(ValueError,match='active'): electric_field_optical_node(s,component='y')


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_exact_reduction_and_native_face_arithmetic(dtype):
    from lcprop.pr.unified.operators import Geometry,field
    from lcprop.pr.unified._backend import MaterialBackend
    p=(.01*np.sin(np.arange(64)*2*np.pi/64)).astype(dtype)
    s=state(p,b=np.array([.125],dtype=dtype))
    s2=state(np.repeat(p[:,None],3,axis=1),lengths=(2*np.pi,3.),b=np.array([.125,0],dtype=dtype))
    a=electric_field_optical_node(s);b=electric_field_optical_node(s2)
    for aa,bb in ((a,b),(delta_n_from_E(a,gain_length_product=10,interaction_length_um=4000,wavelength_um=.633),
                        delta_n_from_E(b,gain_length_product=10,interaction_length_um=4000,wavelength_um=.633)),
                  (phase(a),phase(b))):
        np.testing.assert_array_equal(bb,np.repeat(aa[:,None],3,axis=1))
    assert np.count_nonzero(electric_field_optical_node(s2,component='y'))==0
    g=Geometry(s2.spatial.active_shape,s2.spatial.normalized_lengths,MaterialBackend('numpy',p.dtype.name))
    for c,reference in zip(('x','y'),field(g,s2.psi,s2.b)):
        np.testing.assert_array_equal(electric_field_face(s2,component=c),reference)


def test_requested_axis_only_bounded_operations(monkeypatch):
    import lcprop.pr.unified.projection as module
    s=state(np.ones((8,6)))
    calls=[];original=np.roll
    def roll(a,shift,axis):
        calls.append((a.shape,shift,axis));return original(a,shift,axis=axis)
    monkeypatch.setattr(np,'roll',roll)
    out=module.electric_field_optical_node(s)
    assert calls==[((8,6),-1,0),((8,6),1,0)]
    assert out.shape==(8,6) and not np.shares_memory(out,s.psi)
    with pytest.raises(ValueError,match='identity'): module.electric_field_optical_node(s,identity='future')
    with pytest.raises(ValueError): module.electric_field_optical_node(replace(s,psi=s.psi.astype(np.float32)))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_two_active_axes_frozen_refinement(dtype):
    errors=[]
    for N in (32,64,128):
        x=np.arange(N)*2*np.pi/N
        p=(.01*np.sin(x[:,None])+.02*np.cos(x[None,:])).astype(dtype)
        s=state(p,b=np.array([.125,-.25],dtype=dtype))
        ex=electric_field_optical_node(s,component='x')
        ey=electric_field_optical_node(s,component='y')
        errors.append([np.sqrt(np.mean((ex-(.125-.01*np.cos(x[:,None])))**2)),
                       np.sqrt(np.mean((ey-(-.25+.02*np.sin(x[None,:])))**2))])
    rates=np.log2(np.array(errors[:-1])/np.array(errors[1:]))
    assert np.all((rates>1.85)&(rates<2.15))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_native_projection_parity_when_available(dtype,monkeypatch):
    cp=pytest.importorskip('cupy')
    try: cp.zeros(1)
    except Exception: pytest.skip('CuPy/CUDA unavailable')
    p=(.01*np.sin(np.arange(64)*2*np.pi/64)).astype(dtype)
    expected=electric_field_optical_node(state(p))
    gpu=state(cp.asarray(p),backend='cupy')
    original=cp.asnumpy
    def guard(*args,**kwargs): raise AssertionError('projection host transfer')
    monkeypatch.setattr(cp,'asnumpy',guard)
    result=electric_field_optical_node(gpu);screen=phase(result)
    np.testing.assert_allclose(original(result),expected,rtol=0,atol=4e-7 if dtype==np.float32 else 1e-14)
    np.testing.assert_allclose(original(screen),phase(expected),rtol=0,atol=4e-7 if dtype==np.float32 else 1e-14)

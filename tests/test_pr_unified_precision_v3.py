"""Independent exp oracle, explicit policy and no narrowing of carrier products."""
from decimal import Decimal, localcontext
import numpy as np
import pytest
from lcprop.pr.unified import _positive as positive
from lcprop.pr.unified.specs import PRMaterialPrecisionSpec, POSITIVE_PRECISION, MIXED_PRECISION

EDGES = [np.float32(positive.Q_MIN),
    np.nextafter(np.float32(positive.Q_MIN),np.float32(np.inf)),
    np.float32(-90), np.float32(np.log(np.finfo(np.float32).tiny)),
    np.nextafter(np.float32(np.log(np.finfo(np.float32).tiny)),np.float32(-np.inf)),
    np.nextafter(np.float32(np.log(np.finfo(np.float32).tiny)),np.float32(np.inf)),
    np.float32(0), np.float32(80), np.float32(positive.Q_MAX)]


@pytest.mark.parametrize('value',EDGES)
def test_supported_edges_positive_wide_and_detached(value):
    q=np.array([value],dtype=np.float32);before=q.tobytes()
    n=positive.carrier(np,q);product=positive.carrier_product(np,q)
    with localcontext() as ctx:
        ctx.prec=100
        expected=float(Decimal(float(value)).exp())
    assert n.dtype==np.float64 and n[0]>0 and np.isfinite(n).all()
    np.testing.assert_allclose(n,[expected],rtol=3e-13,atol=0)
    np.testing.assert_array_equal(n,product)
    product[:]=0
    assert n[0]>0 and q.tobytes()==before
    assert not np.shares_memory(n,product)


@pytest.mark.parametrize('value',[
    np.nextafter(np.float32(positive.Q_MIN),np.float32(-np.inf)),
    np.nextafter(np.float32(positive.Q_MAX),np.float32(np.inf)),
    np.float32(-105),np.float32(np.nan),np.float32(np.inf),np.float32(-np.inf)])
def test_domain_rejection_before_exp_no_clipping(value,monkeypatch):
    q=np.array([value],np.float32);before=q.tobytes()
    def forbidden(*args,**kwargs):raise AssertionError('out-of-domain exp evaluated')
    monkeypatch.setattr(np,'exp',forbidden)
    assert not positive.domain_ok(np,q)
    with pytest.raises(positive.CarrierDomainError):positive.carrier(np,q)
    assert q.tobytes()==before


@pytest.mark.parametrize('dt',['float16','int32'])
def test_wrong_canonical_dtype_rejected(dt):
    with pytest.raises(TypeError):positive.carrier(np,np.zeros(2,dt))


def test_exp_receives_only_wide_operands(monkeypatch):
    old=np.exp;seen=[]
    def checked(a):
        seen.append(a.dtype)
        assert a.dtype==np.float64
        return old(a)
    monkeypatch.setattr(np,'exp',checked)
    positive.carrier(np,np.array([-90.,.1],np.float32))
    assert seen==[np.dtype('float64')]


def test_precision_identity_and_no_dtype_override():
    p=PRMaterialPrecisionSpec(POSITIVE_PRECISION,'float32',output_dtype='float32')
    assert p.state_dtype=='float32' and p.output_dtype=='float32'
    assert p.carrier_dtype==p.coefficient_dtype=='float64'
    old=PRMaterialPrecisionSpec(MIXED_PRECISION,'float32',output_dtype='float32')
    assert old.carrier_dtype==old.coefficient_dtype=='float32'
    with pytest.raises(ValueError):PRMaterialPrecisionSpec(POSITIVE_PRECISION)
    with pytest.raises(ValueError):PRMaterialPrecisionSpec(POSITIVE_PRECISION,'float32',linear_dtype='float32',output_dtype='float32')
    with pytest.raises(TypeError):positive.carrier_product(np,np.zeros(2,np.float32),dtype='float32')


@pytest.mark.parametrize('value',EDGES)
def test_native_carrier64_edges_when_available(value):
    cp=pytest.importorskip('cupy',reason='CuPy/CUDA unavailable')
    try:q=cp.asarray([value],dtype=cp.float32)
    except Exception:pytest.skip('CuPy/CUDA unavailable')
    n=positive.carrier(cp,q);out=positive.carrier_product(cp,q)
    assert n.dtype==cp.float64 and out.dtype==cp.float64
    assert bool(cp.all(n>0).item()) and bool(cp.all(n==out).item())
    expected=positive.carrier(np,np.array([value],np.float32))
    np.testing.assert_allclose(cp.asnumpy(out),expected,rtol=3e-13,atol=0)


def test_low_carrier_flux_and_coefficient_storage_remain_wide():
    from lcprop.pr.unified._backend import MaterialBackend
    from lcprop.pr.unified._scalable import WideJacobian
    from tests.test_pr_unified_scalable import electrical
    B = MaterialBackend('numpy', 'float32')
    g = positive.Geometry((4, 4), (2., 3.), B)
    q = np.full(g.shape, -90., np.float32)
    p = np.zeros_like(q); b = np.array([.15, -.08], np.float32)
    I = np.ones_like(q)
    faces = positive.ARITHMETIC.flux(g, I, q, p, b)
    assert all(f.dtype == np.float64 and np.isfinite(f).all() for f in faces)
    J = WideJacobian(g, I, q, p, b, electrical(B, 'fixed'))
    assert J.n.dtype == np.float64 and J.n.min() > 0
    assert J.qcenter.dtype == J.pcenter.dtype == np.float64
    assert np.max(abs(J.qcenter)) > 0

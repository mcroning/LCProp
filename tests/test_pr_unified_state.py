"""M1 array metadata/lifetime contracts, including simulated device metadata."""
from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from lcprop.pr.unified.specs import (
    FIXED_FIELD, MIXED_PRECISION, UNBIASED, PRElectricalClosureSpec,
    PRMaterialPrecisionSpec, PRUnifiedSpatialSpec,
)
from lcprop.pr.unified.state import PRMaterialDiagnostics, PRTransportIntensity, PRUnifiedMaterialState


def state(dtype='float64',batched=False,full=False):
    s=(PRUnifiedSpatialSpec((4,3),(2.,3.),('x','y')) if full else
       PRUnifiedSpatialSpec((4,),(2.,),batch_axes=('y',) if batched else (),batch_shape=(3,) if batched else ()))
    precision=PRMaterialPrecisionSpec() if dtype=='float64' else PRMaterialPrecisionSpec(MIXED_PRECISION,'float32',output_dtype='float32')
    return PRUnifiedMaterialState(np.zeros(s.field_shape,dtype),np.zeros(s.field_shape,dtype),
        np.zeros(s.harmonic_shape,dtype),s,PRElectricalClosureSpec(UNBIASED,s.dimension,(0.,)*s.dimension),precision,'numpy')


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('batched,full',[(False,False),(True,False),(False,True)])
def test_state_layout(dtype,batched,full):
    a=state(dtype,batched,full);a.validate_structure()
    assert a.b.shape==a.spatial.harmonic_shape
    with pytest.raises(ValueError):replace(a,b=np.zeros((1,),dtype) if a.b.shape!=(1,) else np.zeros((2,),dtype)).validate_structure()
    with pytest.raises(ValueError):replace(a,q=np.zeros((1,)+a.q.shape,dtype)).validate_structure()


@pytest.mark.parametrize('change', [
    {'backend':'auto'}, {'backend':'cupy'}, {'state_id':'unknown_v2'}, {'material_id':'unknown'},
    {'q':[0.,0.,0.,0.]}, {'q':np.zeros(4,'float32')}, {'q':np.zeros(4,'complex128')},
    {'psi':np.zeros(4,'int64')}, {'b':np.zeros((1,),'float32')},
    {'q':np.zeros(4,dtype='>f8')},
])
def test_invalid_state(change):
    with pytest.raises(ValueError):replace(state(),**change).validate_structure()


def test_closure_dimension_and_required_types():
    a=state()
    with pytest.raises(ValueError):replace(a,closure=PRElectricalClosureSpec(FIXED_FIELD,2,(0.,0.))).validate_structure()
    with pytest.raises(TypeError):replace(a,closure=None).validate_structure()
    with pytest.raises(TypeError):replace(a,spatial=None).validate_structure()


def test_borrowing_no_physical_scan_or_mutation(monkeypatch):
    a=state();a.q[:]=np.nan;a.psi[:]=7;a.b[:]=3
    initial=[v.tobytes() for v in (a.q,a.psi,a.b)]
    def forbidden(*args,**kwargs):raise AssertionError('array allocation/conversion/scan forbidden')
    with monkeypatch.context() as m:
        for name in ['array','asarray','copy','empty','zeros','ones','isfinite','mean','exp','sum']:
            m.setattr(np,name,forbidden)
        b=replace(a);b.validate_structure()
    assert b.q is a.q and b.psi is a.psi and b.b is a.b
    assert [v.tobytes() for v in (a.q,a.psi,a.b)]==initial
    a.psi[0]=8
    assert b.psi[0]==8  # Borrowed: frozen dataclass does not freeze the buffer.
    with pytest.raises(FrozenInstanceError):a.q=None
    assert a is not b and a != b  # No ndarray-valued equality.


def test_device_metadata_no_import_transfer_or_scan(monkeypatch):
    class DeviceArray:
        def __init__(self,shape,device=0):self.shape=shape;self.dtype=np.dtype('float64');self.device=SimpleNamespace(id=device)
        def __array__(self,*a,**k):raise AssertionError('host conversion')
        def get(self,*a,**k):raise AssertionError('device readback')
        def copy(self,*a,**k):raise AssertionError('copy')
        def __iter__(self):raise AssertionError('scan')
    # A metadata double, not a claim of native CuPy qualification.
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace(ndarray=DeviceArray))
    a=state();a=replace(a,q=DeviceArray((4,)),psi=DeviceArray((4,)),b=DeviceArray((1,)),backend='cupy')
    a.validate_structure()
    with pytest.raises(ValueError):replace(a,b=DeviceArray((1,),device=1)).validate_structure()
    with pytest.raises(ValueError):replace(a,b=np.zeros(1)).validate_structure()


def intensity():
    a=state();return PRTransportIntensity(a.q,a.spatial,a.precision,a.backend,2.,.01,0.)


def test_intensity_borrowing_and_no_positivity_claim():
    a=intensity();a.values[:]=-1;a.validate_structure()
    assert a.total_background==.01 and a.peak_intensity_reference==2.
    assert replace(a).values is a.values


@pytest.mark.parametrize('change', [
    {'peak_intensity_reference':0.}, {'peak_intensity_reference':float('nan')},
    {'dark_intensity':-.1}, {'uniform_background':float('inf')},
    {'dark_intensity':1e308, 'uniform_background':1e308},
    {'intensity_id':'unknown'}, {'normalization_id':'unknown'}, {'units':'physical'},
    {'values':np.zeros(4,'float32')}, {'backend':'auto'},
])
def test_intensity_rejections(change):
    with pytest.raises(ValueError):replace(intensity(),**change).validate_structure()


def test_diagnostics_are_metadata_not_acceptance():
    values=[['gauss_rms',2.]];limits=[['gauss_rms',1.]]
    d=PRMaterialDiagnostics(values,limits,'reported');values[0][1]=99;limits[0][1]=99
    assert d.observations==(('gauss_rms',2.),) and d.limits==(('gauss_rms',1.),)
    assert not hasattr(d,'passed')  # Above limit is reported, not judged in M1.
    PRMaterialDiagnostics().validate_structure()
    for args in [dict(status='accepted'),dict(diagnostics_id='unknown'),dict(observations=(('x',1.),)),
                 dict(status='reported',observations=(('x',1.),('x',2.))),
                 dict(status='reported',limits=(('x',-1.),))]:
        with pytest.raises(ValueError):PRMaterialDiagnostics(**args)


def test_namespace_import_is_documentation_only():
    # Fresh interpreter: importing the namespace must not pull solvers/backends/registries.
    code='''
import sys
import lcprop.pr
before=set(sys.modules)
import lcprop.pr.unified as module
added=set(sys.modules)-before
assert added == {'lcprop.pr.unified'}, added
assert not any(k for k in vars(module) if not k.startswith('__'))
'''
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    subprocess.run([sys.executable,'-B','-c',code],env=env,check=True,capture_output=True,text=True)


def test_intensity_validation_has_no_array_operations(monkeypatch):
    a=intensity();before=a.values.tobytes()
    def forbidden(*args,**kwargs):raise AssertionError('array operation forbidden')
    with monkeypatch.context() as m:
        for name in ['array','asarray','copy','empty','zeros','ones','isfinite','mean','exp','sum']:
            m.setattr(np,name,forbidden)
        a.validate_structure()
    assert a.values.tobytes()==before

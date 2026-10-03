"""M5 observations must not change any accepted scientific state."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest

from tests.test_pr_unified_workflow import request
from lcprop.core.execution import CancellationToken
from lcprop.optics.splitstep import total_intensity
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.pr.unified import workflow as w
from lcprop.pr.unified import products as p
from lcprop.pr.unified.operators import Geometry,flux
from lcprop.pr.unified._backend import MaterialBackend
from lcprop.pr.source import channel_peak_intensity_reference,pr_driving_intensity
from lcprop.core.grid import make_grid
from lcprop.optics.splitstep import hop_linear_inplace,scalar_angular_spectrum_kernel


def all_selection(dim=1):
    names=tuple(n for n in p.FIELDS if dim==2 or '_y_' not in n)
    return p.UnifiedSelection(True,True,True,True,names,tuple(n for n in names if n!='harmonic_field'),True)


def digest(result):
    s=result.scientific
    arrays=(s.boundary_field,s.material_state.q,s.material_state.psi,s.material_state.b)
    return dict(arrays=[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
        ledger=hashlib.sha256(json.dumps(s.ledger,sort_keys=True).encode()).hexdigest())


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('dim',[1,2])
@pytest.mark.parametrize('scattering',[False,True])
def test_selection_preserves_science_and_fields(dtype,dim,scattering,record_property):
    r=request(2,dtype,dim,scatter=scattering)
    original=r.initial_A.tobytes()
    minimal=p.run_unified_products(r)
    selected=p.run_unified_products(r,selection=all_selection(dim))
    assert minimal.scientific.status==selected.scientific.status=='completed'
    assert digest(minimal)==digest(selected)
    baseline=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True))
    for name in ('q','psi','b'):
        np.testing.assert_array_equal(getattr(baseline.material_state,name),getattr(selected.scientific.material_state,name))
    np.testing.assert_array_equal(baseline.boundary_field,selected.scientific.boundary_field)
    assert r.initial_A.tobytes()==original
    assert selected.scientific.collection is None and selected.scientific.products=={}
    assert selected.next_address==dict(cell_index=2,canonical_slab=4 if scattering else None)
    np.testing.assert_array_equal(selected.coordinates['boundary_z_um'],[0.,2.,4.])
    np.testing.assert_array_equal(selected.coordinates['material_z_um'],[2.,4.])
    for c in ('x','y') if dim==2 else ('x',):
        location=selected.locations[f'electric_field_{c}_face']
        assert location['orientation']=='positive' and location['longitudinal']=='material_z_um'
        assert location['offset_um']==([.5,0.] if c=='x' else [0.,1.])
    assert selected.locations['electric_field_x_optical_node']['projection']==w.PROJECTION_ID
    record_property('product_independence_hashes',json.dumps(digest(selected),sort_keys=True))


def test_independent_reduced_native_current_and_cuts():
    r=request(1)
    A=r.initial_A.copy();grid=make_grid(r.grid,real_dtype=np.float64)
    peak=channel_peak_intensity_reference(A,xp=np)
    kernel=scalar_angular_spectrum_kernel(grid.fxy2_um,dz=2.,wavelength=.633,n_ref=2.4,complex_dtype=A.dtype,xp=np)
    hop_linear_inplace(A,kernel,xp=np)
    I=pr_driving_intensity(A,peak_intensity_reference=peak,background_intensity=.01,xp=np)
    result=p.run_unified_products(r,selection=all_selection())
    s=result.scientific.material_state
    g=Geometry((16,),(16.,),MaterialBackend('numpy','float64'))
    expected=np.column_stack([flux(g,I[:,j],s.q[:,j],s.psi[:,j],s.b[j])[0] for j in range(4)])
    np.testing.assert_array_equal(result.arrays['hopping_current_x_face'],expected)
    ix,iy=np.argmin(abs(grid.x_um)),np.argmin(abs(grid.y_um))
    np.testing.assert_array_equal(result.arrays['hopping_current_x_face_cut_x'][0],expected[:,iy])
    np.testing.assert_array_equal(result.arrays['optical_cut_y'][-1],result.scientific.boundary_field[:,ix,:])
    np.testing.assert_array_equal(result.arrays['intensity_cut_x'][0],total_intensity(r.initial_A,xp=np)[:,iy])
    spectrum=direction_cosine_spectrum(result.scientific.boundary_field,dx_um=1.,dy_um=2.,wavelength_um=.633,refractive_index=2.4)
    np.testing.assert_array_equal(result.arrays['far_field_intensity'],spectrum.intensity)


@pytest.mark.parametrize('mode',['fail','cancel','mutate'])
def test_unaccepted_collector_output_is_discarded(mode,monkeypatch):
    r=request(3,scatter=True);token=CancellationToken();original=p._Collector.prepare
    def prepare(self,previous,A,state,I,grid,record,ids):
        result=original(self,previous,A,state,I,grid,record,ids)
        if record is not None and record['cell_index']==1:
            if mode=='fail': raise RuntimeError('collector fault')
            if mode=='cancel': token.cancel()
            if mode=='mutate': A[:]=0
        return result
    monkeypatch.setattr(p._Collector,'prepare',prepare)
    result=p.run_unified_products(r,selection=all_selection(),cancellation_token=token)
    assert result.scientific.completed_cells==1 and result.scientific.reached_z_um==2
    assert result.scientific.status==('cancelled' if mode=='cancel' else 'failed')
    assert result.arrays['optical_cut_x'].shape[0]==2
    assert result.arrays['potential_node_cut_x'].shape[0]==1
    assert result.next_address==dict(cell_index=1,canonical_slab=2)
    np.testing.assert_array_equal(result.arrays['potential_node'],result.scientific.material_state.psi)


def test_launch_only_and_collector_failure_before_launch(monkeypatch):
    r=request(1);token=CancellationToken();token.cancel()
    out=p.run_unified_products(r,selection=all_selection(),cancellation_token=token)
    assert out.scientific.status=='cancelled' and out.scientific.material_state is None
    assert out.arrays['optical_cut_x'].shape[0]==1
    assert not any(k.startswith('potential_') for k in out.arrays)
    def fail(*a,**kw): raise RuntimeError('launch observation failed')
    monkeypatch.setattr(p._Collector,'prepare',fail)
    bad=p.run_unified_products(r,selection=all_selection())
    assert bad.scientific.status=='failed' and bad.scientific.boundary_field is None
    assert bad.arrays=={} and len(bad.coordinates['boundary_z_um'])==0


def test_no_y_products_or_full_volume_selection():
    with pytest.raises(ValueError):p.run_unified_products(request(),selection=p.UnifiedSelection(material_fields=('hopping_current_y_face',)))
    with pytest.raises(ValueError):p.UnifiedSelection(material_fields=('full_material_volume',))
    with pytest.raises(ValueError):p.UnifiedSelection(material_cuts=('harmonic_field',))


def test_bounded_series_never_retains_scientific_snapshots():
    result=p.run_unified_products(request(4),selection=p.UnifiedSelection(optical_cuts=True,intensity_cuts=True,
        material_cuts=('potential_node','hopping_current_x_face')))
    assert result.scientific.status=='completed'
    for key,value in result.arrays.items():
        assert '_cut_' in key
        assert value.shape in ((5,1,16),(5,1,4),(5,16),(5,4),(4,16),(4,4))
    assert result.scientific.collection is None


def test_selected_postprocessing_failure_retains_accepted_science(monkeypatch):
    r=request(1);baseline=p.run_unified_products(r)
    original=np.stack
    def fail(arrays,*a,**kw):
        if arrays and getattr(arrays[0],'shape',())==(1,16): raise RuntimeError('cut assembly failure')
        return original(arrays,*a,**kw)
    monkeypatch.setattr(np,'stack',fail)
    out=p.run_unified_products(r,selection=p.UnifiedSelection(optical_cuts=True))
    assert out.scientific.status=='failed' and out.scientific.completed_cells==1
    assert out.scientific.failure['stage']=='selected_products'
    assert digest(out)==digest(baseline)


def test_minimal_has_no_observation_and_optical_only_borrows_no_material(monkeypatch):
    original=p._Collector.prepare;observed=[]
    def prepare(self,previous,A,state,I,grid,record,ids):
        observed.append((A is not None,state is not None,I is not None))
        return original(self,previous,A,state,I,grid,record,ids)
    monkeypatch.setattr(p._Collector,'prepare',prepare)
    assert p.run_unified_products(request(1)).scientific.status=='completed'
    assert observed==[]
    assert p.run_unified_products(request(1),selection=p.UnifiedSelection(optical_cuts=True)).scientific.status=='completed'
    assert observed==[(True,False,False),(True,False,False)]
    observed.clear()
    assert p.run_unified_products(request(1),selection=p.UnifiedSelection(material_cuts=('potential_node',))).scientific.status=='completed'
    assert observed==[(False,False,False),(False,True,True)]

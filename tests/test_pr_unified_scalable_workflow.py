"""Predeclared complete-workflow gates: no fitted phase or post-hoc budgets.

Float64 rtol/atol 1e-8/1e-8; state32 v1-v3 2e-4/2e-5.
Far-field integrated absolute error / launch sampled norm <= relative budget.
All cases use the deterministic accepted launch; 1,2,4 cells and gains .1,1.
"""
from dataclasses import replace, asdict
import hashlib
import json
import numpy as np
import pytest
from tests.test_pr_unified_workflow import request
from lcprop.pr.unified import workflow as w, codec
from lcprop.pr.unified.specs import *
from lcprop.pr.unified.solver_specs import *
from lcprop.pr.unified.products import UnifiedSelection, run_unified_products

SOLVER = PRUnifiedSolverSpec(SCALABLE, ITERATIVE_POLICY)
FIELDS = ('q_log_carrier','potential_node','harmonic_field','carrier_node',
          'transport_intensity_node','electric_field_x_face','electric_field_y_face','hopping_current_x_face',
          'hopping_current_y_face','electric_field_x_optical_node','material_phase_optical_node')
RICH = UnifiedSelection(launch=True,boundary_intensity=True,optical_cuts=True,
    intensity_cuts=True,material_fields=FIELDS,far_field=True,intensity_volume=True,
    material_volumes=('potential_node','carrier_node','electric_field_x_optical_node'))


def frozen_face_current_atol(I, reference_carrier, spacing):
    """S2/v3 qualify.py and committed S4-M1 current contract, unchanged."""
    return max(2e-5, 16*np.finfo(np.float32).eps*I.max()*reference_carrier.max()/min(spacing))


def compare_product(key, actual, reference, direct, r, rt, at):
    if key.startswith('hopping_current_') and r.precision.state_dtype == 'float32':
        spacing = tuple(L/n for L,n in zip(r.spatial.normalized_lengths,r.spatial.active_shape))
        at = frozen_face_current_atol(direct.arrays['transport_intensity_node'],
                                     direct.arrays['carrier_node'], spacing)
    np.testing.assert_allclose(actual,reference,rtol=rt,atol=at,err_msg=key)


def test_frozen_current_allowance_rejects_just_beyond():
    allowance=frozen_face_current_atol(np.ones((2,2)),np.ones((2,2)),(.04,.2))
    np.testing.assert_allclose(allowance,0.,rtol=2e-4,atol=allowance)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.nextafter(allowance,np.inf),0.,rtol=2e-4,atol=allowance)


def scalable(r):
    precision = (PRMaterialPrecisionSpec(POSITIVE_PRECISION,'float32',output_dtype='float32')
                 if r.precision.state_dtype=='float32' else r.precision)
    return replace(r,solver=SOLVER,precision=precision)


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('closure',[UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT,OPEN_TRANSVERSE])
@pytest.mark.parametrize('cells,scatter,gain',[(1,False,.1),(2,True,.1),(4,True,1.)])
def test_complete_overlap(precision,closure,cells,scatter,gain,monkeypatch,record_property):
    r=request(cells,precision,2,scatter,closure)
    r=replace(r,material=replace(r.material,gain_length_product=gain))
    events=[];original=w.canonical_scattering_phase_increment
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);events.append(hashlib.sha256(value.tobytes()).hexdigest());return value
    monkeypatch.setattr(w,'canonical_scattering_phase_increment',capture)
    a=run_unified_products(r,selection=RICH)
    b=run_unified_products(scalable(r),selection=RICH)
    assert events[:cells]==events[cells:] if scatter else events==[]
    record_property('scattering_phase_hashes',json.dumps(events[:cells]))
    assert a.scientific.status==b.scientific.status=='completed',(a.scientific.failure,b.scientific.failure)
    assert a.scientific.completed_cells==b.scientific.completed_cells==cells
    assert a.scientific.reached_z_um==b.scientific.reached_z_um
    rt,at=(2e-4,2e-5) if precision=='float32' else (1e-8,1e-8)
    np.testing.assert_allclose(b.scientific.boundary_field,a.scientific.boundary_field,rtol=rt,atol=at)
    for key in a.arrays:
        compare_product(key,b.arrays[key],a.arrays[key],a,r,rt,at)
    for key in a.coordinates:np.testing.assert_array_equal(a.coordinates[key],b.coordinates[key])
    error=np.sum(abs(a.arrays['far_field_intensity']-b.arrays['far_field_intensity']))
    assert error/np.sum(a.arrays['far_field_intensity'])<=rt
    assert [x['canonical_slabs'] for x in a.scientific.ledger]==[x['canonical_slabs'] for x in b.scientific.ledger]
    assert b.arrays['carrier_node'].dtype==np.float64
    assert b.arrays['carrier_node_volume'].dtype==np.float64
    from tests._pr_unified_oracle import evaluate_plane
    fixed=(True,True) if closure in (UNBIASED,FIXED_FIELD) else ((True,False) if closure==OPEN_TRANSVERSE else (False,False))
    for result in (a,b):
        st=result.scientific.material_state
        evaluate_plane(result.arrays['transport_intensity_node'],st.q,st.psi,st.b,
                       r.spatial.normalized_lengths,fixed,r.closure.target,closure==UNBIASED,precision)
    # |exp(i a)-exp(i b)| <= |a-b|, plus predeclared state roundoff.
    Eerror=np.max(abs(b.arrays['electric_field_x_optical_node']-a.arrays['electric_field_x_optical_node']))
    bound=2*abs(gain)/cells*Eerror+8*np.finfo(precision).eps
    assert np.max(abs(b.arrays['material_phase_optical_node']-a.arrays['material_phase_optical_node']))<=bound
    payload=asdict(r);payload['initial_A']=hashlib.sha256(r.initial_A.tobytes()).hexdigest()
    record_property('request_scientific_metadata',json.dumps(payload,sort_keys=True))
    record_property('product_max_differences',json.dumps({k:float(np.max(abs(b.arrays[k]-a.arrays[k]))) for k in a.arrays}))
    record_property('endpoint_max',float(np.max(abs(b.scientific.boundary_field-a.scientific.boundary_field))))
    record_property('farfield_integrated_relative',float(error/np.sum(a.arrays['far_field_intensity'])))
    record_property('phase_lipschitz_bound',float(bound))


@pytest.mark.parametrize('precision',['float32','float64'])
def test_selection_observer_identity(precision):
    r=scalable(request(2,precision,2,True));observed=[]
    minimal=run_unified_products(r)
    interactive=run_unified_products(r,selection=UnifiedSelection(boundary_intensity=True,intensity_cuts=True),observer=lambda event:None)
    full=run_unified_products(r,selection=RICH,observer=observed.append)
    assert len(observed)==2
    for other in (interactive,full):
        assert other.scientific.status==minimal.scientific.status=='completed'
        np.testing.assert_array_equal(other.scientific.boundary_field,minimal.scientific.boundary_field)
        for name in ('q','psi','b'):np.testing.assert_array_equal(getattr(other.scientific.material_state,name),getattr(minimal.scientific.material_state,name))
        assert other.scientific.ledger==minimal.scientific.ledger


def geometry(r,shape,backend):
    return replace(r,grid=replace(r.grid,Nx=shape[0],Ny=shape[1]),
        spatial=replace(r.spatial,active_shape=shape),backend=backend,initial_A=None)


def test_metadata_routing_without_device(monkeypatch):
    monkeypatch.setattr(w,'get_backend',lambda *a:pytest.fail('device probe'))
    r=geometry(scalable(request(1,dimension=2)),(128,128),'cupy')
    assert w._validate(r,w.UnifiedProductSelection())==1
    assert r.solver.identity==SCALABLE
    with pytest.raises(ValueError,match='12,288'):w._validate(replace(r,solver=legacy_solver(r.spatial)),w.UnifiedProductSelection())
    with pytest.raises(ValueError,match='envelope'):w._validate(geometry(r,(4096,4096),'cupy'),w.UnifiedProductSelection())
    assert w._validate(replace(r,backend='numpy'),w.UnifiedProductSelection())==1
    with pytest.raises(ValueError,match='envelope'):w._validate(geometry(r,(4096,4096),'numpy'),w.UnifiedProductSelection())
    with pytest.raises(ValueError,match='solver/precision'):w._validate(replace(r,precision=PRMaterialPrecisionSpec(MIXED_PRECISION,'float32',output_dtype='float32')),w.UnifiedProductSelection())


@pytest.mark.parametrize('dimension',[1,2])
def test_legacy_codec_and_explicit_rejection(dimension):
    r=replace(request(1,dimension=dimension),persistence_schema=codec.LEGACY_REQUEST_SCHEMA)
    meta=codec.request_metadata(r)
    assert 'solver' not in meta
    decoded=codec.decode_request(codec.encode_request(r))
    # Stored config remains the exact legacy vocabulary.
    assert decoded.config==meta
    out=codec._request(meta,r.initial_A)
    assert out.solver.identity==(REDUCED if dimension==1 else DIRECT)
    assert out.precision==r.precision
    if dimension==2:
        with pytest.raises(ValueError,match='legacy request'):codec.encode_request(scalable(r))
        modern=replace(scalable(r),persistence_schema=codec.REQUEST_SCHEMA)
        assert codec.decode_request(codec.encode_request(modern)).materialize().solver==SOLVER
        result=run_unified_products(modern)
        assert codec.decode_result(codec.encode_result(result)).scientific.status=='completed'


def test_failure_retains_previous_acceptance(monkeypatch):
    from lcprop.pr.unified import scalable_workflow as adapter
    from lcprop.pr.unified._krylov import Failure
    original=adapter.solve_material;calls=[];observed=[]
    cause=np.linalg.LinAlgError('native fault');error=Failure('linear fault',state={'marker':1},trace=[{'iteration':0}])
    error.stage='gmres'
    def fail_second(*a,**k):
        calls.append(1)
        if len(calls)==2:raise error from cause
        return original(*a,**k)
    monkeypatch.setattr(adapter,'solve_material',fail_second)
    out=run_unified_products(scalable(request(2,dimension=2,scatter=True)),selection=RICH,observer=observed.append)
    s=out.scientific
    assert s.status=='failed' and s.completed_cells==1 and s.reached_z_um==2
    assert len(observed)==1 and len(s.ledger)==1 and calls==[1,1]
    assert s.failure['material_exception'] is error and error.__cause__ is cause
    assert error.state=={'marker':1} and error.trace==[{'iteration':0}]
    assert out.arrays['intensity_volume'].shape[0]==2
    assert out.arrays['carrier_node_volume'].shape[0]==1


@pytest.mark.parametrize('precision',['float32','float64'])
def test_anisotropic_bridge(precision):
    r=request(1,precision,2)
    shape=(384,32)
    A=np.broadcast_to(1+.01*np.cos(2*np.pi*np.arange(384)[:,None]/384),shape)[None].astype(r.initial_A.dtype).copy()
    r=replace(geometry(r,shape,'numpy'),initial_A=A)
    a=run_unified_products(r,selection=RICH)
    b=run_unified_products(scalable(r),selection=RICH)
    assert a.scientific.status==b.scientific.status=='completed',(a.scientific.failure,b.scientific.failure)
    rt,at=(2e-4,2e-5) if precision=='float32' else (1e-8,1e-8)
    for key in a.arrays:compare_product(key,b.arrays[key],a.arrays[key],a,r,rt,at)
    np.testing.assert_allclose(b.scientific.boundary_field,a.scientific.boundary_field,rtol=rt,atol=at)


def test_scalable_exact_order(monkeypatch):
    from lcprop.pr.unified import scalable_workflow as adapter
    events=[]
    for module,name,label in [(w,'hop_linear_inplace','P'),(w,'pr_driving_intensity','I'),
        (adapter,'solve_with_diagnostics','material'),(w,'electric_field_optical_node','projection'),
        (w,'apply_response_screen_inplace','phase'),(w,'canonical_scattering_phase_increment','S'),
        (w,'_prepare_acceptance','bookkeeping')]:
        original=getattr(module,name)
        def wrap(*a,_original=original,_label=label,**k):
            events.append(_label)
            return _original(*a,**k)
        monkeypatch.setattr(module,name,wrap)
    out=w.run_unified_static(scalable(request(1,dimension=2,scatter=True)),observer=lambda event:events.append('accepted'))
    assert out.status=='completed',out.failure
    assert events==['P','I','material','projection','phase','S','phase','bookkeeping','accepted']


def test_cancelled_v3_empty_carrier_volume():
    from lcprop.core.execution import CancellationToken
    token=CancellationToken();token.cancel()
    out=run_unified_products(scalable(request(2,'float32',2)),selection=RICH,cancellation_token=token)
    assert out.scientific.status=='cancelled' and out.scientific.completed_cells==0
    assert out.arrays['carrier_node_volume'].shape[0]==0
    assert out.arrays['carrier_node_volume'].dtype==np.float64


def test_legacy_workflow_bytes_against_committed_driver():
    import subprocess,types,sys
    name='lcprop.pr.unified._pre_s4m2_workflow'
    old=types.ModuleType(name);old.__package__='lcprop.pr.unified';sys.modules[name]=old
    source=subprocess.check_output(['git','show','4eca4a116babf901b85cfa3cddd663b4ca1ace88:src/lcprop/pr/unified/workflow.py'],text=True)
    exec(compile(source,name,'exec'),old.__dict__)
    for dimension in (1,2):
        for precision in ('float32','float64'):
            r=request(2,precision,dimension,True)
            kwargs=dict(vars(r));kwargs.pop('solver');kwargs.pop('persistence_schema')
            a=old.run_unified_static(old.UnifiedStaticRequest(**kwargs),selection=old.UnifiedProductSelection(material_state=True,carrier=True,far_field=True))
            b=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True,carrier=True,far_field=True))
            assert a.status==b.status=='completed'
            assert a.identities==b.identities and a.ledger==b.ledger
            for field in ('q','psi','b'):np.testing.assert_array_equal(getattr(a.material_state,field),getattr(b.material_state,field))
            np.testing.assert_array_equal(a.boundary_field,b.boundary_field)
            np.testing.assert_array_equal(a.products['far_field'].intensity,b.products['far_field'].intensity)

"""Dedicated schemas, no legacy loader or continuation semantics."""
from dataclasses import replace
import io
import json
import zipfile
import numpy as np
import pytest

from tests.test_pr_unified_workflow import request
from tests.test_pr_unified_products import all_selection,digest
from lcprop.core.execution import CancellationToken
from lcprop.pr.unified import codec as c
from lcprop.pr.unified import products as p
from lcprop.pr.unified import workflow as w
from lcprop.pr.unified.specs import UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT,OPEN_TRANSVERSE,PRElectricalClosureSpec


def rewrite(payload,change):
    source=zipfile.ZipFile(io.BytesIO(payload));metadata=json.loads(source.read('metadata.json'))
    change(metadata);out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as target:
        for name in source.namelist():
            target.writestr(name,json.dumps(metadata) if name=='metadata.json' else source.read(name))
    return out.getvalue()


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('dim,closure',[(1,UNBIASED),(1,FIXED_FIELD),(1,PRESCRIBED_CURRENT),(2,UNBIASED),(2,FIXED_FIELD),(2,PRESCRIBED_CURRENT),(2,OPEN_TRANSVERSE)])
def test_request_result_roundtrip(dtype,dim,closure,tmp_path):
    r=request(1,dtype,dim,scatter=True,closure=closure);selection=all_selection(dim)
    c.save_request(tmp_path/'request.pru',r,selection=selection)
    stored=c.load_request(tmp_path/'request.pru')
    restored=stored.materialize()
    assert c.request_metadata(restored)==c.request_metadata(r)
    np.testing.assert_array_equal(restored.initial_A,r.initial_A)
    assert stored.selection==selection
    result=p.run_unified_products(restored,selection=stored.selection)
    assert result.scientific.status=='completed',result.scientific.failure
    c.save_result(tmp_path/'result.pru',result);loaded=c.load_result(tmp_path/'result.pru')
    assert loaded.schema==p.PRODUCTS_SCHEMA and loaded.selection==selection
    assert loaded.locations==result.locations and loaded.next_address==result.next_address
    assert loaded.scientific.identities==result.scientific.identities
    assert loaded.scientific.ledger==result.scientific.ledger
    assert digest(loaded)==digest(result)
    for group in ('arrays','coordinates'):
        assert set(getattr(loaded,group))==set(getattr(result,group))
        for name,array in getattr(result,group).items():
            restored_array=getattr(loaded,group)[name]
            assert restored_array.dtype==array.dtype and restored_array.shape==array.shape
            assert restored_array.tobytes()==array.tobytes()


def test_a7_provenance_and_stored_backend_is_not_runtime():
    r=request(1);r=replace(r,closure=PRElectricalClosureSpec.a7(.2,.01),material=replace(r.material,applied_field=.2))
    payload=c.encode_request(r)
    payload=rewrite(payload,lambda m:m['config'].update(backend='cupy'))
    stored=c.decode_request(payload)
    assert stored.config['backend']=='cupy' and isinstance(stored.launch,np.ndarray)
    cpu=stored.materialize(backend='numpy')
    assert cpu.closure.reservoir_field==.2 and cpu.closure.background_intensity==.01 and cpu.closure.target==(.002,)
    assert cpu.backend=='numpy' and stored.config['backend']=='cupy'


@pytest.mark.parametrize('kind',['before','cancel','failure'])
def test_partial_accepted_state_persistence(kind,monkeypatch):
    r=request(3,scatter=True);token=CancellationToken();original=w._prepare_acceptance
    if kind=='before': token.cancel()
    def stop(ledger,*args):
        if len(ledger)==1:
            if kind=='failure': raise RuntimeError('bookkeeping failure')
            token.cancel()
        return original(ledger,*args)
    monkeypatch.setattr(w,'_prepare_acceptance',stop)
    result=p.run_unified_products(r,selection=all_selection(),cancellation_token=token)
    restored=c.decode_result(c.encode_result(result))
    assert restored.scientific.status==result.scientific.status
    assert restored.scientific.completed_cells==(0 if kind=='before' else 1)
    assert restored.scientific.reached_z_um==(0 if kind=='before' else 2)
    assert restored.scientific.failure==result.scientific.failure
    assert restored.next_address==result.next_address
    np.testing.assert_array_equal(restored.scientific.boundary_field,result.scientific.boundary_field)
    if kind!='before':assert digest(restored)==digest(result)


@pytest.mark.parametrize('change',[
    lambda m:m.update(schema='pr_static'),
    lambda m:m.update(normalization='fixed_I0'),
    lambda m:m['config'].update(workflow_identity='pr_static'),
    lambda m:m['config'].update(arithmetic_identity='midpoint'),
    lambda m:m['config'].update(projection_identity='centered_psi'),
    lambda m:m['config']['precision'].update(linear_dtype='float32'),
    lambda m:m['config']['closure'].update(identity='arbitrary_U_V'),
    lambda m:m['arrays']['launch'].update(dtype='complex64'),
    lambda m:m['arrays']['launch'].update(sha256='0'*64),
    lambda m:m['selection'].update(full_volume=True),
])
def test_bad_request_records_fail(change):
    with pytest.raises(ValueError):c.decode_request(rewrite(c.encode_request(request(1)),change))


@pytest.fixture(scope='module')
def result_payload():return c.encode_result(p.run_unified_products(request(1,scatter=True),selection=all_selection()))


@pytest.mark.parametrize('change',[
    lambda m:m.update(schema='pr_static_result'),
    lambda m:m.update(products_schema='future'),
    lambda m:m['scientific'].update(completed_cells=0),
    lambda m:m['scientific'].update(status='cancelled',failure={'stage':'x'}),
    lambda m:m['scientific'].update(reached_z_um=1.),
    lambda m:m['scientific']['material_z_um'].__setitem__(0,1.),
    lambda m:m['scientific']['ledger'][0].update(canonical_slabs=[0,3]),
    lambda m:m['scientific']['ledger'][0]['observations'].update({'column_0.gauss_max':1.}),
    lambda m:m['next_address'].update(cell_index=0),
    lambda m:m['state'].update(gauge='arbitrary'),
    lambda m:m['state'].update(state_identity='E_only'),
    lambda m:m['locations']['electric_field_x_face'].update(offset_um=[0.,0.]),
    lambda m:m['locations']['electric_field_x_optical_node'].update(projection='new'),
    lambda m:m['arrays']['state/q'].update(shape=[64]),
])
def test_bad_result_records_fail(result_payload,change):
    with pytest.raises(ValueError): c.decode_result(rewrite(result_payload,change))


def test_failed_launch_and_product_error_roundtrip(monkeypatch):
    r=request(1);a=r.initial_A.copy();a[0,0,0]=np.nan
    result=p.run_unified_products(replace(r,initial_A=a),selection=all_selection())
    loaded=c.decode_result(c.encode_result(result))
    assert loaded.scientific.boundary_field is None and loaded.scientific.completed_cells==0
    def fail(request,selection,result,xp,arrays,coordinates,collection):
        coordinates.update(x_um=collection['x'],y_um=collection['y'])
        raise RuntimeError('postprocess failure')
    monkeypatch.setattr(p,'_finish_products',fail)
    result=p.run_unified_products(r,selection=p.UnifiedSelection(optical_cuts=True))
    loaded=c.decode_result(c.encode_result(result))
    assert loaded.scientific.status=='failed' and loaded.scientific.completed_cells==1
    assert loaded.scientific.failure['stage']=='selected_products'


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_native_selected_codec_when_available(dtype):
    cp=pytest.importorskip('cupy')
    try: cp.zeros(1)
    except Exception: pytest.skip('CuPy/CUDA unavailable')
    r=request(1,dtype);r=replace(r,initial_A=cp.asarray(r.initial_A),backend='cupy')
    result=p.run_unified_products(r,selection=all_selection())
    assert result.scientific.status=='completed',result.scientific.failure
    restored=c.decode_result(c.encode_result(result))
    assert restored.scientific.identities['backend']=='cupy'
    assert restored.scientific.material_state.backend=='numpy'
    for name,value in result.arrays.items():
        assert restored.arrays[name].dtype==value.dtype
        assert restored.arrays[name].tobytes()==cp.asnumpy(value).tobytes()


@pytest.fixture(scope='module')
def accepted_provenance_records():
    complete=p.run_unified_products(request(2),selection=p.UnifiedSelection(launch=True))
    token=CancellationToken();token.cancel()
    launch_only=p.run_unified_products(request(2),cancellation_token=token)
    partial=p.run_unified_products(request(1))
    # A one-cell accepted prefix of the two-cell request, with unchanged science.
    ids=dict(partial.scientific.identities,grid=complete.scientific.identities['grid'])
    partial=replace(partial,scientific=replace(partial.scientific,
        identities=ids,status='cancelled',reason='cancelled after accepted cell'))
    failed=replace(partial,scientific=replace(partial.scientific,status='failed',reason='diagnostic failure',
        failure=dict(stage='optical_hop',cell_index=1,type='RuntimeError',reason='diagnostic failure')))
    return dict(completed=complete,partial=partial,failed=failed,cancelled=launch_only)


BAD_PROVENANCE = [
    ('missing_groups',lambda d:d.pop('coherence_groups')),
    ('few_groups',lambda d:d.update(coherence_groups=[])),
    ('many_groups',lambda d:d.update(coherence_groups=['a','b'])),
    ('string_groups',lambda d:d.update(coherence_groups='a')),
    ('null_groups',lambda d:d.update(coherence_groups=None)),
    ('nested_groups',lambda d:d.update(coherence_groups=[['a']])),
    ('empty_group',lambda d:d.update(coherence_groups=[' '])),
    ('numeric_group',lambda d:d.update(coherence_groups=[3])),
    ('missing_peak',lambda d:d.pop('peak_intensity_reference')),
    ('nan_peak',lambda d:d.update(peak_intensity_reference=float('nan'))),
    ('inf_peak',lambda d:d.update(peak_intensity_reference=float('inf'))),
    ('zero_peak',lambda d:d.update(peak_intensity_reference=0.)),
    ('negative_peak',lambda d:d.update(peak_intensity_reference=-1.)),
    ('string_peak',lambda d:d.update(peak_intensity_reference='1')),
    ('bool_peak',lambda d:d.update(peak_intensity_reference=True)),
    ('list_peak',lambda d:d.update(peak_intensity_reference=[1.])),
]


@pytest.mark.parametrize('status',['completed','partial','failed','cancelled'])
@pytest.mark.parametrize('name,change',BAD_PROVENANCE,ids=[v[0] for v in BAD_PROVENANCE])
def test_accepted_provenance_required_on_encode_and_decode(accepted_provenance_records,status,name,change):
    result=accepted_provenance_records[status]
    payload=c.encode_result(result)
    with pytest.raises(ValueError):
        c.decode_result(rewrite(payload,lambda m:change(m['scientific']['identities'])))
    ids=dict(result.scientific.identities);change(ids)
    with pytest.raises(ValueError):
        c.encode_result(replace(result,scientific=replace(result.scientific,identities=ids)))


@pytest.mark.parametrize('status',['completed','cancelled'])
def test_retained_launch_reference_consistency(accepted_provenance_records,status):
    result=accepted_provenance_records[status]
    change=lambda d:d.update(peak_intensity_reference=d['peak_intensity_reference']*2)
    with pytest.raises(ValueError):
        c.decode_result(rewrite(c.encode_result(result),lambda m:change(m['scientific']['identities'])))
    ids=dict(result.scientific.identities);change(ids)
    with pytest.raises(ValueError):c.encode_result(replace(result,scientific=replace(result.scientific,identities=ids)))


@pytest.mark.parametrize('status',['failed','cancelled'])
def test_no_accepted_launch_may_omit_derived_provenance(status):
    r=request(1);a=r.initial_A.copy();a[:]=np.nan
    result=p.run_unified_products(replace(r,initial_A=a))
    ids=dict(result.scientific.identities)
    ids.pop('coherence_groups',None);ids.pop('peak_intensity_reference',None)
    s=replace(result.scientific,identities=ids,status=status,
        failure=result.scientific.failure if status=='failed' else None)
    loaded=c.decode_result(c.encode_result(replace(result,scientific=s)))
    assert loaded.scientific.boundary_field is None
    assert 'coherence_groups' not in loaded.scientific.identities
    assert 'peak_intensity_reference' not in loaded.scientific.identities


def test_evolved_endpoint_is_not_used_as_launch_reference(accepted_provenance_records):
    result=accepted_provenance_records['partial']
    loaded=c.decode_result(c.encode_result(result))
    assert loaded.scientific.identities==result.scientific.identities


def test_multichannel_coherence_provenance_roundtrip():
    r=request(1)
    a=np.concatenate((r.initial_A,.5j*r.initial_A),axis=0)
    r=replace(r,initial_A=a,coherence_groups=('shared','shared'))
    result=p.run_unified_products(r,selection=p.UnifiedSelection(launch=True))
    loaded=c.decode_result(c.encode_result(result))
    assert loaded.scientific.identities==result.scientific.identities
    assert loaded.scientific.identities['coherence_groups']==('shared','shared')
    np.testing.assert_array_equal(loaded.scientific.boundary_field,result.scientific.boundary_field)

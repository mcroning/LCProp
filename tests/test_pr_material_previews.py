"""Observational product parity, bounded retention, and installed-viewer contracts."""
from dataclasses import replace
import hashlib
import numpy as np
import pytest
from lcprop.pr import material_previews as mp
from lcprop.pr.unified import products,integration as a
from tests.test_pr_unified_workflow import request
from tests.test_pr_unified_products import digest
from tests.test_pr_unified_integration import fresh,app,window,apply
from tests.test_pr_unified_scalable_workflow import scalable,SOLVER
from lcprop.core.execution import CancellationToken
from lcprop.gui.workspace import Workspace

@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('dim',[1,2])
def test_static_preview_is_observational(dtype,dim):
    r=request(2,dtype,dim,scatter=True)
    if dim==2:r=scalable(r)
    sel=a.selection_for_policy('fast')
    before=products.run_unified_products(r,selection=sel)
    after=products.run_unified_products(r,selection=sel,preview=True)
    assert after.scientific.status=='completed',after.scientific.failure
    assert digest(before)==digest(after)
    for key in before.arrays:assert before.arrays[key].tobytes()==after.arrays[key].tobytes()
    expected={'optical_intensity','potential_node','carrier_node','electric_field_x_optical_node','electric_field_x_face'}
    if dim==2:expected.add('electric_field_y_face')
    assert set(after.viewer_previews)==expected
    for name,p in after.viewer_previews.items():
        assert p['metadata']['coordinates']['z']==([0.,2.,4.] if name=='optical_intensity' else [2.,4.])
        assert p['data'].nbytes<=mp.FAST_MPR_TARGET_BYTES
    if dim==2 and dtype=='float32':assert after.viewer_previews['carrier_node']['data'].dtype==np.float64

@pytest.mark.parametrize('policy',['minimal','fast','interactive','full','analysis:unified_face_x_volume'])
def test_policy_codec_and_viewer(app,policy):
    r=fresh(2);out=a.execute_unified(r,result_policy=policy)
    enc=a._encode_result(out);reopened=a._decode_result(enc.payload.metadata,enc.payload.arrays)
    assert digest(out.run)==digest(reopened.run)
    assert bool(out.run.viewer_previews)==(policy=='fast')
    assert set(out.run.viewer_previews)==set(reopened.run.viewer_previews)
    w=Workspace();data=a.unified_to_run_data(reopened);w.set_run_data(data)
    for name,rec in reopened.run.viewer_previews.items():
        key='preview_'+name
        index=w.image_pane.field_selector.findData(key+'_xy');assert index>=0
        w.image_pane.field_selector.setCurrentIndex(index)
        w.longitudinal_pane.select_volume(key)
        assert w.longitudinal_pane.field_selector.currentData()==key
        np.testing.assert_array_equal(data.fields[key].data,rec['data'])
    w.close()

def test_partial_does_not_keep_candidate_preview(monkeypatch):
    r=request(3,dimension=2);token=CancellationToken();original=products._Collector.prepare
    def hook(self,prev,A,state,I,g,record,ids):
        c=original(self,prev,A,state,I,g,record,ids)
        if record is not None and record['cell_index']==1:token.cancel()
        return c
    monkeypatch.setattr(products._Collector,'prepare',hook)
    out=products.run_unified_products(r,selection=a.selection_for_policy('fast'),preview=True,cancellation_token=token)
    assert out.scientific.completed_cells==1
    for n,p in out.viewer_previews.items():assert p['metadata']['coordinates']['z']==([0.,2.] if n=='optical_intensity' else [2.])

@pytest.mark.parametrize('dtype',['float32','float64'])
def test_256_style_preview_bound_and_block_means(dtype):
    plane=np.arange(256*256,dtype=dtype).reshape(256,256)
    out,xe,ye=mp.reduce_plane(plane,np)
    reference=np.array([[plane[x0:x1,y0:y1].mean(dtype=np.float64) for y0,y1 in zip(ye[:-1],ye[1:])] for x0,x1 in zip(xe[:-1],xe[1:])],dtype=dtype)
    np.testing.assert_array_equal(out,reference)
    assert out.shape==(96,96)
    indices=mp.plane_indices(400,(256,256),out.itemsize)
    assert len(indices)*out.nbytes<=mp.FAST_MPR_TARGET_BYTES
    rec=mp.record(np.stack([out]*len(indices)),name='psi',x=np.arange(256),y=np.arange(256),z=indices,source_shape=(400,256,256),indices=indices,xe=xe,ye=ye)
    bad=dict(rec,data=np.zeros((100,256,256),dtype=dtype))
    with pytest.raises(ValueError,match='budget'):mp.validate({'psi':bad})

@pytest.mark.parametrize('transverse',[False,True])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_td_codec_preview_no_science_mutation(transverse,dtype,app):
    if transverse:
        from tests.test_pr_transverse_timedependent_transport import _request,PR_TRANSVERSE_TIMEDEPENDENT_OPERATION as op
        from lcprop.pr.transverse.timedependent_transport_codec import encode_pr_transverse_timedependent_transport_result as encode,decode_pr_transverse_timedependent_transport_result as decode
    else:
        from tests.test_pr_timedependent_transport import _request,PR_TIMEDEPENDENT_OPERATION as op
        from lcprop.pr.timedependent_transport_codec import encode_pr_timedependent_transport_result as encode,decode_pr_timedependent_transport_result as decode
    state=op.run(_request(precision=dtype,steps=1))
    attrs=['A_initial','A_final','source_intensity_stack']+(['psi_initial','psi_final'] if transverse else ['E_initial','E_final'])
    before={k:getattr(state,k).tobytes() for k in attrs}
    encoded=encode(state,'fast');out=decode(encoded.payload.metadata,encoded.payload.arrays)
    assert all(getattr(state,k).tobytes()==v for k,v in before.items())
    assert out.A_final.tobytes()==state.A_final.tobytes()
    previews=out.diagnostics['material_previews']
    assert set(previews)==({'psi','E_x','E_y','carrier'} if transverse else {'E_x'})
    data=op.to_run_data(out);w=Workspace();w.set_run_data(data)
    for preview in previews.values():
        assert preview['metadata']['coordinates']['z']==out.intensity_preview_metadata['preview_coordinates_um']['z']
    for name,p in previews.items():
        key='preview_'+name
        assert w.image_pane.field_selector.findData(key+'_xy')>=0
        w.image_pane.field_selector.setCurrentIndex(w.image_pane.field_selector.findData(key+'_xy'))
        w.longitudinal_pane.select_volume(key)
        assert w.longitudinal_pane.field_selector.currentData()==key
        assert p['data'].dtype==np.dtype(dtype)
    again=encode(out,'fast');final=decode(again.payload.metadata,again.payload.arrays)
    for k in previews:np.testing.assert_array_equal(previews[k]['data'],final.diagnostics['material_previews'][k]['data'])
    w.close()

def test_physical_labels_and_dimension_availability(window):
    apply(window,fresh(1))
    assert not window._analysis_actions['unified_face_y_volume'].isEnabled()
    assert 'Material potential' in window._analysis_actions['unified_potential_volume'].text()
    assert 'Projected material response' in window._analysis_actions['unified_optical_field_volume'].text()
    apply(window,replace(fresh(2),solver=SOLVER))
    assert window._analysis_actions['unified_face_y_volume'].isEnabled()
    assert window.result_policy_selector.findData('minimal')>=0
    assert window.result_policy_selector.itemText(window.result_policy_selector.findData('fast'))=='Fast / Exploratory'

def test_fast_static_256_metadata_and_synthetic_accepted_fields(app,monkeypatch):
    from lcprop.core.backend import BackendSpec
    from lcprop.pr.unified.state import PRUnifiedMaterialState
    from lcprop.core.grid import make_grid
    r=replace(fresh(2,precision='float32'),solver=SOLVER,backend=BackendSpec('cupy','float32',False))
    r=replace(r,grid=replace(r.grid,Nx=256,Ny=256))
    plan=a.resource_plan(r,'fast');assert plan['presentation']['fast_material_and_optical_preview_upper_bytes']==6*mp.FAST_MPR_TARGET_BYTES
    core=a.core_request(r);core=replace(core,backend='numpy')
    state=PRUnifiedMaterialState(q=np.zeros((256,256),np.float32),psi=np.zeros((256,256),np.float32),b=np.zeros(2,np.float32),spatial=core.spatial,closure=core.closure,precision=core.precision,backend='numpy')
    g=make_grid(core.grid,real_dtype=np.float32)
    ids=dict(backend='numpy',coherence_groups=['a'],material_parameters={'gain_length_product':.1},grid={'z_length_um':core.grid.z_length_um},wavelength_um=.633)
    A=np.ones((1,256,256),np.complex64);I=np.ones((256,256),np.float32)
    builder=mp.StaticPreviews(core);c=builder.prepare(None,A,None,None,g,None,ids,np)
    c=builder.prepare(c,A,state,I,g,dict(cell_index=0,z_end_um=2.,material_weight_um=2.),ids,np)
    from types import SimpleNamespace
    import lcprop.core.backend as backend
    transferred=[]
    def guarded(value):
        assert value.ndim!=3 or max(value.shape[1:])<=96,'full-plane preview transfer'
        transferred.append(value.nbytes);return value
    monkeypatch.setattr(backend,'asnumpy',guarded)
    records=builder.finish(dict(bounded_previews=c,x=g.x_um,y=g.y_um),SimpleNamespace(completed_cells=1),np)
    assert set(records)=={'optical_intensity','potential_node','carrier_node','electric_field_x_optical_node','electric_field_x_face','electric_field_y_face'}
    assert max(transferred)<=2*96*96*8
    from lcprop.products.data_model import RunData,FieldCollection
    data=mp.add_fields(RunData('synthetic',fields=FieldCollection()),records);w=Workspace();w.set_run_data(data)
    for name in records:
        assert w.image_pane.field_selector.findData('preview_'+name+'_xy')>=0
        w.image_pane.field_selector.setCurrentIndex(w.image_pane.field_selector.findData('preview_'+name+'_xy'))
        w.longitudinal_pane.select_volume('preview_'+name)
        assert w.longitudinal_pane.field_selector.currentData()=='preview_'+name
    w.close()
    # A broken reducer must fail before any full-resolution host transfer.
    c['potential_node'][0]=(2.,np.zeros((256,256),np.float32),np.arange(257),np.arange(257))
    n=len(transferred)
    with pytest.raises(ValueError,match='transfer exceeds'):builder.finish(dict(bounded_previews={'potential_node':c['potential_node']},x=g.x_um,y=g.y_um),SimpleNamespace(completed_cells=1),np)
    assert len(transferred)==n

def test_result_provenance_independent_of_next_selection(window):
    out=a.execute_unified(fresh(),result_policy='fast')
    assert mp.displayed_policy(out)=='Fast / Exploratory'
    window._set_product_policy('full')
    assert mp.displayed_policy(out)=='Fast / Exploratory'
    encoded=a._encode_result(out);reopened=a._decode_result(encoded.payload.metadata,encoded.payload.arrays)
    assert mp.displayed_policy(reopened)=='Fast / Exploratory'

@pytest.mark.parametrize('reason',['snapshot verification failed','SSH authentication failed before sbatch'])
def test_remote_pre_submission_failure_never_running(window,app,monkeypatch,reason):
    import time
    from types import SimpleNamespace
    import lcprop.pr.gui.main_window as module
    monkeypatch.setattr(module,'report_failure',lambda *a,**k:None)
    fake=SimpleNamespace(name='Mock Slurm');window.slurm_runner=fake
    seen=[];original=window.results_panel.workspace.set_operation_status
    def status(value):seen.append(value);return original(value)
    monkeypatch.setattr(window.results_panel.workspace,'set_operation_status',status)
    def fail(*a,**k):raise RuntimeError(reason)
    window._start_background(fresh(),summary='metadata fixture',run_label='Running',runner_callable=fail,execution_runner=fake)
    deadline=time.monotonic()+5
    while window._background_running and time.monotonic()<deadline:
        app.processEvents();time.sleep(.005)
    assert not window._background_running
    assert 'Validating/Preparing' in seen
    assert not any('running' in s.lower() for s in seen)

@pytest.mark.parametrize('policy',['minimal','fast','interactive','full'])
def test_normal_package_save_reopen(policy,tmp_path):
    from lcprop.transport.defaults import default_transport_registry
    from lcprop.transport.io import write_request_package,read_request_package,write_result_package,read_result_package
    registry=default_transport_registry();r=fresh(2);out=a.execute_unified(r,result_policy=policy)
    path=tmp_path/policy
    write_request_package(path,registry=registry,material_id='pr',workflow_id=a.WORKFLOW_ID,request=r,run_id='preview-test',execution_target='slurm',result_policy=policy)
    req=read_request_package(path,registry=registry)
    write_result_package(path,codec=req.codec,result=out,request_envelope=req.envelope)
    loaded=read_result_package(path,registry=registry).result
    assert digest(out.run)==digest(loaded.run)
    assert loaded.result_policy==policy
    for n,v in out.run.viewer_previews.items():
        np.testing.assert_array_equal(v['data'],loaded.run.viewer_previews[n]['data'])
        assert v['metadata']==loaded.run.viewer_previews[n]['metadata']

def test_displayed_result_policy_stays_when_next_request_changes(window):
    r=fresh();apply(window,r);window._set_product_policy('fast')
    out=window._run_registered(r);window._on_finished(out)
    assert 'Fast / Exploratory' in window.results_panel.workspace.operation_status.text()
    window._set_product_policy('full')
    assert 'Fast / Exploratory' in window.results_panel.workspace.operation_status.text()

def test_remote_lifecycle_presentation_distinguishes_phases(window):
    from lcprop.transport.status import RemoteRunStatus,RemoteRunState
    for state in (RemoteRunState.SUBMITTING,RemoteRunState.PENDING,RemoteRunState.RUNNING,RemoteRunState.RETRIEVING,RemoteRunState.COMPLETED):
        window._on_progress(RemoteRunStatus('fixture','slurm',state=state))
        assert window.results_panel.workspace.operation_status.text()=='Execution status: '+state.value.title()

@pytest.mark.parametrize('dtype',['float32','float64'])
def test_reused_backend_preview_arithmetic(dtype):
    class Backend:
        def __getattr__(self,key):return getattr(np,key)
    rng=np.random.default_rng(12);a=rng.normal(size=(257,201)).astype(dtype);before=a.tobytes()
    expected,_,_=mp.reduce_plane(a,np);actual,_,_=mp.reduce_plane(a,Backend())
    assert expected.tobytes()==actual.tobytes()
    assert a.tobytes()==before

@pytest.mark.parametrize('dtype',['float32','float64'])
def test_native_preview_reduction_when_available(dtype):
    try:
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():pytest.skip('CuPy/CUDA unavailable')
    except ImportError:pytest.skip('CuPy/CUDA unavailable')
    except Exception:pytest.skip('CuPy/CUDA unavailable')
    a=np.random.default_rng(21).normal(size=(257,201)).astype(dtype)
    expected,_,_=mp.reduce_plane(a,np);actual,_,_=mp.reduce_plane(cp.asarray(a),cp)
    np.testing.assert_array_equal(cp.asnumpy(actual),expected)

def test_legacy_optical_catalog_does_not_expand_with_unified_fields():
    from lcprop.pr.selected_products import OPTICAL_PRODUCTS
    from lcprop.transport.result_policy import ANALYSIS_PRODUCTS
    assert OPTICAL_PRODUCTS==('input_intensity','output_intensity','far_field_intensity','complex_input','complex_output')
    assert set(OPTICAL_PRODUCTS)<set(ANALYSIS_PRODUCTS)

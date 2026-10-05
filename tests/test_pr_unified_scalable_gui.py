"""S4-M3 schemas/planning/GUI only; bounded CPU and metadata-only GPU targets."""
from dataclasses import replace,asdict
import io,json,zipfile,hashlib
import numpy as np
import pytest
from tests.test_pr_unified_scalable_workflow import request,scalable,RICH,SOLVER
from tests.test_pr_unified_integration import fresh,app,window,apply
from lcprop.pr.unified import codec,products,integration as a,workflow as w
from lcprop.pr.unified.solver_specs import *
from lcprop.pr.unified.specs import *
from lcprop.core.backend import BackendSpec
from lcprop.core.execution import CancellationToken


def meta(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:return json.loads(z.read('metadata.json'))


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('mode',['minimal','full','cancel','failure'])
def test_scalable_archive(precision,mode,monkeypatch):
    r=scalable(request(2,precision,2,True));token=CancellationToken()
    if mode=='cancel':token.cancel()
    if mode=='failure':
        from lcprop.pr.unified import scalable_workflow as adapter
        from lcprop.pr.unified._krylov import Failure
        original=adapter.solve_material;count=0
        def fail(*a,**k):
            nonlocal count
            count+=1
            if count==2:
                exc=Failure('linear stop',trace=[{'iteration':0,'linear':[{'true_relative':.5}]}]);exc.stage='gmres'
                raise exc from np.linalg.LinAlgError('backend stop')
            return original(*a,**k)
        monkeypatch.setattr(adapter,'solve_material',fail)
    selected=products.UnifiedSelection() if mode=='minimal' else RICH
    result=products.run_unified_products(r,selection=selected,cancellation_token=token)
    expected={'minimal':'completed','full':'completed','cancel':'cancelled','failure':'failed'}[mode]
    assert result.scientific.status==expected,result.scientific.failure
    packet=codec.encode_result(result);out=codec.decode_result(packet)
    assert meta(packet)['schema']==codec.RESULT_SCHEMA and out.schema==products.PRODUCTS_SCHEMA
    assert out.scientific.identities['solver']==asdict(SOLVER)
    assert out.scientific.identities['precision']['identity']==r.precision.identity
    assert out.scientific.status==expected and out.scientific.completed_cells==result.scientific.completed_cells
    for k in result.arrays:
        assert result.arrays[k].dtype==out.arrays[k].dtype
        np.testing.assert_array_equal(result.arrays[k],out.arrays[k])
        if k.startswith('carrier_node'):assert out.arrays[k].dtype==np.float64
    for k in result.coordinates:np.testing.assert_array_equal(result.coordinates[k],out.coordinates[k])
    np.testing.assert_array_equal(result.scientific.boundary_field,out.scientific.boundary_field)
    if out.scientific.material_state:
        for k in ('q','psi','b'):np.testing.assert_array_equal(getattr(result.scientific.material_state,k),getattr(out.scientific.material_state,k))
    if mode=='failure':
        failure=out.scientific.failure['material_failure']
        assert failure['stage']=='gmres' and failure['cause']['type']=='LinAlgError'
        assert failure['trace'][0]['linear'][0]['true_relative']==.5
    again=codec.decode_result(codec.encode_result(out));assert again.schema==out.schema
    stored=codec.decode_request(codec.encode_request(r))
    assert stored.materialize().solver==SOLVER


@pytest.mark.parametrize('dimension',[1,2])
def test_legacy_request_and_results_keep_origin(dimension):
    r=replace(request(1,dimension=dimension),persistence_schema=codec.LEGACY_REQUEST_SCHEMA)
    payload=codec.encode_request(r)
    assert 'solver' not in meta(payload)['config']
    stored=codec.decode_request(payload)
    assert stored.materialize().solver==legacy_solver(r.spatial)
    assert meta(codec.encode_request(stored.materialize()))['schema']==codec.LEGACY_REQUEST_SCHEMA
    for origin in ('pr_unified_static_products_v1','pr_unified_static_products_v2'):
        result=products.run_unified_products(r)
        identity=dict(result.scientific.identities);identity.pop('solver')
        result=replace(result,schema=origin,scientific=replace(result.scientific,identities=identity))
        for _ in range(3):
            payload=codec.encode_result(result);assert meta(payload)['schema']==codec.LEGACY_RESULT_SCHEMA
            result=codec.decode_result(payload);assert result.schema==origin
            assert 'solver' not in result.scientific.identities


@pytest.mark.parametrize('n,precision,backend',[(96,'float32','numpy'),(128,'float32','cupy'),(256,'float32','cupy'),(512,'float64','cupy')])
def test_metadata_envelopes(n,precision,backend,monkeypatch,window):
    monkeypatch.setattr(a,'get_backend',lambda *a:pytest.fail('GPU probe'))
    r=fresh(2,precision=precision);r=replace(r,solver=SOLVER,grid=replace(r.grid,Nx=n,Ny=n),backend=BackendSpec(backend,precision,False))
    apply(window,r);gui=window.build_request()
    assert a.encode_fresh(gui)==a.encode_fresh(r)
    a.validate_fresh(r);plan=a.resource_plan(r)
    summary=window.describe_request(gui)
    assert SCALABLE in summary and a.core_request(r).precision.identity in summary
    assert plan['support_envelope'] in summary and backend in summary
    assert plan['direct_solver'] is None and plan['scalable_solver']['workspace_components']['carrier64']==n*n*8
    assert plan['solver_identity']==SCALABLE
    assert plan['measured_native_peak'] is None
    if backend=='cupy':
        with pytest.raises(ValueError,match='12,288'):a.validate_fresh(replace(r,solver=PRUnifiedSolverSpec(DIRECT,DIRECT_POLICY)))
    with pytest.raises(ValueError,match='envelope'):a.validate_fresh(replace(r,grid=replace(r.grid,Nx=1024,Ny=1024)))


@pytest.mark.parametrize('kind',[UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT,OPEN_TRANSVERSE])
@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('identity',[DIRECT,SCALABLE])
def test_gui_headless_exact(window,kind,precision,identity,record_property):
    r=replace(fresh(2,kind,precision),solver=PRUnifiedSolverSpec(identity,ITERATIVE_POLICY if identity==SCALABLE else DIRECT_POLICY))
    apply(window,r);got=window.build_request()
    assert a.encode_fresh(got)==a.encode_fresh(r)
    assert a.core_request(got).precision.identity==(POSITIVE_PRECISION if identity==SCALABLE and precision=='float32' else MIXED_PRECISION if precision=='float32' else DOUBLE_PRECISION)
    decoded=a.decode_fresh(a.encode_fresh(got));assert decoded.solver==r.solver
    from tests.test_pr_unified_integration import science_equal
    gui=a.execute_unified(got,result_policy='fast');headless=a.execute_unified(r,result_policy='fast')
    assert gui.status==headless.status=='completed'
    science_equal(gui.run.scientific,headless.run.scientific)
    for key in ('far_field_intensity',):
        np.testing.assert_array_equal(gui.run.arrays[key],headless.run.arrays[key])
    for key in ('s_x','s_y'):
        np.testing.assert_array_equal(gui.run.coordinates[key],headless.run.coordinates[key])
    record_property('gui_headless_identity',json.dumps(dict(
        request_sha256=hashlib.sha256(json.dumps(a.encode_fresh(got),sort_keys=True).encode()).hexdigest(),
        endpoint_sha256=hashlib.sha256(gui.run.scientific.boundary_field.tobytes()).hexdigest(),
        farfield_sha256=hashlib.sha256(gui.run.arrays['far_field_intensity'].tobytes()).hexdigest(),
        solver=identity,precision=precision,closure=kind)))


def test_fresh_legacy_mapping():
    r=replace(fresh(2,precision='float32'),schema=a.LEGACY_FRESH_SCHEMA)
    payload=a.encode_fresh(r);assert 'solver' not in payload
    out=a.decode_fresh(payload);assert out.solver.identity==DIRECT
    assert a.core_request(out).precision.identity==MIXED_PRECISION
    assert a.encode_fresh(out)==payload


def test_carrier64_common_viewer_and_no_science_on_reopen(monkeypatch):
    r=replace(fresh(2,precision='float32'),solver=SOLVER)
    out=a.execute_unified(r,result_policy='full')
    assert out.status=='completed'
    archive=codec.encode_result(out.run)
    def forbidden(*a,**k):pytest.fail('reopen ran science')
    monkeypatch.setattr(w,'hop_linear_inplace',forbidden)
    loaded=codec.decode_result(archive)
    data=a.unified_to_run_data(a.UnifiedExecutionResult(loaded,'full','numpy'))
    carrier=loaded.arrays['carrier_node_volume']
    assert carrier.dtype==np.float64
    field=data.fields['carrier_node_volume']
    assert field.data.dtype==np.float64
    np.testing.assert_array_equal(field.data,carrier)
    corrupted=dict(loaded.arrays);corrupted['carrier_node_volume']=carrier.astype(np.float32)
    with pytest.raises(ValueError,match='precision'):codec.encode_result(replace(loaded,arrays=corrupted))


def test_gui_fresh_default_and_legacy_resave(window):
    r=replace(fresh(2),schema=a.LEGACY_FRESH_SCHEMA)
    apply(window,r)
    assert window.build_request().schema==a.LEGACY_FRESH_SCHEMA
    assert a.encode_fresh(window.build_request())==a.encode_fresh(r)
    window.evolution_panel._fresh_unified()
    assert window.build_request().solver.identity==SCALABLE
    assert window.build_request().schema==a.FRESH_SCHEMA


def test_transport_versions_still_registered():
    from lcprop.transport.defaults import default_transport_registry
    c=a.UNIFIED_TRANSPORT_CODEC
    assert c.supported_request_codec_versions==(2,1)
    assert c.supported_result_codec_versions==(2,1)
    default_transport_registry()


def test_failure_nonfinite_trace_is_explicit_metadata():
    from lcprop.pr.unified._krylov import Failure
    exc=Failure('breakdown',trace=[{'norm':float('inf')}]);exc.stage='pcg'
    result=codec._failure_metadata(dict(stage='material_equilibrium',cell_index=0,type='Failure',reason='breakdown',material_exception=exc))
    assert result['material_failure']['trace'][0]['norm']=={'nonfinite_float':'inf'}
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('key,value',[
    ('solver',dict(identity='auto',linear_policy=ITERATIVE_POLICY)),
    ('precision',dict(identity=MIXED_PRECISION,state_dtype='float32')),
])
def test_fresh_rejects_contradictory_or_unknown_metadata(key,value):
    r=replace(fresh(2,precision='float32'),solver=SOLVER)
    payload=a.encode_fresh(r);payload[key].update(value)
    with pytest.raises(ValueError):a.decode_fresh(payload)


@pytest.mark.parametrize('precision',['float32','float64'])
def test_partial_and_selection_identity(precision):
    r=replace(fresh(2,precision=precision),solver=SOLVER)
    full=a.execute_unified(r,result_policy='full')
    for policy in ('fast','interactive'):
        other=a.execute_unified(r,result_policy=policy)
        from tests.test_pr_unified_integration import science_equal
        science_equal(full.run.scientific,other.run.scientific)
    token=CancellationToken()
    partial=a.execute_unified(r,result_policy='full',cancellation_token=token,
        progress_callback=lambda event:token.cancel())
    assert partial.status=='cancelled' and partial.run.scientific.completed_cells==1
    restored=codec.decode_result(codec.encode_result(partial.run))
    assert restored.scientific.status=='cancelled'
    assert len(restored.coordinates['boundary_z_um'])==2
    assert len(restored.coordinates['material_z_um'])==1
    np.testing.assert_array_equal(restored.scientific.boundary_field,partial.run.scientific.boundary_field)


@pytest.mark.parametrize('target',['local','slurm'])
@pytest.mark.parametrize('identity',[SCALABLE,DIRECT,REDUCED])
def test_real_run_cost_preflight(window,monkeypatch,target,identity):
    from lcprop.pr.gui.run_cost import classify_pr_run_cost,LocalRunCostClass
    r=replace(fresh(1 if identity==REDUCED else 2),
        solver=PRUnifiedSolverSpec(identity,ITERATIVE_POLICY if identity==SCALABLE else DIRECT_POLICY))
    apply(window,r)
    selector=window.execution_target_selector
    if selector.findData(target)<0:selector.addItem(target,target)
    selector.blockSignals(True);selector.setCurrentIndex(selector.findData(target));selector.blockSignals(False)
    plan=a.resource_plan(r)
    assessed=classify_pr_run_cost(r,execution_target=target)
    expected=sum(plan['bytes'].values())
    if identity!=SCALABLE:expected+=plan['direct_solver']['dense_factorization_scenario_bytes']
    assert assessed.work_score==expected*plan['cells']
    if target=='slurm':assert assessed.classification==LocalRunCostClass.NORMAL
    boundaries=[]
    # Only external/source/runner availability is stubbed; actual Run, summary,
    # validation and cost guard execute, stopping before any scientific work.
    def validate(request):
        a.validate_fresh(request);boundaries.append(('preflight',target))
    monkeypatch.setattr(window,'_validate_execution_request',validate)
    monkeypatch.setattr(window,'_start_background',lambda request,**kwargs:boundaries.append(('dispatch',target)))
    window.run_clicked()
    assert boundaries==[('preflight',target),('dispatch',target)]


def test_real_run_unsupported_scalable(window,monkeypatch):
    import lcprop.pr.gui.main_window as gui
    r=replace(fresh(2),solver=SOLVER);apply(window,r)
    window.grid_panel.set_grid(replace(r.grid,Nx=1024,Ny=1024))
    errors=[]
    monkeypatch.setattr(gui,'report_failure',lambda *args:errors.append(str(args)))
    monkeypatch.setattr(window,'_start_background',lambda *a,**k:pytest.fail('unsupported dispatch'))
    window.run_clicked()
    assert len(errors)==1 and 'envelope' in errors[0]


@pytest.mark.parametrize('legacy',[True,False])
def test_failed_load_restores_provenance(window,monkeypatch,tmp_path,legacy):
    from lcprop.persistence import load_experiment,save_experiment
    original=replace(fresh(2),schema=a.LEGACY_FRESH_SCHEMA) if legacy else replace(fresh(2),solver=SOLVER)
    apply(window,original)
    first=window.save_experiment_to(tmp_path/'original.json');window.load_experiment_from(first)
    accepted=window.evolution_panel._loaded_unified_request
    before=a.encode_fresh(window.build_request());controls=window._capture_experiment_gui_state()
    other=replace(fresh(2),solver=SOLVER,grid=replace(original.grid,z_length_um=original.grid.z_length_um*2))
    second=save_experiment(other,tmp_path/'other.json',material_id='pr',workflow_id=a.WORKFLOW_ID)
    def fail(intent):
        assert window.evolution_panel._loaded_unified_request==other
        raise ValueError('injected after application')
    with monkeypatch.context() as m:
        m.setattr(window,'_restore_execution_intent',fail)
        with pytest.raises(ValueError,match='injected'):window.load_experiment_from(second)
    assert window.evolution_panel._loaded_unified_request is accepted
    assert window._capture_experiment_gui_state()==controls
    assert a.encode_fresh(window.build_request())==before
    saved=window.save_experiment_to(tmp_path/'saved.json')
    restored=load_experiment(saved,expected_material_id='pr').request
    assert a.encode_fresh(restored)==before and restored.schema==original.schema
    window.load_experiment_from(saved)
    assert a.encode_fresh(window.build_request())==before


@pytest.mark.parametrize('path',[('solver',),('precision',),('closure',),
    ('solver','identity'),('solver','linear_policy'),('solver','convergence_policy'),
    ('solver','initialization_policy'),('precision','identity'),('closure','identity'),('closure','dimension')])
def test_fresh_missing_metadata_rejected(path):
    payload=a.encode_fresh(replace(fresh(2,precision='float32'),solver=SOLVER))
    parent=payload
    for key in path[:-1]:parent=parent[key]
    del parent[path[-1]]
    with pytest.raises(ValueError,match='identity|schema fields'):a.decode_fresh(payload)


@pytest.mark.parametrize('path',[('config','solver'),('config','precision'),
    ('config','solver','convergence_policy'),('config','solver','initialization_policy'),
    ('config','closure','identity'),('config','spatial','active_axes'),('selection',),
    ('selection','intensity_volume'),('selection','material_volumes')])
def test_prepared_missing_metadata_rejected(path):
    from tests.test_pr_unified_codec import rewrite
    payload=codec.encode_request(scalable(request(1,dimension=2)))
    def remove(meta):
        parent=meta
        for key in path[:-1]:parent=parent[key]
        del parent[path[-1]]
    with pytest.raises(ValueError,match='schema|solver'):
        codec.decode_request(rewrite(payload,remove))


def test_malformed_load_never_mutates_gui(window,monkeypatch,tmp_path):
    import lcprop.pr.gui.main_window as gui
    original=replace(fresh(2),schema=a.LEGACY_FRESH_SCHEMA);apply(window,original)
    before=window._capture_experiment_gui_state();provenance=window.evolution_panel._loaded_unified_request
    malformed=a.encode_fresh(replace(fresh(2),solver=SOLVER));del malformed['solver']['initialization_policy']
    def decode(*args,**kwargs):return a.decode_fresh(malformed)
    with monkeypatch.context() as m:
        m.setattr(gui,'load_experiment',decode)
        with pytest.raises(ValueError,match='schema fields'):window.load_experiment_from(tmp_path/'malformed.json')
    assert window._capture_experiment_gui_state()==before
    assert window.evolution_panel._loaded_unified_request is provenance
    from lcprop.persistence import load_experiment
    saved=window.save_experiment_to(tmp_path/'untouched.json')
    assert a.encode_fresh(load_experiment(saved,expected_material_id='pr').request)==a.encode_fresh(original)


def complete_fresh_payload():
    from lcprop.optics.screens import RasterSource,ScreenPlacement,IntensityRasterScreen,ChannelLaunchElements
    screen=IntensityRasterScreen(RasterSource.from_array(np.ones((2,2))),ScreenPlacement())
    r=replace(fresh(2,precision='float32',scatter=True),solver=SOLVER,
        launch_elements=(ChannelLaunchElements(0,(screen,)),))
    return a.encode_fresh(r)


def mapping_paths(value,path=()):
    if isinstance(value,dict):
        for key,item in value.items():
            yield path+(key,)
            yield from mapping_paths(item,path+(key,))
    elif isinstance(value,list):
        for i,item in enumerate(value):
            yield from mapping_paths(item,path+(i,))


COMPLETE_TREE_PATHS=tuple(mapping_paths(complete_fresh_payload()))
NULLABLE_TREE_PATHS={('scattering',),('material','characteristic_wavenumber_per_um_override'),
    ('closure','reservoir_field'),('closure','background_intensity'),
    ('launch_elements',0,'elements',0,'source','asset_id'),
    ('launch_elements',0,'elements',0,'source','encoded_bytes_base64')}


@pytest.mark.parametrize('path',COMPLETE_TREE_PATHS)
@pytest.mark.parametrize('mode',['missing','null'])
def test_complete_fresh_tree_rejection(path,mode,monkeypatch):
    if mode=='null' and path in NULLABLE_TREE_PATHS:
        payload=complete_fresh_payload();parent=payload
        for key in path[:-1]:parent=parent[key]
        parent[path[-1]]=None
        assert a.encode_fresh(a.decode_fresh(payload))==payload
        return
    payload=complete_fresh_payload();parent=payload
    for key in path[:-1]:parent=parent[key]
    if mode=='missing':del parent[path[-1]]
    else:parent[path[-1]]=None
    import lcprop.persistence.experiments as experiments
    monkeypatch.setattr(experiments,'decode_beam_stack',lambda *args:pytest.fail('constructed beam before complete validation'))
    with pytest.raises(ValueError) as error:a.decode_fresh(payload)
    assert str(path[-1]) in str(error.value),str(error.value)


def test_explicit_defaults_and_legacy_material_defaults():
    payload=complete_fresh_payload()
    payload['material']['gain_length_product']=0.0;payload['material']['dark_intensity']=0.01
    assert a.encode_fresh(a.decode_fresh(payload))==payload
    legacy=a.encode_fresh(replace(fresh(2),schema=a.LEGACY_FRESH_SCHEMA))
    expected=a.decode_fresh(legacy)
    for key in ('gain_length_product','dark_intensity'):legacy['material'].pop(key)
    restored=a.decode_fresh(legacy)
    assert restored.schema==a.LEGACY_FRESH_SCHEMA
    assert restored.material.gain_length_product==0.0 and restored.material.dark_intensity==0.01
    assert restored.solver.identity==DIRECT and restored.solver==expected.solver
    assert a.decode_fresh(a.encode_fresh(restored))==restored


@pytest.mark.parametrize('path',COMPLETE_TREE_PATHS)
def test_complete_tree_malformed_gui_load(window,tmp_path,path):
    from lcprop.persistence import load_experiment
    original=replace(fresh(2),schema=a.LEGACY_FRESH_SCHEMA);apply(window,original)
    before=window._capture_experiment_gui_state();owned=window.evolution_panel._loaded_unified_request
    valid=window.save_experiment_to(tmp_path/'valid.json')
    document=json.loads(valid.read_text());payload=complete_fresh_payload();parent=payload
    for key in path[:-1]:parent=parent[key]
    del parent[path[-1]]
    document['request_payload']=payload
    malformed=tmp_path/'malformed.json';malformed.write_text(json.dumps(document))
    with pytest.raises(ValueError):window.load_experiment_from(malformed)
    assert window._capture_experiment_gui_state()==before
    assert window.evolution_panel._loaded_unified_request is owned
    saved=window.save_experiment_to(tmp_path/'saved.json')
    assert a.encode_fresh(load_experiment(saved,expected_material_id='pr').request)==a.encode_fresh(original)

"""M6 adapter-only certification, tiny CPU fixtures; no new scientific oracles."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('MPLCONFIGDIR','/tmp/lcprop-m6-mpl')
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from lcprop.core.backend import BackendSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.unified import integration as a, codec, products, workflow
from lcprop.pr.unified.specs import UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT,A7_CURRENT,OPEN_TRANSVERSE,PRElectricalClosureSpec
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.request_adapter import apply_pr_request
from tests.test_pr_local_plane_workflow import request as old_request
from lcprop.transport.defaults import default_transport_registry,default_transport_operations
from lcprop.transport.io import write_request_package,read_request_package,write_result_package,read_result_package
from lcprop.persistence import save_experiment,load_experiment


def fresh(d=1,kind=UNBIASED,precision='float64',scatter=False):
    r=old_request(precision=precision,scattering=scatter,supplied=False)
    r=replace(r,beams=replace(r.beams,channels=tuple(replace(c,coherence_group=g) for c,g in zip(r.beams.channels,r.beams.coherence_groups))))
    c=PRElectricalClosureSpec.a7(.02,r.material.background_intensity) if kind==A7_CURRENT else PRElectricalClosureSpec(
        kind,d,(0.,)*d if kind==UNBIASED else (.02,)+(0.,)*(d-1))
    return a.UnifiedFreshRequest(r.grid,r.beams,replace(r.material,applied_field=c.reservoir_field or 0.),c,r.backend,
        scattering=r.scattering)

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])

@pytest.fixture
def window(app,monkeypatch):
    import lcprop.pr.gui.main_window as m
    monkeypatch.setattr(m,'report_failure',lambda *args,**kwargs:pytest.fail(str(args)))
    w=PRMainWindow();yield w;w.close()


def apply(w,r):
    apply_pr_request(r,material_panel=w.material_panel,beam_panel=w.beam_panel,grid_panel=w.grid_panel,evolution_panel=w.evolution_panel)
    w._update_product_controls()


def exact(left,right):
    assert left.dtype==right.dtype and left.shape==right.shape
    assert left.tobytes()==right.tobytes()


def science_equal(left,right):
    assert left.status==right.status
    assert left.completed_cells==right.completed_cells and left.reached_z_um==right.reached_z_um
    assert left.ledger==right.ledger and left.identities==right.identities
    assert left.boundary_z_um==right.boundary_z_um and left.material_z_um==right.material_z_um
    exact(left.boundary_field,right.boundary_field)
    if left.material_state is not None:
        for key in ('q','psi','b'):exact(getattr(left.material_state,key),getattr(right.material_state,key))

CASES=[(1,UNBIASED),(1,A7_CURRENT),(1,PRESCRIBED_CURRENT),(1,FIXED_FIELD),
       (2,UNBIASED),(2,FIXED_FIELD),(2,PRESCRIBED_CURRENT),(2,OPEN_TRANSVERSE)]

@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('dimension,kind',CASES)
def test_gui_headless_request_and_results(window,precision,dimension,kind,tmp_path,record_property):
    direct=fresh(dimension,kind,precision,scatter=True);apply(window,direct)
    gui=window.build_request();assert a.encode_fresh(gui)==a.encode_fresh(direct)
    # M5 schema, independent serialization, comparing metadata + exact numerical launch.
    x=codec.decode_request(codec.encode_request(independent_prepared(direct)))
    y=codec.decode_request(codec.encode_request(a.prepare_request(gui)))
    assert x.config==y.config;exact(x.launch,y.launch)
    policy='analysis:far_field_intensity';window._set_product_policy(policy)
    headless=products.run_unified_products(independent_prepared(direct),selection=a.selection_for_policy(policy))
    dispatched=window._run_registered(gui);out=dispatched.result.run
    assert out.scientific.status=='completed',out.scientific.failure
    science_equal(headless.scientific,out.scientific)
    for key in headless.arrays:exact(headless.arrays[key],out.arrays[key])
    for key in headless.coordinates:exact(headless.coordinates[key],out.coordinates[key])
    field=dispatched.run_data.fields['far_field_intensity']
    assert field.data is out.arrays['far_field_intensity']
    for key in ('s_x','s_y'):assert field.coordinates[key] is out.coordinates[key]
    assert dispatched.run_data.fields['boundary_intensity'].data is out.arrays['boundary_intensity']
    window._on_finished(dispatched);assert window.run_status=='completed'
    selector=window.results_panel.workspace.image_pane.field_selector
    assert selector.findData('far_field_intensity')>=0
    selector.setCurrentIndex(selector.findData('far_field_intensity'))
    assert a.WORKFLOW_ID in window.describe_request(gui)
    assert not window.evolution_panel.max_coupled_passes.isVisible()
    assert window.evolution_panel.material_iterations.isHidden()
    hashes={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in out.arrays.items()}
    hashes.update({key:hashlib.sha256(getattr(out.scientific.material_state,key).tobytes()).hexdigest() for key in ('q','psi','b')})
    hashes['endpoint']=hashlib.sha256(out.scientific.boundary_field.tobytes()).hexdigest()
    hashes['request_metadata']=hashlib.sha256(json.dumps(x.config,sort_keys=True).encode()).hexdigest()
    for key,value in out.coordinates.items():hashes['coordinate_'+key]=hashlib.sha256(value.tobytes()).hexdigest()
    hashes['ledger']=hashlib.sha256(json.dumps(out.scientific.ledger,sort_keys=True).encode()).hexdigest()
    record_property('exact_gui_headless_identities',json.dumps(hashes,sort_keys=True))
    (tmp_path/'identity.json').write_text(json.dumps(hashes,indent=2))

@pytest.mark.parametrize('policy',['fast','full','analysis:far_field_intensity','analysis:complex_output,far_field_intensity'])
def test_normal_transport_and_persistence(window,tmp_path,policy):
    r=fresh();apply(window,r);window._set_product_policy(policy)
    p=window.save_experiment_to(tmp_path/'fresh.json');loaded=load_experiment(p,expected_material_id='pr')
    assert loaded.request==r;window.load_experiment_from(p);assert window.build_request()==r
    registry=default_transport_registry();assert registry.codec('pr',a.WORKFLOW_ID) is a.UNIFIED_TRANSPORT_CODEC
    write_request_package(tmp_path/'transport',registry=registry,material_id='pr',workflow_id=a.WORKFLOW_ID,
        request=r,run_id='m6-test',execution_target='slurm',result_policy=policy)
    decoded=read_request_package(tmp_path/'transport',registry=registry);assert decoded.request==r
    out=window._run_registered(r)
    write_result_package(tmp_path/'transport',codec=decoded.codec,result=out.result,request_envelope=decoded.envelope)
    restored=read_result_package(tmp_path/'transport',registry=registry)
    science_equal(out.result.run.scientific,restored.result.run.scientific)
    for key in out.result.run.arrays:exact(out.result.run.arrays[key],restored.result.run.arrays[key])

@pytest.mark.parametrize('precision',['float32','float64'])
def test_farfield_selection_observational_and_no_display_recalculation(precision,monkeypatch):
    r=fresh(precision=precision,scatter=True)
    no=a.execute_unified(r);yes=a.execute_unified(r,result_policy='analysis:far_field_intensity')
    science_equal(no.run.scientific,yes.run.scientific)
    def forbidden(*a,**k):raise AssertionError('display must use retained products')
    monkeypatch.setattr(workflow,'direction_cosine_spectrum',forbidden)
    monkeypatch.setattr(workflow,'solve_static_material',forbidden)
    monkeypatch.setattr(workflow,'hop_linear_inplace',forbidden)
    data=a.unified_to_run_data(yes);assert data.fields['far_field_intensity'].data is yes.run.arrays['far_field_intensity']

@pytest.mark.parametrize('mode',['cancel','fail'])
def test_partial_farfield_is_accepted_endpoint(window,monkeypatch,mode):
    r=fresh();apply(window,r);window._set_product_policy('analysis:far_field_intensity')
    original=workflow.solve_static_material;count=0;token=CancellationToken()
    def solve(*args,**kwargs):
        nonlocal count
        count+=1
        if count==2:
            if mode=='fail':raise RuntimeError('injected material failure')
            token.cancel()
        return original(*args,**kwargs)
    monkeypatch.setattr(workflow,'solve_static_material',solve)
    out=window._run_registered(r,cancellation_token=token)
    s=out.result.run.scientific
    assert s.status==('cancelled' if mode=='cancel' else 'failed')
    assert s.completed_cells==1 and s.reached_z_um==2.
    from lcprop.optics.farfield import direction_cosine_spectrum
    from lcprop.core.grid import make_grid
    grid=make_grid(r.grid)
    ref=direction_cosine_spectrum(s.boundary_field,dx_um=grid.dx_um,dy_um=grid.dy_um,wavelength_um=.633,refractive_index=r.material.refractive_index,coherence_groups=r.beams.coherence_groups,xp=np)
    exact(ref.intensity,out.result.run.arrays['far_field_intensity'])
    window._on_finished(out);assert window.run_status in ('stopped','failed')
    assert 'accepted z=2' in out.run_data.fields['far_field_intensity'].display_name


def test_controls_guards_estimator_and_identity(window):
    from lcprop.pr.runtime_estimator import estimate_pr_resources
    for d,kind in CASES:
        r=fresh(d,kind);apply(window,r);panel=window.evolution_panel.unified_closure
        assert panel.closure(r.material.background_intensity)==r.closure
        assert panel.condition.findData(A7_CURRENT)>=0 if d==1 else panel.condition.findData(A7_CURRENT)<0
        assert window.material_panel.applied_field.isEnabled() is False
        assert a.PROJECTION_ID in window.describe_request(r)
        plan=a.resource_plan(r);estimate=estimate_pr_resources(r)
        assert estimate.calibration_id==plan['schema'];assert plan['longitudinal_full_volume_bytes']==0
    r=fresh(2);large=replace(r,grid=replace(r.grid,Nx=4096,Ny=64))
    with pytest.raises(ValueError,match='12,288'):a.resource_plan(large)
    reduced=replace(large,closure=fresh().closure,solver=fresh().solver)
    assert a.resource_plan(reduced)['independent_columns']==64
    for key in ('workflow_identity','arithmetic_identity','projection_identity'):
        with pytest.raises(ValueError):a.decode_fresh(dict(a.encode_fresh(r),**{key:'legacy'}))


def test_local_i_legacy_and_td_isolation(window):
    from lcprop.pr.published_static import PublishedStaticRequest,PR_PUBLISHED_STATIC_WORKFLOW
    from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec,PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,PRTransverseRunRequest
    from lcprop.pr.specs import PRRunRequest,PR_TIMEDEPENDENT_WORKFLOW
    old=old_request(supplied=False)
    r=PublishedStaticRequest(**vars(old));apply(window,r)
    assert type(window.build_request()) is PublishedStaticRequest
    assert window._workflow_id_for_request(r)==PR_PUBLISHED_STATIC_WORKFLOW
    nonlinear=replace(r,material_response=PRTransverseMaterialResponseSpec(model='nonlinear'));apply(window,nonlinear)
    assert type(window.build_request()) is PublishedStaticRequest
    panel=window.evolution_panel;panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
    assert type(window.build_request()) is PRRunRequest
    panel.set_workflow_id(PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW)
    assert type(window.build_request()) is PRTransverseRunRequest


def test_fresh_gui_nonlinear_defaults_unified(window):
    window.evolution_panel.evolution.setCurrentIndex(0)
    window.material_panel.dark_irradiance.setText('0')
    window.material_panel.uniform_irradiance.setText('0')
    assert isinstance(window.build_request(),a.UnifiedFreshRequest)
    window.evolution_panel.transport_model.setCurrentIndex(1)
    assert window.evolution_panel.workflow_id()==a.WORKFLOW_ID


def independent_prepared(r):
    from lcprop.core.backend import get_backend
    from lcprop.core.grid import make_grid
    from lcprop.optics.launch import build_launch,OpticalLaunchContext
    from lcprop.pr.unified.specs import PRUnifiedSpatialSpec,PRMaterialPrecisionSpec,MIXED_PRECISION
    b=get_backend(r.backend);g=make_grid(r.grid,xp=b.xp,real_dtype=b.real_dtype)
    launch=build_launch(r.beams,g,complex_dtype=b.complex_dtype,launch_elements=r.launch_elements,
        context=OpticalLaunchContext(g,r.material.refractive_index,r.grid.z_length_um))
    k=r.material.characteristic_wavenumber_per_um
    spatial=(PRUnifiedSpatialSpec((r.grid.Nx,),(k*r.grid.x_aperture_um,),batch_shape=(r.grid.Ny,),batch_axes=('y',))
        if r.closure.dimension==1 else PRUnifiedSpatialSpec((r.grid.Nx,r.grid.Ny),
            (k*r.grid.x_aperture_um,k*r.grid.y_aperture_um),active_axes=('x','y')))
    precision=PRMaterialPrecisionSpec() if r.backend.precision=='float64' else PRMaterialPrecisionSpec(
        MIXED_PRECISION,'float32',output_dtype='float32')
    return workflow.UnifiedStaticRequest(r.grid,spatial,r.closure,launch.A0,material=r.material,
        precision=precision,backend=r.backend.backend,wavelength_um=r.beams.channels[0].wavelength_um,
        coherence_groups=r.beams.coherence_groups,scattering=r.scattering)


def test_dispatch_calls_certified_owners_exactly_once_per_cell(window,monkeypatch):
    r=fresh(scatter=True);apply(window,r);calls={}
    for module,name in [(products,'run_unified_products'),(workflow,'solve_static_material'),
                        (workflow,'electric_field_optical_node'),(workflow,'canonical_scattering_phase_increment')]:
        old=getattr(module,name)
        def wrapped(*args,_name=name,_old=old,**kwargs):
            calls[_name]=calls.get(_name,0)+1
            return _old(*args,**kwargs)
        monkeypatch.setattr(module,name,wrapped)
    out=window._run_registered(window.build_request())
    assert out.result.status=='completed'
    assert calls==dict(run_unified_products=1,solve_static_material=2,
        electric_field_optical_node=2,canonical_scattering_phase_increment=2)


def test_prelaunch_failure_has_no_fabricated_image(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('bad launch')
    monkeypatch.setattr(workflow,'resolve_material_illumination',fail)
    out=a.execute_unified(fresh(),result_policy='analysis:far_field_intensity')
    assert out.status=='failed' and out.run.scientific.boundary_field is None
    assert 'far_field_intensity' not in a.unified_to_run_data(out).fields
    restored=codec.decode_result(codec.encode_result(out.run))
    assert restored.scientific.boundary_field is None


def test_explicit_new_request_from_loaded_legacy(window):
    from lcprop.pr.published_static import PublishedStaticRequest
    from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
    old=PublishedStaticRequest(**vars(old_request(supplied=False)))
    apply(window,replace(old,material_response=PRTransverseMaterialResponseSpec(model='nonlinear')))
    assert isinstance(window.build_request(),PublishedStaticRequest)
    window.evolution_panel.fresh_unified_button.click()
    assert isinstance(window.build_request(),a.UnifiedFreshRequest)
    assert window.build_request().closure.identity==UNBIASED


def test_execution_target_does_not_change_scientific_metadata():
    r=fresh();cpu=a.encode_fresh(r);gpu=a.encode_fresh(replace(r,backend=replace(r.backend,backend='cupy')))
    assert cpu.pop('backend')['precision']==gpu.pop('backend')['precision']
    assert cpu==gpu


def test_normal_gui_run_button_preflight_and_planning(window,monkeypatch):
    r=fresh();apply(window,r)
    assert window._local_run_cost_guard(window.build_request())
    started=[]
    monkeypatch.setattr(window,'_validate_execution_request',lambda request: a.validate_fresh(request))
    monkeypatch.setattr(window,'_start_background',lambda request,**kwargs: started.append((request,kwargs)))
    window.run_clicked()
    assert len(started)==1 and a.encode_fresh(started[0][0])==a.encode_fresh(r)
    assert 'Workflow: '+a.WORKFLOW_ID in started[0][1]['summary']


def test_axis_transition_refreshes_controls_without_stale_coupling(window):
    panel=window.evolution_panel
    panel.transport_model.setCurrentIndex(1)
    panel.material_response.setCurrentIndex(panel.material_response.findData('linearized'))
    panel.evolution.setCurrentIndex(0)
    panel.transport_model.setCurrentIndex(0)
    assert panel.workflow_id()==a.WORKFLOW_ID
    assert panel.material_iterations.isHidden() and panel.max_coupled_passes.isHidden()
    panel.material_response.setCurrentIndex(panel.material_response.findData('field_linear_local_intensity'))
    panel.set_workflow_id(a.WORKFLOW_ID)
    assert panel.workflow_id()==a.WORKFLOW_ID


def test_selecting_farfield_requests_analysis_and_gui_size_guard(window):
    apply(window,fresh())
    window._set_product_policy('fast')
    window._analysis_actions['far_field_intensity'].setChecked(True)
    assert window.result_policy_selector.currentData()=='analysis:far_field_intensity'
    assert a.selection_for_policy(window.result_policy_selector.currentData()).far_field
    window.evolution_panel.transport_model.setCurrentIndex(1)
    window.grid_panel.Nx.setValue(4096);window.grid_panel.Ny.setValue(64)
    with pytest.raises(ValueError,match='envelope'):window.build_request()
    window.evolution_panel.transport_model.setCurrentIndex(0)
    assert window.build_request().closure.dimension==1


def test_unrepresentable_closure_target_is_rejected_without_rounding(window):
    from lcprop.pr.gui.request_adapter import validate_pr_gui_request_representable
    r=fresh(kind=FIXED_FIELD)
    r=replace(r,closure=replace(r.closure,target=(0.1234567890123456,)))
    with pytest.raises(ValueError,match='electrical target exactly'):
        validate_pr_gui_request_representable(r)


def test_loaded_td_does_not_pin_new_static_to_legacy(window):
    from lcprop.pr.specs import PR_TIMEDEPENDENT_WORKFLOW
    window.material_panel.dark_irradiance.setText('0')
    window.material_panel.uniform_irradiance.setText('0')
    panel=window.evolution_panel
    panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
    td=window.build_request();apply(window,td)
    panel.evolution.setCurrentIndex(panel.evolution.findData('static'))
    assert panel.workflow_id()==a.WORKFLOW_ID
    assert isinstance(window.build_request(),a.UnifiedFreshRequest)


@pytest.mark.parametrize('dimension,kind',CASES)
def test_electrical_summary_physical_meaning(window,dimension,kind):
    from lcprop.pr.gui.main_window import electrical_closure_summary
    r=fresh(dimension,kind)
    before=a.encode_fresh(r)
    text=electrical_closure_summary(r.closure)
    assert text in window.describe_request(r)
    assert a.encode_fresh(r)==before
    assert r.closure.identity in text
    if kind!=A7_CURRENT:
        for forbidden in ('A7','reservoir','E_app','I_b','J_ext'):
            assert forbidden not in text
    if kind==UNBIASED:
        assert 'Unbiased / zero-flux' in text
        assert 'target' not in text and 'prescribed' not in text.lower()
    elif kind==FIXED_FIELD:
        assert f'mean-field target b = {r.closure.target}' in text
        assert 'current' not in text
    elif kind==PRESCRIBED_CURRENT:
        assert f'mean-current target <J> = {r.closure.target}' in text
        assert 'mean-field target' not in text and 'fixed' not in text
        assert 'mean internal field b adjusts to carry this current' in text
    elif kind==A7_CURRENT:
        assert f'reservoir/applied parameter E_app = {r.closure.reservoir_field}' in text
        assert f'I_b = {r.closure.background_intensity}' in text
        assert f'J_ext = E_app * I_b = {r.closure.target[0]}' in text
        assert 'mean internal field b adjusts to carry the current' in text
        assert 'need not equal E_app' in text
        assert 'b = E_app' not in text
    else:
        assert f'x mean-field target b_x = {r.closure.target[0]}' in text
        assert 'open-circuit y' in text and '<J_y> = 0' in text
        assert 'mean y field b_y adjusts to enforce zero net y current' in text and 'b_y = 0' not in text
        assert 'mean-current target' not in text


def test_electrical_summary_rejects_unknown_identity():
    from lcprop.pr.gui.main_window import electrical_closure_summary
    # A corrupted persisted identity must not acquire generic current semantics.
    closure=fresh().closure
    object.__setattr__(closure,'identity','unknown_closure_v1')
    with pytest.raises(ValueError,match='unknown or unsupported electrical closure'):
        electrical_closure_summary(closure)


def test_installed_interactive_selectors_progress_and_reopen(window,record_property):
    r=fresh(scatter=True);apply(window,r);window._set_product_policy('fast')
    events=[]
    def callback(e):
        events.append(e);window._on_progress(e)
        assert f'Accepted cells {e.completed_units}/' in window.status_label.text()
        assert e.latest_field_state.preview.dtype==np.float32
        assert e.latest_field_state.preview.nbytes<=65536
    out=window._run_registered(window.build_request(),progress_callback=callback)
    assert [e.completed_units for e in events]==[1,2]
    assert events[-1].current_coordinate==out.result.run.scientific.reached_z_um
    window._on_finished(out)
    selector=window.results_panel.workspace.image_pane.field_selector
    names=[selector.itemText(i) for i in range(selector.count())]
    for name in ('Output Plane Intensity','Intensity xz','Intensity yz','Intensity Cut X','Intensity Cut Y','Output Far-Field Intensity'):
        assert any(name in s for s in names),names
    assert window.results_panel.workspace.longitudinal_pane.field_selector.count()>0
    restored=codec.decode_result(codec.encode_result(out.result.run))
    view=a.unified_to_run_data(a.UnifiedExecutionResult(restored,'fast','numpy'))
    for axis in ('x','y'):
        field=view.fields['intensity_'+axis+'z']
        exact(field.data,restored.arrays['intensity_cut_'+axis])
        exact(field.coordinates['z'],restored.coordinates['boundary_z_um'])
        assert field.coordinates['paired_cut_key']=='intensity_'+('y' if axis=='x' else 'x')+'z'
    assert 'Execution target:' in window.describe_request(r)
    assert window.describe_request(r).count('Execution target:')==1
    record_property('selector_entries',json.dumps(names))


def test_progress_gui_cancellation_consistent(window):
    token=CancellationToken();events=[]
    def callback(event):
        events.append(event);window._on_progress(event);token.cancel()
    out=window._run_registered(fresh(scatter=True),progress_callback=callback,cancellation_token=token)
    assert out.result.status=='cancelled'
    assert len(events)==out.result.run.scientific.completed_cells==1
    assert events[-1].current_coordinate==out.result.run.scientific.reached_z_um
    window._on_finished(out)
    assert window.run_status=='stopped'


def test_preexisting_minimal_archive_remains_readable():
    # Old M6 fast envelopes intentionally had no far-field selection.
    old=products.run_unified_products(independent_prepared(fresh()),
        selection=products.UnifiedSelection(boundary_intensity=True,intensity_cuts=True))
    from tests.test_pr_unified_codec import rewrite
    def legacy_metadata(m):
        m.update(schema=codec.LEGACY_RESULT_SCHEMA,products_schema='pr_unified_static_products_v1')
        m['scientific']['identities'].pop('solver')
    package=np.frombuffer(rewrite(codec.encode_result(old),legacy_metadata),dtype=np.uint8)
    decoded=a._decode_result({'backend':'numpy','result_policy':'fast'},{'unified_package':package})
    assert 'far_field_intensity' not in decoded.run.arrays


def test_progress_and_cut_resource_breakdown():
    r=fresh();plan=a.resource_plan(r)
    presentation=plan['presentation']
    assert presentation['longitudinal_intensity_cut_bytes']==3*(r.grid.Nx+r.grid.Ny)*8
    assert presentation['ephemeral_preview_max_bytes']==65536
    assert presentation['retained_progress_history_bytes']==0
    assert plan['longitudinal_full_volume_bytes']==0

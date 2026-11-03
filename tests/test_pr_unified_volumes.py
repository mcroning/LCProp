"""Selected result volumes: independent bounded references and real common viewer."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox
from lcprop.core.execution import CancellationToken
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.unified import products as p, workflow as w, codec, integration as a
from lcprop.gui.workspace import Workspace
from tests.test_pr_unified_workflow import request
from tests.test_pr_unified_integration import app, window, fresh, apply, exact, science_equal
from tests.test_pr_unified_codec import rewrite


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('mode',['complete','before','cancel','bookkeeping','observer'])
def test_selected_volumes_independent_stacks_and_accepted_only(monkeypatch,precision,mode):
    r=request(3,precision,scatter=True);token=CancellationToken();snapshots=[]
    prepare=p._Collector.prepare;book=w._prepare_acceptance
    def capture(self,previous,A,state,I,grid,record,ids):
        snap={'intensity_volume':total_intensity(A,coherence_groups=r.coherence_groups,xp=np).copy()}
        if state is not None:
            face=state.b[...,0]-(np.roll(state.psi,-1,axis=0)-state.psi)/(state.spatial.normalized_lengths[0]/state.spatial.active_shape[0])
            snap.update(potential_node_volume=state.psi.copy(),carrier_node_volume=np.exp(state.q),
                electric_field_x_optical_node_volume=(face+np.roll(face,1,axis=0))*face.dtype.type(.5))
        snapshots.append(snap)
        return prepare(self,previous,A,state,I,grid,record,ids)
    def bookkeeping(ledger,*args):
        if mode=='bookkeeping' and len(ledger)==1:raise RuntimeError('injected bookkeeping')
        return book(ledger,*args)
    def observer(event):
        assert not any('iteration_' in key for key in event.diagnostics)
        if mode=='cancel':token.cancel()
        if mode=='observer':raise RuntimeError('injected observer')
    monkeypatch.setattr(p._Collector,'prepare',capture);monkeypatch.setattr(w,'_prepare_acceptance',bookkeeping)
    if mode=='before':token.cancel()
    result=p.run_unified_products(r,selection=a.selection_for_policy('interactive'),cancellation_token=token,observer=observer)
    count=3 if mode=='complete' else 0 if mode=='before' else 1
    assert result.scientific.completed_cells==count,result.scientific.failure
    assert result.scientific.status==('completed' if mode=='complete' else 'failed' if mode in ('bookkeeping','observer') else 'cancelled')
    exact(result.coordinates['boundary_z_um'],np.arange(count+1,dtype=np.float64)*r.grid.dz_um)
    exact(result.coordinates['material_z_um'],np.arange(1,count+1,dtype=np.float64)*r.grid.dz_um)
    for name in ('intensity_volume','potential_node_volume','carrier_node_volume','electric_field_x_optical_node_volume'):
        start=0 if name=='intensity_volume' else 1
        expected=np.stack([s[name] for s in snapshots[start:count+1]]) if count or start==0 else np.empty((0,16,4),dtype=precision)
        exact(result.arrays[name],expected)
    restored=codec.decode_result(codec.encode_result(result))
    for name,value in result.arrays.items():exact(restored.arrays[name],value)
    assert restored.scientific.ledger==result.scientific.ledger
    assert result.scientific.collection is None


@pytest.mark.parametrize('precision',['float32','float64'])
def test_product_modes_observational(precision,record_property):
    r=fresh(precision=precision,scatter=True);results={}
    for mode in ('fast','interactive','full'):
        results[mode]=a.execute_unified(r,result_policy=mode)
    baseline=results['fast'].run
    for mode,result in results.items():
        science_equal(baseline.scientific,result.run.scientific)
        exact(baseline.arrays['far_field_intensity'],result.run.arrays['far_field_intensity'])
        assert any(k.endswith('_volume') for k in result.run.arrays)==(mode!='fast')
    s=baseline.scientific
    identities={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(endpoint=s.boundary_field,q=s.material_state.q,psi=s.material_state.psi,b=s.material_state.b,far_field=baseline.arrays['far_field_intensity']).items()}
    identities['ledger']=hashlib.sha256(json.dumps(s.ledger,sort_keys=True).encode()).hexdigest()
    record_property('mode_identity',json.dumps(identities,sort_keys=True))


@pytest.mark.parametrize('reopen',[False,True])
def test_common_linked_viewer_actual_slices_and_coordinates(app,reopen):
    result=a.execute_unified(fresh(),result_policy='interactive')
    if reopen:result=replace(result,run=codec.decode_result(codec.encode_result(result.run)))
    data=a.unified_to_run_data(result);workspace=Workspace();workspace.set_run_data(data)
    pane=workspace.longitudinal_pane;image=workspace.image_pane
    for name,zkey in [('intensity_volume','boundary_z_um'),('potential_node_volume','material_z_um')]:
        volume=result.run.arrays[name]
        image.field_selector.setCurrentIndex(image.field_selector.findData(name+'_xy'))
        pane.select_volume(name)
        assert pane.field_selector.currentData()==name
        for iz in range(volume.shape[0]):
            pane.z_plane_slider.setValue(iz)
            exact(image.image_view._field.data,volume[iz])
            assert image.field_selector.currentData()==name+'_xy'
            exact(data.fields[name].coordinates['z'],result.run.coordinates[zkey])
            assert f'z = {result.run.coordinates[zkey][iz]:g}' in pane.z_plane_label.text()
        for ix,iy in [(0,0),(3,2),(volume.shape[1]-1,volume.shape[2]-1)]:
            pane.x_cut_slider.setValue(ix);pane.y_cut_slider.setValue(iy)
            exact(pane.xz_view._field.data,volume[:,:,iy]);exact(pane.yz_view._field.data,volume[:,ix,:])
            exact(pane.xz_view._field.coordinates['z'],result.run.coordinates[zkey])
            exact(pane.yz_view._field.coordinates['y'],result.run.coordinates['y_um'])
            assert pane._ix==ix and pane._iy==iy
    assert 'Output Far-Field Intensity' in data.fields['far_field_intensity'].display_name
    assert data.fields['far_field_intensity'].data is result.run.arrays['far_field_intensity']
    for key in ('material_rms','material_max','carrier_min','carrier_max'):
        exact(data.curves[key].x,result.run.coordinates['material_z_um'])
        assert len(data.curves[key].y)==result.run.scientific.completed_cells
    for key,suffix,reducer in [('carrier_min','carrier_min',min),('carrier_max','carrier_max',max),('material_rms','_rms',max)]:
        expected=np.array([reducer(v for k,v in row['observations'].items() if 'iteration_' not in k and k.endswith(suffix)) for row in result.run.scientific.ledger])
        exact(data.curves[key].y,expected)
    assert data.fields['potential_node_volume'].coordinates['product_location']['gauge']==codec.GAUGE
    assert data.fields['electric_field_x_optical_node_volume'].coordinates['product_location']['projection']==a.PROJECTION_ID
    workspace.close()


def test_minimal_reopened_every_image_selection_and_no_volume(app):
    out=a.execute_unified(fresh(),result_policy='fast')
    out=replace(out,run=codec.decode_result(codec.encode_result(out.run)))
    data=a.unified_to_run_data(out);workspace=Workspace();workspace.set_run_data(data)
    selector=workspace.image_pane.field_selector
    assert not any(f.data.ndim==3 for f in data.fields.values())
    for i in range(selector.count()):
        selector.setCurrentIndex(i)
        assert workspace.image_pane.image_view.image is not None
    workspace.close()


def test_gui_defaults_modes_resources_and_explicit_budget(window,monkeypatch):
    apply(window,fresh());assert window.result_policy_selector.currentData()=='interactive'
    assert a.selection_for_policy('interactive').material_volumes==a.DEFAULT_MATERIAL_VOLUMES
    r=fresh();base=a.resource_plan(r,'fast');full=a.resource_plan(r,'full');n=r.grid.Nx*r.grid.Ny;cells=full['cells']
    assert full['longitudinal_product_bytes']==dict(intensity_volume=(cells+1)*n*8,**{k+'_volume':cells*n*8 for k in a.DEFAULT_MATERIAL_VOLUMES})
    assert base['longitudinal_full_volume_bytes']==0
    assert base['bytes']['scientific_state']==full['bytes']['scientific_state']
    for product in a.VOLUME_PRODUCTS:
        if product=='unified_face_y_volume':continue
        plan=a.resource_plan(r,'analysis:'+product)
        assert plan['longitudinal_full_volume_bytes']==cells*n*8
    bigger=replace(r,grid=replace(r.grid,Nx=4096,Ny=64,z_length_um=4000.,dz_um=50.))
    before=a.encode_fresh(bigger);calls=[]
    monkeypatch.setattr(QMessageBox,'question',lambda *args:calls.append(args) or QMessageBox.StandardButton.No)
    assert not window._unified_volume_guard(bigger);assert len(calls)==1
    assert a.encode_fresh(bigger)==before
    window._set_product_policy('fast');assert window._unified_volume_guard(bigger)
    assert len(calls)==1


def test_v1_cannot_claim_volume_selection_and_v2_policy_cannot_drop_it():
    out=a.execute_unified(fresh(),result_policy='interactive')
    payload=codec.encode_result(out.run)
    with pytest.raises(ValueError,match='schema fields'):
        codec.decode_result(rewrite(payload,lambda m:m['selection'].pop('intensity_volume')))
    with pytest.raises(ValueError,match='schema'):codec.decode_result(rewrite(payload,lambda m:m.update(products_schema='pr_unified_static_products_v1')))
    minimal=a.execute_unified(fresh(),result_policy='fast')
    with pytest.raises(ValueError,match='selection/policy'):
        a._decode_result(dict(backend='numpy',result_policy='full'),{'unified_package':np.frombuffer(codec.encode_result(minimal.run),dtype=np.uint8)})


def test_native_face_coordinates_are_not_node_resampled():
    out=a.execute_unified(fresh(d=2),result_policy='analysis:unified_face_x_volume,unified_face_y_volume')
    data=a.unified_to_run_data(out)
    for axis,size,aperture in [('x',fresh().grid.Nx,fresh().grid.x_aperture_um),('y',fresh().grid.Ny,fresh().grid.y_aperture_um)]:
        name='electric_field_'+axis+'_face_volume'
        exact(data.fields[name].coordinates[axis],out.run.coordinates[axis+'_um']+aperture/size/2)
        assert data.fields[name].coordinates['product_location']['transverse']=='oriented_face'


@pytest.mark.parametrize('precision',['float32','float64'])
def test_gpu_volumes_stay_backend_resident_until_explicit_export(precision,monkeypatch):
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()==0:pytest.skip('CuPy/CUDA unavailable')
    r=request(2,precision,scatter=True);r=replace(r,backend='cupy',initial_A=cp.asarray(r.initial_A))
    original=cp.asnumpy;transfers=[]
    def guard(value,*args,**kwargs):
        transfers.append((value.shape,value.nbytes))
        assert value.ndim<2,'scientific plane transfer during retention'
        return original(value,*args,**kwargs)
    monkeypatch.setattr(cp,'asnumpy',guard)
    baseline=p.run_unified_products(r)
    selected=p.run_unified_products(r,selection=a.selection_for_policy('interactive'))
    assert selected.scientific.status=='completed',selected.scientific.failure
    assert bool(cp.all(baseline.scientific.boundary_field==selected.scientific.boundary_field))
    for name in a.DEFAULT_MATERIAL_VOLUMES:
        assert isinstance(selected.arrays[name+'_volume'],cp.ndarray)
    assert isinstance(selected.arrays['intensity_volume'],cp.ndarray)
    monkeypatch.setattr(cp,'asnumpy',original)
    restored=codec.decode_result(codec.encode_result(selected))
    for name,value in selected.arrays.items():exact(restored.arrays[name],original(value))


def test_partial_reopened_view_status_and_material_ledger(app):
    token=CancellationToken()
    out=a.execute_unified(fresh(),result_policy='interactive',cancellation_token=token,
        progress_callback=lambda event:token.cancel())
    out=replace(out,run=codec.decode_result(codec.encode_result(out.run)))
    data=a.unified_to_run_data(out);workspace=Workspace();workspace.set_run_data(data)
    assert 'Cancelled' in workspace.result_ownership.text()
    assert out.run.scientific.completed_cells==1
    assert data.fields['intensity_volume'].data.shape[0]==2
    assert data.fields['potential_node_volume'].data.shape[0]==1
    assert 'cancelled' in data.fields['far_field_intensity'].display_name
    workspace.close()


def test_failure_before_launch_has_no_volume_or_curves(monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('injected before launch acceptance')
    monkeypatch.setattr(p._Collector,'prepare',fail)
    result=a.execute_unified(fresh(),result_policy='interactive')
    data=a.unified_to_run_data(result)
    assert result.run.scientific.boundary_field is None
    assert not list(data.fields.items()) and not list(data.curves.items())
    assert result.run.scientific.status=='failed'


@pytest.mark.parametrize('policy',['fast','interactive','full'])
def test_v1_provenance_repeated_save_reopen_and_viewer(app,policy,tmp_path,record_property):
    import io
    import zipfile
    legacy='pr_unified_static_products_v1'
    # Original committed V1 modes: Full has far field, Fast/Interactive do not;
    # none has longitudinal volumes. Build only these retained products.
    original=p.run_unified_products(a.prepare_request(fresh()),selection=p.UnifiedSelection(
        boundary_intensity=True,intensity_cuts=True,far_field=policy=='full'))
    def v1_header(m):
        m['products_schema']=legacy
        m['schema']=codec.LEGACY_RESULT_SCHEMA
        m['scientific']['identities'].pop('solver')
        m['selection'].pop('intensity_volume');m['selection'].pop('material_volumes')
    payload=rewrite(codec.encode_result(original),v1_header)
    metadata=dict(backend='numpy',result_policy=policy)
    def hashes(run):
        return {group+'/'+key:hashlib.sha256(value.tobytes()).hexdigest()
            for group in ('arrays','coordinates') for key,value in getattr(run,group).items()}
    before=hashes(original);cycles=[]
    for cycle in range(3):
        loaded=a._decode_result(metadata,{'unified_package':np.frombuffer(payload,dtype=np.uint8)})
        assert loaded.run.schema==legacy and loaded.result_policy==policy
        assert hashes(loaded.run)==before
        identity=dict(original.scientific.identities);identity.pop("solver")
        science_equal(replace(original.scientific,identities=identity),loaded.run.scientific)
        assert not loaded.run.selection.intensity_volume and not loaded.run.selection.material_volumes
        assert ('far_field_intensity' in loaded.run.arrays)==(policy=='full')
        # Both standalone save/load and the registered transport codec preserve identity.
        path=tmp_path/f'{policy}-{cycle}.pru';codec.save_result(path,loaded.run)
        assert codec.load_result(path).schema==legacy
        encoded=a._encode_result(loaded,policy)
        payload=encoded.payload.arrays['unified_package'].tobytes()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            stored=json.loads(archive.read('metadata.json'))
        assert stored['products_schema']==legacy
        assert 'intensity_volume' not in stored['selection']
        assert 'material_volumes' not in stored['selection']
        reopened=a._decode_result(encoded.payload.metadata,encoded.payload.arrays)
        assert reopened.run.schema==legacy and hashes(reopened.run)==before
        workspace=Workspace();data=a.unified_to_run_data(reopened);workspace.set_run_data(data)
        assert not any(f.data.ndim==3 for f in data.fields.values())
        assert ('far_field_intensity' in data.fields)==(policy=='full')
        selector=workspace.image_pane.field_selector
        for i in range(selector.count()):
            selector.setCurrentIndex(i)
            assert workspace.image_pane.image_view.image is not None
        workspace.close();cycles.append(dict(schema=reopened.run.schema,hashes=hashes(reopened.run)))
    record_property('v1_repeated_identity',json.dumps(dict(policy=policy,before=before,cycles=cycles),sort_keys=True))


def test_legacy_provenance_does_not_weaken_v2_or_allow_v1_volumes():
    legacy='pr_unified_static_products_v1'
    full=a.execute_unified(fresh(),result_policy='full')
    with pytest.raises(ValueError,match='legacy'):
        codec.encode_result(replace(full.run,schema=legacy))
    old=p.run_unified_products(a.prepare_request(fresh()),selection=p.UnifiedSelection(
        boundary_intensity=True,intensity_cuts=True,far_field=False))
    # Relabelling old products V2 does not give them the new policy contract.
    for policy in ('fast','interactive','full'):
        raw=np.frombuffer(codec.encode_result(old),dtype=np.uint8)
        with pytest.raises(ValueError,match='selection/policy'):
            a._decode_result(dict(backend='numpy',result_policy=policy),dict(unified_package=raw))
    # A genuine Full selection must have its selected arrays even if other fields match.
    arrays=dict(full.run.arrays);arrays.pop('potential_node_volume')
    with pytest.raises(ValueError,match='product selection'):
        codec.encode_result(replace(full.run,arrays=arrays))

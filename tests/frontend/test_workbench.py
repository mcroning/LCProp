import asyncio
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import monotonic,sleep
from types import SimpleNamespace
import zipfile
import pytest
from pr_colab.batch_requests import preset,validate,ROUTES
from pr_colab.batch_session import BatchSession
from pr_colab.runtime_policy import RuntimePolicy,CompatibleDevices,CompatibleDevice
from pr_colab.batch_files import archive_completed,unpack_completed
from pr_colab.pr_workbench import PRWorkbench,open_pr_workbench
from lcprop.notebook.batch_device import GPUDevice,GPUReadiness,DeviceUnavailable
from lcprop.notebook.device_policy import Hardware,CAPABILITY_PROFILE
from lcprop.core.execution import RunProgress

ROUTES_PR=['pr_static','pr','pr_xy_static','pr_xy_td','pr_unified','pr_published']
GIB=1024**3

def session(**kw):return BatchSession(fixed_route='pr',memory_available=lambda:64*GIB,**kw)

def app(tmp_path):return PRWorkbench(files_root=tmp_path,memory_available=lambda:64*GIB)

def hashes(path):
    with zipfile.ZipFile(path) as z:return {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}

@pytest.mark.parametrize('route',ROUTES_PR)
def test_native_mapping_and_roundtrip(route,tmp_path):
    r=preset(route); payload,_=validate(route,r)
    assert ROUTES[route][1].decode_request(payload)==r
    s=session();s.switch(route);s.draft=r
    path=tmp_path/'draft.json';s.save_draft(path);s.load_draft(path)
    assert s.draft==r
    s.preflight(route,r)

@pytest.mark.parametrize('route',ROUTES_PR)
def test_route_forms_and_edit_preservation(route,tmp_path):
    a=app(tmp_path);a.session.switch(route);a._sync()
    old=a.session.draft
    a.grid_inputs['Nx'].value=96
    assert a.session.draft.grid.Nx==96
    a.beam_plane.fields['x0_um'].value='5'
    assert a.session.draft.beams.channels[0].x0_um==5
    a.grid_inputs['Ny'].value=56
    assert a.session.draft.beams.channels[0].x0_um==5
    edited=deepcopy(a.session.draft)
    a.session.switch('pr' if route!='pr' else 'pr_static');a._sync()
    a.session.switch(route);a._sync()
    assert a.session.draft==edited


def test_memory_warning_override_and_invalid_remain_blocked(tmp_path):
    a=PRWorkbench(files_root=tmp_path,memory_available=lambda:2*GIB)
    assert a.run_button.disabled and 'estimated' in a.run_reason.value
    a.memory_risk.value=True
    assert not a.run_button.disabled
    a.grid_inputs['Nx'].value=-1
    assert a.run_button.disabled
    a.grid_inputs['Nx'].value=64
    assert not a.memory_risk.value and a.run_button.disabled
    with pytest.raises(ValueError):RuntimePolicy().check(100,None)
    RuntimePolicy(True).check(100,None)
    b=PRWorkbench(files_root=tmp_path,memory_available=lambda:1)
    b.memory_risk.value=True
    assert b.run_button.disabled and 'plane' in b.run_reason.value


def fake_gpu(free=32*GIB):
    hw=Hardware('Mock L4','mock-id',(8,9),13000,13000,'cupy-cuda13x','14.0.1')
    gpu=GPUDevice(0,'Mock L4',nullcontext,lambda _:GPUReadiness(CAPABILITY_PROFILE,0,free,32*GIB,monotonic()))
    return hw,gpu


def test_backend_not_fixture_gated_explicit_optin_and_memory():
    calls=[]
    p=CompatibleDevices(probe=lambda:(calls.append(1) or fake_gpu()))
    p.working=3*GIB
    assert p.resolve('pr_xy_static','arbitrary','numpy','float64',(1024,1024)).resolved=='numpy'
    assert not calls
    assert p.resolve('pr_xy_static','edited','auto','float64',(1024,1024)).resolved=='numpy'
    with pytest.raises(DeviceUnavailable):p.resolve('pr_xy_static','edited','cupy','float64',(1024,1024))
    p.allow_unqualified=True
    assert p.resolve('pr_xy_static','edited','cupy','float64',(1024,1024)).resolved=='cupy'
    p.probe=lambda:fake_gpu(2*GIB)
    with pytest.raises(DeviceUnavailable):p.resolve('pr','x','cupy','float64',(64,48))
    p.memory_override=True
    assert p.resolve('pr','x','cupy','float64',(64,48)).resolved=='cupy'
    assert p.resolve('pr','x','cupy','float32',(64,48)).precision=='float32'
    p.probe=lambda:(_ for _ in ()).throw(DeviceUnavailable('changed runtime'))
    assert p.resolve('pr','x','auto','float64',(64,48)).resolved=='numpy'
    with pytest.raises(DeviceUnavailable,match='changed runtime'):p.resolve('pr','x','cupy','float64',(64,48))


def test_invalid_typed_controls_no_silent_request_change(tmp_path):
    a=app(tmp_path);before=a.session.draft
    a.solver_inputs['dt_normalized'].value='nonsense'
    assert a.run_button.disabled and a.session.draft==before
    a.redisplay(hosted=False)
    assert a.solver_inputs['dt_normalized'].value=='nonsense'
    a.engine_workflow.value='pr_xy_td'
    assert a.session.route=='pr'


def test_more_than_two_beams_and_edit_approval_revocation(tmp_path):
    a=app(tmp_path)
    a.beam_plane._add(True);a.beam_plane._add(True)
    assert len(a.session.draft.beams.channels)==3
    a.unqualified_opt_in.value=True;a.memory_risk.value=True
    a.beam_plane.fields['x0_um'].value='5'
    assert not a.unqualified_opt_in.value and not a.memory_risk.value

@pytest.mark.parametrize('fixture',['synthetic','legacy'])
def test_saved_result_all_nine_and_views(fixture,tmp_path,saved_archive):
    a=app(tmp_path)
    path=Path(__file__).resolve().parents[2]/'frontend/pr_colab/synthetic_pr.zip'
    if fixture=='legacy':
        path=saved_archive('legacy')
    a.open_path(path);record=a.session.completed
    assert record is not None and record.package_files
    field=next(k for k,v in record.products.fields.items() if v.data.ndim==3)
    a.field.value=field
    for cut in ('x–y','x–z','y–z'):
        a.cut.value=cut
        for aspect in ('Physical','Square'):
            a.aspect.value=aspect;a._draw();assert not a.plot_message.value
    z=archive_completed(a.session.export_completed(tmp_path/'export'))
    assert len(hashes(z))==9 and hashes(z)==hashes(path)
    before=a.session.draft
    namespace={'lcprop_pr_workbench':a};open_pr_workbench(namespace,hosted=False)
    assert a.session.draft==before and a.session.completed is record


def test_fake_progress_stop_failure_prior_result_and_auto_review(tmp_path):
    a=app(tmp_path);a.open_example();prior=a.session.completed
    class Fake:
        def __init__(self):self.fail=False
        def run_operation(self,op,r,**kw):
            if self.fail:raise MemoryError('fake allocation failure')
            for i in range(50):
                kw['progress_callback'](RunProgress(op.workflow_id,'running',i,50,i,'z','um',i*.01))
                if kw['cancellation_token'].is_cancelled():break
                sleep(.015)
            result=replace(prior.result,status='cancelled' if kw['cancellation_token'].is_cancelled() else 'completed')
            return SimpleNamespace(result=result,run_data=prior.products,message='fake only')
    async def exercise():
        fake=Fake();a.session.runner=fake
        assert a.session.start(a.session.generation)
        await asyncio.sleep(.25)
        assert a.session.busy and a.session.progress is not None
        a.session.stop();await a.session.task
        assert a.session.status=='stopped' and a.session.completed is prior and a.session.partial is not None
        fake.fail=True
        assert a.session.start(a.session.generation);await a.session.task
        assert a.session.status=='failed' and 'MemoryError' in a.session.message and a.session.completed is prior
        a.device_control.value='auto'
        a.run_button.click()
        assert not a.session.busy and 'Auto proposes CPU' in a.notice.value
        a.run_button.click();await a.session.task
        assert a.session.status=='failed'
    asyncio.run(exercise())

@pytest.mark.parametrize('route',['pr','pr_xy_td'])
def test_continuation_fake_dispatch_preserves_run_draft(route,tmp_path,monkeypatch):
    from pr_colab import continuation
    from lcprop.runners.base import WorkflowOperation
    s=session();s.switch(route);original=s.draft
    cp=SimpleNamespace(request=original)
    continued=replace(original,solver=replace(original.solver,Nt=7))
    monkeypatch.setattr(continuation,'describe',lambda *args:{'source':'fake checkpoint'})
    marker=WorkflowOperation('pr',ROUTES[route][0].workflow_id,lambda r:None,lambda r:None)
    monkeypatch.setattr(continuation,'operation_for',lambda *args:marker)
    calls=[]
    class Fake:
        def run_operation(self,operation,request,**kw):
            calls.append((operation,request,kw))
            raise MemoryError('fake failure after checkpoint dispatch')
    async def exercise():
        s.runner=Fake()
        assert s.start_continuation(route,continued,cp)
        assert s.draft is original
        await s.task
        assert s.status=='failed' and s.draft is original
    asyncio.run(exercise())
    assert calls[0][0] is marker and calls[0][1].solver.Nt==7


def test_advanced_native_fields_and_engine_limits(tmp_path):
    a=app(tmp_path)
    a.request_controls[('optical_boundary','mode')][0].value='sponge'
    assert a.session.draft.optical_boundary.mode=='sponge'
    a.request_controls[('material','temperature_K')][0].value='300'
    assert a.session.draft.material.temperature_K==300
    a.engine_workflow.value='pr_xy_static'
    assert ('solver','material_solver','max_newton_iterations') in a.request_controls
    a.request_controls[('solver','material_solver','max_newton_iterations')][0].value=31
    assert a.session.draft.solver.material_solver.max_newton_iterations==31
    assert ('material_response','model') in a.request_controls
    assert ('boundary','profile_id') in a.request_controls
    from lcprop.pr.unified.integration import UnifiedFreshRequest
    from lcprop.pr.unified.specs import PRElectricalClosureSpec,UNBIASED
    from lcprop.pr.unified.solver_specs import PRUnifiedSolverSpec,DIRECT,DIRECT_POLICY
    r=preset('pr_unified')
    r=replace(r,grid=replace(r.grid,Nx=128,Ny=128),closure=PRElectricalClosureSpec(UNBIASED,2,(0.,0.)),
              solver=PRUnifiedSolverSpec(DIRECT,DIRECT_POLICY))
    with pytest.raises(ValueError,match='12,288|12288'):validate('pr_unified',r)


def test_auto_decision_changes_require_new_confirmation(tmp_path):
    a=app(tmp_path);p=a.session.device_policy
    p.probe=lambda:fake_gpu();a.device_control.value='auto';a.unqualified_opt_in.value=True
    class Fake:
        def run_operation(self,*args,**kw):raise MemoryError('fake dispatch')
    async def exercise():
        a.session.runner=Fake();a.run_button.click()
        assert 'Auto proposes GPU' in a.notice.value and not a.session.busy
        p.probe=lambda:(_ for _ in ()).throw(DeviceUnavailable('lost GPU'))
        a.run_button.click()
        assert 'Auto proposes CPU' in a.notice.value and not a.session.busy
        a.run_button.click();await a.session.task
        assert a.session.status=='failed'
    asyncio.run(exercise())


def test_runtime_timeout_fake_work(tmp_path):
    a=app(tmp_path);a.open_example();prior=a.session.completed
    class Fake:
        def run_operation(self,op,r,**kw):
            while not kw['cancellation_token'].is_cancelled():sleep(.01)
            return SimpleNamespace(result=replace(prior.result,status='cancelled'),run_data=prior.products,message='fake timeout')
    async def exercise():
        a.session.runner=Fake();a.session.time_limit=.1
        a.session.start(a.session.generation);await a.session.task
        assert a.session.status=='stopped' and a.session.completed is prior and a.session.partial is not None
    asyncio.run(exercise())


def test_rejected_zip_preserves_result_and_draft(tmp_path):
    a=app(tmp_path);a.open_example();prior=a.session.completed;draft=a.session.draft
    path=tmp_path/'bad.zip';path.write_bytes(b'not a zip')
    with pytest.raises(zipfile.BadZipFile):a.open_path(path)
    assert a.session.completed is prior and a.session.draft is draft


def test_raster_native_codec_and_preview(tmp_path):
    from PIL import Image
    import numpy as np
    from pr_colab.raster import decode
    from lcprop.optics.screens import ChannelLaunchElements,IntensityRasterScreen,ScreenPlacement
    pixels=np.array([[[255,0,0,0],[0,255,0,255]],[[0,0,255,12],[255,255,255,255]]],dtype=np.uint8)
    path=tmp_path/'input.png';Image.fromarray(pixels).save(path)
    source=decode(path)
    rgb=pixels[:,:,:3].astype(np.uint32)
    expected=((19595*rgb[:,:,0]+38470*rgb[:,:,1]+7471*rgb[:,:,2]+32768)>>16).astype(np.uint8)
    np.testing.assert_equal(source.grayscale,expected)
    screen=IntensityRasterScreen(source,ScreenPlacement(width_um=20.,height_um=20.))
    r=preset('pr');r=replace(r,launch_elements=(ChannelLaunchElements(0,(screen,)),))
    payload,_=validate('pr',r);decoded=ROUTES['pr'][1].decode_request(payload)
    assert decoded.launch_elements==r.launch_elements
    from pr_colab.pr_workbench_preview import editing_preview
    field=editing_preview('pr',r)
    assert np.isfinite(field.data).all()

@pytest.mark.parametrize('workflow',['pr_static','pr_transverse_static'])
def test_native_saved_static_nine_member_roundtrip(workflow,tmp_path,saved_archive):
    fixture=saved_archive(workflow)
    a=app(tmp_path);a.open_path(fixture)
    assert a.session.completed is not None
    archive=archive_completed(a.session.export_completed(tmp_path/'export'))
    assert len(hashes(archive))==9 and hashes(archive)==hashes(fixture)


def test_three_beam_incoherent_groups_are_distinct():
    from pr_colab.batch_requests import beam_stack
    r=preset('pr');c=r.beams.channels[0]
    channels=(c,replace(c,name='b'),replace(c,name='c'))
    stack=beam_stack(r.beams,channels,'incoherent')
    assert len(set(stack.coherence_groups))==3


def test_desktop_beam_metadata_roundtrip(tmp_path):
    from lcprop.adapters.launchplane import beam_stack_to_launchplane
    from launchplane.serialization import beam_stack_to_dict
    from lcprop.persistence.experiments import write_experiment_file
    r=preset('pr');s=session()
    presentation={'beam_editor':{'provider':'launchplane','schema_version':1,'beam_stack':beam_stack_to_dict(beam_stack_to_launchplane(r.beams))}}
    # Use the codec's own optional presentation format validation.
    path=tmp_path/'desktop.json'
    write_experiment_file(path,r,material_id='pr',workflow_id=ROUTES['pr'][0].workflow_id,
                          registry=s.experiments,presentation_payload=presentation)
    s.load_draft(path)
    assert s.draft==r and s.presentation==presentation

@pytest.mark.parametrize('route',ROUTES_PR)
def test_auto_review_for_every_workflow(route,tmp_path):
    a=app(tmp_path);a.engine_workflow.value=route
    a.device_control.value='auto'
    assert a.session.device_intent=='auto' and not a.run_button.disabled
    async def exercise():
        a.run_button.click()
        assert not a.session.busy and 'Auto proposes CPU' in a.notice.value
    asyncio.run(exercise())

@pytest.mark.parametrize('route',ROUTES_PR)
def test_dispatch_typed_request_for_every_workflow(route):
    s=session();s.switch(route);seen=[]
    class Fake:
        def run_operation(self,op,r,**kw):
            seen.append((op,type(r),r.backend.backend))
            kw['progress_callback'](RunProgress(op.workflow_id,'running',1,2,1,'z','um',.01))
            raise MemoryError('mock allocation failure')
    async def exercise():
        s.runner=Fake();s.start(s.generation);await s.task
        assert s.status=='failed' and 'mock allocation' in s.message
    asyncio.run(exercise())
    assert seen==[(ROUTES[route][0],type(preset(route)),'numpy')]


def test_float32_gpu_compatibility_check_is_dtype_specific_and_cached(monkeypatch):
    import sys
    from pr_colab.runtime_policy import precision32_gpu
    calls=[]
    cp=SimpleNamespace(complex64='complex64',zeros=lambda *a,**kw:None,
        fft=SimpleNamespace(fft2=lambda a:(calls.append(1) or SimpleNamespace(dtype='complex64'))),
        cuda=SimpleNamespace(get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
    monkeypatch.setitem(sys.modules,'cupy',cp)
    _,gpu=fake_gpu();wrapped=precision32_gpu(gpu)
    wrapped.readiness((64,48));wrapped.readiness((64,48))
    assert len(calls)==1
    cp.fft.fft2=lambda a:SimpleNamespace(dtype='complex128')
    with pytest.raises(DeviceUnavailable,match='complex64'):precision32_gpu(gpu).readiness((64,48))


def test_standard_catalog_availability_is_truthful(tmp_path):
    from importlib import resources
    from pr_colab.raster import decode
    from lcprop.pr.image_sources import standard_image_catalog
    from lcprop.pr.portable_launch import encode_raster_source,decode_raster_source
    catalog=standard_image_catalog()
    if not catalog:
        from pr_colab.raster import install
        panel=install(app(tmp_path))
        assert list(panel.children[1].options)==[("Uploaded file","")]
        return  # Exact engine wheel ships an empty catalog; do not invent assets.
    asset=catalog[0]
    with resources.as_file(resources.files('lcprop.pr.assets.images').joinpath(asset.resource_filename)) as path:
        source=decode(path)
    source=replace(source,source_kind='standard',asset_id=asset.asset_id,display_name=asset.display_name)
    assert decode_raster_source(encode_raster_source(source))==source


def test_saved_reopen_overrides_cached_draft_view_without_erasing_drafts(tmp_path):
    a=app(tmp_path)
    for route in ROUTES_PR:
        a.session.switch(route);a._sync()
        a.beam_plane.fields['x0_um'].value='5'
        a.source.value='Draft preview'
    drafts={}
    for route in ROUTES_PR:
        a.session.switch(route);a._sync();drafts[route]=deepcopy(a.session.draft)
    a.open_example()
    assert a.source.value=='Completed' and a.sections.selector.value==1
    assert a.field.value!='input_preview'
    prior=a.session.completed;source=a.source.value
    with pytest.raises(Exception):a._open_uploaded('broken.zip',b'not a ZIP')
    assert a.session.completed is prior and a.source.value==source
    for route in ROUTES_PR:
        a.session.switch(route);a._sync();assert a.session.draft==drafts[route]


def test_backend_readiness_is_truthful_and_does_not_probe(tmp_path):
    a=app(tmp_path)
    a.session.device_policy.probe=lambda:pytest.fail('Display must not probe CUDA')
    assert 'CPU inputs ready' in a.run_reason.value
    a.device_control.value='auto'
    assert 'review pending' in a.run_reason.value and 'Ready' not in a.run_reason.value
    a.device_control.value='cupy'
    assert a.run_button.disabled and 'opt-in' in a.run_reason.value
    a.unqualified_opt_in.value=True
    assert 'checks pending' in a.run_reason.value and not a.run_button.disabled


def test_widget_protocol_matches_previously_working_colab_manager(tmp_path):
    import ipywidgets as widgets
    if widgets.__version__!='7.7.1':pytest.skip('Pinned rendering environment test')
    a=app(tmp_path);a.open_example()
    states=widgets.Widget.get_manager_state()['state'].values()
    assert states
    assert not any(s['model_name']=='TextStyleModel' for s in states)
    assert all(s['model_module_version'].startswith('1.') for s in states if s['model_module']=='@jupyter-widgets/controls')

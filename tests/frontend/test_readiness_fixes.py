"""Regressions for PR readiness findings; no scientific runner or CUDA."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pytest
from pr_colab.batch_requests import preset, validate, ROUTES
from pr_colab.batch_session import BatchSession
from pr_colab.batch_beams import objects
from pr_colab.metadata import INACTIVE_LAUNCH
from pr_colab.pr_workbench import PRWorkbench

GIB=1024**3

def session(route='pr'):
    s=BatchSession(fixed_route='pr',memory_available=lambda:64*GIB)
    s.switch(route)
    return s

def walk(widget):
    yield widget
    for child in getattr(widget,'children',()):yield from walk(child)

def control(app,description):
    return next(w for w in walk(app.advanced) if getattr(w,'description','')==description)

@pytest.fixture
def screened(tmp_path):
    from PIL import Image
    from pr_colab.raster import decode
    from lcprop.optics.screens import ChannelLaunchElements, IntensityRasterScreen, ScreenPlacement
    file=tmp_path/'screen.png';Image.new('L',(2,2),128).save(file)
    screen=IntensityRasterScreen(decode(file),ScreenPlacement(width_um=20.,height_um=20.))
    s=session();beam=s.draft.beams.channels[0]
    s.draft=replace(s.draft,beams=replace(s.draft.beams,channels=tuple(replace(beam,name=n) for n in ('alpha','beta','gamma'))),
                    launch_elements=(ChannelLaunchElements(1,(screen,)),ChannelLaunchElements(2,(replace(screen,invert=True),))))
    return s

def owners(s):
    return {s.draft.beams.channels[a.channel_index].name:a.elements for a in s.draft.launch_elements}

@pytest.mark.parametrize('action',['delete_preceding','disable_preceding','reorder','delete_owner'])
def test_launch_screens_follow_beam_identity(screened,action):
    s=screened;before=owners(s);items=list(objects(s.draft.beams,s.presentation))
    if action=='delete_preceding':items=items[1:]
    if action=='disable_preceding':items[0]=(items[0][0],False)
    if action=='reorder':items=items[::-1]
    if action=='delete_owner':items=items[:1]+items[2:]
    s.edit_beams(items,s.generation)
    assert owners(s)==({k:v for k,v in before.items() if k!='beta'} if action=='delete_owner' else before)
    validate(s.route,s.draft)
    assert 'beta' not in s.presentation.get(INACTIVE_LAUNCH,{})


def test_disabled_owner_survives_experiment_roundtrip_and_reenable(screened,tmp_path):
    s=screened;expected=owners(s);items=list(objects(s.draft.beams,s.presentation));items[1]=(items[1][0],False)
    s.edit_beams(items,s.generation)
    assert set(owners(s))=={'gamma'} and set(s.presentation[INACTIVE_LAUNCH])=={'beta'}
    path=tmp_path/'disabled.json';s.save_draft(path)
    loaded=session();loaded.load_draft(path)
    assert loaded.presentation==s.presentation
    items=[(beam,True) for beam,_ in objects(loaded.draft.beams,loaded.presentation)]
    loaded.edit_beams(items,loaded.generation)
    assert owners(loaded)==expected and INACTIVE_LAUNCH not in loaded.presentation
    validate(loaded.route,loaded.draft)


def test_gui_delete_preceding_beam_does_not_retarget_mask(screened,tmp_path):
    app=PRWorkbench(files_root=tmp_path,memory_available=lambda:64*GIB)
    app.session.draft=screened.draft;app.session.generation+=1;app._sync()
    app.beam_plane.beam.value='alpha';app.beam_plane._delete();app.beam_plane.sync()
    assert set(owners(app.session))=={'beta','gamma'}
    assert owners(app.session)==owners(screened)
    assert not app.run_button.disabled


def test_rejected_edit_keeps_all_attachment_intent(screened):
    s=screened;draft=s.draft;presentation=deepcopy(s.presentation)
    items=list(objects(s.draft.beams,s.presentation));items[1]=(items[0][0],True)
    with pytest.raises(ValueError):s.edit_beams(items,s.generation)
    assert s.draft is draft and s.presentation==presentation

@pytest.mark.parametrize('route',['pr','pr_static','pr_xy_td','pr_xy_static','pr_unified','pr_published'])
def test_time_and_form_errors_are_independent(route,tmp_path):
    app=PRWorkbench(files_root=tmp_path,memory_available=lambda:64*GIB)
    app.engine_workflow.value=route
    timeout=control(app,'Cooperative time budget (s)')
    field=app.grid_inputs['x_aperture_um'];old=app.session.draft.grid.x_aperture_um
    field.value='nonsense';assert app.run_button.disabled
    timeout.value=601
    assert app.run_button.disabled and field.value=='nonsense' and app.session.draft.grid.x_aperture_um==old
    app.session.start=lambda *a,**kw:pytest.fail('Invalid visible controls dispatched')
    app.run_button.click()
    field.value=str(old);assert not app.run_button.disabled
    timeout.value=0;assert app.run_button.disabled
    app.grid_inputs['Nx'].value=80
    assert app.run_button.disabled and timeout.value==0 and app.session.time_limit==601
    app.run_button.click()
    field.value='also invalid';timeout.value=602
    assert app.run_button.disabled
    field.value=str(old);assert not app.run_button.disabled

@pytest.mark.parametrize('route',['pr_static','pr_unified','pr_published'])
@pytest.mark.parametrize('policy',['full','fast','analysis'])
def test_selectable_retention_restored_on_load(route,policy,tmp_path):
    s=session(route);s.result_policy=policy;path=tmp_path/'draft.json';s.save_draft(path)
    s.result_policy='fast' if policy=='full' else 'full';s.load_draft(path)
    assert s.result_policy==policy and s.execution_intent.retrieval_policy==policy
    app=PRWorkbench(files_root=tmp_path,memory_available=lambda:64*GIB)
    app.session=s;app._make_editor()
    assert control(app,'Result products').value==policy

@pytest.mark.parametrize('route',['pr','pr_static','pr_xy_td','pr_unified','pr_published'])
def test_native_valid_float32_draft_reopens(route,tmp_path):
    s=session(route);s.draft=replace(s.draft,backend=replace(s.draft.backend,precision='float32'))
    validate(route,s.draft);expected=s.draft;path=tmp_path/'float32.json';s.save_draft(path)
    s.load_draft(path)
    assert s.draft==expected and s.draft.backend.precision=='float32' and s.execution_intent.precision=='float32'


def test_native_invalid_float32_static_stays_rejected(tmp_path):
    s=session('pr_xy_static');s.draft=replace(s.draft,backend=replace(s.draft.backend,precision='float32'))
    with pytest.raises(ValueError,match='float64'):s.save_draft(tmp_path/'invalid.json')


def test_intent_rejection_is_atomic_and_explained(tmp_path):
    s=session('pr_static');path=tmp_path/'draft.json';s.save_draft(path)
    payload=json.loads(path.read_text());payload['execution_intent']['target']='slurm'
    payload['execution_intent']['cluster_profile']='example';payload['execution_intent']['resource_profile']='example'
    path.write_text(json.dumps(payload));before=s.draft;policy=s.result_policy
    with pytest.raises(ValueError,match='Slurm'):s.load_draft(path)
    assert s.draft is before and s.result_policy==policy


def test_disabled_screen_survives_desktop_metadata_and_other_edits(screened,tmp_path):
    from lcprop.adapters.launchplane import beam_stack_to_launchplane
    from launchplane.serialization import beam_stack_to_dict
    s=screened
    s.presentation={'beam_editor':{'provider':'launchplane','schema_version':1,
        'beam_stack':beam_stack_to_dict(beam_stack_to_launchplane(s.draft.beams))}}
    expected=owners(s)
    items=list(objects(s.draft.beams,s.presentation));items[1]=(items[1][0],False)
    s.edit_beams(items,s.generation)
    s.set_draft(s.route,replace(s.draft,grid=replace(s.draft.grid,Nx=80)),s.generation)
    assert set(s.presentation[INACTIVE_LAUNCH])=={'beta'}
    path=tmp_path/'desktop-disabled.json';s.save_draft(path)
    restored=session();restored.load_draft(path)
    restored.edit_beams([(beam,True) for beam,_ in objects(restored.draft.beams,restored.presentation)],restored.generation)
    assert owners(restored)==expected
    validate(restored.route,restored.draft)


def test_deleting_disabled_screen_owner_removes_only_its_metadata(screened):
    s=screened;items=list(objects(s.draft.beams,s.presentation));items[1]=(items[1][0],False)
    s.edit_beams(items,s.generation)
    remaining=[(beam,on) for beam,on in objects(s.draft.beams,s.presentation) if beam.name!='beta']
    gamma=owners(s)['gamma'];s.edit_beams(remaining,s.generation)
    assert owners(s)=={'gamma':gamma} and INACTIVE_LAUNCH not in s.presentation


def test_fixed_product_workflow_does_not_silently_accept_reduced_retention(tmp_path):
    s=session();path=tmp_path/'draft.json';s.save_draft(path)
    payload=json.loads(path.read_text());payload['execution_intent']['retrieval_policy']='analysis'
    path.write_text(json.dumps(payload));before=s.draft
    with pytest.raises(ValueError,match='fixed native products'):s.load_draft(path)
    assert s.draft is before and s.result_policy=='full'

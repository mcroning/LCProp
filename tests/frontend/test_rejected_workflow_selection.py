"""Rejected navigation must preserve drafts/text and accurately identify the model."""
import pytest
from pr_colab.pr_workbench import PRWorkbench

ROUTES = ['pr','pr_static','pr_xy_td','pr_xy_static','pr_unified','pr_published']

@pytest.mark.parametrize('route,selector', [(r,c) for r in ROUTES for c in ['transport','evolution','engine']
    if c=='engine' or r not in ('pr_unified','pr_published')])
def test_rejected_switch_restores_selectors_without_rebuilding(route,selector,tmp_path):
    a=PRWorkbench(files_root=tmp_path,memory_available=lambda:64*1024**3)
    a.open_example()
    a.session.switch(route);a._sync()
    a.beam_plane.fields['x0_um'].value='5'
    s=a.session
    expected_pair=('full' if route.startswith('pr_xy') or (route=='pr_unified' and s.draft.closure.dimension==2) else 'reduced',
                   'td' if route in ('pr','pr_xy_td') else 'static')
    before=s.draft; completed=s.completed; state=dict(s.route_state)
    text=a.grid_inputs['dz_um']; valid=text.value
    text.value='invalid dz'
    assert a.run_button.disabled
    nav={'transport':a.transport_control,'evolution':a.evolution_control,'engine':a.engine_workflow}[selector]
    target={'transport':'full' if expected_pair[0]=='reduced' else 'reduced',
            'evolution':'td' if expected_pair[1]=='static' else 'static',
            'engine':'pr_xy_td' if route!='pr_xy_td' else 'pr'}[selector]
    # Repeated rejection also exercises observer re-entry protection.
    for _ in range(2):
        nav.value=target
        assert (a.transport_control.value,a.evolution_control.value)==expected_pair
        assert a.engine_workflow.value==s.route==route
        assert a.grid_inputs['dz_um'] is text and text.value=='invalid dz'
        assert s.draft is before and s.completed is completed and s.route_state==state
        assert a.run_button.disabled and 'invalid' in a._form_invalid
    text.value=valid
    assert not a.run_button.disabled
    assert a.engine_workflow.value==s.route==route
    assert (a.transport_control.value,a.evolution_control.value)==expected_pair
    assert s.draft==before and s.completed is completed and s.route_state==state
    # The same navigation works once valid; it was rejected, not queued silently.
    nav.value=target
    assert s.route!=route


@pytest.mark.parametrize('error_kind',['solver','budget'])
@pytest.mark.parametrize('selector',['transport','evolution','engine'])
def test_original_td_solver_rejection_and_budget_error(error_kind,selector,tmp_path):
    a=PRWorkbench(files_root=tmp_path,memory_available=lambda:64*1024**3)
    a.open_example();before=a.session.draft;completed=a.session.completed
    def walk(w):
        yield w
        for child in getattr(w,'children',()):yield from walk(child)
    field=(a.solver_inputs['dt_normalized'] if error_kind=='solver' else
           next(w for w in walk(a.advanced) if getattr(w,'description','')=='Cooperative time budget (s)'))
    valid=field.value
    field.value='invalid solver' if error_kind=='solver' else 0
    controls={'transport':(a.transport_control,'full'),'evolution':(a.evolution_control,'static'),
              'engine':(a.engine_workflow,'pr_xy_td')}
    control,target=controls[selector];control.value=target
    assert field.value==('invalid solver' if error_kind=='solver' else 0)
    assert a.run_button.disabled and a.session.draft is before
    assert a.session.completed is completed
    field.value=valid
    assert not a.run_button.disabled
    assert (a.transport_control.value,a.evolution_control.value,a.engine_workflow.value)==('reduced','td','pr')
    assert a.session.route=='pr' and a.session.completed is completed

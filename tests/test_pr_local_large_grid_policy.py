"""Support/planning only: no large solver runs in the unit suite."""
from dataclasses import replace
import numpy as np
import pytest
from tests.test_pr_unified_integration import fresh,app,window,apply
from tests.test_pr_unified_scalable_workflow import SOLVER
from lcprop.pr.unified import integration as a,local_planning as lp
from lcprop.pr.unified.solver_specs import PRUnifiedSolverSpec,DIRECT,DIRECT_POLICY
from lcprop.core.backend import BackendSpec


def local(n,precision='float64'):
    r=fresh(2,precision=precision)
    return replace(r,solver=SOLVER,grid=replace(r.grid,Nx=n,Ny=n))


@pytest.mark.parametrize('n',[96,128,192,256,384,512])
def test_local_supported_and_memory_metadata(n,monkeypatch):
    r=local(n);before=a.encode_fresh(r)
    monkeypatch.setattr(a,'get_backend',lambda *args:pytest.fail('backend probe'))
    monkeypatch.setattr(lp,'physical_memory_bytes',lambda:16*1024**3)
    a.validate_fresh(r);plan=a.resource_plan(r,policy='minimal')
    assert plan['local_assessment']['classification']==('comfortable' if n<=256 else 'large/slow')
    assert a.encode_fresh(r)==before
    assert 'qualified <=512' in plan['support_envelope']
    direct=replace(r,solver=PRUnifiedSolverSpec(DIRECT,DIRECT_POLICY))
    if n>=128:
        with pytest.raises(ValueError,match='12,288'):a.validate_fresh(direct)
    else:a.validate_fresh(direct)


def test_precision_and_native_limits_unchanged():
    with pytest.raises(ValueError,match='envelope'):a.validate_fresh(local(128,'float32'))
    with pytest.raises(ValueError,match='envelope'):a.validate_fresh(local(513))
    for n,precision in [(256,'float32'),(512,'float64')]:
        r=replace(local(n,precision),backend=BackendSpec('cupy',precision,False))
        a.validate_fresh(r)
        with pytest.raises(ValueError,match='envelope'):a.validate_fresh(replace(r,grid=replace(r.grid,Nx=n+1)))


def test_ram_is_advice_not_identity_or_scientific_support(monkeypatch):
    r=local(256);before=a.encode_fresh(r)
    monkeypatch.setattr(lp,'physical_memory_bytes',lambda:1024)
    plan=a.resource_plan(r);assert plan['local_assessment']['classification']=='memory-risk'
    assert a.encode_fresh(r)==before
    a.validate_fresh(r)  # no dynamic RAM-based scientific rejection
    monkeypatch.setattr(lp,'physical_memory_bytes',lambda:None)
    plan=a.resource_plan(r);assert plan['local_assessment']['physical_ram_bytes'] is None
    assert 'incomplete' in plan['local_assessment']['warnings'][-1]
    fake={'bytes':{'a':600},'presentation':{}}
    assert lp.assess_local_resources(fake,(128,128),physical_ram_bytes=1000)['classification']=='memory-risk'
    assert lp.assess_local_resources(fake,(128,128),physical_ram_bytes=1001)['classification']=='comfortable'


@pytest.mark.parametrize('n',[128,256])
def test_actual_gui_run_preflight_dispatch(n,window,monkeypatch):
    r=local(n);apply(window,r)
    assert a.encode_fresh(window.build_request())==a.encode_fresh(r)
    monkeypatch.setattr(lp,'physical_memory_bytes',lambda:16*1024**3)
    calls=[]
    monkeypatch.setattr(window,'_validate_execution_request',lambda r:a.validate_fresh(r))
    monkeypatch.setattr(window,'_start_background',lambda request,**kw:calls.append(request))
    monkeypatch.setattr(window,'_confirm_very_expensive_local_run',lambda *args:'run_local')
    window._set_product_policy('minimal');window.run_clicked()
    assert len(calls)==1
    assert a.encode_fresh(calls[0])==a.encode_fresh(r)
    assert 'qualified <=512' in window.describe_request(r)


def test_gui_memory_warning_does_not_dispatch_on_no(window,monkeypatch):
    from lcprop.pr.gui import main_window as gui
    r=local(128);apply(window,r)
    monkeypatch.setattr(lp,'physical_memory_bytes',lambda:1024)
    seen=[]
    def answer(parent,title,message,*args):
        seen.append((title,message));return gui.QMessageBox.StandardButton.No
    monkeypatch.setattr(gui.QMessageBox,'question',answer)
    assert not window._unified_volume_guard(r)
    assert seen[0][0]=='Local memory risk'


def test_td_and_reduced_large_grid_metadata_unchanged():
    from tests.test_pr_transverse_timedependent_transport import _request
    r=_request();r=replace(r,grid=replace(r.grid,Nx=512,Ny=512))
    from lcprop.pr.gui.request_adapter import validate_pr_gui_workflow_request
    validate_pr_gui_workflow_request(r)
    from tests.test_pr_timedependent_transport import _request as reduced_td_request
    td=reduced_td_request();td=replace(td,grid=replace(td.grid,Nx=512,Ny=512))
    with pytest.raises(ValueError,match='conservative PR limit'):
        validate_pr_gui_workflow_request(td)  # Euler dt guard remains physical, not a grid ceiling.
    old=reduced_td_request().grid
    td=replace(td,grid=replace(td.grid,x_aperture_um=512*old.x_aperture_um/old.Nx,
        y_aperture_um=512*old.y_aperture_um/old.Ny))
    validate_pr_gui_workflow_request(td)
    reduced=fresh(1);reduced=replace(reduced,grid=replace(reduced.grid,Nx=1024,Ny=512))
    a.validate_fresh(reduced)

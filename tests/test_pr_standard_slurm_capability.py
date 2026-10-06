"""Metadata-only GUI/Slurm resolution of already qualified Static combinations."""
from dataclasses import replace
import pytest
from lcprop.core.backend import BackendSpec
from lcprop.pr.unified.integration import validate_fresh, resource_plan, WORKFLOW_ID
from lcprop.pr.unified.solver_specs import PRUnifiedSolverSpec, SCALABLE, ITERATIVE_POLICY, DIRECT, DIRECT_POLICY
from lcprop.pr.unified.specs import UNBIASED, PRESCRIBED_CURRENT
from lcprop.transport.defaults import default_transport_operations, default_transport_registry
from tests.test_pr_unified_integration import fresh, app, window, apply


@pytest.mark.parametrize('shape,precision', [((128,128),'float64'),((256,256),'float64'),
    ((512,512),'float64'),((256,256),'float32'),((384,32),'float64')])
def test_qualified_slurm_gui_request(window,monkeypatch,shape,precision):
    from lcprop.pr.gui import main_window as gui
    r=fresh(2,precision=precision)
    r=replace(r,grid=replace(r.grid,Nx=shape[0],Ny=shape[1]),
              backend=BackendSpec('cupy',precision,False),solver=PRUnifiedSolverSpec(SCALABLE,ITERATIVE_POLICY))
    apply(window,r);built=window.build_request()
    validate_fresh(built);plan=resource_plan(built)
    assert plan['scalable_solver'] and plan['bytes']['scientific_state']>0
    from types import SimpleNamespace
    window.slurm_runner=SimpleNamespace(registered_operations=default_transport_operations())
    window.runner=window.slurm_runner
    # External source availability only; scientific preflight and capability
    # resolution execute. No connection, deployment or device allocation.
    monkeypatch.setattr(window.remote_execution_controls,'validate_backend',lambda backend: None)
    monkeypatch.setattr(gui,'source_preflight',lambda *args: None)
    assert window._slurm_supports_workflow(WORKFLOW_ID)
    window._validate_execution_request(built)
    registry=default_transport_registry()
    assert any(op.workflow_id==WORKFLOW_ID for op in default_transport_operations())


@pytest.mark.parametrize('shape,precision', [((513,512),'float64'),((1024,1024),'float64'),((512,512),'float32')])
def test_outside_qualified_envelope_remains_rejected(shape,precision):
    r=fresh(2,precision=precision)
    r=replace(r,grid=replace(r.grid,Nx=shape[0],Ny=shape[1]),backend=BackendSpec('cupy',precision,False),
              solver=PRUnifiedSolverSpec(SCALABLE,ITERATIVE_POLICY))
    with pytest.raises(ValueError,match=f'outside scalable commissioned envelope: requested {shape[0]}x{shape[1]}, cupy, {precision}'):
        resource_plan(r)


def test_direct_bound_and_biased_bridge_rejected():
    r=fresh(2);r=replace(r,grid=replace(r.grid,Nx=128,Ny=128),backend=BackendSpec('cupy','float64',False),
                       solver=PRUnifiedSolverSpec(DIRECT,DIRECT_POLICY))
    with pytest.raises(ValueError,match='12,288'):validate_fresh(r)
    r=replace(r,grid=replace(r.grid,Nx=384,Ny=32),backend=BackendSpec('cupy','float32',False),
              solver=PRUnifiedSolverSpec(SCALABLE,ITERATIVE_POLICY),
              closure=replace(r.closure,identity=PRESCRIBED_CURRENT))
    with pytest.raises(ValueError,match='384x32 exception requires the unbiased'):validate_fresh(r)

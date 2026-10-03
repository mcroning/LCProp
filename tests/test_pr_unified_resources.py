from dataclasses import replace
from pathlib import Path
import subprocess
import ast
import numpy as np
import pytest
from tests.test_pr_unified_workflow import request
from lcprop.pr.unified.resources import estimate_resources
from lcprop.pr.unified.products import UnifiedSelection
from lcprop.pr.unified.specs import PRUnifiedSpatialSpec


def test_reduced_column_scaling_and_selection_cost():
    r=request(3);base=estimate_resources(r)
    r2=replace(r,grid=replace(r.grid,Ny=8),spatial=replace(r.spatial,batch_shape=(8,)),
               initial_A=np.repeat(r.initial_A,2,axis=2))
    doubled=estimate_resources(r2)
    assert doubled['bytes']['scientific_state']==2*base['bytes']['scientific_state']
    assert doubled['bytes']['float64_sparse_workspace']==base['bytes']['float64_sparse_workspace']
    assert doubled['direct_solver']==base['direct_solver']
    selected=estimate_resources(r,selection=UnifiedSelection(optical_cuts=True,material_cuts=('potential_node',)))
    assert selected['bytes']['selected_products']==4*(16+4)*16+3*(16+4)*8
    assert selected['longitudinal_full_volume_bytes']==0 and selected['measured_native_peak'] is None


def test_connected_guard_and_precision():
    r=request(1,dimension=2);plan=estimate_resources(r)
    assert plan['active_dimensions']==2 and plan['direct_solver']['domain_nodes']==64
    big=replace(r,grid=replace(r.grid,Nx=128,Ny=128),spatial=PRUnifiedSpatialSpec((128,128),(16.,8.),active_axes=('x','y')))
    with pytest.raises(ValueError,match='scope'):estimate_resources(big)
    single=estimate_resources(request(1,'float32'))
    double=estimate_resources(request(1,'float64'))
    assert single['precision_identity']=='state32_linear64_bernoulli64_v1'
    assert single['bytes']['scientific_state']*2==double['bytes']['scientific_state']
    assert single['bytes']['float64_sparse_workspace']==double['bytes']['float64_sparse_workspace']


def test_no_new_normal_dispatch_or_legacy_codec_edits():
    root=Path(__file__).resolve().parents[1]
    for path in (root/'src/lcprop').rglob('*.py'):
        if '/pr/unified/' not in str(path):
            assert 'pr_unified_static_request_v1' not in path.read_text()
            assert 'pr.unified.codec' not in path.read_text()
    for relative in ('src/lcprop/pr/published_static_codec.py','src/lcprop/pr/evolution.py','src/lcprop/pr/unified/projection.py'):
        assert (root/relative).read_bytes()==subprocess.check_output(['git','show','f662512041af05d752caf591f8e5e0a87c431f76:'+relative],cwd=root)


def test_m4_cell_scientific_ast_is_unchanged():
    root=Path(__file__).resolve().parents[1];path='src/lcprop/pr/unified/workflow.py'
    before=subprocess.check_output(['git','show','f662512041af05d752caf591f8e5e0a87c431f76:'+path],cwd=root).decode()
    after=(root/path).read_text()
    def scientific(text):
        tree=ast.parse(text)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run')
        loop=next(n for n in ast.walk(run) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='k')
        def stage(node,name):
            return isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and node.value.value==name
        start=next(i for i,n in enumerate(loop.body) if stage(n,'optical_hop'))
        stop=next(i for i,n in enumerate(loop.body) if stage(n,'bookkeeping'))
        return [ast.dump(n,include_attributes=False) for n in loop.body[start:stop]]
    assert scientific(before)==scientific(after)

"""Solver identity validation and isolation from existing Product dispatch."""
from dataclasses import replace
import ast
import inspect
from pathlib import Path
import numpy as np
import pytest
from lcprop.pr.unified.specs import (
    PRUnifiedSpatialSpec, PRMaterialPrecisionSpec, PRElectricalClosureSpec,
    POSITIVE_PRECISION, UNBIASED,
)
from lcprop.pr.unified.solver_specs import (
    PRUnifiedSolverSpec, REDUCED, DIRECT, SCALABLE, DIRECT_POLICY, ITERATIVE_POLICY,
)
from lcprop.pr.unified import _scalable, _krylov, _newton, operators
from lcprop.pr.unified.static import solve_static_material
from tests.test_pr_unified_scalable import request, closure, SOLVER


@pytest.mark.parametrize('identity,policy',[(REDUCED,DIRECT_POLICY),(DIRECT,DIRECT_POLICY),(SCALABLE,ITERATIVE_POLICY)])
def test_explicit_solver_dimension_and_precision(identity,policy):
    s=PRUnifiedSolverSpec(identity,policy)
    dim=1 if identity==REDUCED else 2
    spatial=PRUnifiedSpatialSpec((8,)*dim,(4.,)*dim,('x',) if dim==1 else ('x','y'))
    c=PRElectricalClosureSpec(UNBIASED,dim,(0.,)*dim)
    s.validate_material(spatial,c,PRMaterialPrecisionSpec())
    p=PRMaterialPrecisionSpec(POSITIVE_PRECISION,'float32',output_dtype='float32')
    if identity==SCALABLE:s.validate_material(spatial,c,p)
    else:
        with pytest.raises(ValueError):s.validate_material(spatial,c,p)
    with pytest.raises(ValueError):replace(s,linear_policy='auto')
    with pytest.raises(ValueError):replace(s,initialization_policy='warm')
    with pytest.raises(ValueError):replace(s,convergence_policy='relaxed')


@pytest.mark.parametrize('identity',['auto','unknown','pr_unified_connected_scalable_v2'])
def test_unknown_identity_rejected(identity):
    with pytest.raises(ValueError):PRUnifiedSolverSpec(identity,ITERATIVE_POLICY)


def test_reduced_batch_not_connected_and_direct_guard():
    spatial=PRUnifiedSpatialSpec((4096,),(6.,),batch_shape=(64,),batch_axes=('y',))
    c=PRElectricalClosureSpec(UNBIASED,1,(0.,))
    PRUnifiedSolverSpec(REDUCED,DIRECT_POLICY).validate_material(spatial,c,PRMaterialPrecisionSpec())
    with pytest.raises(ValueError):SOLVER.validate_material(spatial,c,PRMaterialPrecisionSpec())
    s=PRUnifiedSpatialSpec((128,128),(6.,6.),('x','y'))
    with pytest.raises(ValueError,match='12,288'):
        PRUnifiedSolverSpec(DIRECT,DIRECT_POLICY).validate_material(s,closure('zero'),PRMaterialPrecisionSpec())
    SOLVER.validate_material(s,closure('zero'),PRMaterialPrecisionSpec())


def test_v3_cannot_enter_legacy_material_solver(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('legacy solve reached')
    monkeypatch.setattr(_newton,'solve_domain',forbidden)
    r=request(np.ones((8,4),np.float32),(4.,2.))
    with pytest.raises(ValueError,match='explicit scalable'):solve_static_material(r,closure=closure('zero'))


def test_existing_dispatch_has_no_scalable_import():
    root=Path(__file__).parents[1]/'src/lcprop'
    for path in root.rglob('*.py'):
        if path.name in ('_scalable.py','_krylov.py','solver_specs.py','_positive.py'):continue
        tree=ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                text=ast.get_source_segment(path.read_text(),n)
                permitted = {
                    'pr/unified/scalable_workflow.py': {'from ._scalable import solve_material'},
                    'pr/unified/workflow.py': {'from .solver_specs import PRUnifiedSolverSpec, SCALABLE, legacy_solver, validate_execution'},
                    'pr/unified/codec.py': {'from .solver_specs import legacy_solver'},
                }
                if text in permitted.get(path.relative_to(root).as_posix(), set()):
                    continue
                assert '_scalable' not in text and '_krylov' not in text and 'solver_specs' not in text,path


def test_product_core_no_runtime_research_import_or_host_plane_conversion():
    from lcprop.pr.unified import _positive
    for module in (_scalable,_krylov,_positive):
        source=inspect.getsource(module)
        for forbidden in ('from paths','predecessor.','results/','asnumpy(','.get(', 'np.asarray(', 'np.array('):
            assert forbidden not in source
    # Existing independent-column routine remains its one-axis path.
    assert 'solve_domain' in inspect.getsource(_newton.solve_column)
    assert '_scalable' not in inspect.getsource(_newton.solve_column)


def test_krylov_true_residual_and_fail_closed():
    b=np.array([1.,-2.,3.]);A=lambda x:np.array([2.,3.,4.])*x;M=lambda x:x/3
    for solve in (_krylov.pcg,_krylov.gmres):
        record=[];x=solve(np,A,b,M,1e-10,record)
        assert np.linalg.norm(A(x)-b)<=1e-10*np.linalg.norm(b)
        assert record[-1]['true_relative']<=1e-10
    with pytest.raises(_krylov.Failure,match='curvature'):
        _krylov.pcg(np,lambda x:-x,b,M,.1,[])
    with pytest.raises(_krylov.Failure,match='annihilated'):
        _krylov.gmres(np,A,b,lambda x:0*x,.1,[])


def test_actual_matrix_free_allocations_are_transverse(monkeypatch):
    from lcprop.pr.unified._backend import MaterialBackend
    from lcprop.pr.unified._positive import Geometry
    from tests.test_pr_unified_scalable import electrical
    g=Geometry((8,4),(6.,3.),MaterialBackend('numpy','float32'))
    I=np.ones(g.shape,np.float32);q=np.zeros_like(I);p=q.copy();b=np.zeros(2,np.float32)
    J=_scalable.WideJacobian(g,I,q,p,b,electrical(g.backend,'current'))
    def arrays(value):
        if isinstance(value,np.ndarray):yield value
        elif isinstance(value,(tuple,list)):
            for v in value:yield from arrays(v)
    owned={id(a):a for value in vars(J).values() for a in arrays(value)}
    assert all(a.dtype==np.float64 for a in owned.values())
    assert sum(a.nbytes for a in owned.values())==136*g.size+32
    assert all(a.ndim<=2 for a in owned.values())
    seen=[];old=np.zeros
    def capture(shape,*args,**kwargs):
        result=old(shape,*args,**kwargs);seen.append((result.shape,result.nbytes));return result
    monkeypatch.setattr(np,'zeros',capture)
    # Uniform bordered operator exposes the genuine restarted-GMRES allocation.
    rhs=np.arange(2*g.size+2,dtype=float)/100
    M=_scalable.preconditioner(g,I,q,electrical(g.backend,'current'),False)
    _krylov.gmres(np,J,rhs,M,.1,[])
    assert ((2*g.size+2,61),8*(2*g.size+2)*61) in seen
    assert ((61,60),8*61*60) in seen
    assert all(len(shape)<=2 for shape,_ in seen)


def test_reduced_solver_never_allocates_scalable_work(monkeypatch):
    from tests.test_pr_unified_static import transport,zero
    def forbidden(*args,**kwargs):raise AssertionError('2D/scalable work in reduced solve')
    monkeypatch.setattr(_scalable,'solve_material',forbidden)
    monkeypatch.setattr(_scalable,'WideJacobian',forbidden)
    monkeypatch.setattr(np.fft,'fftn',forbidden)
    I=1.1+.1*np.cos(np.arange(16)[:,None])+np.zeros((16,3))
    state,d=solve_static_material(transport(I),closure=zero())
    assert state.q.shape==(16,3) and state.b.shape==(3,1)
    assert dict(d.observations)['columns']==3

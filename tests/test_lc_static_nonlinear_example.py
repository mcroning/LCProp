"""Bounded regression of the real example and unchanged historical fixture."""
import ast
import json
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import pytest
from lcprop.lc.workflows.static import run_static

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT/'examples/lc_static_nonlinear_cpu.py'))


def test_request_matches_existing_converged_regression():
    tree = ast.parse((ROOT/'tests/test_static_local_zmarch.py').read_text())
    namespace = dict(API)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('_workflow', '_request_parts')]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'original-fixture','exec'),namespace)
    expected = namespace['StaticRunRequest'](**namespace['_request_parts'](),
        solver=namespace['StaticSolverOptions'](workflow=namespace['_workflow'](),
                                               max_iterations=3,static_max_coupled_passes=3))
    assert API['build_request']() == expected


def test_fixed_baseline_changes_only_workflow():
    from dataclasses import replace
    request = API['build_request']()
    baseline = API['fixed_request'](request)
    assert baseline.initial_theta is None  # authoritative Product dark-bias initialization
    assert replace(baseline,solver=request.solver) == request
    assert replace(baseline.solver,workflow=request.solver.workflow) == request.solver
    assert request.solver.static_residual_rms_tol == .005
    assert request.solver.static_residual_max_tol == .02


def test_example_exact_canonical_parity_and_gates():
    completed, elapsed, warnings = API['execute'](API['build_request']())
    actual = completed.result
    start = perf_counter()
    def budget(_):
        if perf_counter()-start > 120:
            raise TimeoutError('Bounded canonical parity budget exceeded')
    result = run_static(API['build_request'](), progress_callback=budget)
    for field in ('A_initial', 'A_final', 'theta_final', 'theta_bias', 'intensity_stack'):
        np.testing.assert_array_equal(getattr(result,field),getattr(actual,field))
    expected = API['diagnostics'](completed,elapsed,warnings)
    from dataclasses import asdict
    assert [asdict(x) for x in result.iteration_records] == expected['iteration_history']
    assert [asdict(x) for x in result.slice_summaries] == expected['slice_summaries']
    assert result.status == 'completed' and result.all_slices_converged is True
    assert result.max_final_residual_rms <= .005
    assert result.max_final_residual_max <= .02
    assert np.max(np.abs(np.diff(result.theta_final,axis=0))) > 1e-4
    assert np.isfinite(result.A_final).all() and np.isfinite(result.theta_final).all()


def test_width_postprocessing_does_not_change_arrays():
    # Exercise width postprocessing with coordinate-coded data, no scientific solve.
    from types import SimpleNamespace
    from lcprop.core.grid import make_grid
    request = API['build_request']()
    grid = make_grid(request.grid, real_dtype=np.float64)
    plane = np.zeros((128,128));plane[30,40]=1
    run = SimpleNamespace(result=SimpleNamespace(request=request,intensity_stack=plane[None]))
    original=plane.copy()
    np.testing.assert_array_equal(API['widths'](run),np.zeros((1,2)))
    np.testing.assert_array_equal(plane,original)

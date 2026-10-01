"""Small-grid tests only; no native jobs or full A5 array allocations."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from lcprop.pr import static_workflow as workflow
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
from tests.test_pr_static_workflow import _static_request


def baseline_module():
    # Isolate every modified module in the committed numerical helper chain.
    def load(path, name, package):
        source = subprocess.check_output(['git', 'show',
            '3b1a267f28e7558742490b4bb84efb5784a83e78:src/lcprop/'+path], text=True)
        module = types.ModuleType(name)
        module.__package__ = package
        sys.modules[name] = module
        exec(compile(source, 'committed_'+path, 'exec'), module.__dict__)
        return module
    split = load('optics/splitstep.py', 'lcprop.optics._bridge_control', 'lcprop.optics')
    helper = load('pr/workflow.py', 'lcprop.pr._bridge_helper', 'lcprop.pr')
    for name in ('advance_prepared_response', 'scalar_angular_spectrum_kernel', 'total_intensity'):
        setattr(helper, name, getattr(split, name))
    module = load('pr/static_workflow.py', 'lcprop.pr._bridge_committed_control', 'lcprop.pr')
    for name in ('advance_pr_slice_with_midpoint_source', '_apply_canonical_scattering_after_slice',
                 '_canonical_scattering_phase_for_slice', '_validate_canonical_scattering_for_grid'):
        setattr(module, name, getattr(helper, name))
    module.scalar_angular_spectrum_kernel = split.scalar_angular_spectrum_kernel
    return module


@pytest.mark.parametrize('linearized', [False])
@pytest.mark.parametrize('boundary', ['periodic', 'tukey'])
@pytest.mark.parametrize('scattering', [False, True])
@pytest.mark.parametrize('policy', ['full', 'analysis:complex_output'])
def test_no_hooks_bitwise_committed_solver(linearized, policy, scattering, boundary):
    from lcprop.optics.boundaries import TransverseBoundarySpec
    request = replace(_static_request(Nz=3), optical_boundary=TransverseBoundarySpec(mode=boundary))
    if linearized:
        request = replace(request, material_response=PRTransverseMaterialResponseSpec(
            model='field_linear_local_intensity'))
    if scattering:
        from lcprop.pr.scattering import PRCanonicalScatteringSpec
        request = replace(request, scattering=PRCanonicalScatteringSpec(
            epsilon=.00002, transverse_correlation_um=.4, realization_seed=7,
            canonical_dz_um=request.grid.dz_um))
    old = baseline_module().run_pr_static(request, result_policy=policy)
    new = workflow.run_pr_static(request, result_policy=policy)
    for name in ('A_initial', 'A_final', 'E_initial', 'E_final',
                 'source_intensity_stack', 'residual_stack'):
        a, b = getattr(old, name), getattr(new, name)
        if a is None:
            assert b is None
        else:
            np.testing.assert_array_equal(a, b)
    assert old.replay_diagnostics == new.replay_diagnostics
    assert old.converged == new.converged
    assert old.slice_summaries == new.slice_summaries or [vars(x) for x in old.slice_summaries] == [vars(x) for x in new.slice_summaries]
    assert [vars(x) for x in old.iteration_records] == [vars(x) for x in new.iteration_records]


def test_post_phase_only_after_acceptance_and_during_replay():
    calls, hops = [], []
    from lcprop.optics.splitstep import hop_linear_inplace
    def hop(A, kernel, *, xp):
        hops.append(1)
        hop_linear_inplace(A, kernel, xp=xp)
    def phase(A, k, *, replay, grid, xp):
        calls.append((k, replay))
        A *= xp.exp(1j * .003 * (k + 1))
    request = replace(_static_request(Nz=3), material_response=PRTransverseMaterialResponseSpec(
        model='field_linear_local_intensity'))
    control = workflow.run_pr_static(request)
    result = workflow.run_pr_static(request, _research_hooks=workflow._PRStaticResearchHooks(
        linear_hop=hop, post_interval=phase))
    assert result.converged
    assert calls == [(k, False) for k in range(3)] + [(k, True) for k in range(3)]
    assert len(hops) > len(calls)  # Material trial advances do not consume RNG.
    np.testing.assert_allclose(control.source_intensity_stack, result.source_intensity_stack, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result.A_final, control.A_final * np.exp(.018j), rtol=2e-14, atol=2e-14)
    assert all(result.replay_diagnostics[k] for k in ('field_consistent', 'source_consistent', 'residual_consistent'))


def test_midpoint_source_includes_real_window_before_phase():
    from lcprop.pr.workflow import advance_pr_slice_with_midpoint_source
    from lcprop.pr.source import pr_driving_intensity
    A = np.ones((1, 8, 6), complex)
    window = np.linspace(.1, 1, 8)[:, None] * np.ones((1, 6))
    def apply(A, *, xp):
        A *= window
    out, source = advance_pr_slice_with_midpoint_source(A, np.zeros((8, 6)),
        kernel=np.ones((8, 6)), optical_substeps=1, dz_um=1., wavelength_um=.633,
        interaction_length_um=10., gain_length_product=1., peak_intensity_reference=1.,
        background_intensity=.01, coherence_groups=('a',), xp=np,
        _research_real_window=apply)
    np.testing.assert_array_equal(A, np.ones_like(A))
    np.testing.assert_allclose(out, window[None])
    np.testing.assert_allclose(source, (1 + window**2)/2 + .01)


def test_rejected_backtracks_do_not_consume_screen(monkeypatch):
    original = workflow.solve_pr_reduced_field_linear_intensity
    def oversized(intensity, **kwargs):
        answer = original(intensity, **kwargs)
        return replace(answer, E=answer.E * 8.)
    monkeypatch.setattr(workflow, 'solve_pr_reduced_field_linear_intensity', oversized)
    calls = []
    def phase(A, k, *, replay, grid, xp):
        calls.append((k, replay))
    request = replace(_static_request(Nz=2), material_response=PRTransverseMaterialResponseSpec(
        model='field_linear_local_intensity'))
    result = workflow.run_pr_static(request, _research_hooks=workflow._PRStaticResearchHooks(post_interval=phase))
    assert any(not r.accepted for r in result.iteration_records)
    assert calls == [(0, False), (1, False), (0, True), (1, True)]


def test_hooks_reject_canonical_scattering_before_backend(monkeypatch):
    from lcprop.pr.scattering import PRCanonicalScatteringSpec
    request = replace(_static_request(), scattering=PRCanonicalScatteringSpec(
        epsilon=.02, transverse_correlation_um=.4, realization_seed=0, canonical_dz_um=1.))
    monkeypatch.setattr(workflow, 'get_backend', lambda *a: pytest.fail('backend should not initialize'))
    with pytest.raises(ValueError, match='canonical scattering'):
        workflow.run_pr_static(request, _research_hooks=workflow._PRStaticResearchHooks())


@pytest.mark.parametrize('seam', ['workflow', 'splitstep'])
def test_committed_control_detects_no_hook_helper_regression(monkeypatch, seam):
    from lcprop.pr import workflow as helper
    from lcprop.optics import splitstep
    baseline = baseline_module()
    request = _static_request(Nz=2)
    before = baseline.run_pr_static(request)
    if seam == 'workflow':
        original = helper.advance_pr_slice_with_midpoint_source
        def changed(*args, **kwargs):
            field, source = original(*args, **kwargs)
            return field * .9, source
        monkeypatch.setattr(workflow, 'advance_pr_slice_with_midpoint_source', changed)
    else:
        original = splitstep.hop_linear_inplace
        def changed(A, *args, **kwargs):
            original(A, *args, **kwargs)
            A *= .9
        monkeypatch.setattr(splitstep, 'hop_linear_inplace', changed)
    actual = workflow.run_pr_static(request)
    control = baseline.run_pr_static(request)
    np.testing.assert_array_equal(before.A_final, control.A_final)
    assert not np.array_equal(actual.A_final, control.A_final)

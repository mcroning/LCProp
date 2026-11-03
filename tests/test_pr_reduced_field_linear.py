"""Local-intensity paper response, independent algebra and production integration."""
from dataclasses import replace
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
import pytest
import lcprop.persistence
from lcprop.core.backend import BackendSpec
from lcprop.pr.reduced_field_linear import (
    PRReducedFieldLinearSpec, solve_pr_reduced_field_linear_intensity,
    reduced_field_linear_residual,
)
from lcprop.pr.reduced_linearized import PRReducedLinearizedSpec, reduced_linearized_symbol
from lcprop.pr.source import pr_driving_intensity, channel_peak_intensity_reference
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
from lcprop.pr.static_workflow import run_pr_static, PRStaticWorkflowOptions
from lcprop.pr.experiment_codec import encode_pr_static_request, decode_pr_static_request
from lcprop.pr.static_transport_codec import encode_pr_static_transport_request, decode_pr_static_transport_request
from tests.test_pr_static_workflow import _static_request

MODEL = 'field_linear_local_intensity'

def request():
    r = _static_request(Nz=2)
    return replace(r, material_response=PRTransverseMaterialResponseSpec(model=MODEL),
                   solver=PRStaticWorkflowOptions(max_coupled_passes=20))


def matrices(n, dx):
    eye = np.eye(n)
    return ((np.roll(eye, -1, axis=0)-np.roll(eye, 1, axis=0))/(2*dx),
            (np.roll(eye, -1, axis=0)-2*eye+np.roll(eye, 1, axis=0))/dx**2)


@pytest.mark.parametrize('precision,tolerance', [('float64', 2e-13), ('float32', 3e-6)])
@pytest.mark.parametrize('bias', [0., .7, -.4])
@pytest.mark.parametrize('nx', [15, 16])
def test_dense_independent_paper_equation_and_residual(precision, tolerance, bias, nx):
    dtype = np.dtype(precision)
    x = np.arange(nx)[:, None]
    I = (1.3 + .65*np.cos(2*np.pi*x/nx) + np.arange(3)[None, :]*.1).astype(dtype)
    spec = PRReducedFieldLinearSpec(bias, .3, .52)
    d1, d2 = matrices(nx, spec.dx_normalized)
    q = (d1@I + bias*.3)/I
    expected = np.linalg.solve(np.eye(nx)+bias*d1-d2, q)
    result = solve_pr_reduced_field_linear_intensity(I, spec=spec, backend=BackendSpec('numpy', precision, False))
    np.testing.assert_allclose(result.E, expected, atol=tolerance, rtol=tolerance)
    assert result.E.dtype == dtype
    assert result.denominator.dtype == (np.complex64 if precision=='float32' else np.complex128)
    assert np.max(np.abs(result.residual)) < tolerance*10
    # A fixed reference drift E_app*I_b/I0 would not satisfy this biased oracle.
    np.testing.assert_allclose(result.residual,
        result.E + bias*(d1@result.E)-d2@result.E-q, atol=tolerance*10)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_physical_intensity_positivity_no_floor(value):
    I = np.ones((8, 3)); I[2, 1] = value
    with pytest.raises(ValueError, match='strictly positive'):
        solve_pr_reduced_field_linear_intensity(I, spec=PRReducedFieldLinearSpec(0, .01, .5))


def test_authoritative_optical_dark_uniform_source_and_exact_denominator():
    A = np.array([[[1, 0], [2, 0], [3, 0], [2, 0], [1, 0], [0, 0], [1, 0], [2, 0]]], dtype=complex)
    reference = channel_peak_intensity_reference(A, xp=np)
    I = pr_driving_intensity(A, peak_intensity_reference=reference, background_intensity=.01+.03, xp=np)
    np.testing.assert_array_equal(I, abs(A[0])**2/9+.04)
    assert I.min() == .04
    result = solve_pr_reduced_field_linear_intensity(I, spec=PRReducedFieldLinearSpec(0, .04, .5))
    d1, d2 = matrices(8, .5)
    np.testing.assert_allclose((np.eye(8)-d2)@result.E, (d1@I)/I, atol=3e-15)
    assert not np.allclose((np.eye(8)-d2)@result.E, d1@I)


@pytest.mark.parametrize('bias', [0., .8])
def test_constant_intensity(bias):
    I = np.full((16, 2), 2.)
    result = solve_pr_reduced_field_linear_intensity(I, spec=PRReducedFieldLinearSpec(bias, .3, .52))
    np.testing.assert_array_equal(result.E, np.full_like(I, bias*.3/2))


def test_fourier_modes_symbols_and_nyquist():
    n=32; dx=.5228141586169444
    spec=PRReducedFieldLinearSpec(0, .01, dx)
    old=reduced_linearized_symbol(n, spec=PRReducedLinearizedSpec(1, 0, .01, dx))
    for mode in (1, 5, 15, 16):
        x=np.arange(n)[:,None]; theta=2*np.pi*mode/n
        I=1+.4*np.cos(theta*x)
        result=solve_pr_reduced_field_linear_intensity(I, spec=spec)
        expected_q=-.4*np.sin(theta)/dx*np.sin(theta*x)/I
        np.testing.assert_allclose(np.fft.ifft(np.fft.fft(result.E,axis=0)*(1+old.k2_squared[:,None]),axis=0).real, expected_q, atol=3e-15)
        np.testing.assert_array_equal(result.denominator.real,1+old.k2_squared)
        if mode==16: np.testing.assert_allclose(result.E,0,atol=2e-16)


@pytest.mark.parametrize('precision', ['float64','float32'])
def test_midpoint_replay_no_newton_and_selected_analysis(monkeypatch, precision):
    import lcprop.pr.static_workflow as workflow
    from lcprop.pr.scattering import PRCanonicalScatteringSpec
    def forbidden(*args, **kwargs): raise AssertionError('nonlinear solver called')
    monkeypatch.setattr(workflow,'solve_pr_static_intensity_batched',forbidden)
    r=request(); r=replace(r,backend=replace(r.backend,precision=precision),scattering=PRCanonicalScatteringSpec(
        epsilon=.002,transverse_correlation_um=.4,realization_seed=19,canonical_dz_um=1.,
        algorithm_version='canonical_phase_slabs_v2_cross_backend'))
    result=run_pr_static(r)
    assert result.converged
    assert result.replay_diagnostics['field_consistent']
    assert result.replay_diagnostics['source_consistent']
    assert result.replay_diagnostics['residual_converged']
    assert result.replay_diagnostics['canonical_scattering']
    assert result.material_response_summary['model']==MODEL
    assert 'reference_intensity' not in result.material_response_summary
    fast=run_pr_static(r,result_policy='analysis')
    assert fast.replay_diagnostics==result.replay_diagnostics


def test_save_load_transport_and_reject_old_identity():
    r=replace(request(), initial_A=None)
    encoded=encode_pr_static_request(r)
    assert encoded['material_response']=={'model': MODEL}
    assert decode_pr_static_request(encoded)==r
    portable=encode_pr_static_transport_request(r)
    assert decode_pr_static_transport_request(portable.payload.metadata,portable.payload.arrays)==r
    for bad in ({'model':'linearized','reference_intensity':1.}, {'model':MODEL,'reference_intensity':1.}):
        encoded['material_response']=bad
        with pytest.raises(ValueError,match='retired|does not use'):
            decode_pr_static_request(encoded)
        metadata=dict(portable.payload.metadata, material_response=bad)
        with pytest.raises(ValueError,match='retired|does not use'):
            decode_pr_static_transport_request(metadata,{})


def test_gui_inspect_and_separate_transverse_reference():
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.main_window import PRMainWindow
    app=QApplication.instance() or QApplication([])
    window=PRMainWindow(); panel=window.evolution_panel
    panel.set_workflow_id('pr_static')
    assert panel.material_response.findData('linearized') == -1
    panel.material_response.setCurrentIndex(panel.material_response.findData(MODEL))
    assert panel.reference_intensity.isHidden()
    r=window.build_request(); summary=window.describe_request(r)
    assert r.material_response.to_payload()=={'model':MODEL}
    assert 'local intensity' in summary
    assert 'I₀' not in summary and 'equilibrium field' not in summary
    from lcprop.pr.gui.request_adapter import apply_pr_request
    restored = decode_pr_static_request(encode_pr_static_request(r))
    panel.material_response.setCurrentIndex(panel.material_response.findData('nonlinear'))
    apply_pr_request(restored, material_panel=window.material_panel,
        beam_panel=window.beam_panel, grid_panel=window.grid_panel, evolution_panel=panel)
    assert window.build_request() == restored
    assert panel.reference_intensity.isHidden()
    panel.set_workflow_id('pr_timedependent')
    assert panel.material_response.count()==1
    assert panel.material_response.currentData()=='nonlinear'
    panel.set_workflow_id('pr_transverse_static')
    assert panel.material_response.findData(MODEL)==-1
    panel.material_response.setCurrentIndex(panel.material_response.findData('linearized'))
    assert not panel.reference_intensity.isHidden()
    window.close()


@pytest.mark.parametrize('precision', ['float32','float64'])
def test_cupy_parity_without_host_planes(precision, monkeypatch):
    try:
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount(): pytest.skip('no CUDA device')
    except Exception: pytest.skip('CuPy/CUDA unavailable locally')
    I=(1.1+.6*np.cos(2*np.pi*np.arange(32)[:,None]/32))*np.ones((1,3))
    I=I.astype(precision); spec=PRReducedFieldLinearSpec(.4,.1,.52)
    expected=solve_pr_reduced_field_linear_intensity(I,spec=spec,backend=BackendSpec('numpy',precision,False))
    def forbidden(*a,**kw): raise AssertionError('host plane transfer')
    with monkeypatch.context() as patch:
        patch.setattr(cp,'asnumpy',forbidden)
        actual=solve_pr_reduced_field_linear_intensity(cp.asarray(I),spec=spec,backend=BackendSpec('cupy',precision,False))
    np.testing.assert_allclose(cp.asnumpy(actual.E),expected.E,atol=2e-6 if precision=='float32' else 2e-14)

@pytest.mark.parametrize('target', ['local','slurm'])
def test_dispatch_envelope_and_exact_product_retrieval_locally(tmp_path, target):
    # Exercise the same worker entry point and envelope used by Slurm; no SSH,
    # scheduler or remote runner is invoked by this bounded local test.
    from lcprop.transport.defaults import default_transport_registry
    from lcprop.transport.io import write_request_package, read_result_package
    from lcprop.transport.executor import execute_run_directory
    from lcprop.pr.operations import PR_STATIC_OPERATION
    registry=default_transport_registry()
    write_request_package(tmp_path,registry=registry,material_id='pr',workflow_id='pr_static',
        request=request(),run_id='local-i-'+target,execution_target=target,
        result_policy='analysis:far_field_intensity')
    assert execute_run_directory(tmp_path,registry=registry,operations=(PR_STATIC_OPERATION,))==0
    decoded=read_result_package(tmp_path,registry=registry)
    assert decoded.result.converged
    assert decoded.result.material_response_summary['model']==MODEL
    assert decoded.result.selected_products['exact']['far_field_intensity'].dtype==np.float64

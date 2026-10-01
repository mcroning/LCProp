"""Bounded completion diagnostics; no production-sized arrays are allocated."""
from fractions import Fraction

import numpy as np
import pytest

import lcprop.pr.static_workflow as workflow


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_completion_metrics_and_comparisons_match_direct(dtype, strided, monkeypatch):
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', 31)
    rng = np.random.default_rng(483)
    a = rng.normal(size=(7, 13, 11)).astype(dtype)
    if strided:
        a = a[:, ::2, ::-1]
    b = a + dtype(1e-6)
    rms, maximum, within = workflow._completion_residual_metrics(
        a, xp=np, rms_tolerance=2.0)
    reference = np.sqrt(np.mean(a.astype(np.float64) ** 2))
    assert rms == pytest.approx(reference, rel=3e-15)
    assert maximum == np.max(np.abs(a))
    assert within
    difference, consistent = workflow._completion_comparison(
        a, b, xp=np, rtol=1e-5, atol=1e-8)
    assert difference == np.max(np.abs(a-b))
    assert consistent == np.all(np.isclose(a, b, rtol=1e-5, atol=1e-8))


@pytest.mark.parametrize('offset', [-1, 0, 1])
def test_exact_rms_boundary_is_independent_of_chunking(offset, monkeypatch):
    threshold = 0.25
    value = threshold if offset == 0 else np.nextafter(threshold, 0 if offset < 0 else 1)
    a = np.full((3, 7, 5), value)
    exact = sum((Fraction(float(v)) ** 2 for v in a.flat), Fraction())
    expected = exact <= a.size * Fraction(threshold) ** 2
    for cap in (3, 19, 1000):
        monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', cap)
        _, _, within = workflow._completion_residual_metrics(a, xp=np, rms_tolerance=threshold)
        assert within == expected


@pytest.mark.parametrize('value', [1e300, 1e-300, 0.0])
def test_scaled_rms_avoids_square_overflow_underflow(value):
    a = np.full((2, 3, 4), value)
    rms, maximum, within = workflow._completion_residual_metrics(a, xp=np, rms_tolerance=1.0)
    assert rms == pytest.approx(value, rel=2e-15, abs=0)
    assert maximum == value
    assert within == (value <= 1)


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_residual_fails(value):
    a = np.ones((2, 3, 4))
    a[0, 0, 0] = value
    assert not workflow._completion_residual_metrics(a, xp=np, rms_tolerance=2)[2]


def test_comparison_boundaries_complex_and_nonfinite(monkeypatch):
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', 3)
    a = np.array([0, .1, np.nextafter(.1, 1), np.inf, np.nan, 1j]).reshape(2, 3)
    b = np.array([0, 0, 0, np.inf, np.nan, 1j]).reshape(2, 3)
    with np.errstate(invalid='ignore'):
        maximum, consistent = workflow._completion_comparison(a, b, xp=np, rtol=0, atol=.1)
        assert np.isnan(maximum)
        assert consistent == np.all(np.isclose(a, b, rtol=0, atol=.1))


def test_all_diagnostic_operations_are_chunk_bounded(monkeypatch):
    cap = 32
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', cap)

    class Guarded(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            for item in inputs:
                if isinstance(item, np.ndarray):
                    assert item.size <= cap, 'full-volume diagnostic operation'
            result = getattr(ufunc, method)(
                *(np.asarray(v) if isinstance(v, Guarded) else v for v in inputs), **kwargs)
            return result.view(Guarded) if isinstance(result, np.ndarray) else result

    class Backend:
        float64 = np.float64

        def __getattr__(self, name):
            def call(*args, **kwargs):
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        assert arg.size <= cap
                out = getattr(np, name)(*args, **kwargs)
                return out.view(Guarded) if isinstance(out, np.ndarray) else out
            return call

    a = np.linspace(-2, 2, 7*13*11).reshape(7, 13, 11).view(Guarded)
    b = np.asarray(a).copy().view(Guarded)
    workflow._completion_residual_metrics(a, xp=Backend(), rms_tolerance=3)
    assert workflow._completion_comparison(a, b, xp=Backend(), rtol=1e-11, atol=1e-12) == (0, True)


@pytest.mark.parametrize("linearized", [False, True])
def test_workflow_matches_old_direct_completion(monkeypatch, linearized):
    from tests.test_pr_static_workflow import _static_request
    request = _static_request(Nz=3)
    if linearized:
        from dataclasses import replace
        from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
        request = replace(request, material_response=PRTransverseMaterialResponseSpec(
            model="field_linear_local_intensity"))
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', 7)
    bounded = workflow.run_pr_static(request)

    def direct_metrics(a, *, xp, rms_tolerance):
        rms, maximum = workflow._residual_metrics(a, xp=xp)
        return rms, maximum, rms <= rms_tolerance

    def direct_comparison(a, b, *, xp, rtol, atol):
        return float(xp.max(xp.abs(a-b))), bool(xp.all(xp.isclose(a, b, rtol=rtol, atol=atol)))

    monkeypatch.setattr(workflow, '_completion_residual_metrics', direct_metrics)
    monkeypatch.setattr(workflow, '_completion_comparison', direct_comparison)
    direct = workflow.run_pr_static(request)
    assert bounded.converged == direct.converged
    assert bounded.status == direct.status
    for name in ('A_final', 'E_final', 'source_intensity_stack', 'residual_stack'):
        np.testing.assert_array_equal(getattr(bounded, name), getattr(direct, name))
    for name, value in direct.replay_diagnostics.items():
        if isinstance(value, float):
            assert bounded.replay_diagnostics[name] == pytest.approx(value, rel=3e-15, abs=0)
        else:
            assert bounded.replay_diagnostics[name] == value


def test_host_result_ownership_and_no_duplicate_device_transfer(monkeypatch):
    a = np.arange(12.)
    assert not np.shares_memory(a, workflow._owned_host_result(a))
    transferred = a.copy()
    device = object()
    monkeypatch.setattr(workflow, 'asnumpy', lambda value: transferred if value is device else value)
    assert workflow._owned_host_result(device) is transferred


def test_cupy_completion_parity_when_available(monkeypatch):
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', 31)
    a = np.random.default_rng(19).normal(size=(7, 13, 11))
    cpu = workflow._completion_residual_metrics(a, xp=np, rms_tolerance=2)
    gpu = workflow._completion_residual_metrics(cp.asarray(a), xp=cp, rms_tolerance=2)
    assert gpu[:2] == pytest.approx(cpu[:2], rel=3e-15)
    assert gpu[2] == cpu[2]
    b = a + 1e-6
    assert workflow._completion_comparison(cp.asarray(a), cp.asarray(b), xp=cp,
        rtol=1e-5, atol=1e-8) == workflow._completion_comparison(
            a, b, xp=np, rtol=1e-5, atol=1e-8)


def test_mixed_boundary_decision_uses_exact_definition(monkeypatch):
    threshold = 0.25
    a = np.array([np.nextafter(threshold, 0), np.nextafter(threshold, 1)]).reshape(1, 2, 1)
    expected = sum((Fraction(float(v)) ** 2 for v in a.flat), Fraction()) <= (
        a.size * Fraction(threshold) ** 2)
    for cap in (1, 2, 17):
        monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', cap)
        assert workflow._completion_residual_metrics(a, xp=np, rms_tolerance=threshold)[2] == expected


def test_native_geometry_chunk_plan_without_allocating_volume():
    shape = (80, 8192, 4096)
    sizes = [np.prod([s.stop-s.start for s in index])
             for index in workflow._completion_slices(shape)]
    assert sum(sizes) == 80*8192*4096
    assert max(sizes) <= workflow._COMPLETION_CHUNK_ELEMENTS


@pytest.mark.parametrize('threshold', [0.25, 1e-300, 1e300, np.nextafter(0., 1.)])
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_exact_boundary_across_float64_range(threshold, direction, monkeypatch):
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', 17)
    value = threshold if direction == 0 else np.nextafter(
        threshold, 0. if direction < 0 else np.inf)
    residual = np.full((3, 7, 5), value)
    _, _, within = workflow._completion_residual_metrics(
        residual, xp=np, rms_tolerance=threshold)
    assert within == (value <= threshold)


def test_exact_square_summary_all_exponents_and_strided_samples():
    rng = np.random.default_rng(1103)
    # Include every finite exponent and random significands, plus endpoints.
    bits = (np.arange(2047, dtype=np.uint64) << np.uint64(52)) | rng.integers(
        0, 1 << 52, size=2047, dtype=np.uint64)
    values = np.concatenate((bits.view(np.float64), [0., -0., np.nextafter(0., 1.),
        np.finfo(np.float64).max, -np.finfo(np.float64).max]))[::-1]
    summary = workflow._exact_square_summary(values, xp=np)
    actual = sum(int(v) << (16*i) for i, v in enumerate(summary))
    expected = sum((Fraction(float(v)) ** 2 for v in values), Fraction()) * (1 << 2148)
    assert expected.denominator == 1
    assert actual == expected.numerator


@pytest.mark.parametrize('count', [257, 509, 2048])
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_boundary_fallback_only_exports_fixed_summaries(count, direction, monkeypatch):
    cap = 512
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', cap)
    operations = []

    class Guarded(np.ndarray):
        def __array_ufunc__(self, ufunc, method, *args, **kwargs):
            for arg in args:
                if isinstance(arg, np.ndarray):
                    assert arg.size <= cap, 'unbounded diagnostic temporary'
            if 'out' in kwargs:
                kwargs['out'] = tuple(np.asarray(v) if isinstance(v, Guarded) else v
                                      for v in kwargs['out'])
            result = getattr(ufunc, method)(
                *(np.asarray(v) if isinstance(v, Guarded) else v for v in args), **kwargs)
            if isinstance(result, np.ndarray):
                assert result.size <= cap
                return result.view(Guarded)
            return result

    class Backend:
        float64, uint64, int64 = np.float64, np.uint64, np.int64

        def __getattr__(self, name):
            def call(*args, **kwargs):
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        assert arg.size <= cap
                operations.append(name)
                result = getattr(np, name)(*args, **kwargs)
                if isinstance(result, np.ndarray):
                    assert result.size <= cap
                    return result.view(Guarded)
                return result
            return call

    exported = []
    summaries = []
    original = workflow._exact_square_summary

    def summary(chunk, *, xp):
        result = original(chunk, xp=xp)
        summaries.append(id(result))
        return result

    def transfer(value):
        assert id(value) == summaries[-1], 'residual samples exported'
        assert value.shape == (264,)
        assert value.dtype == np.float64
        exported.append(value.nbytes)
        return np.asarray(value)

    monkeypatch.setattr(workflow, '_exact_square_summary', summary)
    monkeypatch.setattr(workflow, 'asnumpy', transfer)
    threshold = .25
    value = threshold if direction == 0 else np.nextafter(
        threshold, 0 if direction < 0 else 1)
    residual = np.full((1, count, 1), value).view(Guarded)
    _, _, within = workflow._completion_residual_metrics(
        residual, xp=Backend(), rms_tolerance=threshold)
    chunks = len(list(workflow._completion_slices(residual.shape)))
    assert within == (direction <= 0)
    assert len(exported) == chunks
    assert exported == [2112] * chunks
    # Fixed Python histogram work per chunk, not per residual element.
    assert operations.count('bincount') == 14 * chunks


def test_cupy_exact_boundary_summary_when_available(monkeypatch):
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    monkeypatch.setattr(workflow, '_COMPLETION_CHUNK_ELEMENTS', 17)
    for threshold in (.25, 1e-300, 1e300, np.nextafter(0., 1.)):
        for value in (np.nextafter(threshold, 0), threshold, np.nextafter(threshold, np.inf)):
            a = np.full((3, 7, 5), value)
            np.testing.assert_array_equal(cp.asnumpy(workflow._exact_square_summary(cp.asarray(a), xp=cp)),
                                          workflow._exact_square_summary(a, xp=np))
            assert workflow._completion_residual_metrics(cp.asarray(a), xp=cp,
                rms_tolerance=threshold)[2] == (value <= threshold)


def test_exact_histogram_maximum_chunk_carry_and_counts():
    count = 262144
    value = np.finfo(np.float64).max
    summary = workflow._exact_square_summary(np.full((1, count), value), xp=np)
    actual = sum(int(v) << (16*i) for i, v in enumerate(summary))
    expected = Fraction(float(value)) ** 2 * count * (1 << 2148)
    assert expected.denominator == 1
    assert actual == expected.numerator
    assert summary.shape == (264,)
    assert np.all(summary < 2**38)

"""Exact historical presentation arithmetic, isolated from solver normalization."""
from dataclasses import replace
import numpy as np
import pytest

import lcprop.pr.longitudinal_cuts as cuts
import lcprop.pr.static_workflow as workflow
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.static_transport_codec import (
    encode_pr_static_transport_result, decode_pr_static_transport_result,
)
from tests.test_pr_static_fast_construction import request_for


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
@pytest.mark.parametrize('shape', [(1, 15, 7), (3, 17, 8), (5, 2, 71)])
def test_bounded_reference_exact_historical_numpy(monkeypatch, dtype, shape):
    rng = np.random.default_rng(53)
    a = (rng.normal(size=shape) + 1j*rng.normal(size=shape)).astype(dtype)
    a = a[:, ::-1, ::-1]  # Strided launch views must not require a plane copy.
    before = a.tobytes()
    monkeypatch.setattr(cuts, '_PRESENTATION_CHUNK_ELEMENTS', 31)
    transfers = []
    class Device:
        def __init__(self, value): self.value = value
        @property
        def ndim(self): return self.value.ndim
        @property
        def shape(self): return self.value.shape
        def __getitem__(self, key): return Device(self.value[key])
    def host(chunk):
        assert chunk.value.ndim == 2 and chunk.value.size <= 31
        transfers.append(chunk.value.nbytes)
        return chunk.value.copy()
    expected = channel_peak_intensity_reference(a, xp=np)
    assert cuts.presentation_peak_intensity_reference(Device(a), asnumpy=host).hex() == expected.hex()
    assert sum(transfers) == a.nbytes
    assert a.tobytes() == before
    def forbidden(*args): raise AssertionError('host launch was copied to host again')
    assert cuts.presentation_peak_intensity_reference(a, asnumpy=forbidden).hex() == expected.hex()


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_raw_cut_transfer_shared_arithmetic_and_nonmutation(precision):
    source = np.random.default_rng(7).uniform(.3, 2., (3, 12, 4)).astype(precision)
    saved = source.tobytes()
    grid = dict(Nx=12, Ny=4, dx_um=2., dy_um=4.)
    options = dict(grid_summary=grid, peak_intensity_reference=.00935869237154356,
                   background_intensity=.30000000000000004)
    raw = [source[:, :, 1], source[:, 5, :]]
    seen = []
    def transfer(value):
        expected = raw[len(seen)]
        assert np.shares_memory(value, source)
        assert value.tobytes() == expected.tobytes()
        seen.append(value.shape)
        return value.copy()
    expected = cuts.extract_longitudinal_optical_intensity_cuts(source, **options)
    actual = cuts.extract_backend_longitudinal_optical_intensity_cuts(source, asnumpy=transfer, **options)
    assert actual.xz.tobytes() == expected.xz.tobytes()
    assert actual.yz.tobytes() == expected.yz.tobytes()
    assert source.tobytes() == saved
    assert seen == [(3, 12), (3, 4)]


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('scattering', [False, True])
def test_one_ulp_scientific_reference_does_not_leak_into_products(monkeypatch, precision, scattering):
    r = request_for(True, scattering, precision)
    original = workflow.channel_peak_intensity_reference
    scientific = []
    def shifted(a, *, xp):
        reference = original(a, xp=xp)
        value = float(np.nextafter(reference, np.inf))
        scientific.append((reference, value))
        return value
    monkeypatch.setattr(workflow, 'channel_peak_intensity_reference', shifted)
    previews = []
    original_preview = workflow.make_fast_intensity_preview
    def preview(source, **kw):
        previews.append(kw['peak_intensity_reference'])
        return original_preview(source, **kw)
    monkeypatch.setattr(workflow, 'make_fast_intensity_preview', preview)
    original_presentation = workflow.presentation_peak_intensity_reference
    def presentation(*args, **kwargs):
        # This helper is reached only after scientific completion/replay.
        import inspect
        frame = inspect.currentframe().f_back
        assert frame.f_locals['completed_slices'] == 3
        assert 'field_consistent' in frame.f_locals
        return original_presentation(*args, **kwargs)
    monkeypatch.setattr(workflow, 'presentation_peak_intensity_reference', presentation)
    full = workflow.run_pr_static(r)
    fast = workflow.run_pr_static(r, result_policy='fast')
    encoded = encode_pr_static_transport_result(full, 'fast')
    expected = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    encoded_fast = encode_pr_static_transport_result(fast, 'fast')
    decoded = decode_pr_static_transport_result(encoded_fast.payload.metadata, encoded_fast.payload.arrays)
    assert len(scientific) == 2 and scientific[0] == scientific[1]
    assert scientific[0][0] != scientific[0][1]
    assert previews == [scientific[0][0]]
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        assert getattr(fast, name).tobytes() == getattr(expected, name).tobytes()
        assert getattr(decoded, name).tobytes() == getattr(expected, name).tobytes()
    assert fast.intensity_preview_metadata == expected.intensity_preview_metadata
    assert fast.iteration_records == full.iteration_records
    assert fast.replay_diagnostics == full.replay_diagnostics


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('scattering', [False, True])
def test_native_presentation_exact_when_available(precision, scattering):
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount(): pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError: pytest.skip('CUDA unavailable')
    request = request_for(True, scattering, precision)
    request = replace(request, backend=replace(request.backend, backend='cupy'))
    full = workflow.run_pr_static(request)
    fast = workflow.run_pr_static(request, result_policy='fast')
    encoded = encode_pr_static_transport_result(full, 'fast')
    old = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        assert getattr(fast, name).tobytes() == getattr(old, name).tobytes()
    assert fast.intensity_preview_metadata == old.intensity_preview_metadata
    launch = cp.asarray(full.A_initial)
    before = cp.asnumpy(launch).tobytes()
    assert cuts.presentation_peak_intensity_reference(launch, asnumpy=cp.asnumpy).hex() == channel_peak_intensity_reference(full.A_initial, xp=np).hex()
    assert cp.asnumpy(launch).tobytes() == before


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('scattering', [False, True])
@pytest.mark.parametrize('completed', [0, 1, 2])
def test_cancelled_presentation_exact(precision, scattering, completed):
    from lcprop.core.execution import CancellationToken
    request = request_for(True, scattering, precision)
    def run(policy):
        token = CancellationToken()
        if completed == 0:
            token.cancel()
        def progress(value):
            if value.message == 'PR static z slice completed' and value.completed_units == completed:
                token.cancel()
        return workflow.run_pr_static(request, result_policy=policy,
            cancellation_token=token, progress_callback=progress)
    full, fast = run('full'), run('fast')
    assert fast.completed_slices == full.completed_slices == completed
    assert fast.status == full.status == 'cancelled'
    assert not fast.converged and not full.converged
    assert fast.iteration_records == full.iteration_records
    assert fast.slice_summaries == full.slice_summaries
    assert fast.power_initial == full.power_initial
    assert fast.power_final == full.power_final
    encoded = encode_pr_static_transport_result(full, 'fast')
    expected = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        actual, wanted = getattr(fast, name), getattr(expected, name)
        if completed:
            assert actual.tobytes() == wanted.tobytes()
        else:
            assert actual is wanted is None
    assert fast.intensity_preview_metadata == expected.intensity_preview_metadata
    assert fast.replay_diagnostics == full.replay_diagnostics


@pytest.mark.parametrize('dtype', ['complex64', 'complex128'])
def test_host_acceptance_uploads_owned_chunks_without_readback(monkeypatch, dtype):
    original = (np.arange(120).reshape(2, 12, 5) * .001 + .03j).astype(dtype)
    expected = original.copy()
    expected_reference = channel_peak_intensity_reference(expected, xp=np)
    monkeypatch.setattr(cuts, '_PRESENTATION_CHUNK_ELEMENTS', 17)
    pending = []
    import weakref
    released = []
    class UploadOnly:
        @staticmethod
        def empty(shape, dtype): return np.empty(shape, dtype=dtype)
        @staticmethod
        def asarray(value):
            assert value.flags.owndata and value.size <= 17
            assert not np.shares_memory(value, original)
            assert all(ref() is None for ref in released)
            pending.append(value)
            return value
    def synchronize():
        assert len(pending) == 1  # Upload storage is live until completion.
        released.append(weakref.ref(pending.pop()))
    accepted, reference = cuts.copy_host_launch_for_presentation(
        original, xp=UploadOnly(), synchronize=synchronize)
    original.fill(9 + 4j)
    assert accepted.tobytes() == expected.tobytes()
    assert reference.hex() == expected_reference.hex()
    assert all(ref() is None for ref in released)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('native', [False, True])
def test_supplied_host_launch_scalar_survives_caller_mutation(monkeypatch, precision, native):
    if native:
        cp = pytest.importorskip('cupy')
        try:
            if not cp.cuda.runtime.getDeviceCount(): pytest.skip('no CUDA device')
        except cp.cuda.runtime.CUDARuntimeError: pytest.skip('CUDA unavailable')
    dtype = np.complex64 if precision == 'float32' else np.complex128
    launch = (.03 + .004*np.cos(np.arange(12)[:, None]) + np.zeros((12, 4)))[None].astype(dtype)
    expected_reference = channel_peak_intensity_reference(launch, xp=np)
    base = request_for(True, True, precision)
    base = replace(base, backend=replace(base.backend, backend='cupy' if native else 'numpy'))
    original_transfer = workflow.asnumpy
    original_scientific_reference = workflow.channel_peak_intensity_reference
    original_reference = workflow.presentation_peak_intensity_reference
    def reference(value, **kw):
        assert isinstance(value, np.ndarray), 'presentation launch read back from GPU'
        assert not np.shares_memory(value, supplied[0])
        return original_reference(value, **kw)
    monkeypatch.setattr(workflow, 'presentation_peak_intensity_reference', reference)
    supplied = []
    def run(policy, mutate):
        host = launch.copy(); supplied[:] = [host]
        def scientific_reference(value, **kwargs):
            # Mutation immediately after launch acceptance, before propagation.
            # Avoid progress callbacks, which intentionally export live fields.
            answer = original_scientific_reference(value, **kwargs)
            if mutate: host.fill(5 + 6j)
            return answer
        def transfer(value):
            if policy == 'fast':
                assert value.dtype.kind != 'c', 'unrequested complex launch/endpoint readback'
            return original_transfer(value)
        with monkeypatch.context() as patch:
            patch.setattr(workflow, 'asnumpy', transfer)
            patch.setattr(workflow, 'channel_peak_intensity_reference', scientific_reference)
            return workflow.run_pr_static(replace(base, initial_A=host), result_policy=policy)
    full = run('full', True)
    assert full.A_initial.tobytes() == launch.tobytes()
    assert channel_peak_intensity_reference(full.A_initial, xp=np).hex() == expected_reference.hex()
    fast = run('fast', True)
    control = run('fast', False)
    encoded = encode_pr_static_transport_result(full, 'fast')
    expected = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    encoded_fast = encode_pr_static_transport_result(fast, 'fast')
    decoded = decode_pr_static_transport_result(encoded_fast.payload.metadata, encoded_fast.payload.arrays)
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        for result in (fast, control, decoded):
            assert getattr(result, name).tobytes() == getattr(expected, name).tobytes()
    assert fast.iteration_records == control.iteration_records == full.iteration_records
    assert fast.replay_diagnostics == control.replay_diagnostics == full.replay_diagnostics


@pytest.mark.parametrize('policy', ['full', 'fast', 'interactive', 'analysis',
                                    'analysis:output_intensity', 'analysis:complex_input'])
@pytest.mark.parametrize('outcome', ['cancelled', 'not_converged', 'zero', 'failure', 'host'])
def test_partial_presentation_transfer_boundary(monkeypatch, policy, outcome):
    _check_partial_transfer_boundary(monkeypatch, policy, outcome, native=False)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('outcome', ['cancelled', 'host', 'zero', 'failure'])
def test_native_partial_presentation_transfer_boundary(monkeypatch, precision, outcome):
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    _check_partial_transfer_boundary(monkeypatch, 'fast', outcome,
                                     native=True, precision=precision)


def _check_partial_transfer_boundary(monkeypatch, policy, outcome, *, native,
                                     precision='float64'):
    """Spy only on presentation transfers; Full's requested arrays are separate.

    CPU coverage uses a device proxy at the streaming-helper boundary. Conditional
    native coverage observes real CuPy chunks without proxying scientific arrays.
    """
    import hashlib
    import inspect
    from lcprop.core.execution import CancellationToken
    request = request_for(True, True, precision)
    monkeypatch.setattr(cuts, '_PRESENTATION_CHUNK_ELEMENTS', 17)
    if native:
        request = replace(request, backend=replace(request.backend, backend='cupy'))
    if outcome == 'host':
        dtype = np.complex64 if precision == 'float32' else np.complex128
        request = replace(request, initial_A=np.full((1, 12, 4), .03 + .002j, dtype=dtype))
    reference_calls, transfers, scientific_hashes = [], [], []
    original_reference = workflow.presentation_peak_intensity_reference
    original_comparison = workflow._completion_comparison
    original_asnumpy = workflow.asnumpy
    product_started = [False]

    def comparison(actual, expected, **kw):
        assert not product_started[0], 'scientific diagnostics resumed after presentation'
        scientific_hashes.append(tuple(hashlib.sha256(
            original_asnumpy(value).tobytes()).hexdigest() for value in (actual, expected)))
        difference, consistent = original_comparison(actual, expected, **kw)
        # Deterministic failed completion diagnostic, without changing any state.
        return difference, False if outcome == 'not_converged' else consistent

    class DeviceProxy:
        def __init__(self, value): self.value = value
        @property
        def ndim(self): return self.value.ndim
        @property
        def shape(self): return self.value.shape
        def __getitem__(self, key): return DeviceProxy(self.value[key])

    def reference(value, **kw):
        if 'asnumpy' not in kw:  # Accepted CPU host launch, not a readback.
            return original_reference(value, **kw)
        frame = inspect.currentframe().f_back
        assert frame.f_locals['completed_slices'] > 0
        assert 'converged' in frame.f_locals and 'residual_consistent' in frame.f_locals
        assert not frame.f_locals['converged']
        product_started[0] = True
        reference_calls.append(value.nbytes)
        if isinstance(value, np.ndarray) and policy == 'analysis:complex_input':
            return original_reference(value, **kw)  # Already requested host field.
        def transfer(chunk):
            data = chunk.value if isinstance(chunk, DeviceProxy) else chunk
            assert data.size <= cuts._PRESENTATION_CHUNK_ELEMENTS
            transfers.append(data.nbytes)
            return original_asnumpy(data).copy()
        device = value if native else DeviceProxy(value)
        return original_reference(device, asnumpy=transfer)

    def run(selected):
        product_started[0] = False
        scientific_hashes.clear()
        token = CancellationToken()
        if outcome == 'zero': token.cancel()
        def progress(event):
            if event.message == 'PR static z slice completed':
                assert not product_started[0], 'slice accepted after presentation'
                if outcome in ('cancelled', 'host') and event.completed_units == 1:
                    token.cancel()
        return workflow.run_pr_static(request, result_policy=selected,
            cancellation_token=token, progress_callback=progress)

    monkeypatch.setattr(workflow, '_completion_comparison', comparison)
    monkeypatch.setattr(workflow, 'presentation_peak_intensity_reference', reference)
    if outcome == 'failure':
        def fail(*args, **kwargs):
            raise RuntimeError('injected before first scientific slice')
        monkeypatch.setattr(workflow, 'channel_peak_intensity_reference', fail)
        with pytest.raises(RuntimeError, match='before first scientific slice'):
            run(policy)
        assert not reference_calls and not transfers and not scientific_hashes
        return
    full = run('full')
    baseline_hashes = list(scientific_hashes)
    result = run(policy)
    assert scientific_hashes == baseline_hashes
    assert result.iteration_records == full.iteration_records
    assert result.slice_summaries == full.slice_summaries
    assert result.replay_diagnostics == full.replay_diagnostics
    assert result.completed_slices == full.completed_slices
    assert result.status == full.status == ('not_converged' if outcome == 'not_converged' else 'cancelled')
    assert not result.converged and not full.converged
    needs_reference = policy != 'full' and outcome not in ('zero', 'host')
    assert len(reference_calls) == int(needs_reference)
    needs_readback = needs_reference and policy != 'analysis:complex_input'
    assert sum(transfers) == (full.A_initial.nbytes if needs_readback else 0)
    if policy != 'full':
        encoded = encode_pr_static_transport_result(full, 'fast')
        expected = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
        for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
            actual, wanted = getattr(result, name), getattr(expected, name)
            if result.completed_slices:
                assert actual.dtype == wanted.dtype and actual.tobytes() == wanted.tobytes()
            else:
                assert actual is wanted is None
        assert result.intensity_preview_metadata == expected.intensity_preview_metadata

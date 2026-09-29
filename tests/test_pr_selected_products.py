"""Selected products preserve science without transporting endpoint dependencies."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from tests.test_pr_static_fast_construction import request_for
from lcprop.pr.static_workflow import run_pr_static
from lcprop.pr.static_transport_codec import encode_pr_static_transport_result, decode_pr_static_transport_result
from lcprop.pr.operations import PR_STATIC_OPERATION
from lcprop.pr.selected_products import construct_selected_products, display_shape
from lcprop.transport.result_policy import ANALYSIS_PRODUCTS, static_product_selection
from lcprop.pr.visualization import block_average_2d
from lcprop.optics.splitstep import total_intensity
from lcprop.optics.farfield import direction_cosine_spectrum


@pytest.fixture(scope='module')
def full():
    return run_pr_static(request_for(True, True))


def selected_from(full, policy):
    return construct_selected_products(full.A_initial, full.A_final, policy=policy,
        grid=full.grid_summary, launch=full.launch_summary, xp=np, asnumpy=np.asarray)


@pytest.mark.parametrize('name', ANALYSIS_PRODUCTS)
def test_analysis_independent_exact_products(name, full):
    result = run_pr_static(request_for(True, True), result_policy='analysis:' + name)
    encoded = encode_pr_static_transport_result(result, 'analysis:' + name)
    restored = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    assert (restored.A_initial is not None) == (name == 'complex_input')
    assert (restored.A_final is not None) == (name == 'complex_output')
    old = encode_pr_static_transport_result(full, "fast")
    legacy = PR_STATIC_OPERATION.to_run_data(decode_pr_static_transport_result(old.payload.metadata, old.payload.arrays))
    if name in ('input_intensity', 'output_intensity', 'far_field_intensity'):
        np.testing.assert_array_equal(restored.selected_products['exact'][name], legacy.fields[name].data)
    else:
        np.testing.assert_array_equal(getattr(restored, 'A_initial' if name == 'complex_input' else 'A_final'),
                                      getattr(full, 'A_initial' if name == 'complex_input' else 'A_final'))
    assert restored.replay_diagnostics == full.replay_diagnostics
    assert restored.iteration_records == full.iteration_records
    assert restored.slice_summaries == full.slice_summaries
    assert restored.power_final == full.power_final
    assert set(restored.selected_products['exact']) == ({name} if 'intensity' in name else set())
    PR_STATIC_OPERATION.to_run_data(restored)


def test_large_previews_bound_intensity_construction_and_transfers(monkeypatch):
    import lcprop.pr.selected_products as module
    rng = np.random.default_rng(713)
    shape = (1031, 65)  # nondivisible bins, no large scientific allocation
    initial = (rng.normal(size=(2, *shape)) + 1j*rng.normal(size=(2, *shape))) * .01
    final = initial * np.exp(1j * np.arange(shape[0])[None, :, None] * .02)
    grid = {'dx_um': .2, 'dy_um': .4}
    launch = {'coherence_groups': ['a', 'b'], 'wavelengths_um': [.633, .633], 'refractive_index': 2.4}
    original = module.total_intensity
    calls, transfers = [], []
    def intensity(a, **kwargs):
        assert a.shape[-2:] != shape
        calls.append(a.shape)
        return original(a, **kwargs)
    def host(value):
        assert value.ndim <= 1  # row sums and coordinates only
        transfers.append(value.shape)
        return np.asarray(value)
    monkeypatch.setattr(module, 'total_intensity', intensity)
    result = module.construct_selected_products(initial, final, policy='interactive', grid=grid, launch=launch, xp=np, asnumpy=host)
    assert calls and transfers and not result['exact']
    expected = {
        'input_intensity': total_intensity(initial, coherence_groups=['a','b']),
        'output_intensity': total_intensity(final, coherence_groups=['a','b']),
        'far_field_intensity': direction_cosine_spectrum(final, dx_um=.2, dy_um=.4,
             wavelength_um=.633, refractive_index=2.4, coherence_groups=['a','b']).intensity,
    }
    for name, plane in expected.items():
        reference, xe, ye = block_average_2d(plane, max_x=display_shape(shape)[0], max_y=display_shape(shape)[1])
        record = result['previews'][name]
        np.testing.assert_allclose(record['data'], reference.astype(np.float32), rtol=2e-7)
        meta = record['metadata']
        assert meta['visualization_only'] and meta['original_shape'] == list(shape)
        for axis, edges in zip(meta['axes'], (xe, ye)):
            assert meta['block_bounds'][axis] == edges.tolist()
            original_coords = result['exact_coordinates'][axis]
            np.testing.assert_array_equal(meta['coordinates'][axis], [original_coords[a:b].mean() for a,b in zip(edges[:-1],edges[1:])])


@pytest.mark.parametrize('damage', ['missing_preview', 'missing_exact', 'bounds', 'availability', 'complex'])
def test_selected_codec_fails_closed(damage):
    result = run_pr_static(request_for(True, False), result_policy='analysis:output_intensity')
    encoded = encode_pr_static_transport_result(result, 'analysis:output_intensity')
    meta = copy.deepcopy(encoded.payload.metadata)
    arrays = dict(encoded.payload.arrays)
    products = meta['selected_products']
    if damage == 'missing_preview':
        products['previews'].pop('input_intensity')
    elif damage == 'missing_exact':
        products['exact'].clear()
    elif damage == 'bounds':
        products['previews']['input_intensity']['metadata']['block_bounds']['x'][0] = 1
    elif damage == 'availability':
        products['availability']['output_intensity'] = 'not_selected'
    else:
        meta['A_initial'] = {'__lcprop_array__': 'injected'}
        arrays['injected'] = np.zeros((1, result.grid_summary['Nx'], result.grid_summary['Ny']), complex)
    with pytest.raises(ValueError):
        decode_pr_static_transport_result(meta, arrays)


def test_historical_fast_remains_exact_endpoint_artifact(full):
    encoded = encode_pr_static_transport_result(full, 'fast')
    encoded.payload.metadata.pop('selected_products', None)  # actual historical schema
    old = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    np.testing.assert_array_equal(old.A_initial, full.A_initial)
    assert old.retention_summary['policy'] == 'fast'
    assert old.selected_products is None
    assert PR_STATIC_OPERATION.to_run_data(old).fields['input_intensity'].data.dtype == np.float64
    assert static_product_selection('fast') == ('interactive', ())


def test_packaging_failure_removes_only_owned_temporary(tmp_path, monkeypatch):
    from lcprop.transport import executor
    from lcprop.transport.defaults import default_transport_registry
    from lcprop.transport.io import write_request_package
    registry = default_transport_registry()
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id='pr_static',
        request=request_for(True, False), run_id='cleanup-test', execution_target='local', result_policy='fast')
    sibling = tmp_path / 'output.incomplete.unrelated'
    sibling.mkdir()
    (sibling/'keep').write_bytes(b'untouched')
    seen = []
    def fail(*args, output_directory, **kwargs):
        output_directory.mkdir()
        (output_directory/'result_arrays.npz.tmp').write_bytes(b'partial-scientific-data')
        seen.append(output_directory)
        raise OSError('injected packaging failure')
    monkeypatch.setattr(executor, 'write_result_package', fail)
    assert executor.execute_run_directory(tmp_path, registry=registry) == 1
    assert seen and not seen[0].exists()
    assert (sibling/'keep').read_bytes() == b'untouched'
    assert (tmp_path/'request'/'request.json').exists()
    assert 'injected packaging failure' in (tmp_path/'output'/'failure.json').read_text()


def test_display_budget():
    assert display_shape((8192,4096)) == (1024,512)
    assert display_shape((4096,8192)) == (512,1024)
    # Cuts, MPR, endpoint previews and full 1-D scientific coordinates.
    assert 80*(8192+4096)*8 + 80*96*96*4 + 3*1024*512*4 + 2*(8192+4096)*8 == 17301504


def test_gui_display_is_bounded_without_losing_exact_data(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from lcprop.gui.views.image_view import ImageView
    from lcprop.products.data_model import make_field
    app = QApplication.instance() or QApplication([])
    a = np.arange(2053*1031, dtype=float).reshape(2053,1031)
    field = make_field('output_intensity', 'Exact output', a, ('x','y'), 'intensity', {'x':'um','y':'um'})
    view = ImageView()
    monkeypatch.setattr(view, 'draw_idle', lambda: None)
    artist = view.image
    view.set_field(field, extent=(-2,2,-1,1))
    assert max(view.image.get_array().shape) <= 1024
    assert view._field.data is a and view._raw_shape == a.shape
    assert tuple(view.image.get_extent()) == (-2,2,-1,1)
    cached = view._display_data
    view.set_field(field, extent=(-2,2,-1,1))
    assert view._display_data is cached and view.image is artist
    view.close()


def test_gui_analysis_intent_roundtrip(window, tmp_path):
    from lcprop.pr.static_workflow import PR_STATIC_WORKFLOW
    w = window
    w.evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    w._set_product_policy('analysis:complex_output,output_intensity')
    before = w.build_request()
    path = tmp_path/'analysis.json'
    w.save_experiment_to(path)
    w._set_product_policy('full')
    w.load_experiment_from(path)
    assert w.build_request() == before
    assert w._current_execution_intent().retrieval_policy == 'analysis:complex_output,output_intensity'
    assert w._analysis_actions['output_intensity'].isChecked()
    assert not w._analysis_actions['input_intensity'].isChecked()
    w._set_product_policy('fast')
    assert w.result_policy_selector.currentText() == 'Interactive'


# Reuse the catalog-isolated GUI fixture; no machine profile or SSH access.
from tests.test_pr_execution_intent import window


@pytest.mark.parametrize('policy', ['interactive', 'analysis:output_intensity,complex_output'])
def test_executor_selected_roundtrip_and_request_identity(tmp_path, policy):
    from lcprop.transport.defaults import default_transport_registry
    from lcprop.transport.io import write_request_package, read_result_package
    from lcprop.transport.executor import execute_run_directory
    from lcprop.pr.static_transport_codec import encode_pr_static_transport_request
    registry = default_transport_registry()
    request = request_for(True, True)
    before = encode_pr_static_transport_request(request).payload.metadata
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id='pr_static',
        request=request, run_id='selected-execution', execution_target='local', result_policy=policy)
    assert execute_run_directory(tmp_path, registry=registry) == 0
    received = read_result_package(tmp_path, registry=registry)
    assert received.result.selected_products['policy'] == static_product_selection(policy)[0]
    assert received.result.A_initial is None
    assert encode_pr_static_transport_request(request).payload.metadata == before
    assert not list(tmp_path.glob('output.incomplete.*'))


def test_autoscale_not_repeated_on_same_scientific_field(monkeypatch):
    from PySide6.QtWidgets import QApplication
    import lcprop.gui.views.image_pane as module
    from lcprop.products.data_model import make_field
    app = QApplication.instance() or QApplication([])
    pane = module.ImagePane()
    field = make_field('exact', 'Exact', np.arange(2048*512.).reshape(2048,512), ('x','y'), 'intensity', {})
    calls = []
    original = module.display_limits
    def limits(f):
        calls.append(f)
        return original(f)
    monkeypatch.setattr(module, 'display_limits', limits)
    a = pane._limits_for_field(field)
    assert pane._limits_for_field(field) == a
    assert len(calls) == 1
    pane.close()


def test_cupy_selected_products_when_available():
    cp = pytest.importorskip('cupy')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    request = request_for(True, True)
    request = replace(request, backend=replace(request.backend, backend='cupy'))
    full = run_pr_static(request)
    selected = run_pr_static(request, result_policy='analysis:output_intensity,far_field_intensity')
    expected = selected_from(full, 'analysis:output_intensity,far_field_intensity')
    for key in expected['exact']:
        np.testing.assert_allclose(selected.selected_products['exact'][key], expected['exact'][key], rtol=1e-12, atol=1e-15)
    assert selected.replay_diagnostics == full.replay_diagnostics


def test_carrier_scalars_preserved_without_endpoint_host_transfer(monkeypatch):
    import lcprop.pr.carrier_power as module
    rng = np.random.default_rng(42)
    initial = rng.normal(size=(2,16,12)) + 1j*rng.normal(size=(2,16,12))
    final = initial * np.exp(1j*np.arange(16)[None,:,None])
    options = dict(dx_um=.5, dy_um=.75, coherence_groups=('same','same'),
                   carrier_channels=({'kx_rad_per_um': 1., 'ky_rad_per_um': 0.},
                                     {'kx_rad_per_um': -1., 'ky_rad_per_um': 0.}))
    expected = module.carrier_power_diagnostic(initial, final, **options)
    class Backend:
        def __getattr__(self, name):
            return getattr(np, name)
    transfers = []
    def scalar_summary(value):
        assert value.size <= 2
        transfers.append(value.size)
        return value
    monkeypatch.setattr(module, 'asnumpy', scalar_summary)
    actual = module.carrier_power_diagnostic(initial, final, xp=Backend(), bounded=True, **options)
    assert actual == expected
    assert transfers == [2, 2, 2, 2]


def test_unsupported_workflow_cannot_silently_execute_selected_policy():
    from lcprop.transport.envelopes import RequestEnvelope
    from lcprop.runners.local import LocalRunner
    from lcprop.runners.base import WorkflowOperation
    envelope = RequestEnvelope(run_id='unsupported', material_id='pr', workflow_id='pr_timedependent',
        codec_id='codec', codec_version=1, scientific_backend_requested='numpy', request_payload={}, result_policy='analysis')
    with pytest.raises(ValueError, match='only by reduced PR Static'):
        envelope.to_dict()
    calls = []
    operation = WorkflowOperation('other', 'other', lambda request: calls.append(request), lambda result: result)
    with pytest.raises(ValueError, match='does not support'):
        LocalRunner((operation,)).run_operation(operation, None, _result_policy='interactive')
    assert not calls


def test_content_revision_refreshes_pixels_and_limits(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from lcprop.gui.views.image_pane import ImagePane
    from lcprop.products.data_model import make_field
    app = QApplication.instance() or QApplication([])
    pane = ImagePane()
    monkeypatch.setattr(pane.image_view, 'draw_idle', lambda: None)
    data = np.ones((1031, 65))
    field = make_field('revision', 'Revision', data, ('x','y'), 'intensity')
    pane.image_view.set_field(field)
    cached = pane.image_view._display_data
    assert pane._limits_for_field(field) == (0., 1.)
    pane.image_view.set_field(field)
    assert pane.image_view._display_data is cached
    data[:] = 7
    field = replace(field, content_revision=1)
    pane.image_view.set_field(field)
    assert pane.image_view._display_data is not cached
    np.testing.assert_array_equal(pane.image_view._display_data, 7)
    assert pane._limits_for_field(field) == (0., 7.)
    # Different display semantics must invalidate even with identical bytes.
    assert pane._limits_for_field(replace(field, kind='field'))[0] > 0
    pane.close()


@pytest.mark.parametrize('policy', ['interactive', 'fast', 'analysis:complex_input', 'full'])
def test_policy_switch_has_workflow_valid_payload(window, tmp_path, policy):
    w = window
    w.evolution_panel.set_workflow_id('pr_static')
    w._set_product_policy(policy)
    path = tmp_path/'policy.json'
    w.save_experiment_to(path)
    w.load_experiment_from(path)
    w.evolution_panel.set_workflow_id('pr_timedependent')
    assert w.result_policy_selector.currentData() == ('full' if policy == 'full' else 'fast')
    assert w._current_execution_intent().retrieval_policy in ('fast', 'full')
    w.evolution_panel.set_workflow_id('pr_static')
    assert w.result_policy_selector.currentData() == ('full' if policy == 'full' else 'fast')
    assert w.result_policy_selector.itemText(0) == 'Interactive'


@pytest.mark.parametrize('policy', ['interactive', 'analysis:complex_input', 'analysis:complex_output'])
def test_missing_analysis_products_do_not_dispatch(monkeypatch, policy):
    from types import SimpleNamespace
    from dataclasses import dataclass
    import lcprop.pr.optional_image_analysis as module
    @dataclass
    class Base:
        result: object
        run_data: object
        kind: str = 'pr_static'
        message: str = 'Propagation succeeded'
    request = request_for(True, False)
    result = run_pr_static(request, result_policy=policy)
    base = Base(result, PR_STATIC_OPERATION.to_run_data(result))
    monkeypatch.setattr(module, 'analyze_image_amplification_result', lambda *a, **k: pytest.fail('must not dispatch'))
    augmented = module.add_optional_image_analysis(base, request, module.PRImageAnalysisSelection(0, 1))
    assert augmented.result is result
    diagnostic = augmented.run_data.diagnostics['optional_image_analysis'].values
    assert diagnostic['status'] == 'not_selected'
    assert 'complex_input and complex_output' in diagnostic['reason']


@pytest.mark.parametrize('same_resolution', [False, True])
@pytest.mark.parametrize('index', [(0,0), (512,256), (1023,511)])
def test_linked_preview_uses_physical_coordinates(index, same_resolution):
    from PySide6.QtWidgets import QApplication
    from lcprop.gui.workspace import Workspace
    from lcprop.products.data_model import make_field, FieldCollection
    app = QApplication.instance() or QApplication([])
    result = run_pr_static(request_for(True, False))
    data = PR_STATIC_OPERATION.to_run_data(result)
    # Nonuniform integer block boundaries, exactly as stored by preview products.
    xbins = np.linspace(0, 8195, 1025).astype(int)
    ybins = np.linspace(0, 4099, 513).astype(int)
    x = (xbins[:-1] + xbins[1:] - 1) / 2 - 4097
    y = (ybins[:-1] + ybins[1:] - 1) / 2 - 2049
    tx = x if same_resolution else np.linspace(x[0], x[-1], 96)
    ty = y if same_resolution else np.linspace(y[0], y[-1], 96)
    fields = FieldCollection([
        ('preview', make_field('preview', 'Preview', np.ones((len(x),len(y))), ('x','y'), 'intensity', coordinates={'x':x,'y':y})),
        ('volume', make_field('volume', 'Volume', np.ones((2,len(tx),len(ty))), ('z','x','y'), 'intensity', coordinates={'x':tx,'y':ty,'z':np.array([0.,1.])})),
    ])
    workspace = Workspace()
    workspace.set_run_data(replace(data, fields=fields))
    workspace.longitudinal_pane.select_volume('volume')
    workspace.image_pane._position_selected(*index)
    assert workspace.longitudinal_pane._ix == np.argmin(abs(tx-x[index[0]]))
    assert workspace.longitudinal_pane._iy == np.argmin(abs(ty-y[index[1]]))
    workspace.close()


@pytest.mark.parametrize('fails', [False, True])
def test_selected_complex_endpoints_allow_analysis(monkeypatch, fails):
    from types import SimpleNamespace
    from dataclasses import dataclass
    from lcprop.products.data_model import DiagnosticData, DiagnosticCollection
    import lcprop.pr.optional_image_analysis as module
    @dataclass
    class Base:
        result: object
        run_data: object
        kind: str = 'pr_static'
        message: str = 'Propagation succeeded'
    request = request_for(True, False)
    result = run_pr_static(request, result_policy='analysis:complex_input,complex_output')
    data = PR_STATIC_OPERATION.to_run_data(result)
    diagnostics = DiagnosticCollection(list(data.diagnostics.items()))
    diagnostics.add('carrier_power', DiagnosticData('carrier_power', 'Carrier', {'status':'ok'}))
    base = Base(result, replace(data, diagnostics=diagnostics))
    # Applicability is independent of availability; isolate the dispatch boundary.
    monkeypatch.setattr(module, 'PRImageAmplificationExperimentRequest', lambda **kw: SimpleNamespace(validate=lambda: None))
    monkeypatch.setattr(module, 'image_amplification_base_capabilities', lambda: [SimpleNamespace(workflow_id='pr_static', validation_status='validated')])
    calls = []
    def analyze(*args, **kwargs):
        calls.append(True)
        if fails:
            raise RuntimeError('genuine analyzer failure')
        return SimpleNamespace(result=SimpleNamespace(analysis_status='completed', analysis_message='ok'), run_data=base.run_data)
    monkeypatch.setattr(module, 'analyze_image_amplification_result', analyze)
    augmented = module.add_optional_image_analysis(base, request, module.PRImageAnalysisSelection(0, 1))
    assert calls == [True]
    assert augmented.result is result
    assert augmented.run_data.diagnostics['optional_image_analysis'].values['status'] == ('failed' if fails else 'completed')


def test_optional_analysis_pre_run_product_advice(window, monkeypatch):
    window.evolution_panel.set_workflow_id('pr_static')
    monkeypatch.setattr(window.input_panel, 'requests_optional_image_analysis', lambda: True)
    window._set_product_policy('interactive')
    assert 'requires complex_input and complex_output' in window.describe_request(window.build_request())
    assert window.result_policy_selector.currentData() == 'interactive'


def test_far_field_click_does_not_link_angle_to_spatial_volume():
    from PySide6.QtWidgets import QApplication
    from lcprop.gui.workspace import Workspace
    app = QApplication.instance() or QApplication([])
    data = PR_STATIC_OPERATION.to_run_data(run_pr_static(request_for(True, False), result_policy='interactive'))
    workspace = Workspace()
    workspace.set_run_data(data)
    pane = workspace.image_pane
    pane.field_selector.setCurrentIndex(pane.field_selector.findData('far_field_intensity'))
    before = workspace.longitudinal_pane.guide_coordinates()
    pane._position_selected(0, 0)
    assert workspace.longitudinal_pane.guide_coordinates() == before
    workspace.close()

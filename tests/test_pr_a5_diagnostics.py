"""Exact selected fields only; recovered A5 definitions, not propagation replay."""
from dataclasses import replace
from pathlib import Path
import hashlib
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lcprop.pr.a5_diagnostics import (
    EXACT_KEY, FROZEN_ELLIPSE, IDENTITIES, exact_far_field_diagnostics,
    add_static_a5_diagnostics, _primary, _edge_occupancy,
)
from lcprop.pr.far_field import far_field_product
from lcprop.pr.far_field_mask import ellipse_exclusion, mask_diagnostic
from lcprop.products.data_model import FieldCollection, DiagnosticCollection
from tests.test_pr_far_field_mask import summary
from lcprop.core.beams import BeamChannel


def exact(shape=(67, 48), zero=False):
    grid = {'dx_um': 1., 'dy_um': 2.}
    launch = dict(summary(BeamChannel(w1_um=20, w2_um=40, psi_rad=.4)),
                  wavelengths_um=[.633], refractive_index=2.4)
    sx = np.fft.fftshift(np.fft.fftfreq(shape[0], d=1.))*.633/2.4
    sy = np.fft.fftshift(np.fft.fftfreq(shape[1], d=2.))*.633/2.4
    data = np.zeros(shape) if zero else np.random.default_rng(8).uniform(size=shape)
    field = replace(far_field_product(data, sx, sy), key=EXACT_KEY,
                    quantity='direction_cosine_power_density')
    return field, grid, launch


def test_exact_current_matches_authoritative_mask_and_raw_unchanged():
    field, grid, launch = exact()
    before = field.data.copy()
    old_mask, old = mask_diagnostic(field, launch)
    current, historical = exact_far_field_diagnostics(field, grid, launch)
    assert current['mask_mode'] == 'current_launch'
    assert historical['mask_mode'] == 'frozen_historical_A5'
    assert current['carriers'] != historical['carriers']
    assert historical['carriers'] == [FROZEN_ELLIPSE]
    for new, prior in [('surviving_norm', 'total_field_norm'), ('off_carrier_norm', 'off_carrier_field_norm'),
                       ('off_carrier_fraction', 'off_carrier_fraction'), ('excluded_pixel_count', 'excluded_pixel_count')]:
        assert current[new] == pytest.approx(old[prior], rel=2e-14)
    np.testing.assert_array_equal(field.data, before)
    assert historical['rectangle']['status'] == 'unavailable'
    assert historical['provenance']['comparison_contract_sha256'] == IDENTITIES['comparison_contract_sha256']
    json.dumps((current, historical), allow_nan=False)


def test_union_cross_terms_and_inclusive_boundary():
    # Diagonal exact covariance: radius-four boundary is exactly representable.
    ellipses = [{'center_s': [0., 0.], 'covariance_s': [[1., 0.], [0., 1.]]},
                {'center_s': [4., 0.], 'covariance_s': [[1., .5], [.5, 1.]]}]
    sx, sy = np.arange(-6., 8.), np.arange(-5., 6.)
    x, y = sx[:, None], sy[None, :]
    expected = (x*x+y*y <= 16) | (((x-4)**2-(x-4)*y+y*y)/.75 <= 16)
    np.testing.assert_array_equal(ellipse_exclusion(sx, sy, ellipses), expected)
    result = _primary(np.ones(expected.shape), sx, sy, ellipses)
    assert result['excluded_pixel_count'] == expected.sum()
    assert result['off_carrier_fraction'] == pytest.approx((~expected).mean())


@pytest.mark.parametrize('damage', ['preview', 'quantity', 'axes', 'negative', 'nan', 'grid', 'launch'])
def test_exact_contract_fails_closed(damage):
    field, grid, launch = exact()
    if damage == 'preview':
        field = replace(field, key='far_field_intensity')
    elif damage == 'quantity':
        field = replace(field, quantity='display')
    elif damage == 'axes':
        field = replace(field, coordinates={**field.coordinates, 's_x': field.coordinates['s_x']+.001})
    elif damage in ('negative', 'nan'):
        field.data[0, 0] = -1 if damage == 'negative' else np.nan
    elif damage == 'grid':
        grid['dx_um'] = 2
    else:
        launch.pop('wavelengths_um')
    with pytest.raises((ValueError, KeyError)):
        exact_far_field_diagnostics(field, grid, launch)


def test_unavailable_preview_only_and_zero_denominator():
    field, grid, launch = exact(zero=True)
    for value in exact_far_field_diagnostics(field, grid, launch):
        assert value['off_carrier_fraction'] is None
        assert value['fraction_unavailable_reason']
    fields = FieldCollection([('far_field_intensity', replace(field, key='far_field_intensity'))])
    diagnostics = DiagnosticCollection()
    add_static_a5_diagnostics(fields, diagnostics, SimpleNamespace(grid_summary=grid, launch_summary=launch))
    assert len(fields) == 1
    for value in diagnostics.values():
        assert value.values['status'] == 'unavailable'
        assert 'Analysis → Far Field Intensity' in value.values['reason']


def test_no_full_plane_diagnostic_temporaries(monkeypatch):
    field, grid, launch = exact(shape=(91, 73))
    shape = field.data.shape
    # Ufunc and allocation guards see actual intermediates, not only a mocked reducer.
    for name in ('isfinite', 'abs', 'hypot', 'arctan2', 'zeros', 'ones', 'empty', 'full'):
        original = getattr(np, name)
        def guard(*args, _fn=original, **kwargs):
            result = _fn(*args, **kwargs)
            assert np.shape(result) != shape, (_fn, np.shape(result))
            return result
        monkeypatch.setattr(np, name, guard)
    current, historical = exact_far_field_diagnostics(field, grid, launch)
    assert current['status'] == historical['status'] == 'available'


# Exact Research blobs recovered from commit e58cfe263c91f850b60b6b0acef2a062b232cf3f.
# Override permits another persistent copy; pinned hashes still apply.
REFERENCE = Path(os.environ.get('LCPROP_A5_DIAGNOSTIC_FIXTURES',
    str(Path(__file__).resolve().parents[1]/'results/pr_frozen_a5_diagnostic_fixtures')))
DEFINITIONS = REFERENCE/'L50_4394756_l2_readiness/definitions'


def load_reference(name, digest):
    path = DEFINITIONS/name
    if not path.exists():
        pytest.fail(f'Missing checksum-bound Research fixture: {path}; see tests/fixtures/frozen_a5.txt')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location('a5_reference_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promoted_compact_and_edge_match_frozen_functions():
    from lcprop.pr.a5_artifacts import compact_artifacts
    original = load_reference('a5_package.py', IDENTITIES['artifact_source_sha256'])
    edge = load_reference('a5_spectral_edge.py', IDENTITIES['edge_source_sha256'])
    field, _, _ = exact(shape=(71, 59))
    sx, sy = field.coordinates['s_x'], field.coordinates['s_y']
    mask = ellipse_exclusion(sx, sy, [FROZEN_ELLIPSE])
    masked = field.data.copy(); masked[mask] = np.nan
    for value, excluded in ((field.data, None), (masked, lambda i: mask[i])):
        expected = original.compact_artifacts(value, sx, sy, FROZEN_ELLIPSE['center_s'][0])
        actual = compact_artifacts(field.data, sx, sy, FROZEN_ELLIPSE['center_s'][0], excluded_row=excluded)
        assert actual == expected
    expected = edge.spectral_edge_occupancy(field.data)
    actual = _edge_occupancy(field.data)['bands']
    for band in expected:
        for axis in expected[band]:
            assert actual[band][axis]['fraction'] == pytest.approx(expected[band][axis]['fraction'], rel=2e-14)
            assert actual[band][axis]['selected_bins'] == expected[band][axis]['selected_bins']


@pytest.mark.parametrize('directory,case', [('L50_4394756_l2_readiness', 'L50'), ('L2_4414263_final_reproduction', 'L2')])
def test_preserved_historical_maps(directory, case):
    root = REFERENCE/directory/case/'evidence'
    if not root.exists():
        pytest.fail(f'Missing checksum-bound Research maps: {root}; see tests/fixtures/frozen_a5.txt')
    contract = REFERENCE/directory/'comparison_contract.json'
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == IDENTITIES['comparison_contract_sha256']
    expected_hashes = {
        's_x.npy': '0f9a2e4b5fdd5b5927ab141819ab29a8e14abdc3809cad95d2c2f832b648d8a9',
        's_y.npy': '373664b8363ae361575198a3aa1d12689202fe7803e3d2d803fcebee90ae43af',
        'metrics.json': {'L50': '31b071fe4380b85e3ad6981b224ab200be69541a28807dcdbe5bf0bc527700a8',
                         'L2': '90192f2cf6e1edd559b8c6de85342d73a92441ef14b1f715c9d89e1b9bcbc756'}[case],
        'static_final_spectrum.npy': {'L50': '5dd64cfaf4dcf683598c3b92f29f1107553abc1fbb94feabd55ba3da187d5f8e',
                                      'L2': 'e212368dcb35befe94110afa387062943d0d694e67f5e766a34dcccf5d268f5e'}[case],
    }
    for name, digest in expected_hashes.items():
        actual = hashlib.sha256()
        with (root/name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                actual.update(block)
        assert actual.hexdigest() == digest
    # Float32 saved maps are regression evidence, NOT exact Analysis inputs in production.
    data = np.load(root/'static_final_spectrum.npy', mmap_mode='r')
    sx, sy = (np.load(root/(key+'.npy'), mmap_mode='r') for key in ('s_x', 's_y'))
    assert data.shape == (8192, 4096) and data.dtype == np.float32
    assert sx.shape == (8192,) and sy.shape == (4096,)
    definitions = json.loads(contract.read_text())['definitions']
    assert definitions['coordinate_convention'] == 'absolute in-medium s_x=kx/k_internal, s_y=ky/k_internal'
    assert definitions['spectral_standard_deviations'] == 4.0
    field = replace(far_field_product(data, sx, sy), key=EXACT_KEY, quantity='direction_cosine_power_density')
    _, historical = exact_far_field_diagnostics(field, {'dx_um':1000/8192, 'dy_um':1000/4096},
                                              {'wavelengths_um':[.633], 'refractive_index':2.4})
    expected = json.loads((root/'metrics.json').read_text())['static_final']
    for new, old in [('surviving_norm','total_field_norm'), ('off_carrier_norm','off_carrier_field_norm'),
                     ('off_carrier_fraction','off_carrier_fraction')]:
        assert historical[new] == pytest.approx(expected[old], rel=2e-7)
    rect = historical['rectangle']
    assert rect['historical_rectangle_fraction'] == pytest.approx(expected['L_rectangle_off_fraction'], rel=2e-7)
    assert rect['support_clipped_rectangle_fraction'] == pytest.approx(expected['historical_executable_rectangle_fraction_after_diagnostic_free_clip'], rel=2e-7)
    for new, old in [('canonical_artifacts','artifact_diagnostics'), ('masked_artifacts','masked_artifact_diagnostics')]:
        for metric in ('high_frequency_rho_gt_0_25_fraction','radial_roughness_L1_second_difference_over_L1'):
            assert historical[new][metric] == pytest.approx(expected[old][metric], rel=2e-6)
    assert historical['excluded_pixel_count'] == 32
    assert historical['edge_occupancy']['interpretation_blocker'] is False


def test_selected_static_codec_and_gui_tables_without_complex_endpoints(app):
    from tests.test_pr_static_products import _synthetic_static_result
    from lcprop.pr.selected_products import construct_selected_products
    from lcprop.pr.static_transport_codec import encode_pr_static_transport_result, decode_pr_static_transport_result
    from lcprop.pr.products import pr_static_result_to_run_data
    from lcprop.gui.workspace import Workspace
    from tests.test_pr_live_results import close_widget
    result = _synthetic_static_result()
    launch = dict(result.launch_summary, **summary(BeamChannel()), wavelengths_um=[.633], refractive_index=2.4)
    result = replace(result, launch_summary=launch)
    products = construct_selected_products(result.A_initial, result.A_final, policy='analysis:far_field_intensity',
                                          grid=result.grid_summary, launch=launch, xp=np, asnumpy=np.asarray)
    fast = encode_pr_static_transport_result(result, "fast")
    result = decode_pr_static_transport_result(fast.payload.metadata, fast.payload.arrays)
    products['carrier_power'] = {}
    products['observation'] = {'completed_slices': 2, 'z_reached_um': 10., 'longitudinal_products': 'available'}
    result = replace(result, selected_products=products, A_initial=None, A_final=None,
                     retention_summary={'policy':'analysis', 'omitted_fields': [*result.retention_summary['omitted_fields'], 'A_initial', 'A_final']})
    encoded = encode_pr_static_transport_result(result, 'analysis:far_field_intensity')
    restored = decode_pr_static_transport_result(encoded.payload.metadata, encoded.payload.arrays)
    data = pr_static_result_to_run_data(restored)
    assert restored.A_initial is restored.A_final is None
    assert data.diagnostics['a5_current_launch'].values['status'] == 'available'
    ws = Workspace()
    try:
        ws.set_run_data(data)
        assert 'Historical A5 comparison mask' in ws.diagnostics_view.toPlainText()
        assert ws.table_pane.selector.findData('a5_frozen_historical_A5') >= 0
        assert EXACT_KEY in data.fields
    finally:
        close_widget(ws)


from tests.test_pr_live_results import app, collect_test_objects_on_gui_thread

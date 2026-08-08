import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_ALGORITHM,
    PRCanonicalScatteringSpec,
    canonical_scattering_phase_increment,
    canonical_scattering_provenance,
    canonical_slab_seed,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static_streaming import (
    PRStreamingStaticOptions,
    PRStreamingStaticRequest,
    PR_STREAMING_PRODUCTION,
    _volume_noise_phase,
    run_pr_static_streaming,
    volume_noise_seed_for_slice,
)
from lcprop.pr.static_workflow import PRStaticWorkflowOptions


def _spec(*, seed=12345):
    return PRCanonicalScatteringSpec(
        epsilon=0.02,
        transverse_correlation_um=1.0,
        realization_seed=seed,
        canonical_dz_um=2.0,
    )


def _phase(spec, *, z_start_um, dz_um, shape=(32, 16), apertures=(32.0, 16.0)):
    return canonical_scattering_phase_increment(
        spec,
        z_start_um=z_start_um,
        dz_um=dz_um,
        z_length_um=100.0,
        Nx=shape[0],
        Ny=shape[1],
        x_aperture_um=apertures[0],
        y_aperture_um=apertures[1],
        real_dtype=np.float64,
        xp=np,
    )


def test_canonical_scattering_is_deterministic_across_independent_specs():
    first = _phase(_spec(), z_start_um=20.0, dz_um=10.0)
    second = _phase(_spec(), z_start_um=20.0, dz_um=10.0)

    assert np.array_equal(first, second)


def test_canonical_scattering_is_access_order_independent():
    spec = _spec()
    forward = {
        z: _phase(spec, z_start_um=z, dz_um=2.0) for z in (0.0, 2.0, 20.0, 68.0)
    }
    reverse = {
        z: _phase(spec, z_start_um=z, dz_um=2.0)
        for z in (68.0, 20.0, 2.0, 0.0)
    }

    for z in forward:
        assert np.array_equal(forward[z], reverse[z])


def test_coarse_interval_is_sum_of_same_fine_canonical_increments():
    spec = _spec()
    coarse = _phase(spec, z_start_um=0.0, dz_um=50.0)
    fine = np.sum(
        [_phase(spec, z_start_um=float(z), dz_um=2.0) for z in range(0, 50, 2)],
        axis=0,
    )

    relative = np.linalg.norm(coarse - fine) / np.linalg.norm(fine)
    assert relative < 8e-16
    assert np.max(np.abs(coarse - fine)) < 2e-16


def test_canonical_scattering_statistics_are_white_in_z_and_scale_by_interval():
    one_slab = []
    four_slabs = []
    next_slab = []
    for seed in range(160):
        spec = _spec(seed=seed)
        one_slab.append(_phase(spec, z_start_um=0.0, dz_um=2.0))
        next_slab.append(_phase(spec, z_start_um=2.0, dz_um=2.0))
        four_slabs.append(_phase(spec, z_start_um=0.0, dz_um=8.0))
    one = np.asarray(one_slab)
    next_one = np.asarray(next_slab)
    four = np.asarray(four_slabs)

    standard_deviation = float(np.std(one))
    assert abs(float(np.mean(one))) < 0.02 * standard_deviation
    assert float(np.var(four)) / float(np.var(one)) == pytest.approx(4.0, rel=0.08)
    assert abs(float(np.corrcoef(one.ravel(), next_one.ravel())[0, 1])) < 0.02

    spectrum = np.mean(np.abs(np.fft.fft2(one, axes=(-2, -1))) ** 2, axis=0)
    fx = np.fft.fftfreq(one.shape[-2])[:, None]
    fy = np.fft.fftfreq(one.shape[-1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    low = float(np.mean(spectrum[(radius > 0.02) & (radius < 0.15)]))
    high = float(np.mean(spectrum[radius > 0.45]))
    assert high < 0.08 * low


def test_manifest_provenance_is_partition_independent_and_complete():
    spec = _spec()
    common = dict(
        z_length_um=100.0,
        Nx=32,
        Ny=16,
        x_aperture_um=32.0,
        y_aperture_um=16.0,
        real_dtype=np.float64,
        xp=np,
    )
    first = canonical_scattering_provenance(spec, **common)
    second = canonical_scattering_provenance(spec, **common)

    assert first == second
    assert first["algorithm_version"] == PR_CANONICAL_SCATTERING_ALGORITHM
    assert first["canonical_slab_count"] == 50
    assert first["mode"] == "partition_independent_canonical_phase_slabs"
    assert len(first["canonical_seed_sha256_le_u32"]) == 64
    assert len(first["configuration_sha256"]) == 64
    assert first["first_canonical_seed"] == canonical_slab_seed(spec, 0)
    assert first["last_canonical_seed"] == canonical_slab_seed(spec, 49)


def test_canonical_intervals_must_align_and_modes_are_mutually_exclusive():
    spec = _spec()
    with pytest.raises(ValueError, match="align"):
        _phase(spec, z_start_um=1.0, dz_um=2.0)
    with pytest.raises(ValueError, match="align"):
        _phase(spec, z_start_um=0.0, dz_um=3.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        PRStreamingStaticOptions(
            volume_noise_epsilon=0.02,
            volume_noise_seed=1,
            partition_independent_scattering=spec,
        ).validate()


def test_explicit_legacy_mode_retains_trusted_per_slice_formula_exactly():
    grid_spec = GridSpec(
        Nx=12,
        Ny=6,
        x_aperture_um=24.0,
        y_aperture_um=18.0,
        z_length_um=15.0,
        dz_um=5.0,
    )
    grid = make_grid(grid_spec, real_dtype=np.float64)
    options = PRStreamingStaticOptions(
        volume_noise_epsilon=0.02,
        volume_noise_correlation_um=0.4,
        volume_noise_seed=12345,
    )
    request = SimpleNamespace(solver=options, grid=grid_spec)
    actual = _volume_noise_phase(1, request=request, grid=grid, xp=np)

    seed = volume_noise_seed_for_slice(options, z_index=1, longitudinal_slices=3)
    scale = math.sqrt(0.02 / 3 * 4.0 * math.pi)
    raw = np.random.RandomState(seed).normal(0.0, scale, size=(12, 6))
    sigma_x = 0.4 * 12 / 24.0
    sigma_y = 0.4 * 6 / 18.0
    expected = gaussian_filter(raw, sigma=(sigma_x, sigma_y)) * math.sqrt(
        sigma_x * sigma_y
    )

    assert np.array_equal(actual, expected)


def _streaming_request(*, dz_um):
    grid = GridSpec(
        Nx=20,
        Ny=10,
        x_aperture_um=40.0,
        y_aperture_um=20.0,
        z_length_um=20.0,
        dz_um=dz_um,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=8.0,
                waist_y_um=6.0,
                coherence_group="readiness",
            ),
        ),
        coherence="coherent",
    )
    return PRStreamingStaticRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=0.1,
            gain_length_product=0.05,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRStreamingStaticOptions(
            coupled=PRStaticWorkflowOptions(
                max_coupled_passes=12,
                residual_rms_tolerance=1e-8,
                residual_max_tolerance=1e-7,
            ),
            mode=PR_STREAMING_PRODUCTION,
            deterministic_replay=True,
            partition_independent_scattering=PRCanonicalScatteringSpec(
                epsilon=0.01,
                transverse_correlation_um=0.8,
                realization_seed=9876,
                canonical_dz_um=2.0,
            ),
        ),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def test_small_coarse_and_fine_runs_share_scattering_identity_and_replay():
    coarse = run_pr_static_streaming(_streaming_request(dz_um=10.0))
    fine = run_pr_static_streaming(_streaming_request(dz_um=2.0))

    coarse_noise = coarse.memory_policy["volume_noise"]
    fine_noise = fine.memory_policy["volume_noise"]
    assert coarse_noise["configuration_sha256"] == fine_noise["configuration_sha256"]
    assert (
        coarse_noise["canonical_seed_sha256_le_u32"]
        == fine_noise["canonical_seed_sha256_le_u32"]
    )
    assert coarse.replay_diagnostics["consistent"]
    assert fine.replay_diagnostics["consistent"]
    assert np.all(np.isfinite(coarse.A_final))
    assert np.all(np.isfinite(fine.A_final))


def test_canonical_scattering_is_backend_native_on_cupy_when_available():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.arange(1)
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")

    spec = _spec()
    common = dict(
        spec=spec,
        z_length_um=20.0,
        Nx=12,
        Ny=6,
        x_aperture_um=24.0,
        y_aperture_um=12.0,
        real_dtype=cp.float64,
        xp=cp,
    )
    coarse = canonical_scattering_phase_increment(
        z_start_um=0.0, dz_um=10.0, **common
    )
    fine = cp.zeros_like(coarse)
    for z_start in range(0, 10, 2):
        fine += canonical_scattering_phase_increment(
            z_start_um=float(z_start), dz_um=2.0, **common
        )

    assert isinstance(coarse, cp.ndarray)
    assert bool(cp.allclose(coarse, fine, rtol=2e-14, atol=2e-15).item())

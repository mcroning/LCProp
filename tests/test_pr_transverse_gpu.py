from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import (
    PRTransverseRunRequest,
    PRTransverseSolverOptions,
    run_pr_transverse_timedependent,
)
from lcprop.pr.transverse.transport import (
    imex_euler_step,
    potential_rhs,
    state_from_potential,
)
import lcprop.pr.transverse.workflow as transverse_workflow


def _cupy_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
        cp.cuda.Device().compute_capability
    except Exception as exc:
        pytest.skip(f"CuPy device unavailable: {exc}")
    return cp


def _request(*, precision: str, scattering: bool = False):
    return PRTransverseRunRequest(
        grid=GridSpec(
            Nx=18,
            Ny=16,
            x_aperture_um=54.0,
            y_aperture_um=48.0,
            dz_um=4.0,
            z_length_um=8.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    waist_x_um=12.0,
                    waist_y_um=9.0,
                    tilt_x_rad_per_um=0.08,
                    tilt_y_rad_per_um=-0.04,
                    coherence_group="transverse-gpu",
                ),
            )
        ),
        material=PRMaterialSpec(
            dark_intensity=0.2,
            uniform_background_intensity=0.1,
            gain_length_product=0.08,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseSolverOptions(
            Nt=2,
            dt_normalized=1.0e-4,
            optical_substeps=1,
        ),
        backend=BackendSpec("numpy", precision, False),
        scattering=(
            PRCanonicalScatteringSpec(
                realization_seed=8912,
                canonical_dz_um=4.0,
                epsilon=1.0e-3,
                transverse_correlation_um=1.5,
                algorithm_version=PR_CANONICAL_SCATTERING_V2,
            )
            if scattering
            else None
        ),
    )


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("kind", ("uniform", "y_independent", "transverse"))
def test_cupy_transport_and_imex_match_numpy(precision, kind):
    cp = _cupy_device()
    dtype = np.float32 if precision == "float32" else np.float64
    rng = np.random.default_rng(20260818)
    shape = (3, 19, 17)
    if kind == "uniform":
        psi = np.zeros(shape, dtype=dtype)
        intensity = np.full(shape, 0.31, dtype=dtype)
    elif kind == "y_independent":
        x = np.arange(shape[-2]) * 2.0 * np.pi / shape[-2]
        profile = 2.0e-3 * np.cos(x) + 7.0e-4 * np.sin(2.0 * x)
        psi = np.broadcast_to(profile[None, :, None], shape).copy().astype(dtype)
        intensity = np.broadcast_to(
            (0.4 + 0.1 * np.cos(x))[None, :, None], shape
        ).copy().astype(dtype)
    else:
        psi = rng.normal(scale=8.0e-4, size=shape).astype(dtype)
        intensity = (0.2 + rng.random(shape)).astype(dtype)

    kwargs = dict(dx_normalized=0.37, dy_normalized=0.43)
    expected_state = state_from_potential(psi, **kwargs)
    actual_state = state_from_potential(cp.asarray(psi), xp=cp, **kwargs)
    expected_rhs = potential_rhs(psi, intensity, **kwargs)
    actual_rhs = potential_rhs(
        cp.asarray(psi), cp.asarray(intensity), xp=cp, **kwargs
    )
    expected_step = imex_euler_step(
        psi, intensity, dt_normalized=2.0e-4, **kwargs
    )
    actual_step = imex_euler_step(
        cp.asarray(psi),
        cp.asarray(intensity),
        dt_normalized=2.0e-4,
        xp=cp,
        **kwargs,
    )

    tolerance = 8.0e-6 if precision == "float32" else 3.0e-12
    for expected, actual in (
        (expected_state.psi, actual_state.psi),
        (expected_state.carrier_density, actual_state.carrier_density),
        (expected_state.E_x, actual_state.E_x),
        (expected_state.E_y, actual_state.E_y),
        (expected_rhs, actual_rhs),
        (expected_step, actual_step),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual), expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("precision", ("float32", "float64"))
@pytest.mark.parametrize("scattering", (False, True))
def test_cupy_workflow_matches_numpy_without_fallback(precision, scattering):
    _cupy_device()
    numpy_request = _request(precision=precision, scattering=scattering)
    cupy_request = replace(
        numpy_request,
        backend=BackendSpec("cupy", precision, False),
    )
    expected = run_pr_transverse_timedependent(numpy_request)
    actual = run_pr_transverse_timedependent(cupy_request)

    assert actual.backend_summary["backend"] == "cupy"
    assert actual.backend_summary["is_gpu"] is True
    assert actual.completed_steps == expected.completed_steps
    tolerance = 2.0e-5 if precision == "float32" else 8.0e-12
    for expected_array, actual_array in (
        (expected.psi_final, actual.psi_final),
        (expected.A_final, actual.A_final),
    ):
        np.testing.assert_allclose(
            actual_array, expected_array, rtol=tolerance, atol=tolerance
        )
    for key in (
        "carrier_relative_drift_max",
        "curl_rms",
        "curl_max",
        "gauss_rms",
        "gauss_max",
        "optical_power_relative_drift",
    ):
        assert actual.diagnostics[key] == pytest.approx(
            expected.diagnostics[key], rel=tolerance, abs=tolerance
        )
    assert actual.diagnostics["finite_material_state"] is True
    assert actual.diagnostics["finite_optical_state"] is True


def test_cupy_workflow_source_intensity_matches_numpy(monkeypatch):
    cp = _cupy_device()
    request = _request(precision="float64", scattering=True)
    original = transverse_workflow.imex_euler_step
    captured: list[np.ndarray] = []

    def capture(psi, intensity, **kwargs):
        captured.append(np.asarray(cp.asnumpy(intensity) if kwargs["xp"] is cp else intensity).copy())
        return original(psi, intensity, **kwargs)

    monkeypatch.setattr(transverse_workflow, "imex_euler_step", capture)
    run_pr_transverse_timedependent(request)
    numpy_sources = captured.copy()
    captured.clear()
    run_pr_transverse_timedependent(
        replace(request, backend=BackendSpec("cupy", "float64", False))
    )
    cupy_sources = captured.copy()

    assert len(numpy_sources) == len(cupy_sources) == request.solver.Nt
    for expected, actual in zip(numpy_sources, cupy_sources):
        np.testing.assert_allclose(actual, expected, rtol=8.0e-12, atol=8.0e-12)

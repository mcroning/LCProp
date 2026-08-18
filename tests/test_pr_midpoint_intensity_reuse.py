from __future__ import annotations

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, normalized_power
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
import lcprop.pr.workflow as workflow


def _case(*, xp, nz: int, optical_substeps: int, groups: tuple[str, ...]):
    real_dtype = np.float64 if xp is np else xp.float32
    complex_dtype = np.complex128 if xp is np else xp.complex64
    grid_spec = GridSpec(
        Nx=12,
        Ny=8,
        x_aperture_um=30.0,
        y_aperture_um=20.0,
        dz_um=2.0,
        z_length_um=2.0 * nz,
    )
    beams = BeamStack(
        channels=tuple(
            BeamChannel(
                name=f"beam-{index}",
                wavelength_um=0.633,
                power_mW=1.0 + index,
                waist_x_um=5.0 + index,
                waist_y_um=4.0,
                x0_um=-2.0 + 2.0 * index,
                tilt_x_rad_per_um=0.03 * (index + 1),
                coherence_group=group,
            )
            for index, group in enumerate(groups)
        ),
        coherence="incoherent",
    )
    material = PRMaterialSpec(
        dark_intensity=0.1,
        uniform_background_intensity=0.05,
        applied_field=0.2,
        gain_length_product=1.3,
        refractive_index=2.4,
        characteristic_wavenumber_per_um_override=0.5,
    )
    request = PRRunRequest(
        grid=grid_spec,
        beams=beams,
        material=material,
        solver=PRSolverOptions(
            Nt=1,
            dt_normalized=1e-3,
            optical_substeps=optical_substeps,
        ),
        backend=BackendSpec(
            backend="numpy" if xp is np else "cupy",
            precision="float64" if xp is np else "float32",
            verbose=False,
        ),
    )
    grid = make_grid(grid_spec, xp=xp, real_dtype=real_dtype)
    launch = build_launch(beams, grid, complex_dtype=complex_dtype)
    A0 = launch.A0.copy()
    peak_reference = channel_peak_intensity_reference(A0, xp=xp)
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um / optical_substeps,
        wavelength=0.633,
        n_ref=material.refractive_index,
        xp=xp,
    )
    x = xp.arange(grid.Nx, dtype=real_dtype)[None, :, None]
    z = xp.arange(grid.Nz, dtype=real_dtype)[:, None, None]
    E = 0.01 * xp.sin(2.0 * np.pi * (x + z) / grid.Nx)
    E = xp.broadcast_to(E, (grid.Nz, grid.Nx, grid.Ny)).copy()
    kwargs = {
        "request": request,
        "grid": grid,
        "kernel": kernel,
        "peak_reference": peak_reference,
        "wavelength_um": 0.633,
    }
    return A0, E, kwargs


def _legacy_optical_pass(A0, E, **kwargs):
    request = kwargs["request"]
    grid = kwargs["grid"]
    A = A0.copy()
    source = grid.xp.empty(E.shape, dtype=grid.real_dtype)
    for index in range(grid.Nz):
        A, source[index] = workflow.advance_pr_slice_with_midpoint_source(
            A,
            E[index],
            kernel=kwargs["kernel"],
            optical_substeps=request.solver.optical_substeps,
            dz_um=grid.dz_um,
            wavelength_um=kwargs["wavelength_um"],
            interaction_length_um=request.grid.z_length_um,
            gain_length_product=request.material.gain_length_product,
            peak_intensity_reference=kwargs["peak_reference"],
            background_intensity=request.material.background_intensity,
            coherence_groups=request.beams.coherence_groups,
            xp=grid.xp,
        )
    return A, source


@pytest.mark.parametrize("nz", (1, 4))
@pytest.mark.parametrize("optical_substeps", (1, 3))
@pytest.mark.parametrize(
    "groups",
    (
        ("left", "right"),
        ("shared", "shared"),
        ("shared", "shared", "other"),
    ),
    ids=("incoherent", "coherent", "multiple-groups"),
)
def test_reused_midpoint_intensity_is_exact_numpy(
    nz,
    optical_substeps,
    groups,
):
    A0, E, kwargs = _case(
        xp=np,
        nz=nz,
        optical_substeps=optical_substeps,
        groups=groups,
    )

    expected_A, expected_source = _legacy_optical_pass(A0, E, **kwargs)
    actual_A, actual_source = workflow._optical_pass(A0, E, **kwargs)

    np.testing.assert_array_equal(actual_A, expected_A)
    np.testing.assert_array_equal(actual_source, expected_source)
    assert normalized_power(actual_A, kwargs["grid"]) == normalized_power(
        expected_A,
        kwargs["grid"],
    )


@pytest.mark.parametrize("nz", (1, 5))
def test_optical_pass_evaluates_intensity_nz_plus_one_times(monkeypatch, nz):
    A0, E, kwargs = _case(
        xp=np,
        nz=nz,
        optical_substeps=1,
        groups=("shared", "shared"),
    )
    original = workflow.pr_driving_intensity
    calls = []

    def counted(*args, **call_kwargs):
        calls.append(args[0])
        return original(*args, **call_kwargs)

    monkeypatch.setattr(workflow, "pr_driving_intensity", counted)
    workflow._optical_pass(A0, E, **kwargs)

    assert len(calls) == nz + 1


def test_cancellation_between_slices_does_not_reuse_stale_intensity(monkeypatch):
    A0, E, kwargs = _case(
        xp=np,
        nz=4,
        optical_substeps=2,
        groups=("left", "right"),
    )
    token = CancellationToken()
    original = workflow.pr_driving_intensity
    call_count = 0

    def cancel_after_first_exit(*args, **call_kwargs):
        nonlocal call_count
        result = original(*args, **call_kwargs)
        call_count += 1
        if call_count == 2:
            token.cancel()
        return result

    monkeypatch.setattr(
        workflow,
        "pr_driving_intensity",
        cancel_after_first_exit,
    )

    with pytest.raises(workflow._PRCancellationRequested) as captured:
        workflow._optical_pass(
            A0,
            E,
            **kwargs,
            cancellation_token=token,
            cancellation_stage="reuse-test",
        )

    assert captured.value.stage == "reuse-test"
    assert call_count == 2


def test_reused_midpoint_intensity_is_exact_cupy():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed without a visible CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy is installed without a usable CUDA runtime")

    A0, E, kwargs = _case(
        xp=cp,
        nz=3,
        optical_substeps=2,
        groups=("shared", "shared", "other"),
    )
    expected_A, expected_source = _legacy_optical_pass(A0, E, **kwargs)
    actual_A, actual_source = workflow._optical_pass(A0, E, **kwargs)

    assert bool(cp.array_equal(actual_A, expected_A).item())
    assert bool(cp.array_equal(actual_source, expected_source).item())

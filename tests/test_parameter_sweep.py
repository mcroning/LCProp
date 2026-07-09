

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import BiasSpec, GridSpec, LCMaterial
from lcprop.core.requests import OutputOptions, StaticRunRequest, StaticSolverOptions
from lcprop.workflows.soliton import SolitonRequest
from lcprop.workflows.sweep import ParameterSweepRequest, run_parameter_sweep


def make_base_static_request() -> StaticRunRequest:
    return StaticRunRequest(
        grid=GridSpec(Nx=24, Ny=24, dz_um=20.0, x_aperture_um=24.0, y_aperture_um=24.0, z_length_um=80.0),
        material=LCMaterial(),
        bias=BiasSpec(V_bias=0.9144),
        beams=BeamStack(channels=(BeamChannel(power_mW=0.05, waist_x_um=3.0, waist_y_um=3.0),)),
        solver=StaticSolverOptions(max_iterations=1),
        output=OutputOptions(),
    )


def test_parameter_sweep_soliton_power_smoke():
    base = SolitonRequest(
        base=make_base_static_request(),
        mode="00",
        max_outer=2,
        theta_steps_per_outer=2,
        field_mix=0.25,
        tol_residual_rms=1e9,
        tol_residual_max=1e9,
    )
    request = ParameterSweepRequest(
        experiment="soliton",
        parameter="power_mW",
        values=(0.05, 0.08),
        base=base,
        continuation=True,
    )

    result = run_parameter_sweep(request)

    assert result.kind == "ParameterSweepResult"
    assert result.experiment == "soliton"
    assert result.parameter == "power_mW"
    assert result.values == (0.05, 0.08)
    assert result.continuation is True
    assert result.metrics["n_points"] == 2
    assert len(result.results) == 2
    assert len(result.samples) == 2
    assert result.samples[0]["requested_power_mW"] == 0.05
    assert result.samples[1]["requested_power_mW"] == 0.08
    assert result.results[0].mode == "00"
    assert result.results[1].mode == "00"


def test_parameter_sweep_rejects_unsupported_parameter():
    base = SolitonRequest(base=make_base_static_request())
    request = ParameterSweepRequest(
        experiment="soliton",
        parameter="V_bias",  # type: ignore[arg-type]
        values=(0.9, 1.0),
        base=base,
    )

    try:
        request.validate()
    except ValueError as exc:
        assert "parameter='power_mW'" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
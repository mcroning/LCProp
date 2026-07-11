from lcprop.core.requests import TimeDependentRunRequest, TimeDependentSolverOptions
from lcprop.products.data_model import (
    from_static_result,
    from_timedependent_result,
    from_soliton_result,
    from_soliton_existence_result,
    to_run_data,
)
from lcprop.workflows import run_static, run_timedependent, run_soliton, run_soliton_existence
from lcprop.workflows.soliton import SolitonRequest
from lcprop.workflows.soliton_existence import SolitonExistenceRequest
from tests.test_all_workflows import make_base_static_request


def test_static_result_to_run_data():
    result = run_static(make_base_static_request())
    data = from_static_result(result)
    assert data.workflow == "static"
    assert data.fields["final_intensity"].data.shape == (24, 24)
    assert data.fields["theta"].data.shape == (24, 24)


def test_timedependent_result_to_run_data():
    base = make_base_static_request()
    result = run_timedependent(
        TimeDependentRunRequest(
            grid=base.grid,
            material=base.material,
            bias=base.bias,
            beams=base.beams,
            solver=TimeDependentSolverOptions(Nt=1),
            output=base.output,
        )
    )
    data = from_timedependent_result(result)
    assert data.workflow == "timedependent"
    assert data.fields["theta_stack"].data.ndim == 3


def test_soliton_result_to_run_data():
    result = run_soliton(
        SolitonRequest(
            base=make_base_static_request(),
            max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    data = from_soliton_result(result)
    assert data.workflow == "soliton"
    assert "final_intensity" in data.fields
    assert "summary" in data.diagnostics


def test_soliton_existence_result_to_run_data():
    result = run_soliton_existence(
        SolitonExistenceRequest(
            base=make_base_static_request(),
            powers_mW=(0.03, 0.05),
            soliton_max_outer=1,
            theta_steps_per_outer=1,
            tol_residual_rms=1e9,
            tol_residual_max=1e9,
        )
    )
    data = from_soliton_existence_result(result)
    assert data.workflow == "soliton_existence"
    assert "beta" in data.curves
    assert len(data.curves["beta"].x) == 2


def test_to_run_data_dispatch():
    data = to_run_data(run_static(make_base_static_request()))
    assert data.workflow == "static"


def test_soliton_transverse_refinement_options_validate():
    req = SolitonRequest(
        base=make_base_static_request(),
        refine_transverse=True,
        transverse_max_outer=12,
        transverse_theta_steps_per_outer=7,
        transverse_field_mix=0.4,
        transverse_theta_mix=0.6,
    )

    req.validate()

    assert req.refine_transverse is True
    assert req.transverse_max_outer == 12
    assert req.transverse_theta_steps_per_outer == 7

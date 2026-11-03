from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import lcprop.persistence  # initialize experiment/transport registrations
import lcprop.pr.workflow as workflow_module
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.pr.experiment_codec import (
    PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
    decode_pr_timedependent_request,
    encode_pr_timedependent_request,
)
from lcprop.pr.gui.run_cost import classify_pr_run_cost
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.persistence import (
    PR_CHECKPOINT_SCHEMA_VERSION,
    PR_CHECKPOINT_SUPPORTED_SCHEMA_VERSIONS,
    load_pr_checkpoint,
    save_pr_checkpoint,
)
from lcprop.pr.products import pr_result_to_run_data
from lcprop.pr.reduced_linearized import PRReducedLinearizedSpec
from lcprop.pr.reduced_linearized_timedependent import (
    solve_pr_reduced_linearized_timedependent,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_EULER_INTEGRATOR,
    PR_EXACT_MODAL_INTEGRATOR,
)
from lcprop.pr.timedependent_transport_codec import (
    PR_TIMEDEPENDENT_TRANSPORT_CODEC,
    decode_pr_timedependent_transport_request,
    decode_pr_timedependent_transport_result,
    encode_pr_timedependent_transport_request,
    encode_pr_timedependent_transport_result,
)
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PRTransverseMaterialResponseSpec,
)
from lcprop.pr.workflow import continue_pr_timedependent, run_pr_timedependent
from lcprop.transport.result_policy import FAST_RESULT_POLICY
from lcprop.transport.defaults import default_transport_operations


def _request(
    *,
    linearized: bool = True,
    precision: str = "float64",
    steps: int = 2,
    bias: float = 0.5,
) -> PRRunRequest:
    request = PRRunRequest(
        grid=GridSpec(
            Nx=12,
            Ny=6,
            x_aperture_um=24.0,
            y_aperture_um=12.0,
            dz_um=4.0,
            z_length_um=8.0,
        ),
        beams=BeamStack(
            channels=(BeamChannel(
                wavelength_um=0.633,
                power_mW=1.0,
                coherence_group='reduced-linearized-td',
                w1_um=8.0,
                w2_um=7.0,
            ),)
        ),
        material=PRMaterialSpec(
            dark_intensity=0.25,
            uniform_background_intensity=0.15,
            applied_field=bias,
            gain_length_product=0.01,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRSolverOptions(
            Nt=steps,
            dt_normalized=0.2 if linearized else 0.001,
            optical_substeps=1,
            integrator=(
                PR_EXACT_MODAL_INTEGRATOR if linearized else PR_EULER_INTEGRATOR
            ),
        ),
        backend=BackendSpec(backend="numpy", precision=precision, verbose=False),
    )
    if not linearized:
        return request
    return replace(
        request,
        material_response=PRTransverseMaterialResponseSpec(
            model=PR_MATERIAL_RESPONSE_LINEARIZED,
            reference_intensity=1.5,
        ),
    )


def _source(request: PRRunRequest, dtype=np.float64):
    nx, ny = request.grid.Nx, request.grid.Ny
    nz = round(request.grid.z_length_um / request.grid.dz_um)
    x = np.arange(nx)[:, None] * 2.0 * np.pi / nx
    y = np.arange(ny)[None, :] * 2.0 * np.pi / ny
    plane = 1.5 + 0.08 * np.cos(2.0 * x) + 0.03 * np.sin(3.0 * x + y)
    return np.stack([plane + 0.01 * index for index in range(nz)]).astype(dtype)


def _fixed_source(monkeypatch, source, calls=None):
    def optical(A0, E, **kwargs):
        if calls is not None:
            calls.append(np.asarray(E).copy())
        result = (
            A0.copy(),
            kwargs["grid"].xp.asarray(source, dtype=kwargs["grid"].real_dtype),
        )
        return result

    monkeypatch.setattr(workflow_module, "_optical_pass", optical)


def _assert_result_physics_equal(actual, expected):
    for name in (
        "A_initial",
        "A_final",
        "E_initial",
        "E_final",
        "source_intensity_stack",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.completed_steps == expected.completed_steps
    assert actual.time_normalized == expected.time_normalized


def test_old_positional_request_signature_defaults_to_nonlinear_without_shifting():
    expected = _request(linearized=False)
    positional = PRRunRequest(
        expected.grid,
        expected.beams,
        expected.material,
        expected.solver,
        expected.backend,
        expected.launch_elements,
        expected.initial_A,
        expected.initial_E,
        expected.scattering,
        expected.optical_boundary,
    )
    for name in (
        "grid", "beams", "material", "solver", "backend", "initial_A",
        "initial_E", "scattering", "launch_elements", "optical_boundary",
    ):
        assert getattr(positional, name) is getattr(expected, name)
    assert positional.material_response.model == PR_MATERIAL_RESPONSE_NONLINEAR














def test_nonlinear_default_produces_the_preexisting_result_exactly():
    request = _request(linearized=False, steps=1)
    explicit = replace(request, material_response=PRTransverseMaterialResponseSpec())
    implicit_result = run_pr_timedependent(request)
    explicit_result = run_pr_timedependent(explicit)
    _assert_result_physics_equal(implicit_result, explicit_result)







@pytest.mark.parametrize("model", ["linearized", "field_linear_local_intensity"])
def test_reduced_td_retired_or_unsupported_response_fails_before_execution(model):
    request = replace(_request(linearized=False), material_response=PRTransverseMaterialResponseSpec(
        model=model, reference_intensity=1.0 if model=="linearized" else None))
    for action in (run_pr_timedependent, encode_pr_timedependent_request,
                   encode_pr_timedependent_transport_request):
        with pytest.raises(ValueError, match="Reduced TD supports nonlinear hopping only"):
            action(request)
    # Decoders fail too; no automatic reinterpretation of saved experiments.
    payload = encode_pr_timedependent_request(_request(linearized=False))
    payload["material_response"] = {"model": model, "reference_intensity": 1.0 if model=="linearized" else None}
    with pytest.raises(ValueError, match="Reduced TD supports nonlinear hopping only"):
        decode_pr_timedependent_request(payload)

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.lc import LC_MATERIAL_ID
from lcprop.lc.requests import (
    OutputOptions,
    RuntimeOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.lc.specs import BiasSpec, LCMaterial
from lcprop.persistence import (
    EXPERIMENT_FORMAT,
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentFormatError,
    ExperimentMaterialError,
    ExperimentPayloadError,
    ExperimentPresentationError,
    ExperimentRuntimeStateError,
    ExperimentSchemaError,
    ExperimentWorkflowError,
    load_experiment,
    save_experiment,
)
from lcprop.persistence.experiments import (
    ExperimentCodecRegistry,
    ExperimentRequestCodec,
    read_experiment_file,
    write_experiment_file,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_MATERIAL_ID,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)


def _angle_q() -> tuple[float, float]:
    tx = math.tan(0.031)
    ty = math.tan(-0.017)
    norm = math.sqrt(1.0 + tx * tx + ty * ty)
    k_launch = 2.0 * math.pi / 0.633
    return k_launch * tx / norm, k_launch * ty / norm


def _launchplane_stack():
    model = pytest.importorskip("launchplane.model")
    return model.BeamStackDefinition(
        beams=(
            model.BeamDefinition.from_launch_angles(
                name="angle-originated",
                wavelength_um=0.633,
                power_mW=1.25,
                x_um=-3.0,
                y_um=4.0,
                waist_x_um=7.0,
                waist_y_um=8.0,
                angle_x_rad=0.031,
                angle_y_rad=-0.017,
                launch_medium_index=1.0,
                phase_rad=0.2,
                coherence_group="laser-a",
            ),
            model.BeamDefinition(
                name="legacy-phase-slope",
                wavelength_um=0.532,
                power_mW=0.75,
                x_um=6.0,
                y_um=-2.0,
                waist_x_um=9.0,
                waist_y_um=10.0,
                tilt_x_rad_per_um=0.712345678901234,
                tilt_y_rad_per_um=-0.412345678901234,
                launch_medium_index=None,
                launch_input_mode="transverse_wavevector",
                phase_rad=-0.3,
                coherence_group="laser-b",
            ),
            model.BeamDefinition(
                name="disabled-editor-beam",
                tilt_x_rad_per_um=1.1,
                launch_medium_index=None,
                launch_input_mode="transverse_wavevector",
                coherence_group="unused",
                enabled=False,
            ),
        )
    )


def _beams():
    qx, qy = _angle_q()
    return BeamStack(
        channels=(
            BeamChannel(
                name="angle-originated",
                wavelength_um=0.633,
                power_mW=1.25,
                x0_um=-3.0,
                y0_um=4.0,
                waist_x_um=7.0,
                waist_y_um=8.0,
                tilt_x_rad_per_um=qx,
                tilt_y_rad_per_um=qy,
                phase_rad=0.2,
                coherence_group="laser-a",
            ),
            BeamChannel(
                name="legacy-phase-slope",
                wavelength_um=0.532,
                power_mW=0.75,
                x0_um=6.0,
                y0_um=-2.0,
                waist_x_um=9.0,
                waist_y_um=10.0,
                tilt_x_rad_per_um=0.712345678901234,
                tilt_y_rad_per_um=-0.412345678901234,
                phase_rad=-0.3,
                coherence_group="laser-b",
            ),
        )
    )


def _presentation() -> dict:
    serialization = pytest.importorskip("launchplane.serialization")
    return {
        "beam_editor": {
            "provider": "launchplane",
            "schema_version": 1,
            "beam_stack": serialization.beam_stack_to_dict(
                _launchplane_stack()
            ),
        }
    }


def _grid() -> GridSpec:
    return GridSpec(
        Nx=32,
        Ny=24,
        dz_um=3.25,
        x_aperture_um=80.0,
        y_aperture_um=90.0,
        z_length_um=130.0,
    )


def _lc_static_request() -> StaticRunRequest:
    return StaticRunRequest(
        grid=_grid(),
        material=LCMaterial(
            name="roundtrip-lc",
            ne=1.71,
            no=1.49,
            K=8.2e-12,
            delta_epsilon=12.5,
        ),
        bias=BiasSpec(
            V_bias=1.2,
            theta_bc=0.1,
            theta_min=0.0,
            theta_max=1.4,
            theta_center=0.7,
            b_override=2.3,
        ),
        beams=_beams(),
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            max_iterations=17,
            tolerance_rms=2e-6,
            tolerance_max=3e-5,
            static_residual_rms_tol=4e-4,
            static_residual_max_tol=5e-3,
            static_delta_theta_rms_tol=6e-6,
            static_delta_theta_max_tol=7e-5,
            static_max_relax_iterations=19,
            static_max_coupled_passes=11,
            record_iteration_history=False,
        ),
        output=OutputOptions(run_dir=None, save_slices=False, save_full=True),
        runtime=RuntimeOptions(
            precision="float32",
            optical_substeps_enabled=False,
            optical_dn_max_est=0.031,
            optical_max_phase_per_substep_rad=0.21,
            optical_max_substeps=23,
        ),
    )


def _lc_timedependent_request() -> TimeDependentRunRequest:
    return TimeDependentRunRequest(
        grid=_grid(),
        material=_lc_static_request().material,
        bias=_lc_static_request().bias,
        beams=_beams(),
        solver=TimeDependentSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            Nt=29,
            dt=0.000321,
            gamma_z=0.17,
            max_picard_iter=7,
            tolerance_update=8e-7,
        ),
        output=OutputOptions(run_dir=None, save_slices=True, save_full=True),
        runtime=RuntimeOptions(
            precision="float64",
            optical_substeps_enabled=True,
            optical_dn_max_est=0.025,
            optical_max_phase_per_substep_rad=0.19,
            optical_max_substeps=31,
        ),
    )


def _pr_material() -> PRMaterialSpec:
    return PRMaterialSpec(
        dark_intensity=0.02,
        uniform_background_intensity=0.3,
        applied_field=1.7,
        gain_length_product=-4.2,
        refractive_index=2.41,
        relative_permittivity=2400.0,
        mobile_charge_density_m3=7.1e22,
        temperature_K=295.0,
        characteristic_wavenumber_per_um_override=0.123,
    )


def _pr_static_request() -> PRStaticRunRequest:
    return PRStaticRunRequest(
        grid=_grid(),
        beams=_beams(),
        material=_pr_material(),
        solver=PRStaticWorkflowOptions(
            material_solver=PRStaticSolverOptions(
                max_iterations=27,
                residual_rms_tolerance=2e-8,
                residual_max_tolerance=3e-7,
                max_backtracks=13,
                minimum_step_scale=2.0**-15,
                armijo_fraction=2e-4,
            ),
            max_coupled_passes=12,
            residual_rms_tolerance=4e-7,
            residual_max_tolerance=5e-6,
            max_backtracks=11,
            minimum_step_scale=2.0**-14,
            armijo_fraction=3e-4,
            optical_substeps=5,
            replay_rtol=6e-6,
            replay_atol=7e-7,
            record_iteration_history=False,
        ),
        backend=BackendSpec(backend="numpy", precision="float32", verbose=False),
    )


def _pr_timedependent_request() -> PRRunRequest:
    return PRRunRequest(
        grid=_grid(),
        beams=_beams(),
        material=_pr_material(),
        solver=PRSolverOptions(
            Nt=37,
            dt_normalized=0.0125,
            optical_substeps=7,
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        ),
        backend=BackendSpec(backend="auto", precision="float64", verbose=True),
    )


@pytest.mark.parametrize(
    ("material_id", "workflow_id", "request_factory"),
    (
        (LC_MATERIAL_ID, "static", _lc_static_request),
        (LC_MATERIAL_ID, "timedependent", _lc_timedependent_request),
        (PR_MATERIAL_ID, PR_STATIC_WORKFLOW, _pr_static_request),
        (PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW, _pr_timedependent_request),
    ),
)
def test_initial_request_codecs_round_trip_exactly(
    tmp_path,
    material_id,
    workflow_id,
    request_factory,
):
    request = request_factory()
    path = tmp_path / f"{workflow_id}.lcprop.json"

    assert save_experiment(
        request,
        path,
        material_id=material_id,
        workflow_id=workflow_id,
    ) == path
    loaded = load_experiment(path, expected_material_id=material_id)

    assert loaded.request == request
    assert loaded.material_id == material_id
    assert loaded.workflow_id == workflow_id
    assert loaded.presentation_payload == {}
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == EXPERIMENT_FORMAT
    assert document["schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert document["request_payload"]["beams"]["channels"][0][
        "tilt_x_rad_per_um"
    ] == request.beams.channels[0].tilt_x_rad_per_um
    assert document["request_payload"]["beams"]["channels"][1][
        "tilt_x_rad_per_um"
    ] == 0.712345678901234
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not (tmp_path / f"{workflow_id}.lcprop.json.tmp").exists()


def test_pr_static_auto_tolerance_policy_round_trips_as_none(tmp_path):
    request = replace(
        _pr_static_request(),
        solver=PRStaticWorkflowOptions(
            material_solver=None,
            residual_rms_tolerance=None,
            residual_max_tolerance=None,
            replay_rtol=None,
            replay_atol=None,
        ),
    )
    path = tmp_path / "auto-tolerances.lcprop.json"

    save_experiment(
        request,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_STATIC_WORKFLOW,
    )
    loaded = load_experiment(path)

    assert loaded.request == request
    assert loaded.request.solver.material_solver is None
    assert loaded.request.solver.residual_rms_tolerance is None


def test_registry_accepts_future_workflow_without_shared_io_changes(tmp_path):
    @dataclass(frozen=True)
    class DummyRequest:
        value: int

    codec = ExperimentRequestCodec(
        material_id="future-material",
        workflow_id="soliton",
        request_type=DummyRequest,
        encode_request=lambda request: {
            "schema_version": 1,
            "value": request.value,
        },
        decode_request=lambda payload: DummyRequest(value=int(payload["value"])),
    )
    registry = ExperimentCodecRegistry()
    registry.register(codec)
    path = tmp_path / "future.lcprop.json"

    write_experiment_file(
        path,
        DummyRequest(41),
        material_id="future-material",
        workflow_id="soliton",
        registry=registry,
    )
    loaded = read_experiment_file(path, registry=registry)

    assert loaded.request == DummyRequest(41)
    assert registry.codecs == (codec,)
    with pytest.raises(ExperimentWorkflowError, match="future-workflow"):
        registry.codec("future-material", "future-workflow")


def test_launchplane_presentation_round_trip_preserves_modes_and_disabled_beam(
    tmp_path,
):
    request = _pr_timedependent_request()
    path = tmp_path / "presentation.lcprop.json"

    save_experiment(
        request,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
        presentation_payload=_presentation(),
    )
    loaded = load_experiment(path)

    assert loaded.presentation_payload == _presentation()
    beams = loaded.presentation_payload["beam_editor"]["beam_stack"]["beams"]
    assert beams[0]["launch_input_mode"] == "angle"
    assert beams[0]["launch_medium_index"] == 1.0
    assert beams[1]["launch_input_mode"] == "transverse_wavevector"
    assert beams[1]["launch_medium_index"] is None
    assert beams[2]["enabled"] is False


def test_launchplane_presentation_cannot_override_canonical_q(tmp_path):
    request = _pr_timedependent_request()
    presentation = _presentation()
    presentation["beam_editor"]["beam_stack"]["beams"][0][
        "tilt_x_rad_per_um"
    ] += 0.01

    with pytest.raises(ExperimentPresentationError, match="disagree"):
        save_experiment(
            request,
            tmp_path / "mismatch.lcprop.json",
            material_id=PR_MATERIAL_ID,
            workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
            presentation_payload=presentation,
        )

    path = tmp_path / "tampered-presentation.lcprop.json"
    save_experiment(
        request,
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
        presentation_payload=_presentation(),
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    document["presentation_payload"]["beam_editor"]["beam_stack"]["beams"][
        0
    ]["tilt_x_rad_per_um"] += 0.01
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ExperimentPresentationError, match="disagree"):
        load_experiment(path)


@pytest.mark.parametrize(
    ("request_obj", "material_id", "workflow_id", "field"),
    (
        (
            replace(_lc_static_request(), initial_A=np.zeros((1, 2, 2))),
            LC_MATERIAL_ID,
            "static",
            "initial_A",
        ),
        (
            replace(_lc_timedependent_request(), initial_theta=np.zeros((2, 2))),
            LC_MATERIAL_ID,
            "timedependent",
            "initial_theta",
        ),
        (
            replace(_pr_static_request(), initial_E=np.zeros((1, 2, 2))),
            PR_MATERIAL_ID,
            PR_STATIC_WORKFLOW,
            "initial_E",
        ),
        (
            replace(_pr_timedependent_request(), initial_A=np.zeros((1, 2, 2))),
            PR_MATERIAL_ID,
            PR_TIMEDEPENDENT_WORKFLOW,
            "initial_A",
        ),
    ),
)
def test_runtime_state_is_rejected_instead_of_silently_omitted(
    tmp_path,
    request_obj,
    material_id,
    workflow_id,
    field,
):
    with pytest.raises(ExperimentRuntimeStateError, match=field):
        save_experiment(
            request_obj,
            tmp_path / "runtime-state.lcprop.json",
            material_id=material_id,
            workflow_id=workflow_id,
        )


def test_machine_local_lc_output_directory_is_rejected(tmp_path):
    request = replace(
        _lc_static_request(),
        output=replace(_lc_static_request().output, run_dir=tmp_path / "run"),
    )

    with pytest.raises(ExperimentRuntimeStateError, match="run_dir"):
        save_experiment(
            request,
            tmp_path / "output-path.lcprop.json",
            material_id=LC_MATERIAL_ID,
            workflow_id="static",
        )


def test_material_mismatch_precedes_request_decoder(tmp_path):
    calls = []

    @dataclass(frozen=True)
    class Request:
        value: int

    registry = ExperimentCodecRegistry()
    registry.register(
        ExperimentRequestCodec(
            material_id="pr",
            workflow_id="workflow",
            request_type=Request,
            encode_request=lambda request: {"value": request.value},
            decode_request=lambda payload: calls.append(payload) or Request(1),
        )
    )
    path = tmp_path / "wrong-material.lcprop.json"
    write_experiment_file(
        path,
        Request(1),
        material_id="pr",
        workflow_id="workflow",
        registry=registry,
    )

    with pytest.raises(ExperimentMaterialError, match="expected material_id 'lc'"):
        read_experiment_file(
            path,
            registry=registry,
            expected_material_id="lc",
        )
    assert calls == []


def _write_document(path, document) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _valid_document(tmp_path):
    path = tmp_path / "valid.lcprop.json"
    save_experiment(
        _pr_timedependent_request(),
        path,
        material_id=PR_MATERIAL_ID,
        workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_invalid_envelopes_and_payloads_are_rejected(tmp_path):
    malformed = tmp_path / "malformed.lcprop.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExperimentFormatError, match="malformed"):
        load_experiment(malformed)

    path, document = _valid_document(tmp_path)
    document["format"] = "not-lcprop"
    _write_document(path, document)
    with pytest.raises(ExperimentFormatError, match="format marker"):
        load_experiment(path)

    path, document = _valid_document(tmp_path)
    document["schema_version"] = 999
    _write_document(path, document)
    with pytest.raises(ExperimentSchemaError, match="999"):
        load_experiment(path)

    path, document = _valid_document(tmp_path)
    document["workflow_id"] = "future_pr_workflow"
    _write_document(path, document)
    with pytest.raises(ExperimentWorkflowError, match="future_pr_workflow"):
        load_experiment(path)

    path, document = _valid_document(tmp_path)
    document["material_id"] = "unknown-material"
    _write_document(path, document)
    with pytest.raises(ExperimentMaterialError, match="unknown-material"):
        load_experiment(path)

    path, document = _valid_document(tmp_path)
    document["request_payload"]["schema_version"] = 999
    _write_document(path, document)
    with pytest.raises(ExperimentSchemaError, match="999"):
        load_experiment(path)

    path, document = _valid_document(tmp_path)
    del document["request_payload"]["grid"]
    _write_document(path, document)
    with pytest.raises(ExperimentPayloadError, match="missing.*grid"):
        load_experiment(path)

    path, document = _valid_document(tmp_path)
    document["request_payload"]["grid"]["Nx"] = "not-an-integer"
    _write_document(path, document)
    with pytest.raises(ExperimentPayloadError, match="invalid PR"):
        load_experiment(path)


def test_missing_required_envelope_field_is_rejected(tmp_path):
    path, document = _valid_document(tmp_path)
    del document["material_id"]
    _write_document(path, document)

    with pytest.raises(ExperimentFormatError, match="material_id"):
        load_experiment(path)

from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.requests import (
    OutputOptions, StaticRunRequest, StaticSolverOptions,
    TimeDependentRunRequest, TimeDependentSolverOptions,
)
from lcprop.lc.results import StaticIterationRecord, StaticSliceSummary
from lcprop.lc.specs import BiasSpec, LCMaterial
from lcprop.lc.workflows import (
    continue_static, run_timedependent, timedependent_state_from_static_result,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
)
from lcprop.pr.transverse.transport_codec import (
    decode_pr_transverse_static_transport_result,
    encode_pr_transverse_static_transport_result,
)
from lcprop.transport.artifacts import (
    sha256_file, verify_artifact_bundle, write_artifact_bundle,
)
from lcprop.transport.defaults import default_transport_operations, default_transport_registry
from lcprop.transport.envelopes import (
    RequestEnvelope, TransportCodecError, TransportSchemaError,
    TransportVerificationError,
)
from lcprop.transport.executor import execute_run_directory
from lcprop.transport.io import (
    read_failure_package,
    read_request_package,
    read_result_package,
    runner_result_from_package,
    write_request_package,
    write_result_package,
)
from lcprop.transport.status import (
    RemoteRunState, RemoteRunStatus, scheduler_state_to_remote_state,
    transition_remote_status,
)


def _beam_stack(group="remote"):
    return BeamStack(channels=(BeamChannel(
        wavelength_um=0.633, waist_x_um=8.0, waist_y_um=8.0,
        tilt_x_rad_per_um=0.05, tilt_y_rad_per_um=-0.03,
        coherence_group=group,
    ),))


def _lc_request():
    return StaticRunRequest(
        grid=GridSpec(Nx=12, Ny=10, x_aperture_um=30.0, y_aperture_um=25.0,
                      dz_um=5.0, z_length_um=10.0),
        material=LCMaterial(), bias=BiasSpec(), beams=_beam_stack(),
        solver=StaticSolverOptions(), output=OutputOptions(),
    )


def _pr_request():
    return PRTransverseStaticRunRequest(
        grid=GridSpec(Nx=12, Ny=12, x_aperture_um=30.0, y_aperture_um=30.0,
                      dz_um=5.0, z_length_um=5.0),
        beams=_beam_stack("pr-remote"),
        material=PRMaterialSpec(
            dark_intensity=0.4, uniform_background_intensity=0.1,
            gain_length_product=1e-3,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=3),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def _assert_arrays_equal(left, right, names):
    for name in names:
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


def _assert_nested_equal(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        np.testing.assert_allclose(left, right)
    elif isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_remote_status_distinguishes_scheduler_and_gui_ready_completion():
    running = RemoteRunStatus(
        run_id="run-1", execution_target="slurm", state=RemoteRunState.RUNNING
    )
    finished = transition_remote_status(running, RemoteRunState.SCIENTIFICALLY_FINISHED)
    assert not finished.gui_product_ready
    retrieving = transition_remote_status(finished, RemoteRunState.RETRIEVING)
    verifying = transition_remote_status(retrieving, RemoteRunState.VERIFYING)
    reconstructing = transition_remote_status(verifying, RemoteRunState.RECONSTRUCTING)
    completed = transition_remote_status(reconstructing, RemoteRunState.COMPLETED)
    assert completed.gui_product_ready
    assert scheduler_state_to_remote_state("COMPLETED") == RemoteRunState.SCIENTIFICALLY_FINISHED
    with pytest.raises(ValueError):
        transition_remote_status(running, RemoteRunState.COMPLETED)


@pytest.mark.parametrize("kind", ["lc", "pr"])
def test_request_result_roundtrip_and_product_regeneration(tmp_path, kind):
    registry = default_transport_registry()
    operation = LC_STATIC_OPERATION if kind == "lc" else PR_TRANSVERSE_STATIC_OPERATION
    request = _lc_request() if kind == "lc" else _pr_request()
    run_dir = tmp_path / kind
    write_request_package(
        run_dir, registry=registry, material_id=operation.material_id,
        workflow_id=operation.workflow_id, request=request, run_id=f"{kind}-run",
        execution_target="local", provenance={"git_sha": "test"},
    )
    decoded_request = read_request_package(run_dir, registry=registry).request
    assert decoded_request == request
    completed = operation.run(decoded_request)
    codec = registry.codec(operation.material_id, operation.workflow_id)
    write_result_package(
        run_dir, codec=codec, result=completed,
        request_envelope=read_request_package(run_dir, registry=registry).envelope,
    )
    decoded = read_result_package(run_dir, registry=registry).result
    if kind == "lc":
        _assert_arrays_equal(completed, decoded, ("A_final", "theta_final", "theta_bias"))
        assert decoded.checkpoint is not None
        assert decoded.checkpoint.request_fingerprint == completed.checkpoint.request_fingerprint
        _assert_arrays_equal(completed.checkpoint, decoded.checkpoint, (
            "A_next", "theta_seed", "theta_stack", "intensity_stack",
        ))
        td_request = TimeDependentRunRequest(
            grid=request.grid, material=request.material, bias=request.bias,
            beams=request.beams, solver=TimeDependentSolverOptions(Nt=1),
            output=request.output, runtime=request.runtime,
        )
        initialized = timedependent_state_from_static_result(decoded, td_request)
        np.testing.assert_array_equal(
            initialized.initial_theta, decoded.checkpoint.theta_stack
        )
        td_result = run_timedependent(initialized)
        assert td_result.status == "completed"
    else:
        _assert_arrays_equal(completed, decoded, (
            "A_final", "psi_final", "source_intensity_stack",
            "equilibrium_residual_stack", "td_rhs_residual_stack",
        ))
        assert decoded.replay_diagnostics == completed.replay_diagnostics
        assert decoded.diagnostics.keys() == completed.diagnostics.keys()
    regenerated = runner_result_from_package(
        run_dir, registry=registry, operations=default_transport_operations()
    )
    assert regenerated.result.status == completed.status
    assert regenerated.run_data is not None
    original_data = operation.to_run_data(completed)
    assert regenerated.run_data.workflow == original_data.workflow
    assert regenerated.run_data.longitudinal_enabled == original_data.longitudinal_enabled
    assert regenerated.run_data.longitudinal_message == original_data.longitudinal_message
    assert regenerated.run_data.geometry.units == original_data.geometry.units
    for axis in ("x", "y", "z"):
        _assert_nested_equal(
            getattr(regenerated.run_data.geometry, axis),
            getattr(original_data.geometry, axis),
        )
    assert tuple(regenerated.run_data.fields.keys()) == tuple(original_data.fields.keys())
    assert tuple(regenerated.run_data.curves.keys()) == tuple(original_data.curves.keys())
    assert tuple(regenerated.run_data.diagnostics.keys()) == tuple(original_data.diagnostics.keys())
    for key in original_data.fields:
        original_field = original_data.fields[key]
        transported_field = regenerated.run_data.fields[key]
        assert transported_field.axes == original_field.axes
        assert transported_field.units == original_field.units
        assert transported_field.value_unit == original_field.value_unit
        np.testing.assert_allclose(transported_field.data, original_field.data)
        _assert_nested_equal(transported_field.coordinates, original_field.coordinates)
    for key in original_data.curves:
        original_curve = original_data.curves[key]
        transported_curve = regenerated.run_data.curves[key]
        assert transported_curve.x_label == original_curve.x_label
        assert transported_curve.y_label == original_curve.y_label
        assert transported_curve.units == original_curve.units
        np.testing.assert_allclose(transported_curve.x, original_curve.x)
        np.testing.assert_allclose(transported_curve.y, original_curve.y)
    for key in original_data.diagnostics:
        original_diagnostic = original_data.diagnostics[key]
        transported_diagnostic = regenerated.run_data.diagnostics[key]
        assert transported_diagnostic.display_name == original_diagnostic.display_name
        _assert_nested_equal(transported_diagnostic.values, original_diagnostic.values)


def test_lc_stopped_checkpoint_typed_records_and_metadata_roundtrip(tmp_path):
    registry = default_transport_registry()
    request = _lc_request()
    token = CancellationToken()

    def stop_after_first(progress):
        if progress.completed_units == 1:
            token.cancel()

    stopped_base = LC_STATIC_OPERATION.run(
        request, cancellation_token=token, progress_callback=stop_after_first
    )
    assert stopped_base.status == "stopped"
    iteration = StaticIterationRecord(
        z_index=0, z_um=2.5, optical_pass=2, coupled_pass=1,
        relax_iteration=3, residual_rms=1e-4, residual_max=2e-4,
        delta_theta_rms=3e-5, delta_theta_max=4e-5,
        theta_min=0.1, theta_max=0.8, intensity_peak=0.02,
        normalized_intensity_integral=1.0, converged=True,
        residual_before_refresh_rms=2e-4, residual_before_refresh_max=3e-4,
        residual_after_refresh_rms=1e-4, residual_after_refresh_max=2e-4,
    )
    summary = StaticSliceSummary(
        z_index=0, z_um=2.5, optical_passes=2, relaxation_iterations=3,
        final_residual_rms=1e-4, final_residual_max=2e-4,
        final_delta_theta_rms=3e-5, final_delta_theta_max=4e-5,
        theta_min=0.1, theta_max=0.8, converged=True,
        termination_reason="residual_tolerance",
    )
    stopped_checkpoint = replace(
        stopped_base.checkpoint,
        iteration_records=(iteration,), slice_summaries=(summary,),
    )
    stopped = replace(
        stopped_base, checkpoint=stopped_checkpoint,
        iteration_records=(iteration,), slice_summaries=(summary,),
    )
    run_dir = tmp_path / "lc-stopped"
    write_request_package(
        run_dir, registry=registry, material_id="lc", workflow_id="static",
        request=request, run_id="lc-stopped", execution_target="local",
    )
    write_result_package(
        run_dir, codec=registry.codec("lc", "static"), result=stopped,
        request_envelope=read_request_package(run_dir, registry=registry).envelope,
    )
    restored = read_result_package(run_dir, registry=registry).result
    assert restored.status == "stopped"
    assert restored.checkpoint.status == "stopped"
    assert restored.checkpoint.next_slice_index == stopped_checkpoint.next_slice_index
    assert restored.checkpoint.completed_slices == stopped_checkpoint.completed_slices
    assert restored.checkpoint.z_reached_um == stopped_checkpoint.z_reached_um
    assert restored.checkpoint.iteration_records == (iteration,)
    assert restored.checkpoint.slice_summaries == (summary,)
    continued = continue_static(request, restored.checkpoint)
    assert continued.status == "completed"
    assert continued.completed_slices == int(
        round(request.grid.z_length_um / request.grid.dz_um)
    )


@pytest.mark.parametrize("coherent", [True, False])
def test_pr_request_roundtrip_preserves_coherence_groups(tmp_path, coherent):
    base = _pr_request().beams.channels[0]
    groups = ("shared", "shared") if coherent else ("one", "two")
    template = _pr_request()
    request = replace(template, beams=BeamStack(channels=(
        replace(base, name="one", coherence_group=groups[0]),
        replace(base, name="two", phase_rad=0.3, coherence_group=groups[1]),
    )), initial_A=np.zeros((2, 12, 12), dtype=np.complex128),
        initial_psi=np.zeros((1, 12, 12), dtype=np.float64))
    registry = default_transport_registry()
    run_dir = tmp_path / str(coherent)
    write_request_package(
        run_dir, registry=registry, material_id="pr",
        workflow_id="pr_transverse_static", request=request,
        run_id=f"coherence-{coherent}", execution_target="local",
    )
    restored = read_request_package(run_dir, registry=registry).request
    assert tuple(c.coherence_group for c in restored.beams.channels) == groups
    assert (len(set(groups)) == 1) is coherent
    np.testing.assert_array_equal(restored.initial_A, request.initial_A)
    np.testing.assert_array_equal(restored.initial_psi, request.initial_psi)


def test_pr_continuation_and_cancellation_provenance_roundtrip(tmp_path):
    registry = default_transport_registry()
    operation = PR_TRANSVERSE_STATIC_OPERATION
    request = _pr_request()
    result = operation.run(request)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "continuation_used": True,
        "visibility_schedule": (0.0, 0.25, 0.5, 0.75, 1.0),
        "continuation_stages": (
            {"visibility": 0.0, "status": "converged", "converged": True},
            {"visibility": 0.25, "status": "cancelled", "converged": False},
        ),
        "direct_attempt_status": "not_converged",
        "direct_attempt_termination_reason": "coupled_line_search_failed",
        "requested_final_visibility": 1.0,
        "last_attempted_visibility": 0.25,
        "final_visibility": None,
        "continuation_succeeded": False,
        "continuation_cancelled": True,
        "continuation_cancellation_visibility": 0.25,
        "continuation_failure_visibility": None,
        "returned_state_source": "cancelled_continuation_stage",
        "termination_reason": "cancelled_at_accepted_boundary",
    })
    cancelled = replace(result, status="cancelled", converged=False, diagnostics=diagnostics)
    run_dir = tmp_path / "cancelled"
    write_request_package(
        run_dir, registry=registry, material_id=operation.material_id,
        workflow_id=operation.workflow_id, request=request, run_id="cancelled-run",
        execution_target="local",
    )
    write_result_package(
        run_dir, codec=registry.codec(*operation.key), result=cancelled,
        request_envelope=read_request_package(run_dir, registry=registry).envelope,
    )
    restored = read_result_package(run_dir, registry=registry).result
    assert restored.status == "cancelled"
    assert restored.diagnostics["continuation_cancelled"] is True
    assert restored.diagnostics["returned_state_source"] == "cancelled_continuation_stage"
    assert restored.diagnostics["visibility_schedule"] == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert isinstance(restored.diagnostics["continuation_stages"], tuple)
    residual = np.asarray(restored.equilibrium_residual_stack, dtype=np.float64)
    assert np.isclose(np.sqrt(np.mean(residual**2)), restored.diagnostics["equilibrium_residual_rms"])
    assert np.isclose(np.max(np.abs(residual)), restored.diagnostics["equilibrium_residual_max"])


def test_pr_successful_continuation_provenance_roundtrip(tmp_path):
    import lcprop.pr.transverse.static_workflow as workflow

    registry = default_transport_registry()
    request = _pr_request()
    direct = PR_TRANSVERSE_STATIC_OPERATION.run(request)
    continued = workflow._run_pr_transverse_static_visibility_continuation(
        request, direct_result=direct
    )
    assert continued.diagnostics["continuation_succeeded"] is True
    run_dir = tmp_path / "continued"
    write_request_package(
        run_dir, registry=registry, material_id="pr",
        workflow_id="pr_transverse_static", request=request,
        run_id="continued-run", execution_target="local",
    )
    write_result_package(
        run_dir, codec=registry.codec("pr", "pr_transverse_static"),
        result=continued,
        request_envelope=read_request_package(run_dir, registry=registry).envelope,
    )
    restored = read_result_package(run_dir, registry=registry).result
    assert restored.diagnostics["continuation_succeeded"] is True
    assert restored.diagnostics["final_visibility"] == 1.0
    assert restored.diagnostics["returned_state_source"] == "final_full_visibility_stage"
    assert restored.replay_diagnostics == continued.replay_diagnostics


def test_result_backend_requested_resolved_compatibility(tmp_path):
    registry = default_transport_registry()
    cpu_request = _pr_request()
    cpu_result = PR_TRANSVERSE_STATIC_OPERATION.run(cpu_request)
    for requested, accepted in (("numpy", True), ("auto", True), ("cupy", False)):
        request = replace(
            cpu_request,
            backend=BackendSpec(backend=requested, precision="float64", verbose=False),
        )
        run_dir = tmp_path / requested
        write_request_package(
            run_dir, registry=registry, material_id="pr",
            workflow_id="pr_transverse_static", request=request,
            run_id=f"backend-{requested}", execution_target="local",
        )
        write_result_package(
            run_dir, codec=registry.codec("pr", "pr_transverse_static"),
            result=cpu_result,
            request_envelope=read_request_package(run_dir, registry=registry).envelope,
        )
        if accepted:
            assert read_result_package(run_dir, registry=registry).result is not None
        else:
            with pytest.raises(TransportVerificationError, match="incompatible"):
                read_result_package(run_dir, registry=registry)


@pytest.mark.parametrize("array_name", [
    "A_initial", "A_final", "psi_initial", "psi_final",
    "source_intensity_stack", "equilibrium_residual_stack",
    "td_rhs_residual_stack",
])
def test_pr_result_rejects_incompatible_array_shapes(array_name):
    result = PR_TRANSVERSE_STATIC_OPERATION.run(_pr_request())
    encoded = encode_pr_transverse_static_transport_result(result)
    metadata = dict(encoded.payload.metadata)
    arrays = dict(encoded.payload.arrays)
    key = metadata[array_name]["__lcprop_array__"]
    arrays[key] = arrays[key][..., :-1]
    with pytest.raises(TransportCodecError, match=f"{array_name} shape"):
        decode_pr_transverse_static_transport_result(metadata, arrays)


def test_pr_result_rejects_inconsistent_continuation_provenance():
    result = PR_TRANSVERSE_STATIC_OPERATION.run(_pr_request())
    encoded = encode_pr_transverse_static_transport_result(result)
    metadata = dict(encoded.payload.metadata)
    diagnostics = dict(metadata["diagnostics"])
    diagnostics.update({
        "continuation_used": True,
        "visibility_schedule": [0.0, 1.0],
        "continuation_stages": [
            {"visibility": 0.0, "status": "converged", "converged": True},
        ],
        "direct_attempt_status": "not_converged",
        "direct_attempt_termination_reason": "coupled_line_search_failed",
        "requested_final_visibility": 1.0,
        "last_attempted_visibility": 0.0,
        "final_visibility": None,
        "continuation_succeeded": True,
        "continuation_cancelled": False,
        "continuation_cancellation_visibility": None,
        "continuation_failure_visibility": None,
        "returned_state_source": "final_full_visibility_stage",
    })
    metadata["diagnostics"] = diagnostics
    with pytest.raises(TransportCodecError, match="must reach requested visibility"):
        decode_pr_transverse_static_transport_result(metadata, encoded.payload.arrays)


def test_headless_executor_uses_registered_operation_and_writes_verified_result(tmp_path):
    registry = default_transport_registry()
    run_dir = tmp_path / "headless"
    write_request_package(
        run_dir, registry=registry, material_id="lc", workflow_id="static",
        request=_lc_request(), run_id="headless-run", execution_target="local",
    )
    assert execute_run_directory(run_dir) == 0
    decoded = read_result_package(run_dir, registry=registry)
    assert decoded.result.status == "completed"
    assert (run_dir / "output" / "RESULT_READY.json").is_file()


def test_headless_module_entrypoint_runs_without_qt_or_scheduler(tmp_path):
    registry = default_transport_registry()
    run_dir = tmp_path / "module"
    write_request_package(
        run_dir, registry=registry, material_id="lc", workflow_id="static",
        request=_lc_request(), run_id="module-run", execution_target="local",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "lcprop.transport.executor", "--run-dir", str(run_dir)],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "PySide6" not in completed.stderr
    assert read_result_package(run_dir, registry=registry).result.status == "completed"


def test_checksum_corruption_is_rejected_before_array_loading(tmp_path):
    registry = default_transport_registry()
    run_dir = tmp_path / "corrupt"
    write_request_package(
        run_dir, registry=registry, material_id="lc", workflow_id="static",
        request=replace(_lc_request(), initial_theta=np.zeros((12, 10))),
        run_id="corrupt-run", execution_target="local",
    )
    arrays = run_dir / "request" / "request_arrays.npz"
    before = sha256_file(arrays)
    with arrays.open("ab") as stream:
        stream.write(b"corrupt")
    assert sha256_file(arrays) != before
    with pytest.raises(TransportVerificationError):
        read_request_package(run_dir, registry=registry)


@pytest.mark.parametrize(
    ("envelope_filename", "ready_filename"),
    (("request.json", "READY.json"), ("result.json", "RESULT_READY.json")),
)
def test_artifact_envelope_manifest_run_id_mismatch_is_rejected(
    tmp_path, envelope_filename, ready_filename,
):
    bundle = tmp_path / envelope_filename
    write_artifact_bundle(
        bundle, run_id="manifest-run", envelope_filename=envelope_filename,
        envelope={"run_id": "envelope-run", "array_manifest": {}},
        arrays_filename="arrays.npz", arrays={}, ready_filename=ready_filename,
    )
    with pytest.raises(TransportVerificationError, match="run_id mismatch"):
        verify_artifact_bundle(
            bundle, envelope_filename=envelope_filename,
            arrays_filename="arrays.npz", ready_filename=ready_filename,
        )


def test_missing_artifact_and_unsupported_schema_fail_explicitly(tmp_path):
    registry = default_transport_registry()
    missing_dir = tmp_path / "missing"
    write_request_package(
        missing_dir, registry=registry, material_id="lc", workflow_id="static",
        request=_lc_request(), run_id="missing-run", execution_target="local",
    )
    (missing_dir / "request" / "READY.json").unlink()
    with pytest.raises(TransportVerificationError, match="missing required artifact"):
        read_request_package(missing_dir, registry=registry)

    schema_dir = tmp_path / "schema"
    write_request_package(
        schema_dir, registry=registry, material_id="lc", workflow_id="static",
        request=_lc_request(), run_id="schema-run", execution_target="local",
    )
    # Schema edits necessarily invalidate the checksum and are rejected before decode.
    request_path = schema_dir / "request" / "request.json"
    value = json.loads(request_path.read_text())
    value["transport_schema_version"] = 999
    request_path.write_text(json.dumps(value))
    with pytest.raises(TransportVerificationError):
        read_request_package(schema_dir, registry=registry)

    envelope = RequestEnvelope(
        run_id="schema", material_id="lc", workflow_id="static",
        codec_id="lc.static.request", codec_version=1,
        scientific_backend_requested="numpy", request_payload={},
        transport_schema_version=999,
    )
    with pytest.raises(TransportSchemaError):
        RequestEnvelope.from_dict(envelope.to_dict())

    codec_dir = tmp_path / "codec"
    write_request_package(
        codec_dir, registry=registry, material_id="lc", workflow_id="static",
        request=_lc_request(), run_id="codec-run", execution_target="local",
    )
    decoded = read_request_package(codec_dir, registry=registry)
    incompatible = replace(decoded.codec, request_codec_version=999)
    replacement = type(registry)()
    replacement.register(incompatible)
    with pytest.raises(TransportCodecError, match="unsupported request codec"):
        read_request_package(codec_dir, registry=replacement)


def test_remote_status_failure_and_backend_transitions():
    pending = RemoteRunStatus(
        run_id="cancel", execution_target="slurm", state=RemoteRunState.PENDING,
        scientific_backend_requested="auto", scientific_backend_resolved="unresolved",
    )
    requested = transition_remote_status(pending, RemoteRunState.CANCEL_REQUESTED)
    assert transition_remote_status(requested, RemoteRunState.CANCELLED).terminal
    for target in (RemoteRunState.TIMEOUT, RemoteRunState.OUT_OF_MEMORY):
        running = replace(pending, state=RemoteRunState.RUNNING)
        assert transition_remote_status(running, target).terminal
    resolved = replace(pending, scientific_backend_resolved="cupy")
    assert resolved.scientific_backend_requested == "auto"
    assert resolved.scientific_backend_resolved == "cupy"


def test_headless_failure_is_structured_and_nonzero(tmp_path):
    registry = default_transport_registry()
    run_dir = tmp_path / "failed"
    write_request_package(
        run_dir, registry=registry, material_id="lc", workflow_id="static",
        request=_lc_request(), run_id="failed-run", execution_target="local",
    )
    class BrokenOperation:
        pass
    # An empty operation composition forces a bounded execution failure.
    assert execute_run_directory(run_dir, registry=registry, operations=()) == 1
    failure = read_failure_package(run_dir)
    assert failure.failure_category == "scientific_process_failed"
    assert failure.exception_type == "KeyError"
    assert failure.traceback


def test_transport_source_has_no_qt_slurm_or_material_branching():
    root = Path(__file__).parents[1] / "src" / "lcprop" / "transport"
    shared = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
        if path.name != "defaults.py"
    )
    assert "PySide6" not in shared
    assert "sbatch" not in shared
    assert "isinstance(request, StaticRunRequest)" not in shared
    assert "allow_pickle=False" in (root / "artifacts.py").read_text(encoding="utf-8")


def test_importing_shared_transport_does_not_import_material_code():
    probe = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import lcprop.transport; "
            "assert 'lcprop.lc.transport_codec' not in sys.modules; "
            "assert 'lcprop.pr.transverse.transport_codec' not in sys.modules"
        )],
        capture_output=True, text=True, check=False,
    )
    assert probe.returncode == 0, probe.stderr

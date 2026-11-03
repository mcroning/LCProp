"""Optional image analysis of ordinary PR results, without scientific dispatch."""

from dataclasses import dataclass, replace
from time import perf_counter

from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.products.data_model import DiagnosticCollection, DiagnosticData
from lcprop.pr.image_amplification import (
    PRImageAmplificationExperimentRequest,
    analyze_image_amplification_result, image_amplification_base_capabilities,
)


@dataclass(frozen=True)
class PRImageAnalysisSelection:
    """Session-local role snapshot; not a request codec or durable identity."""

    pump_channel_index: int | None
    signal_channel_index: int | None


def add_optional_image_analysis(
    base, request, selection, *, cancellation_token=None, progress_callback=None,
    base_runtime=0.0,
):
    """Keep ordinary result/status authoritative even if analysis is unavailable.

    The base operation has already completed exactly once. Historical requests
    and composite status semantics remain on their existing compatibility path.
    """
    data = base.run_data
    status = "unavailable"
    reason = ""
    validation = "not_assessed"
    try:
        if getattr(base.result, "status", None) not in ("completed", "converged"):
            reason = "Base propagation did not complete successfully."
        elif cancellation_token is not None and cancellation_token.is_cancelled():
            status, reason = "cancelled", "Image analysis cancelled after propagation."
        elif (getattr(base.result, "A_initial", None) is None
              or getattr(base.result, "A_final", None) is None):
            status = "not_selected"
            reason = ("Optional image analysis requires complex_input and complex_output. "
                      "Select both in Analysis, or select Full, before running.")
        elif (selection.pump_channel_index is None
              or selection.signal_channel_index is None):
            reason = "Select Pump and Signal roles to request specialized image analysis."
        else:
            composite = PRImageAmplificationExperimentRequest(
                base_workflow_id=base.kind,
                base_request=request,
                launch_configuration=LaunchConfiguration(
                    request.beams, request.launch_elements,
                ),
                pump_channel_index=selection.pump_channel_index,
                signal_channel_index=selection.signal_channel_index,
            )
            try:
                composite.validate()
            except (TypeError, ValueError) as exc:
                reason = str(exc)
            else:
                validation = next(
                    c.validation_status for c in image_amplification_base_capabilities()
                    if c.workflow_id == base.kind
                )
                if request.material_response.model in ("linearized", "field_linear_local_intensity"):
                    validation = "compatible_validation_pending"
                # Reuse the existing two-carrier applicability/quality decision;
                # do not invent an N-carrier algorithm or alter historical IA.
                carrier = data.diagnostics.get("carrier_power")
                if carrier is None or carrier.values.get("status") != "ok":
                    reason = (
                        "Carrier-resolved image analysis unavailable: "
                        + str(carrier.values.get("reason") if carrier else
                              "carrier separation evidence is unavailable")
                    )
                else:
                    analyzed = analyze_image_amplification_result(
                        composite, base,
                        cancellation_token=cancellation_token,
                        progress_callback=progress_callback,
                        base_runtime=base_runtime,
                        benchmark_started_at=perf_counter() - base_runtime,
                    )
                    status = analyzed.result.analysis_status
                    reason = analyzed.result.analysis_message
                    if status == "completed":
                        data = analyzed.run_data
    except Exception as exc:
        status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        data = base.run_data

    diagnostics = DiagnosticCollection(list(data.diagnostics.items()))
    # Optional augmentation may add image products, but never reinterpret or
    # overwrite the ordinary run's diagnostics (including carrier_gain).
    for key, diagnostic in base.run_data.diagnostics.items():
        diagnostics.add(key, diagnostic)
    diagnostics.add("optional_image_analysis", DiagnosticData(
        "optional_image_analysis", "Optional image analysis",
        {"status": status, "reason": reason, "validation_status": validation,
         "propagation_status": base.result.status,
         "pump_channel_index": selection.pump_channel_index,
         "signal_channel_index": selection.signal_channel_index,
         "role_identity": "session_enabled_channel_index"},
    ))
    return replace(
        base,
        run_data=replace(data, workflow=base.run_data.workflow, diagnostics=diagnostics),
        message=f"{base.message}\nOptional image analysis {status}: {reason}",
    )

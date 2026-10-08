"""Shared presentation of request attempts; never executes a workflow."""
from __future__ import annotations

import traceback


def begin_request(window, operation: str) -> None:
    window.results_panel.workspace.begin_request(operation)
    window.tabs.setCurrentWidget(window.results_panel)


def inspect_request(window, builder) -> None:
    """Capture editor values and validate/describe without dispatching a worker."""
    if window._background_running:
        return
    workspace = window.results_panel.workspace
    workspace.mark_previous()
    workspace.operation_boundary("Inspect Request")
    workspace.set_operation_status("Validating / Preparing preview")
    try:
        request = builder()
        window._validate_execution_request(request)
        summary = window.describe_request(request)
    except ValueError as exc:
        # Expected editor/capability rejection is a user-facing validation result.
        # Keep unexpected implementation failures on the diagnostic traceback path.
        report_failure(window, 'Invalid request: ' + str(exc))
    except Exception:
        report_failure(window, traceback.format_exc())
    else:
        workspace.set_request_summary("Pre-run inspection (not executed)\n" + summary)
        workspace.set_operation_status("Request validated; not executed")
        workspace.tabs.setCurrentWidget(workspace.request_summary)
    window.tabs.setCurrentWidget(window.results_panel)


def report_failure(window, formatted_traceback: str) -> None:
    """Keep the complete exception while making its cause visible above results."""
    lines = formatted_traceback.strip().splitlines()
    cause = lines[-1] if lines else "Unknown failure"
    if 'nyquist' in formatted_traceback.lower():
        cause = 'Optical launch exceeds grid sampling (Nyquist); adjust beam angles or grid/aperture'
    elif len(cause) > 180:
        cause = cause[:177] + '…'
    workspace = window.results_panel.workspace
    workspace.finish_attempt("State at failure")
    workspace.set_operation_status(
        f"Failed — {cause}. Review the request/configuration; details in Console."
    )
    workspace.append_console(formatted_traceback)


def execution_summary(window, request, *, runner=None) -> str:
    base = request
    while hasattr(base, "base"):
        base = base.base
    backend = getattr(base, "backend", None)
    requested = getattr(backend, "backend", "numpy")
    precision = getattr(backend, "precision", None)
    if precision is None:
        precision = getattr(getattr(base, "runtime", None), "precision", "float64")
    runner = window.runner if runner is None else runner
    remote = runner is window.slurm_runner and window.slurm_runner is not None
    lines = [
        "Execution target: " + ("Slurm" if remote else "Local"),
        f"Requested backend: {requested}",
        f"Precision: {precision}",
        "Resolved backend: " + (
            "NumPy (LC workflow)" if backend is None else
            "unresolved until execution; no device probe performed"
        ),
    ]
    profile = None
    if remote:
        # Runner configuration owns the cluster; dispatch may override its resource.
        if window._explicit_slurm_runner is not None:
            config = getattr(runner, "config", None)
            cluster = getattr(config, "cluster_profile", "provided runner")
            resource = window.remote_execution_controls.runner_kwargs().get(
                "resource_profile",
                getattr(config, "default_resource_profile", "provided runner"),
            )
            profiles = getattr(config, "resource_profiles", ())
            profile = next((p for p in profiles if p.name == resource), None)
        else:
            controls = window.remote_execution_controls
            selected = controls.selected_cluster()
            cluster = selected.name if selected is not None else "not selected"
            resource = controls.selected_resource_name() or "not selected"
            if selected is not None and resource != "not selected":
                profile = selected.profile(resource)
        lines.extend([f"Cluster: {cluster}", f"Resource: {resource}"])
        selector = getattr(window, "result_policy_selector", None)
        policy = selector.currentData() if selector is not None else "full"
        lines.append(f"Retrieval policy: {policy}")
        if requested == "numpy" and profile is not None and profile.gpus > 0:
            lines.append("Warning: NumPy scientific computation will not use the allocated GPU.")
    if type(base).__name__.startswith("PRTransverse"):
        lines.append("Full-transverse workflows require explicit NumPy or CuPy; Auto is not supported.")
    lines.append("Comparison resource estimates are not the configured execution plan.")
    return "\n".join(lines)


def source_preflight(window) -> None:
    if window.runner is window.slurm_runner and window.slurm_runner is not None:
        preflight = getattr(window.runner, "preflight_source", None)
        if preflight is not None:
            preflight()

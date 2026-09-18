"""Presentation of retained soliton qualification; never runs a solver."""
from __future__ import annotations

GATES = (
    ("field_rel", "Field relative change", "tol_field"),
    ("dtheta_rms", "Theta update RMS", "tol_theta"),
    ("residual_rms", "Director residual RMS", "tol_residual_rms"),
    ("residual_max", "Director residual maximum", "tol_residual_max"),
)
CONVERGENCE_LABEL = "Solver converged within configured limits"
CONTINUATION_NOTE = (
    "Continuation initializes a subsequent member from a previously converged "
    "member and can improve branch following. All configured strict convergence "
    "criteria must still pass in the same completed outer iteration. "
    "Nonconvergence does not establish physical nonexistence or instability."
)


def qualification(result):
    """Use the last completed outer row, not post-polish/rounded metrics."""
    history = getattr(result, "history", ()) or ()
    final = history[-1] if history else {}
    request = getattr(result, "request", None)
    metrics = getattr(result, "metrics", {})
    transverse = metrics.get("solver") == "transverse_eigen" or "optical_residual" in final
    gates = list(GATES)
    if transverse:
        gates.append(("optical_residual", "Optical residual (outer iteration)", "tol_field"))
    rows = []
    for key, label, tolerance_key in gates:
        value = final.get(key)
        tolerance = getattr(request, tolerance_key, None)
        rows.append({
            "criterion": label, "key": key, "value": value,
            "tolerance": tolerance, "comparison": "<",
            "qualification": (
                "Unavailable" if value is None or tolerance is None else
                "Pass" if value < tolerance else "Fail"
            ),
            "value_source": "Last completed outer iteration" if history else "No completed outer row retained",
        })
    reason = metrics.get("convergence_status")
    if reason is None:
        reason = "converged" if result.converged else getattr(result, "status", "unknown")
    status = {
        "converged": CONVERGENCE_LABEL,
        "max_outer_reached": "Solver did not converge within configured limits",
        "stopped": "Stopped / cancelled",
        "cancelled": "Stopped / cancelled",
        "running": "Running; not yet qualified",
    }.get(reason, "Solver nonconverged" if not result.converged else CONVERGENCE_LABEL)
    return {
        "solver_status": status,
        "termination_reason": reason,
        "completed_iterations": getattr(result, "completed_iterations", None),
        "iteration_budget": getattr(request, "max_outer", getattr(result, "total_iterations", None)),
        "qualification_source": "Last completed outer iteration; strict gates must pass together",
        "polishing_note": (
            "Transverse eigenpair final polishing does not update the outer convergence flag. "
            "Returned fields and final optical residual can differ from qualification values."
            if transverse else "No physical existence or stability conclusion follows from nonconvergence."
        ),
    }, rows


def enrich_sample(row, result):
    enriched = dict(row)
    summary, gates = qualification(result)
    enriched.update(summary)
    for gate in gates:
        key = gate['key']
        enriched[key + '_qualification_value'] = gate['value']
        enriched[key + '_tolerance'] = gate['tolerance']
        enriched[key + '_gate'] = gate['qualification']
    return enriched, gates

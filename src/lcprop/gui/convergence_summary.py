"""Read-only explanations of existing member and convergence-gate products."""
from lcprop.gui.number_format import format_number


def member_explanations(run_data):
    gates_product = run_data.diagnostics.get("convergence_gates")
    if gates_product is None:
        return []
    gates = gates_product.values.get("rows", ())
    members = run_data.diagnostics.get("members")
    samples = run_data.diagnostics.get("table")
    rows = members.values.get("rows", ()) if members else ()
    if not rows and samples:
        rows = samples.values.get("rows", ())
    if not rows:
        return []  # No member identity/status: do not invent a sweep summary.
    explanations = []
    for position, row in enumerate(rows):
        identity = row.get("requested_index", row.get("i", position))
        power = row.get("requested_power_mW", row.get("power_mW"))
        label = f"Member {identity}"
        if power is not None:
            label += f" — {format_number(power)} mW"
        status = row.get("status", "Unavailable")
        converged = row.get("converged")
        convergence = "Unavailable" if converged is None else "Yes" if converged else "No"
        reason = row.get("termination_reason", "Unavailable")
        reason = {"max_outer_reached": "Reached the iteration limit",
                  "converged": "All configured criteria satisfied",
                  "stopped": "Stopped / cancelled", "cancelled": "Stopped / cancelled"}.get(reason, reason)
        lines = [f"Execution status: {status}.",
                 f"Solver converged within configured limits: {convergence}.",
                 f"Termination: {reason}.",
                 f"Iterations: {format_number(row.get('completed_iterations'))} / {format_number(row.get('iteration_budget'))}."]
        member_gates = [g for g in gates if g.get("member") == identity]
        failed = [g for g in member_gates if g.get("qualification") == "Fail"]
        unknown = [g for g in member_gates if g.get("qualification") not in {"Pass", "Fail"}]
        for gate in failed:
            lines.append(f"{gate['criterion']} = {format_number(gate.get('value'), exact=True)}; "
                         f"required {gate.get('comparison', '<')} {format_number(gate.get('tolerance'), exact=True)} (failed).")
        if unknown or not member_gates:
            lines.append("Some convergence evidence is unavailable; inspect Samples / Tables.")
        elif failed:
            if len(failed) < len(member_gates):
                lines.append("Other configured convergence gates passed.")
        else:
            lines.append("All retained convergence gates passed.")
        if row.get("error_text"):
            lines.append(str(row["error_text"]))
        lines.append("Nonconvergence alone does not establish physical nonexistence or instability. "
                     "Exact qualification comes from retained outer iterations, not final polishing; see Convergence gates.")
        if converged is None:
            label += f" — Execution {status}; convergence unavailable"
        elif converged:
            label += " — Solver converged"
        else:
            label += " — Solver not converged"
        if failed:
            label += "; " + ", ".join(g["criterion"] for g in failed) + " failed"
        explanations.append((identity, label, "\n".join(lines)))
    return explanations

"""Report the bounded PR pump/signal numerical-readiness case.

Run from the repository root with the lcprop environment::

    python scripts/checks/pr_image_amplification_readiness.py
"""

from lcprop.pr.readiness import run_image_amplification_readiness


def main() -> None:
    result = run_image_amplification_readiness()
    print(f"grid={result.request.grid}")
    print(f"solver={result.request.solver}")
    print(
        "dt_normalized,explicit_euler_limit,semi_implicit_limit="
        f"{result.request.solver.dt_normalized:.12g},"
        f"{result.explicit_euler_dt_limit:.12g},"
        f"{result.semi_implicit_dt_limit:.12g}"
    )
    print(
        "final_residual_rms,final_residual_max="
        f"{result.final_residual_rms:.12g},{result.final_residual_max:.12g}"
    )
    print(
        "static_relative_l2_error,static_max_error="
        f"{result.static_relative_l2_error:.12g},"
        f"{result.static_max_error:.12g}"
    )
    print(
        "zero_response_output_matched_powers="
        f"{result.zero_response_output_matched_powers}"
    )
    print(
        "final_response_output_matched_powers="
        f"{result.final_response_output_matched_powers}"
    )
    print(
        "signal_matched_power_relative_change="
        f"{result.signal_matched_power_relative_change:.12g}"
    )
    print(
        "normalized_power_relative_drift="
        f"{result.normalized_power_relative_drift:.12g}"
    )
    print(f"runtime_s={result.runtime_s:.6g}")


if __name__ == "__main__":
    main()

"""Research-scale PR two-plane-wave convergence check.

Run from the repository root with the lcprop environment::

    python scripts/checks/pr_two_beam_coupling.py
"""

from dataclasses import replace
import math

from lcprop.pr.coupling import PlaneWaveCouplingSpec, run_plane_wave_coupling


def main() -> None:
    base = PlaneWaveCouplingSpec(
        Nx=256,
        Ny=4,
        x_aperture_um=4.0 * math.pi * 4.0 / 0.76,
        positive_mode_index=4,
        beam_ratio=0.01,
        gain_length_product=0.3,
        Nt=2250,
        dt_normalized=0.0025,
    )
    print("dz_um,time,gamma_measured,gamma_expected,relative_error,power_drift,runtime_s")
    for dz_um in (10.0, 5.0, 2.5, 1.25):
        result = run_plane_wave_coupling(replace(base, dz_um=dz_um))
        relative_error = (
            result.gamma_p_L_measured / result.gamma_p_L_expected - 1.0
        )
        print(
            f"{dz_um:g},{result.run_result.time_normalized:g},"
            f"{result.gamma_p_L_measured:.12g},"
            f"{result.gamma_p_L_expected:.12g},"
            f"{relative_error:.6g},"
            f"{result.output_total_power-result.input_total_power:.6g},"
            f"{result.runtime_s:.6g}"
        )


if __name__ == "__main__":
    main()

"""Generate a standalone static torque-balance review figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.lc import (
    BiasSpec,
    LCMaterial,
    OutputOptions,
    StaticRunRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    run_static,
)
from lcprop.lc.static_torque_balance import (
    build_static_torque_balance_data,
    plot_static_torque_balance,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("value must be an integer >= 3")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a static LCProp case and plot its fixed-x torque balance.",
    )
    parser.add_argument("--nx", type=_positive_int, default=128)
    parser.add_argument("--ny", type=_positive_int, default=128)
    parser.add_argument("--z-length-um", type=_positive_float, default=1500.0)
    parser.add_argument("--dz-um", type=_positive_float, default=5.0)
    parser.add_argument("--x-aperture-um", type=_positive_float, default=75.0)
    parser.add_argument("--y-aperture-um", type=_positive_float, default=100.0)

    parser.add_argument("--beam-waist-um", type=_positive_float, default=5.0)
    parser.add_argument(
        "--beam-separation-um",
        type=_nonnegative_float,
        default=20.0,
        help="center-to-center y separation; two beams are placed at ±separation/2",
    )
    parser.add_argument(
        "--beam-power-mw",
        type=_positive_float,
        default=1.0,
        help="power per beam; overridden when --total-power-mw is supplied",
    )
    parser.add_argument(
        "--total-power-mw",
        type=_positive_float,
        default=None,
        help="optional total power divided equally across active beams",
    )

    beam_count = parser.add_mutually_exclusive_group()
    beam_count.add_argument(
        "--two-beams",
        dest="two_beams",
        action="store_true",
        help="launch two mutually incoherent beams (default)",
    )
    beam_count.add_argument(
        "--single-beam",
        dest="two_beams",
        action="store_false",
        help="launch one beam centered at y=0",
    )
    parser.set_defaults(two_beams=True)

    parser.add_argument(
        "--x-um",
        type=float,
        default=0.0,
        help="requested physical x coordinate for the interior torque cut",
    )
    parser.add_argument(
        "--vbias",
        type=_nonnegative_float,
        default=0.9144,
        help="LC bias voltage in volts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("static_torque_balance.png"),
    )
    return parser


def _beam_stack(args: argparse.Namespace) -> BeamStack:
    beam_count = 2 if args.two_beams else 1
    power_mw = (
        args.total_power_mw / beam_count
        if args.total_power_mw is not None
        else args.beam_power_mw
    )
    if args.two_beams:
        half_separation_um = 0.5 * args.beam_separation_um
        channels = (
            BeamChannel(
                name="lower beam",
                power_mW=power_mw,
                waist_x_um=args.beam_waist_um,
                waist_y_um=args.beam_waist_um,
                x0_um=0.0,
                y0_um=-half_separation_um,
                coherence_group="lower",
            ),
            BeamChannel(
                name="upper beam",
                power_mW=power_mw,
                waist_x_um=args.beam_waist_um,
                waist_y_um=args.beam_waist_um,
                x0_um=0.0,
                y0_um=half_separation_um,
                coherence_group="upper",
            ),
        )
    else:
        channels = (
            BeamChannel(
                name="single beam",
                power_mW=power_mw,
                waist_x_um=args.beam_waist_um,
                waist_y_um=args.beam_waist_um,
                x0_um=0.0,
                y0_um=0.0,
                coherence_group="single",
            ),
        )
    return BeamStack(channels=channels, coherence="incoherent")


def _request(args: argparse.Namespace) -> StaticRunRequest:
    workflow = StaticWorkflowOptions(
        strategy="local_self_consistent",
        theta_solver="picard_cn",
        optics_solver="splitstep",
        coupling="self_consistent",
    )
    return StaticRunRequest(
        grid=GridSpec(
            Nx=args.nx,
            Ny=args.ny,
            dz_um=args.dz_um,
            x_aperture_um=args.x_aperture_um,
            y_aperture_um=args.y_aperture_um,
            z_length_um=args.z_length_um,
        ),
        material=LCMaterial(),
        bias=BiasSpec(V_bias=args.vbias),
        beams=_beam_stack(args),
        solver=StaticSolverOptions(
            workflow=workflow,
            record_iteration_history=False,
        ),
        output=OutputOptions(),
    )


def main() -> None:
    args = _parser().parse_args()

    dx_um = args.x_aperture_um / args.nx
    x_um = (np.arange(args.nx) - 0.5 * (args.nx - 1)) * dx_um
    interior_x_um = x_um[1:-1]
    x_index = int(np.argmin(np.abs(interior_x_um - args.x_um))) + 1
    print(f"requested_x_um = {args.x_um}")
    print(f"selected_x_index = {x_index}")
    print(f"selected_x_um = {x_um[x_index]}", flush=True)

    result = run_static(_request(args))
    data = build_static_torque_balance_data(result, x_index=x_index)
    fig, axes = plot_static_torque_balance(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")

    print(f"image_path = {args.output.resolve()}")
    print(f"source_theta_shape = {np.shape(result.theta_final)}")
    print(f"source_intensity_shape = {np.shape(result.theta_intensity_stack)}")
    print(f"elastic_shape = {data.elastic.shape}")
    print(f"drive_shape = {data.drive.shape}")
    print(f"residual_shape = {data.residual.shape}")
    print(f"relative_shape = {data.relative_residual.shape}")
    print(f"torque_clim = {axes[0, 0].images[0].get_clim()}")
    print(f"residual_clim = {axes[1, 0].images[0].get_clim()}")
    print(f"relative_clim = {axes[1, 1].images[0].get_clim()}")
    for key, value in data.metrics.items():
        print(f"{key} = {value}")


if __name__ == "__main__":
    main()

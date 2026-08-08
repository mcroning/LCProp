#!/usr/bin/env python3
"""Small CPU readiness check for partition-independent PR scattering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import scipy

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.pr.scattering import (
    PRCanonicalScatteringSpec,
    canonical_scattering_phase_increment,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static_streaming import (
    PRStreamingStaticOptions,
    PRStreamingStaticRequest,
    PR_STREAMING_PRODUCTION,
    run_pr_static_streaming,
)
from lcprop.pr.static_workflow import PRStaticWorkflowOptions


FINE_DZ_UM = 2.0
COARSE_DZ_UM = 10.0
Z_LENGTH_UM = 20.0
NX = 32
NY = 16
X_APERTURE_UM = 64.0
Y_APERTURE_UM = 32.0


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _scattering_spec() -> PRCanonicalScatteringSpec:
    return PRCanonicalScatteringSpec(
        epsilon=0.02,
        transverse_correlation_um=0.8,
        realization_seed=20260808,
        canonical_dz_um=FINE_DZ_UM,
    )


def _phase(z_start_um: float, dz_um: float) -> np.ndarray:
    return canonical_scattering_phase_increment(
        _scattering_spec(),
        z_start_um=z_start_um,
        dz_um=dz_um,
        z_length_um=Z_LENGTH_UM,
        Nx=NX,
        Ny=NY,
        x_aperture_um=X_APERTURE_UM,
        y_aperture_um=Y_APERTURE_UM,
        real_dtype=np.float64,
        xp=np,
    )


def _request(dz_um: float) -> PRStreamingStaticRequest:
    grid = GridSpec(
        Nx=NX,
        Ny=NY,
        x_aperture_um=X_APERTURE_UM,
        y_aperture_um=Y_APERTURE_UM,
        z_length_um=Z_LENGTH_UM,
        dz_um=dz_um,
    )
    beams = BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                waist_x_um=12.0,
                waist_y_um=8.0,
                coherence_group="scattering-readiness",
            ),
        ),
        coherence="coherent",
    )
    return PRStreamingStaticRequest(
        grid=grid,
        beams=beams,
        material=PRMaterialSpec(
            dark_intensity=0.1,
            gain_length_product=0.05,
            refractive_index=2.4,
            characteristic_wavenumber_per_um_override=0.2,
        ),
        solver=PRStreamingStaticOptions(
            coupled=PRStaticWorkflowOptions(
                optical_substeps=int(round(dz_um / FINE_DZ_UM)),
                max_coupled_passes=12,
                residual_rms_tolerance=1e-8,
                residual_max_tolerance=1e-7,
            ),
            mode=PR_STREAMING_PRODUCTION,
            deterministic_replay=True,
            partition_independent_scattering=_scattering_spec(),
        ),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coarse_intervals = []
    fine_intervals = []
    interval_errors = []
    for coarse_start in np.arange(0.0, Z_LENGTH_UM, COARSE_DZ_UM):
        coarse = _phase(float(coarse_start), COARSE_DZ_UM)
        fine = np.sum(
            [
                _phase(float(fine_start), FINE_DZ_UM)
                for fine_start in np.arange(
                    coarse_start,
                    coarse_start + COARSE_DZ_UM,
                    FINE_DZ_UM,
                )
            ],
            axis=0,
        )
        coarse_intervals.append(coarse)
        fine_intervals.append(fine)
        interval_errors.append(
            {
                "z_start_um": float(coarse_start),
                "relative_l2": float(np.linalg.norm(coarse - fine) / np.linalg.norm(fine)),
                "max_abs": float(np.max(np.abs(coarse - fine))),
            }
        )

    coarse_total = np.sum(coarse_intervals, axis=0)
    fine_total = np.sum(fine_intervals, axis=0)
    started = perf_counter()
    coarse_run = run_pr_static_streaming(_request(COARSE_DZ_UM))
    coarse_seconds = perf_counter() - started
    started = perf_counter()
    fine_run = run_pr_static_streaming(_request(FINE_DZ_UM))
    fine_seconds = perf_counter() - started

    coarse_provenance = coarse_run.memory_policy["volume_noise"]
    fine_provenance = fine_run.memory_policy["volume_noise"]
    field_relative = float(
        np.linalg.norm(coarse_run.A_final - fine_run.A_final)
        / np.linalg.norm(fine_run.A_final)
    )
    record = {
        "purpose": (
            "readiness only; output-field differences measure longitudinal "
            "discretization and screen placement, not Figure A5 ring suppression"
        ),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "configuration": {
            "shape": [NX, NY],
            "aperture_um": [X_APERTURE_UM, Y_APERTURE_UM],
            "z_length_um": Z_LENGTH_UM,
            "coarse_dz_um": COARSE_DZ_UM,
            "fine_dz_um": FINE_DZ_UM,
            "coarse_optical_substeps": int(COARSE_DZ_UM / FINE_DZ_UM),
            "fine_optical_substeps": 1,
        },
        "scattering_identity": {
            "configuration_sha256": coarse_provenance["configuration_sha256"],
            "canonical_seed_sha256_le_u32": coarse_provenance[
                "canonical_seed_sha256_le_u32"
            ],
            "coarse_and_fine_provenance_equal": coarse_provenance == fine_provenance,
            "coarse_total_phase_sha256": _sha256(coarse_total),
            "fine_total_phase_sha256": _sha256(fine_total),
            "total_phase_relative_l2": float(
                np.linalg.norm(coarse_total - fine_total) / np.linalg.norm(fine_total)
            ),
            "total_phase_max_abs": float(np.max(np.abs(coarse_total - fine_total))),
            "interval_errors": interval_errors,
            "provenance": coarse_provenance,
        },
        "workflow": {
            "coarse_status": coarse_run.status,
            "fine_status": fine_run.status,
            "coarse_replay_consistent": coarse_run.replay_diagnostics["consistent"],
            "fine_replay_consistent": fine_run.replay_diagnostics["consistent"],
            "coarse_runtime_s": coarse_seconds,
            "fine_runtime_s": fine_seconds,
            "output_field_relative_l2": field_relative,
            "coarse_power_relative_drift": (
                coarse_run.power_final - coarse_run.power_initial
            )
            / coarse_run.power_initial,
            "fine_power_relative_drift": (
                fine_run.power_final - fine_run.power_initial
            )
            / fine_run.power_initial,
        },
    }
    if not coarse_provenance == fine_provenance:
        raise AssertionError("coarse and fine scattering provenance differs")
    if record["scattering_identity"]["total_phase_relative_l2"] >= 2e-15:
        raise AssertionError("coarse/fine integrated scattering phases disagree")
    if not (
        coarse_run.replay_diagnostics["consistent"]
        and fine_run.replay_diagnostics["consistent"]
    ):
        raise AssertionError("deterministic workflow replay failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

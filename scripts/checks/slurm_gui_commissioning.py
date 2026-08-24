"""Bounded GUI-to-Slurm commissioning checks for canonical remote execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.core.backend import BackendSpec
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.static_workflow import PR_TRANSVERSE_STATIC_WORKFLOW
from pr_coherent_grating_diagnosis import fixture as coherent_fixture
from lcprop.runners.local import LocalRunner
from lcprop.runners.slurm import SlurmExecutionConfig, SlurmRunner
from lcprop.transport.defaults import (
    default_transport_operations, default_transport_registry,
)


SOURCE_SHA = "eb9382c0843a25136309728e57d0a7fb4dae333f"
SOURCE_PATH = "/cluster/tufts/cglab/mcroning/lcprop_runs/lcprop-remote-eb9382c0843a"


def _runner(local_root: Path) -> SlurmRunner:
    return SlurmRunner(
        SlurmExecutionConfig(
            host="mcroning@login.pax.tufts.edu",
            remote_run_root="/cluster/tufts/cglab/mcroning/lcprop_runs",
            remote_python="/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python",
            remote_source_path=SOURCE_PATH,
            source_git_sha=SOURCE_SHA,
            local_artifact_root=local_root,
            poll_interval=5.0,
        ),
        default_transport_operations(),
        registry=default_transport_registry(),
    )


def run_lc(output: Path) -> dict:
    app = QApplication.instance() or QApplication([])
    window = LCPropMainWindow(slurm_runner=_runner(output / "artifacts"))
    window.grid_panel.Nx.setValue(16)
    window.grid_panel.Ny.setValue(16)
    window.grid_panel.dz_um.setValue(10.0)
    window.grid_panel.z_length_um.setValue(20.0)
    window.solver_panel.max_iterations.setValue(2)
    request = window.build_request()
    local = LocalRunner((LC_STATIC_OPERATION,)).run_registered(
        "lc", "static", request
    )
    window.execution_target_selector.setCurrentIndex(1)
    window.run_static_clicked()
    deadline = time.monotonic() + 900.0
    while window._background_running:
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("LC GUI Slurm smoke timed out")
        time.sleep(0.02)
    app.processEvents()
    remote = window.last_static_result
    if remote is None:
        raise RuntimeError("LC GUI did not retain a remote static result")
    metrics = {
        "case": "lc_static_cpu_gui_slurm",
        "source_sha": SOURCE_SHA,
        "status": remote.status,
        "a_relative_l2": float(
            np.linalg.norm(remote.A_final - local.result.A_final)
            / max(np.linalg.norm(local.result.A_final), 1e-30)
        ),
        "theta_relative_l2": float(
            np.linalg.norm(remote.theta_final - local.result.theta_final)
            / max(np.linalg.norm(local.result.theta_final), 1e-30)
        ),
        "run_data_presented": (
            window.results_panel.workspace.image_pane._run_data is not None
        ),
        "runner_label": window.runner_label.text(),
        "console": window.results_panel.workspace.console.toPlainText(),
    }
    window.shutdown_background_run()
    window.close()
    return metrics


def run_pr(output: Path) -> dict:
    app = QApplication.instance() or QApplication([])
    window = PRMainWindow(slurm_runner=_runner(output / "artifacts"))
    window.evolution_panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    window.grid_panel.Nx.setValue(64)
    window.grid_panel.Ny.setValue(64)
    window.grid_panel.dz_um.setValue(5.0)
    window.grid_panel.z_length_um.setValue(5.0)
    window.evolution_panel.max_coupled_passes.setValue(8)
    window.evolution_panel.backend.setCurrentText("cupy")
    window.evolution_panel.precision.setCurrentText("float64")
    request = window.build_request()
    local_request = replace(
        request,
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )
    local = LocalRunner((PR_TRANSVERSE_STATIC_OPERATION,)).run_registered(
        "pr", PR_TRANSVERSE_STATIC_WORKFLOW, local_request
    )
    window.execution_target_selector.setCurrentIndex(1)
    window.run_clicked()
    deadline = time.monotonic() + 1800.0
    while window._background_running:
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("PR GUI Slurm smoke timed out")
        time.sleep(0.02)
    app.processEvents()
    remote = window.last_result
    status = window.last_remote_status
    if remote is None or status is None:
        raise RuntimeError("PR GUI did not retain remote result/status")
    metrics = {
        "case": "pr_transverse_static_h200_gui_slurm",
        "source_sha": SOURCE_SHA,
        "job_id": status.remote_job_id,
        "requested_backend": status.scientific_backend_requested,
        "resolved_backend": status.scientific_backend_resolved,
        "device_summary": status.device_summary,
        "status": remote.status,
        "authoritative_residual_rms": remote.diagnostics[
            "equilibrium_residual_rms"
        ],
        "authoritative_residual_max": remote.diagnostics[
            "equilibrium_residual_max"
        ],
        "a_relative_l2": float(
            np.linalg.norm(remote.A_final - local.result.A_final)
            / max(np.linalg.norm(local.result.A_final), 1e-30)
        ),
        "psi_relative_l2": float(
            np.linalg.norm(remote.psi_final - local.result.psi_final)
            / max(np.linalg.norm(local.result.psi_final), 1e-30)
        ),
        "power_drift": float(remote.power_final / remote.power_initial - 1.0),
        "run_data_presented": (
            window.results_panel.workspace.image_pane._run_data is not None
        ),
        "console": window.results_panel.workspace.console.toPlainText(),
    }
    window.shutdown_background_run()
    window.close()
    return metrics


def _seconds_between(left, right):
    if not left or not right:
        return None
    return (datetime.fromisoformat(right) - datetime.fromisoformat(left)).total_seconds()


def run_coherent(output: Path) -> dict:
    app = QApplication.instance() or QApplication([])
    window = PRMainWindow(slurm_runner=_runner(output / "artifacts"))
    request = coherent_fixture(
        n=512, coherent=True, backend="cupy", precision="float64"
    )
    summary = window.describe_request(request)
    started = time.monotonic()
    window.execution_target_selector.setCurrentIndex(1)
    window._start_background(
        request,
        summary=summary,
        runner_callable=window._run_registered,
        run_label="Running",
    )
    deadline = time.monotonic() + 1800.0
    while window._background_running:
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("coherent PR GUI Slurm case timed out")
        time.sleep(0.02)
    app.processEvents()
    wall = time.monotonic() - started
    remote = window.last_result
    status = window.last_remote_status
    if remote is None or status is None:
        raise RuntimeError("coherent PR GUI did not retain remote result/status")
    diagnostics = remote.diagnostics
    history = window.remote_status_history
    retrieving = next(
        (item for item in history if item.state.value == "retrieving"), None
    )
    metrics = {
        "case": "pr_transverse_static_coherent_512_h200_gui_slurm",
        "source_sha": SOURCE_SHA,
        "job_id": status.remote_job_id,
        "requested_backend": status.scientific_backend_requested,
        "resolved_backend": status.scientific_backend_resolved,
        "device_summary": status.device_summary,
        "status": remote.status,
        "converged": remote.converged,
        "wall_seconds": wall,
        "post_scheduler_seconds": _seconds_between(
            None if retrieving is None else retrieving.retrieval_started_at,
            status.completed_at,
        ),
        "authoritative_residual_rms": diagnostics["equilibrium_residual_rms"],
        "authoritative_residual_max": diagnostics["equilibrium_residual_max"],
        "carrier_minimum": diagnostics["carrier_minimum"],
        "continuation": {
            key: diagnostics.get(key) for key in (
                "continuation_used", "continuation_succeeded",
                "continuation_cancelled", "final_visibility",
                "direct_attempt_status", "direct_attempt_termination_reason",
                "returned_state_source",
            )
        },
        "continuation_stages": diagnostics.get("continuation_stages"),
        "replay": remote.replay_diagnostics,
        "power_drift": float(remote.power_final / remote.power_initial - 1.0),
        "diagnostic_subset": {
            key: diagnostics.get(key) for key in (
                "potential_mean", "potential_mean_max_abs",
                "curl_rms", "curl_max", "gauss_rms", "gauss_max",
            )
        },
        "remote_state_sequence": [item.state.value for item in history],
        "run_data_presented": (
            window.results_panel.workspace.image_pane._run_data is not None
        ),
        "console": window.results_panel.workspace.console.toPlainText(),
    }
    window.shutdown_background_run()
    window.close()
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("lc", "pr", "coherent"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metrics = (
        run_lc(args.output)
        if args.case == "lc"
        else run_pr(args.output)
        if args.case == "pr"
        else run_coherent(args.output)
    )
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

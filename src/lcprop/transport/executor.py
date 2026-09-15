"""Scheduler-neutral headless executor for one prepared transport run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import traceback
import uuid

from lcprop.runners.local import LocalRunner
from lcprop.core.execution import RunProgress
from lcprop.transport.defaults import default_transport_operations, default_transport_registry
from lcprop.transport.envelopes import FailureEnvelope
from lcprop.transport.io import read_request_package, write_failure_package, write_result_package


def _provenance() -> dict[str, str]:
    return {
        "executor": "lcprop.transport.executor",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_progress(run_dir: Path, progress: RunProgress | dict) -> bool:
    """Best-effort atomic scheduler telemetry, isolated from run outcomes."""

    temporary: Path | None = None
    try:
        if isinstance(progress, RunProgress):
            diagnostics = progress.diagnostics or {}
            payload = {
                "schema_version": 1,
                "workflow": progress.workflow,
                "status": progress.status,
                "phase": diagnostics.get("phase", "running"),
                "message": progress.message,
                "completed_units": int(progress.completed_units),
                "total_units": int(progress.total_units),
                "current_coordinate": float(progress.current_coordinate),
                "coordinate_name": progress.coordinate_name,
                "coordinate_unit": progress.coordinate_unit,
                "elapsed_wall_time": float(progress.elapsed_wall_time),
            }
        else:
            payload = dict(progress)
            payload.setdefault("schema_version", 1)
        target = run_dir / "progress.json"
        temporary = run_dir / f"progress.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    except Exception:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass
        return False
    return True


def execute_run_directory(run_directory: str | Path, *, registry=None, operations=None) -> int:
    run_dir = Path(run_directory)
    registry = default_transport_registry() if registry is None else registry
    operations = default_transport_operations() if operations is None else tuple(operations)
    decoded = None
    temporary = run_dir / f"output.incomplete.{uuid.uuid4().hex}"
    try:
        _write_progress(run_dir, {
            "status": "running",
            "phase": "initialization",
            "message": "Running — initialization",
        })
        decoded = read_request_package(run_dir, registry=registry)
        runner = LocalRunner(operations)
        completed = runner.run_registered(
            decoded.envelope.material_id,
            decoded.envelope.workflow_id,
            decoded.request,
            _prepare_products=False,
            progress_callback=lambda progress: _write_progress(run_dir, progress),
        )
        _write_progress(run_dir, {
            "status": "running",
            "phase": "packaging",
            "message": (
                "Packaging Fast result"
                if decoded.envelope.result_policy == "fast"
                else "Packaging Full result"
            ),
        })
        write_result_package(
            run_dir,
            codec=decoded.codec,
            result=completed.result,
            request_envelope=decoded.envelope,
            provenance=_provenance(),
            output_directory=temporary,
        )
        temporary.rename(run_dir / "output")
        _write_progress(run_dir, {
            "status": "completed",
            "phase": "completed",
            "message": "Completed",
        })
        return 0
    except Exception as exc:
        _write_progress(run_dir, {
            "status": "failed",
            "phase": "failed",
            "message": str(exc) or type(exc).__name__,
        })
        run_id = "unknown-run" if decoded is None else decoded.envelope.run_id
        material_id = "unknown" if decoded is None else decoded.envelope.material_id
        workflow_id = "unknown" if decoded is None else decoded.envelope.workflow_id
        failure = FailureEnvelope(
            run_id=run_id,
            material_id=material_id,
            workflow_id=workflow_id,
            failure_category="scientific_process_failed",
            message=str(exc) or type(exc).__name__,
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),
            provenance=_provenance(),
        )
        failure_directory = run_dir / f"output.incomplete.{uuid.uuid4().hex}"
        write_failure_package(run_dir, failure, output_directory=failure_directory)
        failure_directory.rename(run_dir / "output")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    return execute_run_directory(args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())

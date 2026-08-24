"""Scheduler-neutral headless executor for one prepared transport run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import traceback
import uuid

from lcprop.runners.local import LocalRunner
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


def execute_run_directory(run_directory: str | Path, *, registry=None, operations=None) -> int:
    run_dir = Path(run_directory)
    registry = default_transport_registry() if registry is None else registry
    operations = default_transport_operations() if operations is None else tuple(operations)
    decoded = None
    temporary = run_dir / f"output.incomplete.{uuid.uuid4().hex}"
    try:
        decoded = read_request_package(run_dir, registry=registry)
        runner = LocalRunner(operations)
        completed = runner.run_registered(
            decoded.envelope.material_id,
            decoded.envelope.workflow_id,
            decoded.request,
            _prepare_products=False,
        )
        write_result_package(
            run_dir,
            codec=decoded.codec,
            result=completed.result,
            request_envelope=decoded.envelope,
            provenance=_provenance(),
            output_directory=temporary,
        )
        temporary.rename(run_dir / "output")
        return 0
    except Exception as exc:
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

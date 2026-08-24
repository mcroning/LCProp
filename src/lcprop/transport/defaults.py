"""Explicit application composition for the initially supported transports."""

import os
from pathlib import Path

from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.transport_codec import LC_STATIC_TRANSPORT_CODEC
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.transport_codec import PR_TRANSVERSE_STATIC_TRANSPORT_CODEC
from lcprop.transport.codecs import TransportCodecRegistry


def default_transport_registry() -> TransportCodecRegistry:
    registry = TransportCodecRegistry()
    registry.register(LC_STATIC_TRANSPORT_CODEC)
    registry.register(PR_TRANSVERSE_STATIC_TRANSPORT_CODEC)
    return registry


def default_transport_operations():
    return (LC_STATIC_OPERATION, PR_TRANSVERSE_STATIC_OPERATION)


def default_slurm_runner_from_environment():
    """Compose the supported remote operations when cluster settings exist."""

    source = os.environ.get("LCPROP_SLURM_SOURCE_PATH")
    sha = os.environ.get("LCPROP_SLURM_SOURCE_SHA")
    if not source or not sha:
        return None
    from lcprop.runners.slurm import SlurmExecutionConfig, SlurmRunner

    config = SlurmExecutionConfig(
        host=os.environ.get(
            "LCPROP_SLURM_HOST", "mcroning@login-p03.pax.tufts.edu"
        ),
        remote_run_root=os.environ.get(
            "LCPROP_SLURM_RUN_ROOT",
            "/cluster/tufts/cglab/mcroning/lcprop_runs",
        ),
        remote_python=os.environ.get(
            "LCPROP_SLURM_PYTHON",
            "/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python",
        ),
        remote_source_path=source,
        source_git_sha=sha,
        local_artifact_root=Path(
            os.environ.get(
                "LCPROP_SLURM_LOCAL_ARTIFACT_ROOT",
                str(Path.home() / "LCProp-results" / "remote"),
            )
        ),
    )
    return SlurmRunner(
        config,
        default_transport_operations(),
        registry=default_transport_registry(),
    )


__all__ = [
    "default_slurm_runner_from_environment",
    "default_transport_operations",
    "default_transport_registry",
]

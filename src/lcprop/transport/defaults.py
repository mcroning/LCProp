"""Explicit application composition for the initially supported transports."""

import os
from pathlib import Path
from typing import Mapping

from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.transport_codec import LC_STATIC_TRANSPORT_CODEC
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.transport_codec import PR_TRANSVERSE_STATIC_TRANSPORT_CODEC
from lcprop.runners.cluster_profiles import (
    ClusterCatalog,
    ClusterConfigError,
    ClusterProfile,
    load_cluster_profiles,
    select_cluster_profile,
)
from lcprop.transport.codecs import TransportCodecRegistry


def default_transport_registry() -> TransportCodecRegistry:
    registry = TransportCodecRegistry()
    registry.register(LC_STATIC_TRANSPORT_CODEC)
    registry.register(PR_TRANSVERSE_STATIC_TRANSPORT_CODEC)
    return registry


def default_transport_operations():
    return (LC_STATIC_OPERATION, PR_TRANSVERSE_STATIC_OPERATION)


def make_slurm_runner(
    *,
    cluster: ClusterProfile,
    remote_source_path: str,
    source_git_sha: str,
    local_artifact_root: Path | None = None,
):
    """Compose one runner from explicit site and pre-staged source inputs."""

    from lcprop.runners.slurm import SlurmExecutionConfig, SlurmRunner

    config = SlurmExecutionConfig(
        host=cluster.host,
        remote_run_root=cluster.remote_run_root,
        remote_python=cluster.remote_python,
        remote_source_path=remote_source_path,
        source_git_sha=source_git_sha,
        local_artifact_root=(
            Path.home() / "LCProp-results" / "remote"
            if local_artifact_root is None
            else Path(local_artifact_root)
        ),
        resource_profiles=cluster.resource_profiles,
        default_resource_profile=cluster.default_resource_profile,
        poll_interval=cluster.poll_interval,
    )
    return SlurmRunner(
        config,
        default_transport_operations(),
        registry=default_transport_registry(),
    )


def default_slurm_runner_from_environment(
    *,
    catalog: ClusterCatalog | None = None,
    cluster_name: str | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Compose configured Slurm execution while source staging remains explicit."""

    env = os.environ if environ is None else environ
    if catalog is None:
        profiles = load_cluster_profiles(environ=env)
        cluster = select_cluster_profile(profiles, cluster_name, environ=env)
    else:
        profiles = catalog
        cluster = select_cluster_profile(profiles, cluster_name, environ={})
    if cluster is None:
        return None
    source = env.get("LCPROP_SLURM_SOURCE_PATH")
    sha = env.get("LCPROP_SLURM_SOURCE_SHA")
    if bool(source) != bool(sha):
        raise ClusterConfigError(
            "LCPROP_SLURM_SOURCE_PATH and LCPROP_SLURM_SOURCE_SHA must be set together"
        )
    if not source or not sha:
        return None
    local_root = Path(
        env.get(
            "LCPROP_SLURM_LOCAL_ARTIFACT_ROOT",
            str(Path.home() / "LCProp-results" / "remote"),
        )
    )
    return make_slurm_runner(
        cluster=cluster,
        remote_source_path=source,
        source_git_sha=sha,
        local_artifact_root=local_root,
    )


__all__ = [
    "default_slurm_runner_from_environment",
    "default_transport_operations",
    "default_transport_registry",
    "make_slurm_runner",
]

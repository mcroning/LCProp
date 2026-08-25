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
    remote_source_path: str | None = None,
    source_git_sha: str | None = None,
    local_artifact_root: Path | None = None,
    local_source: Path | None = None,
    transport=None,
    source_deployment_manager=None,
):
    """Compose one runner with automatic deployment or an explicit override."""

    from lcprop.runners.slurm import SlurmExecutionConfig, SlurmRunner
    from lcprop.runners.source_deployment import SourceDeploymentManager

    if bool(remote_source_path) != bool(source_git_sha):
        raise ClusterConfigError(
            "remote_source_path and source_git_sha must be supplied together"
        )
    if remote_source_path is None and source_deployment_manager is None:
        source_deployment_manager = SourceDeploymentManager(
            host=cluster.host,
            source_root=cluster.source_root,
            local_source=(
                Path(__file__).resolve().parents[3]
                if local_source is None
                else Path(local_source)
            ),
            transport=transport,
        )

    config = SlurmExecutionConfig(
        host=cluster.host,
        remote_run_root=cluster.remote_run_root,
        remote_python=cluster.remote_python,
        remote_source_path=remote_source_path,
        source_git_sha=source_git_sha,
        cluster_profile=cluster.name,
        local_artifact_root=(
            Path.home() / "LCProp-results" / "remote"
            if local_artifact_root is None
            else Path(local_artifact_root)
        ),
        resource_profiles=cluster.resource_profiles,
        default_resource_profile=cluster.default_resource_profile,
        poll_interval=cluster.poll_interval,
        cleanup_remote_on_success=cluster.cleanup_remote_on_success,
    )
    return SlurmRunner(
        config,
        default_transport_operations(),
        transport=transport,
        source_deployment_manager=source_deployment_manager,
        registry=default_transport_registry(),
    )


def default_slurm_runner_from_environment(
    *,
    catalog: ClusterCatalog | None = None,
    cluster_name: str | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Compose configured Slurm execution with automatic committed-source staging."""

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

"""Portable user-local configuration for LCProp Slurm execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from lcprop.runners.slurm import SlurmResourceProfile, validate_remote_path


_CLUSTER_FIELDS = {
    "host",
    "remote_run_root",
    "remote_python",
    "source_root",
    "poll_interval",
    "cleanup_remote_on_success",
    "default_resource_profile",
    "profiles",
}
_PROFILE_FIELDS = {
    "partition",
    "qos",
    "time_limit",
    "cpus",
    "memory_gb",
    "gpus",
    "gres",
    "setup_commands",
    "require_cupy",
    "minimum_device_count",
    "expected_device_pattern",
}
_SECRET_FIELD_PARTS = {
    "password",
    "passphrase",
    "token",
    "secret",
    "private_key",
    "identity_file",
}


class ClusterConfigError(ValueError):
    """One actionable user-local cluster configuration error."""


def compose_ssh_host(username: str, login_host: str) -> str:
    """Compose the model's credential-free ``user@host`` SSH endpoint."""

    username = username.strip()
    login_host = login_host.strip()
    if username and not re.fullmatch(r"[A-Za-z0-9_.-]+", username):
        raise ClusterConfigError("SSH username contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", login_host):
        raise ClusterConfigError("login host must be a hostname without a username")
    return f"{username}@{login_host}" if username else login_host


def split_ssh_host(host: str) -> tuple[str, str]:
    """Split a validated SSH endpoint into user-facing username and host fields."""

    if "@" not in host:
        return "", host
    return tuple(host.split("@", 1))  # type: ignore[return-value]


def default_cluster_config_path(
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the platform-appropriate default cluster configuration path."""

    env = os.environ if environ is None else environ
    explicit = env.get("LCPROP_CLUSTER_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    if sys.platform == "win32":
        base = Path(env.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "LCProp" / "clusters.toml"
    if sys.platform == "darwin":
        return (
            Path.home() / "Library" / "Application Support" / "LCProp" / "clusters.toml"
        )
    base = Path(env.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "lcprop" / "clusters.toml"


def _remote_path(value: str, *, context: str, field: str) -> str:
    if not isinstance(value, str):
        raise ClusterConfigError(f"{context}.{field}: must be a string")
    try:
        return validate_remote_path(value)
    except ValueError as exc:
        raise ClusterConfigError(f"{context}.{field}: {exc}") from exc


def _required_string(values: dict, field: str, *, context: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ClusterConfigError(f"{context}.{field}: must be a non-empty string")
    return value


def _integer(
    values: dict,
    field: str,
    *,
    context: str,
    default: int | None = None,
    minimum: int = 0,
) -> int | None:
    value = values.get(field, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ClusterConfigError(f"{context}.{field}: must be an integer >= {minimum}")
    return value


def _boolean(values: dict, field: str, *, context: str, default: bool) -> bool:
    value = values.get(field, default)
    if not isinstance(value, bool):
        raise ClusterConfigError(f"{context}.{field}: must be a boolean")
    return value


def _reject_unknown(values: dict, allowed: set[str], *, context: str) -> None:
    for key in values:
        lowered = str(key).lower()
        if any(part in lowered for part in _SECRET_FIELD_PARTS):
            raise ClusterConfigError(
                f"{context}.{key}: credentials and secrets must not be stored here"
            )
        if key not in allowed:
            raise ClusterConfigError(f"{context}.{key}: unknown field")


@dataclass(frozen=True)
class ClusterProfile:
    """One named SSH/Slurm site and its material-neutral resource profiles."""

    name: str
    host: str
    remote_run_root: str
    remote_python: str
    source_root: str
    resource_profiles: tuple[SlurmResourceProfile, ...]
    poll_interval: float = 5.0
    default_resource_profile: str | None = None
    cleanup_remote_on_success: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.name):
            raise ValueError(
                "cluster name must contain only letters, digits, '.', '_', '-'"
            )
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", self.host):
            raise ValueError("host must be an SSH hostname or user@hostname")
        for field in ("remote_run_root", "remote_python", "source_root"):
            try:
                validate_remote_path(getattr(self, field))
            except ValueError as exc:
                raise ValueError(f"{field}: {exc}") from exc
        if isinstance(self.poll_interval, bool) or self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if not isinstance(self.cleanup_remote_on_success, bool):
            raise ValueError("cleanup_remote_on_success must be boolean")
        names = tuple(profile.name for profile in self.resource_profiles)
        if not names:
            raise ValueError("at least one resource profile is required")
        if len(set(names)) != len(names):
            raise ValueError("resource profile names must be unique")
        if (
            self.default_resource_profile is not None
            and self.default_resource_profile not in names
        ):
            raise ValueError("default_resource_profile is not registered")

    def profile(self, name: str) -> SlurmResourceProfile:
        matches = [value for value in self.resource_profiles if value.name == name]
        if len(matches) != 1:
            raise KeyError(
                f"unknown resource profile {name!r} on cluster {self.name!r}"
            )
        return matches[0]


@dataclass(frozen=True)
class ClusterCatalog:
    """One validated user-local multi-cluster catalog."""

    clusters: tuple[ClusterProfile, ...] = ()
    default_cluster: str | None = None
    config_path: Path | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        names = tuple(cluster.name for cluster in self.clusters)
        if len(set(names)) != len(names):
            raise ValueError("cluster names must be unique")
        if self.default_cluster is not None and self.default_cluster not in names:
            raise ValueError("default_cluster is not registered")

    def __bool__(self) -> bool:
        return bool(self.clusters)

    def __getitem__(self, name: str) -> ClusterProfile:
        matches = [value for value in self.clusters if value.name == name]
        if len(matches) != 1:
            raise KeyError(f"unknown cluster {name!r}")
        return matches[0]


def _parse_resource_profile(
    name: str,
    values: object,
    *,
    context: str,
) -> SlurmResourceProfile:
    if not isinstance(values, dict):
        raise ClusterConfigError(f"{context}: must be a TOML table")
    _reject_unknown(values, _PROFILE_FIELDS, context=context)
    setup = values.get("setup_commands", [])
    if not isinstance(setup, list) or not all(
        isinstance(value, str) and value.strip() for value in setup
    ):
        raise ClusterConfigError(
            f"{context}.setup_commands: must be an array of non-empty strings"
        )
    gres = values.get("gres")
    if gres is not None and (not isinstance(gres, str) or not gres.strip()):
        raise ClusterConfigError(f"{context}.gres: must be a non-empty string")
    pattern = values.get("expected_device_pattern")
    if pattern is not None:
        if not isinstance(pattern, str) or not pattern:
            raise ClusterConfigError(
                f"{context}.expected_device_pattern: must be a non-empty string"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ClusterConfigError(
                f"{context}.expected_device_pattern: invalid regular expression: {exc}"
            ) from exc
    try:
        return SlurmResourceProfile(
            name=name,
            partition=_required_string(values, "partition", context=context),
            qos=_required_string(values, "qos", context=context),
            time_limit=_required_string(values, "time_limit", context=context),
            cpus=int(_integer(values, "cpus", context=context, minimum=1)),
            memory_gb=int(_integer(values, "memory_gb", context=context, minimum=1)),
            gpus=int(_integer(values, "gpus", context=context, default=0)),
            gres=gres,
            setup_commands=tuple(setup),
            require_cupy=_boolean(
                values, "require_cupy", context=context, default=False
            ),
            minimum_device_count=_integer(
                values,
                "minimum_device_count",
                context=context,
                default=None,
                minimum=1,
            ),
            expected_device_pattern=pattern,
        )
    except (TypeError, ValueError) as exc:
        raise ClusterConfigError(f"{context}: {exc}") from exc


def _parse_cluster(name: str, values: object, *, path: Path) -> ClusterProfile:
    context = f"{path}: clusters.{name}"
    if not isinstance(values, dict):
        raise ClusterConfigError(f"{context}: must be a TOML table")
    _reject_unknown(values, _CLUSTER_FIELDS, context=context)
    profiles = values.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ClusterConfigError(f"{context}.profiles: must be a non-empty table")
    profile_values = tuple(
        _parse_resource_profile(
            profile_name,
            profile,
            context=f"{context}.profiles.{profile_name}",
        )
        for profile_name, profile in profiles.items()
    )
    poll = values.get("poll_interval", 5.0)
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or poll <= 0:
        raise ClusterConfigError(f"{context}.poll_interval: must be positive")
    default_profile = values.get("default_resource_profile")
    if default_profile is not None and not isinstance(default_profile, str):
        raise ClusterConfigError(
            f"{context}.default_resource_profile: must be a string"
        )
    try:
        return ClusterProfile(
            name=name,
            host=_required_string(values, "host", context=context),
            remote_run_root=_remote_path(
                values.get("remote_run_root"),
                context=context,
                field="remote_run_root",
            ),
            remote_python=_remote_path(
                values.get("remote_python"),
                context=context,
                field="remote_python",
            ),
            source_root=_remote_path(
                values.get("source_root"), context=context, field="source_root"
            ),
            resource_profiles=profile_values,
            poll_interval=float(poll),
            default_resource_profile=default_profile,
            cleanup_remote_on_success=_boolean(
                values,
                "cleanup_remote_on_success",
                context=context,
                default=True,
            ),
        )
    except ValueError as exc:
        raise ClusterConfigError(f"{context}: {exc}") from exc


def load_cluster_profiles(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ClusterCatalog:
    """Load one strict catalog; an absent file is a safe empty catalog."""

    env = os.environ if environ is None else environ
    resolved = (
        Path(path).expanduser()
        if path is not None
        else default_cluster_config_path(environ=env)
    )
    if not resolved.exists():
        return ClusterCatalog(config_path=resolved)
    try:
        with resolved.open("rb") as stream:
            values = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ClusterConfigError(f"{resolved}: cannot load TOML: {exc}") from exc
    if not isinstance(values, dict):
        raise ClusterConfigError(f"{resolved}: top level must be a TOML table")
    _reject_unknown(
        values, {"schema_version", "default_cluster", "clusters"}, context=str(resolved)
    )
    version = values.get("schema_version", 1)
    if isinstance(version, bool) or version != 1:
        raise ClusterConfigError(f"{resolved}: unsupported schema_version {version!r}")
    raw_clusters = values.get("clusters", {})
    if not isinstance(raw_clusters, dict):
        raise ClusterConfigError(f"{resolved}: clusters must be a TOML table")
    default_cluster = values.get("default_cluster")
    if default_cluster is not None and not isinstance(default_cluster, str):
        raise ClusterConfigError(f"{resolved}: default_cluster must be a string")
    try:
        return ClusterCatalog(
            clusters=tuple(
                _parse_cluster(name, cluster, path=resolved)
                for name, cluster in raw_clusters.items()
            ),
            default_cluster=default_cluster,
            config_path=resolved,
            schema_version=version,
        )
    except ValueError as exc:
        raise ClusterConfigError(f"{resolved}: {exc}") from exc


def select_cluster_profile(
    catalog: ClusterCatalog,
    name: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ClusterProfile | None:
    """Select a cluster and apply documented environment field overrides."""

    env = os.environ if environ is None else environ
    selected_name = name or env.get("LCPROP_CLUSTER") or catalog.default_cluster
    if selected_name is None:
        if len(catalog.clusters) == 1:
            selected_name = catalog.clusters[0].name
        else:
            return None
    try:
        cluster = catalog[selected_name]
    except KeyError as exc:
        raise ClusterConfigError(
            f"{catalog.config_path}: selected cluster {selected_name!r} is not configured"
        ) from exc
    overrides: dict = {}
    env_fields = {
        "LCPROP_SLURM_HOST": "host",
        "LCPROP_SLURM_RUN_ROOT": "remote_run_root",
        "LCPROP_SLURM_PYTHON": "remote_python",
        "LCPROP_SLURM_SOURCE_ROOT": "source_root",
        "LCPROP_RESOURCE_PROFILE": "default_resource_profile",
    }
    for env_name, field in env_fields.items():
        if env.get(env_name):
            overrides[field] = env[env_name]
    if env.get("LCPROP_SLURM_POLL_INTERVAL"):
        try:
            overrides["poll_interval"] = float(env["LCPROP_SLURM_POLL_INTERVAL"])
        except ValueError as exc:
            raise ClusterConfigError(
                "LCPROP_SLURM_POLL_INTERVAL: must be a positive number"
            ) from exc
    try:
        return replace(cluster, **overrides)
    except ValueError as exc:
        raise ClusterConfigError(f"environment override: {exc}") from exc


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _catalog_toml(catalog: ClusterCatalog) -> str:
    """Serialize the deliberately small version-1 catalog deterministically."""

    lines = [f"schema_version = {catalog.schema_version}"]
    if catalog.default_cluster is not None:
        lines.append(f"default_cluster = {_toml_string(catalog.default_cluster)}")
    for cluster in catalog.clusters:
        prefix = f"clusters.{cluster.name}"
        lines.extend(
            [
                "",
                f"[{prefix}]",
                f"host = {_toml_string(cluster.host)}",
                f"remote_run_root = {_toml_string(cluster.remote_run_root)}",
                f"remote_python = {_toml_string(cluster.remote_python)}",
                f"source_root = {_toml_string(cluster.source_root)}",
                f"poll_interval = {cluster.poll_interval!r}",
                "cleanup_remote_on_success = "
                + ("true" if cluster.cleanup_remote_on_success else "false"),
            ]
        )
        if cluster.default_resource_profile is not None:
            lines.append(
                "default_resource_profile = "
                + _toml_string(cluster.default_resource_profile)
            )
        for profile in cluster.resource_profiles:
            lines.extend(
                [
                    "",
                    f"[{prefix}.profiles.{_toml_string(profile.name)}]",
                    f"partition = {_toml_string(profile.partition)}",
                    f"qos = {_toml_string(profile.qos)}",
                    f"time_limit = {_toml_string(profile.time_limit)}",
                    f"cpus = {profile.cpus}",
                    f"memory_gb = {profile.memory_gb}",
                    f"gpus = {profile.gpus}",
                ]
            )
            if profile.gres is not None:
                lines.append(f"gres = {_toml_string(profile.gres)}")
            if profile.setup_commands:
                commands = ", ".join(_toml_string(v) for v in profile.setup_commands)
                lines.append(f"setup_commands = [{commands}]")
            if profile.require_cupy:
                lines.append("require_cupy = true")
            if profile.minimum_device_count is not None:
                lines.append(f"minimum_device_count = {profile.minimum_device_count}")
            if profile.expected_device_pattern is not None:
                lines.append(
                    "expected_device_pattern = "
                    + _toml_string(profile.expected_device_pattern)
                )
    return "\n".join(lines) + "\n"


def write_cluster_profiles(
    catalog: ClusterCatalog,
    path: str | Path | None = None,
) -> ClusterCatalog:
    """Atomically validate and replace one user-local cluster catalog."""

    resolved = Path(path or catalog.config_path or default_cluster_config_path())
    resolved = resolved.expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = _catalog_toml(replace(catalog, config_path=resolved))
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=resolved.parent,
            prefix=f".{resolved.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        validated = load_cluster_profiles(temporary)
        os.replace(temporary, resolved)
        temporary = None
        return replace(validated, config_path=resolved)
    except (OSError, ClusterConfigError) as exc:
        if isinstance(exc, ClusterConfigError):
            raise
        raise ClusterConfigError(f"{resolved}: cannot save TOML: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def upsert_cluster_profile(
    catalog: ClusterCatalog,
    cluster: ClusterProfile,
    *,
    make_default: bool = False,
) -> ClusterCatalog:
    """Return a catalog with one cluster inserted or replaced in place."""

    values = list(catalog.clusters)
    for index, existing in enumerate(values):
        if existing.name == cluster.name:
            values[index] = cluster
            break
    else:
        values.append(cluster)
    default = (
        cluster.name
        if make_default or catalog.default_cluster is None
        else catalog.default_cluster
    )
    return replace(catalog, clusters=tuple(values), default_cluster=default)


def delete_cluster_profile(
    catalog: ClusterCatalog,
    name: str,
) -> ClusterCatalog:
    """Return a catalog without one named cluster."""

    values = tuple(cluster for cluster in catalog.clusters if cluster.name != name)
    if len(values) == len(catalog.clusters):
        raise KeyError(f"unknown cluster {name!r}")
    default = catalog.default_cluster
    if default == name:
        default = values[0].name if values else None
    return replace(catalog, clusters=values, default_cluster=default)


__all__ = [
    "ClusterCatalog",
    "ClusterConfigError",
    "ClusterProfile",
    "compose_ssh_host",
    "delete_cluster_profile",
    "default_cluster_config_path",
    "load_cluster_profiles",
    "select_cluster_profile",
    "split_ssh_host",
    "upsert_cluster_profile",
    "write_cluster_profiles",
]

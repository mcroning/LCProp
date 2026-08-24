from dataclasses import replace
import inspect
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lcprop.runners.cluster_profiles import (
    ClusterCatalog,
    ClusterConfigError,
    ClusterProfile,
    load_cluster_profiles,
    select_cluster_profile,
)
from lcprop.runners.slurm import SlurmResourceProfile
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.transport import defaults as transport_defaults
from lcprop.transport.defaults import (
    default_slurm_runner_from_environment,
    make_slurm_runner,
)


SHA = "3d9c20bf5f407ea13997adf58ea632e46a12c9d9"
UNSAFE_REMOTE_PATHS = (
    "/scratch/bad path",
    "/scratch/bad;command",
    "/scratch/$(command)",
    "/scratch/`command`",
    "/scratch/bad\npath",
    "/scratch/'quoted'",
    '/scratch/"quoted"',
    "/scratch/bad\\path",
    "/scratch/bad|pipe",
    "/scratch/bad>redirect",
    "/scratch/bad&command",
)


def _catalog_text(*, default_profile: str = "cpu-small") -> str:
    return f"""
schema_version = 1
default_cluster = "alpha"

[clusters.alpha]
host = "user@login.alpha.edu"
remote_run_root = "/scratch/user/runs"
remote_python = "/scratch/user/env/bin/python"
source_root = "/scratch/user/sources"
poll_interval = 2.5
default_resource_profile = "{default_profile}"

[clusters.alpha.profiles.cpu-small]
partition = "batch"
qos = "normal"
time_limit = "00:15:00"
cpus = 2
memory_gb = 8
gpus = 0

[clusters.alpha.profiles.generic-gpu]
partition = "gpu"
qos = "normal"
time_limit = "00:20:00"
cpus = 2
memory_gb = 16
gpus = 1
gres = "gpu:1"
setup_commands = ["module load cuda"]
require_cupy = true
minimum_device_count = 1

[clusters.alpha.profiles.h200]
partition = "gpu"
qos = "normal"
time_limit = "00:20:00"
cpus = 2
memory_gb = 16
gpus = 1
gres = "gpu:h200:1"
setup_commands = ["module load cuda/12.9.0"]
require_cupy = true
minimum_device_count = 1
expected_device_pattern = "H200"
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _explicit_cluster() -> ClusterProfile:
    return ClusterProfile(
        name="explicit",
        host="explicit@login.example.edu",
        remote_run_root="/explicit/runs",
        remote_python="/explicit/bin/python",
        source_root="/explicit/sources",
        resource_profiles=(
            SlurmResourceProfile(
                "cpu", "batch", "normal", "00:10:00", 1, 2
            ),
        ),
        default_resource_profile="cpu",
    )


def test_absent_configuration_returns_empty_catalog(tmp_path):
    path = tmp_path / "missing.toml"
    catalog = load_cluster_profiles(path)
    assert not catalog
    assert catalog.clusters == ()
    assert catalog.config_path == path


def test_valid_toml_loads_cluster_and_multiple_resource_profiles(tmp_path):
    path = _write(tmp_path / "clusters.toml", _catalog_text())
    catalog = load_cluster_profiles(path)
    cluster = catalog["alpha"]
    assert catalog.default_cluster == "alpha"
    assert cluster.host == "user@login.alpha.edu"
    assert cluster.poll_interval == 2.5
    assert cluster.default_resource_profile == "cpu-small"
    assert tuple(profile.name for profile in cluster.resource_profiles) == (
        "cpu-small",
        "generic-gpu",
        "h200",
    )
    assert cluster.profile("generic-gpu").gres == "gpu:1"
    assert cluster.profile("h200").expected_device_pattern == "H200"


def test_multiple_clusters_are_independently_selectable(tmp_path):
    text = _catalog_text() + """

[clusters.beta]
host = "login.beta.edu"
remote_run_root = "/work/runs"
remote_python = "/work/env/bin/python"
source_root = "/work/sources"

[clusters.beta.profiles.a100]
partition = "accelerated"
qos = "research"
time_limit = "01:00:00"
cpus = 4
memory_gb = 32
gpus = 1
gres = "gpu:a100:1"
require_cupy = true
minimum_device_count = 1
expected_device_pattern = "A100"
"""
    catalog = load_cluster_profiles(_write(tmp_path / "clusters.toml", text))
    assert tuple(cluster.name for cluster in catalog.clusters) == ("alpha", "beta")
    assert catalog["beta"].profile("a100").gres == "gpu:a100:1"


def test_invalid_default_profile_is_actionable(tmp_path):
    path = _write(
        tmp_path / "clusters.toml", _catalog_text(default_profile="missing")
    )
    with pytest.raises(ClusterConfigError) as caught:
        load_cluster_profiles(path)
    assert str(path) in str(caught.value)
    assert "default_resource_profile" in str(caught.value)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[clusters.alpha\n", "cannot load TOML"),
        (
            "schema_version=1\n[clusters.alpha]\nhost='login'\n",
            "profiles",
        ),
        (_catalog_text().replace("cpus = 2", "cpus = 0", 1), "cpus"),
        (
            _catalog_text().replace(
                'expected_device_pattern = "H200"',
                'expected_device_pattern = "["',
            ),
            "expected_device_pattern",
        ),
    ],
)
def test_malformed_or_invalid_configuration_is_actionable(
    tmp_path, text, message
):
    path = _write(tmp_path / "clusters.toml", text)
    with pytest.raises(ClusterConfigError) as caught:
        load_cluster_profiles(path)
    assert str(path) in str(caught.value)
    assert message in str(caught.value)


def test_secret_fields_are_rejected(tmp_path):
    text = _catalog_text().replace(
        'host = "user@login.alpha.edu"',
        'host = "user@login.alpha.edu"\npassword = "do-not-store"',
    )
    with pytest.raises(ClusterConfigError, match="must not be stored"):
        load_cluster_profiles(_write(tmp_path / "clusters.toml", text))


@pytest.mark.parametrize("unsafe_path", UNSAFE_REMOTE_PATHS)
@pytest.mark.parametrize(
    ("field", "original"),
    (
        ("remote_run_root", "/scratch/user/runs"),
        ("remote_python", "/scratch/user/env/bin/python"),
        ("source_root", "/scratch/user/sources"),
    ),
)
def test_toml_remote_paths_reject_shell_unsafe_values(
    tmp_path, unsafe_path, field, original
):
    toml_value = (
        unsafe_path.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )
    text = _catalog_text().replace(
        f'{field} = "{original}"',
        f'{field} = "{toml_value}"',
    )
    with pytest.raises(ClusterConfigError, match=field):
        load_cluster_profiles(_write(tmp_path / "clusters.toml", text))


@pytest.mark.parametrize("unsafe_path", UNSAFE_REMOTE_PATHS)
@pytest.mark.parametrize(
    ("environment_name", "field"),
    (
        ("LCPROP_SLURM_RUN_ROOT", "remote_run_root"),
        ("LCPROP_SLURM_PYTHON", "remote_python"),
        ("LCPROP_SLURM_SOURCE_ROOT", "source_root"),
    ),
)
def test_environment_remote_path_overrides_are_revalidated(
    tmp_path, unsafe_path, environment_name, field
):
    catalog = load_cluster_profiles(
        _write(tmp_path / "clusters.toml", _catalog_text())
    )
    with pytest.raises(ClusterConfigError, match=field):
        select_cluster_profile(
            catalog,
            environ={environment_name: unsafe_path},
        )


@pytest.mark.parametrize("unsafe_path", UNSAFE_REMOTE_PATHS)
@pytest.mark.parametrize(
    "field", ("remote_run_root", "remote_python", "source_root")
)
def test_direct_cluster_profile_rejects_shell_unsafe_paths(unsafe_path, field):
    with pytest.raises(ValueError, match=field):
        replace(_explicit_cluster(), **{field: unsafe_path})


def test_environment_overrides_selected_user_profile(tmp_path):
    catalog = load_cluster_profiles(
        _write(tmp_path / "clusters.toml", _catalog_text())
    )
    selected = select_cluster_profile(
        catalog,
        environ={
            "LCPROP_SLURM_HOST": "override@login.example.edu",
            "LCPROP_SLURM_RUN_ROOT": "/override/runs",
            "LCPROP_SLURM_POLL_INTERVAL": "7.5",
            "LCPROP_RESOURCE_PROFILE": "generic-gpu",
        },
    )
    assert selected is not None
    assert selected.host == "override@login.example.edu"
    assert selected.remote_run_root == "/override/runs"
    assert selected.poll_interval == 7.5
    assert selected.default_resource_profile == "generic-gpu"


def test_explicit_constructor_values_are_not_replaced_by_environment(tmp_path):
    cluster = _explicit_cluster()
    runner = make_slurm_runner(
        cluster=cluster,
        remote_source_path="/explicit/snapshot",
        source_git_sha=SHA,
        local_artifact_root=tmp_path,
    )
    assert runner.config.host == "explicit@login.example.edu"
    assert runner.config.remote_run_root == "/explicit/runs"
    assert runner.config.profile("cpu").partition == "batch"


def test_default_composition_has_no_personal_or_tufts_defaults(tmp_path):
    source = inspect.getsource(transport_defaults)
    assert "mcroning" not in source
    assert "tufts" not in source.lower()
    assert default_slurm_runner_from_environment(
        catalog=ClusterCatalog(config_path=tmp_path / "missing.toml"),
        environ={},
    ) is None


def test_explicit_source_pair_composes_profile_runner(tmp_path):
    catalog = ClusterCatalog(clusters=(_explicit_cluster(),), default_cluster="explicit")
    runner = default_slurm_runner_from_environment(
        catalog=catalog,
        environ={
            "LCPROP_SLURM_SOURCE_PATH": "/explicit/snapshot",
            "LCPROP_SLURM_SOURCE_SHA": SHA,
            "LCPROP_SLURM_LOCAL_ARTIFACT_ROOT": str(tmp_path),
        },
    )
    assert runner is not None
    assert runner.config.remote_source_path == "/explicit/snapshot"
    assert runner.config.source_git_sha == SHA


def test_user_config_discovery_composes_runner_without_source_edits(tmp_path):
    config_path = _write(tmp_path / "clusters.toml", _catalog_text())
    runner = default_slurm_runner_from_environment(
        environ={
            "LCPROP_CLUSTER_CONFIG": str(config_path),
            "LCPROP_SLURM_SOURCE_PATH": "/deployed/snapshot",
            "LCPROP_SLURM_SOURCE_SHA": SHA,
            "LCPROP_SLURM_LOCAL_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        }
    )
    assert runner is not None
    assert runner.config.host == "user@login.alpha.edu"
    assert runner.config.default_resource_profile == "cpu-small"
    assert runner.config.profile("generic-gpu").gres == "gpu:1"


def test_normal_profile_composition_no_longer_requires_manual_source_environment(
    tmp_path,
):
    catalog = ClusterCatalog(
        clusters=(_explicit_cluster(),), default_cluster="explicit"
    )
    runner = default_slurm_runner_from_environment(
        catalog=catalog,
        environ={"LCPROP_SLURM_LOCAL_ARTIFACT_ROOT": str(tmp_path)},
    )
    assert runner is not None
    assert runner.config.remote_source_path is None
    assert runner.config.source_git_sha is None
    assert runner._source_deployment_manager is not None


def test_explicit_source_pair_must_be_complete():
    catalog = ClusterCatalog(clusters=(_explicit_cluster(),), default_cluster="explicit")
    with pytest.raises(ClusterConfigError, match="must be set together"):
        default_slurm_runner_from_environment(
            catalog=catalog,
            environ={"LCPROP_SLURM_SOURCE_PATH": "/explicit/snapshot"},
        )


@pytest.mark.parametrize(
    ("name", "gres", "pattern"),
    [
        ("h100", "gpu:h100:1", "H100"),
        ("a100", "gpu:a100:1", "A100"),
        ("l40s", "gpu:l40s:1", "L40S"),
    ],
)
def test_device_specific_profiles_need_no_source_changes(name, gres, pattern):
    profile = SlurmResourceProfile(
        name,
        "gpu",
        "normal",
        "00:20:00",
        2,
        16,
        gpus=1,
        gres=gres,
        require_cupy=True,
        minimum_device_count=1,
        expected_device_pattern=pattern,
    )
    assert profile.gres == gres
    assert profile.expected_device_pattern == pattern


def test_gpu_profile_requires_cupy_in_toml(tmp_path):
    text = _catalog_text().replace(
        "require_cupy = true",
        "require_cupy = false",
        1,
    )
    with pytest.raises(ClusterConfigError, match="require_cupy=true"):
        load_cluster_profiles(_write(tmp_path / "clusters.toml", text))


def test_cluster_profile_environment_override_validation_is_actionable():
    catalog = ClusterCatalog(clusters=(_explicit_cluster(),), default_cluster="explicit")
    with pytest.raises(ClusterConfigError, match="POLL_INTERVAL"):
        select_cluster_profile(
            catalog, environ={"LCPROP_SLURM_POLL_INTERVAL": "not-a-number"}
        )


def test_profile_model_has_no_material_or_credential_fields():
    fields = set(ClusterProfile.__dataclass_fields__)
    assert not fields.intersection(
        {"material_id", "workflow_id", "password", "token", "private_key"}
    )


def test_material_guis_do_not_hard_code_site_resource_profile_names():
    lc_source = inspect.getsource(LCPropMainWindow._run_registered)
    pr_source = inspect.getsource(PRMainWindow._run_registered)
    assert "CPU small" not in lc_source
    assert "H200 small" not in pr_source
    assert "resource_profile" not in lc_source
    assert "resource_profile" not in pr_source

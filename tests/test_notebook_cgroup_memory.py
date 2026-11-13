"""Memory admission checks using local copies of Linux proc and cgroup data."""
from pathlib import Path

import pytest

from lcprop.notebook import batch_resources


@pytest.fixture
def proc_files(monkeypatch):
    files = {"/proc/meminfo": "MemAvailable: 11912932 kB\n"}

    def read_text(path):
        try:
            return files[str(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    monkeypatch.setattr(Path, "read_text", read_text)
    return files


def test_colab_namespace_mount(proc_files):
    proc_files.update({
        "/proc/self/cgroup": "0::/../../jupyter-children\n",
        "/proc/self/mountinfo": (
            "936 935 0:27 / /sys/fs/cgroup ro - cgroup2 cgroup rw\n"
            "945 940 0:27 /../../jupyter-children /var/colab/cgroup/jupyter-children "
            "ro - cgroup2 cgroup2 rw\n"
        ),
        "/sys/fs/cgroup/memory.max": "max",
        "/var/colab/cgroup/jupyter-children/memory.max": str(4096 * batch_resources.MIB),
        "/var/colab/cgroup/jupyter-children/memory.current": str(1024 * batch_resources.MIB),
    })
    assert batch_resources.available_memory() == 3072 * batch_resources.MIB


@pytest.mark.parametrize("version", [1, 2])
def test_memory_mount_root_and_ancestors(proc_files, version):
    if version == 1:
        group, filesystem = "5:memory:/batch/jobs/task", "cgroup cgroup rw,memory"
        limit, usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
    else:
        group, filesystem = "0::/batch/jobs/task", "cgroup2 cgroup rw"
        limit, usage = "memory.max", "memory.current"
    proc_files.update({
        "/proc/self/cgroup": group,
        "/proc/self/mountinfo": rf"7 1 0:27 /batch /some\040mount ro - {filesystem}",
        f"/some mount/{limit}": "5000",
        f"/some mount/{usage}": "4500",
        f"/some mount/jobs/{limit}": "4000",
        f"/some mount/jobs/{usage}": "100",
        f"/some mount/jobs/task/{limit}": "2000",
        f"/some mount/jobs/task/{usage}": "300",
    })
    assert batch_resources.available_memory() == 500


@pytest.mark.parametrize("group,limit,usage", [
    ("/other", "1024", "64"),
    ("/batch/../other", "1024", "64"),
    ("/batch", None, "64"),
    ("/batch", "1024", None),
    ("/batch", "invalid", "64"),
    ("/batch", "1024", "-1"),
])
def test_unknown_memory_fails_closed(proc_files, group, limit, usage):
    proc_files.update({
        "/proc/self/cgroup": f"0::{group}",
        "/proc/self/mountinfo": "7 1 0:27 /batch /mount ro - cgroup2 cgroup rw",
    })
    if limit is not None:
        proc_files["/mount/memory.max"] = limit
    if usage is not None:
        proc_files["/mount/memory.current"] = usage
    available = batch_resources.available_memory()
    assert available is None
    with pytest.raises(ValueError, match="availability is unknown"):
        batch_resources.require_work(1, 0, available)

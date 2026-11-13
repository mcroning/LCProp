"""Conservative CPU admission charges for the notebook's supported routes."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re

from lcprop.core.grid import grid_metadata

MIB = 1024**2
RESULT_LIMIT = 256 * MIB
WORK_LIMIT = 2560 * MIB
RESERVE = 512 * MIB

# Retained volumes, overlapping work volumes, packed volumes, auxiliary MiB.
CHARGES = {
    "lc": (7, 14, 6, 8, 8),
    "lc_td": (9, 18, None, 8, 8),
    "pr_static": (9, 18, 4, 24, 16),
    "pr": (9, 32, 5, 24, 32),
}


@dataclass(frozen=True)
class Charge:
    retained: int
    working: int
    export: int | None


def estimate(route, request):
    grid = grid_metadata(request.grid)
    plane = 8 * grid.Nx * grid.Ny
    volume = plane * grid.Nz
    channels = len(request.beams.channels)
    retained, work, packed, retained_misc, work_misc = CHARGES[route]
    steps = getattr(request.solver, "Nt", 1)
    passes = getattr(request.solver, "max_coupled_passes",
                     getattr(request.solver, "static_max_coupled_passes", 1))
    iterations = getattr(request.solver, "static_max_relax_iterations", 1)
    history = 512 * (steps + grid.Nz * passes * iterations)
    retained_extra = max(retained_misc * MIB, history)
    work_extra = max(work_misc * MIB, history)
    return Charge(
        retained * volume + 64 * channels * plane + retained_extra,
        2 * (work * volume + 128 * channels * plane + work_extra),
        None if packed is None else packed * volume + 64 * channels * plane + max(8 * MIB, history),
    )


def _memory_cgroup_paths(group, version, mountinfo):
    """Map a process cgroup to its visible mount and visible ancestors."""
    group = PurePosixPath(group)
    if not group.is_absolute():
        raise ValueError("Cgroup membership must be absolute")
    for line in mountinfo.splitlines():
        mount, filesystem = line.split(" - ", 1)
        kind, _, options = filesystem.split()
        if version == 1:
            if kind != "cgroup" or "memory" not in options.split(","):
                continue
        elif kind != "cgroup2":
            continue

        # Mountinfo escapes whitespace and backslashes in path fields.
        root, mountpoint = (
            PurePosixPath(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), field))
            for field in mount.split()[3:5]
        )
        if not group.is_relative_to(root):
            continue
        relative = group.relative_to(root)
        # Colab can report namespace-relative ".." in both the group and mount
        # root. Match those prefixes, but never traverse above a mountpoint.
        if ".." in relative.parts or not mountpoint.is_absolute() or ".." in mountpoint.parts:
            continue
        folder = Path(mountpoint) / relative
        while True:
            yield folder
            if folder == mountpoint:
                break
            folder = folder.parent


def available_memory():
    """Return available Linux memory constrained by the process's memory cgroup."""
    try:
        info = Path("/proc/meminfo").read_text().splitlines()
        available = next(int(line.split()[1]) * 1024 for line in info
                         if line.startswith("MemAvailable:"))
        groups = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
        v2 = next((name for _, controllers, name in groups if not controllers), None)
        v1 = next((name for _, controllers, name in groups
                   if "memory" in controllers.split(",")), None)
        if v1 is not None:
            group, version = v1, 1
            ceiling, usage = "memory.limit_in_bytes", "memory.usage_in_bytes"
        elif v2 is not None:
            group, version = v2, 2
            ceiling, usage = "memory.max", "memory.current"
        else:
            return None
        mountinfo = Path("/proc/self/mountinfo").read_text()
        folders = set(_memory_cgroup_paths(group, version, mountinfo))
        if not folders or available < 0:
            return None
        for folder in folders:
            limit = (folder / ceiling).read_text().strip()
            if version == 2 and limit == "max":
                continue
            maximum = int(limit)
            used = int((folder / usage).read_text())
            if maximum < 0 or used < 0:
                return None
            available = min(available, maximum - used)
        return max(0, available)
    except (OSError, ValueError, StopIteration, IndexError):
        return None


def require_work(new_work, resident, available):
    if available is None:
        raise ValueError("Runtime memory availability is unknown")
    if new_work + resident > WORK_LIMIT or new_work + RESERVE > available:
        raise ValueError("Working-memory budget exceeded; release an unneeded result or edit the request")

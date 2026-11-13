"""Conservative CPU admission charges for the notebook's supported routes."""
from dataclasses import dataclass
from pathlib import Path

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
            root = Path("/sys/fs/cgroup/memory")
            group, ceiling, usage = v1, "memory.limit_in_bytes", "memory.usage_in_bytes"
        elif v2 is not None:
            root = Path("/sys/fs/cgroup")
            group, ceiling, usage = v2, "memory.max", "memory.current"
        else:
            return None
        folder = root / group.lstrip("/")
        for candidate in (folder, *folder.parents):
            if not candidate.is_relative_to(root):
                break
            limit = (candidate / ceiling).read_text().strip()
            if limit != "max":
                used = int((candidate / usage).read_text())
                available = min(available, int(limit) - used)
        return max(0, available)
    except (OSError, ValueError, StopIteration, IndexError):
        return None


def require_work(new_work, resident, available):
    if available is None:
        raise ValueError("Runtime memory availability is unknown")
    if new_work + resident > WORK_LIMIT or new_work + RESERVE > available:
        raise ValueError("Working-memory budget exceeded; release an unneeded result or edit the request")

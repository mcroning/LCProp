"""Local resource advice only; never changes requests or scientific acceptance."""
import os
import subprocess
import sys


def physical_memory_bytes():
    """Best-effort physical capacity, never free RAM or an allocation probe."""
    try:
        if sys.platform == 'darwin':
            value = int(subprocess.check_output(
                ['/usr/sbin/sysctl', '-n', 'hw.memsize'], timeout=1,
                stderr=subprocess.DEVNULL).strip())
        else:
            value = int(os.sysconf('SC_PHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
        return value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def assess_local_resources(plan, shape, *, physical_ram_bytes=None):
    """Deterministic given estimator, shape and RAM; advice is not request identity.

    Summed estimator scenarios deliberately overcount overlapping buffers. A
    60%-of-physical-RAM warning reserves capacity for macOS and other programs;
    it is not a measured peak or a scientific grid ceiling. No available-memory
    sampling influences the result. Unknown capacity remains explicit.
    """
    if physical_ram_bytes is not None and physical_ram_bytes <= 0:
        raise ValueError('physical RAM must be positive or unknown')
    estimated = sum(plan['bytes'].values()) + sum(plan.get('presentation', {}).values())
    large = max(shape) > 256
    risk = physical_ram_bytes is not None and estimated >= .6 * physical_ram_bytes
    classification = 'memory-risk' if risk else 'large/slow' if large else 'comfortable'
    warnings = []
    if large:
        warnings.append('Above the 256-per-axis interactive recommendation; local runtime can be large. '
                        'Supported but expensive/not recommended interactively. '
                        'Scientific support is closure/backend/precision-specific; see the support envelope. '
                        'Bounded qualification does not guarantee long multi-plane runtime.')
    if risk:
        warnings.append('Estimated array/workspace scenarios reach 60% of physical RAM. '
                        'Close other workloads or select fewer retained products; no settings are changed.')
    if physical_ram_bytes is None:
        warnings.append('Physical RAM unavailable; memory-risk assessment is incomplete.')
    return dict(classification=classification, estimated_scenario_bytes=estimated,
                physical_ram_bytes=physical_ram_bytes, warnings=tuple(warnings),
                scope='planning only; unsupported configurations rejected by solver/backend validation')

"""Resolve notebook execution intent before admitting a bounded run."""
import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from math import isfinite
from time import monotonic

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.grid import grid_metadata
from lcprop.notebook.batch_resources import MIB


class DeviceUnavailable(RuntimeError):
    """A GPU cannot be used for the requested run."""


# Provisional policy for local fake-runner qualification only.
GPU_PROFILE = "pr-td-t4-gate2-mock-v1"
GPU_CAP = 2048 * MIB
GPU_RESERVE = 1024 * MIB
GPU_TELEMETRY_MAX_AGE_S = 5.0


@dataclass(frozen=True)
class GPUReadiness:
    profile: str
    ordinal: int
    free_bytes: int
    total_bytes: int
    sampled_at: float


@dataclass(frozen=True)
class GPUDevice:
    ordinal: int
    name: str
    bind: Callable[[], AbstractContextManager]
    readiness: Callable[[tuple[int, int]], GPUReadiness] | None = None


@dataclass(frozen=True)
class RunDevice:
    requested: str
    resolved: str
    precision: str
    gpu: GPUDevice | None = None
    fallback_reason: str | None = None
    gpu_work_bytes: int | None = None

    @property
    def label(self):
        if self.gpu is None:
            return "CPU"
        return f"GPU {self.gpu.ordinal} ({self.gpu.name})"

    def context(self):
        return nullcontext() if self.gpu is None else self.gpu.bind()

    def verify_gpu_memory(self, shape):
        if self.gpu is None:
            return
        if self.gpu.readiness is None or self.gpu_work_bytes is None:
            raise DeviceUnavailable("GPU memory readiness is unavailable. Select CPU.")
        try:
            sample = self.gpu.readiness(shape)
        except (OSError, MemoryError) as error:
            raise DeviceUnavailable(
                "GPU memory telemetry check failed. Check device readiness or select CPU."
            ) from error
        if (not isinstance(sample, GPUReadiness) or sample.profile != GPU_PROFILE
                or type(sample.ordinal) is not int or sample.ordinal != self.gpu.ordinal):
            raise DeviceUnavailable("GPU readiness profile or device identity does not match. Select CPU.")
        if (type(sample.free_bytes) is not int or type(sample.total_bytes) is not int
                or not 0 <= sample.free_bytes <= sample.total_bytes or sample.total_bytes <= 0
                or type(sample.sampled_at) not in (int, float) or not isfinite(sample.sampled_at)
                or not 0 <= monotonic() - sample.sampled_at <= GPU_TELEMETRY_MAX_AGE_S):
            raise DeviceUnavailable("GPU memory telemetry is invalid or stale. Check readiness or select CPU.")
        if self.gpu_work_bytes > GPU_CAP or self.gpu_work_bytes + GPU_RESERVE > sample.free_bytes:
            raise DeviceUnavailable("GPU memory budget is insufficient. Free device memory or select CPU.")

    def metadata(self):
        metadata = {
            "requested_backend": self.requested,
            "resolved_backend": self.resolved,
            "precision": self.precision,
            "real_dtype": self.precision,
            "complex_dtype": "complex128" if self.precision == "float64" else "complex64",
            "device_ordinal": None if self.gpu is None else self.gpu.ordinal,
            "device_name": self.label,
            "fallback_reason": self.fallback_reason,
        }
        if self.gpu_work_bytes is not None:
            metadata.update(gpu_profile=GPU_PROFILE, gpu_work_bytes=self.gpu_work_bytes)
        return metadata


def probe_cupy(spec):
    """Opt-in qualification probe. Public notebook controllers do not call it yet."""
    try:
        import cupy as cp
    except (ImportError, OSError) as error:
        raise DeviceUnavailable(f"CuPy is unavailable: {error}") from error

    device_errors = (ImportError, OSError, MemoryError)
    try:
        device_errors += (cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError)
        fft_error = getattr(getattr(cp.cuda, "cufft", None), "CuFFTError", None)
        if fft_error is not None:
            device_errors += (fft_error,)
        backend = get_backend(spec)
        device = cp.cuda.Device()
        with device:
            sample = backend.xp.zeros(8, dtype=backend.real_dtype)
            spectrum = backend.xp.fft.fft(sample.astype(backend.complex_dtype))
            backend.xp.cuda.get_current_stream().synchronize()
            if spectrum.dtype != backend.complex_dtype:
                raise DeviceUnavailable("GPU FFT did not preserve the requested precision")
            name = cp.cuda.runtime.getDeviceProperties(device.id)["name"]
    except device_errors as error:
        raise DeviceUnavailable(f"GPU probe failed: {error}") from error
    except RuntimeError as error:
        # get_backend wraps its allocation failure for explicit CuPy requests.
        if isinstance(error.__cause__, device_errors):
            raise DeviceUnavailable(f"GPU probe failed: {error}") from error
        raise
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    ordinal = device.id
    return GPUDevice(ordinal, str(name), lambda: cp.cuda.Device(ordinal))


def resolve_device(route, request, gpu_probe=None):
    if route in {"lc", "lc_td"}:
        return RunDevice("numpy", "numpy", request.runtime.precision)

    if route not in {"pr", "pr_static"}:
        raise ValueError(f"Unsupported notebook workflow: {route}")
    spec = request.backend
    spec.validate()
    if spec.precision != "float64":
        raise ValueError("The PR notebook requires float64 precision")
    if spec.backend == "numpy":
        return RunDevice("numpy", "numpy", spec.precision)
    try:
        reason = gpu_scope_reason(route, request, gpu_probe)
        if reason:
            raise DeviceUnavailable(reason)
        gpu = gpu_probe(BackendSpec("cupy", spec.precision, verbose=False))
        if (not isinstance(gpu, GPUDevice) or type(gpu.ordinal) is not int
                or gpu.ordinal < 0 or not callable(gpu.bind)):
            raise TypeError("GPU probe must return a GPUDevice with a valid ordinal and binder")
        if gpu.name not in {"Tesla T4", "NVIDIA T4"}:
            raise DeviceUnavailable("GPU requires a T4. Select CPU for this device.")
        # Scope matched the full, single-channel frozen request above. Charge
        # provisional headroom for simultaneous real volumes, optical planes
        # and opaque device workspace. These are test charges, not VRAM bounds.
        grid = request.grid
        plane = 8 * grid.Nx * grid.Ny
        work_bytes = 32 * plane * grid_metadata(grid).Nz
        work_bytes += 128 * plane + 256 * MIB
        resolved = RunDevice(spec.backend, "cupy", spec.precision, gpu=gpu,
                             gpu_work_bytes=work_bytes)
        if gpu.readiness is None:
            raise DeviceUnavailable("GPU memory readiness is unavailable. Select CPU.")
        try:
            with resolved.context():
                resolved.verify_gpu_memory((grid.Nx, grid.Ny))
        except (OSError, MemoryError) as error:
            raise DeviceUnavailable(
                "GPU device binding failed. Reconnect the GPU runtime or select CPU."
            ) from error
    except DeviceUnavailable as error:
        if spec.backend == "auto":
            return RunDevice("auto", "numpy", spec.precision, fallback_reason=str(error))
        raise
    return resolved


# Batch 4 archived 20 µm and 1 mm requests, encoded by the current experiment
# codec with only backend intent normalized to NumPy.
_TD_GPU_FIXTURES = frozenset({
    "535e0f9794203ff2562ba5f6900f09eee29c78428c4a901ca19416e2b312680c",
    "f576ac6e5ca2ee45b0338c6f69cca2e58dd460626a965afd7aef7b582d78c72f",
})


def gpu_scope_reason(route, request, gpu_probe):
    if route != "pr":
        return "GPU is restricted to the qualified reduced PR TD envelope. Select CPU for PR Static."
    from lcprop.notebook.batch_requests import ROUTES

    payload = ROUTES[route][1].encode_request(request)
    payload["backend"]["backend"] = "numpy"
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(canonical.encode()).hexdigest() not in _TD_GPU_FIXTURES:
        return "GPU is restricted to the qualified reduced PR TD envelope. Select CPU or restore a frozen fixture."
    if gpu_probe is None:
        return "GPU qualification is pending for this notebook. Select CPU."
    return ""

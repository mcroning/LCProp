"""Resolve notebook execution intent before admitting a bounded run."""
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace

from lcprop.core.backend import BackendSpec, get_backend


class DeviceUnavailable(RuntimeError):
    """A GPU cannot be used for the requested run."""


@dataclass(frozen=True)
class GPUDevice:
    ordinal: int
    name: str
    bind: Callable[[], AbstractContextManager]


@dataclass(frozen=True)
class RunDevice:
    requested: str
    resolved: str
    precision: str
    gpu: GPUDevice | None = None
    fallback_reason: str | None = None

    @property
    def label(self):
        if self.gpu is None:
            return "CPU"
        return f"GPU {self.gpu.ordinal} ({self.gpu.name})"

    def context(self):
        return nullcontext() if self.gpu is None else self.gpu.bind()

    def metadata(self):
        return {
            "requested_backend": self.requested,
            "resolved_backend": self.resolved,
            "precision": self.precision,
            "real_dtype": self.precision,
            "complex_dtype": "complex128" if self.precision == "float64" else "complex64",
            "device_ordinal": None if self.gpu is None else self.gpu.ordinal,
            "device_name": self.label,
            "fallback_reason": self.fallback_reason,
        }


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
    if spec.backend == "auto":
        return RunDevice("auto", "numpy", spec.precision)
    from lcprop.notebook.batch_requests import preset
    reference = preset("pr")
    if (route != "pr" or request.grid not in (reference.grid, replace(reference.grid, z_length_um=20.))
            or request.solver.integrator != reference.solver.integrator
            or request.solver.optical_substeps != reference.solver.optical_substeps
            or request.solver.Nt not in (1, 2, 3) or request.solver.dt_normalized > 0.001):
        raise DeviceUnavailable("GPU is restricted to the qualified reduced PR TD envelope")
    if gpu_probe is None:
        raise DeviceUnavailable("GPU qualification is pending for this notebook")
    gpu = gpu_probe(BackendSpec("cupy", spec.precision, verbose=False))
    if (not isinstance(gpu, GPUDevice) or type(gpu.ordinal) is not int
            or gpu.ordinal < 0 or not callable(gpu.bind)):
        raise TypeError("GPU probe must return a GPUDevice with a valid ordinal and binder")
    return RunDevice(spec.backend, "cupy", spec.precision, gpu=gpu)

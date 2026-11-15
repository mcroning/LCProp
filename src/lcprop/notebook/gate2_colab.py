"""Opt-in T4 readiness adapter for Gate 2 qualification with a fake runner."""

from contextlib import ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError, version
import platform
import subprocess
from time import monotonic

from lcprop.notebook.batch_device import (
    DeviceUnavailable, GPUDevice, GPUReadiness, GPU_PROFILE,
)

PYTHON_VERSION = "3.13.16"
PACKAGE_VERSIONS = {"numpy": "2.4.6", "scipy": "1.16.3", "cupy-cuda13x": "14.0.1"}
DRIVER_VERSION = "580.82.07"
CUDA_RUNTIME_VERSION = 13000


def _verify_host():
    try:
        packages_match = all(version(name) == pin for name, pin in PACKAGE_VERSIONS.items())
    except PackageNotFoundError:
        packages_match = False
    if (platform.system() != "Linux" or platform.machine() != "x86_64"
            or platform.python_version() != PYTHON_VERSION or not packages_match):
        raise DeviceUnavailable("Gate 2 software profile differs from the pinned stack. Select CPU.")
    try:
        drivers = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5,
        ).split()
    except (OSError, subprocess.SubprocessError) as error:
        raise DeviceUnavailable("Cannot read the T4 driver version. Select CPU.") from error
    if drivers != [DRIVER_VERSION]:
        raise DeviceUnavailable("Gate 2 requires the pinned driver on one visible T4. Select CPU.")


def probe_gate2_t4(spec):
    """Supply bounded readiness and fresh telemetry. Inject only in qualification."""
    spec.validate()
    if spec.backend != "cupy" or spec.precision != "float64":
        raise ValueError("Gate 2 requires explicit CuPy float64")
    _verify_host()
    try:
        import cupy as cp
        from cupy.cuda import cufft
    except (ImportError, OSError) as error:
        raise DeviceUnavailable("CuPy or cuFFT is unavailable. Select CPU.") from error

    runtime = cp.cuda.runtime
    operational = (OSError, ImportError, MemoryError, runtime.CUDARuntimeError,
                   cp.cuda.memory.OutOfMemoryError, cufft.CuFFTError)

    def verify_device():
        if runtime.getDevice() != ordinal or runtime.getDeviceProperties(ordinal)["name"] not in (
                b"Tesla T4", "Tesla T4", b"NVIDIA T4", "NVIDIA T4"):
            raise DeviceUnavailable("Gate 2 device identity changed. Select CPU or reconnect.")

    try:
        if runtime.getDeviceCount() != 1 or runtime.runtimeGetVersion() != CUDA_RUNTIME_VERSION:
            raise DeviceUnavailable("Gate 2 requires one T4 and CUDA runtime 13.0. Select CPU.")
        ordinal = 0
        name = runtime.getDeviceProperties(ordinal)["name"]
        if name not in (b"Tesla T4", "Tesla T4", b"NVIDIA T4", "NVIDIA T4"):
            raise DeviceUnavailable("Gate 2 requires a Tesla T4. Select CPU.")
    except operational as error:
        raise DeviceUnavailable("CUDA device discovery failed. Reconnect or select CPU.") from error

    @contextmanager
    def bind():
        # Guard context entry and cleanup without swallowing runner exceptions.
        stack = ExitStack()
        try:
            stack.enter_context(cp.cuda.Device(ordinal))
        except operational as error:
            raise DeviceUnavailable("CUDA binding failed. Reconnect or select CPU.") from error
        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                stack.close()
            except operational as error:
                message = "CUDA context cleanup failed. Reconnect the runtime."
                if primary is None:
                    raise DeviceUnavailable(message) from error
                primary.cuda_cleanup_error = error
                if hasattr(primary, "add_note"):
                    primary.add_note(f"{message} {type(error).__name__}: {error}")

    def readiness(shape):
        if shape != (64, 48):
            raise DeviceUnavailable("Gate 2 FFT readiness is restricted to 64 × 48. Select CPU.")
        try:
            verify_device()
            real = cp.zeros(shape, dtype=cp.float64)
            spectrum = cp.fft.fft2(real.astype(cp.complex128))
            cp.cuda.get_current_stream().synchronize()
            if real.dtype != cp.float64 or spectrum.dtype != cp.complex128:
                raise DeviceUnavailable("Gate 2 FFT did not preserve float64/complex128. Select CPU.")
            del spectrum, real
            free, total = runtime.memGetInfo()
        except operational as error:
            raise DeviceUnavailable("CUDA FFT or memory telemetry failed. Reconnect or select CPU.") from error
        return GPUReadiness(GPU_PROFILE, ordinal, free, total, monotonic())

    return GPUDevice(ordinal, name.decode() if isinstance(name, bytes) else name, bind, readiness)

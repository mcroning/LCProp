"""Frontend resource and backend decisions; no numerical qualification claims."""
from dataclasses import dataclass, replace
import math
from time import monotonic
from lcprop.notebook.batch_device import RunDevice, DeviceUnavailable, GPUReadiness
from lcprop.notebook.device_policy import probe_cuda, CAPABILITY_PROFILE
from .batch_resources import MIB


@dataclass
class RuntimePolicy:
    override: bool = False
    last_warning: str = ''

    def check(self, estimate, available, *, label='memory'):
        if type(estimate) is not int or estimate < 0:
            raise ValueError('Invalid memory estimate')
        if available is not None and (type(available) is not int or available < 0):
            raise ValueError('Invalid runtime memory telemetry')
        self.last_warning = ''
        if available is None or estimate > available * 0.6:
            self.last_warning = (f'{label}: estimated {estimate/MIB:.0f} MiB; '
                + ('available memory unknown' if available is None else f'{available/MIB:.0f} MiB available')
                + '. Estimate is conservative, not a guarantee. Allocation can fail or the runtime can terminate. '
                  'Save prior results; explicitly acknowledge this risk to proceed.')
            if not self.override:
                raise ValueError(self.last_warning)
        return self.last_warning


@dataclass(frozen=True)
class CompatibleDevice(RunDevice):
    """Runtime memory check without historical fixture caps."""
    memory_override: bool = False

    def verify_gpu_memory(self, shape):
        if self.gpu is None:
            return
        if self.gpu.readiness is None:
            raise DeviceUnavailable('GPU memory telemetry is unavailable')
        sample = self.gpu.readiness(shape)
        if (not isinstance(sample, GPUReadiness) or sample.profile != self.readiness_profile
                or sample.ordinal != self.gpu.ordinal
                or type(sample.free_bytes) is not int or type(sample.total_bytes) is not int
                or not 0 <= sample.free_bytes <= sample.total_bytes or sample.total_bytes <= 0
                or not math.isfinite(sample.sampled_at)
                or not 0 <= monotonic() - sample.sampled_at <= 5):
            raise DeviceUnavailable('GPU memory telemetry or device attribution is invalid/stale')
        # One complex128 optical plane is a necessary allocation, not a model of peak use.
        minimum = int(shape[0]) * int(shape[1]) * (16 if self.precision == 'float64' else 8)
        if minimum > sample.free_bytes:
            raise DeviceUnavailable('A single required optical plane cannot fit in free GPU memory')
        if self.gpu_work_bytes > sample.free_bytes * .6 and not self.memory_override:
            raise DeviceUnavailable(f'GPU working estimate {self.gpu_work_bytes/MIB:.0f} MiB; '
                f'free {sample.free_bytes/MIB:.0f} MiB / total {sample.total_bytes/MIB:.0f} MiB. '
                'Acknowledge demanding-memory risk explicitly to proceed; this is an estimate.')


class CompatibleDevices:
    def __init__(self, *, probe=None):
        self.probe = probe or probe_cuda
        self.allow_unqualified = False
        self.memory_override = False
        self.working = 0
        self.request = None
        self.detected = None
        self.last_memory = None

    def reason(self, route, digest):
        # Compatibility is checked against the actual typed request at dispatch.
        # Qualification is reported separately, never inferred from shape or device model.
        return ''

    def resolve(self, route, digest, intent, precision, shape):
        if intent == 'numpy':
            return CompatibleDevice(intent, 'numpy', precision)
        if intent not in {'auto', 'cupy'}:
            raise ValueError('Unknown device intent')
        try:
            if not self.allow_unqualified:
                raise DeviceUnavailable('This edited request/runtime is not numerically qualified. '
                    'Opt in to compatible unqualified GPU execution, or choose CPU.')
            hardware, gpu = self.probe()
            if precision == "float32" and self.probe is probe_cuda:
                gpu = precision32_gpu(gpu)
            self.detected = hardware
            device = CompatibleDevice(intent, 'cupy', precision, gpu=gpu,
                gpu_work_bytes=self.working, readiness_profile=CAPABILITY_PROFILE,
                hardware=hardware, qualification='Unqualified request/runtime; explicit opt-in',
                memory_override=self.memory_override)
            with device.context():
                device.verify_gpu_memory(shape)
            return device
        except (DeviceUnavailable, ImportError, OSError, RuntimeError) as error:
            if intent == 'auto':
                return CompatibleDevice(intent, 'numpy', precision, fallback_reason=str(error),
                                        hardware=self.detected)
            raise DeviceUnavailable(str(error)) from error


def precision32_gpu(gpu):
    """Add a dtype-specific tiny FFT to the existing identity/library guard."""
    checked=False
    def readiness(shape):
        nonlocal checked
        sample=gpu.readiness(shape)
        if not checked:
            import cupy as cp
            result=cp.fft.fft2(cp.zeros((8,8),dtype=cp.complex64))
            cp.cuda.get_current_stream().synchronize()
            if result.dtype != cp.complex64:
                raise DeviceUnavailable('CUDA FFT did not preserve complex64 precision')
            checked=True
        return sample
    return replace(gpu,readiness=readiness)

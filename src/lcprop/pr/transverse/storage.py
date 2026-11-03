"""Explicit longitudinal ownership for full-transverse TD.

Storage is execution metadata, never a different material model. Native bounded
execution remains unqualified until a resident/bounded native overlap passes.
"""
from dataclasses import dataclass
import math
import numpy as np

from lcprop.core.backend import asnumpy, scalar_float
from .transport import state_from_potential
from .diagnostics import state_diagnostics


@dataclass(frozen=True)
class TDStoragePolicy:
    mode: str = "resident"
    chunk_planes: int = 1
    host_budget_bytes: int | None = None

    def validate(self):
        if self.mode not in ("resident", "bounded"):
            raise ValueError("unknown transverse TD storage mode")
        if type(self.chunk_planes) is not int or self.chunk_planes < 1:
            raise ValueError("chunk_planes must be a positive integer")
        if self.host_budget_bytes is not None and (
            type(self.host_budget_bytes) is not int or self.host_budget_bytes <= 0
        ):
            raise ValueError("host budget must be a positive byte count")

    def plan(self, shape, dtype):
        self.validate()
        plane = math.prod(shape[-2:]) * np.dtype(dtype).itemsize
        volume = plane * shape[0]
        k = min(self.chunk_planes, shape[0])
        # Includes accepted/candidate/source/initial/phase, result ownership,
        # continuation checkpoint and serialization copies. No derived volume.
        host = 12 * volume + 16 * plane
        if self.mode == "bounded" and self.host_budget_bytes is not None and host > self.host_budget_bytes:
            raise ValueError("bounded TD host ownership estimate exceeds host budget")
        return dict(mode=self.mode, chunk_planes=k, longitudinal_backing_bytes=5*volume,
                    host_admission_bytes=host, pinned_staging_bytes=0,
                    device_array_allowance_bytes=(40*k*plane+64*plane if self.mode == "bounded"
                                                  else 48*volume+64*plane),
                    external_fft_workspace_bytes=None, native_qualified=False)


def allocation_snapshot(arrays, *, xp=np):
    """Count unique backing allocations, without copying or scanning values.

    Array bytes exclude opaque library scratch. Pool reservations include used
    blocks; they must not be added to owned bytes as disjoint memory categories.
    """
    bases = {}
    logical = 0
    for a in arrays:
        logical += int(a.nbytes)
        b = a
        while getattr(b, "base", None) is not None and hasattr(b.base, "nbytes"):
            b = b.base
        if xp is np:
            key, size = id(b), int(b.nbytes)
        else:
            mem = b.data.mem
            key, size = int(mem.ptr), int(mem.size)
        bases[key] = max(bases.get(key, 0), size)
    record = dict(logical_array_bytes=logical, unique_backing_bytes=sum(bases.values()),
                  residence="host" if xp is np else "device", pool_used_bytes=None,
                  pool_reserved_bytes=None, device_free_bytes=None, device_total_bytes=None,
                  external_cuda_fft_bytes=None)
    if xp is not np:
        pool = xp.get_default_memory_pool()
        free, total = xp.cuda.runtime.memGetInfo()
        record.update(pool_used_bytes=int(pool.used_bytes()), pool_reserved_bytes=int(pool.total_bytes()),
                      device_free_bytes=int(free), device_total_bytes=int(total))
    return record


class BoundedStorage:
    def __init__(self, policy, shape, dtype, xp):
        self.plan = policy.plan(shape, dtype)
        self.k = self.plan['chunk_planes']
        self.shape, self.dtype, self.xp = shape, np.dtype(dtype), xp
        self.max_uploaded_planes = 0
        self.samples = []

    def slices(self):
        for start in range(0, self.shape[0], self.k):
            yield slice(start, min(start+self.k, self.shape[0]))

    def upload(self, host):
        if host.ndim == 3:
            if host.shape[0] > self.k:
                raise ValueError("unbounded longitudinal upload")
            self.max_uploaded_planes = max(self.max_uploaded_planes, host.shape[0])
        return self.xp.asarray(host)

    def record(self, *arrays):
        # Bounded metadata retention, not scientific-array history.
        snap = allocation_snapshot(arrays, xp=self.xp)
        if not self.samples or snap['unique_backing_bytes'] > self.samples[0]['unique_backing_bytes']:
            self.samples[:] = [snap]

    def initial(self, value):
        host = np.zeros(self.shape, dtype=self.dtype)
        if value is not None:
            if value.shape != self.shape:
                raise ValueError("initial_psi shape mismatch")
            for s in self.slices():
                host[s] = asnumpy(value[s])
        for s in self.slices():
            block = self.upload(host[s]).copy()
            if not bool(scalar_float(self.xp.all(self.xp.isfinite(block)))):
                raise ValueError("initial_psi must contain only finite values")
            block -= self.xp.mean(block, axis=(-2, -1), keepdims=True)
            host[s] = asnumpy(block)
        return host

    def step(self, function, psi, source, *, check, **kwargs):
        candidate = np.empty_like(psi)
        for s in self.slices():
            check("material_chunk_boundary")
            old, driving = self.upload(psi[s]), self.upload(source[s])
            result = function(old, driving, xp=self.xp, **kwargs)
            self.record(old, driving, result)
            candidate[s] = asnumpy(result)
            del old, driving, result
        return candidate

    def summarize(self, psi, *, diagnostics=False, **kwargs):
        sums = np.empty(self.shape[0], dtype=np.float64)
        minimum, physical = float('inf'), True
        weighted = {}; maxima = {}; integrals = []
        finite = True
        for s in self.slices():
            state = state_from_potential(self.upload(psi[s]), xp=self.xp, **kwargs)
            arrays = (state.psi, state.carrier_density, state.E_x, state.E_y)
            self.record(*arrays)
            ok = self.xp.all(self.xp.isfinite(state.psi) & self.xp.isfinite(state.carrier_density)
                             & self.xp.isfinite(state.E_x) & self.xp.isfinite(state.E_y)
                             & (state.carrier_density > 0))
            physical = physical and bool(scalar_float(ok))
            minimum = min(minimum, scalar_float(self.xp.min(state.carrier_density)))
            sums[s] = asnumpy(self.xp.sum(state.carrier_density, axis=(-2, -1)))
            if diagnostics:
                d = state_diagnostics(state, xp=self.xp, **{k: v for k, v in kwargs.items() if k != "applied_field_x"})
                n = state.psi.size
                for key, value in d.items():
                    if key == 'carrier_integrals_per_z':
                        integrals.extend(value.tolist())
                    elif key == 'finite_material_state':
                        finite = finite and value
                    elif key.endswith('_rms'):
                        weighted.setdefault(key, []).append(value*value*n)
                    else:
                        maxima[key] = max(maxima.get(key, 0.), value)
            del state, arrays
        d = {key: math.sqrt(math.fsum(values)/math.prod(self.shape)) for key, values in weighted.items()}
        d.update(maxima)
        if diagnostics:
            d.update(carrier_integrals_per_z=np.asarray(integrals), finite_material_state=finite)
        return dict(carrier_sums=sums, carrier_minimum=minimum, physical=physical, diagnostics=d)


def plan_td_storage(shape, dtype, *, device_budget_bytes, host_budget_bytes,
                    fft_workspace_bytes, allow_bounded=False):
    """Pre-execution planner; callers must supply a qualified workspace bound.

    This does not enable bounded GPU dispatch or enlarge a scientific envelope.
    Native admission must reserve its safety margin before passing these budgets.
    """
    if any(type(v) is not int or v <= 0 for v in (device_budget_bytes, host_budget_bytes)):
        raise ValueError("positive device and host budgets required")
    if type(fft_workspace_bytes) is not int or fft_workspace_bytes < 0:
        raise ValueError("explicit nonnegative FFT workspace bound required")
    resident = TDStoragePolicy().plan(shape, dtype)
    if resident['host_admission_bytes'] > host_budget_bytes:
        raise ValueError("TD host ownership estimate exceeds host budget")
    if resident['device_array_allowance_bytes'] + fft_workspace_bytes <= device_budget_bytes:
        return TDStoragePolicy('resident', 1, host_budget_bytes)
    if not allow_bounded:
        raise ValueError("resident estimate exceeds device budget; bounded path not authorized")
    plane = math.prod(shape[-2:]) * np.dtype(dtype).itemsize
    k = min(shape[0], (device_budget_bytes-fft_workspace_bytes-64*plane)//(40*plane))
    if k < 1:
        raise ValueError("one full transverse plane and workspace cannot fit device budget")
    return TDStoragePolicy('bounded', int(k), host_budget_bytes)

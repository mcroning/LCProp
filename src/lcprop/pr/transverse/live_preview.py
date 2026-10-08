"""Optional bounded observations of complete passes through accepted material.

No scientific state, history, propagation or device transfer is owned here.
"""
from dataclasses import dataclass
from threading import Lock
from time import monotonic
import math
import numpy as np


def supported(request):
    return (getattr(getattr(request, 'backend', None), 'backend', None) == 'numpy'
            and getattr(getattr(request, 'material_response', None), 'model', None) == 'nonlinear'
            and getattr(getattr(request, 'solver', None), 'integrator', None) == 'spectral_imex_euler'
            and getattr(request, 'resolved_optical_coupling', None) == 'frozen_material_published_optical_first_v1')


def next_segment_number(checkpoint):
    """Initial execution is 1; persisted continuation entries count from 1.

    Unknown or inconsistent lineage gets a neutral label, never a GUI counter.
    The workflow remains responsible for scientific checkpoint validation.
    """
    lineage = checkpoint.record.get('lineage')
    if not isinstance(lineage, list) or any(
        not isinstance(entry, dict) or type(entry.get('segment_index')) is not int
        or entry['segment_index'] != index
        for index, entry in enumerate(lineage, 1)
    ):
        return None
    return len(lineage) + 2


def can_preserve_preview(request, checkpoint, bound_identity):
    """Presentation eligibility; never repairs or changes a scientific request."""
    if checkpoint is None or bound_identity != checkpoint.identity:
        return False
    from .continuation import compatibility_key
    try:
        return compatibility_key(request) == checkpoint.record['compatibility_key']
    except (ValueError, TypeError, KeyError):
        return False


@dataclass(frozen=True)
class OpticalPassPreview:
    run_id: str
    segment_id: str
    pass_id: int
    observed_step: int
    segment_step: int
    cumulative_time: float
    intensity_xz: np.ndarray
    x_um: np.ndarray
    z_um: np.ndarray
    y_um: float
    original_shape: tuple
    final_replay: bool
    segment_number: int | None = None
    total_steps: int | None = None
    sampling: str = 'point samples; complete published arriving-source pass'


class LatestPreview:
    """One pending immutable buffer; no array-bearing queued Qt signal."""
    def __init__(self, run_id):
        self.run_id = run_id
        self._lock = Lock()
        self._pending = None
        self._last = -1
        self._segment_id = None
        self.closed = False
        self.error = None

    def publish(self, frame):
        with self._lock:
            if (self.closed or frame.run_id != self.run_id or frame.pass_id <= self._last
                    or (self._segment_id is not None and frame.segment_id != self._segment_id)):
                return
            self._segment_id = frame.segment_id
            self._last = frame.pass_id
            self._pending = frame

    def take(self):
        with self._lock:
            frame, self._pending = self._pending, None
            return frame

    def close(self):
        with self._lock:
            self.closed = True
            self._pending = None


class AcceptedPassObserver:
    def __init__(self, mailbox, *, segment_id, interval_seconds=.5, clock=monotonic,
                 prior_steps=0, start_time=0., time_at=None, segment_number=1, total_steps=None):
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError('preview interval must be finite and nonnegative')
        self.mailbox = mailbox
        self.segment_id = segment_id
        self.segment_number = segment_number
        self.total_steps = total_steps
        self.interval = interval_seconds
        self.clock = clock
        self.prior_steps = prior_steps
        self.start_time = start_time
        self.time_at = time_at
        self._last_at = None
        self._last_pass = -1
        self.published = 0

    def with_origin(self, prior_steps, start_time, time_at=None, segment_number=None):
        return AcceptedPassObserver(self.mailbox, segment_id=self.segment_id,
            interval_seconds=self.interval, clock=self.clock,
            prior_steps=prior_steps, start_time=start_time, time_at=time_at,
            segment_number=segment_number, total_steps=self.total_steps)

    def observe(self, source, *, grid, reference, background, step, dt, pass_id, final=False):
        # Presentation failure must never abort or accept a scientific step.
        try:
            now = self.clock()
            if pass_id <= self._last_pass or (not final and self._last_at is not None
                                         and now-self._last_at < self.interval):
                return
            nz, nx, ny = source.shape
            ix = np.rint(np.linspace(0, nx-1, min(nx,128))).astype(int)
            iz = np.rint(np.linspace(0, nz-1, min(nz,256))).astype(int)
            # Existing nearest-to-zero cell-center convention (first on ties).
            cy = (ny-1)//2
            scale = 0. if getattr(reference,'optical_scale_W_cm2',None) == 0 else float(reference)
            values = np.array((source[iz[:,None],ix[None,:],cy]-background)*scale,
                              dtype=np.float64,copy=True)
            x = (ix-.5*(nx-1))*grid.dx_um
            z = (iz+1)*grid.dz_um
            if not np.isfinite(values).all():
                raise ValueError('nonfinite optical preview')
            for a in (values,x,z):a.setflags(write=False)
            self.mailbox.publish(OpticalPassPreview(self.mailbox.run_id,self.segment_id,
                pass_id,self.prior_steps+step,step,(self.time_at(step,dt) if self.time_at else self.start_time+step*dt),
                values,x,z,(cy-.5*(ny-1))*grid.dy_um,tuple(source.shape),final,
                segment_number=self.segment_number,total_steps=self.total_steps))
            self._last_pass = pass_id
            self._last_at = self.clock()
            self.published += 1
        except Exception as exc:
            self.mailbox.error = f'Live preview unavailable: {type(exc).__name__}: {exc}'

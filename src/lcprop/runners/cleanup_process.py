"""POSIX cleanup-only process isolation and bounded diagnostic capture."""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time

CLEANUP_STREAM_LIMIT_BYTES = 8192  # 4096-byte head + 4096-byte tail per stream
CLEANUP_READ_BYTES = 16384
CLEANUP_TERM_GRACE_SECONDS = 0.25
CLEANUP_PIPE_DRAIN_SECONDS = 0.25
CLEANUP_REAP_SECONDS = 1.0
CLEANUP_ERROR_TEXT_LIMIT = 20480


class _Excerpt:
    def __init__(self):
        self.head = b""
        self.tail = b""
        self.observed = 0
        self.complete = False

    def feed(self, data):
        self.observed += len(data)
        half = CLEANUP_STREAM_LIMIT_BYTES // 2
        take = min(half - len(self.head), len(data))
        self.head += data[:take]
        rest = data[take:]
        if rest:
            self.tail = (self.tail + rest[-half:])[-half:]

    def record(self):
        retained = self.head + self.tail
        return {
            "text": retained.decode("utf-8", errors="replace"),
            "truncated": self.observed > len(retained),
            "bytes_observed": self.observed,
            "bytes_retained": len(retained),
            "retention_limit_bytes": CLEANUP_STREAM_LIMIT_BYTES,
            "capture_complete": self.complete,
        }


class CleanupOutput(str):
    def __new__(cls, diagnostics):
        value = super().__new__(cls, diagnostics["stdout"]["text"].strip())
        value.cleanup_diagnostics = diagnostics
        return value


def _signal_group(pid, sig):
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def run_cleanup_command(arguments, *, timeout):
    """Drain bounded excerpts; TERM/KILL only this attempt's new POSIX session.

    No communicate(), pipe-reader threads, unbounded wait(), or retry. Descendants
    inheriting the session/group are included, even when they hold pipes open.
    """
    if os.name != "posix":
        raise OSError("isolated cleanup requires POSIX process groups")
    streams = {name: _Excerpt() for name in ("stdout", "stderr")}
    selector = None
    failure = None
    timed_out = False
    shutdown_errors = []
    pending_interrupt = None
    reporting_failed = False

    def remember_interrupt(exc):
        nonlocal pending_interrupt
        if not isinstance(exc, Exception) and pending_interrupt is None:
            pending_interrupt = exc

    def note(phase, exc):
        # Diagnostics never authorize or prevent finalization. Even an exception
        # whose __str__ fails must not skip the next close, signal, or reap.
        nonlocal reporting_failed
        remember_interrupt(exc)
        try:
            shutdown_errors.append(f"{phase}: {type(exc).__name__}: {str(exc)[:512]}")
        except BaseException as reporting_exc:
            reporting_failed = True
            remember_interrupt(reporting_exc)

    def pump(wait):
        for key, _ in selector.select(max(0., wait)):
            try:
                data = os.read(key.fileobj.fileno(), CLEANUP_READ_BYTES)
            except BlockingIOError:
                continue
            if data:
                streams[key.data].feed(data)
            else:
                streams[key.data].complete = True
                selector.unregister(key.fileobj)
                key.fileobj.close()

    process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        try:
            selector = selectors.DefaultSelector()
            for name in streams:
                pipe = getattr(process, name)
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, name)
            deadline = time.monotonic() + timeout
            while selector.get_map() or process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                pump(min(remaining, .05))
        except BaseException as exc:
            failure = True
            note("execution", exc)
    finally:
        # This outer ownership finalizer cannot be skipped by an interruption
        # or secondary reporting fault. Nested finally blocks make KILL,
        # resource closure, and reap independent obligations.
        try:
            # Never reap between TERM and KILL: the leader pins the owned PGID.
            # A normal poll-confirmed completion ends ownership, so no signal
            # is sent to that (potentially reusable) PID/group afterward.
            if process.returncode is None:
                try:
                    try:
                        _signal_group(process.pid, signal.SIGTERM)
                    except BaseException as exc:
                        note("TERM", exc)
                    try:
                        deadline = time.monotonic() + CLEANUP_TERM_GRACE_SECONDS
                        while selector is not None and time.monotonic() < deadline:
                            pump(min(.05, deadline - time.monotonic()))
                    except BaseException as exc:
                        note("grace drain", exc)
                finally:
                    try:
                        _signal_group(process.pid, signal.SIGKILL)
                    except BaseException as exc:
                        # Includes Darwin's already-dead zombie-group EPERM.
                        # No destructive signal is sent after the reap below.
                        note("KILL", exc)
        finally:
            try:
                try:
                    deadline = time.monotonic() + CLEANUP_PIPE_DRAIN_SECONDS
                    while selector is not None and selector.get_map() and time.monotonic() < deadline:
                        pump(min(.05, deadline - time.monotonic()))
                except BaseException as exc:
                    note("final drain", exc)
            finally:
                try:
                    # Close actual Popen pipes, including unregistered pipes.
                    # Each close/report failure is isolated from the next one.
                    for name in streams:
                        try:
                            getattr(process, name).close()
                        except BaseException as exc:
                            note(f"close {name}", exc)
                finally:
                    try:
                        try:
                            if selector is not None:
                                selector.close()
                        except BaseException as exc:
                            note("selector close", exc)
                    finally:
                        try:
                            process.wait(timeout=CLEANUP_REAP_SECONDS)
                        except BaseException as exc:
                            note("reap", exc)
                            # Only a nonblocking check; no repeated wait budget.
                            try:
                                if process.poll() is None:
                                    note("termination uncertain", RuntimeError("child not confirmed reaped"))
                            except BaseException as poll_exc:
                                note("termination uncertain", poll_exc)
    # Interruptions retain their identity and propagate only after ownership
    # finalization, never as a successful cleanup result.
    if pending_interrupt is not None:
        raise pending_interrupt
    if reporting_failed:
        shutdown_errors.append("cleanup diagnostic recording failed; details unavailable")
    diagnostics = {name: stream.record() for name, stream in streams.items()}
    out = (streams['stdout'].head + streams['stdout'].tail)
    err = (streams['stderr'].head + streams['stderr'].tail)
    if timed_out:
        failure = subprocess.TimeoutExpired(arguments, timeout, output=out, stderr=err)
    elif failure is not None or shutdown_errors:
        failure = OSError("cleanup execution/shutdown indeterminate: "
                          + "; ".join(shutdown_errors))
    elif process.returncode:
        failure = subprocess.CalledProcessError(process.returncode, arguments, output=out, stderr=err)
    else:
        return CleanupOutput(diagnostics)
    failure.cleanup_shutdown_errors = shutdown_errors
    failure.cleanup_diagnostics = diagnostics
    raise failure


def cleanup_failure_detail(exc):
    """Bound even diagnostic exceptions from alternate/custom transports."""
    diagnostics = getattr(exc, 'cleanup_diagnostics', None)
    if diagnostics is None:
        diagnostics = {}
        for name in ('stdout', 'stderr'):
            value = getattr(exc, name, None) or b''
            # Alternate transports may supply strings; cap before encoding.
            unknown_bytes = isinstance(value, str) and len(value) > CLEANUP_STREAM_LIMIT_BYTES
            if isinstance(value, str):
                value = (value[:4096] + value[-4096:] if unknown_bytes else value).encode('utf-8')
            excerpt = _Excerpt()
            excerpt.feed(value)
            record = excerpt.record()
            if unknown_bytes:
                record.update(truncated=True, bytes_observed=None)
            diagnostics[name] = record
    detail = f"{type(exc).__name__}: {str(exc)[:1024]}"
    for issue in getattr(exc, "cleanup_shutdown_errors", ()):
        detail += "\n" + issue
    for name, record in diagnostics.items():
        detail += f"\n{name} (truncated={record['truncated']}): {record['text']}"
    return detail[:CLEANUP_ERROR_TEXT_LIMIT], diagnostics

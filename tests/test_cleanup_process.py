"""No SSH/network: real local process groups behind cleanup orchestration."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from lcprop.runners.cleanup_process import (
    CLEANUP_STREAM_LIMIT_BYTES, CLEANUP_ERROR_TEXT_LIMIT, run_cleanup_command,
)
from lcprop.runners.slurm import SlurmRunner, SubprocessRemoteTransport
from lcprop.transport.defaults import default_transport_registry
from lcprop.transport.status import RemoteRunState
from tests.test_slurm_runner import FakeTransport, _config, _request, LC_STATIC_OPERATION, _cleanup_commands


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # On Linux an orphan can remain a zombie until the system subreaper reaps it;
    # it is terminated, cannot hold pipes, and cannot execute any cleanup work.
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists() and stat.read_text().split(') ', 1)[1].startswith('Z '):
        return False
    return True


def _wait_dead(pid):
    end = time.monotonic() + 2
    while _alive(pid) and time.monotonic() < end:
        time.sleep(.01)
    assert not _alive(pid)


def _noise_program(streams, exitcode):
    return f'''import os, sys
for fd in {streams!r}:
    os.write(fd, b'HEAD-CONTEXT\\n')
for i in range(128):
    for fd in {streams!r}:
        os.write(fd, b'x'*16384)
for fd in {streams!r}:
    os.write(fd, b'FINAL-ERROR-CONTEXT\\n')
sys.exit({exitcode})
'''


@pytest.mark.parametrize('streams', [(1,), (2,), (1, 2)])
@pytest.mark.parametrize('exitcode', [0, 1])
def test_noisy_capture_drains_both_streams_with_fixed_head_tail(streams, exitcode):
    command = [sys.executable, '-c', _noise_program(streams, exitcode)]
    if exitcode:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            run_cleanup_command(command, timeout=5)
        diagnostics = caught.value.cleanup_diagnostics
    else:
        output = run_cleanup_command(command, timeout=5)
        diagnostics = output.cleanup_diagnostics
    for fd, name in [(1, 'stdout'), (2, 'stderr')]:
        record = diagnostics[name]
        assert record['capture_complete']
        assert record['retention_limit_bytes'] == CLEANUP_STREAM_LIMIT_BYTES
        assert record['bytes_retained'] <= CLEANUP_STREAM_LIMIT_BYTES
        if fd in streams:
            assert record['truncated'] and record['bytes_observed'] > 2_000_000
            assert record['text'].startswith('HEAD-CONTEXT')
            assert record['text'].endswith('FINAL-ERROR-CONTEXT\n')
        else:
            assert record['bytes_observed'] == 0 and not record['truncated']


@pytest.mark.parametrize('timeout_case', [False, True])
def test_real_cleanup_process_returns_verified_result_with_bounded_provenance(tmp_path, monkeypatch, timeout_case):
    pidfile = tmp_path/'pids.json'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    if timeout_case:
        child = _noise_program((1, 2), 0).replace('sys.exit(0)', 'import time; time.sleep(60)')
        program = f'''import os,sys,subprocess,time,signal,json
child=subprocess.Popen([sys.executable,'-c',{child!r}])
open({str(pidfile)!r},'w').write(json.dumps([os.getpid(),child.pid]))
def term(sig, frame):
    child.wait(timeout=1)  # Reap TERM-killed descendant; parent still needs KILL.
signal.signal(signal.SIGTERM,term)
while True: time.sleep(60)
'''
    else:
        program = _noise_program((1, 2), 1)
    real = SubprocessRemoteTransport()
    monkeypatch.setattr(real, '_ssh_command', lambda *a: [sys.executable, '-c', program])
    class Transport(FakeTransport):
        def ssh_with_timeout(self, host, *args, timeout):
            self.commands.append((host, args))
            assert timeout == 30.0  # Production policy unchanged; shorten only surrogate.
            return real.ssh_with_timeout(host, *args, timeout=1 if timeout_case else 5)
    transport = Transport()
    states = []
    runner = SlurmRunner(_config(tmp_path), (LC_STATIC_OPERATION,), transport=transport,
                        registry=default_transport_registry(), sleep=lambda _: None)
    start = time.monotonic()
    try:
        result = runner.run_registered('lc', 'static', _request(), progress_callback=states.append)
        assert time.monotonic() - start < 8
        assert result.run_data is not None
        assert states[-1].state == RemoteRunState.COMPLETED
        assert states[-2].state == RemoteRunState.CLEANING
        record = result.operational_provenance['remote_cleanup']
        assert record['outcome'] == ('timed_out' if timeout_case else 'failed')
        assert len(record['error']) <= CLEANUP_ERROR_TEXT_LIMIT
        assert len(json.dumps(record)) < 40000
        for name in ('stdout','stderr'):
            stream = record['diagnostics'][name]
            assert stream['truncated']
            assert stream['bytes_retained'] == CLEANUP_STREAM_LIMIT_BYTES
            assert 'HEAD-CONTEXT' in stream['text'] and 'FINAL-ERROR-CONTEXT' in stream['text']
        assert len(_cleanup_commands(transport)) == 1
        assert record['remote_artifacts_retained'] is None
        assert unrelated.poll() is None  # Never signal the parent's or another run's group.
        if timeout_case:
            parent, child = json.loads(pidfile.read_text())
            _wait_dead(parent)
            _wait_dead(child)
            with pytest.raises(ChildProcessError):
                os.waitpid(parent, os.WNOHANG)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)


@pytest.mark.parametrize('fault', ['create', 'register', 'term', 'grace', 'final', 'wait'])
def test_shutdown_faults_still_kill_owned_group_and_deliver_result(tmp_path, monkeypatch, fault):
    import lcprop.runners.cleanup_process as cleanup
    pidfile = tmp_path/'fault-pids.json'
    child_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'
    program = f'''import os,sys,subprocess,time,signal,json
signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c',{child_code!r}])
open({str(pidfile)!r},'w').write(json.dumps([os.getpid(),child.pid]))
while True: time.sleep(60)
'''
    original_popen = subprocess.Popen
    original_selector = cleanup.selectors.DefaultSelector
    original_signal = cleanup._signal_group
    processes = []
    signals = []
    injected = []
    unrelated = original_popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        deadline = time.monotonic() + 3
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pidfile.exists()
        if fault == 'wait':
            def broken_wait(*args, **kwargs):
                injected.append('wait')
                raise OSError('injected wait fault')
            monkeypatch.setattr(process, 'wait', broken_wait)
        return process

    class Selector:
        def __init__(self):
            if fault == 'create':
                injected.append('create')
                raise OSError('injected selector creation fault')
            self.inner = original_selector()
        def register(self, *args):
            if fault == 'register':
                injected.append('register')
                raise OSError('injected registration fault')
            return self.inner.register(*args)
        def select(self, timeout):
            phase = ('final' if signal.SIGKILL in signals else
                     'grace' if signal.SIGTERM in signals else 'running')
            if phase == fault and fault not in injected:
                injected.append(fault)
                raise OSError('injected '+fault+' drain fault')
            return self.inner.select(timeout)
        def __getattr__(self, name):
            return getattr(self.inner, name)

    def send(pid, sig):
        assert pid == processes[0].pid
        signals.append(sig)
        if sig == signal.SIGTERM and fault == 'term':
            injected.append('term')
            raise OSError('injected TERM failure')
        return original_signal(pid, sig)

    monkeypatch.setattr(cleanup.subprocess, 'Popen', popen)
    monkeypatch.setattr(cleanup.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(cleanup, '_signal_group', send)
    real = SubprocessRemoteTransport()
    monkeypatch.setattr(real, '_ssh_command', lambda *a: [sys.executable, '-c', program])
    class Transport(FakeTransport):
        def ssh_with_timeout(self, host, *args, timeout):
            self.commands.append((host, args))
            return real.ssh_with_timeout(host, *args, timeout=.2)
    states = []
    runner = SlurmRunner(_config(tmp_path), (LC_STATIC_OPERATION,), transport=Transport(),
                        registry=default_transport_registry(), sleep=lambda _: None)
    started = time.monotonic()
    try:
        result = runner.run_registered('lc', 'static', _request(), progress_callback=states.append)
        assert time.monotonic() - started < 6
        assert injected and signal.SIGKILL in signals
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        assert result.run_data is not None and states[-1].state == RemoteRunState.COMPLETED
        outcome = result.operational_provenance['remote_cleanup']
        assert outcome['outcome'] in ('timed_out', 'indeterminate')
        assert 'injected' in outcome['error']
        assert outcome['remote_artifacts_retained'] is None
        assert unrelated.poll() is None
        parent, child = json.loads(pidfile.read_text())
        _wait_dead(child)
        if fault == 'wait' and processes[0].returncode is None:
            assert 'termination uncertain' in outcome['error']
        else:
            _wait_dead(parent)
            with pytest.raises(ChildProcessError):
                os.waitpid(parent, os.WNOHANG)
    finally:
        # The wait-fault case may truthfully report an unreaped child. Test-only
        # recovery uses the real bounded wait, never another production retry.
        for process in processes:
            original_popen.wait(process, timeout=2)
        unrelated.terminate()
        original_popen.wait(unrelated, timeout=2)


def test_final_gui_cleanup_message_bounded_without_cropping_provenance(tmp_path):
    from lcprop.gui.remote_execution import remote_status_text
    from lcprop.runners.cleanup_process import _Excerpt
    root = '/' + '/'.join(['r' * 200] * 19)  # Valid components; total path <4096.
    diagnostics = {}
    for name in ('stdout', 'stderr'):
        excerpt = _Excerpt()
        excerpt.feed((name + '-HEAD').encode() + b'x' * 50000 + b'FINAL-SSH-ERROR')
        excerpt.complete = True
        diagnostics[name] = excerpt.record()
    class Transport(FakeTransport):
        def ssh_with_timeout(self, host, *args, timeout):
            self.commands.append((host, args))
            exc = subprocess.CalledProcessError(1, ['ssh', host, *args])
            exc.cleanup_diagnostics = diagnostics
            raise exc
    states = []
    runner = SlurmRunner(_config(tmp_path, remote_run_root=root), (LC_STATIC_OPERATION,),
                        transport=Transport(), registry=default_transport_registry(), sleep=lambda _: None)
    result = runner.run_registered('lc', 'static', _request(), progress_callback=states.append)
    record = result.operational_provenance['remote_cleanup']
    status = states[-1]
    assert record['target'] == status.remote_cleanup_target
    assert record['target'].startswith(root + '/') and len(record['target']) < 4096
    message = remote_status_text(status)
    assert len(message) <= CLEANUP_ERROR_TEXT_LIMIT
    assert 'cleanup message truncated' in message
    assert message.startswith('Runner: Slurm') and message.endswith('FINAL-SSH-ERROR')
    assert record['diagnostics'] == diagnostics
    for stream in record['diagnostics'].values():
        assert stream['bytes_retained'] == 8192 and stream['truncated'] and stream['capture_complete']


@pytest.mark.parametrize('fault', [
    'grace_interrupt', 'final_interrupt', 'final_error', 'system_exit',
    'custom_base', 'formatting', 'close_formatting', 'pipe_close',
])
def test_interrupt_and_reporting_faults_finalize_before_propagation(tmp_path, monkeypatch, fault):
    import lcprop.runners.cleanup_process as cleanup
    class StopCleanup(BaseException):
        pass
    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError('diagnostic formatting failed')
    interrupt = (SystemExit(17) if fault == 'system_exit' else
                 StopCleanup('custom interruption') if fault == 'custom_base' else
                 KeyboardInterrupt('cleanup interrupted'))
    interrupt_case = fault in ('grace_interrupt', 'final_interrupt', 'system_exit', 'custom_base')
    pidfile = tmp_path/'owned.json'
    ready = tmp_path/'descendant-ready'
    child_code = f'''import signal,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
open({str(ready)!r},'w').close()
time.sleep(60)
'''
    program = f'''import os,sys,subprocess,time,signal,json
signal.signal(signal.SIGTERM,signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-c',{child_code!r}])
while not os.path.exists({str(ready)!r}): time.sleep(.01)
open({str(pidfile)!r},'w').write(json.dumps([os.getpid(),child.pid]))
time.sleep(60)
'''
    real_popen = subprocess.Popen
    real_selector = cleanup.selectors.DefaultSelector
    real_signal = cleanup._signal_group
    processes, events, signals = [], [], []
    injected = set()
    unrelated = real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)

    class Pipe:
        def __init__(self, pipe, name):
            self.pipe, self.name = pipe, name
        def fileno(self):
            return self.pipe.fileno()
        def close(self):
            events.append('close '+self.name)
            self.pipe.close()
            if self.name == 'stdout' and fault in ('close_formatting', 'pipe_close') and 'close' not in injected:
                injected.add('close')
                raise UnprintableError() if fault == 'close_formatting' else OSError('pipe close failed')

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        deadline = time.monotonic() + 3
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pidfile.exists()
        process.stdout = Pipe(process.stdout, 'stdout')
        process.stderr = Pipe(process.stderr, 'stderr')
        real_wait = process.wait
        def wait(*args, **kwargs):
            events.append('reap')
            assert kwargs['timeout'] == cleanup.CLEANUP_REAP_SECONDS
            return real_wait(*args, **kwargs)
        monkeypatch.setattr(process, 'wait', wait)
        return process

    class Selector:
        def __init__(self):
            self.inner = real_selector()
        def select(self, timeout):
            phase = 'final' if signal.SIGKILL in signals else 'grace' if signals else 'running'
            wanted = ('grace' if fault in ('grace_interrupt', 'custom_base') else 'final')
            if phase == wanted and 'drain' not in injected:
                if interrupt_case or fault == 'final_error':
                    injected.add('drain')
                    raise interrupt if interrupt_case else OSError('final drain failed')
            return self.inner.select(timeout)
        def close(self):
            events.append('selector close')
            self.inner.close()
        def __getattr__(self, name):
            return getattr(self.inner, name)

    def send(pid, sig):
        assert pid == processes[0].pid
        signals.append(sig)
        if sig == signal.SIGTERM and fault == 'formatting':
            injected.add('formatting')
            raise UnprintableError()
        return real_signal(pid, sig)

    monkeypatch.setattr(cleanup.subprocess, 'Popen', popen)
    monkeypatch.setattr(cleanup.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(cleanup, '_signal_group', send)
    start = time.monotonic()
    try:
        with pytest.raises(type(interrupt) if interrupt_case else subprocess.TimeoutExpired) as caught:
            cleanup.run_cleanup_command([sys.executable, '-c', program], timeout=.2)
        assert time.monotonic() - start < 5
        assert injected
        if interrupt_case:
            assert caught.value is interrupt  # Preserve the original interruption.
        elif fault in ('formatting', 'close_formatting'):
            assert 'cleanup diagnostic recording failed' in caught.value.cleanup_shutdown_errors[-1]
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        assert 'close stdout' in events and 'close stderr' in events
        assert 'selector close' in events and events[-1] == 'reap'
        assert processes[0].returncode is not None
        with pytest.raises(ChildProcessError):
            os.waitpid(processes[0].pid, os.WNOHANG)
        parent, child = json.loads(pidfile.read_text())
        _wait_dead(child)
        assert unrelated.poll() is None
        assert processes[0].stdout.pipe.closed and processes[0].stderr.pipe.closed
    finally:
        unrelated.terminate()
        real_popen.wait(unrelated, timeout=2)

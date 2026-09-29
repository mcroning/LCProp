"""Exercise the real SSH serialization through a local POSIX-shell surrogate."""
import json
import subprocess
import sys

import pytest

from lcprop.runners.cluster_connection import (
    ConnectionTestRemoteTransport, _login_shell_command,
)
from lcprop.runners.slurm import SubprocessRemoteTransport
from lcprop.runners.distribution_deployment import InstalledDeploymentManager, VERIFY_PROGRAM
from lcprop.runners.source_deployment import SourceDeploymentError
from tests.test_distribution_deployment import installed, resolve


@pytest.fixture
def local_ssh(monkeypatch):
    """Model OpenSSH joining its command operands, without invoking SSH."""
    run, popen = subprocess.run, subprocess.Popen
    calls = []

    def translate(argv):
        if argv[0] == 'ssh':
            assert argv[:4] == ['ssh', '-o', 'BatchMode=yes', 'example.invalid']
            calls.append(argv)
            return ['/bin/sh', '-c', ' '.join(argv[4:])]
        return argv

    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw: run(translate(argv), **kw))
    monkeypatch.setattr(subprocess, 'Popen', lambda argv, **kw: popen(translate(argv), **kw))
    return calls


@pytest.mark.parametrize('bounded', [False, True])
@pytest.mark.parametrize('arguments', [
    ['ordinary', '--leading-option', '-x'],
    ['two words', '/path with spaces/file'],
    ['semi;colon', "single'quote", 'double"quote'],
    ['line one\nline two', 'tab\there'],
    ['$HOME', '$(printf substituted)', '`printf substituted`', '*?[a-z]', 'a&b|c>d<e', r'a\b'],
    ['', ' ', ''],
])
def test_remote_argv_round_trip(local_ssh, bounded, arguments):
    argv = (sys.executable, '-I', '-S', '-c',
            'import json,sys\nprint(json.dumps(sys.argv[1:]))\n', *arguments)
    if bounded:
        result = ConnectionTestRemoteTransport().ssh_with_timeout(
            'example.invalid', *argv, timeout_seconds=10)
    else:
        result = SubprocessRemoteTransport().ssh('example.invalid', *argv)
    assert json.loads(result) == arguments
    assert len(local_ssh[-1]) == 5  # one serialized remote command


def test_explicit_shell_payload_is_not_double_quoted(local_ssh):
    payload = _login_shell_command((
        "export LCPROP_TEST_LABEL='space; dollar $ and quote'",
        "printf '%s' \"$LCPROP_TEST_LABEL\"",
    ))
    assert ConnectionTestRemoteTransport().ssh_with_timeout(
        'example.invalid', *payload, timeout_seconds=10
    ) == 'space; dollar $ and quote'


@pytest.mark.parametrize('arguments', [(), ('',), ('printf', 'bad\0argument')])
def test_unsupported_command_rejected_before_ssh(local_ssh, arguments):
    with pytest.raises(ValueError):
        SubprocessRemoteTransport().ssh('example.invalid', *arguments)
    assert not local_ssh


def test_scp_argument_construction_is_unchanged(monkeypatch, tmp_path):
    commands = []
    transport = SubprocessRemoteTransport()
    monkeypatch.setattr(transport, '_run', lambda argv: commands.append(argv))
    local = tmp_path / 'local file'
    transport.upload('example.invalid', local, '/remote/path')
    transport.download('example.invalid', '/remote/path', tmp_path / 'download')
    assert commands == [
        ['scp', '-q', '-r', str(local), 'example.invalid:/remote/path'],
        ['scp', '-q', '-r', 'example.invalid:/remote/path', str(tmp_path / 'download')],
    ]


def snapshot(installed, tmp_path):
    artifact = resolve(installed)
    target = tmp_path / "snapshot with 'quotes' $ and ; spaces"
    for name, data in artifact.files:
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (target / '.lcprop-artifact.json').write_text(json.dumps(artifact.manifest))
    return target, artifact.digest


def verifier_manager(installed):
    return InstalledDeploymentManager(
        host='example.invalid', source_root='/unused', remote_python=sys.executable,
        package_file=installed / 'lcprop/__init__.py', transport=SubprocessRemoteTransport())


def test_actual_multiline_verifier_survives_remote_shell(local_ssh, installed, tmp_path):
    target, digest = snapshot(installed, tmp_path)
    # The former argv-to-OpenSSH contract fails at the shell before verification.
    broken = subprocess.run(['/bin/sh', '-c', ' '.join(
        [sys.executable, '-I', '-S', '-c', VERIFY_PROGRAM, str(target), digest])],
        text=True, capture_output=True)
    assert broken.returncode != 0
    assert SubprocessRemoteTransport().ssh(
        'example.invalid', sys.executable, '-I', '-S', '-c', VERIFY_PROGRAM,
        str(target), digest) == digest
    assert verifier_manager(installed)._verify_distribution(str(target), digest)


def test_content_mismatch_has_explicit_identity_classification(local_ssh, installed, tmp_path):
    target, digest = snapshot(installed, tmp_path)
    (target / 'src/lcprop/__init__.py').write_text('modified artifact')
    with pytest.raises(SourceDeploymentError) as error:
        verifier_manager(installed)._verify_distribution(str(target), digest)
    assert error.value.category == 'snapshot_identity_mismatch'
    assert 'LCPROP_ARTIFACT_IDENTITY_FAILURE: payload size mismatch' in str(error.value)
    assert error.value.__cause__.returncode == 65


@pytest.mark.parametrize('failure', ['missing-python', 'syntax', 'runtime', 'response', 'unmarked-65', 'ssh'])
def test_verifier_execution_failure_is_not_identity_mismatch(local_ssh, installed, tmp_path, monkeypatch, failure):
    target, digest = snapshot(installed, tmp_path)
    manager = verifier_manager(installed)
    expected = ''
    if failure == 'missing-python':
        manager.remote_python = str(tmp_path / 'missing-python')
        expected = 'missing-python'
    elif failure == 'ssh':
        def fail(*args):
            raise subprocess.CalledProcessError(255, ['ssh'], stderr='connection lost')
        monkeypatch.setattr(manager._transport, 'ssh', fail)
        expected = 'connection lost'
    else:
        programs = {
            'syntax': ('not valid python!', 'SyntaxError'),
            'runtime': ('raise RuntimeError("technical failure")', 'technical failure'),
            'response': ('print("wrong response")', 'unexpected verification response'),
            'unmarked-65': ('import sys; print("unrelated error",file=sys.stderr); sys.exit(65)', 'unrelated error'),
        }
        program, expected = programs[failure]
        monkeypatch.setattr('lcprop.runners.distribution_deployment.REMOTE_VERIFY_PROGRAM', program)
    with pytest.raises(SourceDeploymentError) as error:
        manager._verify_distribution(str(target), digest)
    assert error.value.category == 'snapshot_verification_failed'
    assert expected in str(error.value)
    stderr = getattr(error.value.__cause__, 'stderr', None)
    if failure != 'response':
        assert stderr and '\nstderr:\n' + stderr in str(error.value)
    if failure == 'missing-python':
        assert error.value.__cause__.returncode == 127

"""Local-only installed-product and fake-remote deployment acceptance tests."""
import base64
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from lcprop.runners.distribution_deployment import (
    InstalledDeploymentManager, VERIFY_PROGRAM, resolve_installed_distribution,
)
from lcprop.runners.source_deployment import SourceDeploymentError
from tests.test_source_deployment import LocalRemoteTransport, _repository, _run


def set_active_package(monkeypatch, package):
    import lcprop
    from copy import copy
    spec = copy(lcprop.__spec__)
    spec.origin = str(package / '__init__.py')
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr(lcprop, '__file__', spec.origin)
    monkeypatch.setattr(lcprop, '__path__', [str(package)])
    monkeypatch.setattr(lcprop, '__spec__', spec)


def record(root):
    info = root / 'lcprop-0.1.0.dist-info'
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name != 'RECORD':
            b = p.read_bytes()
            rows.append([p.relative_to(root).as_posix(), 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).decode().rstrip('='), str(len(b))])
    rows.append([info.name + '/RECORD', '', ''])
    with (info / 'RECORD').open('w', newline='') as f:
        csv.writer(f).writerows(rows)


@pytest.fixture
def installed(tmp_path):
    root = tmp_path / 'site'; (root / 'lcprop/assets').mkdir(parents=True)
    (root / 'lcprop/__init__.py').write_text('VALUE = 1\n')
    (root / 'lcprop/assets/data.json').write_text('{"data":1}\n')
    info = root / 'lcprop-0.1.0.dist-info'; info.mkdir()
    (info / 'METADATA').write_text('Metadata-Version: 2.4\nName: lcprop\nVersion: 0.1.0\nRequires-Python: >=3.10\n')
    (info / 'WHEEL').write_text('Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
    record(root)
    return root


def resolve(root):
    return resolve_installed_distribution(root / 'lcprop/__init__.py', [metadata.PathDistribution(root / 'lcprop-0.1.0.dist-info')])


class Remote(LocalRemoteTransport):
    def ssh(self, host, *args):
        if args[0] == sys.executable:
            self.commands.append((host, args))
            return subprocess.run([sys.executable, '-B', '-c', args[args.index('-c')+1], str(self._path(args[-2])), args[-1]], text=True, capture_output=True, check=True).stdout
        return super().ssh(host, *args)


def manager(root, remote):
    return InstalledDeploymentManager(host='example.invalid', source_root='/sources',
        package_file=root / 'lcprop/__init__.py', remote_python=sys.executable,
        distributions=[metadata.PathDistribution(root / 'lcprop-0.1.0.dist-info')], transport=remote)


def test_deterministic_content_identity_excludes_installation_state(installed, tmp_path):
    first = resolve(installed)
    other = tmp_path / 'other'; shutil.copytree(installed, other)
    (other / 'lcprop/__pycache__').mkdir()
    (other / 'lcprop/__pycache__/cache.pyc').write_bytes(b'not deployed')
    second = resolve(other)
    assert first.digest == second.digest
    assert first.archive(tmp_path / 'one.tar') == second.archive(tmp_path / 'two.tar')
    assert (tmp_path / 'one.tar').read_bytes() == (tmp_path / 'two.tar').read_bytes()
    with tarfile.open(tmp_path / 'one.tar') as bundle:
        assert all('__pycache__' not in n and 'RECORD' not in n for n in bundle.getnames())
    (other / 'lcprop/assets/data.json').write_text('{"data":2}\n'); record(other)
    assert resolve(other).version == first.version
    assert resolve(other).digest != first.digest


@pytest.mark.parametrize('problem', ['modified', 'missing', 'unowned', 'symlink', 'native', 'traversal', 'duplicate', 'unhashed', 'editable', 'binary-wheel'])
def test_invalid_installations_fail_closed(installed, problem):
    p = installed / 'lcprop/assets/data.json'; rec = installed / 'lcprop-0.1.0.dist-info/RECORD'
    if problem == 'modified': p.write_text('changed')
    if problem == 'missing': p.unlink()
    if problem == 'unowned': (installed / 'lcprop/extra.py').write_text('bad')
    if problem == 'symlink': p.unlink(); p.symlink_to(installed / 'lcprop/__init__.py')
    if problem == 'native': (installed / 'lcprop/native.so').write_bytes(b'ELF'); record(installed)
    if problem == 'traversal': rec.write_text(rec.read_text() + '../secret,,\n')
    if problem == 'duplicate': rec.write_text(rec.read_text() + rec.read_text().splitlines()[0] + '\n')
    if problem == 'unhashed':
        rows=list(csv.reader(rec.open())); rows[0][1]=''
        with rec.open('w', newline='') as f: csv.writer(f).writerows(rows)
    if problem == 'editable':
        (rec.parent / 'direct_url.json').write_text('{"dir_info":{"editable":true}}'); record(installed)
    if problem == 'binary-wheel':
        (rec.parent / 'WHEEL').write_text('Root-Is-Purelib: false\nTag: cp312-cp312-macosx_11_0_arm64\n'); record(installed)
    with pytest.raises(SourceDeploymentError): resolve(installed)


def test_active_import_ownership(installed, tmp_path):
    with pytest.raises(SourceDeploymentError, match='ownership|matches'):
        resolve_installed_distribution(tmp_path / 'other/lcprop/__init__.py', [metadata.PathDistribution(installed / 'lcprop-0.1.0.dist-info')])
    dist=metadata.PathDistribution(installed / 'lcprop-0.1.0.dist-info')
    with pytest.raises(SourceDeploymentError, match='ambiguous'):
        resolve_installed_distribution(installed / 'lcprop/__init__.py', [dist,dist])


def test_staging_reuse_and_corrupt_cached_payload(installed, tmp_path):
    remote=Remote(tmp_path / 'remote'); m=manager(installed,remote)
    m.preflight(); assert remote.commands == []
    first=m.resolve_or_stage(); uploads=remote.upload_count
    assert first.source_git_sha is None
    assert '/dist-sha256-' in first.remote_source_path
    assert not (remote._path(first.remote_source_path)/'.lcprop-source-sha').exists()
    assert m.resolve_or_stage().reused_existing_snapshot
    assert remote.upload_count==uploads
    (remote._path(first.remote_source_path)/'src/lcprop/__init__.py').write_text('changed')
    with pytest.raises(SourceDeploymentError, match='snapshot_identity_mismatch'):m.resolve_or_stage()
    assert remote.upload_count==uploads


def test_binding_rejects_change_after_preflight(installed, tmp_path):
    remote=Remote(tmp_path/'remote');m=manager(installed,remote);binding=m.preflight()
    (installed/'lcprop/assets/data.json').write_text('{"change":true}');record(installed)
    with pytest.raises(SourceDeploymentError,match='source_changed'):m.resolve_or_stage(binding=binding)
    assert remote.upload_count==0


@pytest.mark.parametrize('failure',['upload','extract'])
def test_failed_distribution_staging_never_publishes(installed,tmp_path,failure):
    remote=Remote(tmp_path/'remote');remote.corrupt_archive_upload=failure=='upload';remote.fail_extract=failure=='extract'
    with pytest.raises(SourceDeploymentError):manager(installed,remote).resolve_or_stage()
    assert not list((tmp_path/'remote/sources').glob('dist-sha256-*'))


def test_concurrent_publisher_verified(installed,tmp_path):
    remote=Remote(tmp_path/'remote');remote.simulate_winning_finalizer=True
    assert manager(installed,remote).resolve_or_stage().reused_existing_snapshot


@pytest.mark.parametrize('failure', ['ssh', 'command', 'command-not-found', 'os-error', 'timeout', 'absent'])
def test_staging_existence_probe_preserves_unknown_vs_absent(installed, tmp_path, failure):
    class ProbeRemote(Remote):
        probed_after_extraction = False
        original_error = None

        def ssh(self, host, *args):
            if args[:2] == ('test', '-e') and '/.staging-' in args[2]:
                # Exercise resolve_or_stage -> _stage -> verifier, after real
                # local archive extraction. Do not stub either existence helper.
                path = self._path(args[2])
                assert (path / 'src/lcprop/__init__.py').is_file()
                assert any(command[0] == 'tar' for _, command in self.commands)
                self.probed_after_extraction = True
                if failure == 'absent':
                    shutil.rmtree(path)
                    return super().ssh(host, *args)
                if failure == 'os-error':
                    error = OSError('probe transport could not start')
                elif failure == 'timeout':
                    error = subprocess.TimeoutExpired(args, 10, stderr=b'probe timed out')
                else:
                    code = {'ssh': 255, 'command': 2, 'command-not-found': 127}[failure]
                    error = subprocess.CalledProcessError(code, args, stderr='probe execution failed')
                self.original_error = error
                raise error
            return super().ssh(host, *args)

    remote = ProbeRemote(tmp_path / 'remote')
    with pytest.raises(SourceDeploymentError) as caught:
        manager(installed, remote).resolve_or_stage()
    assert remote.probed_after_extraction
    assert not list((tmp_path / 'remote/sources').glob('dist-sha256-*'))
    if failure == 'absent':
        assert caught.value.category == 'snapshot_identity_mismatch'
        assert 'staging disappeared' in str(caught.value)
    else:
        assert caught.value.category == 'snapshot_verification_failed'
        assert caught.value.__cause__ is remote.original_error
        assert 'probe' in str(caught.value)
        stderr = getattr(remote.original_error, 'stderr', None)
        if stderr:
            if isinstance(stderr, bytes):
                stderr = stderr.decode()
            assert stderr in str(caught.value)


def test_checked_snapshot_absence_still_allows_first_stage(installed, tmp_path):
    remote = Remote(tmp_path / 'remote')
    deployment = manager(installed, remote)
    assert deployment._verify_distribution('/sources/missing', resolve(installed).digest) is False
    assert remote.upload_count == 0
    result = deployment.resolve_or_stage()
    assert not result.reused_existing_snapshot
    assert deployment.resolve_or_stage().reused_existing_snapshot


@pytest.mark.parametrize(('stdout', 'stderr'), [
    ('', ''),
    ('', '** WARNING: connection is not using a post-quantum key exchange algorithm.\n'
         '** This session may be vulnerable to "store now, decrypt later" attacks.\n'
         '** The server may need to be upgraded. See https://openssh.com/pq.html\n'),
    ('site login banner\n', ''),
    ('site login banner\n', 'Warning: SSH connection diagnostic\n'),
])
def test_first_distribution_deployment_cache_miss_publishes(installed, tmp_path, stdout, stderr):
    class FirstDeploymentRemote(Remote):
        cache_misses = 0
        staging_checks = 0

        def ssh(self, host, *args):
            if args[:2] == ('test', '-e'):
                path = self._path(args[2])
                if path.name.startswith('dist-sha256-') and not path.exists():
                    self.cache_misses += 1
                    self.commands.append((host, args))
                    raise subprocess.CalledProcessError(1, args, output=stdout, stderr=stderr)
                if path.name.startswith('.staging-'):
                    assert (path / 'src/lcprop/__init__.py').is_file()
                    self.staging_checks += 1
            return super().ssh(host, *args)

    remote = FirstDeploymentRemote(tmp_path / 'remote')
    deployment = manager(installed, remote)
    result = deployment.resolve_or_stage()
    assert remote.cache_misses == 1
    assert remote.upload_count == 2  # archive and manifest
    assert remote.staging_checks == 1
    commands = [args for _, args in remote.commands]
    assert any(args[0] == 'tar' for args in commands)
    assert any(args[0] == sys.executable and '/.staging-' in args[-2] for args in commands)
    assert any(args[0] == 'ln' for args in commands)
    published = remote._path(result.remote_source_path)
    assert published.is_symlink() and (published / 'src/lcprop/__init__.py').is_file()
    assert not result.reused_existing_snapshot
    uploads = remote.upload_count
    assert deployment.resolve_or_stage().reused_existing_snapshot
    assert remote.upload_count == uploads


@pytest.mark.parametrize('failure', ['ssh', 'command', 'os-error', 'timeout'])
def test_published_cache_probe_failure_never_stages(installed, tmp_path, failure):
    class FailedCacheRemote(Remote):
        def ssh(self, host, *args):
            assert args[:2] == ('test', '-e')
            assert '/dist-sha256-' in args[2]
            if failure == 'os-error':
                raise OSError('cache probe could not start')
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(args, 10, stderr='cache probe timed out')
            raise subprocess.CalledProcessError(
                255 if failure == 'ssh' else 2, args, stderr='cache probe execution failed')

    remote = FailedCacheRemote(tmp_path / 'remote')
    with pytest.raises(SourceDeploymentError) as caught:
        manager(installed, remote).resolve_or_stage()
    assert caught.value.category == 'snapshot_verification_failed'
    assert 'cache probe' in str(caught.value)
    assert caught.value.__cause__ is not None
    assert remote.upload_count == 0


def test_git_identity_bound_and_never_falls_back(tmp_path):
    from lcprop.runners.source_deployment import SourceDeploymentManager
    root,_=_repository(tmp_path);m=SourceDeploymentManager(host='example.invalid',source_root='/sources',local_source=root,transport=Remote(tmp_path/'remote'))
    binding=m.preflight();(root/'src/lcprop/__init__.py').write_text('VALUE=2\n')
    with pytest.raises(SourceDeploymentError,match='source_dirty'):m.resolve_or_stage(binding=binding)
    _run('git','add','src',cwd=root);_run('git','commit','-qm','new',cwd=root)
    with pytest.raises(SourceDeploymentError,match='source_changed'):m.resolve_or_stage(binding=binding)


def test_gui_preflight_and_fake_runner_reach_submission_preparation(installed,tmp_path):
    from types import SimpleNamespace
    from lcprop.gui.request_transparency import source_preflight
    from lcprop.runners.slurm import SlurmRunner
    from lcprop.transport.defaults import default_transport_registry
    from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
    from tests.test_slurm_runner import _config, _pr_td_request
    class StopBeforeSubmit(Remote):
        def ssh(self,host,*args):
            if args[0]=='sbatch':
                assert 'artifact' in self._path(args[-1]).read_text()
                raise RuntimeError('submission preparation reached; no job executed')
            return super().ssh(host,*args)
        def upload(self,host,local,remote):
            if Path(local).is_dir():
                shutil.copytree(local,self._path(remote)/Path(local).name)
                return
            super().upload(host,local,remote)
    remote=StopBeforeSubmit(tmp_path/'remote');m=manager(installed,remote)
    runner=SlurmRunner(_config(tmp_path,remote_source_path=None,source_git_sha=None),
        (PR_TIMEDEPENDENT_OPERATION,),transport=remote,source_deployment_manager=m,
        registry=default_transport_registry())
    window=SimpleNamespace(runner=runner,slurm_runner=runner)
    source_preflight(window);assert not remote.commands
    with pytest.raises(RuntimeError,match='submission preparation reached'):
        runner.run_registered('pr','pr_timedependent',_pr_td_request())


def test_real_wheel_install_without_wheel_or_checkout(tmp_path):
    """Build/install locally; child resolves its real active import, no wheel remains."""
    repo=Path(__file__).resolve().parents[1];project=tmp_path/'build';project.mkdir()
    shutil.copytree(repo/'src/lcprop',project/'src/lcprop',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['pyproject.toml','README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
        shutil.copyfile(repo/name,project/name)
    wheels=tmp_path/'wheels'
    subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-index','-w',str(wheels),str(project)],check=True,capture_output=True)
    env=tmp_path/'env'
    subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True,capture_output=True)
    python=env/'bin/python';wheel=next(wheels.glob('*.whl'))
    subprocess.run([str(python),'-m','pip','install','--no-deps','--no-index','--ignore-installed',str(wheel)],check=True,capture_output=True)
    shutil.rmtree(wheels);shutil.rmtree(project)
    code='''
import lcprop,sys,json,tarfile,pathlib,subprocess,os
from lcprop.runners.distribution_deployment import automatic_deployment_manager, VERIFY_PROGRAM, BOOTSTRAP_PROGRAM
from lcprop.runners.slurm import SlurmRunner
from lcprop.gui.request_transparency import source_preflight
from types import SimpleNamespace
assert 'site-packages' in lcprop.__file__ and lcprop.__file__.startswith(sys.prefix)
m=automatic_deployment_manager(host='example.invalid',source_root='/sources',remote_python=sys.executable)
r=object.__new__(SlurmRunner);r.config=SimpleNamespace(remote_source_path=None);r._source_deployment_manager=m
source_preflight(SimpleNamespace(runner=r,slurm_runner=r))
a=m.preflight().identity;dest=pathlib.Path('payload');dest.mkdir();a.archive('payload.tar')
with tarfile.open('payload.tar') as t:t.extractall(dest,filter='data')
(dest/'.lcprop-artifact.json').write_text(json.dumps(a.manifest))
subprocess.run([sys.executable,'-B','-c',VERIFY_PROGRAM,str(dest),a.digest],check=True)
subprocess.run([sys.executable,'-I','-S','-c',BOOTSTRAP_PROGRAM+'\\nimport lcprop.transport.executor',str(dest),a.digest],check=True)
child_env={**os.environ,'PYTHONPATH':str((dest/'src').resolve()),'PYTHONDONTWRITEBYTECODE':'1'}
origin='import lcprop,pathlib; assert pathlib.Path(lcprop.__file__).resolve()==pathlib.Path("src/lcprop/__init__.py").resolve(); import lcprop.transport.executor'
subprocess.run([sys.executable,'-B','-c',origin],cwd=dest,env=child_env,check=True)
subprocess.run([sys.executable,'-B','-m','lcprop.transport.executor','--help'],cwd=dest,env=child_env,check=True)
print(json.dumps({'digest':a.digest,'import':lcprop.__file__}))
'''
    envvars={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'};envvars.pop('PYTHONPATH',None)
    run=subprocess.run([str(python),'-B','-c',code],cwd=tmp_path,env=envvars,text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'site-packages' in run.stdout


@pytest.mark.parametrize('problem', ['dirty', 'staged', 'untracked', 'symlink', 'submodule', 'lfs'])
def test_automatic_checkout_provider_never_falls_back(tmp_path, monkeypatch, problem):
    import lcprop
    from lcprop.runners.distribution_deployment import automatic_deployment_manager
    from lcprop.runners.source_deployment import SourceDeploymentManager

    root, sha = _repository(tmp_path)
    package = root / 'src/lcprop'
    if problem in ('dirty', 'staged'):
        (package / '__init__.py').write_text('CHANGED = True\n')
        if problem == 'staged':
            _run('git', 'add', 'src', cwd=root)
    elif problem == 'untracked':
        (package / 'extra.py').write_text('EXTRA = True\n')
    elif problem == 'symlink':
        (package / 'link.py').symlink_to('__init__.py')
        _run('git', 'add', 'src', cwd=root)
        _run('git', 'commit', '-qm', 'link fixture', cwd=root)
    elif problem == 'submodule':
        _run('git', 'update-index', '--add', '--cacheinfo',
             '160000', sha, 'src/lcprop/submodule', cwd=root)
        _run('git', 'commit', '-qm', 'gitlink fixture', cwd=root)
    elif problem == 'lfs':
        (package / 'asset.png').write_text(
            'version https://git-lfs.github.com/spec/v1\n'
            'oid sha256:' + '0' * 64 + '\nsize 9999\n')
        _run('git', 'add', 'src', cwd=root)
        _run('git', 'commit', '-qm', 'LFS fixture', cwd=root)
    set_active_package(monkeypatch, package)
    # A source import must not consult a different installed copy at all.
    def unexpected_owner_lookup(*args, **kwargs):
        raise AssertionError('Git provider selection attempted distribution fallback')
    monkeypatch.setattr('lcprop.runners.distribution_deployment.owning_distribution',
                        unexpected_owner_lookup)
    deployment = automatic_deployment_manager(
        host='example.invalid', source_root='/sources', remote_python=sys.executable)
    assert type(deployment) is SourceDeploymentManager
    with pytest.raises(SourceDeploymentError):
        deployment.preflight()


def test_remote_verifier_rejects_tampering_even_with_optimization(installed, tmp_path):
    artifact = resolve(installed)
    dest = tmp_path / 'snapshot'
    for name, data in artifact.files:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (dest / '.lcprop-artifact.json').write_text(json.dumps(artifact.manifest))
    command = [sys.executable, '-B', '-O', '-c', VERIFY_PROGRAM, str(dest), artifact.digest]
    subprocess.run(command, check=True, capture_output=True)
    (dest / 'src/lcprop/__init__.py').write_text('tampered')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'payload size mismatch' in result.stderr


def test_generated_job_uses_isolated_same_process_launcher(tmp_path):
    from lcprop.runners.slurm import SlurmRunner
    from lcprop.transport.defaults import default_transport_registry
    from lcprop.runners.distribution_deployment import LAUNCH_PROGRAM
    from tests.test_slurm_runner import _config, CPU_SMALL
    import shlex

    runner = SlurmRunner(_config(tmp_path, remote_python=sys.executable), (),
                         registry=default_transport_registry())
    script = runner._script('/runs/not-executed', CPU_SMALL,
                            remote_source_path='/snapshot', artifact_digest='a' * 64)
    assert '-I -S -c ' + shlex.quote(LAUNCH_PROGRAM) in script
    assert "runpy.run_module" in script


def test_executor_records_distribution_identity_without_git_sha():
    from lcprop.transport.executor import _provenance

    source = {'source_kind': 'installed_distribution', 'source_git_sha': None,
              'source_content_sha256': 'a' * 64, 'distribution_version': '0.1.0'}
    provenance = _provenance(source)
    assert provenance['source_artifact'] == source
    assert provenance['python_executable'] == sys.executable
    assert provenance['lcprop_import_origin'].endswith('lcprop/__init__.py')
    assert 'numpy' in provenance['dependencies']
    assert 'source_artifact' not in _provenance({'source_kind': 'git_snapshot'})

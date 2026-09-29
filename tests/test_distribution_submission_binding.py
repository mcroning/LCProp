"""Regression tests for active ownership and submission-scoped execution binding."""
from dataclasses import replace
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest

from lcprop.runners.distribution_deployment import (
    automatic_deployment_manager, LAUNCH_PROGRAM, VERIFY_PROGRAM,
)
from lcprop.runners.source_deployment import SourceDeploymentError, SourceDeploymentManager
from tests.test_distribution_deployment import installed, record, resolve, manager, Remote, set_active_package
from tests.test_source_deployment import _repository, _run


@pytest.mark.parametrize('location', ['.venv/lib/site-packages', 'build/lib'])
def test_nested_unowned_import_never_deploys_enclosing_git(tmp_path, monkeypatch, location):
    import lcprop
    root, _ = _repository(tmp_path)
    package = root / location / 'lcprop'
    package.mkdir(parents=True)
    active = package / '__init__.py'
    active.write_text('DIFFERENT_PRODUCT = True\n')
    set_active_package(monkeypatch, package)
    with pytest.raises(SourceDeploymentError, match='neither the supported Git source tree'):
        automatic_deployment_manager(host='example.invalid', source_root='/sources',
                                     remote_python=sys.executable)


def test_valid_installed_distribution_nested_in_checkout_keeps_its_own_identity(installed, tmp_path, monkeypatch):
    import lcprop
    from importlib import metadata
    from lcprop.runners.distribution_deployment import InstalledDeploymentManager
    root, _ = _repository(tmp_path)
    site = root / '.venv/site-packages'
    shutil.copytree(installed, site)
    set_active_package(monkeypatch, site / 'lcprop')
    monkeypatch.setattr(metadata, 'distributions', lambda: [metadata.PathDistribution(site / 'lcprop-0.1.0.dist-info')])
    selected = automatic_deployment_manager(host='example.invalid', source_root='/sources', remote_python=sys.executable)
    assert isinstance(selected, InstalledDeploymentManager)
    assert selected.preflight().identity.digest == resolve(installed).digest


@pytest.mark.parametrize('kind', ['git', 'distribution'])
def test_submission_bindings_can_advance_without_weakening_in_attempt_checks(installed, tmp_path, kind):
    remote = Remote(tmp_path / 'remote')
    if kind == 'git':
        root, _ = _repository(tmp_path)
        deployment = SourceDeploymentManager(host='example.invalid', source_root='/sources', local_source=root, transport=remote)
        def advance():
            (root / 'src/lcprop/__init__.py').write_text('NEXT = True\n')
            _run('git', 'add', 'src', cwd=root)
            _run('git', 'commit', '-qm', 'next fixture identity', cwd=root)
    else:
        deployment = manager(installed, remote)
        def advance():
            (installed / 'lcprop/__init__.py').write_text('NEXT = True\n')
            record(installed)
    first = deployment.preflight()
    completed = deployment.resolve_or_stage(binding=first)
    advance()
    with pytest.raises(SourceDeploymentError, match='source_changed'):
        deployment.resolve_or_stage(binding=first)
    second = deployment.preflight()
    updated = deployment.resolve_or_stage(binding=second)
    assert updated.remote_source_path != completed.remote_source_path
    assert deployment.resolve_or_stage(binding=second).reused_existing_snapshot


def test_runner_consumes_and_abandons_one_attempt_at_a_time(installed, tmp_path, monkeypatch):
    from lcprop.runners.slurm import SlurmRunner
    from lcprop.transport.defaults import default_transport_registry
    from tests.test_slurm_runner import _config
    deployment = manager(installed, Remote(tmp_path / 'remote'))
    runner = SlurmRunner(_config(tmp_path, remote_source_path=None, source_git_sha=None), (),
                         registry=default_transport_registry(), source_deployment_manager=deployment)
    def attempt(*args, source_submission, **kwargs):
        return deployment.resolve_or_stage(binding=source_submission)
    monkeypatch.setattr(runner, '_run_registered_attempt', attempt)
    runner.preflight_source()
    first = runner.run_registered('test', 'test', None)
    runner.preflight_source()
    (installed / 'lcprop/__init__.py').write_text('B = True\n')
    record(installed)
    with pytest.raises(SourceDeploymentError, match='source_changed'):
        runner.run_registered('test', 'test', None)
    assert runner._pending_source_submission is None
    runner.preflight_source()
    second = runner.run_registered('test', 'test', None)
    assert second.remote_source_path != first.remote_source_path
    runner.preflight_source()
    runner.abandon_source_submission()
    assert runner._pending_source_submission is None


def _stage(artifact, destination):
    for name, data in artifact.files:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination / '.lcprop-artifact.json').write_text(json.dumps(artifact.manifest))


@pytest.fixture
def executor_artifact(installed):
    """Real executor/transport, with a verified test-only dispatch sentinel."""
    source = Path(__file__).resolve().parents[1] / 'src/lcprop'
    shutil.rmtree(installed / 'lcprop')
    for path in source.rglob('*'):
        if path.suffix in ('.py', '.json', '.png') and '__pycache__' not in path.parts:
            target = installed / 'lcprop' / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    executor = installed / 'lcprop/transport/executor.py'
    text = executor.read_text()
    text = text.replace('        runner = LocalRunner(operations)',
                        '        (run_dir / "dispatch.txt").write_text("verified dispatch")\n'
                        '        return 0\n        runner = LocalRunner(operations)')
    text = text.replace('        from lcprop.runners.distribution_deployment import verify_execution_binding',
                        '        if os.environ.get("LCPROP_TEST_TAMPER"):\n'
                        '            Path(__file__).write_text(Path(__file__).read_text() + "\\n# changed\\n")\n'
                        '        from lcprop.runners.distribution_deployment import verify_execution_binding')
    executor.write_text(text)
    record(installed)
    return resolve(installed)


@pytest.mark.parametrize('scenario', ['valid', 'retarget', 'tamper', 'user-site', 'unbound', 'request-mismatch', 'in-process-tamper'])
def test_real_executor_launch_binding_precedes_dispatch(executor_artifact, tmp_path, scenario):
    from lcprop.transport.io import write_request_package
    from lcprop.transport.defaults import default_transport_registry
    from tests.test_slurm_runner import _request
    artifact = executor_artifact
    physical = tmp_path / 'physical'
    _stage(artifact, physical)
    link = tmp_path / 'published'
    link.symlink_to(physical, target_is_directory=True)
    run = tmp_path / 'run'
    run.mkdir()
    write_request_package(run, registry=default_transport_registry(), material_id='lc',
                          workflow_id='static', request=_request(), run_id='binding-test',
                          resource_profile='CPU small', provenance={
                              'source_kind': 'installed_distribution',
                              'source_content_sha256': '0' * 64 if scenario == 'request-mismatch' else artifact.digest,
                              'remote_source_path': str(link),
                          })
    # This succeeds for A, before the substitution the review identified.
    subprocess.run([sys.executable, '-I', '-S', '-c', VERIFY_PROGRAM, str(link), artifact.digest],
                   check=True, capture_output=True)
    if scenario == 'retarget':
        files = tuple((n, b + b'\nCHANGED=True\n' if n == 'src/lcprop/__init__.py' else b)
                      for n, b in artifact.files)
        alternative = tmp_path / 'alternative'
        _stage(replace(artifact, files=files), alternative)
        link.unlink()
        link.symlink_to(alternative, target_is_directory=True)
    if scenario == 'tamper':
        (physical / 'src/lcprop/__init__.py').write_text('CHANGED=True\n')
    env = dict(os.environ)
    if scenario == 'in-process-tamper':
        env['LCPROP_TEST_TAMPER'] = '1'
    if scenario == 'user-site':
        hostile = tmp_path / 'hostile'
        hostile.mkdir()
        (hostile / 'sitecustomize.py').write_text('raise RuntimeError("startup contamination")\n')
        (hostile / 'lcprop.py').write_text('raise RuntimeError("wrong LCProp")\n')
        env.update(PYTHONPATH=str(hostile), PYTHONUSERBASE=str(hostile))
        (physical / 'lcprop.py').write_text('raise RuntimeError("cwd shadow")\n')
    command = [sys.executable, '-I', '-S', '-c', LAUNCH_PROGRAM, str(link), artifact.digest, str(run)]
    if scenario == 'unbound':
        env.update(PYTHONPATH=str(physical / 'src'), PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, '-B', '-m', 'lcprop.transport.executor', '--run-dir', str(run)]
    completed = subprocess.run(command, cwd=physical, env=env, text=True, capture_output=True)
    if scenario in ('valid', 'user-site'):
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (run / 'dispatch.txt').read_text() == 'verified dispatch'
    else:
        assert completed.returncode != 0, completed.stdout + completed.stderr
        assert not (run / 'dispatch.txt').exists()


@pytest.mark.parametrize('scenario', [
    'linked_initializer', 'hardlinked_initializer', 'extended_path',
    'redirected_path', 'extended_spec', 'canonical', 'detached',
])
def test_real_import_package_tree_ownership(tmp_path, scenario):
    root, _ = _repository(tmp_path)
    canonical = root / 'src/lcprop'
    (canonical / 'ownership_probe.py').write_text("VALUE = 'canonical'\n")
    _run('git', 'add', 'src', cwd=root)
    _run('git', 'commit', '-qm', 'canonical probe fixture', cwd=root)
    if scenario == 'detached':
        _run('git', 'checkout', '--detach', '-q', cwd=root)
    nested = root / 'build/lib/lcprop'
    nested.mkdir(parents=True)
    if scenario == 'hardlinked_initializer':
        os.link(canonical / '__init__.py', nested / '__init__.py')
    else:
        (nested / '__init__.py').symlink_to(canonical / '__init__.py')
    (nested / 'ownership_probe.py').write_text("VALUE = 'nested'\n")
    (nested / 'outside_only.py').write_text("VALUE = 'outside'\n")
    repo = Path(__file__).resolve().parents[1]
    code = r'''
import importlib, pathlib, sys
sys.path.insert(0, sys.argv[1])
from lcprop.runners.distribution_deployment import automatic_deployment_manager
from lcprop.runners.source_deployment import SourceDeploymentError, SourceDeploymentManager
# Retain the actual provider under review, but import a genuinely fresh fixture
# package rather than replacing its __file__ while retaining the original path.
for name in tuple(sys.modules):
    if name == 'lcprop' or name.startswith('lcprop.'):
        del sys.modules[name]
canonical, nested = map(pathlib.Path, sys.argv[2:4])
scenario = sys.argv[4]
linked = scenario in ('linked_initializer', 'hardlinked_initializer')
package = nested if linked else canonical
sys.path.insert(0, str(package.parent))
importlib.invalidate_caches()
import lcprop
assert pathlib.Path(lcprop.__file__).parent == package
assert list(lcprop.__path__) == [str(package)]
if scenario == 'linked_initializer':
    assert pathlib.Path(lcprop.__file__).resolve() == canonical/'__init__.py'
if scenario == 'hardlinked_initializer':
    assert pathlib.Path(lcprop.__file__).samefile(canonical/'__init__.py')
if scenario == 'extended_path':
    lcprop.__path__.append(str(nested))
    import lcprop.outside_only
    assert pathlib.Path(lcprop.outside_only.__file__).parent == nested
if scenario == 'redirected_path':
    lcprop.__path__ = [str(nested)]
if scenario == 'extended_spec':
    lcprop.__spec__.submodule_search_locations = [str(canonical), str(nested)]
import lcprop.ownership_probe
assert lcprop.ownership_probe.VALUE == ('nested' if linked or scenario == 'redirected_path' else 'canonical')
if linked:
    assert pathlib.Path(lcprop.ownership_probe.__file__).parent == nested
try:
    selected = automatic_deployment_manager(host='example.invalid', source_root='/sources', remote_python=sys.executable, transport=object())
except SourceDeploymentError:
    assert scenario not in ('canonical', 'detached')
else:
    assert scenario in ('canonical', 'detached'), 'noncanonical package selected a source provider'
    assert type(selected) is SourceDeploymentManager
    assert selected.preflight().kind == 'git'
print('verified actual import and ownership decision: ' + scenario)
'''
    result = subprocess.run(
        [sys.executable, '-I', '-B', '-c', code, str(repo / 'src'),
         str(canonical), str(nested), scenario],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mixed_installed_package_path_cannot_bypass_git_ownership_rejection(installed, tmp_path, monkeypatch):
    import lcprop
    from importlib import metadata
    set_active_package(monkeypatch, installed / 'lcprop')
    outside = tmp_path / 'outside'
    outside.mkdir()
    monkeypatch.setattr(lcprop, '__path__', [str(installed / 'lcprop'), str(outside)])
    monkeypatch.setattr(metadata, 'distributions', lambda: [metadata.PathDistribution(installed / 'lcprop-0.1.0.dist-info')])
    with pytest.raises(SourceDeploymentError, match='unambiguous package search path'):
        automatic_deployment_manager(host='example.invalid', source_root='/sources', remote_python=sys.executable)

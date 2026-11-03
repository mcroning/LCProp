"""Execution-only notification settings; all remote interactions are mocked."""
from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from lcprop.gui.remote_execution import RemoteExecutionDialog
from lcprop.runners.cluster_profiles import ClusterCatalog, ClusterProfile, load_cluster_profiles, write_cluster_profiles
from lcprop.runners.slurm import SlurmRunner
from lcprop.transport.defaults import default_transport_registry, make_slurm_runner
import lcprop.runners.slurm as slurm
from tests.test_slurm_runner import _config, _request, CPU_SMALL, SHA

EMAIL = 'person+jobs@example.org'


def cluster(**kwargs):
    return ClusterProfile('test', 'user@login.example.org', '/runs', '/env/python',
                          '/sources', (CPU_SMALL,), default_resource_profile=CPU_SMALL.name, **kwargs)


def script(tmp_path, **kwargs):
    return SlurmRunner(_config(tmp_path, **kwargs), registry=default_transport_registry())._script('/runs/fixed', CPU_SMALL)


def test_off_is_exact_pre_feature_script(tmp_path):
    default = script(tmp_path)
    # Captured from Git HEAD b888e6d's original generator with this exact fixture.
    assert hashlib.sha256(default.encode()).hexdigest() == '736bc4cc01a156d5d3d34df4e7f3c88e42b8fd75ad51bc489936953c8a0a54fa'
    assert '--mail-' not in default
    assert script(tmp_path, notification_email=EMAIL, notification_events='') == default


@pytest.mark.parametrize('events', ['END', 'FAIL', 'END,FAIL', 'BEGIN,END,FAIL'])
def test_exact_directives(tmp_path, events):
    value = script(tmp_path, notification_email=EMAIL, notification_events=events)
    lines = value.splitlines()
    assert lines.count('#SBATCH --mail-user=' + EMAIL) == 1
    assert lines.count('#SBATCH --mail-type=' + events) == 1
    assert '\n'.join(line for line in lines if not line.startswith('#SBATCH --mail-')) + '\n' == script(tmp_path)


@pytest.mark.parametrize('email', ['', ' ', 'a@b.org\n#SBATCH --nodes=99', 'a@b.org\r', 'a@b.org\0',
                                  'a@b.org,b@c.org', 'a@b.org;touch /tmp/x', '$(id)@b.org',
                                  'a@-b.org', 'a..b@example.org', 'not-an-email', None, 17])
def test_invalid_address_rejected_at_both_boundaries(tmp_path, email):
    with pytest.raises(ValueError, match='email'):
        cluster(notification_email=email, notification_events='END')
    with pytest.raises(ValueError, match='email'):
        _config(tmp_path, notification_email=email, notification_events='END')


@pytest.mark.parametrize('events', ['ALL', 'END\n#SBATCH --nodes=99', 'FAIL,END', ['END'], None])
def test_only_named_event_sets_allowed(tmp_path, events):
    with pytest.raises(ValueError, match='events'):
        cluster(notification_email=EMAIL, notification_events=events)
    with pytest.raises(ValueError, match='events'):
        _config(tmp_path, notification_email=EMAIL, notification_events=events)


@pytest.mark.parametrize('events', ['', 'END', 'FAIL', 'END,FAIL', 'BEGIN,END,FAIL'])
def test_catalog_and_runner_roundtrip(tmp_path, events):
    c = cluster(notification_email=EMAIL, notification_events=events)
    path = tmp_path / 'clusters.toml'
    write_cluster_profiles(ClusterCatalog((c,), c.name, path))
    loaded = load_cluster_profiles(path)[c.name]
    assert loaded == c
    runner = make_slurm_runner(cluster=loaded, remote_source_path='/snapshot', source_git_sha=SHA, local_artifact_root=tmp_path)
    assert runner.config.notification_email == EMAIL
    assert runner.config.notification_events == events


def test_historical_catalog_defaults(tmp_path):
    path = tmp_path / 'clusters.toml'
    write_cluster_profiles(ClusterCatalog((cluster(),), 'test', path))
    assert 'notification_' not in path.read_text()
    loaded = load_cluster_profiles(path)['test']
    assert loaded.notification_events == loaded.notification_email == ''
    path.write_text(path.read_text().replace('[clusters.test]', '[clusters.test]\nnotification_events = "END"'))
    with pytest.raises(ValueError, match='email is required'):load_cluster_profiles(path)


def test_gui_save_reload_and_bounded_validation(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    path = tmp_path / 'clusters.toml'
    dialog = RemoteExecutionDialog(ClusterCatalog((cluster(),), 'test', path))
    messages = []
    monkeypatch.setattr('lcprop.gui.remote_execution.QMessageBox.warning', lambda *args: messages.append(args[2]))
    dialog.notification_events.setCurrentIndex(dialog.notification_events.findData('END,FAIL'))
    dialog.save_profile()
    assert len(messages) == 1 and 'email is required' in messages[0]
    assert not path.exists()
    dialog.notification_email.setText(EMAIL)
    dialog.save_profile()
    assert len(messages) == 1
    reopened = RemoteExecutionDialog(load_cluster_profiles(path))
    assert reopened.notification_email.text() == EMAIL
    assert reopened.notification_events.currentData() == 'END,FAIL'
    for events in ('', 'END', 'FAIL', 'END,FAIL', 'BEGIN,END,FAIL'):
        reopened.notification_events.setCurrentIndex(reopened.notification_events.findData(events))
        assert reopened._form_cluster().notification_events == events
    reopened._new_cluster()
    assert reopened.notification_events.currentData() == '' and reopened.notification_email.text() == ''
    reopened.close(); dialog.close()
    assert app is not None


def test_real_runner_request_and_source_provenance_unchanged(tmp_path, monkeypatch):
    # Stop after the real request codec/export, before upload/submission or science.
    class Captured(Exception):pass
    original = slurm.write_request_package
    packages = []
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        packages.append(json.loads((result / 'request.json').read_text()))
        raise Captured
    monkeypatch.setattr(slurm, 'write_request_package', capture)
    class Transport:
        def ssh(self, host, *args):
            assert args == ('cat', '/snapshot/.lcprop-source-sha')
            return SHA
    monkeypatch.setattr(slurm, 'uuid4', lambda: SimpleNamespace(hex='fixed'))
    request = _request()
    for index, events in enumerate(('', 'END,FAIL')):
        runner = make_slurm_runner(cluster=cluster(notification_email=EMAIL, notification_events=events),
                                   remote_source_path='/snapshot', source_git_sha=SHA,
                                   local_artifact_root=tmp_path / str(index), transport=Transport())
        with pytest.raises(Captured):runner.run_registered('lc', 'static', request)
    assert packages[0] == packages[1]
    assert EMAIL not in json.dumps(packages)
    assert 'notification_' not in json.dumps(packages)

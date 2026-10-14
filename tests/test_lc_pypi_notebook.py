"""Exercise the actual PyPI notebook cells without installations or scientific solves."""
import io
import json
import sys
import subprocess
import urllib.request
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / 'notebooks/lc_static_pypi_cpu.ipynb'

@pytest.fixture
def cells(monkeypatch):
    import IPython.display
    import lcprop.notebook.interactive_lc as ui
    from lcprop.runners.local import LocalRunner
    monkeypatch.setattr(IPython.display, 'display', lambda *a, **k: None)
    def forbidden(*a, **k):
        pytest.fail('Setup invoked installation or scientific execution')
    monkeypatch.setattr(ui, 'execute', forbidden)
    monkeypatch.setattr(LocalRunner, 'run_operation', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    artifact = ROOT / 'results/Research/lcprop-pypi-stage-5b-production-release-v1/artifacts/lcprop-0.3.0-py3-none-any.whl'
    digest = '9ab34f4ec244716b3e6f0431e4cac1e0d8a1ebeb17a7064c6422f1ed6b4e3d60'
    metadata = {'urls': [{'filename': artifact.name, 'digests': {'sha256': digest}, 'url': 'https://files.pythonhosted.org/fixture.whl'}]}
    def fetch(url, **kwargs):
        return io.BytesIO(json.dumps(metadata).encode() if url.endswith('/json') else artifact.read_bytes())
    monkeypatch.setattr(urllib.request, 'urlopen', fetch)
    nb = json.loads(NB.read_text()); ns = {}
    exec(''.join(nb['cells'][1]['source']), ns)
    return ns, ''.join(nb['cells'][2]['source']), metadata


def test_real_initialization_and_repeat_preserves_snapshot(cells):
    ns, setup, _ = cells
    exec(setup, ns)
    from lcprop.notebook.lc_static_demo import build_request
    app = ns['app']; assert app.request() == build_request()
    assert app.time_budget.value == 120 and app.completed is None
    sentinel = object(); app.completed = sentinel
    exec(setup, ns)
    assert ns['app'] is app and app.completed is sentinel
    assert ns['installation_provenance']['runtime_files_verified'] == 267
    assert not any(n in sys.modules for n in ('PySide6', 'cupy', 'launchplane'))


def test_wrong_published_hash_preserves_app(cells):
    ns, setup, metadata = cells
    exec(setup, ns); previous = ns['app']
    metadata['urls'][0]['digests']['sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='Published wheel identity'): exec(setup, ns)
    assert ns['app'] is previous


def test_download_integrity(cells, monkeypatch):
    ns, setup, metadata = cells
    monkeypatch.setattr(urllib.request, 'urlopen', lambda url, **k: io.BytesIO(json.dumps(metadata).encode() if url.endswith('/json') else b'corrupt'))
    with pytest.raises(RuntimeError, match='Downloaded wheel identity'): exec(setup, ns)
    assert 'app' not in ns


def test_missing_dependency(cells, monkeypatch):
    import importlib
    ns, setup, _ = cells; original = importlib.import_module
    def imp(name, *a, **k):
        if name == 'ipywidgets': raise ImportError('fixture')
        return original(name, *a, **k)
    monkeypatch.setattr(importlib, 'import_module', imp)
    with pytest.raises(RuntimeError, match='Missing notebook dependencies'): exec(setup, ns)


def test_loaded_wrong_version(cells, monkeypatch):
    import importlib.metadata as md
    ns, setup, _ = cells; original = md.version
    monkeypatch.setattr(md, 'version', lambda name: '0.2' if name == 'lcprop' else original(name))
    with pytest.raises(RuntimeError, match='different LCProp is loaded'): exec(setup, ns)


def test_install_failure_and_pin(cells, monkeypatch):
    import importlib.metadata as md
    from types import SimpleNamespace
    ns, setup, _ = cells; original = md.version
    for n in list(sys.modules):
        if n == 'lcprop' or n.startswith('lcprop.'): monkeypatch.delitem(sys.modules, n)
    def version(name):
        if name == 'lcprop': raise md.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(md, 'version', version)
    calls = []
    def run(command, **kwargs):
        calls.append(command); return SimpleNamespace(returncode=1, stderr='network unavailable')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Installation failed'): exec(setup, ns)
    assert calls[0][-3:] == ['--index-url', 'https://pypi.org/simple', 'lcprop[colab]==0.3.0']


def test_notebook_presentation_and_no_outputs():
    nb = json.loads(NB.read_text())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            assert cell['metadata']['cellView'] == 'form'
            assert cell['outputs'] == [] and cell['execution_count'] is None
    assert len(nb['cells'][2]['source']) < 12
    assert 'third-party' in ''.join(nb['cells'][0]['source'])


def test_successful_install_dispatch(cells, monkeypatch):
    import importlib.metadata as md
    from types import SimpleNamespace
    ns, setup, _ = cells; original = md.version; installed = [False]
    for n in list(sys.modules):
        if n == 'lcprop' or n.startswith('lcprop.'): monkeypatch.delitem(sys.modules, n)
    def version(name):
        if name == 'lcprop' and not installed[0]: raise md.PackageNotFoundError(name)
        return original(name)
    monkeypatch.setattr(md, 'version', version)
    def install(command, **kwargs):
        assert command[-1] == 'lcprop[colab]==0.3.0'
        installed[0] = True
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', install)
    exec(setup, ns)
    assert ns['app'].completed is None
    assert ns['installation_provenance']['version'] == '0.3.0'


def test_installed_file_tampering(cells, monkeypatch):
    ns, setup, _ = cells; original = Path.read_bytes
    def read(path):
        if str(path).endswith('/lcprop/__init__.py'): return b'altered'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    with pytest.raises(RuntimeError, match='Installed package identity'): exec(setup, ns)


def test_loaded_dependency_version_mismatch(cells, monkeypatch):
    import numpy
    ns, setup, _ = cells
    monkeypatch.setattr(numpy, '__version__', '0.0')
    with pytest.raises(RuntimeError, match='Loaded numpy differs'): exec(setup, ns)

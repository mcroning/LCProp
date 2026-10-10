"""Bootstrap checks with no installer or scientific solver execution."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def boot():
    spec=importlib.util.spec_from_file_location('bootstrap_under_test',ROOT/'examples/lc_colab_bootstrap.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m




def test_manifest_and_engine(boot):
    p=ROOT/'results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/expected-installed-files.json'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==boot.MANIFEST_SHA256
    manifest=json.loads(p.read_text());assert len(manifest)==264
    boot.verify_engine(manifest)
    with pytest.raises(RuntimeError,match='identity mismatch'):boot.verify_engine({'__init__.py':'wrong'})


def test_download_checks(boot,monkeypatch):
    monkeypatch.setattr(boot,'urlopen',lambda *a,**k:io.BytesIO(b'good'))
    assert boot.download('url',hashlib.sha256(b'good').hexdigest())==b'good'
    with pytest.raises(RuntimeError,match='identity mismatch'):boot.download('url','wrong')
    def fail(*a,**k):raise OSError('404')
    monkeypatch.setattr(boot,'urlopen',fail)
    with pytest.raises(RuntimeError,match='Pinned download unavailable'):boot.download('url','wrong')


def test_preimported_dependencies_reused(boot,monkeypatch):
    import numpy,scipy,matplotlib,ipywidgets
    monkeypatch.setattr(boot,'pip_install',lambda args:pytest.fail('Must reuse installed dependencies'))
    boot.dependencies()


def test_missing_and_incompatible_dependencies(boot,monkeypatch):
    calls=[];missing={'ipywidgets'}
    monkeypatch.setattr(boot.importlib.util,'find_spec',lambda n:None if n in missing else True)
    monkeypatch.setattr(boot,'pip_install',lambda args:(calls.append(args),missing.clear()))
    monkeypatch.setattr(boot.importlib,'import_module',lambda n:SimpleNamespace(__version__='1'))
    monkeypatch.setattr(boot.metadata,'version',lambda n:'1')
    boot.dependencies();assert calls==[['ipywidgets']]
    monkeypatch.setattr(boot.metadata,'version',lambda n:'2')
    with pytest.raises(RuntimeError,match='restart the runtime'):boot.dependencies()


def test_pip_failure_quiet_actionable(boot,monkeypatch):
    calls=[]
    def run(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(returncode=1,stderr='Package unavailable',stdout='')
    monkeypatch.setattr(boot.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='Package unavailable'):boot.pip_install(['x'])
    assert '--no-deps' in calls[0][0] and calls[0][1]['capture_output']


def test_real_widget_repeated_setup(boot,monkeypatch,tmp_path):
    import lcprop
    def download(url,h,**kwargs):
        p=(ROOT/url.split('/8c2e8b1f4bc4477787c2e686c38d4636180f3d75/')[1]) if url==boot.MANIFEST_URL else ROOT/'examples'/url.rsplit('/',1)[-1]
        b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==h;return b
    monkeypatch.setattr(boot,'download',download)
    monkeypatch.setattr(boot,'pip_install',lambda args:pytest.fail('No installation needed'))
    monkeypatch.chdir(tmp_path)
    # Keep imported helper state isolated from other tests.
    import sys
    for name in ('lc_static_interactive','lc_static_nonlinear_cpu'):monkeypatch.delitem(sys.modules,name,raising=False)
    demo,app=boot.initialize()
    assert app.request()==demo.build_request() and app.completed is None
    marker=demo.CompletedRun('{}',b'preserved','{}');app.completed=marker
    monkeypatch.setattr(demo,'execute',lambda *a,**k:pytest.fail('No solver allowed'))
    assert boot.initialize(app)[1] is app and app.completed is marker
    assert not any(n.split('.')[0] in ('PySide6','launchplane','cupy') for n in sys.modules)


def test_broken_dependency_import(boot,monkeypatch):
    monkeypatch.setattr(boot.importlib.util,'find_spec',lambda n:True)
    def broken(name):raise ImportError('binary ABI mismatch')
    monkeypatch.setattr(boot.importlib,'import_module',broken)
    with pytest.raises(RuntimeError,match='binary ABI mismatch'):boot.dependencies()

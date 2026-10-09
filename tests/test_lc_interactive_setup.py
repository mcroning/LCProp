"""Execute the notebook's real setup cell with local transport/install fixtures."""
import ast
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def cell():
    nb=json.loads((ROOT/'notebooks/lc_static_interactive_cpu.ipynb').read_text())
    tree=ast.parse(''.join(nb['cells'][1]['source']))
    tree.body.pop()  # Defer only the actual setup invocation.
    ns={};exec(compile(tree,'notebook-setup','exec'),ns)
    return ns

@pytest.fixture
def transport(cell,monkeypatch):
    def fetch(url,timeout):
        return io.BytesIO((ROOT/'examples'/url.rsplit('/',1)[-1]).read_bytes())
    monkeypatch.setitem(cell,'urlopen',fetch)


def test_pinned_helper_hashes(cell,transport):
    assert len(cell['ENGINE_REV'])==40 and len(cell['HELPER_REV'])==40
    for name in cell['HELPER_HASHES']:
        assert cell['fetch_helper'](name)==(ROOT/'examples'/name).read_bytes()


def test_bad_hash(cell,monkeypatch):
    monkeypatch.setitem(cell,'urlopen',lambda *a,**k:io.BytesIO(b'wrong'))
    with pytest.raises(RuntimeError,match='identity mismatch'):cell['fetch_helper']('lc_static_interactive.py')


def test_unpublished_revision(cell,monkeypatch):
    def fail(*a,**k):raise OSError('404')
    monkeypatch.setitem(cell,'urlopen',fail)
    with pytest.raises(RuntimeError,match='publication'):cell['setup']()


@pytest.fixture
def setup_fixture(cell,transport,monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(cell,'verify_engine',lambda:tmp_path/'site-packages/lcprop')
    monkeypatch.setitem(cell,'sys',SimpleNamespace(executable=sys.executable,modules={},path=[]))
    fake=SimpleNamespace(InteractiveLC=lambda:SimpleNamespace(widget='widget'),environment=lambda:{})
    monkeypatch.setitem(cell,'importlib',SimpleNamespace(invalidate_caches=lambda:None,
        util=SimpleNamespace(find_spec=lambda n:True),import_module=lambda n:fake))
    calls=[]
    import subprocess
    monkeypatch.setitem(cell,'subprocess',SimpleNamespace(run=lambda args,check:calls.append(args),CalledProcessError=subprocess.CalledProcessError))
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda x:None)
    return calls


def test_setup_success_and_rerun(cell,setup_fixture):
    demo,app=cell['setup']()
    assert cell['ENGINE_URL'] in setup_fixture[0]
    assert not any(x in ' '.join(setup_fixture[0]).lower() for x in ('pyside','launchplane','cupy'))
    cell['app']=app;cell['sys'].modules['lcprop']=object()
    assert cell['setup']()[1] is app
    assert len(setup_fixture)==1


def test_install_failure(cell,setup_fixture):
    def fail(*a,**k):raise OSError('pip unavailable')
    cell['subprocess'].run=fail
    with pytest.raises(RuntimeError,match='Installation failed'):cell['setup']()


def test_missing_dependency(cell,setup_fixture):
    cell['importlib'].util.find_spec=lambda n:None if n=='ipywidgets' else True
    with pytest.raises(RuntimeError,match='Missing dependencies: ipywidgets'):cell['setup']()


def test_installed_engine_manifest(cell,monkeypatch):
    # Existing qualified disposable installation; no installation or calculation.
    root=cell['verify_engine']()
    assert root.is_dir()
    monkeypatch.setitem(cell,'ENGINE_FILES',{'__init__.py':'0'*64})
    with pytest.raises(RuntimeError,match='Engine identity mismatch'):cell['verify_engine']()

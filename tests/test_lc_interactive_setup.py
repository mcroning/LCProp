"""Pinned notebook integration checks; no solver or installation."""
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


def test_notebook_binding(boot):
    nb=json.loads((ROOT/'notebooks/lc_static_interactive_cpu.ipynb').read_text())
    source=''.join(nb['cells'][1]['source'])
    assert hashlib.sha256((ROOT/'examples/lc_colab_bootstrap.py').read_bytes()).hexdigest() in source
    assert "BOOTSTRAP_REV = '8f2d66a6d778b21c5929ce5196b84a39bfca7a06'" in source
    assert 'PUBLICATION_PENDING' not in source
    assert 'ENGINE_FILES' not in source and len(source.splitlines())<25
    assert nb['cells'][1]['metadata']['cellView']=='form'
    assert boot.ENGINE_REV=='52b00928978a9ec7862a357d38327fd0b9ae857c'
    assert boot.HELPER_REV=='6ead2b5746765d865965162d614be8bb79659827'


def test_actual_setup_cell_and_repeated_snapshot(monkeypatch,tmp_path):
    import sys,subprocess,urllib.request
    import lcprop
    source=''.join(json.loads((ROOT/'notebooks/lc_static_interactive_cpu.ipynb').read_text())['cells'][1]['source'])
    def fetch(url,timeout):
        if url.endswith('/lc_colab_bootstrap.py'):p=ROOT/'examples/lc_colab_bootstrap.py'
        elif url.endswith('/expected-installed-files.json'):
            p=ROOT/'results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/expected-installed-files.json'
        else:p=ROOT/'examples'/url.rsplit('/',1)[-1]
        return io.BytesIO(p.read_bytes())
    monkeypatch.setattr(urllib.request,'urlopen',fetch)
    monkeypatch.setattr(subprocess,'run',lambda *a,**k:pytest.fail('No installation allowed'))
    for name in ('lc_static_interactive','lc_static_nonlinear_cpu'):monkeypatch.delitem(sys.modules,name,raising=False)
    monkeypatch.chdir(tmp_path)
    ns={};exec(compile(source,'notebook','exec'),ns)
    app=ns['app'];demo=ns['demo'];assert app.request()==demo.build_request()
    marker=demo.CompletedRun('{}',b'prior completed arrays','{}');app.completed=marker
    monkeypatch.setattr(demo,'execute',lambda *a,**k:pytest.fail('No scientific execution allowed'))
    exec(compile(source,'notebook','exec'),ns)
    assert ns['app'] is app and app.completed is marker
    assert not any(n.split('.')[0] in ('PySide6','launchplane','cupy') for n in sys.modules)

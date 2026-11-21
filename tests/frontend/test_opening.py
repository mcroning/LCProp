from pathlib import Path
import json,types,os,sys,zipfile,hashlib
import pytest
ROOT=Path(__file__).resolve().parents[2]
KIT=Path(os.environ.get('PR_CAPABILITY_KIT',str(ROOT/'delivery-v5/LCProp-PR-capabilities-setup.zip')))
@pytest.fixture(autouse=True)
def setup_kit_available():
    if not KIT.is_file():pytest.skip('Opening tests need the locally built exact kit: set PR_CAPABILITY_KIT. See frontend/README.md.')

def opener():
    notebook=json.loads((ROOT/'delivery-v5/LCProp-PR-capabilities.ipynb').read_text())
    ns={'__name__':'opening_test'}
    exec(compile(''.join(notebook['cells'][1]['source']),'pinned notebook Open cell','exec'),ns)
    return ns


def test_kit_and_notebook_exact_pins():
    m=opener();members=m['verify_kit'](KIT)
    assert hashlib.sha256(members['lcprop-0.5.0rc3-py3-none-any.whl']).hexdigest()==m['ENGINE_SHA256']
    for path in (ROOT/'frontend/pr_colab').glob('*.py'):
        assert members['frontend/pr_colab/'+path.name]==path.read_bytes()
    assert len(json.loads((ROOT/'delivery-v5/LCProp-PR-capabilities.ipynb').read_text())['cells'])==2


def test_missing_kit_help_no_picker_no_install(tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path);m=opener()
    m['subprocess']=types.SimpleNamespace(CalledProcessError=__import__('subprocess').CalledProcessError,run=lambda *a,**kw:pytest.fail('Unexpected install'))
    m['open_workbench']({})
    assert 'Files sidebar' in capsys.readouterr().out


def test_fresh_install_restart_resume_with_mocked_setup(tmp_path,monkeypatch,capsys):
    m=opener();monkeypatch.chdir(tmp_path)
    kit=KIT
    (tmp_path/m['KIT_NAME']).write_bytes(kit.read_bytes())
    state={'installed':False};calls=[]
    m['sys']=types.SimpleNamespace(modules={},executable=sys.executable,path=[])
    m['installed_matches']=lambda *a:state['installed']
    def fake_install(*a,**kw):calls.append(a);state['installed']=True
    m['subprocess']=types.SimpleNamespace(CalledProcessError=__import__('subprocess').CalledProcessError,run=fake_install)
    m['open_workbench']({})
    assert len(calls)==3 and 'Restart session' in capsys.readouterr().out
    m['open_workbench']({})
    assert len(calls)==3 and 'Restart session' in capsys.readouterr().out
    # Simulate the one requested restart; no actual interpreter/package changes.
    for marker in tmp_path.glob('.pr-capabilities-*/installed.json'):marker.write_text('{"pid":-1}')
    versions={'matplotlib':'3.11.2','ipympl':'0.10.0','ipywidgets':'7.7.1','widgetsnbextension':'3.6.10','jupyterlab_widgets':'1.1.11','ipython-genutils':'0.2.0','Pillow':'12.0','launchplane':'0.1.0.dev0'}
    monkeypatch.setattr(m['importlib'].metadata,'version',lambda name:versions[name])
    m['display_workbench']=lambda ns:'rendered'
    assert m['open_workbench']({})=='rendered' and len(calls)==4


def test_reconnect_never_installs_or_discards_live_panel():
    m=opener();seen=[]
    panel=types.SimpleNamespace(redisplay=lambda **kw:seen.append(kw))
    ns={'lcprop_pr_workbench':panel,'_pr_frontend_identity':m['FRONTEND_SHA256']}
    m['open_workbench'](ns)
    assert len(seen)==1 and ns['lcprop_pr_workbench'] is panel
    ns['_pr_frontend_identity']='other'
    with pytest.raises(RuntimeError,match='Save'):m['open_workbench'](ns)


def test_foreign_loaded_engine_is_rejected_without_replacing_modules(tmp_path):
    m=opener();contents=m['verify_kit'](KIT)
    other=tmp_path/'__init__.py';other.write_text('# Different engine source')
    module=types.SimpleNamespace(__file__=str(other))
    m['sys']=types.SimpleNamespace(modules={'lcprop':module})
    with pytest.raises(RuntimeError,match='Loaded engine differs'):
        m['verify_loaded_engine'](contents['lcprop-0.5.0rc3-py3-none-any.whl'])
    assert m['sys'].modules['lcprop'] is module


def test_headless_codec_subset_is_exact_and_has_no_gui_distribution():
    m=opener();contents=m['verify_kit'](KIT)
    pins=json.loads((ROOT/'delivery-v5/PINS.json').read_text())
    provenance=json.loads((ROOT/'frontend/launchplane/PROVENANCE.json').read_text())
    assert provenance['source_wheel_sha256']==pins['headless_launchplane_source_sha256']
    assert not any(n.startswith('launchplane-') for n in contents)
    for name,digest in provenance['files'].items():
        assert hashlib.sha256(contents['frontend/launchplane/'+name]).hexdigest()==digest
    assert pins['notebook_dependencies']['ipywidgets']=='7.7.1'
    assert 'PySide6' not in ''.join(json.loads((ROOT/'delivery-v5/LCProp-PR-capabilities.ipynb').read_text())['cells'][1]['source'])


def test_dependency_failure_does_not_claim_setup_or_write_restart_receipt(tmp_path,monkeypatch,capsys):
    m=opener();monkeypatch.chdir(tmp_path)
    (tmp_path/m['KIT_NAME']).write_bytes((KIT).read_bytes())
    m['sys']=types.SimpleNamespace(modules={},executable=sys.executable,path=[])
    installed=[]
    m['installed_matches']=lambda *a:bool(installed)
    def check(command,**kw):
        if command[-1]=='check':raise RuntimeError('Dependency check failed: preserve files and use a fresh runtime')
        installed.append(command)
    m['subprocess']=types.SimpleNamespace(CalledProcessError=__import__('subprocess').CalledProcessError,run=check)
    with pytest.raises(RuntimeError,match='Dependency check failed'):m['open_workbench']({})
    assert not list(tmp_path.glob('.pr-capabilities-*/installed.json'))
    assert 'Setup complete' not in capsys.readouterr().out
    assert not any('launchplane-' in arg or 'PySide6' in arg for cmd in installed for arg in cmd)


def test_dependency_failure_preserves_actual_explanation():
    import subprocess
    m=opener()
    def broken(*args,**kwargs):
        raise subprocess.CalledProcessError(1,args[0],output='unrelated-package requires missing-package',stderr='')
    m['subprocess']=types.SimpleNamespace(run=broken,CalledProcessError=subprocess.CalledProcessError)
    with pytest.raises(RuntimeError,match='unrelated-package requires missing-package'):
        m['check_dependencies']()


def test_foreign_headless_codec_module_is_not_hot_replaced(tmp_path,monkeypatch):
    m=opener();monkeypatch.chdir(tmp_path)
    (tmp_path/m['KIT_NAME']).write_bytes((KIT).read_bytes())
    foreign=tmp_path/'foreign.py';foreign.write_text('# incompatible codec')
    module=types.SimpleNamespace(__file__=str(foreign))
    m['sys']=types.SimpleNamespace(modules={'launchplane.model':module},executable=sys.executable,path=[])
    m['installed_matches']=lambda *a:True
    m['check_dependencies']=lambda:None
    monkeypatch.setattr(m['importlib'].metadata,'version',lambda name:m['NOTEBOOK_PINS'].get(name,'12.0'))
    with pytest.raises(RuntimeError,match='Different LaunchPlane code'):m['open_workbench']({})
    assert m['sys'].modules['launchplane.model'] is module

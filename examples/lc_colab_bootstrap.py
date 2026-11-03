"""Notebook bootstrap only; no scientific solver execution."""
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

ENGINE_REV = '52b00928978a9ec7862a357d38327fd0b9ae857c'
HELPER_REV = '6ead2b5746765d865965162d614be8bb79659827'
HELPER_HASHES = {'lc_static_interactive.py': '509d64948ebeb5ddbef0600049d081fd71b32689dd838a10a5223a5c46223672', 'lc_static_nonlinear_cpu.py': '2ffed40dfc82d47b73bd1f9770c2ef4a12a2918e32bd2c137f6a252ed07f90a3'}
ENGINE_URL = f'https://github.com/mcroning/LCProp/archive/{ENGINE_REV}.zip'
MANIFEST_URL = 'https://raw.githubusercontent.com/mcroning/LCProp/8c2e8b1f4bc4477787c2e686c38d4636180f3d75/results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/expected-installed-files.json'
MANIFEST_SHA256 = '71aa29c55973ef422378b7616208e1dea064d64d2fea66cdede22ea567972299'


def download(url, expected, limit=1_000_000):
    try:
        with urlopen(url, timeout=30) as response:
            data = response.read(limit+1)
    except Exception as exc:
        raise RuntimeError('Pinned download unavailable. Check network and publication of the exact revision.') from exc
    if len(data)>limit or hashlib.sha256(data).hexdigest()!=expected:
        raise RuntimeError('Downloaded identity mismatch; stop and use the verified published notebook.')
    return data


def pip_install(arguments):
    try:
        result = subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',
                                 '--quiet','--no-deps',*arguments], capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError('Cannot start pip in this kernel. Check the Python installation.') from exc
    if result.returncode:
        detail=(result.stderr or result.stdout or 'No installer diagnostics')[-3000:]
        raise RuntimeError('Installation failed; existing dependencies were not upgraded.\n'+detail)
    importlib.invalidate_caches()


def dependencies():
    # Reuse installed versions; never replace imported numerical libraries in place.
    names=['numpy','scipy','matplotlib','ipywidgets','IPython']
    if sys.version_info < (3,11): names.append('tomli')
    missing=[n for n in names if importlib.util.find_spec(n) is None]
    if missing: pip_install(missing)
    for name in names:
        try:
            module=importlib.import_module(name)
            installed=metadata.version(name)
        except Exception as exc:
            raise RuntimeError(f'Dependency {name} is missing or cannot import. Check its installation and '
                               'reported error; a restart is needed only after replacing an imported dependency. '
                               f'Details: {exc}') from exc
        loaded=getattr(module,'__version__',installed)
        if loaded!=installed:
            raise RuntimeError(f'{name} loaded version {loaded} differs from installed {installed}; '
                               'restart the runtime to use the installed version. No automatic restart.')


def verify_engine(manifest):
    import lcprop
    root=Path(lcprop.__file__).resolve().parent
    if not any(p in ('site-packages','dist-packages') for p in root.parts):
        raise RuntimeError('LCProp resolves to a checkout. Use an installed-package kernel.')
    for name,expected in manifest.items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Engine identity mismatch: {name}. If LCProp is already imported, '
                               'restart in a clean runtime before setup; no in-process replacement.')
    return root


def initialize(previous_app=None):
    try:
        print('Setup: verifying pinned inputs…')
        manifest=json.loads(download(MANIFEST_URL,MANIFEST_SHA256))
        payloads={n:download(f'https://raw.githubusercontent.com/mcroning/LCProp/{HELPER_REV}/examples/{n}',h)
                  for n,h in HELPER_HASHES.items()}
        print('Setup: checking dependencies and engine…')
        dependencies()
        if 'lcprop' not in sys.modules:
            pip_install([ENGINE_URL])
        verify_engine(manifest)
        folder=Path.cwd()/'.lcprop-colab-helpers'/HELPER_REV
        folder.mkdir(parents=True,exist_ok=True)
        for name,data in payloads.items():
            loaded=sys.modules.get(Path(name).stem)
            if loaded is not None:
                p=Path(loaded.__file__).resolve()
                if p!=(folder/name).resolve() or hashlib.sha256(p.read_bytes()).hexdigest()!=HELPER_HASHES[name]:
                    raise RuntimeError('Conflicting imported helper. Restart runtime before using this revision.')
            (folder/name).write_bytes(data)
        if str(folder.resolve()) not in sys.path:sys.path.insert(0,str(folder.resolve()))
        demo=importlib.import_module('lc_static_interactive')
        from IPython import get_ipython
        shell=get_ipython()
        if shell is not None:shell.run_line_magic('matplotlib','inline')
        app=previous_app if previous_app is not None else demo.InteractiveLC()
        from IPython.display import display
        display(app.widget)
        print('Ready — configure the experiment, then press Run. Setup has not run the solver.')
        return demo,app
    except Exception as exc:
        print(f'Failed — {exc}')
        raise

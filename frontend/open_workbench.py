#@title Open PR workbench — run again after restart or reconnect
from pathlib import Path
import hashlib,importlib.metadata,io,json,os,subprocess,sys,zipfile
KIT_NAME = 'LCProp-PR-capabilities-setup.zip'
KIT_SHA256 = "STAGED"
FRONTEND_SHA256 = "STAGED"
ENGINE_SHA256 = "STAGED"
RENDERER = '3.11.2'
NOTEBOOK_PINS = {'matplotlib':RENDERER,'ipympl':'0.10.0','ipywidgets':'7.7.1',
                 'widgetsnbextension':'3.6.10','jupyterlab_widgets':'1.1.11','ipython-genutils':'0.2.0'}


def installed_matches(wheel, package="lcprop"):
    try:
        dist=importlib.metadata.distribution(package)
        with zipfile.ZipFile(io.BytesIO(wheel)) as z:
            members=[n for n in z.namelist() if n.startswith(package+'/') and not n.endswith('/')]
            return bool(members) and all(Path(dist.locate_file(n)).read_bytes()==z.read(n) for n in members)
    except (OSError,importlib.metadata.PackageNotFoundError):return False


def verify_kit(path):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=KIT_SHA256:raise ValueError('Setup ZIP checksum mismatch; nothing installed')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate setup members')
        pins=json.loads(z.read('PINS.json'))
        if set(names)!=set(pins['members'])|{'PINS.json'}:raise ValueError('Unexpected setup contents')
        contents={name:z.read(name) for name in pins['members']}
    if pins['engine_sha256']!=ENGINE_SHA256 or pins['frontend_sha256']!=FRONTEND_SHA256:
        raise ValueError('Setup identities do not match this notebook')
    for name,value in contents.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Invalid setup path')
        if hashlib.sha256(value).hexdigest()!=pins['members'][name]:raise ValueError('Setup member checksum mismatch')
    return contents


def verify_loaded_engine(wheel):
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        names=set(z.namelist())
        for name,module in tuple(sys.modules.items()):
            if name!='lcprop' and not name.startswith('lcprop.'):continue
            relative=name.replace('.','/')
            member=relative+'/__init__.py' if relative+'/__init__.py' in names else relative+'.py'
            file=getattr(module,'__file__',None)
            if member not in names or not file or Path(file).read_bytes()!=z.read(member):
                raise RuntimeError('Loaded engine differs from the pin. Save work before restarting; no live module was replaced.')


def check_dependencies():
    try:
        subprocess.run([sys.executable,'-m','pip','check'],check=True,
                       capture_output=True,text=True)
    except subprocess.CalledProcessError as error:
        detail=(error.stdout or '')+(error.stderr or '')
        raise RuntimeError('Dependency check failed: '+detail.strip()+
                           ' Preserve saved files and use a fresh disposable CPU runtime; no dependency conflict was hidden.') from error


def display_workbench(namespace):
    try:
        from google.colab import output
    except ImportError:hosted=False
    else:output.enable_custom_widget_manager();hosted=True
    get_ipython().run_line_magic('matplotlib','widget')
    from pr_colab.pr_workbench import open_pr_workbench
    return open_pr_workbench(namespace,hosted=hosted)


def open_workbench(namespace):
    existing=namespace.get('lcprop_pr_workbench')
    if existing is not None:
        if namespace.get('_pr_frontend_identity')!=FRONTEND_SHA256:
            raise RuntimeError('A different workbench is live. Save its drafts/results before a restart; nothing was replaced.')
        return existing.redisplay(hosted='google.colab' in sys.modules)
    kit=Path.cwd()/KIT_NAME
    if not kit.is_file():
        print('Upload '+KIT_NAME+' using the Files sidebar, keep it zipped, then run this same cell again. No picker opens automatically.')
        return
    contents=verify_kit(kit)
    engine_name=next(n for n in contents if n.startswith('lcprop-') and n.endswith('.whl'))
    wheel=contents[engine_name]
    if hashlib.sha256(wheel).hexdigest()!=ENGINE_SHA256:raise ValueError('Engine wheel mismatch')
    root=Path.cwd()/('.pr-capabilities-'+FRONTEND_SHA256[:12])
    marker=root/'installed.json'
    try:receipt=json.loads(marker.read_text())
    except (ValueError,OSError):receipt={}
    if receipt.get('pid')==os.getpid():
        print('Setup complete. Runtime → Restart session once, then rerun this same Open cell. Do not delete the runtime.')
        return
    try:
        ready=(all(importlib.metadata.version(name)==version for name,version in NOTEBOOK_PINS.items())
               and bool(importlib.metadata.version('Pillow')))
    except importlib.metadata.PackageNotFoundError:ready=False
    exact=installed_matches(wheel)
    if not exact or not ready:
        if any(n=='lcprop' or n.startswith('lcprop.') for n in sys.modules):
            raise RuntimeError('LCProp is already loaded. Save work and restart before setup; no live package was changed.')
        root.mkdir(exist_ok=True)
        for name,value in contents.items():
            if name.endswith('.whl'):(root/name).write_bytes(value)
        print('Verified kit. Installing its unchanged engine and notebook rendering dependencies into this disposable runtime.')
        subprocess.run([sys.executable,'-m','pip','install','--force-reinstall','--no-deps',
                        *[str(root/n) for n in contents if n.endswith('.whl')]],check=True)
        subprocess.run([sys.executable,'-m','pip','install','numpy','scipy','Pillow',
                        *[name+'=='+version for name,version in NOTEBOOK_PINS.items()]],check=True)
        if not installed_matches(wheel):
            raise RuntimeError('Installed engine/dependency bytes do not match the pins')
        check_dependencies()
        marker.write_text(json.dumps(dict(pid=os.getpid(),engine=ENGINE_SHA256,frontend=FRONTEND_SHA256)))
        print('Setup complete. Runtime → Restart session once, then rerun this same Open cell. No propagation was started.')
        return
    check_dependencies()
    verify_loaded_engine(wheel)
    for module_name,module in tuple(sys.modules.items()):
        if module_name=='launchplane' or module_name.startswith('launchplane.'):
            member='frontend/'+module_name.replace('.','/')+('/__init__.py' if module_name=='launchplane' else '.py')
            file=getattr(module,'__file__',None)
            if member not in contents or not file or Path(file).read_bytes()!=contents[member]:
                raise RuntimeError('Different LaunchPlane code is loaded. Save work and restart; nothing replaced.')
    for name,value in contents.items():
        if name.startswith('frontend/'):
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
    if 'pr_colab' in sys.modules and namespace.get('_pr_frontend_identity')!=FRONTEND_SHA256:
        raise RuntimeError('Another frontend was imported. Save work before restarting; no hot reload was attempted.')
    sys.path.insert(0,str(root/'frontend'))
    namespace['_pr_frontend_identity']=FRONTEND_SHA256
    print('Opening controls below. If the panel stays blank, the widget renderer failed; preserve your files and report the browser error. Do not repeatedly reinstall. No run starts automatically.')
    return display_workbench(namespace)


if __name__=='__main__':
    try:open_workbench(globals())
    except Exception as error:
        print('Workbench did not open: '+str(error))
        print('No propagation started. Your scientific inputs were not changed. Preserve saved files; use the same Open cell to resume.')

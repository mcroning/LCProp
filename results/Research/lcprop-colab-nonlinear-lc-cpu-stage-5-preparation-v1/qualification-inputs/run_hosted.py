"""Explicit hosted CPU execution supervisor; do not run during preparation."""
from pathlib import Path
import contextlib
import hashlib
import importlib.metadata as md
import io
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time

p=Path(__file__).resolve().parent
out=Path(sys.argv[1]).resolve()
out.mkdir(exist_ok=False,parents=True)
def sha(f): return hashlib.sha256(f.read_bytes()).hexdigest()
def save(name,value): (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
started=time.time()
try:
    manifest=json.loads((p/'inputs-sha256.json').read_text())
    for name,digest in manifest.items(): assert sha(p/name)==digest,name
    import numpy as np, scipy, matplotlib, lcprop
    package=Path(lcprop.__file__).resolve().parent
    assert 'site-packages' in str(package) or 'dist-packages' in str(package)
    expected=json.loads((p/'expected-installed-files.json').read_text())
    for name,digest in expected.items(): assert sha(package/name)==digest,name
    save('installation.json',dict(python=sys.version, executable=sys.executable,platform=platform.platform(),
        numpy=np.__version__, scipy=scipy.__version__,matplotlib=matplotlib.__version__,
        lcprop=md.version('lcprop'),module=str(lcprop.__file__),
        engine_source='52b00928978a9ec7862a357d38327fd0b9ae857c',
        direct_url=md.distribution('lcprop').read_text('direct_url.json'),
        installed_files={name:sha(package/name) for name in expected},
        thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    config=io.StringIO()
    with contextlib.redirect_stdout(config): np.show_config(); scipy.show_config()
    (out/'numerical-config.txt').write_text(config.getvalue())
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(out/'pip-freeze.txt').open('w'),check=True)
    script=out/'executed-example.py'
    shutil.copyfile(p/'lc_static_nonlinear_cpu.py',script)
    assert sha(script)==manifest['lc_static_nonlinear_cpu.py']
    save('execution.json',dict(script_sha256=sha(script),input_manifest_sha256=sha(p/'inputs-sha256.json'),
        command=[sys.executable,'-I',str(script),'--output',str(out/'result')],started_unix=started))
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
        run=subprocess.run([sys.executable,'-I',str(script),'--output',str(out/'result')],
            stdout=stdout,stderr=stderr,timeout=300,env=dict(os.environ,MPLBACKEND='Agg'))
    save('process.json',dict(returncode=run.returncode,elapsed_seconds=time.time()-started,
        children_maxrss_KiB_linux=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss))
    assert run.returncode==0,'Stop: preserve incomplete evidence; no automatic retry'
    assert sha(script)==manifest['lc_static_nonlinear_cpu.py']
    for name,digest in expected.items(): assert sha(package/name)==digest,name
    import compare_saved
    save('comparison.json',compare_saved.compare(p,out/'result'))
except BaseException as error:
    save('failure.json',dict(type=type(error).__name__,message=str(error),elapsed_seconds=time.time()-started))
    raise
finally:
    save('qualification-manifest.json',{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='qualification-manifest.json'})
    archive=Path(shutil.make_archive(str(out), 'zip',out))
    print('Evidence archive:',archive,'SHA-256:',sha(archive))

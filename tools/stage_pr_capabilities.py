"""Package the standalone frontend and unchanged wheels; no install or publication."""
from pathlib import Path
import argparse,hashlib,io,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
ENGINE_SHA='ce9153666272780d3eac63f2c56d2777ae821a260bca6619e26852049cb78edd'
ENGINE_NAME='lcprop-0.5.0rc3-py3-none-any.whl'
LP_SHA='2c6310bb2dff18d0fbbd0978fc37ff11f9de55079af302e399722f01370d40f6'

def sha(data):return hashlib.sha256(data).hexdigest()

def engine_bytes(*, engine_wheel=None, input_kit=None, input_kit_sha256=None):
    if bool(engine_wheel) == bool(input_kit):
        raise ValueError('Supply exactly one --engine-wheel or --input-kit; no workspace fallback')
    if engine_wheel:
        data=Path(engine_wheel).read_bytes()
    else:
        blob=Path(input_kit).read_bytes()
        if not input_kit_sha256 or sha(blob)!=input_kit_sha256:
            raise ValueError('Input kit checksum missing or mismatched')
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if archive.namelist().count(ENGINE_NAME)!=1:
                raise ValueError('Input kit must contain one exact engine wheel')
            data=archive.read(ENGINE_NAME)
    if sha(data)!=ENGINE_SHA:
        raise ValueError('Engine wheel does not match unchanged ce915366 pin')
    return data


def stage(*, engine_wheel=None, input_kit=None, input_kit_sha256=None, output=None):
    data=engine_bytes(engine_wheel=engine_wheel,input_kit=input_kit,input_kit_sha256=input_kit_sha256)
    out=Path(output) if output else ROOT/'delivery-v3'
    # Never replace an earlier delivery or accepted artifact pair.
    out.mkdir(parents=True,exist_ok=False)
    provenance=json.loads((ROOT/'frontend/launchplane/PROVENANCE.json').read_text())
    if provenance['source_wheel_sha256']!=LP_SHA:
        raise ValueError('Headless codec source identity mismatch')
    for name,digest in provenance['files'].items():
        if sha((ROOT/'frontend/launchplane'/name).read_bytes())!=digest:
            raise ValueError('Headless codec source changed: '+name)
    (out/ENGINE_NAME).write_bytes(data)
    payload={ENGINE_NAME:data,
             'LICENSE':(ROOT/'LICENSE').read_bytes(),
             'READ-ME.txt':b'Local PR capability preview. Keep this ZIP intact.\nConnect CPU; upload this setup ZIP through Files; run the supplied notebook Open cell.\nIf requested, restart the session once and rerun the same cell.\nNo propagation starts until an explicit Run or Continue action. Drive is optional.\nEngine bytes are unchanged; this kit is not a published release.\n'}
    for path in sorted((ROOT/'frontend/pr_colab').iterdir()):
        if path.suffix in {'.py','.zip'}:payload['frontend/pr_colab/'+path.name]=path.read_bytes()
    for path in sorted((ROOT/'frontend/launchplane').iterdir()):
        if path.is_file():payload['frontend/launchplane/'+path.name]=path.read_bytes()
    payload['HEADLESS-CODECS.txt']=b'LaunchPlane headless source subset: unchanged __init__.py, model.py, serialization.py from wheel SHA256 2c6310bb2dff18d0fbbd0978fc37ff11f9de55079af302e399722f01370d40f6. License: frontend/launchplane/LICENSE. This is bundled source, not an installed LaunchPlane distribution; no Qt package is required or installed. Desktop LaunchPlane and the engine wheel remain unchanged.\n'
    members={name:sha(data) for name,data in payload.items()}
    frontend=sha(json.dumps({k:v for k,v in members.items() if k.startswith('frontend/')},sort_keys=True).encode())
    manifest=dict(engine_sha256=ENGINE_SHA,frontend_sha256=frontend,members=members,
        headless_launchplane_source_sha256=LP_SHA,
        notebook_dependencies={'ipywidgets':'7.7.1','ipympl':'0.10.0','matplotlib':'3.11.2',
                               'widgetsnbextension':'3.6.10','jupyterlab_widgets':'1.1.11','ipython-genutils':'0.2.0'},
        engine_source='ae3103f (preserved integrated qualification build)',
        scope='frontend only; no solver/GPU or hosted execution during preparation')
    payload['PINS.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    kit=out/'LCProp-PR-capabilities-setup.zip'
    with zipfile.ZipFile(kit,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,(2026,11,21,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data)
    code=(ROOT/'frontend/open_workbench.py').read_text()
    for key,value in [('KIT_SHA256',sha(kit.read_bytes())),('FRONTEND_SHA256',frontend),('ENGINE_SHA256',ENGINE_SHA)]:
        code=code.replace(f'{key} = "STAGED"',f'{key} = "{value}"')
    notes='''# PR workbench — capability preview

1. Connect to a CPU runtime. One prerequisite: upload **LCProp-PR-capabilities-setup.zip** through the Files sidebar; keep it zipped.
2. Run **Open PR workbench** below. If instructed, use **Runtime → Restart session** once, then run this same cell again.
3. Use the workbench. **Files → Open synthetic example** needs no result upload and runs no propagation. Run is always a separate explicit action.

This is a local frontend preview, not a published release. The kit pins the unchanged integrated engine. No Drive connection or file picker opens automatically. Rerun Open after a browser disconnect to reuse the live drafts/results. A deleted/replaced runtime loses unsaved work: save experiments/results before restarting an established session.

CPU is the default. Compatible unqualified GPU requests require explicit opt-in and a compatible CUDA/CuPy environment; this opening cell installs no GPU software. Auto must show its backend decision before dispatch. Existing engine compatibility and numerical limits remain. Estimated memory warnings can be acknowledged; no grids, precision, physics or result retention are silently changed.

The two Transport/Evolution selectors address the four classic routes. **Engine workflow** also exposes desktop Unified Static (electrical closure/solver in Advanced) and published optical-first reduced Static, with distinct arithmetic identities. Detailed/array launch inputs use native codec JSON; no implicit model migration occurs.
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='LCProp-PR-capabilities.ipynb',provenance=[]),kernelspec=dict(name='python3',display_name='Python 3')),
        cells=[dict(cell_type='markdown',metadata={},source=notes.splitlines(True)),
               dict(cell_type='code',execution_count=None,metadata=dict(cellView='form'),outputs=[],source=code.splitlines(True))])
    (out/'LCProp-PR-capabilities.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    (out/'PINS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=[]
    for name in ['LCProp-PR-capabilities-setup.zip','LCProp-PR-capabilities.ipynb',ENGINE_NAME,'PINS.json']:
        lines.append(sha((out/name).read_bytes())+'  '+name)
    (out/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--engine-wheel',type=Path)
    inputs.add_argument('--input-kit',type=Path)
    parser.add_argument('--input-kit-sha256')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(stage(**vars(args)),indent=2))

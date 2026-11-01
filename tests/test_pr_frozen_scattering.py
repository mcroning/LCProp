"""Synthetic dispatch, phase application and corruption tests; no propagation."""
import ast
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from lcprop.pr.frozen_scattering import FrozenScatteringPhases, scattering_context
import lcprop.pr.static_streaming as stream
from tests.test_pr_static_streaming import _request
from tests.streaming_assertions import assert_same


def fixture(tmp_path, *, mode=stream.PR_STREAMING_PRODUCTION, change=None, data=None):
    r=_request(mode=mode)
    r=replace(r,solver=replace(r.solver,volume_noise_epsilon=.02,volume_noise_seeds=(11,22,33)))
    a=np.arange(3*12*6,dtype=np.float64).reshape(3,12,6)/1000 if data is None else data
    p=tmp_path/'phases.npz';np.savez(p,phase=a)
    m=dict(schema=1,units='rad',layout='zxy',dtype='<f8',shape=[3,12,6],indices=[0,1,2],payload_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),context=scattering_context(r),producer='synthetic',coordinates='synthetic grid')
    if change:change(m)
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(m))
    return r,p,mp,hashlib.sha256(mp.read_bytes()).hexdigest(),a

def load(p,mp,h):return FrozenScatteringPhases.from_npz(p,manifest_path=mp,manifest_sha256=h)

def application_block(module):
    tree=ast.parse(Path(module.__file__).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_streaming_pass')
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    disk = next((n for n in loop.body if isinstance(n, ast.If) and 'DiskScatteringPhases' in ast.unparse(n.test)), None)
    if disk is not None:
        return compile(ast.fix_missing_locations(ast.Module(body=[disk], type_ignores=[])), 'phase-block', 'exec')
    ix=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='noise_phase')
    return compile(ast.fix_missing_locations(ast.Module(body=loop.body[ix:ix+2],type_ignores=[])),'phase-block','exec')

def stub_pass(module,seen):
    block=application_block(module)
    def run(**kw):
        a=kw['A0'].copy();seen.append(kw.get('frozen_scattering'))
        for z in range(3):
            ns=dict(module.__dict__,**kw,A=a,z_index=z)
            ns.setdefault("frozen_scattering",None)
            exec(block,ns)
        moments=np.zeros((3,3));p=module.PRStreamingPassResult(tuple(SimpleNamespace() for _ in range(3)),(),moments,moments,moments,moments,{},{},{},{},True)
        return p,a
    return run

@pytest.mark.parametrize('mode',stream.PR_STREAMING_MODES)
def test_all_modes_replay_and_no_regeneration(tmp_path,monkeypatch,mode):
    r,p,mp,h,a=fixture(tmp_path,mode=mode);source=load(p,mp,h);seen=[]
    monkeypatch.setattr(stream,'_run_streaming_pass',stub_pass(stream,seen))
    monkeypatch.setattr(stream,'_volume_noise_phase',lambda *a,**k:pytest.fail('Regenerated external input'))
    result=stream.run_pr_static_streaming(r,frozen_scattering=source)
    expected=result.A_initial.copy()
    for phase in a:expected*=np.exp(1j*phase)[None,:,:]
    assert np.array_equal(result.A_final,expected)
    assert seen==[source,source] and result.replay_diagnostics['consistent']
    assert result.memory_policy['frozen_scattering']['payload_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()

@pytest.mark.parametrize('mode',stream.PR_STREAMING_MODES)
def test_default_dispatch_bitwise_parity(tmp_path,monkeypatch,mode):
    baseline=Path(__file__).resolve().parents[1]/'tests/fixtures/pr_static_streaming_before_frozen.py'
    spec=importlib.util.spec_from_file_location('frozen_phase_baseline',baseline);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    r,*_=fixture(tmp_path,mode=mode)
    for mod in (old,stream):
        monkeypatch.setattr(mod,'_run_streaming_pass',stub_pass(mod,[]))
        monkeypatch.setattr(mod,'_volume_noise_phase',lambda z,**k:np.full((12,6),z*.01))
    assert_same(old.run_pr_static_streaming(r),stream.run_pr_static_streaming(r))

@pytest.mark.parametrize('mutation',[
 lambda m:m.update(indices=[0,2,1]),lambda m:m.update(indices=[False,1,2]),
 lambda m:m.update(shape=[4,12,6]),lambda m:m.update(dtype='<f4'),
 lambda m:m.update(payload_sha256='0'*64),lambda m:m.update(units='degrees')])
def test_manifest_rejections(tmp_path,mutation):
    r,p,mp,h,a=fixture(tmp_path,change=mutation)
    with pytest.raises(ValueError):load(p,mp,h)

@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite(tmp_path,value):
    a=np.zeros((3,12,6));a[0,0,0]=value;r,p,mp,h,_=fixture(tmp_path,data=a)
    with pytest.raises(ValueError):load(p,mp,h)

@pytest.mark.parametrize('mutation',['file','manifest','missing','float32','shape'])
def test_payload_corruption(tmp_path,mutation):
    r,p,mp,h,a=fixture(tmp_path)
    if mutation=='file':p.write_bytes(p.read_bytes()+b'corruption')
    elif mutation=='manifest':mp.write_bytes(mp.read_bytes()+b' ')
    elif mutation=='missing':p.unlink()
    else:
        a=a.astype(np.float32) if mutation=='float32' else a[:2];np.savez(p,phase=a)
        m=json.loads(mp.read_text());m['payload_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();mp.write_text(json.dumps(m));h=hashlib.sha256(mp.read_bytes()).hexdigest()
    with pytest.raises((ValueError,FileNotFoundError)):load(p,mp,h)

def test_immutable_and_context_rejected_before_launch(tmp_path,monkeypatch):
    r,p,mp,h,a=fixture(tmp_path);source=load(p,mp,h)
    with pytest.raises(ValueError):source.phase(0).setflags(write=True)
    with pytest.raises(AttributeError):source._planes=()
    p.unlink();assert np.array_equal(source.phase(0),a[0])
    monkeypatch.setattr(stream,'get_backend',lambda *a:pytest.fail('Launch reached'))
    wrong=replace(r,solver=replace(r.solver,volume_noise_seeds=(11,22,34)))
    with pytest.raises(ValueError,match='context'):stream.run_pr_static_streaming(wrong,frozen_scattering=source)
    with pytest.raises(ValueError,match='coverage'):source.validate_for(r,required_cells=4)
    for i in [-1,3,True,1.5]:
        with pytest.raises(IndexError):source.phase(i)


def test_default_march_equations_structurally_unchanged():
    # Updated from the pre-compiled-backend AST to the independently sealed
    # source qualified in job 6562036. Stage 2A did not alter this file.
    # Exact bytes avoid Python-version-dependent AST representations.
    assert hashlib.sha256(Path(stream.__file__).read_bytes()).hexdigest() == (
        "b3e3fb9ea78ad5434f24df5d743057cafa8a06f6426835879556770d7a4a209e"
    )

@pytest.mark.parametrize('field,value',[('epsilon',.03),('correlation_um',9.),('precision','float32')])
def test_incompatible_metadata_before_backend(tmp_path,monkeypatch,field,value):
    r,p,mp,h,a=fixture(tmp_path,change=lambda m:m['context'].update({field:value}))
    source=load(p,mp,h)
    monkeypatch.setattr(stream,'get_backend',lambda *a:pytest.fail('Invalid source reached backend'))
    with pytest.raises(ValueError,match='context'):stream.run_pr_static_streaming(r,frozen_scattering=source)

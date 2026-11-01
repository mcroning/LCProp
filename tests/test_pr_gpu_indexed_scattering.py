"""Synthetic only. Host doubles test control flow, not GPU RNG equivalence."""
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
import lcprop.pr.gpu_indexed_scattering as gpu
import lcprop.pr.static_streaming as stream
from lcprop.pr.disk_scattering import file_sha256
from tests.test_pr_indexed_scattering import make
from tests.test_pr_frozen_scattering import application_block


class DeviceArray(np.ndarray):
    @property
    def device(self):
        return SimpleNamespace(id=0)


@pytest.fixture
def double(monkeypatch):
    calls=[]
    def random_state(seed, *, method):
        assert method == 101
        calls.append(seed)
        rng=np.random.RandomState(seed)
        return SimpleNamespace(normal=lambda loc,scale,*,size,dtype:rng.normal(loc,scale,size).astype(dtype).view(DeviceArray))
    cp=SimpleNamespace(random=SimpleNamespace(RandomState=random_state),float64=np.float64,
        complex128=np.complex128,ndarray=DeviceArray,empty_like=np.empty_like,
        isfinite=np.isfinite,exp=np.exp,cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda:0),
            get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
    monkeypatch.setattr(gpu,'_gpu',lambda:(cp,SimpleNamespace(CURAND_RNG_PSEUDO_XORWOW=101),gaussian_filter1d))
    monkeypatch.setattr(gpu,'gpu_runtime_fingerprint',lambda:{'synthetic_test_double':True})
    return cp,calls


def recipe(tmp_path,refine=1):
    r,m,p,h,ids=make(tmp_path,refine=refine)
    r=replace(r,backend=replace(r.backend,backend='cupy'))
    m=gpu.gpu_indexed_phase_descriptor(r,original_shape=(17,13),original_aperture_um=(17.,13.),crop=((3,15),(2,10)),events=3,source_identities=ids,execution_dz_um=r.grid.dz_um/refine)
    p.write_text(json.dumps(m))
    return r,m,p,ids


def load(p,ids):
    return gpu.GPUIndexedScatteringPhases(p,descriptor_sha256=file_sha256(p),source_identities=ids)


def test_isolated_order_and_replay(tmp_path,double):
    cp,calls=double;r,m,p,ids=recipe(tmp_path);s=load(p,ids)
    s.validate_for(r,required_cells=3)
    first=s.phase(2).copy();s.phase(0);second=s.phase(2)
    assert np.array_equal(first,second) and calls==[33,11,33]
    second[:]=100
    assert np.array_equal(first,s.phase(2)) # caller mutation cannot affect replay
    fields=[np.ones((2,12,8),complex).view(DeviceArray) for _ in range(2)]
    for f in fields:s.apply_to(f,2,xp=cp)
    assert np.array_equal(*fields)
    g=m['generation'];raw=np.random.RandomState(33).normal(0,g['raw_scale'],size=(17,13))
    expected=gaussian_filter1d(gaussian_filter1d(raw,g['sigma_pixels'][0],axis=0),g['sigma_pixels'][1],axis=1)[3:15,2:10]*g['post_filter_scale']
    assert np.array_equal(first,expected)
    assert s.provenance()['historical_cpu_realization'] is False


@pytest.mark.parametrize('factor',[2,4])
def test_identity_mapping(tmp_path,double,factor):
    cp,calls=double;r,m,p,ids=recipe(tmp_path,refine=factor);s=load(p,ids)
    r=replace(r,grid=replace(r.grid,dz_um=r.grid.dz_um/factor),solver=replace(r.solver,volume_noise_seeds=gpu.gpu_execution_seed_schedule(m)))
    s.validate_for(r,required_cells=3*factor)
    field=np.ones((1,12,8),complex).view(DeviceArray)
    for cell in range(3*factor):
        before=field.copy();n=len(calls);s.apply_to(field,cell,xp=cp)
        if s.event_for_cell(cell) is None:
            assert np.array_equal(field,before) and len(calls)==n
    assert calls==[11,22,33]


@pytest.mark.parametrize('mutation',[
    lambda m:m.update(algorithm='historical'),
    lambda m:m.update(algorithm_sha256='0'*64),
    lambda m:m['implementation'].update(gpu_source='0'*64),
    lambda m:m['generation'].update(rng='default'),
    lambda m:m['generation']['complete_seeds'].__setitem__(0,999),
    lambda m:m['generation'].update(raw_scale=1.),
    lambda m:m['generation'].update(filter='wrap'),
    lambda m:m['execution']['cell_to_event'].__setitem__(0,1),
    lambda m:m.update(dtype='<f4'),
    lambda m:m.update(runtime={'wrong':True}),
    lambda m:m['generation']['crop'][0].__setitem__(0,2),
])
def test_recipe_rejection(tmp_path,double,mutation):
    _,m,p,ids=recipe(tmp_path);mutation(m);p.write_text(json.dumps(m))
    with pytest.raises(ValueError):load(p,ids)


def test_identity_and_fail_before_mutation(tmp_path,double,monkeypatch):
    cp,calls=double;r,m,p,ids=recipe(tmp_path);s=load(p,ids)
    with pytest.raises(ValueError):gpu.GPUIndexedScatteringPhases(p,descriptor_sha256='0'*64,source_identities=ids)
    with pytest.raises(ValueError):load(p,{'wrong':'0'*64})
    with pytest.raises(ValueError):s.validate_for(replace(r,backend=replace(r.backend,backend='numpy')),required_cells=3)
    with pytest.raises(ValueError):s.apply_to(np.ones((1,12,8),complex),0,xp=np)
    for j in [-1,3,True,1.2]:
        with pytest.raises(IndexError):s.phase(j)
    f=np.ones((1,12,8),complex).view(DeviceArray)
    original=gpu._gpu
    def bad_filter(raw,*a,output,**kw):output[:]=np.nan
    monkeypatch.setattr(gpu,'_gpu',lambda:(cp,original()[1],bad_filter))
    with pytest.raises(ValueError,match='Nonfinite'):s.apply_to(f,0,xp=cp)
    assert np.all(f==1)
    p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError):s.phase(0)


@pytest.mark.parametrize('mode',stream.PR_STREAMING_MODES)
def test_common_model_dispatch(tmp_path,double,mode):
    # Execute the actual common application block, never the scientific march.
    cp,calls=double;r,m,p,ids=recipe(tmp_path);r=replace(r,solver=replace(r.solver,mode=mode))
    s=load(p,ids);s.validate_for(r,required_cells=3);block=application_block(stream)
    results=[]
    for replay in range(2):
        a=np.ones((1,12,8),complex).view(DeviceArray)
        for k in range(3):
            exec(block,dict(stream.__dict__,A=a,z_index=k,xp=cp,request=r,frozen_scattering=s))
        results.append(a)
    assert np.array_equal(*results) and calls==[11,22,33]*2


def test_runtime_change_rejected(tmp_path,double,monkeypatch):
    _,_,p,ids=recipe(tmp_path);s=load(p,ids)
    monkeypatch.setattr(gpu,'gpu_runtime_fingerprint',lambda:{'changed':True})
    with pytest.raises(ValueError,match='runtime'):s.phase(0)


def test_actual_cupy_synthetic(tmp_path):
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()==0:pytest.skip('CUDA device unavailable')
    r,m,p,ids=recipe(tmp_path);s=load(p,ids)
    from cupyx.scipy.ndimage import gaussian_filter
    from cupy.cuda import curand
    saved={}
    for j in [2,0,1,2,0]:
        phase=s.phase(j);raw=cp.random.RandomState(m['generation']['complete_seeds'][j],method=curand.CURAND_RNG_PSEUDO_XORWOW).normal(0,m['generation']['raw_scale'],size=(17,13),dtype=cp.float64)
        expected=gaussian_filter(raw,m['generation']['sigma_pixels'],mode='reflect',order=0,truncate=4.)[3:15,2:10]*m['generation']['post_filter_scale']
        assert bool(cp.array_equal(phase,expected))
        data=cp.asnumpy(phase).tobytes()
        assert data==saved.setdefault(j,data)
    f=cp.ones((1,12,8),cp.complex128);s.apply_to(f,2,xp=cp)
    assert bool(cp.array_equal(f,cp.exp(1j*s.phase(2))[None]))


def test_reject_before_runner_backend_and_plane_guard(tmp_path,double,monkeypatch):
    r,m,p,ids=recipe(tmp_path);s=load(p,ids)
    monkeypatch.setattr(stream,'get_backend',lambda *a,**k:pytest.fail('Reached backend before source validation'))
    wrong=replace(r,solver=replace(r.solver,volume_noise_seeds=(12,22,33)))
    with pytest.raises(ValueError,match='context'):stream.run_pr_static_streaming(wrong,frozen_scattering=s)
    monkeypatch.setattr(gpu,'MAX_ORIGINAL_PLANE_BYTES',8)
    with pytest.raises(ValueError,match='allocation guard'):load(p,ids)


def test_original_figure_geometry_metadata_only(tmp_path,double):
    r,_,_,ids=recipe(tmp_path)
    r=replace(r,grid=replace(r.grid,Nx=9600,Ny=3456,x_aperture_um=9600*3000/16384,y_aperture_um=3456*2000/4096,z_length_um=5000.,dz_um=2.),solver=replace(r.solver,volume_noise_correlation_um=.4,volume_noise_seeds=tuple(range(2500))))
    m=gpu.gpu_indexed_phase_descriptor(r,original_shape=(16384,4096),original_aperture_um=(3000.,2000.),crop=((5052,14652),(320,3776)),events=50,source_identities=ids,execution_dz_um=1.)
    assert m['generation']['full_Nz']==2500
    assert m['generation']['raw_scale']==np.sqrt(.02/2500*4*np.pi)
    assert m['execution']['cell_to_event']==[x for j in range(50) for x in (None,j)]
    assert double[1]==[] # metadata only, no Figure 11 noise


def test_actual_cupy_global_state_and_refinement(tmp_path):
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()==0:pytest.skip('CUDA device unavailable')
    r,m,p,ids=recipe(tmp_path,refine=2);s=load(p,ids)
    cp.random.seed(781)
    expected=cp.random.normal(size=17)
    cp.random.seed(781)
    s.phase(2)
    assert bool(cp.array_equal(cp.random.normal(size=17),expected))
    field=cp.ones((2,12,8),cp.complex128)
    s.apply_to(field,0,xp=cp)
    assert bool(cp.all(field==1))
    s.apply_to(field,1,xp=cp)
    reference=cp.exp(1j*s.phase(0))[None]
    assert bool(cp.all(field==reference))


def test_no_recipe_substitution(tmp_path,double):
    r,m,p,ids=recipe(tmp_path)
    with pytest.raises(ValueError):gpu.historical.IndexedScatteringPhases(p,descriptor_sha256=file_sha256(p),source_identities=ids)
    m['algorithm']=gpu.historical._ALGORITHM
    m['generation']['rng']='numpy.RandomState normal, original C order'
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):load(p,ids)


def test_headless_import_does_not_load_cupy():
    import subprocess,sys
    subprocess.run([sys.executable,'-B','-c',
        "import sys; import lcprop.pr.static_streaming; assert 'cupy' not in sys.modules"],check=True)

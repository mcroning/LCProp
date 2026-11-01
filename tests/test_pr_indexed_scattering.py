"""Small synthetic tests only; no Figure 11 generation or propagation."""
from dataclasses import replace
import hashlib,json,tracemalloc,time
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from lcprop.pr.indexed_scattering import (IndexedScatteringPhases,indexed_phase_descriptor,
    execution_seed_schedule,compare_regenerated_phase)
from lcprop.pr.phase_producer import produce_disk_phases
from lcprop.pr.disk_scattering import DiskScatteringPhases,file_sha256
from tests.test_pr_frozen_scattering import fixture,stub_pass
import lcprop.pr.static_streaming as stream


def make(tmp_path,shape=(17,13),crop=((3,15),(2,10)),events=3,refine=1):
 r,*_=fixture(tmp_path)
 r=replace(r,grid=replace(r.grid,Nx=crop[0][1]-crop[0][0],Ny=crop[1][1]-crop[1][0],x_aperture_um=float(crop[0][1]-crop[0][0]),y_aperture_um=float(crop[1][1]-crop[1][0])),solver=replace(r.solver,volume_noise_correlation_um=1.2))
 src=tmp_path/'source-id';src.write_bytes(b'synthetic original inputs');ids={'fixture':file_sha256(src)}
 m=indexed_phase_descriptor(r,original_shape=shape,original_aperture_um=tuple(float(n) for n in shape),crop=crop,events=events,source_identities=ids,execution_dz_um=r.grid.dz_um/refine)
 p=tmp_path/'recipe.json';p.write_text(json.dumps(m));h=file_sha256(p)
 return r,m,p,h,ids


def load(p,h,ids):return IndexedScatteringPhases(p,descriptor_sha256=h,source_identities=ids)

@pytest.mark.parametrize('shape,crop',[((17,13),((3,15),(2,10))),((12,6),((0,12),(0,6))),((17,13),((0,4),(0,5))),((17,13),((13,17),(9,13))),((2,3),((0,2),(0,3)))])
def test_full_reference_disk_and_order(tmp_path,shape,crop):
 r,m,p,h,ids=make(tmp_path,shape,crop);s=load(p,h,ids)
 s.validate_for(r,required_cells=3);g=m['generation']
 mp,dh=produce_disk_phases(tmp_path/'disk',request=r,original_shape=shape,original_aperture_um=tuple(float(n) for n in shape),crop=crop,cells=3,source_files={'fixture':(tmp_path/'source-id',ids['fixture'])})
 disk=DiskScatteringPhases(mp,manifest_sha256=dh,source_identities=ids)
 for j in [2,0,2,1,0]:
  raw=np.random.RandomState(g['complete_seeds'][j]).normal(0,g['raw_scale'],size=shape)
  expected=gaussian_filter(raw,g['sigma_pixels'])*g['post_filter_scale']
  (a,b),(c,d)=crop;got=s.phase(j)
  assert got.tobytes()==expected[a:b,c:d].tobytes()==disk.phase(j).tobytes()
  assert not got.flags.writeable
  with pytest.raises(ValueError):got.setflags(write=True)
 assert s.provenance()['retained_phase_bytes']==0
 with pytest.raises(AttributeError):s._raw=b'x'


def test_overlap(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';a.mkdir();b.mkdir()
 _,_,p,h,i=make(a,crop=((2,14),(1,11)));_,_,q,k,j=make(b,crop=((4,12),(3,9)))
 assert load(p,h,i).phase(1)[2:10,2:8].tobytes()==load(q,k,j).phase(1).tobytes()

@pytest.mark.parametrize('factor',[1,2,4])
def test_physical_events_refinement(tmp_path,factor):
 r,m,p,h,ids=make(tmp_path,refine=factor);s=load(p,h,ids)
 refined=replace(r,grid=replace(r.grid,dz_um=r.grid.dz_um/factor),solver=replace(r.solver,volume_noise_seeds=execution_seed_schedule(m)))
 s.validate_for(refined,required_cells=3*factor)
 stream._validate_request(refined)
 a=np.ones((1,r.grid.Nx,r.grid.Ny),complex);expected=a.copy()
 for cell in range(3*factor):
  before=a.copy();s.apply_to(a,cell,xp=np);event=s.event_for_cell(cell)
  if event is None:assert np.array_equal(a,before)
  else:
   assert (cell+1)*refined.grid.dz_um==m['event_positions_um'][event]
   expected*=np.exp(1j*s.phase(event))[None,:,:]
 assert np.array_equal(a,expected)
 for event in range(3):
  g=m['generation'];full=gaussian_filter(np.random.RandomState(g['complete_seeds'][event]).normal(0,g['raw_scale'],size=(17,13)),g['sigma_pixels'])*g['post_filter_scale']
  assert s.phase(event).tobytes()==full[3:15,2:10].tobytes()

@pytest.mark.parametrize('mode',stream.PR_STREAMING_MODES)
def test_modes_replay(tmp_path,monkeypatch,mode):
 r,m,p,h,ids=make(tmp_path,shape=(12,6),crop=((0,12),(0,6)));r=replace(r,solver=replace(r.solver,mode=mode));s=load(p,h,ids);seen=[]
 monkeypatch.setattr(stream,'_run_streaming_pass',stub_pass(stream,seen))
 monkeypatch.setattr(stream,'_volume_noise_phase',lambda *a,**k:pytest.fail('Default generator used'))
 result=stream.run_pr_static_streaming(r,frozen_scattering=s)
 expected=result.A_initial.copy()
 for j in range(3):expected*=np.exp(1j*s.phase(j))[None,:,:]
 assert np.array_equal(result.A_final,expected) and seen==[s,s] and result.replay_diagnostics['consistent']

@pytest.mark.parametrize('bad',[-1,3,True,1.5,None])
def test_indices(tmp_path,bad):
 _,_,p,h,i=make(tmp_path);s=load(p,h,i)
 with pytest.raises(IndexError):s.phase(bad)
 with pytest.raises(IndexError):s.event_for_cell(bad)

@pytest.mark.parametrize('kind',['seed','shape','dtype','event','map','boolmap','algorithm','runtime','sigma','scale','origin','geometry','huge_refinement'])
def test_invalid_descriptor(tmp_path,kind):
 r,m,p,h,ids=make(tmp_path)
 if kind=='seed':m['generation']['complete_seeds'][0]+=1
 elif kind=='shape':m['shape'][1]+=1
 elif kind=='dtype':m['dtype']='<f4'
 elif kind=='event':m['event_positions_um'][0]+=1
 elif kind=='map':m['execution']['cell_to_event']=[1,0,2]
 elif kind=='boolmap':m['execution']['cell_to_event']=[False,1,2]
 elif kind=='algorithm':m['algorithm_sha256']='0'*64
 elif kind=='runtime':m['runtime']['numpy']='wrong'
 elif kind=='sigma':m['generation']['sigma_pixels'][0]*=2
 elif kind=='scale':m['generation']['raw_scale']*=2
 elif kind=='origin':m['generation']['coordinate_convention']='centered crop'
 elif kind=='geometry':m['generation']['crop'][0][0]+=1
 else:m['execution']['refinement_factor']=100001
 p.write_text(json.dumps(m))
 with pytest.raises(ValueError):load(p,file_sha256(p),ids)


def test_corruption_context_source(tmp_path):
 r,m,p,h,ids=make(tmp_path);s=load(p,h,ids)
 with pytest.raises(ValueError):load(p,'0'*64,ids)
 with pytest.raises(ValueError):load(p,h,{'fixture':'0'*64})
 with pytest.raises(ValueError):s.validate_for(replace(r,grid=replace(r.grid,x_aperture_um=20)),required_cells=3)
 with pytest.raises(ValueError):s.validate_for(r,required_cells=4)
 p.write_text(p.read_text()+' ')
 with pytest.raises(ValueError):s.phase(0)


def test_requested_event_only_and_nonfinite(tmp_path,monkeypatch):
 _,m,p,h,ids=make(tmp_path);s=load(p,h,ids);real=np.random.RandomState;seen=[]
 def factory(seed):seen.append(seed);return real(seed)
 monkeypatch.setattr(np.random,'RandomState',factory);s.phase(2)
 assert seen==[m['generation']['complete_seeds'][2]]
 import scipy.ndimage
 monkeypatch.setattr(scipy.ndimage,'gaussian_filter1d',lambda a,*args,**kw:np.full(a.shape,np.nan))
 with pytest.raises(ValueError):s.phase(0)


def test_memory_scaling(tmp_path):
 peaks=[]
 for nx in [65,2049]:
  d=tmp_path/str(nx);d.mkdir();_,_,p,h,ids=make(d,shape=(nx,31),crop=((nx-40,nx-8),(3,27)))
  s=load(p,h,ids);s.phase(0)
  tracemalloc.start();s.phase(2);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop();peaks.append(peak)
 # Fixed output/halo, changing skipped full-grid height must not allocate it.
 assert peaks[1] < peaks[0]+128*1024


def test_comparison_policy():
 a=np.ones((3,4),dtype='<f8');assert compare_regenerated_phase(a,a)['bitwise']
 b=a.copy();b[0,0]=np.nextafter(1.,2.)
 assert compare_regenerated_phase(a,b)['qualification']=='requires_independent_error_budget'
 with pytest.raises(ValueError):compare_regenerated_phase(a,np.full_like(a,np.inf))


def test_cupy_consumption(tmp_path):
 cp=pytest.importorskip('cupy');_,_,p,h,ids=make(tmp_path);s=load(p,h,ids)
 a=cp.ones((1,12,8),cp.complex128);s.apply_to(a,1,xp=cp)
 np.testing.assert_allclose(cp.asnumpy(a),np.exp(1j*s.phase(1))[None],rtol=0,atol=2e-14)

@pytest.mark.parametrize('dz',[0.,-1.,float('nan'),.3])
def test_bad_execution_spacing(tmp_path,dz):
 r,m,_,_,ids=make(tmp_path)
 with pytest.raises(ValueError):indexed_phase_descriptor(r,original_shape=(17,13),original_aperture_um=(17.,13.),crop=((3,15),(2,10)),events=3,source_identities=ids,execution_dz_um=dz)


def test_large_event_number_does_not_generate_prefix(tmp_path,monkeypatch):
 r,m,p,h,ids=make(tmp_path);r=replace(r,grid=replace(r.grid,z_length_um=30*r.grid.dz_um),solver=replace(r.solver,volume_noise_seeds=tuple(range(30))))
 m=indexed_phase_descriptor(r,original_shape=(17,13),original_aperture_um=(17.,13.),crop=((3,15),(2,10)),events=30,source_identities=ids)
 p.write_text(json.dumps(m));s=load(p,file_sha256(p),ids)
 real=np.random.RandomState;seen=[]
 def factory(seed):seen.append(seed);return real(seed)
 monkeypatch.setattr(np.random,'RandomState',factory)
 s.phase(29);assert seen==[29]


def test_bounded_output_memory(tmp_path):
 measurements=[]
 for nx,ny in [(256,128),(512,256)]:
  d=tmp_path/str(nx);d.mkdir();_,_,p,h,ids=make(d,shape=(nx+8,ny+8),crop=((4,nx+4),(4,ny+4)))
  s=load(p,h,ids);s.phase(0)
  tracemalloc.start();a=s.phase(1);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
  # Mutable output plus immutable return; halo/row work and two 64-KiB
  # streaming identity hashes fit a separately declared 512-KiB allowance.
  assert peak<2*a.nbytes+512*1024
  measurements.append((a.nbytes,peak))
 assert measurements[1][1]-measurements[0][1] < 2*(measurements[1][0]-measurements[0][0])+128*1024

@pytest.mark.parametrize('shape,dtype',[((1,12,8),np.complex64),((1,11,8),np.complex128),((12,8),np.complex128)])
def test_field_contract(tmp_path,shape,dtype):
 _,_,p,h,ids=make(tmp_path);s=load(p,h,ids)
 with pytest.raises(ValueError):s.apply_to(np.ones(shape,dtype),0,xp=np)


def test_bool_endpoint_rejected(tmp_path):
 _,m,p,h,ids=make(tmp_path);m['event_positions_um'][0]=True;p.write_text(json.dumps(m))
 with pytest.raises(ValueError):load(p,file_sha256(p),ids)


def test_filter_weight_memory_bound(tmp_path):
 r,_,_,_,ids=make(tmp_path)
 r=replace(r,grid=replace(r.grid,y_aperture_um=8e-10))
 with pytest.raises(ValueError,match='bounded Gaussian'):
  indexed_phase_descriptor(r,original_shape=(17,13),original_aperture_um=(17.,13e-10),crop=((3,15),(2,10)),events=3,source_identities=ids)

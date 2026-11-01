"""Synthetic construction/application only; no Figure 11 propagation."""
from dataclasses import replace
import hashlib
import json
import tracemalloc
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from lcprop.pr.disk_scattering import DiskScatteringPhases, file_sha256
from lcprop.pr.phase_producer import produce_disk_phases
from tests.test_pr_frozen_scattering import fixture, stub_pass
import lcprop.pr.static_streaming as stream


def make(tmp_path, crop=((0,12),(0,6)), mode=stream.PR_STREAMING_PRODUCTION):
    r,*_ = fixture(tmp_path, mode=mode)
    g = r.grid
    r = replace(r,grid=replace(g,Nx=crop[0][1]-crop[0][0],Ny=crop[1][1]-crop[1][0],
        x_aperture_um=g.x_aperture_um*(crop[0][1]-crop[0][0])/12,
        y_aperture_um=g.y_aperture_um*(crop[1][1]-crop[1][0])/6))
    src=tmp_path/'source';src.write_bytes(b'synthetic source identity')
    digest=file_sha256(src)
    mp,h=produce_disk_phases(tmp_path/'phases',request=r, original_shape=(12,6),
        original_aperture_um=(g.x_aperture_um,g.y_aperture_um),crop=crop,cells=3,
        source_files={'source':(src,digest)})
    return r,mp,h,{'source':digest}


def load(mp,h,ids):
    return DiskScatteringPhases(mp,manifest_sha256=h,source_identities=ids)


@pytest.mark.parametrize('crop',[((0,12),(0,6)),((2,10),(1,5)),((0,8),(0,4)),((5,12),(3,6))])
def test_original_filter_and_overlap(tmp_path,crop):
    r,mp,h,ids=make(tmp_path,crop)
    source=load(mp,h,ids);m=json.loads(mp.read_text());gen=m['generation']
    source.validate_for(r,required_cells=3)
    for i,seed in enumerate(gen['complete_seeds']):
        raw=np.random.RandomState(seed).normal(0,gen['raw_scale'],size=(12,6))
        expected=gaussian_filter(raw,gen['sigma_pixels'])*gen['post_filter_scale']
        (x0,x1),(y0,y1)=crop
        assert np.array_equal(source.phase(i),expected[x0:x1,y0:y1])
        assert not source.phase(i).flags.writeable


@pytest.mark.parametrize('mode',stream.PR_STREAMING_MODES)
def test_all_modes_replay(tmp_path,monkeypatch,mode):
    r,mp,h,ids=make(tmp_path,mode=mode);source=load(mp,h,ids);seen=[]
    monkeypatch.setattr(stream,'_run_streaming_pass',stub_pass(stream,seen))
    monkeypatch.setattr(stream,'_volume_noise_phase',lambda *a,**k:pytest.fail('Regeneration'))
    result=stream.run_pr_static_streaming(r,frozen_scattering=source)
    expected=result.A_initial.copy()
    for i in range(3):expected*=np.exp(1j*source.phase(i))[None,:,:]
    assert np.array_equal(expected,result.A_final)
    assert seen==[source,source] and result.replay_diagnostics['consistent']


@pytest.mark.parametrize('kind',['missing','corrupt','replace','manifest','nan','inf','size'])
def test_recheck_before_use(tmp_path,kind):
    r,mp,h,ids=make(tmp_path);source=load(mp,h,ids)
    p=mp.parent/'phase-000000.f64';p.chmod(0o644)
    if kind=='missing':p.unlink()
    elif kind=='replace':
        data=p.read_bytes();p.unlink();p.write_bytes(data)
    elif kind=='manifest':mp.chmod(0o644);mp.write_bytes(mp.read_bytes()+b' ')
    elif kind=='size':p.write_bytes(b'x')
    else:
        a=np.frombuffer(p.read_bytes(),dtype='<f8').copy()
        a[0]={'nan':np.nan,'inf':np.inf,'corrupt':123.}[kind];p.write_bytes(a.tobytes())
    with pytest.raises((ValueError,OSError)):source.phase(0)
    with pytest.raises((ValueError,OSError)):source.validate_for(r,required_cells=3)


@pytest.mark.parametrize('kind',['order','z','shape','dtype','identity','nonfinite'])
def test_manifest_rejection(tmp_path,kind):
    r,mp,h,ids=make(tmp_path);m=json.loads(mp.read_text())
    if kind=='order':m['planes'].reverse()
    if kind=='z':m['planes'][0]['z_um']+=1
    if kind=='shape':m['shape'][1]+=1
    if kind=='dtype':m['dtype']='<f4'
    if kind=='identity':m['source_identities']['source']='0'*64
    if kind=='nonfinite':
        p=mp.parent/m['planes'][0]['file'];p.chmod(0o644)
        a=np.frombuffer(p.read_bytes(),dtype='<f8').copy();a[0]=np.nan;p.write_bytes(a.tobytes())
        m['planes'][0]['sha256']=file_sha256(p)
    mp.chmod(0o644);mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):load(mp,file_sha256(mp),ids)


def test_bad_context_and_bounds(tmp_path,monkeypatch):
    r,mp,h,ids=make(tmp_path);source=load(mp,h,ids)
    for i in (-1,3,True,1.5):
        with pytest.raises(IndexError):source.phase(i)
    with pytest.raises(ValueError):source.validate_for(r,required_cells=4)
    wrong=replace(r,solver=replace(r.solver,volume_noise_seeds=(1,2,3)))
    monkeypatch.setattr(stream,'get_backend',lambda *a:pytest.fail('Reached backend'))
    with pytest.raises(ValueError):stream.run_pr_static_streaming(wrong,frozen_scattering=source)
    with pytest.raises(AttributeError):source._raw=b''
    with pytest.raises(ValueError):source.phase(0)[0,0]=2


def test_cupy_application_when_available(tmp_path):
    cp=pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount()==0:pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('No CUDA runtime')
    r,mp,h,ids=make(tmp_path);source=load(mp,h,ids)
    source.validate_for(replace(r,backend=replace(r.backend,backend='cupy')),required_cells=3)
    a=cp.ones((1,12,6),dtype=cp.complex128)
    expected=np.ones((1,12,6),dtype=np.complex128)
    for i in range(3):
        source.apply_to(a,i,xp=cp);source.apply_to(expected,i,xp=np)
    np.testing.assert_allclose(cp.asnumpy(a),expected,rtol=1e-14,atol=1e-14)

@pytest.mark.parametrize('field',['complete_seeds','crop','raw_scale','sigma_pixels','full_Nz'])
def test_physical_manifest_rejected(tmp_path,field):
    r,mp,h,ids=make(tmp_path);m=json.loads(mp.read_text());gen=m['generation']
    if field=='complete_seeds':gen[field][0]+=1
    elif field=='crop':gen[field][0][0]+=1
    elif field=='sigma_pixels':gen[field][0]*=2
    else:gen[field]*=2
    mp.chmod(0o644);mp.write_text(json.dumps(m))
    with pytest.raises(ValueError):load(mp,file_sha256(mp),ids)


def large_fixture(tmp_path, *, nx=129, cells=3, crop=None):
    r,*_=fixture(tmp_path)
    ny=65
    if crop is None:crop=((0,nx),(0,ny))
    (x0,x1),(y0,y1)=crop
    r=replace(r,grid=replace(r.grid,Nx=x1-x0,Ny=y1-y0,
        x_aperture_um=float(x1-x0),y_aperture_um=float(y1-y0),z_length_um=5.*cells),
        solver=replace(r.solver,volume_noise_seeds=tuple(range(cells)),volume_noise_correlation_um=2.3))
    src=tmp_path/'source';src.write_bytes(b'fixture');ids={'source':file_sha256(src)}
    mp,h=produce_disk_phases(tmp_path/'phases',request=r,original_shape=(nx,ny),
        original_aperture_um=(float(nx),float(ny)),crop=crop,cells=cells,
        source_files={'source':(src,ids['source'])})
    return r,mp,h,ids


def test_odd_rng_chunks_full_halos_and_overlap(tmp_path):
    paths=[tmp_path/str(i) for i in range(2)]
    for p in paths:p.mkdir()
    _,mp,h,ids=large_fixture(paths[0]);full=load(mp,h,ids)
    _,mp,h,ids=large_fixture(paths[1],crop=((23,101),(13,54)));crop=load(mp,h,ids)
    m=full.provenance()['manifest'];g=m['generation']
    for i in range(3):
        a=full.phase(i)
        raw=np.random.RandomState(i).normal(0,g['raw_scale'],size=(129,65))
        expected=gaussian_filter(raw,g['sigma_pixels'])*g['post_filter_scale']
        assert np.array_equal(a,expected)
        assert np.array_equal(a[23:101,13:54],crop.phase(i))


def _peak(call):
    import gc
    gc.collect()
    tracemalloc.start()
    try:
        value = call()
        _, peak = tracemalloc.get_traced_memory()
        del value
        return peak
    finally:
        tracemalloc.stop()


def _metadata_peak(mp):
    # Up to three decoded manifests coexist (constructor/validation/load).
    # Measure this separately from plane I/O, including the encoded bytes.
    return _peak(lambda: (mp.read_bytes(), [json.loads(mp.read_bytes()) for _ in range(3)]))


def _record_scaling(name, rows):
    import os
    from pathlib import Path
    path = os.environ.get('LCPROP_MEMORY_SCALING_EVIDENCE')
    if path:
        p = Path(path)
        records = json.loads(p.read_text()) if p.exists() else {}
        records[name] = rows
        p.write_text(json.dumps(records, indent=2) + '\n')


def _scaling_bound(rows):
    from lcprop.pr.disk_scattering import _IO_BUFFER_BYTES
    # Four fixed 64-KiB blocks cover sequential I/O buffers and bookkeeping
    # differences; separately measured metadata covers plane-count growth.
    # This allowance is independent of Nx*Ny*Nz and never fitted to a peak.
    return 4*_IO_BUFFER_BYTES + 2*max(row['metadata_peak'] for row in rows)


def _assert_volume_scaling(rows):
    assert max(r['peak'] for r in rows)-min(r['peak'] for r in rows) <= _scaling_bound(rows)


def _assert_plane_scaling(rows):
    first,last=rows[0],rows[-1]
    assert abs((last['peak']-first['peak'])-(last['plane_bytes']-first['plane_bytes'])) <= _scaling_bound(rows)


def _reader_peak(r,mp,h,ids):
    def use():
        source=load(mp,h,ids)
        cells=len(source.provenance()['manifest']['planes'])
        source.validate_for(r,required_cells=cells)
        for _ in range(2):
            for i in range(cells):
                a=source.phase(i)
                del a
        assert source.provenance()['retained_phase_bytes']==0
    return _peak(use)


def test_memory_does_not_scale_with_volume(tmp_path):
    rows=[]
    for cells in (2,12,24):
        p=tmp_path/str(cells);p.mkdir()
        r,mp,h,ids=large_fixture(p,nx=1025,cells=cells)
        rows.append(dict(cells=cells,plane_bytes=1025*65*8,peak=_reader_peak(r,mp,h,ids),metadata_peak=_metadata_peak(mp)))
    _record_scaling('reader_plane_count',rows)
    _assert_volume_scaling(rows)


def test_reader_scales_as_one_plane(tmp_path):
    rows=[]
    for nx in (1025,4097,8193):
        p=tmp_path/str(nx);p.mkdir()
        r,mp,h,ids=large_fixture(p,nx=nx,cells=2)
        rows.append(dict(nx=nx,cells=2,plane_bytes=nx*65*8,peak=_reader_peak(r,mp,h,ids),metadata_peak=_metadata_peak(mp)))
    _record_scaling('reader_plane_size',rows)
    _assert_plane_scaling(rows)


def test_producer_memory_independent_of_original_height(tmp_path):
    rows=[]
    for nx in (129,2049,8193):
        p=tmp_path/str(nx);p.mkdir()
        peak=_peak(lambda:large_fixture(p,nx=nx,cells=1,crop=((20,90),(10,50))))
        rows.append(dict(nx=nx,peak=peak,metadata_peak=_metadata_peak(p/'phases/manifest.json')))
    _record_scaling('producer_original_height',rows)
    _assert_volume_scaling(rows)


def test_producer_does_not_retain_phase_volume(tmp_path):
    rows=[]
    for cells in (1,6,12):
        p=tmp_path/str(cells);p.mkdir()
        peak=_peak(lambda:large_fixture(p,nx=257,cells=cells))
        rows.append(dict(cells=cells,peak=peak,metadata_peak=_metadata_peak(p/'phases/manifest.json')))
    _record_scaling('producer_plane_count',rows)
    _assert_volume_scaling(rows)


@pytest.mark.parametrize('kind',['stack','two_planes','full_grid'])
def test_scaling_gate_rejects_unbounded_regressions(kind):
    # Deliberate counterexamples: ensure the allowance cannot hide the
    # scientific memory failures these tests are meant to detect.
    rows=[dict(peak=600_000,metadata_peak=10_000,plane_bytes=533_000),
          dict(peak=650_000,metadata_peak=20_000,plane_bytes=533_000)]
    if kind=='stack':rows[-1]['peak']+=23*533_000
    if kind=='full_grid':rows[-1]['peak']+=8193*65*8
    if kind=='two_planes':
        rows[-1]['plane_bytes']=8193*65*8
        rows[-1]['peak']=rows[0]['peak']+2*(rows[-1]['plane_bytes']-rows[0]['plane_bytes'])
    with pytest.raises(AssertionError):
        (_assert_plane_scaling if kind=='two_planes' else _assert_volume_scaling)(rows)


def test_modified_between_first_pass_and_replay(tmp_path,monkeypatch):
    r,mp,h,ids=make(tmp_path);source=load(mp,h,ids);seen=[]
    base=stub_pass(stream,seen)
    def march(**kw):
        result=base(**kw)
        p=mp.parent/'phase-000000.f64'
        p.chmod(0o644);p.write_bytes(b'x'*p.stat().st_size)
        return result
    monkeypatch.setattr(stream,'_run_streaming_pass',march)
    with pytest.raises(ValueError,match='SHA256'):stream.run_pr_static_streaming(r,frozen_scattering=source)
    assert len(seen)==2

@pytest.mark.parametrize('bad',['seed_count','spacing','source_hash','workspace'])
def test_producer_fails_before_output(tmp_path,bad):
    r,*_=fixture(tmp_path)
    original_shape=(12,6);aperture=(24.,18.)
    src=tmp_path/'input';src.write_bytes(b'original');digest=file_sha256(src)
    if bad=='seed_count':r=replace(r,solver=replace(r.solver,volume_noise_seeds=(11,)))
    if bad=='spacing':aperture=(25.,18.)
    if bad=='source_hash':digest='0'*64
    if bad=='workspace':
        original_shape=(12,1_000_000);aperture=(24.,3_000_000.)
    with pytest.raises(ValueError):produce_disk_phases(tmp_path/'absent',request=r,
        original_shape=original_shape,original_aperture_um=aperture,crop=((0,12),(0,6)),cells=3,
        source_files={'source':(src,digest)})
    assert not (tmp_path/'absent').exists()


def test_symlink_plane_rejected(tmp_path):
    _,mp,h,ids=make(tmp_path)
    p=mp.parent/'phase-000000.f64';q=mp.parent/'redirect';p.rename(q);p.symlink_to(q)
    with pytest.raises(OSError):load(mp,h,ids)


def test_io_buffers_are_explicit_and_hashing_is_chunk_bounded(tmp_path,monkeypatch):
    import builtins
    from types import SimpleNamespace
    import lcprop.pr.disk_scattering as disk
    import lcprop.pr.phase_producer as producer
    calls=[]
    original_open=builtins.open
    def hashing_open(path,mode,**kw):
        assert kw['buffering']==0
        base=original_open(path,mode,**kw)
        class Reader:
            def __enter__(self):return self
            def __exit__(self,*a):base.close()
            def readinto(self,buffer):
                calls.append(len(buffer));return base.readinto(buffer)
        return Reader()
    monkeypatch.setattr(disk,'open',hashing_open,raising=False)
    blob=tmp_path/'blob';blob.write_bytes(b'x'*(4*disk._IO_BUFFER_BYTES+3))
    assert disk.file_sha256(blob)==hashlib.sha256(blob.read_bytes()).hexdigest()
    assert calls and set(calls)=={disk._IO_BUFFER_BYTES}
    original_fdopen=disk.os.fdopen
    def fdopen(fd,mode,**kw):
        assert kw['buffering']==disk._IO_BUFFER_BYTES
        return original_fdopen(fd,mode,**kw)
    monkeypatch.setattr(disk.os,'fdopen',fdopen)
    original_temp=producer.tempfile.TemporaryFile
    def temporary(**kw):
        assert kw['buffering']==disk._IO_BUFFER_BYTES
        return original_temp(**kw)
    monkeypatch.setattr(producer,'tempfile',SimpleNamespace(TemporaryFile=temporary))
    r,mp,h,ids=make(tmp_path)
    load(mp,h,ids).validate_for(r,required_cells=3)

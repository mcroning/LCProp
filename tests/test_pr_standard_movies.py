"""Bounded movie products are observations, available without Full retrieval."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from lcprop.pr.trajectory import movie_artifacts, unpack_frames
from lcprop.pr.td_movie_products import AcceptedMovieProducts
from lcprop.pr.visualization import downsample_td_movie_frame
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.gui.movie_export import movie_display, save_movie


@pytest.mark.parametrize('transverse', [False, True])
def test_workflow_movies_fast_roundtrip_and_science(transverse):
    if transverse:
        from tests.test_pr_transverse_continuation import request
        from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent as run
        from lcprop.pr.transverse.timedependent_transport_codec import encode_pr_transverse_timedependent_transport_result as encode, decode_pr_transverse_timedependent_transport_result as decode
        state='psi_final'
    else:
        from tests.test_pr_reduced_physical_continuation import request
        from lcprop.pr.workflow import run_pr_timedependent as run
        from lcprop.pr.timedependent_transport_codec import encode_pr_timedependent_transport_result as encode, decode_pr_timedependent_transport_result as decode
        state='E_final'
    r=request();plain=run(r);observed=run(r,progress_callback=lambda _: None)
    np.testing.assert_array_equal(plain.A_final,observed.A_final)
    np.testing.assert_array_equal(getattr(plain,state),getattr(observed,state))
    if hasattr(plain,'source_intensity_stack'):
        np.testing.assert_array_equal(plain.source_intensity_stack, observed.source_intensity_stack)
    packet=encode(observed,result_policy='fast')
    restored=decode(packet.payload.metadata,packet.payload.arrays)
    artifacts=movie_artifacts(restored)
    for key in ('xz','yz','far_field'):
        a=artifacts['td_trajectory_'+key]
        assert len(a.data)<=36 and max(a.data.shape[1:])<=128
        np.testing.assert_array_equal(a.data,movie_artifacts(observed)[a.key].data)
        assert a.metadata['times'][-1]==observed.time_normalized
        assert a.metadata['coordinates']==movie_artifacts(observed)[a.key].metadata['coordinates']
    ff=direction_cosine_spectrum(observed.A_final,dx_um=observed.grid_summary['dx_um'],
        dy_um=observed.grid_summary['dy_um'],wavelength_um=r.beams.channels[0].wavelength_um,
        refractive_index=r.material.refractive_index,coherence_groups=r.beams.coherence_groups)
    a=artifacts['td_trajectory_far_field']
    np.testing.assert_array_equal(a.data[-1],downsample_td_movie_frame(ff.intensity))
    np.testing.assert_array_equal(a.metadata['coordinates']['s_x'],ff.s_x)
    np.testing.assert_array_equal(a.metadata['coordinates']['s_y'],ff.s_y)
    a=artifacts['td_trajectory_xz']
    np.testing.assert_array_equal(a.metadata['coordinates']['z'],
                                  np.arange(1,observed.grid_summary['Nz']+1)*observed.grid_summary['dz_um'])


def test_backend_cut_reduction_transfer_bounds_and_nonmutation():
    from tests.test_pr_td_movie_backend_transfer import BackendStandIn
    xp=BackendStandIn();nx,ny,nz=260,130,135
    rng=np.random.default_rng(5);field=(rng.normal(size=(1,nx,ny))+1j*rng.normal(size=(1,nx,ny)))
    source=np.ones((nz,nx,ny));source[:, :nx//2,:]=2
    before=source.copy();original=field.copy();transfers=[]
    def host(a):
        assert a.ndim<=2 and a.size<=128*128 and a.nbytes<=65536
        transfers.append((a.shape,a.nbytes));return a.copy()
    grid=SimpleNamespace(xp=xp,summary=lambda:dict(dx_um=.2,dy_um=.3,dz_um=.4))
    request=SimpleNamespace(optical_coupling='frozen_material_published_optical_first_v1',
        material=SimpleNamespace(background_intensity=.1,refractive_index=2.3),
        beams=SimpleNamespace(channels=[SimpleNamespace(wavelength_um=.532)]))
    products=AcceptedMovieProducts()
    products.append(field,source,grid=grid,request=request,reference=2.,groups=('a',),asnumpy=host)
    np.testing.assert_array_equal(source,before);np.testing.assert_array_equal(field,original)
    np.testing.assert_array_equal(products.frames['xz'][0],downsample_td_movie_frame((source[:,:,64].T-.1)*2))
    expected_ff=direction_cosine_spectrum(field,dx_um=.2,dy_um=.3,wavelength_um=.532,
        refractive_index=2.3,coherence_groups=('a',))
    np.testing.assert_array_equal(products.frames['far_field'][0],downsample_td_movie_frame(expected_ff.intensity))
    edges=np.linspace(0,nx,129,dtype=int)
    expected_axis=[expected_ff.s_x[a:b].mean() for a,b in zip(edges[:-1],edges[1:])]
    np.testing.assert_array_equal(products.metadata['far_field']['coordinates']['s_x'],expected_axis)
    assert len(transfers)==5 and max(n for _,n in transfers)==65536
    metadata={};products.attach(metadata)
    assert unpack_frames(metadata['additional_movies']['xz']['frames']).shape==(1,128,128)


def artifact():
    from lcprop.products.data_model import ArtifactData
    return ArtifactData('td_trajectory_far_field','Far field',np.array([[[0.,1.],[2.,4.]],[[0.,.1],[.2,.4]]]),
        'application/x-lcprop-trajectory','x',dict(times=[0.,3.],axes=['s_x','s_y'],
        coordinates={'s_x':[-.2,.2],'s_y':[-.1,.1]},value_unit='density',segment_start=0.,availability='Retained'))


def test_global_linear_log_scaling():
    a=artifact();before=a.data.copy()
    linear,limits=movie_display(a.data)
    assert limits==(0.,4.) and linear[1].max()==.4
    log,bounds=movie_display(a.data,logarithmic=True)
    assert bounds[1]-bounds[0]==pytest.approx(8.)
    assert log[0,1,1]-log[1,1,1]==pytest.approx(1.)
    assert log[0,0,0]==log[1,0,0]
    np.testing.assert_array_equal(a.data,before)


def test_missing_ffmpeg_and_external_export(monkeypatch,tmp_path):
    import lcprop.gui.movie_export as export
    monkeypatch.delenv('LCPROP_FFMPEG',raising=False)
    monkeypatch.setattr(export.shutil,'which',lambda _:None)
    with pytest.raises(RuntimeError,match='outside lcprop-new-user'):
        save_movie(artifact(),tmp_path/'x.mp4')
    from matplotlib.figure import Figure  # initialize optional presentation dependencies before subprocess spy
    calls=[]
    monkeypatch.setenv('LCPROP_FFMPEG','/external/presentation/ffmpeg')
    def encode(command,**kwargs):
        calls.append(command);assert len(kwargs['input'])==2*640*480*3
        return SimpleNamespace(stdout=b'mp4')
    monkeypatch.setattr(export.subprocess,'run',encode)
    save_movie(artifact(),tmp_path/'x.mp4',logarithmic=True)
    assert calls[0][0]=='/external/presentation/ffmpeg'
    assert (tmp_path/'x.mp4').read_bytes()==b'mp4'


def test_gui_movie_selection_export_and_missing_encoder(monkeypatch,tmp_path):
    from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
    from lcprop.gui.workspace import Workspace
    from lcprop.products.data_model import RunData,Geometry
    import lcprop.gui.movie_export as export
    app=QApplication.instance() or QApplication([]);w=Workspace();p=w.trajectory_player
    a=artifact();base=replace(a,key='td_trajectory')
    p.set_run_data(RunData(workflow='pr_timedependent',geometry=Geometry(),artifacts={base.key:base,a.key:a}))
    p.quantity.setCurrentIndex(1);p.show_frame(1)
    assert w.image_pane.image_view._field.axes==('s_x','s_y')
    assert p.save.isEnabled()
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *args: (str(tmp_path/'saved'),'MP4'))
    calls=[]
    monkeypatch.setattr(export,'save_movie',lambda *args,**kw: calls.append((args,kw)))
    p.save.click();assert calls[0][0][0] is a and calls[0][0][1].endswith('saved.mp4')
    def fail(*args,**kwargs):raise RuntimeError('ffmpeg unavailable; playback remains')
    monkeypatch.setattr(export,'save_movie',fail)
    messages=[];monkeypatch.setattr(QMessageBox,'warning',lambda *args:messages.append(args[-1]))
    p.save.click();assert 'ffmpeg unavailable' in messages[0] and p.play.isEnabled()
    w.close()


def test_zero_movie_log_is_finite_and_does_not_mutate():
    values=np.zeros((2,4,3),dtype=np.float32);values[1,0,0]=-1e-20
    before=values.copy()
    shown,limits=movie_display(values,logarithmic=True)
    assert np.isfinite(shown).all() and np.isfinite(limits).all()
    assert limits[0]<limits[1]
    np.testing.assert_array_equal(values,before)


def test_movie_size_planning_saturates_with_frame_cap():
    from tests.test_pr_reduced_physical_continuation import request
    from lcprop.pr.runtime_estimator import _result_sizes
    r=request()
    def sizes(nt):
        return _result_sizes(replace(r,solver=replace(r.solver,Nt=nt)),
                             shape=(512,512,1000),static=False,full_transverse=False)
    assert sizes(1000)==sizes(10000)
    assert sizes(1)!=sizes(1000)

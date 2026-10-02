"""Published optical coupling with unchanged transverse material-time physics."""
from dataclasses import replace
import numpy as np
import pytest
from lcprop.core.grid import make_grid
from lcprop.optics.splitstep import scalar_angular_spectrum_kernel
from lcprop.pr import workflow as reduced
from lcprop.pr.source import pr_driving_intensity
from lcprop.pr.specs import PR_TD_PUBLISHED_COUPLING, PR_TD_LEGACY_COUPLING
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2
from lcprop.pr.transverse import workflow as w
from lcprop.pr.transverse.transport import state_from_potential
from lcprop.pr.transverse.projection import project_active_field
from tests.test_pr_transverse_production import _request


@pytest.mark.parametrize('nz', [1, 2, 4])
@pytest.mark.parametrize('scatter', [False, True])
def test_arriving_source_and_operation_order(monkeypatch, nz, scatter):
    request = _request(steps=1)
    request = replace(request, grid=replace(request.grid,z_length_um=nz*request.grid.dz_um),
        scattering=(PRCanonicalScatteringSpec(.02,1.,17,1.,PR_CANONICAL_SCATTERING_V2) if scatter else None))
    grid = make_grid(request.grid,xp=np,real_dtype=np.float64)
    rng = np.random.default_rng(13)
    A = rng.normal(size=(1,grid.Nx,grid.Ny)) + 1j*rng.normal(size=(1,grid.Nx,grid.Ny))
    psi = rng.normal(scale=1e-4,size=(nz,grid.Nx,grid.Ny));before=psi.copy()
    dx=request.material.characteristic_wavenumber_per_um*grid.dx_um
    dy=request.material.characteristic_wavenumber_per_um*grid.dy_um
    kernel=scalar_angular_spectrum_kernel(grid.fxy2_um,dz=grid.dz_um,wavelength=.633,n_ref=2.4,xp=np)
    events=[]
    for name,label in [('hop_linear_inplace','P'),('pr_driving_intensity','I'),('delta_n_from_E','material'),('apply_response_screen_inplace','M')]:
        original=getattr(reduced,name)
        def spy(*args,_fn=original,_label=label,**kw):
            events.append(_label);return _fn(*args,**kw)
        monkeypatch.setattr(reduced,name,spy)
    scattering=w._apply_canonical_scattering_after_slice
    def screen(*args,**kw):
        events.append('S');return scattering(*args,**kw)
    monkeypatch.setattr(w,'_apply_canonical_scattering_after_slice',screen)
    def forbidden(*args,**kw):raise AssertionError('old midpoint reached')
    monkeypatch.setattr(w,'advance_pr_slice_with_midpoint_source',forbidden)
    actual,source=w._optical_pass(A,psi,request=request,grid=grid,kernel=kernel,peak_reference=1.,wavelength_um=.633,dx_normalized=dx,dy_normalized=dy)
    assert events==['P','I','material','M','S']*nz
    state=state_from_potential(psi,dx_normalized=dx,dy_normalized=dy,h_y=request.dielectric.h_y,applied_field_x=0.,xp=np)
    active=project_active_field(state.E_x,state.E_y,profile=request.projection,xp=np)
    ref=A.copy();expected=[]
    for k in range(nz):
        ref=np.fft.ifft2(np.fft.fft2(ref,axes=(-2,-1))*kernel,axes=(-2,-1))
        expected.append(pr_driving_intensity(ref,peak_intensity_reference=1.,background_intensity=request.material.background_intensity,coherence_groups=request.beams.coherence_groups,xp=np))
        ref*=np.exp(-2j*request.material.gain_length_product*grid.dz_um/request.grid.z_length_um*active[k])[None]
        scattering(ref,scattering=request.scattering,z_index=k,grid=grid,z_length_um=request.grid.z_length_um,xp=np)
    np.testing.assert_allclose(actual,ref,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(source,np.array(expected),rtol=2e-14,atol=2e-14)
    np.testing.assert_array_equal(psi,before)
    assert w.advance_pr_published_frozen_slice is reduced.advance_pr_published_frozen_slice


def test_persisted_identity_and_legacy_loading():
    import lcprop.persistence
    from lcprop.pr.experiment_codec import encode_pr_transverse_timedependent_request as encode, decode_pr_transverse_timedependent_request as decode
    from lcprop.pr.transverse.timedependent_transport_codec import encode_pr_transverse_timedependent_transport_request as pack, decode_pr_transverse_timedependent_transport_request as unpack
    req=_request();payload=encode(req)
    assert payload['optical_coupling']==PR_TD_PUBLISHED_COUPLING
    assert decode(payload).resolved_optical_coupling==PR_TD_PUBLISHED_COUPLING
    payload.pop('optical_coupling')
    assert decode(payload).resolved_optical_coupling==PR_TD_LEGACY_COUPLING
    wire=pack(req)
    assert unpack(wire.payload.metadata,wire.payload.arrays).resolved_optical_coupling==PR_TD_PUBLISHED_COUPLING
    old=dict(wire.payload.metadata);old.pop('optical_coupling')
    assert unpack(old,wire.payload.arrays).resolved_optical_coupling==PR_TD_LEGACY_COUPLING
    from lcprop.pr.gui.request_adapter import validate_pr_gui_request_representable
    with pytest.raises(ValueError,match='midpoint'):
        validate_pr_gui_request_representable(decode(payload))


@pytest.mark.parametrize('precision',['float32','float64'])
def test_callbacks_preserve_imex_states_and_detach_progress(monkeypatch,precision):
    request=_request(steps=2,precision=precision)
    original=w.imex_euler_step;traces=[]
    def imex(psi,source,**kw):
        state=psi.copy();forcing=source.copy();result=original(psi,source,**kw)
        np.testing.assert_array_equal(psi,state);np.testing.assert_array_equal(source,forcing)
        traces[-1].append((state.tobytes(),forcing.tobytes(),result.tobytes()))
        return result
    monkeypatch.setattr(w,'imex_euler_step',imex)
    traces.append([]);plain=w.run_pr_transverse_timedependent(request)
    traces.append([])
    def callback(progress):
        assert progress.latest_field_state['psi_current_backend']=='numpy'
        progress.latest_field_state['psi_current'][...]=123  # Owned observation only.
    observed=w.run_pr_transverse_timedependent(request,progress_callback=callback)
    assert traces[0]==traces[1] and len(traces[0])==2
    for name in ('psi_final','A_final','source_intensity_stack'):
        np.testing.assert_array_equal(getattr(plain,name),getattr(observed,name))
    assert observed.diagnostics['optical_coupling']==PR_TD_PUBLISHED_COUPLING


@pytest.mark.parametrize('precision',['float32','float64'])
def test_native_callback_movie_is_backend_first(monkeypatch,precision):
    cp=pytest.importorskip('cupy')
    try:cp.zeros(1)
    except Exception as exc:pytest.skip(f'CuPy/CUDA unavailable: {exc}')
    req=_request(steps=1,precision=precision)
    req=replace(req,backend=replace(req.backend,backend='cupy'))
    original=w.downsample_td_movie_frame;transfers=[]
    def movie(intensity,**kw):
        assert isinstance(intensity,cp.ndarray) and kw['xp'] is cp
        before=cp.asnumpy(intensity)  # Explicit reference instrumentation.
        def transfer(preview):
            assert preview.dtype==cp.float32 and preview.nbytes<=65536
            assert preview.shape==(12,10)
            transfers.append(preview.nbytes)
            return cp.asnumpy(preview)
        value=original(intensity,xp=cp,asnumpy=transfer)
        assert value.tobytes()==original(before).tobytes()
        assert cp.asnumpy(intensity).tobytes()==before.tobytes()
        return value
    monkeypatch.setattr(w,'downsample_td_movie_frame',movie)
    plain=w.run_pr_transverse_timedependent(req)
    def progress(p):
        assert isinstance(p.latest_field_state['psi_current'],cp.ndarray)
        p.latest_field_state['psi_current'][...]=123
    observed=w.run_pr_transverse_timedependent(req,progress_callback=progress)
    assert transfers and max(transfers)==480
    np.testing.assert_array_equal(plain.psi_final,observed.psi_final)
    np.testing.assert_array_equal(plain.A_final,observed.A_final)


def test_source_coordinates_and_result_roundtrip():
    from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
    from lcprop.pr.transverse.timedependent_transport_codec import encode_pr_transverse_timedependent_transport_result as encode, decode_pr_transverse_timedependent_transport_result as decode
    result=w.run_pr_transverse_timedependent(_request(steps=1))
    encoded=encode(result)
    restored=decode(encoded.payload.metadata,encoded.payload.arrays)
    assert restored.diagnostics['optical_coupling']==PR_TD_PUBLISHED_COUPLING
    assert restored.diagnostics['source_sample_z_um']==[5.,10.]
    data=pr_transverse_result_to_run_data(restored)
    np.testing.assert_array_equal(data.geometry.z,[5.,10.])
    np.testing.assert_array_equal(restored.A_final,result.A_final)


def test_material_updates_and_other_workflows_are_byte_unchanged():
    from pathlib import Path
    import subprocess
    base = 'c0ce8f6d8f0cccab29ad815d911464fd97b467b9'
    for path in ('src/lcprop/pr/evolution.py', 'src/lcprop/pr/workflow.py',
                 'src/lcprop/pr/transverse/transport.py',
                 'src/lcprop/pr/transverse/static_workflow.py',
                 'src/lcprop/pr/transverse/marching_static.py'):
        assert Path(path).read_bytes() == subprocess.check_output(['git', 'show', base + ':' + path])
    path = 'src/lcprop/pr/transverse/workflow.py'
    old = subprocess.check_output(['git', 'show', base + ':' + path], text=True)
    new = Path(path).read_text()
    start = '                if request.solver.integrator == PR_TRANSVERSE_IMEX_EULER:'
    end = '        td_scalar_history.append(scalar_row)'
    assert old[old.index(start):old.index(end)] == new[new.index(start):new.index(end)]

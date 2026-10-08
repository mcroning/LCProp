"""Presentation-only fixtures: no material or optical execution."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from dataclasses import replace
import hashlib
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from lcprop.products.data_model import FieldData, FieldCollection, Geometry, RunData
from lcprop.gui.workspace import Workspace
from lcprop.gui.field_selection import spatial_registry, with_transverse_aliases
from lcprop.gui.views.display_scale import volume_limits, DisplayScales, scale_key


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def fixture():
    z, x, y = np.array([1., 3., 7.]), np.array([-3., -1., 1., 3.]), np.array([-2., 2.])
    a = 10000*np.arange(3)[:, None, None]+100*np.arange(4)[None, :, None]+np.arange(2)[None, None, :]
    a = a.astype(float)
    a.flags.writeable = False
    volume = FieldData('psi', 'Potential', a, ('z','x','y'), 'field',
                       units={'z':'um','x':'um','y':'um'}, quantity='psi', value_unit='1',
                       coordinates={'z':z,'x':x,'y':y})
    face = replace(volume, key='face', display_name='Potential', quantity='E',
                   coordinates={'z':z+.25,'x':x+.5,'y':y+.75})
    plane = FieldData('far', 'Far field', np.ones((4,2)), ('s_x','s_y'), 'intensity',
                      coordinates={'s_x':x/10,'s_y':y/10})
    return RunData(workflow='pr_transverse_timedependent', geometry=Geometry(x=x,y=y,z=z),
                   fields=FieldCollection([(f.key,f) for f in (volume,face,plane)]))


def select(w, mode, key):
    index = w.spatial_selector.findData([mode,key])
    assert index >= 0
    w.spatial_selector.setCurrentIndex(index)
    w._select_spatial_product()


def test_three_slices_identity_locations_scales_nonmutation(app):
    r=fixture(); original=hashlib.sha256(r.fields['psi'].data.tobytes()).hexdigest()
    w=Workspace();w.set_run_data(r);select(w,'volume','psi')
    p=w.longitudinal_pane;p.set_cut_indices(2,1);p.z_plane_slider.setValue(2)
    views=(w.image_pane.image_view,p.xz_view,p.yz_view)
    a=r.fields['psi'].data
    for v,expected in zip(views,(a[2],a[:,:,1],a[:,2,:])):
        np.testing.assert_array_equal(v._field.data,expected)
        assert np.shares_memory(v._field.data,a)
        np.testing.assert_array_equal(v.image.get_array(),expected.T)
        assert v.image.get_clim()==(a.min(),a.max())
        assert v._field.value_unit=='1'
    w.display_scales.configure(scale_key(r.fields['psi']),'fixed',(-3.,30000.))
    assert all(v.image.get_clim()==(-3.,30000.) for v in views)
    select(w,'volume','face')
    assert p._physical_position==(1.5,2.75,7.25)
    np.testing.assert_array_equal(w.image_pane.image_view._field.coordinates['x'],r.fields['face'].coordinates['x'])
    assert not w.linked_log.isEnabled()
    assert hashlib.sha256(a.tobytes()).hexdigest()==original
    assert len(r.fields)==3
    w.close()


def test_independent_planes_and_unavailable(app):
    r=fixture();w=Workspace();w.set_run_data(r);select(w,'volume','psi');select(w,'plane','far')
    assert w.image_pane.image_view._field.key=='far'
    assert w.longitudinal_pane.xz_view._field is None
    assert 'Independent' in w.longitudinal_pane.no_data_label.text()
    bad=replace(r.fields['psi'],coordinates={'x':np.array([0.])})
    broken=replace(r,fields=FieldCollection([('psi',bad)]))
    assert 'psi' in spatial_registry(broken)[1]
    w.set_run_data(broken)
    assert 'Unavailable' in w.spatial_selector.itemText(0)
    w.close()


def test_preview_physical_position(app):
    r=fixture();full=r.fields['psi']
    preview=replace(full,key='preview',data=full.data[::2,::2,:],
        coordinates={'z':full.coordinates['z'][::2],'x':full.coordinates['x'][::2],
                     'y':full.coordinates['y'],'preview_metadata':{'role':'block mean'}})
    r.fields.add('preview',preview)
    w=Workspace();w.set_run_data(r);select(w,'volume','psi')
    w.longitudinal_pane.set_cut_indices(2,1);w.longitudinal_pane.z_plane_slider.setValue(2)
    select(w,'volume','preview')
    assert w.longitudinal_pane._physical_position==(1.,2.,7.)
    assert w.image_pane.image_view._field.data.shape==(2,2)
    w.close()


class LazyHost:
    def __init__(self):
        self.shape=(3,600,700);self.ndim=3;self.dtype=np.dtype('float64');self.calls=[]
    def __array__(self,*args):
        raise AssertionError('Whole volume conversion forbidden')
    def __getitem__(self,index):
        shape=tuple(len(range(*s.indices(n))) for s,n in zip(index,self.shape))
        self.calls.append(np.prod(shape)*8)
        return np.ones(shape)


def test_bounded_global_scan_and_cache():
    a=LazyHost();f=FieldData('lazy','Lazy',a,('z','x','y'),'field')
    scales=DisplayScales();assert scales.global_limits(f)[0]<1
    assert max(a.calls)<=1024*1024
    count=len(a.calls);scales.global_limits(f);assert len(a.calls)==count
    scales.global_limits(replace(f,content_revision=1));assert len(a.calls)>count
    assert volume_limits(np.array([np.nan,np.inf,-2.,4.]))==(-2.,4.)


def test_log_and_movie_endpoint_restoration(app):
    from tests.test_pr_standard_movies import artifact
    r=fixture();f=replace(r.fields['psi'],key='I',quantity='optical_intensity',kind='intensity')
    r.fields.add('I',f)
    a=replace(artifact(),key='td_trajectory');r.artifacts[a.key]=a
    w=Workspace();w.set_run_data(r);select(w,'volume','I')
    w.linked_log.setChecked(True)
    from matplotlib.colors import LogNorm
    assert isinstance(w.image_pane.image_view.image.norm,LogNorm)
    select(w,'volume','psi');assert not w.linked_log.isChecked()
    w.longitudinal_pane.z_plane_slider.setValue(2)
    w.trajectory_player.show_frame(1)
    assert w.longitudinal_pane.isHidden() and not w.spatial_selector.isEnabled()
    w.trajectory_player.restore_endpoint()
    assert w.spatial_selector.currentData()==['volume','psi']
    np.testing.assert_array_equal(w.image_pane.image_view._field.data,r.fields['psi'].data[2])
    assert w.trajectory_player.save.isEnabled()
    w.close()


def test_plane_then_same_linked_field_rebuilds_all_panels(app):
    w=Workspace();r=fixture();w.set_run_data(r)
    select(w,'volume','psi');select(w,'plane','far');select(w,'volume','psi')
    assert w.longitudinal_pane.xz_view._field is not None
    assert w.longitudinal_pane.yz_view._field is not None
    w.close()


def test_source_alias_is_independent_endpoint_not_arbitrary_z(app):
    r=fixture();v=r.fields['psi']
    endpoint=replace(v,key='endpoint',data=v.data[-1],axes=('x','y'),source_volume_key='psi')
    r.fields.add(endpoint.key,endpoint)
    w=Workspace();w.set_run_data(r);select(w,'volume','psi')
    w.longitudinal_pane.z_plane_slider.setValue(0)
    select(w,'plane','endpoint')
    np.testing.assert_array_equal(w.image_pane.image_view._field.data,v.data[-1])
    assert r.fields['endpoint'].source_volume_key=='psi'
    w.close()


def test_retained_fixed_cuts_never_claim_xy_volume(app):
    r=fixture();v=r.fields['psi'];co={**v.coordinates,'x_cut_um':1.,'y_cut_um':2.}
    xz=replace(v,key='xz',data=v.data[:,:,1],axes=('z','x'),source_volume_key='missing',
               coordinates={**co,'paired_cut_key':'yz'})
    yz=replace(v,key='yz',data=v.data[:,2,:],axes=('z','y'),source_volume_key='missing',
               coordinates={**co,'paired_cut_key':'xz'})
    r.fields.add('xz',xz);r.fields.add('yz',yz)
    w=Workspace();w.set_run_data(r);select(w,'cuts','__paired_cuts__:xz')
    assert w.image_pane.image_view._field is None
    assert 'unavailable' in w.image_pane.selection_message.text()
    np.testing.assert_array_equal(w.longitudinal_pane.xz_view._field.data,xz.data)
    w.close()


@pytest.mark.parametrize('policy',['fast','full'])
def test_transport_reopening_same_registry_and_borrowed_slices(app,policy):
    # The same codec is the local/Slurm presentation boundary; no remote execution.
    from tests.test_pr_transverse_continuation import request
    from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
    from lcprop.pr.transverse.timedependent_transport_codec import (
        encode_pr_transverse_timedependent_transport_result as encode,
        decode_pr_transverse_timedependent_transport_result as decode)
    from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
    result=run_pr_transverse_timedependent(request())
    packet=encode(result,result_policy=policy)
    reopened=decode(packet.payload.metadata,packet.payload.arrays)
    left=pr_transverse_result_to_run_data(reopened)
    right=pr_transverse_result_to_run_data(decode(packet.payload.metadata,packet.payload.arrays))
    assert spatial_registry(left)[0].keys()==spatial_registry(right)[0].keys()
    before={k:hashlib.sha256(f.data.tobytes()).hexdigest() for k,f in left.fields.items()}
    w=Workspace();w.set_run_data(left)
    keys=list(spatial_registry(left)[0]);assert keys
    for key in keys:
        select(w,'volume',key)
        f=left.fields[key]
        np.testing.assert_array_equal(w.image_pane.image_view._field.data,f.data[w.longitudinal_pane._iz])
        np.testing.assert_array_equal(f.data,right.fields[key].data)
    after={k:hashlib.sha256(f.data.tobytes()).hexdigest() for k,f in left.fields.items()}
    assert before==after
    w.close()


def test_linked_log_preserves_manual_scale_and_linear_reset(app):
    r=fixture();f=replace(r.fields['psi'],key='I',kind='intensity',quantity='optical_intensity')
    r.fields.add('I',f);w=Workspace();w.set_run_data(r);select(w,'volume','I')
    w.linked_log.setChecked(True);w.longitudinal_pane.z_plane_slider.setValue(0)
    w.display_scales.configure(scale_key(f),'fixed',(1.,100000.))
    views=(w.image_pane.image_view,w.longitudinal_pane.xz_view,w.longitudinal_pane.yz_view)
    assert all(v.image.get_clim()==(1.,100000.) for v in views)
    w.linked_log.setChecked(False)
    w.display_scales.configure(scale_key(f),'auto')
    assert all(v.image.get_clim()==(0.,20301.) for v in views)
    w.close()

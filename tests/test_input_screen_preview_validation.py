"""Preview ownership is independent of scientific launch qualification."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from tests.test_input_screen_editor import _enabled_panel
from lcprop.gui.image_sources import _rgba_bytes
from lcprop.gui.panels.input_screen_editor import _array_pixmap, RASTER_PREVIEW_SIZE
from lcprop.optics.screens import prepare_intensity_raster_screen
from lcprop.optics.launch import build_launch
from lcprop.pr.portable_launch import encode_launch_elements

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])

def digest(e):
    return hashlib.sha256(json.dumps(encode_launch_elements(e.launch_elements()),sort_keys=True).encode()).hexdigest()

def pixels(label):
    pix=label.pixmap()
    return None if pix is None or pix.isNull() else _rgba_bytes(pix.toImage())

def write_png(path, reverse=False):
    a=np.full((16,20),255,dtype=np.uint8);a[2:13,3:7]=0;a[8:11,9:18]=64
    if reverse:a=255-a
    assert QImage(a.data,20,16,20,QImage.Format.Format_Grayscale8).copy().save(str(path))


def test_mask_updates_despite_invalid_carrier_and_recovers(app,tmp_path):
    p=_enabled_panel(Nx=128,Ny=128,x_aperture=500,y_aperture=500)
    p.set_optical_context(n_ref=2.4,interaction_length_um=1000.)
    e=p.input_screen_editor
    path=tmp_path/'nonuniform.png';write_png(path)
    e.load_user_image(path);e.width_um.setValue(200);e.height_um.setValue(200)
    assert pixels(e.transformed_preview) is not None
    original_beams=e._beams
    valid=original_beams()
    invalid=replace(valid,channels=tuple(replace(b,theta_ext_rad=.15) for b in valid.channels))
    # Hold the exact bound image/placement fixed while varying only the carrier.
    e._beams=lambda:invalid
    before=digest(e);source_hash=e._source.sha256
    e.refresh_preview()
    assert digest(e)==before and e._source.sha256==source_hash
    assert pixels(e.transformed_preview) is None
    assert 'optical preview unavailable' in e.status.text()
    assert 'Nyquist' in e.status.toolTip()
    grid=p._screen_preview_grid()
    with pytest.raises(ValueError,match='Optical launch sampling invalid'):
        build_launch(invalid,grid,complex_dtype=np.complex128,launch_elements=e.launch_elements(),context=p._preview_launch_context(grid))
    previous=pixels(e.transmission_preview)
    for change in (lambda:e.width_um.setValue(120),lambda:e.invert.setChecked(True),
                   lambda:e.center_x_um.setValue(40)):
        change()
        actual=pixels(e.transmission_preview);assert not np.array_equal(actual,previous)
        sampled=prepare_intensity_raster_screen(e._screen_from_controls(),p._screen_preview_grid())
        expected=_rgba_bytes(_array_pixmap(sampled,xy_axes=True,size=RASTER_PREVIEW_SIZE).toImage())
        np.testing.assert_array_equal(actual,expected)
        assert pixels(e.transformed_preview) is None
        identity=digest(e);e.refresh_preview();assert digest(e)==identity
        previous=actual
    second=tmp_path/'replacement.png';write_png(second,True);e.load_user_image(second)
    assert e._source.sha256!=source_hash
    assert not np.array_equal(pixels(e.transmission_preview),previous)
    after=digest(e);e._beams=original_beams;e.refresh_preview()
    assert pixels(e.transformed_preview) is not None
    assert digest(e)==after and e.status.toolTip()==''
    p.close()


def test_invalid_placement_clears_both_previews_then_recovers(app,tmp_path):
    p=_enabled_panel(Nx=32,Ny=32,x_aperture=100,y_aperture=100);e=p.input_screen_editor
    path=tmp_path/'pattern.png';write_png(path);e.load_user_image(path)
    assert pixels(e.transmission_preview) is not None
    e.width_um.setValue(90);e.center_x_um.setValue(40)
    assert pixels(e.transmission_preview) is None
    assert pixels(e.transformed_preview) is None
    assert 'Screen preview unavailable' in e.status.text()
    e.center_x_um.setValue(0)
    assert pixels(e.transmission_preview) is not None and pixels(e.transformed_preview) is not None
    p.close()

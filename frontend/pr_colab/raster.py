"""Headless image input; native PR raster records/screens remain authoritative."""
from dataclasses import replace
import hashlib
from pathlib import Path
import ipywidgets as W
import numpy as np
from lcprop.pr.image_sources import PRImageSource
from lcprop.optics.screens import ChannelLaunchElements,IntensityRasterScreen,ScreenPlacement


def decode(path,*,admission=lambda size:None):
    from PIL import Image,ImageOps
    path=Path(path)
    admission(4*path.stat().st_size)
    with Image.open(path) as im:
        if im.format not in {'PNG','JPEG','TIFF','BMP','PPM','WEBP'}:
            raise ValueError('Choose a supported raster image, not executable/vector content')
        admission(im.width*im.height*24+4*path.stat().st_size)
        fmt=im.format.lower()
        # Same EXIF orientation and Pillow-compatible grayscale convention as
        # desktop QImage conversion. Alpha is not used as intensity.
        gray=np.asarray(ImageOps.exif_transpose(im).convert('RGBA').convert('L')).copy()
    encoded=path.read_bytes()
    return PRImageSource(source_kind='user',display_name=path.name,basename=path.name,
        sha256=hashlib.sha256(encoded).hexdigest(),width=gray.shape[1],height=gray.shape[0],
        encoded_format=fmt,decoded_mode='L',grayscale=gray,encoded_bytes=encoded)


def install(app):
    from lcprop.pr.image_sources import standard_image_catalog
    catalog=standard_image_catalog()
    source_choice=W.Dropdown(description='Image source',options=[('Uploaded file','')]+[(a.display_name,a.asset_id) for a in catalog],layout=W.Layout(width='98%'))
    path=W.Text(description='Image in Files',placeholder='/content/my-image.png',layout=W.Layout(width='98%'))
    width=W.FloatText(description='Width (µm)',value=20.)
    height=W.FloatText(description='Height (µm)',value=20.)
    x=W.FloatText(description='Center x (µm)',value=0.);y=W.FloatText(description='Center y (µm)',value=0.)
    invert=W.Checkbox(description='Invert intensity transmission',value=False)
    add=W.Button(description='Attach screen to active beam');clear=W.Button(description='Remove active beam screens')
    info=W.HTML()
    def change(remove=False):
        s=app.session;s._editable()
        index=next(i for i,c in enumerate(s.draft.beams.channels) if c.name==app.beam_plane.beam.value)
        prior=tuple(v for v in s.draft.launch_elements if v.channel_index!=index)
        if not remove:
            if source_choice.value:
                from importlib import resources
                asset=next(a for a in catalog if a.asset_id==source_choice.value)
                with resources.as_file(resources.files('lcprop.pr.assets.images').joinpath(asset.resource_filename)) as file:
                    source=decode(file,admission=lambda size:s.runtime_policy.check(size,s.memory_available(),label='image decoding'))
                source=replace(source,source_kind='standard',asset_id=asset.asset_id,display_name=asset.display_name)
            else:
                source=decode(path.value,admission=lambda size:s.runtime_policy.check(size,s.memory_available(),label='image decoding'))
            screen=IntensityRasterScreen(source,ScreenPlacement(center_x_um=x.value,center_y_um=y.value,
                width_um=width.value,height_um=height.value),invert=invert.value)
            screen.validate()
            prior=(*prior,ChannelLaunchElements(index,(screen,)))
        request=replace(s.draft,launch_elements=prior)
        app.session.validator(s.route,request)
        s.set_draft(s.route,request,s.generation);s.generation+=1
        app._refresh_preview();app._sync()
        info.value='Screen removed.' if remove else 'Screen attached; native passive transmission, no renormalization or propagation.'
    def guarded(remove):
        try:change(remove)
        except Exception as error:
            from html import escape
            info.value=escape(str(error))
    add.on_click(lambda _:guarded(False));clear.on_click(lambda _:guarded(True))
    for widget in (source_choice,path,width,height,x,y,invert,add,clear):widget.disabled=app.session.busy
    return W.VBox((W.HTML('Upload a raster in the Files sidebar; select the active beam first. '
        'Attaching replaces that beam’s screens only. Other beams are preserved. '
        'Passive intensity transmission uses the existing engine preprocessing and placement.'),
        source_choice,path,width,height,x,y,invert,add,clear,info))

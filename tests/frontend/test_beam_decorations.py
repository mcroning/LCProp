"""Exercise real ipympl artists locally, including the hosted third-beam path."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ipympl.backend_nbagg import Canvas, FigureManager
from pr_colab.pr_workbench import PRWorkbench
from pr_colab.beam_decorations import BEAM_COLORS, update_decorations
from lcprop.notebook._launch_preview import ray_endpoint
from pr_colab.batch_requests import preset, refractive_index


def widget_canvas():
    figure=Figure(figsize=(3.75,3.75),layout='constrained')
    canvas=Canvas(figure)
    FigureManager(canvas,id(figure))
    return figure,canvas


@pytest.mark.parametrize('route',['pr_static','pr','pr_xy_static','pr_xy_td','pr_unified','pr_published'])
def test_add_duplicate_select_edit_delete_with_actual_widget_canvas(route,tmp_path,monkeypatch):
    monkeypatch.setattr('pr_colab.batch_beam_plane.widget_canvas',widget_canvas)
    a=PRWorkbench(files_root=tmp_path,memory_available=lambda:64*1024**3)
    a.session.switch(route);a._sync()
    p=a.beam_plane
    assert isinstance(p.canvas,Canvas) and p.ax is not None
    # Actual button callbacks (rather than headless _add-only tests).
    p.add.click();p.duplicate.click()
    assert len(a.session.draft.beams.channels)==3
    assert not a.notice.value
    assert len(p.markers)==len(p.labels)==len(p.arrows)==len(p.tips)==3
    p.beam.value='beam-3';p.mode.value='Tilt';p.fields['x0_um'].value='7'
    assert a.session.draft.beams.channels[2].x0_um==7
    assert p.arrows[2].arrow_patch.get_linewidth()==2.5
    assert p.arrows[0].arrow_patch.get_linewidth()==1
    p.canvas.draw()  # real renderer, no display server or solver
    # Disable/re-enable and delete preserve selected identity and refresh artists.
    p.enabled.value=False
    assert len(p.markers)==2
    p.enabled.value=True
    assert len(p.markers)==3
    p.delete.click()
    assert len(p.markers)==2 and len(a.session.draft.beams.channels)==2
    p.close()


def test_many_channels_geometry_and_artist_cleanup():
    r=preset('pr');b=r.beams.channels[0]
    r=replace(r,beams=replace(r.beams,channels=tuple(replace(b,name=f'b{i}',x0_um=float(i)) for i in range(17))))
    f=Figure();FigureCanvasAgg(f)
    p=SimpleNamespace(ax=f.subplots(),markers=[],labels=[],arrows=[],tips=[])
    index=refractive_index('pr',r)
    update_decorations(p,r,16,index,labels=[c.name for c in r.beams.channels])
    old=p.markers+p.labels+p.arrows+p.tips
    assert len(p.markers)==17
    assert [m.get_markeredgecolor() for m in p.markers]==[BEAM_COLORS[i%len(BEAM_COLORS)] for i in range(17)]
    for channel,tip in zip(r.beams.channels,p.tips):
        x,y=ray_endpoint(channel,r,index)
        assert list(tip.get_xdata())==[x] and list(tip.get_ydata())==[y]
    update_decorations(p,r,0,index,labels=[c.name for c in r.beams.channels])
    assert all(artist.axes is None for artist in old)
    assert all(not artist.get_in_layout() for artist in p.markers+p.labels+p.arrows+p.tips)
    assert len(p.ax.lines)==34 and len(p.ax.texts)==34
    f.canvas.draw()

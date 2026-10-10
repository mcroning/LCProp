"""Stage 1C: retained-data presentation checks, no scientific execution."""
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'examples'))
import lc_static_interactive as ui


@pytest.fixture
def snapshot():
    # Coordinate-coded synthetic values exercise an off-center, unequal grid.
    request=ui.request_for_controls(x=-18,y=3)
    x=np.array([-30.,-20.,-10.,0.]);y=np.array([-8.,-4.,0.,4.,8.]);z=np.array([2.5,7.5,12.5])
    bias=np.arange(20,dtype=float).reshape(4,5)/10
    delta=np.arange(60,dtype=float).reshape(3,4,5)/100-.3
    a=dict(x_um=x,y_um=y,z_um=z,intensity_stack=np.arange(60,dtype=float).reshape(3,4,5),
           theta=bias[None]+delta,theta_bias=bias,widths_um=np.ones((3,2)))
    b=io.BytesIO();np.savez(b,**a)
    return ui.CompletedRun(json.dumps(ui.encode_lc_static_request(request)),b.getvalue(),'{}')


def test_reorientation_shared_scales_and_labels(snapshot):
    a,m,views=ui.longitudinal_slices(snapshot)
    np.testing.assert_array_equal(views[2],a['theta'][:,:,3]-a['theta_bias'][:,3])
    np.testing.assert_array_equal(views[3],a['theta'][:,1,:]-a['theta_bias'][1,:])
    before=hashlib.sha256(snapshot.arrays_npz).hexdigest()
    fig=ui.plot_completed(snapshot)
    axes=[ax for ax in fig.axes if ax.get_title()]
    assert len(axes)==4 and len(fig.axes)==7
    assert axes[0].collections[0].norm is axes[1].collections[0].norm
    assert axes[2].collections[0].norm is axes[3].collections[0].norm
    assert axes[2].collections[0].norm.vmin==-axes[2].collections[0].norm.vmax
    assert 'y=4.00' in axes[0].get_title() and 'x=-20.00' in axes[1].get_title()
    for ax,v in zip(axes,views):
        np.testing.assert_array_equal(np.asarray(ax.collections[0].get_array()).reshape(v.T.shape),v.T)
        assert '$z$' in ax.get_xlabel()
    assert before==hashlib.sha256(snapshot.arrays_npz).hexdigest()
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_missing_baseline_rejected(snapshot):
    a=snapshot.arrays();del a['theta_bias']
    with pytest.raises(ValueError,match='theta_bias baseline'):ui.slice_selection(snapshot.request,a)


def test_export_matches_display_snapshot_after_edits(snapshot,tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    form=ui.InteractiveLC();form.completed=snapshot
    monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('Unexpected solver execution'))
    form.controls['x'].value=-17
    form.controls['V_bias'].value=1.
    fig=ui.plot_completed(snapshot);png=io.BytesIO();fig.savefig(png,format='png',dpi=130);plt.close(fig)
    ui.export_completed(form.completed,tmp_path/'saved')
    assert (tmp_path/'saved/fields.png').read_bytes()==png.getvalue()
    assert (tmp_path/'saved/arrays.npz').read_bytes()==snapshot.arrays_npz
    meta=json.loads((tmp_path/'saved/visualization.json').read_text())
    assert meta['requested_x_um']==-18 and meta['V_bias']==.9144
    assert meta['sampled_y_um']==4
    form.restore_button.click();assert form.completed is snapshot
    def failed_run(*args, **kwargs):
        raise ValueError("Synthetic execution failure")
    monkeypatch.setattr(ui, "execute", failed_run)
    form.run_button.click();assert form.completed is snapshot


def test_compact_widgets_and_preview(monkeypatch):
    import IPython.display
    captured=[]
    monkeypatch.setattr(IPython.display,'display',captured.append)
    form=ui.InteractiveLC()
    assert form.path.layout.width==form.load_path.layout.width=='300px'
    for b in (form.run_button,form.restore_button,form.save_button,form.load_button):
        assert 140<=int(b.layout.width[:-2])<=170
    images=[x for x in captured if type(x).__name__=='Image']
    assert images and images[0].layout.width=='375px'
    assert images[0].layout.max_width=='100%'
    from PIL import Image
    with Image.open(io.BytesIO(images[0].value)) as im:assert abs(im.width-im.height)<=1


@pytest.mark.parametrize('limit', [120,180,300])
def test_wall_budget_boundary_progress_first(monkeypatch, limit):
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Progress:
        completed_units: int
        total_units: int = 100
        elapsed_wall_time: float = -1
    clock=iter([10.,10.+limit,10.+limit+.25])
    monkeypatch.setattr(ui,'perf_counter',lambda:next(clock))
    monkeypatch.setattr(ui,'launch_preview',lambda request:None)
    seen=[]
    class Runner:
        def run_operation(self, operation, request, progress_callback):
            progress_callback(Progress(3))  # Equality is allowed.
            assert len(seen)==1
            progress_callback(Progress(4))
            pytest.fail('Must stop at first over-budget boundary')
    monkeypatch.setattr(ui,'LocalRunner',Runner)
    with pytest.raises(TimeoutError) as error:
        ui.execute(ui.build_request(),seen.append,time_limit=limit)
    assert [(p.completed_units,p.elapsed_wall_time) for p in seen]==[(3,limit),(4,limit+.25)]
    assert 'Time budget exhausted; convergence not assessed.' in str(error.value)
    assert 'last completed slice: 4/100' in str(error.value)
    assert f'configured limit: {limit} s' in str(error.value)
    assert f'{limit+.25:.3f} s' in str(error.value)


@pytest.mark.parametrize('limit',[301,float('inf'),float('nan'),0,-1])
def test_invalid_budget_rejected_before_execution(monkeypatch,limit):
    monkeypatch.setattr(ui,'launch_preview',lambda r:pytest.fail('Validation must precede work'))
    with pytest.raises(ValueError,match='at most 300'):
        ui.execute(ui.build_request(),time_limit=limit)


def test_budget_selection_timeout_preserves_snapshot(snapshot,monkeypatch,tmp_path):
    from types import SimpleNamespace
    form=ui.InteractiveLC();form.completed=snapshot
    before=ui.encode_lc_static_request(form.request())
    assert form.time_budget.value==120
    form.result_output.outputs=({"output_type":"display_data","data":{"text/plain":"Previous completed plot"},"metadata":{}},)
    original_outputs=form.result_output.outputs
    calls=[]
    def timeout(request,progress,*,time_limit):
        calls.append(time_limit)
        progress(SimpleNamespace(completed_units=42,total_units=100,elapsed_wall_time=180.25))
        raise TimeoutError('Time budget exhausted; convergence not assessed. Elapsed wall time: 180.250 s; last completed slice: 42/100; configured limit: 180 s.')
    monkeypatch.setattr(ui,'execute',timeout)
    form.time_budget.value=180
    assert calls==[] and form.completed is snapshot
    assert ui.encode_lc_static_request(form.request())==before==ui.encode_lc_static_request(ui.build_request())
    form.run_button.click()
    assert calls==[180] and form.progress.value==42
    assert '180.250 s' in form.status.value and '42/100' in form.status.value
    assert form.completed is snapshot and form.result_output.outputs==original_outputs
    assert not form.time_budget.disabled
    ui.export_completed(form.completed,tmp_path/'after-timeout')
    assert (tmp_path/'after-timeout/arrays.npz').read_bytes()==snapshot.arrays_npz


def test_configure_title_tracks_panel_state(monkeypatch):
    # Only widget state is under test; do not sample a launch or run the solver.
    monkeypatch.setattr(ui.InteractiveLC, '_changed', lambda *args, **kwargs: None)
    form = ui.InteractiveLC()
    assert form.configure.selected_index == 0
    assert form.configure.get_title(0) == 'Configure — collapse to focus on results'
    form.configure.selected_index = None
    assert form.configure.get_title(0) == 'Configure — expand to show setup pane'
    form.configure.selected_index = 0
    assert form.configure.get_title(0) == 'Configure — collapse to focus on results'
    form.configure.selected_index = None
    assert form.configure.get_title(0) == 'Configure — expand to show setup pane'

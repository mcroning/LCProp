"""Presentation only: synthetic completed fields, no scientific execution."""
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from lcprop.notebook import interactive_lc as ui
from lcprop.core.beams import LEGACY_COHERENCE_GROUP
from lcprop.persistence import load_experiment


def snapshot(mode='coherent', *, relative=.375, legacy_flag=None):
    request=ui.build_request()
    first=request.beams.channels[0]
    if mode=='single':
        beams=request.beams
    else:
        second=replace(first,name='second',phase_rad=relative,
                       coherence_group='A' if mode=='coherent' else 'B')
        channels=(first,second)
        if mode=='mixed':
            channels=(first,replace(second,coherence_group='A'),replace(second,name='third',coherence_group='B'))
        if mode=='legacy':
            channels=tuple(replace(b,coherence_group=LEGACY_COHERENCE_GROUP) for b in channels)
        beams=replace(request.beams,channels=channels,coherence=legacy_flag or 'incoherent')
    request=replace(request,beams=beams)
    bias=np.arange(12,dtype=float).reshape(3,4)/100
    intensity=(1+np.arange(24,dtype=float)).reshape(2,3,4)/1000
    arrays=dict(x_um=np.array([-20.,0.,20.]),y_um=np.array([-4.,0.,4.,8.]),z_um=np.array([2.5,7.5]),
        intensity_stack=intensity,theta=bias[None]+intensity,theta_bias=bias,widths_um=np.array([[2.,3.],[4.,5.]]))
    buffer=io.BytesIO();np.savez(buffer,**arrays)
    return ui.CompletedRun(json.dumps(ui.encode_lc_static_request(request)),buffer.getvalue(),'{}')


@pytest.fixture
def form(monkeypatch):
    monkeypatch.setattr(ui.InteractiveLC,'_changed',lambda *a,**k:None)
    monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('No scientific execution'))
    return ui.InteractiveLC()


def test_selector_label_order_and_full_selected_option(form):
    configure=form.configure.children[0]
    assert configure.children.index(form.coherence_panel)<configure.children.index(form.control_grid)
    assert form.coherence_label.value=='<b>Beam coherence</b>'
    assert form.coherence.value=='preserve'
    assert tuple(form.coherence.options)==(('Preserve saved groups','preserve'),
        ('Coherent — same group','coherent'),('Incoherent — distinct groups','incoherent'))
    assert form.coherence.layout.width=='420px' and form.coherence.layout.max_width=='100%'
    assert form.coherence_selection.layout.width=='100%'
    for label,value in form.coherence.options:
        form.coherence.value=value
        assert form.coherence_selection.value=='Selected: '+label


def test_relative_phase_label_and_accessibility(form):
    phase=form.beam2_controls['phase_rad']
    row=next(row for row in form.beam2_form.children if phase in row.children)
    assert row.children[0].value=='Relative phase (rad)'
    assert 'Beam 2 phase minus Beam 1 phase' in phase.tooltip
    assert 'entrance-plane intersection' in phase.tooltip
    form.second_enabled.value=True
    form.coherence.value='coherent'
    assert phase in form._draft_controls() and not phase.disabled
    form.beam2_form.layout.display='flex'
    phase.value=.375
    assert form.request().beams.channels[1].phase_rad==.375


@pytest.mark.parametrize('mode,expected', [('coherent','Coherent'),('incoherent','Incoherent between groups'),
                                         ('single','Single beam'),('mixed','Mixed coherence groups')])
def test_effective_labels_from_saved_groups(mode,expected):
    snap=snapshot(mode)
    before=(snap.request_json,snap.arrays_npz,snap.summary_json)
    short,details=ui.completed_beam_labels(snap)
    assert short.startswith(expected) and details.startswith(expected)
    if mode=='single':
        assert 'Relative phase not applicable' in details and 'Δφ' not in short
    else:
        assert '0.375 rad' in short and '0.375 rad' in details
    assert 'Explicit saved groups:' in details
    if mode=='incoherent':
        assert 'no intensity effect between their distinct groups' in details
        assert 'no inter-group effect' in short
        assert 'Beam 1: A, Beam 2: B' in details
    assert before==(snap.request_json,snap.arrays_npz,snap.summary_json)


def test_legacy_and_contradictory_stack_flag():
    explicit=snapshot('coherent',legacy_flag='incoherent')
    short,details=ui.completed_beam_labels(explicit)
    assert short.startswith('Coherent')
    assert 'Legacy stack flag: incoherent; explicit groups take precedence' in details
    legacy=snapshot('legacy',legacy_flag='coherent')
    assert ui.completed_beam_labels(legacy)[0].startswith('Coherent')
    assert 'Legacy stack:' in ui.completed_beam_labels(legacy)[1]
    # The request does not retain whether a user chose Preserve or explicitly selected a mode.
    assert 'selected Preserve' not in details


def test_completed_summary_survives_edits_failure_and_export(form,monkeypatch,tmp_path):
    import IPython.display
    import matplotlib.pyplot as plt
    monkeypatch.setattr(IPython.display,'display',lambda *a,**k:None)
    snap=snapshot('coherent')
    form._set_request_controls(snap.request)
    monkeypatch.setattr(ui,'execute',lambda *a,**k:snap)
    form.run_button.click()
    summary=form.completed_summary.value
    title=ui.plot_completed(snap)._suptitle.get_text()
    form.second_enabled.value=True
    form.coherence.value='incoherent'
    form.beam2_controls['phase_rad'].value=2.
    form.controls['V_bias'].value=1.
    assert form.completed_summary.value==summary and form.completed is snap
    assert 'Coherent' in summary and '0.375 rad' in summary
    def failure(*a,**k):raise TimeoutError('Synthetic budget failure')
    monkeypatch.setattr(ui,'execute',failure)
    form.run_button.click()
    assert form.completed_summary.value==summary and form.completed is snap
    ui.export_completed(form.completed,tmp_path/'export')
    assert (tmp_path/'export/arrays.npz').read_bytes()==snap.arrays_npz
    assert load_experiment(tmp_path/'export/experiment.lcprop.json').request==snap.request
    assert ui.plot_completed(form.completed)._suptitle.get_text()==title
    form.restore_button.click()
    assert form.completed_summary.value==summary
    plt.close('all')


def test_plot_arrays_scales_coordinates_and_widths_unchanged():
    import ast
    import matplotlib.pyplot as plt
    # Load just the old plotting function; no workflow or other baseline code executes.
    old=subprocess.check_output(['git','show','681f669b5ddb41d6bfd829856933d0ad0e23eb83:src/lcprop/notebook/interactive_lc.py'],text=True)
    tree=ast.parse(old)
    func=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='plot_completed')
    namespace=vars(ui).copy()
    exec(compile(ast.Module(body=[func],type_ignores=[]),'accepted-plot-function','exec'),namespace)
    snap=snapshot('incoherent')
    baseline=namespace['plot_completed'](snap)
    current=ui.plot_completed(snap)
    assert len(baseline.axes)==len(current.axes)==7
    for old_ax,new_ax in zip(baseline.axes,current.axes):
        assert old_ax.get_xlabel()==new_ax.get_xlabel()
        assert old_ax.get_ylabel()==new_ax.get_ylabel()
        assert old_ax.get_xlim()==new_ax.get_xlim() and old_ax.get_ylim()==new_ax.get_ylim()
        assert old_ax.get_title()==new_ax.get_title()
        for old_plot,new_plot in zip(old_ax.collections,new_ax.collections):
            np.testing.assert_array_equal(old_plot.get_array(),new_plot.get_array())
            assert old_plot.get_clim()==new_plot.get_clim()
        for old_line,new_line in zip(old_ax.lines,new_ax.lines):
            np.testing.assert_array_equal(old_line.get_xdata(),new_line.get_xdata())
            np.testing.assert_array_equal(old_line.get_ydata(),new_line.get_ydata())
    assert current.axes[-1].get_ylabel()=='Combined RMS width (µm)'
    assert len(current._suptitle.get_text().splitlines())==2
    assert 'no inter-group effect' in current._suptitle.get_text()
    plt.close(baseline);plt.close(current)


def test_html_summary_escapes_saved_names(form,monkeypatch):
    snap=snapshot()
    request=snap.request
    request=replace(request,beams=replace(request.beams,channels=tuple(
        replace(b,coherence_group='<laser>') for b in request.beams.channels)))
    snap=replace(snap,request_json=json.dumps(ui.encode_lc_static_request(request)))
    form._set_request_controls(snap.request)
    monkeypatch.setattr(ui,'execute',lambda *a,**k:snap)
    monkeypatch.setattr(ui,'display_figure',lambda *a,**k:None)
    form.run_button.click()
    assert '&lt;laser&gt;' in form.completed_summary.value
    assert '<laser>' not in form.completed_summary.value


def test_phase_stays_visible_on_actual_coherence_change(monkeypatch):
    from types import SimpleNamespace
    grid=SimpleNamespace(x_um=np.array([-1.,0.,1.]),y_um=np.array([-1.,0.,1.]))
    monkeypatch.setattr(ui,'launch_preview',lambda r:(grid,SimpleNamespace(coherence_groups=r.beams.coherence_groups),np.ones((3,3))))
    monkeypatch.setattr(ui,'qualify_launch_sampling',lambda *a,**k:SimpleNamespace(warnings=[]))
    monkeypatch.setattr(ui,'display_figure',lambda *a,**k:None)
    monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('No scientific execution'))
    app=ui.InteractiveLC()
    assert app.beam2_form.layout.display=='none'
    app.second_enabled.value=True
    app.coherence.value='coherent'
    assert app.beam2_form.layout.display=='flex'
    assert not app.beam2_controls['phase_rad'].disabled
    app.beam2_controls['phase_rad'].value=.5
    assert app.request().beams.channels[1].phase_rad==.5
    app.coherence.value='incoherent'
    assert app.beam2_form.layout.display=='flex'
    assert not app.beam2_controls['phase_rad'].disabled
    assert 'Selected: Incoherent — distinct groups' in app.coherence_selection.value

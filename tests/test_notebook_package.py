"""Installed-package integration; one bounded canonical comparison pair."""
import json
import subprocess
import sys
import numpy as np
import pytest


def test_core_and_notebook_import_isolation():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'PySide6','launchplane','cupy','matplotlib','ipywidgets','IPython'}:
            raise AssertionError('Optional import attempted: '+fullname)
sys.meta_path.insert(0,Block())
import lcprop
import lcprop.notebook
from lcprop.notebook import interactive_lc, lc_static_demo
assert interactive_lc.build_request()==lc_static_demo.build_request()
'''
    subprocess.run([sys.executable,'-B','-c',script],check=True)


def test_request_and_launch_parity():
    from lcprop.notebook import interactive_lc as ui, lc_static_demo as demo
    from lcprop.core.grid import make_grid
    from lcprop.optics.launch import build_launch, OpticalLaunchContext
    r=demo.build_request();assert ui.request_for_controls()==r
    assert r.solver.static_residual_rms_tol==.005 and r.solver.static_residual_max_tol==.02
    assert r.beams.channels[0].power_mW==1. and r.bias.V_bias==.9144
    g=make_grid(r.grid,real_dtype=np.float64)
    canonical=build_launch(r.beams,g,complex_dtype=np.complex128,
        context=OpticalLaunchContext(grid=g,n_ref=r.material.no,interaction_length_um=r.grid.z_length_um))
    _,launch,I=ui.launch_preview(r)
    np.testing.assert_array_equal(launch.A0,canonical.A0)
    np.testing.assert_array_equal(I,ui.total_intensity(canonical.A0,coherence_groups=canonical.coherence_groups))


def test_widget_initialization_and_label(monkeypatch):
    from lcprop.notebook import interactive_lc as ui
    monkeypatch.setattr(ui,'execute',lambda *a,**k:pytest.fail('Initialization must not run solver'))
    app=ui.InteractiveLC()
    assert app.completed is None and app.time_budget.value==120
    assert app.configure.get_title(0)=='Configure — collapse to focus on results'
    app.configure.selected_index=None
    assert app.configure.get_title(0)=='Configure — expand to show setup pane'


def test_exact_canonical_scientific_parity_and_snapshot(tmp_path):
    from lcprop.notebook import interactive_lc as ui
    from lcprop.lc.workflows.static import run_static
    from lcprop.runners.local import LocalRunner
    from lcprop.lc.operations import LC_STATIC_OPERATION
    request=ui.build_request()
    snapshot=ui.execute(request)
    canonical=run_static(request)
    arrays=snapshot.arrays()
    for key,attr in [('A_initial','A_initial'),('A_final','A_final'),('theta','theta_final'),
                     ('theta_bias','theta_bias'),('intensity_stack','intensity_stack')]:
        np.testing.assert_array_equal(arrays[key],getattr(canonical,attr))
    assert canonical.all_slices_converged and canonical.completed_slices==100
    from dataclasses import asdict
    assert snapshot.summary['slice_summaries']==[asdict(x) for x in canonical.slice_summaries]
    assert snapshot.summary['iteration_history']==[asdict(x) for x in canonical.iteration_records]
    before=snapshot.arrays_npz
    arrays['theta'][:]=0
    assert snapshot.arrays_npz==before
    ui.export_completed(snapshot,tmp_path/'export')
    assert (tmp_path/'export/arrays.npz').read_bytes()==before
    from lcprop.persistence import load_experiment
    assert load_experiment(tmp_path/'export/experiment.lcprop.json').request==request

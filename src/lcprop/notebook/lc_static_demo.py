"""Existing self-consistent LC regression fixture; CPU only, no soliton claim.

Uses installed LCProp. Output is a new diagnostic directory. Fixed comparison:
same launch/material/bias/grid, director frozen at the Product dark-bias solution.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
import lcprop
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.grid import GridSpec, make_grid
from lcprop.persistence import save_experiment
from lcprop.lc.experiment_codec import encode_lc_static_request
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.requests import OutputOptions, StaticRunRequest, StaticSolverOptions, StaticWorkflowOptions
from lcprop.lc.specs import LCMaterial, BiasSpec
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.products.diagnostics import rms_widths
from lcprop.runners.local import LocalRunner


def build_request():
    """Exact single-beam test_static_local_zmarch fixture and solver options."""
    return StaticRunRequest(
        grid=GridSpec(Nx=128, Ny=128, dz_um=5., x_aperture_um=75.,
                      y_aperture_um=100., z_length_um=500.),
        material=LCMaterial(), bias=BiasSpec(),
        beams=BeamStack(channels=(BeamChannel(name='off-axis', power_mW=1.,
            x0_um=-20., y0_um=0., coherence_group='A', w1_um=10., w2_um=10.),)),
        output=OutputOptions(save_slices=True),
        solver=StaticSolverOptions(workflow=StaticWorkflowOptions(
            strategy='local_self_consistent', theta_solver='picard_cn',
            optics_solver='splitstep', coupling='self_consistent'),
            max_iterations=3, static_max_coupled_passes=3))


def fixed_request(request):
    return replace(request, solver=replace(request.solver, workflow=StaticWorkflowOptions()))


def execute(request, *, time_limit=120.):
    """Preflight and public runner, with an operational stop between z slices."""
    encode_lc_static_request(request)
    request.optical_boundary.validate()
    sampling = qualify_launch_sampling(request.beams, request.grid, request.material.no,
                                       boundary=request.optical_boundary)
    sampling.require_valid()
    started = perf_counter()
    def progress(p):
        if perf_counter()-started > time_limit:
            raise TimeoutError('Local CPU budget exceeded; stop without changing the request')
    completed = LocalRunner().run_operation(LC_STATIC_OPERATION, request, progress_callback=progress)
    return completed, perf_counter()-started, list(sampling.warnings)


def widths(completed):
    result = completed.result
    grid = make_grid(result.request.grid, real_dtype=np.float64)
    return np.asarray([rms_widths(plane, grid) for plane in result.intensity_stack])


def diagnostics(completed, elapsed, warnings):
    r = completed.result
    return dict(status=r.status, method=r.method, completed_slices=r.completed_slices,
        total_slices=r.total_slices, all_slices_converged=r.all_slices_converged,
        residual_rms_max=r.max_final_residual_rms, residual_max=r.max_final_residual_max,
        elapsed_seconds_including_products=elapsed,
        normalized_power_initial=r.power_initial, normalized_power_final=r.power_final,
        physical_power_initial_mW=r.physical_power_initial_mW,
        physical_power_final_mW=r.physical_power_final_mW,
        warnings=warnings+list(r.warnings),
        slice_summaries=[asdict(x) for x in r.slice_summaries],
        iteration_history=[asdict(x) for x in r.iteration_records])


def plot_comparison(nonlinear, fixed):
    import matplotlib.pyplot as plt
    r, data = nonlinear.result, nonlinear.run_data
    x,y,z = data.geometry.x,data.geometry.y,data.geometry.z
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,values,title,label in [
        (axes[0,0],data.fields['final_intensity'].data,'Nonlinear output, z=500 µm',r'$1/\mu\mathrm{m}^{2}$'),
        (axes[0,1],r.theta_final[-1],'Last accepted director slice',r'$\theta$ (rad)'),
        (axes[1,0],r.theta_final[-1]-r.theta_bias,'Reorientation relative to dark bias',r'$\Delta\theta$ (rad)')]:
        artist=ax.pcolormesh(x,y,values.T,shading='nearest')
        ax.set(xlabel=r'$x$ ($\mu$m)',ylabel=r'$y$ ($\mu$m)',title=title,aspect='equal')
        fig.colorbar(artist,ax=ax,label=label)
    for run,label,style in [(nonlinear,'self-consistent','-'),(fixed,'fixed dark bias','--')]:
        w=widths(run)
        for i,axis in enumerate(('x','y')):
            axes[1,1].plot(z,w[:,i],style,label=f'{label}: {axis}')
    axes[1,1].set(xlabel=r'$z$ ($\mu$m)',ylabel='RMS width (µm)',title='Slice-average optical intensity')
    axes[1,1].legend(fontsize='small')
    return fig


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='New diagnostic directory')
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    request=build_request()
    nonlinear,elapsed,warnings=execute(request)
    r=nonlinear.result
    summary={'nonlinear':diagnostics(nonlinear,elapsed,warnings),
        'environment':dict(python=platform.python_version(),platform=platform.platform(),
            numpy=np.__version__,scipy=scipy.__version__,lcprop=importlib.metadata.version('lcprop'),
            module=str(Path(lcprop.__file__).resolve())),
        'claim':'Self-consistent slice convergence is not stationary soliton formation.'}
    save_experiment(request,args.output/'nonlinear.lcprop.json',material_id='lc',workflow_id='static')
    # Retain failure/convergence diagnostics before deciding whether to continue.
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if r.status!='completed' or not r.all_slices_converged:
        raise RuntimeError('Original convergence gates not met; no fixed comparison or parameter tuning')
    fixed,t_fixed,w_fixed=execute(fixed_request(request))
    np.testing.assert_array_equal(r.A_initial,fixed.result.A_initial)
    np.testing.assert_array_equal(r.theta_bias,fixed.result.theta_final)
    summary['fixed']=diagnostics(fixed,t_fixed,w_fixed)
    summary['max_reorientation_rad']=float(np.max(np.abs(r.theta_final-r.theta_bias[None])))
    summary['last_slice_rms_width_um']={'nonlinear':widths(nonlinear)[-1].tolist(),'fixed':widths(fixed)[-1].tolist()}
    save_experiment(fixed.result.request,args.output/'fixed.lcprop.json',material_id='lc',workflow_id='static')
    for label,run in [('nonlinear',nonlinear),('fixed',fixed)]:
        s=run.result
        np.savez(args.output/f'{label}.npz',A_initial=s.A_initial,A_final=s.A_final,
            theta=s.theta_final,theta_bias=s.theta_bias,intensity_stack=s.intensity_stack,
            widths_um=widths(run),x_um=run.run_data.geometry.x,y_um=run.run_data.geometry.y,z_um=run.run_data.geometry.z)
    fig=plot_comparison(nonlinear,fixed);fig.savefig(args.output/'comparison.png',dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.iterdir()) if p.is_file()},indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('nonlinear','fixed')},indent=2))
    print(f'Nonlinear: {elapsed:.3f}s, fixed: {t_fixed:.3f}s; all slices converged')


if __name__=='__main__':
    main()

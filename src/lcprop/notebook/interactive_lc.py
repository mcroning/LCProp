"""Example-level ipywidgets presentation; LCProp remains the scientific authority."""
from dataclasses import dataclass, replace
from html import escape
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import lcprop
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.grid import make_grid
from lcprop.persistence import load_experiment, save_experiment
from lcprop.lc.experiment_codec import encode_lc_static_request, decode_lc_static_request
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.splitstep import total_intensity
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.runners.local import LocalRunner
from .lc_static_demo import build_request, widths, diagnostics

_KEEP_SECOND = object()

def request_for_controls(x=-20., y=0., w1=10., w2=10., power=1., V_bias=0.9144, *,
                         base=None, beam1_options=None, second_beam=_KEEP_SECOND, coherence=None):
    request = build_request() if base is None else base
    if second_beam is _KEEP_SECOND:
        second_beam = request.beams.channels[1] if len(request.beams.channels) == 2 else None
    if not 1 <= len(request.beams.channels) <= 2:
        raise ValueError('This introductory interface supports one or two beams only.')
    beam = replace(request.beams.channels[0], x0_um=x, y0_um=y,
                   w1_um=w1, w2_um=w2, power_mW=power)
    if beam1_options:
        beam = replace(beam, **beam1_options)
    channels = (beam,) if second_beam is None else (beam, second_beam)
    stack = BeamStack(channels=channels, coherence=request.beams.coherence)
    if coherence is not None:
        if coherence not in ('coherent', 'incoherent'):
            raise ValueError('Choose coherent, incoherent, or preserve saved groups.')
        # Explicit groups, not the legacy stack flag, determine interference.
        groups = stack.coherence_groups
        if coherence == 'coherent':
            groups = (groups[0],) * len(channels)
        elif len(groups) == 2 and groups[0] == groups[1]:
            groups = (groups[0], groups[0] + ':beam-2')
        stack = BeamStack(channels=tuple(replace(b, coherence_group=g)
                          for b, g in zip(channels, groups)), coherence=coherence)
    request = replace(request, beams=stack,
                      bias=replace(request.bias, V_bias=float(V_bias)))
    if not np.isfinite(request.bias.V_bias):
        raise ValueError("Applied voltage must be finite.")
    encode_lc_static_request(request)
    return request


def require_bounded_request(request):
    """Restrict this teaching UI, not Product support, to the qualified preset shape."""
    preset = build_request()
    if not 1 <= len(request.beams.channels) <= 2:
        raise ValueError('This introductory interface supports one or two beams only.')
    if not np.isfinite(request.bias.V_bias):
        raise ValueError("Applied voltage must be finite.")
    if replace(request, beams=preset.beams, bias=replace(request.bias, V_bias=preset.bias.V_bias)) != preset:
        raise ValueError('This interface requires the qualified grid/material/solver preset; use desktop for other configurations.')
    encode_lc_static_request(request)


def launch_preview(request):
    require_bounded_request(request)
    grid = make_grid(request.grid, real_dtype=np.float64)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128,
        context=OpticalLaunchContext(grid=grid, n_ref=float(request.material.no),
                                    interaction_length_um=float(request.grid.z_length_um)))
    intensity = total_intensity(launch.A0, coherence_groups=launch.coherence_groups)
    return grid, launch, intensity


def environment():
    import scipy
    import matplotlib
    return dict(python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                lcprop=metadata.version('lcprop'), module=str(Path(lcprop.__file__).resolve()),
                direct_url=metadata.distribution('lcprop').read_text('direct_url.json'),
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


@dataclass(frozen=True)
class CompletedRun:
    """Immutable serialized snapshot: public access returns detached decoded values."""
    request_json: str
    arrays_npz: bytes
    summary_json: str

    @property
    def request(self):
        return decode_lc_static_request(json.loads(self.request_json))

    def arrays(self):
        with np.load(io.BytesIO(self.arrays_npz), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}

    @property
    def summary(self):
        return json.loads(self.summary_json)


def execute(request, progress=None, *, time_limit=120.):
    if not np.isfinite(time_limit) or not 0 < time_limit <= 300:
        raise ValueError('Wall-time budget must be positive and at most 300 seconds.')
    require_bounded_request(request)
    # Validate the actual sampled launch, not a hand-normalized GUI approximation.
    launch_preview(request)
    started = perf_counter()
    def report(p):
        elapsed = perf_counter()-started
        if progress is not None:
            progress(replace(p, elapsed_wall_time=elapsed))
        if elapsed > time_limit:
            raise TimeoutError(
                'Time budget exhausted; convergence not assessed. '
                f'Elapsed wall time: {elapsed:.3f} s; '
                f'last completed slice: {p.completed_units}/{p.total_units}; '
                f'configured limit: {time_limit:g} s.')
    completed = LocalRunner().run_operation(LC_STATIC_OPERATION, request, progress_callback=report)
    result = completed.result
    if result.status != 'completed' or not result.all_slices_converged:
        raise RuntimeError('Calculation did not converge completely; previous completed result retained.')
    arrays = dict(A_initial=result.A_initial, A_final=result.A_final,
        theta=result.theta_final, theta_bias=result.theta_bias,
        intensity_stack=result.intensity_stack, widths_um=widths(completed),
        x_um=completed.run_data.geometry.x, y_um=completed.run_data.geometry.y,
        z_um=completed.run_data.geometry.z)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    sampling = qualify_launch_sampling(request.beams, request.grid, request.material.no,
                                       boundary=request.optical_boundary)
    summary = diagnostics(completed, perf_counter()-started, list(sampling.warnings))
    summary['visualization'] = slice_selection(request, arrays)
    summary['environment'] = environment()
    summary['claim'] = 'Self-consistent slice convergence is not stationary soliton formation.'
    return CompletedRun(json.dumps(encode_lc_static_request(request), sort_keys=True),
                        buffer.getvalue(), json.dumps(summary, allow_nan=False))


def slice_selection(request, arrays):
    """Metadata from the completed request and retained coordinates, never controls."""
    x, y, z = (arrays[k] for k in ('x_um', 'y_um', 'z_um'))
    for name, coordinate in zip(('x','y','z'), (x,y,z)):
        if coordinate.ndim != 1 or not coordinate.size or not np.isfinite(coordinate).all():
            raise ValueError(f'Missing/invalid retained {name} coordinates.')
    shape = (len(z), len(x), len(y))
    for name in ('intensity_stack', 'theta'):
        if name not in arrays or arrays[name].shape != shape:
            raise ValueError(f'Required retained {name} volume unavailable; no reconstruction permitted.')
    bias = arrays.get('theta_bias')
    if bias is None or bias.shape != (len(x),len(y)) or not np.isfinite(bias).all():
        raise ValueError('Required corresponding retained theta_bias baseline unavailable.')
    beam = request.beams.channels[0]
    ix = int(np.argmin(np.abs(x-beam.x0_um)))
    iy = int(np.argmin(np.abs(y-beam.y0_um)))
    return dict(V_bias=request.bias.V_bias, requested_x_um=beam.x0_um, requested_y_um=beam.y0_um,
                slice_beam_index=0, slice_beam_name=beam.name,
                width_quantity='combined optical intensity RMS width (not per-beam)',
                x_index=ix, y_index=iy, sampled_x_um=float(x[ix]), sampled_y_um=float(y[iy]),
                volume_axes=['z','x','y'], z_location='retained slice midpoints',
                optical_quantity='retained slice-average optical intensity', director_quantity='theta minus corresponding retained theta_bias',
                director_baseline='completed run theta_bias; no recomputation')


def longitudinal_slices(snapshot):
    if snapshot is None:
        raise ValueError('Run successfully before plotting or exporting.')
    a = snapshot.arrays()
    selection = slice_selection(snapshot.request, a)
    ix, iy = selection['x_index'], selection['y_index']
    return a, selection, (a['intensity_stack'][:,:,iy], a['intensity_stack'][:,ix,:],
                           a['theta'][:,:,iy]-a['theta_bias'][:,iy][None,:],
                           a['theta'][:,ix,:]-a['theta_bias'][ix,:][None,:])


def plot_completed(snapshot):
    import matplotlib.pyplot as plt
    a, selection, slices = longitudinal_slices(snapshot)
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    fig = plt.figure(figsize=(9, 6.2), layout='constrained')
    gs = fig.add_gridspec(3, 2, height_ratios=(1,1,.65))
    titles = [f"Intensity XZ · y={selection['sampled_y_um']:.2f} µm",
              f"Intensity YZ · x={selection['sampled_x_um']:.2f} µm",
              f"Reorientation XZ · y={selection['sampled_y_um']:.2f} µm",
              f"Reorientation YZ · x={selection['sampled_x_um']:.2f} µm"]
    # One normalization per pair; reduce only the displayed retained slices.
    intensity_norm = Normalize(0., max(float(np.max(v)) for v in slices[:2]) or 1.)
    limit = max(float(np.max(np.abs(v))) for v in slices[2:]) or 1.
    director_norm = Normalize(-limit, limit)
    for row in range(2):
        axes = []
        for col in range(2):
            i = row*2+col
            ax = fig.add_subplot(gs[row,col]); axes.append(ax)
            axis = 'x' if col == 0 else 'y'
            artist = ax.pcolormesh(a['z_um'], a[axis+'_um'], slices[i].T, shading='nearest',
                                  norm=intensity_norm if row==0 else director_norm,
                                  cmap='viridis' if row==0 else 'RdBu_r')
            ax.set(xlabel=r'$z$ (µm)', ylabel=axis+' (µm)', title=titles[i])
            ax.title.set_fontsize(10)
            ax.tick_params(labelsize=9)
        bar = fig.colorbar(artist, ax=axes, pad=.015, fraction=.035,
                           format=FormatStrFormatter('%.3g'))
        bar.set_label(r'$1/\mu\mathrm{m}^{2}$' if row==0 else r'$\Delta\theta$ (rad)')
        bar.locator=MaxNLocator(nbins=5); bar.update_ticks()
        bar.ax.tick_params(labelsize=9)
    ax = fig.add_subplot(gs[2,:])
    ax.plot(a['z_um'], a['widths_um'][:,0], label='x')
    ax.plot(a['z_um'], a['widths_um'][:,1], label='y')
    ax.set(xlabel=r'$z$ (µm)', ylabel='Combined RMS width (µm)')
    ax.legend(ncol=2, fontsize=9, loc='upper right')
    ax.tick_params(labelsize=9)
    fig.suptitle(f"Completed: {selection['V_bias']:.3g} V; "
                 f"slices through Beam 1 input ({selection['requested_x_um']:.2f}, {selection['requested_y_um']:.2f}) µm",fontsize=11)
    return fig


def display_figure(fig, *, width_px=900):
    """Cap notebook display size; shrink to the container without upscaling."""
    import ipywidgets as W
    from IPython.display import display
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=110)
    image = W.Image(value=buffer.getvalue(), format='png',
                    layout=W.Layout(width=f'{width_px}px', max_width='100%', height='auto', object_fit='contain'))
    display(image)
    return image


def export_completed(snapshot, directory):
    if snapshot is None:
        raise ValueError('Run successfully before exporting.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_experiment(snapshot.request, directory/'experiment.lcprop.json', material_id='lc', workflow_id='static')
    (directory/'arrays.npz').write_bytes(snapshot.arrays_npz)
    (directory/'summary.json').write_text(snapshot.summary_json+'\n')
    (directory/'visualization.json').write_text(json.dumps(slice_selection(snapshot.request, snapshot.arrays()), indent=2)+'\n')
    fig = plot_completed(snapshot)
    fig.savefig(directory/'fields.png', dpi=130)
    import matplotlib.pyplot as plt
    plt.close(fig)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file()}
    (directory/'sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    return hashes


class InteractiveLC:
    """Small synchronous teaching form; no solver runs from widget change events."""
    def __init__(self):
        import ipywidgets as W
        self.completed = None
        self.base = build_request()
        self.busy = False
        self.loading = False
        def full(): return W.Layout(width='100%', max_width='100%', min_width='0')
        specs = [('x',-20.,'Input x (µm)'),('y',0.,'Input y (µm)'),
                 ('w1',10.,'Principal radius 1 (µm)'),('w2',10.,'Principal radius 2 (µm)'),
                 ('power',1.,'Incident power (mW)'),('V_bias',self.base.bias.V_bias,'Applied voltage (V)')]
        self.controls = {name: W.FloatText(value=value, layout=full()) for name,value,label in specs}
        boxes = [W.VBox([W.HTML(value=escape(label)),self.controls[name]],
                        layout=W.Layout(flex='0 1 220px',min_width='0',max_width='100%'))
                 for name,value,label in specs]
        self.control_grid = W.Box(boxes, layout=W.Layout(display='flex',flex_flow='row wrap',width='100%'))
        # Radians preserve native request values on load; no degree round-trip.
        self._beam_specs = [
            ('theta_ext_rad', 'External incidence (rad)'),
            ('phi_rad', 'Azimuth (rad)'), ('psi_rad', 'Profile roll (rad)'),
            ('wavelength_um', 'Wavelength (µm)'), ('n_ext', 'External index'),
            ('phase_rad', 'Phase at beam axis (rad)')]
        def beam_form(beam, specs):
            controls = {key: W.FloatText(value=getattr(beam,key),layout=full()) for key,label in specs}
            rows = [W.VBox([W.HTML(escape(label)),controls[key]],
                          layout=W.Layout(flex='0 1 220px',min_width='0',max_width='100%'))
                    for key,label in specs]
            return controls, W.Box(rows,layout=W.Layout(display='flex',flex_flow='row wrap',width='100%'))
        self.beam1_controls, first_geometry = beam_form(self.base.beams.channels[0], self._beam_specs)
        second_specs = [('x0_um','Input x (µm)'),('y0_um','Input y (µm)'),
                        ('w1_um','Principal radius 1 (µm)'),('w2_um','Principal radius 2 (µm)'),
                        ('power_mW','Incident power (mW)')] + self._beam_specs[:-1] + [
                        ('phase_rad','Phase relative to Beam 1 (rad)')]
        self.beam2_controls, self.beam2_form = beam_form(self._second_template(), second_specs)
        self.second_enabled = W.Checkbox(value=False,description='Include Beam 2',layout=full())
        self.coherence = W.Dropdown(options=[('Preserve saved groups', 'preserve'),
            ('Coherent — same group','coherent'),('Incoherent — distinct groups','incoherent')],
            value='preserve',layout=W.Layout(width='300px',max_width='100%'))
        self.beam_help = W.HTML('Radii: beam-normal 1/e field (1/e² intensity). '
            'LC requires the same wavelength for both beams. Phase is referenced at each beam axis. '
            'Beam 2 starts at x=20 µm, 1 mW; adding it does not redistribute Beam 1 power. '
            'Multibeam and modified geometry are exploratory; slices use Beam 1 coordinates.',layout=full())
        def compact(width): return W.Layout(width=f'{width}px', max_width='100%', min_width='0')
        def wrap(children): return W.Box(children, layout=W.Layout(display='flex', flex_flow='row wrap', align_items='center', width='100%'))
        self.run_button = W.Button(description='Run', button_style='primary',layout=compact(140))
        self.time_budget = W.Dropdown(options=[('120 seconds',120),('180 seconds',180),('300 seconds',300)],
                                      value=120, layout=compact(170))
        self.restore_button = W.Button(description='Restore preset', tooltip='Restore qualified preset', layout=compact(170))
        self.preview_output = W.Output(layout=full())
        self.result_output = W.Output(layout=full())
        self.status = W.HTML(value='Qualified preset. Editing updates launch preview only.', layout=full())
        self.advisories = W.HTML(layout=full())
        self.progress = W.IntProgress(min=0,max=100,description='Slices',layout=compact(240))
        self.path = W.Text(value='lc-interactive-result',layout=compact(300))
        self.save_button = W.Button(description='Export result',layout=compact(160))
        self.load_path = W.Text(value='',layout=compact(300))
        self.load_button = W.Button(description='Load experiment',layout=compact(160))
        configure = W.VBox([W.HTML('<h3>Configure — Beam 1 and bias</h3>'),self.control_grid,
            first_geometry,self.second_enabled,self.beam2_form,self.coherence,self.beam_help,self.restore_button,
            self.advisories,self.preview_output,W.HTML('Experiment path'),wrap([self.load_path,self.load_button])],layout=full())
        run = W.VBox([W.HTML('<h3>Run</h3>'),W.HTML('Elapsed wall-time budget (not a solver iteration limit)'),wrap([self.time_budget,self.run_button,self.progress,
            W.Box([self.status],layout=compact(420))])],layout=full())
        results = W.VBox([W.HTML('<h3>Results — completed snapshot</h3>'),self.result_output,
            W.HTML('New export folder'),wrap([self.path,self.save_button])],layout=full())
        self.configure = W.Accordion(children=[configure], selected_index=0,layout=full())
        def update_configure_title(change=None):
            title = ('Configure — collapse to focus on results'
                     if self.configure.selected_index == 0
                     else 'Configure — expand to show setup pane')
            self.configure.set_title(0, title)
        self.configure.observe(update_configure_title, names='selected_index')
        update_configure_title()
        self.widget = W.VBox([self.configure,run,results],layout=full())
        self.restore_button.on_click(self._restore)
        self.run_button.on_click(self._run)
        self.save_button.on_click(self._save)
        self.load_button.on_click(self._load)
        for control in self._draft_controls(): control.observe(self._changed, names='value')
        self._changed()

    def _second_template(self):
        if len(self.base.beams.channels) == 2:
            return self.base.beams.channels[1]
        first = self.base.beams.channels[0]
        # Retain legacy grouping consistently if a legacy experiment was loaded.
        from lcprop.core.beams import LEGACY_COHERENCE_GROUP
        group = (LEGACY_COHERENCE_GROUP if first.coherence_group == LEGACY_COHERENCE_GROUP
                 else first.coherence_group + ':beam-2')
        return BeamChannel(name='beam-2', x0_um=20., w1_um=10., w2_um=10.,
                           wavelength_um=first.wavelength_um, coherence_group=group)

    def _draft_controls(self):
        return (*self.controls.values(), *self.beam1_controls.values(),
                *self.beam2_controls.values(), self.second_enabled, self.coherence)

    def request(self):
        options = {k:v.value for k,v in self.beam1_controls.items()}
        second = None
        if self.second_enabled.value:
            template = self._second_template()
            values = {k:v.value for k,v in self.beam2_controls.items()}
            relative = values['phase_rad']
            first_phase = options['phase_rad']
            # Preserve saved phase bytes when controls have not changed: subtraction
            # followed by addition need not round-trip floating point exactly.
            original_phase = self.base.beams.channels[0].phase_rad
            values['phase_rad'] = (template.phase_rad if first_phase == original_phase and
                relative == template.phase_rad-original_phase else first_phase+relative)
            second = replace(template, **values)
        return request_for_controls(base=self.base, beam1_options=options, second_beam=second,
            coherence=None if self.coherence.value == 'preserve' else self.coherence.value,
            **{k:v.value for k,v in self.controls.items()})

    def _changed(self, change=None):
        if self.loading:
            return
        self.beam2_form.layout.display = 'flex' if self.second_enabled.value else 'none'
        from IPython.display import display
        import matplotlib.pyplot as plt
        self.preview_output.clear_output(wait=False)
        self.advisories.value = ""
        try:
            request = self.request()
            grid, launch, intensity = launch_preview(request)
            sampling = qualify_launch_sampling(request.beams, request.grid, request.material.no,
                                               boundary=request.optical_boundary)
            self.advisories.value = '<br>'.join('Launch advisory: '+escape(w) for w in sampling.warnings)
            self.advisories.value += '<br>Effective coherence groups: ' + escape(
                ', '.join(f'Beam {i+1}: {g}' for i,g in enumerate(launch.coherence_groups)))
            with self.preview_output:
                fig, ax = plt.subplots(figsize=(3.75,3.75),layout='constrained')
                artist = ax.pcolormesh(grid.x_um,grid.y_um,intensity.T,shading='nearest')
                ax.set(aspect='equal',xlabel='x (µm)',ylabel='y (µm)',title='Draft launch\nNormalized optical intensity')
                fig.colorbar(artist,ax=ax,label=r'$1/\mu\mathrm{m}^{2}$')
                display_figure(fig, width_px=375); plt.close(fig)
            self.status.value = ('Qualified preset.' if request == build_request() else 'Exploratory configuration.') + ' Draft launch validated; press Run. Existing completed plots/export unchanged.'
        except (ValueError,TypeError) as error:
            self.status.value = f'Invalid draft: {escape(str(error))}. Completed result unchanged.'

    def _run(self, button=None):
        if self.busy: return
        from IPython.display import display
        import matplotlib.pyplot as plt
        self.busy = True
        for control in (*self._draft_controls(), self.run_button, self.load_button, self.restore_button, self.time_budget): control.disabled = True
        self.progress.value = 0
        self.status.value = 'Running; prior completed result remains owned separately.'
        def progress(p):
            self.progress.value = p.completed_units
            self.status.value = f'{p.completed_units}/{p.total_units} slices; elapsed {p.elapsed_wall_time:.1f} s'
        try:
            snapshot = execute(self.request(), progress, time_limit=self.time_budget.value)
            # Render successfully before replacing previous completed ownership.
            fig = plot_completed(snapshot)
            self.completed = snapshot
            self.configure.selected_index = None
            self.result_output.clear_output(wait=True)
            with self.result_output: display_figure(fig)
            plt.close(fig)
            self.status.value = 'Completed: all 100 slices converged. Plots/export belong to this run, not later edits.'
        except Exception as error:
            self.status.value = f'Run rejected/failed: {escape(str(error))}. Previous completed result retained.'
        finally:
            self.busy = False
            for control in (*self._draft_controls(), self.run_button, self.load_button, self.restore_button, self.time_budget): control.disabled = False

    def _save(self, button=None):
        try:
            export_completed(self.completed,self.path.value)
            self.status.value = 'Exported completed snapshot; later control edits are not included.'
        except Exception as error:
            self.status.value = f'Export rejected: {escape(str(error))}'

    def load(self,path):
        # Public loader validates any desktop presentation payload; never strip it.
        loaded = load_experiment(path)
        if loaded.material_id != 'lc' or loaded.workflow_id != 'static':
            raise ValueError('Only LC static experiments are supported here.')
        request = loaded.request
        require_bounded_request(request)
        launch_preview(request)
        self._set_request_controls(request)
        self._changed()

    def _set_request_controls(self, request):
        beam = request.beams.channels[0]
        self.loading = True
        try:
            self.base = request
            for name,value in dict(x=beam.x0_um,y=beam.y0_um,w1=beam.w1_um,w2=beam.w2_um,power=beam.power_mW,V_bias=request.bias.V_bias).items():
                self.controls[name].value=value
            for name,control in self.beam1_controls.items():
                control.value = getattr(beam,name)
            second = self._second_template()
            for name,control in self.beam2_controls.items():
                control.value = (second.phase_rad-beam.phase_rad if name == 'phase_rad'
                                 else getattr(second,name))
            self.second_enabled.value = len(request.beams.channels) == 2
            self.coherence.value = 'preserve'
        finally:
            self.loading=False

    def _restore(self,button=None):
        if self.busy: return
        self._set_request_controls(build_request())
        self._changed()

    def _load(self,button=None):
        try: self.load(self.load_path.value)
        except Exception as error:
            self.status.value = f'Load rejected: {escape(str(error))}. Use canonical LC experiment without Qt editor metadata; file unchanged.'

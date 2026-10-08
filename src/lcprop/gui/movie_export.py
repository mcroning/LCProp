"""Optional movie export, independent of scientific Python dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from lcprop.gui.scientific_labels import scientific_text, unit_label, axis_label


def movie_display(samples, *, logarithmic=False):
    """One transform and one range for the complete retained time series."""
    values = np.asarray(samples)
    if not np.isfinite(values).all():
        raise ValueError('Movie contains nonfinite values')
    if logarithmic:
        peak = float(values.max())
        # Display-only floor, shared by every frame, eight decades below peak.
        # Nonpositive source-inversion roundoff is displayed at that floor;
        # retained samples are unchanged. Signed material display stays linear.
        floor = max(peak * 1e-8, np.finfo(float).tiny)
        values = np.log10(np.maximum(values.astype(np.float64), floor))
    low, high = float(values.min()), float(values.max())
    if not logarithmic and low >= 0:
        low = 0.0
    if high <= low:
        high = low + max(abs(low) * 1e-6, 1.0)
    return values, (low, high)


def save_movie(artifact, destination, *, fps=6.0, logarithmic=False, ffmpeg_path=None):
    """Save retained data only; external ffmpeg may live outside the conda env.

    LCPROP_FFMPEG may name an external executable. No import or installation of
    an ffmpeg Python dependency is required. Playback never requires ffmpeg.
    """
    executable = ffmpeg_path or os.environ.get('LCPROP_FFMPEG') or shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('Save Movie requires an ffmpeg executable on PATH or LCPROP_FFMPEG. '
                           'It may be installed outside lcprop-new-user. Interactive playback remains available.')
    if not np.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError('Invalid movie frame rate')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    values, limits = movie_display(artifact.data, logarithmic=logarithmic)
    metadata = artifact.metadata
    axes = tuple(metadata.get('axes', ('x', 'y')))
    coords = [np.asarray(metadata['coordinates'][a]) for a in axes]
    def extent(c):
        step = float(c[1] - c[0]) if len(c) > 1 else 1.0
        return float(c[0] - step/2), float(c[-1] + step/2)
    figure = Figure(figsize=(6.4, 4.8), dpi=100)
    canvas = FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    image = ax.imshow(values[0].T, origin='lower', aspect='auto',
                      extent=(*extent(coords[0]), *extent(coords[1])),
                      vmin=limits[0], vmax=limits[1])
    ax.set_xlabel(axis_label(axes[0], {axes[0]: '1' if axes[0].startswith('s_') else 'um'}))
    ax.set_ylabel(axis_label(axes[1], {axes[1]: '1' if axes[1].startswith('s_') else 'um'}))
    figure.colorbar(image, ax=ax, label=unit_label(('log10 ' if logarithmic else '') + metadata['value_unit']))
    title = artifact.display_name.replace("Trajectory", "Time evolution").replace("trajectory", "time evolution")
    frames = []
    for index, frame in enumerate(values):
        image.set_data(frame.T)
        ax.set_title(scientific_text(f'{title}\naccepted τ={metadata["times"][index]:.8g}; frame-uniform playback'))
        figure.tight_layout()
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy().tobytes())
    width, height = canvas.get_width_height()
    command = [str(executable), '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s:v', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0', '-an',
               '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-movflags',
               'frag_keyframe+empty_moov', '-f', 'mp4', 'pipe:1']
    try:
        result = subprocess.run(command, input=b''.join(frames), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'Movie export failed ({type(exc).__name__}). Playback and retained data remain available.') from exc
    if not result.stdout:
        raise RuntimeError('Movie encoder returned an empty file')
    Path(destination).write_bytes(result.stdout)

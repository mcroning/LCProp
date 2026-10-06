# PR accepted-time movies and unified Static Slurm support

Both reduced TD and full-transverse TD retain bounded movie products when a
progress observer is attached (including normal GUI and remote-worker runs).
The Results movie selector provides optical xy, optical xz/yz, output far-field,
and the existing sampled material-plane movie. Older archives expose only the
products actually retained; Full retrieval is not needed to play or save Fast
movies. An unobserved headless run does not fabricate missing time samples.

Each quantity stores at most 36 accepted-time frames, each at most 128×128
float32 values. The five channels together retain at most 11,796,480 raw bytes,
independent of the number of accepted steps or the longitudinal grid size.
Compression/base64 and bounded metadata add package overhead. The result-size
planner includes a conservative movie allowance for Fast and Full TD results.
These are visualization products, not full-resolution trajectory checkpoints.

Longitudinal movies are cuts nearest x=0 or y=0 of the existing accepted-state
source map, inverted with the segment's retained illumination reference. The
published coupling uses right-endpoint source coordinates z=h,...,L; historical
couplings retain their source-coordinate convention. Blocks are averaged on the
backend before host transfer, with block-center coordinates. No optical replay
or material solve is added for movies.

The far-field movie calls `optics.farfield.direction_cosine_spectrum` on the
actual accepted optical endpoint. It transforms the complete endpoint before
averaging the resulting intensity; it never transforms a decimated endpoint.
Coherence, continuous-transform normalization, and direction-cosine s_x/s_y axes
are shared with the ordinary Product far field. This is field-norm density,
not physical power in mW. Each ordinary frame transfer is at most 65,536 bytes;
each reduced axis is at most 128 values. No longitudinal volume is transferred
for movie construction.

Playback uses one global linear range across the retained time samples. The
log10 intensity view uses one display-only floor eight decades below the global
peak (including nonpositive source-inversion roundoff at that display floor);
it does not change retained values. Signed material movies remain linear.
Manual limits, if used, remain fixed across time. Frames are spaced uniformly
in playback, with actual accepted characteristic times labelled explicitly;
playback seconds are not physical seconds. Return to endpoint restores the
ordinary Results presentation.

`Save Movie…` exports the selected retained movie to MP4, with coordinate labels,
accepted times and a global color scale. The export uses the selected linear/log
mode and playback frame rate, with its global range computed from retained data.
It does not fetch Full results or execute science. Interactive playback needs no
ffmpeg. Export can use an executable on PATH or the absolute path specified by
`LCPROP_FFMPEG`; that executable may be installed outside `lcprop-new-user`.
Missing/unusable ffmpeg gives a bounded GUI message while playback stays usable.
No optional Python package is installed automatically.

Unified Static is already registered with the Slurm operation/transport
composition. Native qualification job 4902575 (pax009) passed for the connected
scalable core, with certification SHA-256
`fc6a29ee1e1f6dc00f20ab388abbba1a4e0d5c70f321518a451ba752ad53b3bf`.
The reported `outside scalable commissioned envelope` rejection is a
backend/precision/grid check, not a blanket Slurm restriction. Ordinary CuPy
float64 supports at most 512 per axis / 262,144 active nodes, and CuPy state32
at most 256 per axis / 65,536 nodes. The explicit 384×32 exception is unbiased
only. Direct/reference still has its 12,288-active-node limit. Slurm selection
does not enlarge these envelopes or waive resource/retrieval planning. The
rejection now reports the requested combination and the applicable limits.
No solver, precision, scientific tolerance or envelope was changed here.

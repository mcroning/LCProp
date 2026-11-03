# Photorefractive GUI Workflows

The PR GUI exposes the canonical nonlinear two-dimensional transverse
zero-flux static workflow as **Static (2D transverse zero-flux)**. Requests
use the registered `pr_transverse_static` operation through the shared local
runner; GUI widgets do not invoke material solvers directly.

The authoritative material state is the periodic, zero-mean potential
`psi(x,y,z)`. The workflow reconstructs the normalized carrier state and both
transverse space-charge components, `E_x` and `E_y`. The current scalar
electro-optic projection remains `E_active = E_x`; solving transverse
transport does not add `E_y` directly to optical propagation.

Results expose `psi`, carrier `P`, `E_x`, `E_y`, the active field, transport
intensity, authoritative zero-flux residuals, replay evidence, convergence
status, and available Newton/PCG summaries. A nonconverged material solve is
reported explicitly and is not presented as successful.

The historical reduced `pr_static` workflow remains available as **Static
(legacy x-only)** for compatibility and existing experiment files. It is not
the canonical 2D static selection. Persistence for `pr_transverse_static` is
not introduced by this integration.

## Scientific result views

Canonical 2D zero-flux static results provide input/output near-field
intensity, center-line input/output profiles, midpoint optical-intensity
`x-z` and `y-z` sections, and compact `E_x(x,z)` and `E_y(y,z)` sections. The
selected transverse plane can inspect the canonical potential `psi`, carrier
`P`, and both material-field components. `E_y`, `psi`, and `P` remain
diagnostics; scalar optical propagation still uses `E_active = E_x`.
Recorded optical-source and material-state slices are interval-midpoint,
piecewise-constant values labeled at `z = (slice_index + 0.5) * dz`.

The output angular spectrum is available on in-medium direction-cosine axes
`s_x = wavelength * f_x / n_ref` and `s_y = wavelength * f_y / n_ref`. Its
normalization makes the integral over `ds_x ds_y` equal the normalized optical
power. The workspace includes linear intensity, intensity in dB relative to
its peak (with a -120 dB display floor), and an explicitly diagnostic
carrier-masked view. Carrier masking changes only the stored presentation
product; the unmasked spectrum and propagated optical field are preserved.
Each channel mask is centered at `q/k_medium` and uses half-width
`max(two spectral bins, wavelength/(2*n_ref*waist))` on each axis.

No additional full longitudinal material history is recorded for these
views. Material planes and sections are read-only views of state already
retained by the canonical result. Only the two optical cross-sections and
three far-field planes add significant presentation storage. For 80
longitudinal samples and float64 data, their upper-bound payload is about
25.3 MiB on a 1024 x 1024 grid and 98.5 MiB on a 2048 x 2048 grid; float32
payloads are approximately half those values. This estimate excludes the
canonical solver volumes that already exist independently of presentation.

# Self-consistent LC static CPU example

Install the same pinned core LCProp wheel used by `lc_static_cpu.py`, or in a new
environment install the core package at engine commit
`52b00928978a9ec7862a357d38327fd0b9ae857c`, plus NumPy/SciPy/Matplotlib. No Qt,
LaunchPlane or CuPy is needed. Download this standalone script separately; the new
example is not part of the pinned engine commit.

```sh
python lc_static_nonlinear_cpu.py --output new-nonlinear-lc-run
```

The script uses Agg and saves `comparison.png` for viewing; it does not launch a GUI.
Output must be a new directory. No installation or external connection occurs in the
script. Its completed-slice callback stops a calculation exceeding 120 seconds; an
external process timeout is advisable if a platform must enforce a hard budget.

## Recovered scientific configuration

The exact single-beam request is from
`tests/test_static_local_zmarch.py::test_static_local_self_consistent_marches_theta_and_field_along_z`:
128×128, aperture 75×100 µm, length 500 µm, dz=5 µm, 1 mW beam centered at x=-20 µm,
y=0, principal radii 10 µm, wavelength 0.633 µm, normal incidence, coherence group A.
Material/bias defaults: ne=1.7, no=1.5, K=7e-12 N, dielectric anisotropy 13,
bias=0.9144 V, anchored theta_bc=0, periodic optical boundary. Float64 NumPy and
existing optical substep policy are unchanged.

Self-consistent local z marching uses Picard/CN director relaxation and split-step
optics. Original residual RMS 0.005 and maximum 0.02 gates, 200 relaxation iterations
per coupled pass and 3 coupled passes are retained. The historical max_iterations=3
field is preserved too; it is not the per-slice relaxation cap. A preflight advisory
about the entrance envelope approaching the periodic boundary is printed in diagnostics.

The fixed comparison changes **only workflow selection** to fixed_theta/none/frozen.
It reconstructs the same Product dark-bias director (initial_theta=None), not the
converged illuminated director. Launch arrays and bias identity are checked exactly.
All other physical, numerical and optical-boundary settings remain the same.

## Outputs and interpretation

`summary.json` records environment, timing, status, all 100 slice convergence records
and complete director iteration history for the nonlinear calculation. Two Product
experiment JSON files and two selected NPZ diagnostic exports preserve the requests,
fields, coordinates and RMS widths. `sha256.json` binds the exported files. These
NPZs are diagnostics, not full transport/checkpoint archives.

The figure shows output intensity, last-slice director, reorientation relative to
bias, and x/y RMS widths versus z for both workflows. Widths use the public Product
helper on retained **slice-average** intensities at Product midpoint coordinates.
Optical intensity is normalized 1/µm²; director is radians; physical-power diagnostics
in mW remain distinct from the dimensionless optical norm.

Local validation: nonlinear 14.96 s, fixed 0.18 s; all slices converged under original
gates, maximum reorientation 0.0800 rad. Convergence means the recorded per-slice
residual contract was satisfied, not infinite-accuracy global equilibrium.
The beam undergoes transient focusing and broadening; its final widths are larger
than the fixed-bias baseline. This demonstrates nonlinear reorientation/propagation,
**not monotonic beam narrowing, stationary soliton formation or stability**.
No tuning, sweep or tolerance relaxation is performed. A nonconverged run stops
before the comparison and retains available convergence diagnostics.

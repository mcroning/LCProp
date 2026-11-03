# PR Fast Longitudinal Intensity Cuts

## Scope and motivation

PR Fast retrieval previously retained optical endpoint fields and far-field
products but omitted every longitudinal volume. The GUI therefore disabled its
longitudinal pane and directed users to rerun with Full retrieval. This bounded
change retains two optical-intensity cuts while continuing to omit all 3-D
optical and material volumes. It applies to reduced and full-transverse static
and time-dependent PR results, including nonlinear and linearized transverse
material responses.

No propagation, source cadence, material equation, solver, carrier-power, Image
Amplification, run-cost, Slurm, or soliton behavior changed.

## Retained contract

For the canonical centered coordinates,

\[
i_{x0}=\mathop{\rm argmin}_i |x_i|,\qquad
i_{y0}=\mathop{\rm argmin}_j |y_j|,
\]

Fast result payloads optionally retain:

- `longitudinal_intensity_xz`, shape `(Nz, Nx)`, sampled at
  `y_cut_um = y[iy0]`;
- `longitudinal_intensity_yz`, shape `(Nz, Ny)`, sampled at
  `x_cut_um = x[ix0]`;
- the scalar `x_cut_um` and `y_cut_um` coordinates.

The first array axis is propagation `z`; it is not material time. For a TD
result, the cuts describe the final accepted optical propagation state already
used for the endpoint products. Optical intensity follows the existing Full
transverse-static presentation convention: subtract the uniform material
background from the normalized PR-driving source and multiply by the launch
peak-intensity reference.

Odd grids retain the exact zero sample. An even grid has an equal-distance
pair around zero; canonical `argmin` selects the lower index, so, for example,
spacing 0.78125 µm records -0.390625 µm rather than falsely reporting zero.

## Transport and presentation

The additions are optional fields in the existing result schemas. New Fast
payloads checksum and transport the arrays and coordinates, and their retention
summary labels them as nearest-zero Fast cuts. Old Fast payloads without these
fields still decode and retain the prior unavailable-volume message.

The existing longitudinal GUI pane recognizes the paired `(z, x)` and `(z, y)`
fields. It labels both views with their actual retained coordinate and hides the
transverse slice sliders and selection guides because no alternate slice is
available. Full results continue through the existing 3-D selectable-slice
path without presentation changes.

## Equivalence and size

Tests compare the Fast arrays directly with cuts sampled from the same Full
optical source and require exact equality through transport. Coverage includes
reduced static and TD, full-transverse static, and full-transverse nonlinear and
linearized TD cases, plus odd/even coordinate selection.

For a representative float64 `256 × 256 × 256` source volume, the two retained
cuts add

\[
256(256+256)\times 8 = 1{,}048{,}576\ \text{bytes} = 1\ \text{MiB},
\]

whereas one full scalar volume is 128 MiB. The cut pair is therefore 1/128 of
one omitted full volume. Float32 requires half that storage. The actual portable
package also retains the pre-existing optical endpoints and metadata; a package
regression bounds the new array archive by those endpoints, the two cuts, and
small serialization overhead.

## Local validation

Focused retention, transport, product, transverse linearized/nonlinear, and GUI
tests passed locally. The complete PR suite, syntax compilation, and whitespace
checks are recorded in the Development handoff after final validation.

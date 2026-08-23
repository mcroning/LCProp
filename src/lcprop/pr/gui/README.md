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

# PR material products in the common Results viewer

These are observational products. They do not change launch normalization,
accepted states, solver tolerances, propagation order or temporal integration.
No material-time history is added.

## Authoritative fields

| Workflow | Accepted state | Observational fields | Longitudinal coordinates |
| --- | --- | --- | --- |
| Unified Static | q, ψ, harmonic b at each accepted material plane | ψ; normalized carrier from the existing precision-specific carrier operation; x/y face fields; projected optical-node material response E_x | Optical boundaries include z=0; material samples use accepted right endpoints. Face axes retain half-cell offsets. |
| Reduced TD | E_final, (Nz,Nx,Ny), x-only transport with independent y columns | Material E_x only; no invented E_y or potential | Existing material-plane ledger; published optical-first samples are at right endpoints. |
| Full-transverse TD | psi_final, (Nz,Nx,Ny) | ψ, E_x, E_y and carrier reconstructed by the existing spectral state_from_potential operation on original-resolution planes | Existing material-plane ledger; published optical-first samples are at right endpoints. |

Full-transverse reconstruction is differentiation/reconstruction, not an equilibrium
solve or a new time step. Derivatives are never taken on downsampled potential.
TD scientific states are already retained in completed results; Fast transfers
bounded final-state presentation products rather than those full material volumes.

`unified_optical_field_volume` is the projected **material response E_x**, not
complex optical amplitude. Canonical product IDs are unchanged. Potential and
carrier are distinct from material fields and complex optical fields. Reduced
models are not offered y-field products.

## Retrieval policies

| Policy | Unified Static | TD |
| --- | --- | --- |
| Minimal (`minimal`) | Existing endpoint/far field and fixed longitudinal cuts; no 3-D preview | Not introduced for TD |
| Fast / Exploratory (`fast`) | Minimal outputs plus bounded optical intensity, ψ, carrier, projected E_x, x-face E_x and (genuine 2D only) y-face E_y | Existing Fast optical/temporal products plus bounded final material E_x (reduced), or ψ/E_x/E_y/carrier (transverse) |
| Interactive / selected Analysis | Existing selection mechanism and selected volume resolution, unchanged | Existing availability unchanged |
| Full | Existing complete/default/explicit selection contract, unchanged | Existing full final material state and products, unchanged |

Old `fast` unified archives remain readable and are labelled as historical
fixed-cut products when supplemental previews are absent. They are not fabricated
or silently upgraded. The next-request selector never rewrites displayed provenance.

## Bounds and ownership

Each new preview is at most 96×96 transversely and at most 4 MiB of numerical
payload. Longitudinal planes are sampled, not averaged. Unified collection reserves
one slot for the latest accepted plane, including on partial runs. Optical and
material coordinate ledgers remain distinct. Per-field source shape, retained
shape, dtype, byte count, block bounds, actual z indices and coordinates accompany
the data. Coordinates do not claim full resolution.

Transverse reduction uses the existing contiguous-block arithmetic-mean policy,
with float64 accumulation and the source field's output dtype. This preserves block
means, not pointwise extrema. Carrier64 stays float64. GPU reduction reuses the
established backend movie-reduction arithmetic; only bounded results and axes
cross the host boundary. Scientific input buffers are never mutated. No
longitudinal full-resolution volume is accumulated for unified Fast previews.

The maximum numerical preview payload is 24 MiB across six genuine-2D Static
fields, 20 MiB across five reduced Static fields, 16 MiB across four transverse TD
material fields, or 4 MiB for reduced TD material E_x. Coordinates/metadata are
additional. These are limits, not measured native peaks. Existing endpoint,
far-field and optical/temporal products are additional and retain their contracts.

## Persistence and display

Unified previews are an optional, validated **presentation envelope** around the
unchanged M5 scientific archive. Normal Product result transport/save/reopen
preserves them. Standalone core scientific `.pru` archives retain their established
scope; supplemental viewer data belongs to the execution-result envelope.
TD previews use the existing portable diagnostic extension container. Old archives
without that extension remain valid. No scientific schema or canonical q/ψ/b state
meaning is migrated.

Each preview supplies a normal common-viewer volume and linked xy selector with
explicit coordinates. The existing xz/yz linkage, guides and color controls apply.
Results status includes the result-owned retrieval policy. Remote presentation
acknowledges preparation before submission and follows supplied Submitting,
Pending, Running and Retrieving statuses; it does not change runner scheduling.

CPU product, codec, metadata and common-viewer regression coverage qualifies this
change. New native preview checks remain conditional on local CuPy/CUDA; no new
H200 or installed-GUI commissioning is implied.

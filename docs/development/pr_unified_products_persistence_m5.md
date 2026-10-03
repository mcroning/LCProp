# Unified Static products, persistence and resource planning (M5)

M5 is a direct headless/testing layer. There is no GUI, normal workflow registry,
legacy codec modification, checkpoint, resume, continuation or TD integration.

## Entry points and identities

- `products.run_unified_products(request, selection=UnifiedSelection(...))`
- `codec.encode_request/decode_request`, `save_request/load_request`
- `codec.encode_result/decode_result`, `save_result/load_result`
- `resources.estimate_resources(request, selection=...)`

All modules live under `lcprop.pr.unified`. Identities are:

| Record | Identity |
|---|---|
| Request | `pr_unified_static_request_v1` |
| Result | `pr_unified_static_result_v1` |
| Products | `pr_unified_static_products_v1` |
| Canonical Static state | `pr_unified_static_canonical_state_v1` |
| Potential gauge | `zero_mean_potential_per_active_domain_v1` |
| Resource plan | `pr_unified_static_resource_plan_v1` |

Workflow/arithmetic/material/spatial/closure/precision/projection identities
remain exactly M1–M4. Projection is
`adjacent_face_arithmetic_to_optical_node_v1`. Float32 records explicitly retain
`state32_linear64_bernoulli64_v1`, never an ambiguous all-float32 claim.

The request records physical grid/L/dz, normalized active/batch geometry,
named electrical experiment, complete material/gain/dark/uniform parameters,
wavelength, coherence groups, backend intent, canonical scattering, exact
prepared launch array and selected products. The normalization identity is
`pr_channel_peak_reference_v1`; the accepted scalar is retained in result
execution provenance. Old `pr_static` or other identities cannot be decoded as
this workflow, and there is no legacy runtime-state conversion.

## Observation and scientific isolation

M4's hop/source/material/projection/phase/scattering code is unchanged. The
only workflow extension is an internal product transaction after complete
candidate science and before acceptance (plus launch observation at z=0).
The collector receives only the needed detached optical and/or q/psi/b/I
snapshots and detached transverse coordinate arrays, with copied metadata.
Minimal selection uses no observer; optical-only cuts copy no material planes.
It receives no optical kernel/frequency
buffers. Mutation of those snapshots fails closed before acceptance. Only the
internal collector type is accepted; this is not a public arbitrary callback.

The collector constructs a new prospective series mapping without modifying
previous accepted products. M4 promotes it together with optical/material state
and ledger after bookkeeping and the cancellation check. A rejected/cancelled
cell contributes neither optical boundary nor material product. Returning
products never requires replay or another material solve. Derived J uses the
unchanged fitted-flux operator and exact arriving total intensity from that cell.
The snapshots remain O(Nx Ny), not longitudinal volumes.

## Selection, coordinates and locations

The accepted optical endpoint and last accepted canonical q/psi/b are mandatory
scientific records, including partial runs. This is a reproducible scientific
package, not a promise of a tiny display-only Fast payload. Optional products:

- launch, final/partial raw optical boundary intensity;
- raw complex optical cuts and raw optical intensity cuts at boundaries;
- last material q, n, psi, b or dark-inclusive arriving transport intensity;
- native oriented face E_x/J_x, and E_y/J_y only for genuine 2D;
- reconstructed node E_x, delta-n, full material phase;
- reduced cuts of the scalar/transverse material quantities above;
- exact far-field intensity/axes from the actual accepted endpoint.

Harmonic b has domain/batch layout, so a fictitious x-z/y-z cut is rejected.
No reduced y field/current is fabricated. Full longitudinal scientific volumes,
preview schemas and optional complex far-field products are not implemented.

Longitudinal arrays are explicit binary64 `boundary_z_um=(0,h,...,mh)` and
`material_z_um=(h,...,mh)`. They remain distinct and are checked against the
accepted ledger; no implicit common z axis is substituted. The schema uses
explicit coordinates rather than requiring reconstruction by arange, leaving
future variable-step schemas possible without implementing continuation now.

`x_um/y_um` preserve actual optical coordinate dtype and values. Material
location records include transverse node/face/domain location, face component,
positive orientation and physical half-cell offsets. These offsets affect only
transverse positions, never z. Optical-node response products carry the projection
identity. Cuts use nearest-zero transverse samples. Face coordinates are the
stored node axes plus the declared component offset, periodic at the domain edge.

## Archives and validation

Archives contain versioned JSON metadata and NPY arrays with dtype, shape, byte
count and SHA-256 descriptors. No pickle or executable object reconstruction is
accepted. Checksums bind the archive's arrays; they do not constitute a signature
or prove physical acceptance independently of the recorded solver evidence.

Encoding is an **explicit host export** of retained arrays, separate from the
backend-resident science/collector path. CuPy exports are direct `cp.asnumpy`,
without a helper that swallows transfer errors. Decoding never imports CUDA or
runs propagation/material solves. A stored request holds NumPy launch data and
the original backend intent. `stored.materialize()` explicitly allocates a new
runtime launch; `backend='numpy'` is an explicit override, not a hidden fallback.
A requested CuPy upload is synchronized before its owned snapshot is released.

Decoded results hold NumPy q/psi/b and selected arrays, while execution provenance
retains the original backend. The storage backend is not claimed to be the
execution backend. Canonical metadata retains active/batch shape, closure, gauge
and mixed precision. Derived E/n/J/phase never substitute for q/psi/b.

Decoders reject unknown schema/physical identities, incompatible shape/dtype,
wrong selection/location metadata, missing canonical state, inconsistent
completion/coordinates, invalid scattering addresses, array checksum failures
and contradictory stored physical-gate summaries. Material arrays are not
re-solved or regauged on loading. Loading a finite q is not a new carrier
positivity/physical-equilibrium certification. The accepted gate evidence is
preserved as evidence from execution.

Failure/cancellation records include the truthful completed count, reached z,
last accepted state and endpoint, failure reason/stage and next cell/canonical
slab address. That address is provenance only, not a resume API. A postprocessing
failure may leave an explicitly failed record with only the products constructed
before that failure; accepted canonical science is still retained.

## Resource estimates

Plans separate accepted/candidate scientific state, optical working fields,
material workspace, float64 sparse solve buffers, face fluxes, scattering,
isolated observations, selected products, coordinate arrays, transaction/final
assembly workspace, canonical final outputs and host serialization scenario.
Reduced factorization estimates depend on one x column, not Nx*Ny; genuine 2D
uses a connected plane and rejects more than 12,288 nodes before allocation.

Array counts are planning figures. FFT/library caches and sparse factorization
fill are not measured; the reported dense-factorization scenario is not a
guaranteed peak upper bound. Serialization includes host arrays plus archive/
buffer coexistence; it is not network transfer until a caller explicitly sends
an archive. No resource report implies native certification or large-2D feasibility.

## Validation boundary

M5 tests compare exact endpoint, q/psi/b, ledger/physical diagnostics and canonical
addresses across selections and against direct M4 execution. An AST regression
binds the entire M4 cell science block to its committed base. Round trips preserve
selected array bytes, dtype, shapes, coordinate arrays and identities. Collector
fault/mutation/cancellation, failed-launch persistence, postprocessing failure,
unknown/malformed records, bounded output shapes, resource scaling and legacy
isolation are covered. Native CuPy tests remain local skips when unavailable;
no H200 commissioning is claimed here.

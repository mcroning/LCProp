# Unified nonlinear Static GUI integration (M6)

Fresh ordinary Static calculations with Fully nonlinear response select
`pr_static_unified_published_optical_first_v1`. Reduced x-only means independent
y batches on the optical plane; Full x-y means one connected material plane.
Local-I remains the distinct paper Eq. (5) response on the commissioned published
workflow. Tangent and all TD selections retain their existing identities.

Saved legacy requests are never converted implicitly. Representable legacy
requests restore their old workflow; unsupported midpoint/symmetric requests
retain the existing explicit loading rejection. A loaded legacy nonlinear Static
request can be replaced explicitly using **Create fresh unified nonlinear Static
request**. That action starts a new unbiased electrical experiment; it is not a
checkpoint or a state conversion. No continuation is provided.

Electrical conditions are explicit: unbiased zero flux, prescribed internal mean
field, prescribed normalized mean current, reduced A7 reservoir/current, and
full-x-y fixed-x-field/open-y-current. A7 retains E_app and I_b and imposes
J_ext = E_app I_b; b is solved. The open-y condition fixes mean J_y = 0, not b_y.
Dark plus uniform background has the unchanged transport meaning. The old
applied-field widget is disabled for unified requests; closure controls own these
quantities. No arbitrary circuit conditions are offered.

The adapter prepares the launch with the existing Product launch primitive on the
explicit execution backend. Its fresh launch-definition schema is
`pr_unified_static_fresh_launch_v1`; the prepared request remains the unchanged
M5 `UnifiedStaticRequest`. The registered material-neutral operation delegates
once to M5 `run_unified_products`. Local and Slurm use the same registration.
Prepared requests and canonical q/psi/b results use the dedicated M5 codecs;
the transport result envelope carries the exact M5 archive. Fresh experiment
files store the launch definition, not a legacy E-only or continuation state.

No material, projection, optical stepping, scattering or precision implementation
is duplicated. The projection is fixed to
`adjacent_face_arithmetic_to_optical_node_v1`. Float32 means state/output float32,
with float64 sensitive linear solves and Bernoulli evaluation; float64 uses
float64 throughout those components. Solver gates remain certified constants,
not the old coupled-pass/backtracking controls.

## Result modes and common linked views

Fresh local unified requests default to Interactive. Interactive and Full select
optical intensity, potential, carrier exp(q), and optical-response E_x volumes.
Analysis can select these independently, and optionally native x/y face fields.
Minimal/Fast omits volumes and retains fixed nearest-zero intensity cuts,
accepted endpoint/state, diagnostics and far field. Selecting products never
changes scientific request parameters or the accepted scientific march.

Optical volumes contain launch and accepted boundaries at 0,h,...,mh. Material
volumes contain only accepted right-endpoint planes h,2h,...,mh. A cancelled run
before cell 0 may retain launch alone and empty material volumes. Rejected cells
add no samples. The internal collector prepares isolated products before atomic
acceptance, then publishes selected planes after acceptance. Post-acceptance
product/observer failures retain truthful failed status and accepted science.

The common Results Fields panel displays **Input Plane Intensity**, **Output
Plane Intensity**, **Optical Intensity**, and **Output Far-Field Intensity**.
Input Plane requires selected intensity volume (or a separately retained launch).
Optical Intensity, Potential, Carrier and Optical-response E_x have linked xy/xz/yz
views. Select the matching xy entry to link the image, then move z, x and y in the
existing longitudinal pane. Each volume supplies its own z/x/y coordinates;
material z is never borrowed from optical boundaries. Native face fields retain
their half-cell coordinate offsets and are not resampled into node fields.
Potential retains its per-active-domain zero-mean gauge metadata. Optical-response
E_x retains `adjacent_face_arithmetic_to_optical_node_v1` projection provenance;
it is not labelled a native face field.

The common Curves panel presents accepted material residual RMS/max and carrier
min/max versus material z, plus existing component/closure residuals. Summary RMS
is the maximum accepted component RMS across independent columns, not an RMS of
Newton iteration history. Summary max uses Gauss, flux-divergence and closure
maximum residuals. The persisted scientific ledger is the authority for these
curves; intermediate Newton observations are excluded from presentation summaries.

The far field is the existing exact M5 Product spectrum of the accepted endpoint,
with Product s_x/s_y axes. No GUI FFT, optical replay, material solve, new masked
transform, peak normalization or axis fabrication is introduced. The existing
viewer rendering/downsampling convention applies without changing retained raw
values. Complete and partial field titles carry status/reached z. The complex
endpoint remains available in the result archive.

Selected result products use `pr_unified_static_products_v2`; the dedicated codec
preserves volumes, explicit coordinates/locations, gauge/projection and the
accepted diagnostic ledger. Version-1 archives remain readable with no fabricated
volumes or spectrum. A v1 archive cannot claim a v2 volume selection, and a new
Full envelope cannot silently substitute Minimal products. Save/reopen reconstructs
common viewer metadata from Product records without rerunning science.

Scientific working memory remains transverse; selected result storage intentionally
scales with cells. For real-state itemsize r, intensity costs (N+1)NxNy*r and each
selected material volume N*NxNy*r bytes. Reduced x-only batching is independent
of the genuine-2D solver ceiling. Backend arrays stay on backend during retention;
explicit result display/export may transfer selected volumes, separately from
bounded live progress. Assembly can hold a second copy. The estimator reports
individual volume bytes, scientific/workspace scenarios, selected output bytes,
host retrieval array bytes and serialized array bytes separately. Archive metadata,
allocator caches and library overhead remain additional. These are planning
estimates, not measured GPU peaks.

Local selections above a 512-MiB retained-volume warning budget require explicit
confirmation (default No). The warning reports each selected volume and assembly
costs. It never changes grid, scientific parameters or product selection silently;
users may choose Minimal/fewer Analysis products themselves. The budget is a
presentation safeguard, not a claim about installed memory or solver capacity.

M5 planning is used without allocating a launch for preflight. Array estimates
exclude archive metadata/allocator overhead and are not measured peak bounds.
Full x-y requests above 12,288 active nodes fail on every execution target,
including H200. This is an algorithmic reference-solver limit; reduced large
optical planes are not subject to that connected-2D ceiling. No large-2D
feasibility or M6 native GUI certification is claimed.

## Accepted-state progress and installed-GUI views

The optional M4/M5 `observer=None` interface emits an event only after atomic
cell acceptance. The private product collector prepares before acceptance and publishes selected
planes only after acceptance; it is never the GUI progress callback. An event carries accepted cell index/count, total cells,
boundary z, copied final accepted material diagnostics (excluding Newton trial histories), and an independently owned intensity
preview/coordinate pair. It exposes no q/psi/b or mutable scientific buffers.
Frozen event records do not make their detached array buffers immutable.

Cancellation from an observer prevents future cells; it does not revoke the
observed accepted cell. On the final cell there is no future work to cancel.
Observer or preview construction failure returns failed status with stage
`post_acceptance_observer` and the accepted state intact. Postprocessing still
uses that accepted endpoint. No launch event or unaccepted-candidate event is
emitted. Scientific evolution is identical when observation is disabled.

Preview intensity is block-averaged with the commissioned TD backend-first
helper, then transferred as at most 128×128 float32 (65,536 bytes), with two
bounded coordinate vectors. Coordinates are bin-center averages of the actual
Product grid. No full scientific plane crosses to host for progress. Backend
intensity and reduction workspaces are temporary transverse planes; the adapter
reports a separate conservative workspace scenario, not a measured native peak.
GUI events are ephemeral; no longitudinal progress history enters the result
archive. The observer consumer is responsible for storage if it records events.

`Intensity xz` and `Intensity yz` alias M5 fixed nearest-zero intensity cuts for
Minimal results. They have paired-cut metadata and no fictitious source volume;
selecting them cannot dereference a missing volume. Existing common paired fields
remain hidden from the image selector unless explicitly opted in. No common
rendering system or TD scientific/presentation path is replaced.

Native follow-up remains bounded: both precisions, callback/volume selection
identity, backend retention, bounded progress transfers, explicit selected-volume
export/codec, cancellation and accepted-state ordering. This work does not rerun
or supersede the 556-solve native material matrix.

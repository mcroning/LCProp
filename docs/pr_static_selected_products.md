# Selected PR Static products

Result selection is operational envelope/execution-intent state, not part of
`PRStaticRunRequest`. It takes effect after the unchanged solve, independent
replay and completion diagnostics. Local reduced Static and remote execution
honor the same selection.

- **Interactive** (`interactive`, or legacy execution spelling `fast`): actual
  executed input/output intensity previews and canonical far-field preview;
  existing exact nearest-zero x-z/y-z cuts; existing 96×96×Nz MPR preview;
  histories, convergence, scalar diagnostics and provenance. Neither complex
  endpoint nor the four omitted material volumes is exported.
- **Analysis** (`analysis:<comma-separated sorted selection>`): Interactive plus
  any independently selected `input_intensity`, `output_intensity`,
  `far_field_intensity`, `complex_input`, `complex_output`. Bare `analysis`
  selects no additional exact products. Exact intensity does not require
  retaining/exporting its complex endpoint.
- **Full** (`full`): unchanged complete scientific content. Other PR workflows
  retain their existing Fast/Full semantics; Analysis is reduced-Static only.

The GUI labels reduced Static's historical `fast` choice Interactive. Saved
execution intent preserves the operational selection, including Analysis
checkboxes. The result codec is version 4 and accepts historical versions 1–3.
Old Fast artifacts still mean their stored endpoints/cuts/preview; decoding
never relabels them as new previews. Legacy Full-to-Fast codec projection is
retained for historical compatibility, not used by new Fast execution.

## Preview definitions and memory

Maximum display side is 1024, retaining transverse aspect ratio (integer floor,
minimum one sample). For 8192×4096 this is 1024×512. Contiguous blocks use the
existing integer linspace bin boundaries and arithmetic means in float64,
then float32 storage. Coordinates are means of original sample coordinates
within each block. Metadata includes axes, units, normalization, original and
preview dimensions, coordinate vectors, exact block boundaries, original
extent, dtype and `visualization_only=true`.

Input/output previews come from executed backend fields, never reconstructed
request approximations. Canonical far-field FFT/shift, coherence grouping,
`dx*dy` transform scaling and `(n/lambda)^2` density convention are reused.
The transform requires a full complex plane; unselected intensity is formed
only in row blocks and immediately reduced. Reduction rows are limited to
262144 elements where the transverse row itself fits that limit; larger rows
necessarily set the one-row lower bound. Host transfers are row summaries,
coordinates and selected products. FFT workspace remains backend dependent.

Cuts and orthogonal-slice previews use the authoritative NumPy presentation
reference, separately from the scientific backend reference. Exact supplied host
launches carry the scalar from accepted owned bytes; GPU-generated launches
use bounded host streaming after replay. Background subtraction, sampling and
block definitions are unchanged. Native CuPy equivalence requires commissioning.
This streaming is permitted for complete and cancelled/nonconverged partial
results with accepted slices whenever requested retained cuts/previews need the
reference and no accepted-host scalar exists. It occurs only in result
construction after slice acceptance and existing replay/completion have ended;
it does not change scientific state or truthful result status. Zero accepted
slices and failures before result construction cause no presentation-reference
readback. Full construction requires no such reference; its later Fast projection
uses retained host bytes. All Interactive/Analysis policies include baseline
cuts/previews, even when no additional exact Analysis product is selected.
For an 8192×4096 single-channel complex128 GPU-native launch, complete or retained
partial results can require 512 MiB of streamed device-to-host launch traffic,
using bounded host workspace. This is not package storage, Mac/network retrieval,
or a longitudinal-volume transfer.
Carrier scalar diagnostics run on the execution backend using bounded intensity
reductions. Two-carrier partition masks and FFT workspaces still require planes.

Solver-volume aliases are released after completion diagnostics. Trial and
launch endpoint aliases are released before selected construction, and final
backend endpoints after construction. Memory-pool reservations are not live
references and are not forcibly flushed. Full behavior remains unchanged.
Cancellation records the accepted completed length; a zero-length result has
executed-launch endpoints and explicitly unavailable longitudinal products.
Failure receipts are distinct from successful selected-product artifacts.
Selected missing products or inconsistent metadata fail codec validation.

## Rendering and quantitative use

Matplotlib receives at most a 1024-side block-mean image, cached for the selected
array, even when exact Analysis/Full arrays exist. Original scientific arrays,
coordinates and exact selection indices remain available. The title identifies
display reduction. Pan/zoom reuses the artist and reduced pixels. Full-array
color-limit calculation is cached by weak array identity; it is not repeated
on ordinary redraw/control refresh. Exact automatic limits retain their former
semantics; explicit preview products have preview-derived limits.

Exact Analysis intensities are separate fields marked Exact; preview labels and
metadata remain explicit. Complex selections remain in the scientific result
and verified local archive. No preview is promoted to a full-resolution array.

## Projected 8192×4096×80 float64 budget

| Product | Raw bytes |
|---|---:|
| Existing x-z/y-z cuts | 7,864,320 |
| Existing 80×96×96 float32 MPR | 2,949,120 |
| Three 1024×512 float32 previews | 6,291,456 |
| x/y and s_x/s_y float64 coordinate vectors | 196,608 |
| **Interactive arrays** | **17,301,504** |

JSON (including complete iteration history, preview mappings, metadata) and
uncompressed NPZ headers add overhead: expected package scale remains about
18 MB for the commissioned history, not a fixed cap on arbitrary histories.
A serialization-only projection using that saved history measured 17,304,770
NPZ bytes plus 510,416 JSON bytes = 17,815,186 bytes, before small bundle receipts.
Placeholder arrays were used for this budget (NPZ is uncompressed); this is
not a new scientific result or a native measured transfer.
Each selected full-resolution float64 intensity adds 268,435,456 raw bytes.
Each one-channel complex128 endpoint adds 536,870,912 raw bytes. Full still
includes the four 20-GiB material volumes for this geometry.

The existing bundle writer publishes via rename without a second full disk
copy. Peak temporary remote scientific storage, transfer and retained local
package size are the selected bundle size plus compact receipts/logs; live
RAM and FFT workspace are separate. Successful verified retrieval uses existing
remote cleanup, leaving zero scientific-package bytes in the run directory.
There is no deferred retrieval or persistent cluster archive. A packaging
exception removes only that attempt's known `output.incomplete.<uuid>` path,
then writes compact failure evidence. Other attempts, source caches and user
files are untouched. Cleanup errors are recorded rather than hidden. Abrupt
process/node death cannot execute Python cleanup and remains an operational
failure-recovery case; no claim of guaranteed cleanup after an uncatchable kill.

### Display revisions and linked coordinates

`FieldData.content_revision` is a producer-owned monotonic revision. After any
in-place change to data, coordinates, block/normalization metadata or equivalent
log/linear presentation semantics, present a replacement FieldData with an
incremented revision. Unchanged revisions and display state reuse bounded pixels
and automatic limits; no scientific-array hash is computed during redraw.
Explicit fixed/locked color limits remain user-controlled. Fields with changed
kind/quantity/units invalidate automatic limits even with the same data object.

Linked near-field selections pass physical x/y coordinates, using stored preview
block centres; the MPR independently chooses its nearest coordinate. Far fields
have direction-cosine axes and do not link angular clicks to spatial MPR cuts.

Optional image analysis requires both `complex_input` and `complex_output`.
Interactive or insufficient Analysis selections report `not_selected`, with
pre-run advice to select both endpoints or Full. No products are added implicitly.
When switching away from reduced Static, Interactive (including legacy `fast`)
and Analysis map explicitly to TD's `fast`; Full remains `full`. Returning to
Static keeps that compatible selection (Interactive or Full); Analysis selections
remain available for explicit reselection. Labels and submitted values update
atomically. Scientific requests and product byte budgets are unchanged.

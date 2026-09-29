# Reduced Static completion memory

Job 4623802 (request SHA-256
`baf5947b9211478e80b5457357245641a29e65f69c411aa0bc3d4febe5633ebe`)
completed its march and independent replay, then failed allocating the 20-GiB
square of its `(80,8192,4096)` float64 replay residual.

Completion diagnostics now slice existing arrays into views of at most 262,144
elements, splitting existing axes without flattening a strided volume. Each
float64 work array is at most 2 MiB (complex128: 4 MiB). The RMS path holds only
bounded normalized/squared chunks and backend reduction workspace, not another
volume. Comparisons likewise bound differences, absolute values, `isclose`
intermediates and masks. A conservative working allowance is 32 MiB for these
chunk-local elementwise arrays; backend reduction workspace and allocator-pool
reservation are implementation-dependent and are not measured GPU peaks.

RMS remains `sqrt(sum(residual**2)/N)`. A global absolute maximum scales values
before squaring; chunk sums use float64 and scalar sums use `math.fsum`.
This avoids square overflow/underflow for uniformly extreme finite inputs.
NaN/infinite residuals cannot pass convergence. Max and elementwise `isclose`
keep their definitions, including asymmetric relative tolerance and nonfinite
behavior. All chunks contribute; a failed comparison does not suppress the
reported maximum difference.

Float64 reduction order is not bitwise identical to the old backend mean.
Away from tolerance boundaries the conservative forward-error envelope protects
the decision. Near a boundary, a backend integer-digit reduction compares the
exact sum of squares of the represented finite binary64 inputs with
`N * tolerance**2`. No residual samples cross to host for this decision.

Every finite binary64 value has an integer significand M and can be expressed
as `M * 2**-1074 * 2**max(E-1,0)`, where E is its exponent field and the hidden
significand bit is present only for normal values. Thus its square is an integer
in units of `2**-2148`. The backend splits M into four 16-bit limbs, computes
seven square digits with uint64 multiplication/addition and carry propagation,
and shifts them to their exponent-addressed bins. Signs disappear on squaring.
All limb products/coefficient sums fit uint64. This also handles subnormals
without floating square underflow and the largest finite values without overflow.

Each of the fourteen weighted histograms has nonnegative integer weights at
most 65535 and at most 262,144 samples. Each bin sum is less than `2**34`;
combining fourteen histograms stays below `2**38`. Therefore float64 histogram
accumulation is exact regardless of order, including GPU atomic-add order.
The 264-bin summary covers all finite float64 square exponents. Host Python
combines its integer digits using exact integer shifts/addition and compares
against an exact integer representation of `N * tolerance**2`. No arbitrary
acceptance epsilon is introduced, and no per-sample Python/Fraction work remains.

Each boundary chunk exports exactly **264 float64 bins = 2,112 bytes**; the
ordinary path exports scalars. Backend work is O(N); Python work and transfer
are O(number of chunks), with 264 integer terms per chunk. Native geometry
uses 10,240 chunks, at most 20.625 MiB of summary transfers for the complete
boundary pass, rather than 20 GiB of samples. The fallback is more expensive
than ordinary reductions but has no Python loop over residual elements.
Individual chunk temporaries remain at most 2 MiB, plus fixed 2,112-byte
summaries. A conservative simultaneous elementwise-array allowance for the
integer fallback is 64 MiB, excluding backend workspace/pool reservations.

An old rounded mean could accept a value whose true RMS is slightly above
tolerance; the exact boundary decision correctly rejects it. The reported
float64 RMS may itself round to the tolerance in that case. Primary per-slice/
trial reductions and all scientific updates are unchanged.

Audit of adjacent completion operations:

- Full-volume residual square and absolute-value temporaries: chunked.
- Replay/primary source and residual differences and absolute values: chunked.
- Field/source/residual `isclose` intermediates and inverted masks: chunked;
  use chunk `all` instead of building a full inverted mask.
- Convergence aggregation: scalars and slice summaries only.
- Result conversion `asnumpy(volume).copy()`: CuPy's owned host transfer is
  returned directly; NumPy inputs still receive independent copies.
- Power calculations remain transverse field operations (no longitudinal
  volume). Scattering provenance remains compact metadata/seed identities.

The six underlying device volumes are unchanged. The four required host result
volumes are also unchanged: they total **80 GiB** at the native geometry,
before optical fields and downstream products. Fast transport omits volumes
only after construction. Removing redundant host transfer copies does not make
this fit a 64-GiB host-memory request. This remediation therefore addresses the
observed GPU diagnostic OOM, not end-to-end commissioning readiness under the
original resources. Changing retention/construction policy or resources requires
separate scope; no such change is included here. TD is untouched.

Focused tests compare direct definitions and real linearized/nonlinear workflows,
exercise exact tolerance boundaries, extreme/nonfinite values, strided arrays,
complex comparisons, owned host transfers, and a backend/array guard that rejects
full-volume diagnostic operations. Optional CuPy parity remains a native-backend
check when a CUDA device is available.

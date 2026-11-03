# Local PR run-cost guard and responsive cancellation

**Status:** Development and local validation
**Baseline:** `0c338787193a4012f284125e9408b6e103f34e79`
**Branch:** `feature/pr-second-order-static`

## Motivation and scope

Real-user testing exposed two local-only usability hazards. A large nonlinear
full-transverse request could be launched without any indication of its
worst-case iterative work, and Stop was observed only outside an entire
frozen-intensity material solve. A 1024-by-1024 calculation with hundreds of
longitudinal planes could therefore spend minutes inside per-plane Newton and
PCG work before the existing outer-loop token check ran.

This milestone adds a GUI pre-launch cost guard and cooperative cancellation
inside discardable full-transverse static trials. It does not change PR
equations, tolerances, iterations, acceptance tests, optical propagation, or
remote execution policy.

## Coarse local cost classifier

The classifier uses only the immutable request and selected execution target.
It resolves `Nz` with the same grid rounding helper as production and includes:

- transport family (reduced or full transverse);
- static or time-dependent workflow;
- nonlinear or linearized material response;
- `Nx`, `Ny`, and `Nz`;
- local backend and precision;
- coupled, material, Newton, PCG, or TD iteration limits as applicable.

The score is an explainable relative work indicator, not a runtime estimate.
NumPy float64 is the reference factor; float32 and CuPy reduce the coarse local
factor. Thresholds are deliberately separated by algorithm:

| Workflow/model | Potentially Expensive | Very Expensive |
| --- | ---: | ---: |
| full-transverse nonlinear static | `5e9` | `5e11` |
| full-transverse nonlinear TD | `5e9` | `5e10` |
| full-transverse linearized static | `2e10` | `2e11` |
| reduced static or TD | `2e10` | `2e11` |

Nonlinear static work includes coupled passes, per-plane Newton limits, and a
square-root PCG factor. Linearized static work uses the fixed-count Fourier
response factor instead. Consequently a large grid is not automatically given
the same class as the nonlinear Newton/PCG path. A Slurm-selected request is
always outside the local guard and receives no local warning.

The required reference cases classify as follows under local NumPy float64:

| Case | Class |
| --- | --- |
| small reduced nonlinear | Normal |
| small full-transverse nonlinear | Normal |
| 256-by-256-by-100 full-transverse nonlinear | Potentially Expensive |
| 1024-by-1024-by-300 full-transverse nonlinear | Very Expensive |
| same 1024-by-1024-by-300 grid, linearized response | Potentially Expensive |
| extreme nonlinear request selected for Slurm | Normal/local guard bypassed |

## Warning and confirmation behavior

Normal local requests start without an added dialog. Potentially Expensive
local requests show a concise warning and then may start normally. The warning
names the scientific model, grid, Local execution, recommends Slurm/H200 for
large full-transverse nonlinear PR, and explains safe cancellation checkpoints.

Very Expensive local requests require an explicit choice:

- Cancel;
- Use Slurm/H200;
- Run locally anyway.

Choosing Slurm changes only the execution selector and stops the launch. The
user must press Run again to submit explicitly; the guard never submits a
remote job automatically. If Slurm is unavailable, that choice is disabled.

Image Amplification is classified from its prepared base request, so the same
guard applies without changing IA physics or postprocessing.

## Cancellation root cause and checkpoints

Before this change, the full-transverse static outer workflow checked the token
at a coupled-iteration boundary and after the material proposal returned. The
host-volume material adapter could meanwhile solve every plane, and each plane
could execute all Newton, PCG, residual, and line-search work. The coupled
optical trial and refreshed residual volume also lacked internal checks.

The following cooperative checkpoints now bound those discardable sections:

- between frozen-intensity planes;
- before each Newton iteration and between its major residual stages;
- before and after each PCG Jacobian action;
- between material line-search trials;
- between planes of coupled trial projection and physical-state validation;
- between coupled optical-propagation planes;
- between refreshed coupled-residual planes;
- before each coupled line-search trial and before accepting a candidate;
- the existing coupled-iteration and post-material boundaries.

The callback is optional. With no cancellation request, it inserts token reads
only and leaves numerical ordering and algebra unchanged.

## Accepted-state and status semantics

All new inner checkpoints operate on a discardable proposal. Cancellation
raises an internal control-flow signal to the outer workflow, which discards
the partial plane volume, Newton/PCG correction, optical trial, or residual
trial. It does not publish or adopt partial state. The returned result is
`cancelled`, records the precise observation stage, and retains the last
accepted potential and corresponding optical state. A stop after one accepted
coupled iteration returns that exact accepted potential. Repeated token
cancellation remains idempotent, and normal GUI worker teardown re-enables a
subsequent run.

No process force-kill is used. The architectural target is observation at the
next bounded solver checkpoint, not a hard wall-clock guarantee. Local tests
inject cancellation inside residual construction, PCG, plane traversal, and
the coupled optical trial to verify prompt control return without requiring an
unbounded benchmark fixture.

## Scientific and remote non-change

A deterministic regression compares an ordinary solve with the same solve
using an uncancelled token. Status, convergence, iteration records, optical
field, potential, and source volume are bitwise identical. No material
equation, Newton update, PCG recurrence, line-search acceptance rule,
tolerance, or optical operator changed.

Slurm submission, scheduler cancellation, retrieval cancellation, result
preservation, and Fast/Full retrieval code were not modified. Existing remote
GUI and transverse-static regressions remain the remote non-regression gate.

## Local validation

Validation is performed in the repository `lcprop` environment with
`QT_QPA_PLATFORM=offscreen` for GUI tests.

| Gate | Result |
| --- | --- |
| focused classifier, GUI, and cancellation tests | 18 passed |
| focused plus existing transverse workflow, GUI, and remote UI tests | 75 passed |
| transverse nonlinear/linearized, transport, and Slurm regressions | 221 passed, 33 skipped (conditional GPU cases) |
| complete `tests/test_pr*.py` suite | 550 passed, 70 skipped (conditional GPU cases) |
| `python -m compileall -q src tests` | passed |
| `git diff --check` plus checks for new untracked files | passed |

The default shell Python 3.9 environment aborted while importing PySide6
before test collection. All reported validation used the repository's Python
3.12 `lcprop` environment, where PySide6 6.11.1 imports and offscreen GUI tests
normally.

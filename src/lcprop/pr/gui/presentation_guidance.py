"""Read-only GUI explanations; no request mutation or capability decisions."""
from lcprop.pr.unified import solver_specs
from lcprop.pr.unified.specs import (
    PRMaterialPrecisionSpec, PRElectricalClosureSpec, DOUBLE_PRECISION,
    POSITIVE_PRECISION, UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE,
)


def unified_support_guidance():
    closures = [('unbiased', UNBIASED), ('fixed field', FIXED_FIELD),
                ('prescribed current', PRESCRIBED_CURRENT), ('open-y', OPEN_TRANSVERSE)]
    parts = ['Reduced x-only: independent y columns; connected direct/reference uses '
             'its separate active-node bound. Scalable limits per axis:']
    for backend, label in [('numpy', 'NumPy'), ('cupy', 'CuPy')]:
        for dtype, identity in [('float64', DOUBLE_PRECISION), ('float32', POSITIVE_PRECISION)]:
            precision = PRMaterialPrecisionSpec(identity, dtype, output_dtype=dtype)
            limits = [(name, solver_specs.scalable_axis_limit(backend, precision,
                       PRElectricalClosureSpec(kind, 2, (0., 0.)))) for name, kind in closures]
            if len({limit for _, limit in limits}) == 1:
                detail = f'all four closures ≤{limits[0][1]}'
            else:
                detail = ', '.join(f'{name} ≤{limit}' for name, limit in limits)
            parts.append(f'{label} {dtype}: {detail}.')
    parts.append('The unbiased 384×32 bridge is a separate exception. '
                 'Inspect Request applies the selected solver/backend/precision envelope '
                 'and resource guidance; Slurm does not enlarge that envelope. '
                 'Scientific support is separate from interactive runtime recommendations.')
    return ' '.join(parts)


TD_ELECTRICAL_GUIDANCE = (
    'Nonlinear full-transverse TD: periodic transverse boundaries and fixed unbiased '
    'mean internal field b_x = b_y = 0 throughout evolution. The local field may be nonzero '
    'through −∇ψ. Static prescribed-current and open-y closures are not TD controls.'
)


def checkpoint_ownership(checkpoint, *, transverse=False):
    """Describe an existing checkpoint without validating or changing eligibility."""
    if checkpoint is None:
        return 'No TD checkpoint retained.'
    owner = 'Full-transverse TD' if transverse else 'Reduced TD'
    status = checkpoint.record['status'] if transverse else checkpoint.status
    return (f'{owner} · {checkpoint.completed_steps} cumulative steps · '
            f'τ = {checkpoint.time_normalized:.6g} · {status}')


def checkpoint_folder_hint(*, transverse=False):
    files = ('accepted-psi.npy and transverse-continuation.json' if transverse else
             'checkpoint.npz, request.json and provenance.json')
    return 'Choose a destination folder; LCProp writes ' + files + ' into it.'


def accepted_td_time_text(progress):
    """Format accepted TD metadata only; never advance time from a wall clock."""
    from lcprop.pr.transverse.specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    start = progress.segment_start_time
    # Reduced TD current_coordinate is already cumulative. Transverse fresh
    # progress is segment-local; continuation supplies authoritative cumulative_time.
    cumulative = (progress.cumulative_time if start or progress.cumulative_time
                  else progress.current_coordinate)
    segment = (progress.current_coordinate
               if progress.workflow == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
               else progress.segment_elapsed_time if start else progress.current_coordinate)
    text = f"PR cumulative material time τ = {cumulative:.6g}"
    if start:
        text += f"; segment τ = {segment:.6g} (start τ = {start:.6g})"
    return text + f"; step {progress.completed_units}/{progress.total_units}"

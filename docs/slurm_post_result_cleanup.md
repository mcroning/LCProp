# Bounded post-result cleanup

Only after retrieval, verification, decoding and product reconstruction succeed,
the Slurm runner transitions from Reconstructing to Cleaning. It attempts deletion
of the existing validated, exact per-run directory once, using the unchanged
`rm -rf -- <validated path>` operand and POSIX SSH argument serialization.

Post-result cleanup uses a separate timeout-capable transport operation with a
30-second timeout. Other SSH operations keep their existing behavior. Only this
cleanup subprocess uses POSIX `start_new_session=True`. On timeout its isolated
process group receives TERM, followed after 0.25 seconds by KILL, including local
SSH helpers inheriting the group. The leader is not reaped before group signaling,
so its PID cannot be recycled into an unrelated group. Existing SSH masters and
other runs are not signaled. No retry is launched. All post-spawn setup, including selector creation and
registration, is inside the process-ownership region. TERM and grace/drain errors
are captured independently of the unconditional KILL attempt while ownership
remains active. Final drain, each pipe close, selector close, and child reap are
separate protected stages. Shutdown faults are retained as bounded diagnostics.
A failed bounded wait is followed only by a nonblocking reap check; an unconfirmed
child termination is reported as uncertainty, with no additional wait or signal
after reaping.

Nonblocking selector reads drain both pipes throughout execution/termination.
After KILL, pipe draining is limited to 0.25 seconds; remaining pipes are closed.
The direct child is explicitly reaped with a 1-second wait bound. Thus the normal
timeout shutdown allowance is 1.5 seconds beyond the 30-second deadline, excluding
process creation and OS scheduling. An OS-uninterruptible process cannot be
promised to terminate within a userspace deadline; any such failure is reported,
not treated as confirmed deletion. Deliberately session-detaching helpers are
outside the inherited-process-group contract. Remote deletion may have completed
even when local SSH times out.

Each stream retains at most 8,192 raw bytes: a 4,096-byte head and 4,096-byte tail.
Reads use at most 16,384 bytes at a time; discarded output is counted, not stored.
Operational diagnostics report `truncated`, `bytes_observed`, `bytes_retained`,
`retention_limit_bytes`, and `capture_complete` separately for stdout/stderr.
`capture_complete=false` means EOF was not established, so observed bytes need not
be the total emitted. Excerpts decode UTF-8 with replacement. The complete formatted GUI cleanup status (including path and explanatory text)
is capped at 20,480 characters using a marked head/tail excerpt. The uncropped
exact path remains in structured operational provenance. No full cleanup-output log is retained.

A successful cleanup records `completed`. Timeout records `timed_out`; nonzero
remote-command exit records `failed`; SSH exit 255, connection loss and other
indeterminate failures record `indeterminate`. Failure diagnostics preserve bounded stdout/stderr excerpts
and the exact remote path. Remote artifact retention is unknown (`None`)
after an unsuccessful attempt, rather than asserted true or false. Unsupported
custom transports without the bounded operation fail non-fatally; there is no
fallback to unbounded SSH.

The verified scientific result is returned in every cleanup-failure case. Its
scientific status and arrays are unchanged. `RunnerResult.operational_provenance`
and the final RemoteRunStatus record cleanup separately, including its outcome,
timeout, target, error and confirmed completion time when applicable. The GUI's
completion status reports unconfirmed deletion when needed. Cleanup disabled by
existing policy records `not_requested` and retains remote artifacts.

Retrieval, checksum, decoding and product-conversion failures remain result
failures and never enter successful-result cleanup. Recovery from a previously
retrieved bundle does not require scientific recomputation. This change does not
interrupt or repair SSH processes launched by older running GUI instances.

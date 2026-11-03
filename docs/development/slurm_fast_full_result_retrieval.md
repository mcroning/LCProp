# Slurm Fast and Full Result Retrieval

## Status

Development policy for remote PR result retention and retrieval. This policy
does not change any PR equation, solver, convergence rule, or local execution
result.

## User policy

The PR GUI offers two remote-result policies:

- **Fast / Exploratory** (default): retain and retrieve the initial and final
  optical channel fields, compact request/provenance/status/diagnostic records,
  and data needed to construct input/output optical products. Transverse PR
  results also reconstruct the output far field from the retained final
  optical channels. Image Amplification retains the base optical endpoints
  needed by its existing local analysis and specialized products.
- **Full** (explicit opt-in): retain and retrieve the complete canonical result,
  including longitudinal material/source/residual volumes and continuation or
  checkpoint state where the workflow defines them.

The choice is an execution/result-retention policy, not a scientific request
parameter. Local execution remains unchanged. Older request/result envelopes
without a policy field decode as Full.

The reduced PR workflows add `refractive_index` to their existing launch
summary so a decoded Fast result can reconstruct its physical far-field axes.
This is an additive provenance/presentation metadata change in Local and remote
results; it does not change launch fields, propagation, or any scientific
array.

## Ownership and artifact construction

The shared transport envelope carries only the material-neutral policy value
`fast` or `full`. Each registered PR transport codec owns its projection:

| Workflow | Fast-retained arrays | Full-only arrays/state |
| --- | --- | --- |
| `pr_static` | `A_initial`, `A_final` | `E_initial`, `E_final`, source and residual stacks |
| `pr_timedependent` | `A_initial`, `A_final` | `E_initial`, `E_final`, source stack, checkpoint |
| `pr_transverse_static` | `A_initial`, `A_final` | initial/final potential, source, equilibrium-residual and TD-residual stacks |
| `pr_transverse_timedependent` | `A_initial`, `A_final` | initial/final potential stacks |

Projection occurs before `result_arrays.npz` is written. Therefore Fast mode
does not first write, archive, or transfer the omitted arrays. `SlurmRunner`
contains no PR field names or material-specific retention branch.

Decoded Fast results use the existing canonical result types, with omitted
Full-only members explicitly set to `None` and a `retention_summary` recording
the policy and exact omitted fields. Product adapters must not synthesize
missing scientific volumes. The Results workspace disables longitudinal
selection and reports:

> Not retrieved in Fast mode; rerun with Full result retrieval to inspect this volume.

## Size model and the 1024² case

For two complex128 optical channels at 1024², each retained endpoint is about
32 MiB, so the principal Fast array payload is about 64 MiB plus compact JSON
and NPZ metadata. The diagnosed Full result for job 2994555 was approximately
12.56 GiB: four 3.125-GiB longitudinal volumes plus two 32-MiB optical endpoint
arrays. Fast mode therefore removes roughly 12.5 GiB from that transfer before
packaging. Exact sizes vary with channel count, precision, grid, workflow, and
diagnostic records.

## Retrieval progress and failure behavior

The subprocess transport visibly reports bytes transferred, total bytes when
remote `du` is available, percentage, the current retrieved object, and average
transfer rate. These are observational; they do not alter verification.

Stop/Cancel during retrieval terminates and reaps only the local SCP/SSH
process group. The already completed scheduler job is not cancelled, successful
cleanup is not entered, and the complete remote package is retained for a
separately authorized recovery. Retrieval cancellation is intentionally
distinct from resume: the SCP transport still has no automatic retry or
byte-range restart. A later retrieval begins as a separately authorized
operation.

Successful Fast and Full runs use the same Stage-02D cleanup gate: remote run
deletion can occur only after download, checksum verification, canonical
reconstruction, and product conversion. Failed, cancelled, interrupted, or
verification-failed runs remain remote. Automatic age-based deletion of failed
runs is not implemented here; site/user cleanup remains an explicit operation
so failure evidence cannot disappear without a separate retention decision.

## Existing job 2994555

The preserved result is a valid Full artifact and need not be recomputed to
recover the optical endpoints. After the active retrieval is stopped by the
user, a separately authorized compact-recovery operation can verify the remote
manifest/checksums and project `A_initial` and `A_final` into a new Fast-format
artifact. The original remote package should remain untouched until the compact
artifact has itself passed checksum verification and product conversion.

This milestone does not start, stop, or alter that transfer and performs no
cluster operation.

## Validation contract

Validation covers:

- Fast and Full codec behavior for all four PR workflow families;
- old-envelope Full compatibility;
- request/result policy identity binding;
- Fast product conversion and explicit longitudinal unavailability;
- Image Amplification Fast-vs-Full equivalence for all four compatible PR base
  workflows through decoded results, common analysis, specialized arrays,
  metrics, geometry, presentation metadata, and scientific diagnostics;
- GUI default/opt-in behavior;
- progress presentation;
- unchanged Full and local scientific results.

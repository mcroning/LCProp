# Remote Transport Foundation

This package transports canonical material requests and results through strict
JSON metadata and host NumPy NPZ arrays. Material packages own their codecs;
the shared layer owns envelopes, checksums, state vocabulary, and a
scheduler-neutral headless executor.

The canonical material result is authoritative. `RunData` is reconstructed
locally through the existing registered operation's product adapter after all
artifacts and checksums have been verified. NPZ loading always uses
`allow_pickle=False`.

Supported in the initial transport contract:

- LC canonical static requests/results, including static checkpoints;
- PR canonical full-transverse static requests/results, including continuation
  provenance and exported authoritative residuals.

The sibling `lcprop.runners.slurm` module implements material-neutral Slurm
submission, polling, verified retrieval, and reconstruction using these
artifacts.  Application composition remains in `lcprop.transport.defaults`;
the runner itself contains no LC- or PR-specific request/result logic.

The LC and PR applications expose **Execution: Local | Slurm** separately
from the scientific backend. Cluster sites and resource profiles are loaded
from a versioned user-local `clusters.toml`; multiple clusters and multiple
CPU/GPU profiles can coexist without source changes. A missing configuration
is safe and leaves ordinary Local execution available. No credentials are
stored in cluster profiles, requests, or transport artifacts.

The default configuration path is the platform application-config location:
`~/.config/lcprop/clusters.toml` on a typical XDG system,
`~/Library/Application Support/LCProp/clusters.toml` on macOS, and the LCProp
directory under `%APPDATA%` on Windows. `LCPROP_CLUSTER_CONFIG` selects an
explicit path. Explicit constructor values take precedence over environment
overrides, which take precedence over the selected profile; only neutral
values such as polling cadence and local artifact location have package
defaults.

Resource profiles define partition, QOS, time, CPU, memory, GPU count, optional
site-specific GRES, setup commands, and GPU/CuPy preflight requirements. H200
is not universal: a profile may request `gpu:1`, `gpu:h200:1`, `gpu:a100:1`, or
another validated site value. Device-name matching is optional and
profile-specific. Because CuPy is LCProp's only supported GPU scientific
backend, every profile requesting GPUs must set `require_cupy = true`. GPU
execution derives provenance from the scheduler-visible CUDA/CuPy device and
always retrieves its actual name, visible-device count, CuPy version, and CUDA
runtime/driver versions.

`setup_commands` are trusted user-supplied executable shell commands. LCProp
rejects multiline and NUL-containing values, but intentionally executes each
configured command rather than treating it as untrusted data. Cluster-profile
files must therefore be protected and reviewed like shell configuration; do
not place passwords, tokens, private keys, or other credentials in them.

Normal profile-driven Slurm execution automatically resolves the exact local
Git `HEAD`, builds a deterministic archive containing committed `src/` and
`pyproject.toml` content, and stages it beneath the selected cluster's
`source_root`. Both branch and detached clean checkouts are supported. Changes
inside the deployable boundary—including untracked source files—block staging;
untracked research results and other files outside that boundary are not read
or uploaded.

Snapshots are named by the full commit SHA, carry exact SHA and archive-SHA256
markers, and are made operationally read-only before an atomic final link is
published. An existing snapshot is reused only when its full SHA, checksum, and
required `src/lcprop` layout match. A conflicting or incomplete snapshot fails
before `sbatch`; it is never overwritten. Temporary staging paths cannot be
mistaken for finalized snapshots.

`LCPROP_SLURM_SOURCE_PATH` and `LCPROP_SLURM_SOURCE_SHA` remain a paired
advanced override for CI, commissioning, and deliberately pre-staged sources,
but ordinary configured users no longer need them. Automatic deployment from
an installed package without a Git checkout remains deferred and fails with an
actionable request to use that explicit override.

Both material GUIs provide **Configure Remote Execution…** beside the
independent Local/Slurm execution selector. The shared dialog accepts a profile
name, SSH username and login host, remote run/Python/source paths, polling
interval, and one or more resource profiles. Profiles are saved atomically to
the user-local TOML file and become selectable without restarting. Local stays
selected until the user explicitly chooses Slurm. If the file is absent or
invalid, Local remains usable and the GUI displays the reason Slurm is
unavailable.

Saving or editing through the GUI rewrites the complete cluster-profile
catalog deterministically. Cluster and resource profile semantics and their
ordering are preserved, but TOML whitespace, comments, and other original
formatting are normalized and are not retained.

**Test Connection** uses system `ssh` in batch mode to check login access,
`sbatch`/`squeue`/`sacct` availability, the configured Python, and writable run
and source roots. For a GPU profile it also checks that CuPy imports on the
login node; it does not request a GPU or submit a scheduler job. LCProp never
asks for or stores passwords, SSH keys, MFA codes, or tokens. Configure normal
system SSH authentication or an SSH agent before testing.

Minimal profile example:

```toml
schema_version = 1
default_cluster = "example"

[clusters.example]
host = "user@login.example.edu"
remote_run_root = "/scratch/user/lcprop_runs"
remote_python = "/scratch/user/env/bin/python"
source_root = "/scratch/user/lcprop_sources"
cleanup_remote_on_success = true
default_resource_profile = "gpu-standard"

[clusters.example.profiles.gpu-standard]
partition = "gpu"
qos = "normal"
time_limit = "00:30:00"
cpus = 2
memory_gb = 16
gpus = 1
gres = "gpu:1"
setup_commands = ["module load cuda"]
require_cupy = true
minimum_device_count = 1
```

A generic CPU resource can instead be kept alongside it:

```toml
[clusters.example.profiles.cpu-small]
partition = "batch"
qos = "normal"
time_limit = "00:15:00"
cpus = 2
memory_gb = 8
gpus = 0
```

A site-specific accelerator is just another profile. For example, a Tufts
H200 profile may use `gres = "gpu:h200:1"`, `require_cupy = true`, and
`expected_device_pattern = "H200"`. This is an example only; no Tufts path,
partition, GPU model, or setup command is a package default.

Remote GUI completion means scheduler success followed by artifact retrieval,
checksum verification, canonical result reconstruction, and conversion by the
existing product adapter. Scheduler `COMPLETED` alone is not GUI completion.
GPU resource profiles also retrieve the allocated-device execution provenance
so the final status reports the actual device. Retrieval, verification,
reconstruction, and product-conversion errors terminate in a categorized
remote `FAILED` state; confirmed scheduler cancellation terminates in
`CANCELLED`.

Successful remote runs delete their per-run directory by default, but only
after the complete result package has been downloaded, checksum-verified,
reconstructed as the canonical material result, and converted through the
registered local product adapter. Set `cleanup_remote_on_success = false` on a
cluster profile to retain successful remote artifacts. Existing profiles that
omit the field use the safe storage default of cleanup enabled.
The shared configuration dialog exposes the same policy as **Delete remote run
artifacts after successful retrieval**, checked by default.

Scheduler failures, timeouts, out-of-memory terminations, cancellations, and
retrieval, verification, reconstruction, or product-conversion failures retain
their remote artifacts for diagnosis. Cleanup failure is reported as a
nonblocking warning and does not discard the verified local result or turn a
successful scientific run into failure. Cleanup is restricted to the exact
generated `<remote_run_root>/<run_id>` tree. Reusable immutable source
snapshots under `source_root` are a separate lifecycle domain and are never
cleanup targets.

This package does not implement GUI-specific scientific behavior or
execution-target selection.

## Bundle and execution contract

Each run directory contains a verified `request/` bundle and, after headless
execution, an `output/` result or failure bundle. Bundles contain a versioned
JSON envelope, an optional host-array NPZ, `manifest.json`,
`checksums.sha256`, and a ready marker written last. Verification checks the
ready marker, manifest, byte sizes, SHA-256 hashes, NPZ members, dtypes, and
shapes before reconstructing a canonical object.

The scheduler-neutral payload command is:

```bash
python -m lcprop.transport.executor --run-dir /path/to/run
```

It resolves the same registered `WorkflowOperation` used by `LocalRunner`,
but skips presentation conversion on the execution host. Requested and
resolved scientific backends remain distinct envelope fields. A later local
retrieval step verifies and decodes the canonical result, invokes the existing
material product adapter, and only then may mark the run `COMPLETED`.

The remote status vocabulary separately represents submission, queueing,
running, scientific-process completion, retrieval, verification,
reconstruction, GUI-ready completion, cancellation, timeout, out-of-memory,
and failure states. Scheduler completion alone is therefore never reported as
GUI-ready completion.

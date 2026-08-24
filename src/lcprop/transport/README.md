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
actionable request to use that explicit override. GUI cluster setup and profile
selection remain deferred to Stage 02C. Until then, GUI remote runs use the
selected cluster's `default_resource_profile`; they do not contain
package-defined CPU or H200 profile names.

Minimal profile example:

```toml
schema_version = 1
default_cluster = "example"

[clusters.example]
host = "user@login.example.edu"
remote_run_root = "/scratch/user/lcprop_runs"
remote_python = "/scratch/user/env/bin/python"
source_root = "/scratch/user/lcprop_sources"
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

Remote GUI completion means scheduler success followed by artifact retrieval,
checksum verification, canonical result reconstruction, and conversion by the
existing product adapter. Scheduler `COMPLETED` alone is not GUI completion.
GPU resource profiles also retrieve the allocated-device execution provenance
so the final status reports the actual device. Retrieval, verification,
reconstruction, and product-conversion errors terminate in a categorized
remote `FAILED` state; confirmed scheduler cancellation terminates in
`CANCELLED`.

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

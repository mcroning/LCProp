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

This package does not implement Slurm submission, remote polling, SSH, or GUI
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

# Stage 5 — manual hosted Colab CPU qualification

Preparation only. Select a **fresh CPU runtime**, upload `qualification-inputs.zip`
through the Colab file pane. No GPU, Qt, LaunchPlane, Drive or Git checkout is needed.
The wheel is the exact retained Stage 4 wheel; the example is a content-pinned
uncommitted file, not a file claimed to exist at the engine commit.

## 1. Verify before installation

Substitute the archive SHA-256 printed in REPORT.md for `BUNDLE_SHA256` below.
Run this cell once; it rejects existing extracted inputs.

```python
from pathlib import Path
import hashlib, json, zipfile
p = Path('/content/qualification-inputs')
a = Path('/content/qualification-inputs.zip')
assert hashlib.sha256(a.read_bytes()).hexdigest() == 'BUNDLE_SHA256'
assert not p.exists()
with zipfile.ZipFile(a) as z:
    assert all(n.startswith('qualification-inputs/') and '..' not in Path(n).parts for n in z.namelist())
    z.extractall('/content')
for name, digest in json.loads((p/'inputs-sha256.json').read_text()).items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest() == digest, name
```

## 2. Explicit setup

```python
%pip install numpy scipy matplotlib
%pip install --force-reinstall --no-deps /content/qualification-inputs/lcprop-0.1.0-py3-none-any.whl
```

Python >=3.10 is required (fresh hosted Colab is expected to be newer; Python 3.10
also needs `tomli`). No GUI/GPU extras. If libraries were already imported, restart
before proceeding, leaving uploaded files in place. Do not assume Colab storage
survives runtime disposal. These installation cells are manual, not automatic code
inside the scientific example. No dependency-version parity is assumed.

## 3. Execute exactly once and preserve provenance

```python
import sys, subprocess
from pathlib import Path
p = Path('/content/qualification-inputs')
out = Path('/content/lc-nonlinear-colab-results')
assert not out.exists() and not Path(str(out)+'.zip').exists()
run = subprocess.run([sys.executable, '-B', str(p/'run_hosted.py'), str(out)])
print('Qualification supervisor exit:', run.returncode)
```

The supervisor verifies the input manifest and all 264 installed LCProp files,
copies the exact executed script into the evidence directory, records its hash,
command, installation receipt, versions, BLAS configuration and start time, then
executes that copy under `python -I`. It rechecks package/script identities afterward.
The unchanged example binds both completed results, plots and exports within one CLI
invocation; no mutable notebook execution/plot state is involved. Its original
manifest is verified. Exceptions/timeouts preserve available evidence and create an
archive. Never silently rerun or tune parameters after failure.

Expected local nonlinear runtime: 13.6–15 s, fixed comparison ~0.18 s; measured
whole-process peak ~292 MiB. Allow slower hosted execution, imports, plots and
postprocessing: plan **at least 1 GiB available RAM**, roughly several minutes total,
plus installation/download time. This is guidance, not measured hosted performance.
The original example retains its **120 s between-slice timeout per calculation**;
the outer process cap is 300 s. Timeout is a failed operational qualification, not
permission to increase solver iterations, change gates or rerun. Linux child MaxRSS
is captured (KiB, includes other child processes such as pip; not a sampled isolated
solver-memory measurement). Reference arrays add ~50 MiB uncompressed; NumPy
comparison temporaries add overhead. No FFT/director solve occurs during comparison.

## 4. Inspect and download, including failures

```python
import hashlib, json
from IPython.display import display, Image
if (out/'result/comparison.png').exists():
    display(Image(filename=str(out/'result/comparison.png')))
if (out/'comparison.json').exists():
    comparison = json.loads((out/'comparison.json').read_text())
    print('Numerical screen:', comparison['provisional_numerical_screen_pass'])
    print('Diagnostic differences:', comparison['diagnostic_differences'])
print((out/'stdout.txt').read_text() if (out/'stdout.txt').exists() else '')
print((out/'stderr.txt').read_text() if (out/'stderr.txt').exists() else '')
for name,digest in json.loads((out/'qualification-manifest.json').read_text()).items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest() == digest
archive = Path(str(out)+'.zip')
print('Download SHA-256:',hashlib.sha256(archive.read_bytes()).hexdigest())
from google.colab import files
files.download(str(archive))
```

Keep the download digest and note whether all four panels render: endpoint intensity
(z=500 µm), accepted director slice (midpoint z=497.5 µm), reorientation relative to
bias, and both x/y width curves. Check readable physical axes, radian director units,
normalized intensity units and legend. Pixel identity across Matplotlib versions is
not required. Download even failed runs; do not rerun automatically. Later local
review independently verifies ZIP/manifests and records manual plotting success.

## Numerical and scientific acceptance contract (fixed before hosted results)

- Exact input wheel/example hashes, 264 installed file hashes, saved request bytes,
  array shapes/dtypes and x/y/z coordinate bytes required. Retain wheel receipt and
  executed script, not just an engine version string.
- Compare all 18 arrays. SHA-256 over stored-dtype C-order bytes, shape/dtype checked
  separately; report exact identity, differing element count, max/RMS absolute error,
  relative L2, max relative error where |reference| >= max(1e-300, 1e-12*peak).
  No phase alignment, rescaling, clipping or rounding. Zero-reference relative L2
  is null and absolute-error screening remains active.
- A **provisional cross-platform consistency screen**, not an established LC
  backend tolerance: max absolute error <= 1e-12 + 1e-9*reference peak AND relative
  L2 <=1e-9 for nonzero arrays. Scientific scalar/history floats use
  |difference| <=1e-12+1e-9*|reference|. Discrete history, iteration counts, flags,
  termination reasons and warnings must match for this screen. All differences
  are recorded; changed iteration counts require review even if convergence passes.
  These bounds permit accumulated floating-point error far above the ~1e-15
  differences observed in the earlier fixed-LC Colab run, yet are far below the
  0.005/0.02 director residual gates and observed ~0.08 rad response. They are a
  conservative preparation-stage comparison criterion, **not prior evidence of
  nonlinear cross-platform accuracy**. Outside bounds: stop for diagnostic review,
  do not loosen thresholds. Inside bounds still requires provenance/plot/gate review.
- Both runs must complete 100/100 slices. All nonlinear slices must converge with
  original RMS <=0.005 and maximum <=0.02. Fixed-director convergence is not
  applicable. Require finite arrays and consistent diagnostics. Do not reinterpret
  completed execution as convergence.
- Compare full 6,275-record reference relaxation history and 100 slice summaries,
  full beam-width curves, ~0.08002581125 rad maximum reorientation, and all normalized
  and physical power diagnostics. Recompute widths/reorientation and initial/final
  normalized powers from saved arrays; verify consistency with exported summaries.
- Normalized power (~0.9997691121920113 ->0.9997691121920986) is dimensionless.
  Physical angular-flux power (~0.9997230027 ->0.9949732609 mW) is separate;
  no relabeling, rescaling or invented physical-power conservation gate.
- Preserve the original periodic-boundary launch warning. The full width curves
  demonstrate transient narrowing followed by broadening, **not stationary soliton
  formation**, spatial convergence or stability.

The reference is the retained Stage 4 installed-example output, backed by exact
same-environment canonical-workflow tests. No second independently retained canonical
archive or missing internal torque-source volume is invented. Selected NPZ exports
are diagnostic artifacts, not complete transport archives or restart checkpoints.

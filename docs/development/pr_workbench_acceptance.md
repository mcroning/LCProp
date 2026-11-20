# Reduced PR workbench: frontend acceptance record

Date: 2026-11-20. Scope: additive reduced x-only PR Static/TD workbench and
unpublished source-preview notebook. Base: public main
`2873af15818e7f4e675bcc98b124d9547332008d`.

This records UI acceptance, not a new numerical or GPU qualification. Existing
published notebooks and release downloads are not changed by this branch.

## Tested artifacts

| Artifact | SHA-256 |
| --- | --- |
| Development wheel, `lcprop-0.5.0rc3-py3-none-any.whl` | `37830db9ae46e16ae761e5fb41d411c27c472b9e67651763505c749c640f1007` |
| `LCProp-PR-workbench-setup.zip` | `0c5866e044e50b142581c44f981efd601e741617bc94825103fd2ffd1706045d` |
| Clean notebook, `notebooks/pr_workbench_source.ipynb` | `d4c2a841eefbfcc048389e0d5a744a45a61f310bea3416345280cc4fc4a242c3` |

The development wheel retains the source version string; it is **not** the
published rc3 wheel. Its runtime package matches this branch's runtime source.
Later opening-cell corrections changed only the notebook helper and tests, so
those corrections do not change that wheel's identity. The setup ZIP contains
that exact wheel and a short readme. It has not been published as a release asset.
The source-preview notebook needs this separately supplied kit. Developers may
instead install the source checkout as described in the workbench guide.

## Local verification

The workbench and opening-flow tests prohibit real runner/launch execution and
use mocked devices, fake runners, and synthetic saved products. They cover:

- Reduced Static/TD draft switching, CPU default and unchanged GPU guards.
- Independently sampled editing preview versus strict scientific admission.
- Progress, cooperative Stop, failure and retained prior-result behavior.
- Same-controller redisplay, invalid text preservation and result identity labels.
- XY/XZ/YZ field views, aspect selection, file limits, and exact nine-member
  synthetic completed-ZIP reopen/re-export hashes.
- Setup-kit/wheel checksum rejection, live-controller preservation, interrupted
  dependency installation, explicit restart guidance and renderer compatibility.
- Lazy optional imports, source packaging and unchanged default LC/PR layouts.

Relevant commands (with notebook/test extras available in an isolated environment):

```sh
PYTHONPATH=src:. MPLBACKEND=Agg python -m pytest -q \
  tests/test_pr_workbench.py tests/test_pr_workbench_open.py \
  tests/test_notebook_first_batch.py
PYTHONPATH=src:. MPLBACKEND=module://ipympl.backend_nbagg python -m pytest -q \
  tests/test_pr_workbench.py
```

Before the opening corrections, the combined suite passed 94 checks with five
existing optional LaunchPlane interoperability skips; the widget backend passed
17 checks. After the retry and renderer fixes, the 25 affected checks passed.
The final combined suite passed **96 checks with five existing optional skips**.

## Actual Colab CPU checks

A new disposable Python 3.13.18 CPU runtime received the single setup ZIP. The
Open cell verified the kit and installed the exact wheel and notebook dependencies.
After the requested explicit session restart, the same cell opened the workbench.
The final tested renderer was Matplotlib 3.11.2 with ipympl 0.10.0 and ipywidgets
7.7.1. No CUDA package, Drive mount or account-access expansion was required.

Observed: visible beam and advanced controls, CPU selection, included synthetic
example loading, field/cut/aspect selection, and compact plots within the existing
638-pixel browser panel. Vertical page scrolling remains normal. Reloading the
browser and rerunning Open preserved the same live process/controller, a committed
beam edit, and the loaded synthetic result. Installed runtime-package bytes were
checked against the exact wheel. No propagation was dispatched (`last_run` stayed
unset). The temporary diagnostic cell was removed after evidence was saved; the
clean notebook has exactly one code cell and no saved output.

## Failures that informed the opening fixes

An interrupted setup installed LCProp before its rendering dependencies. The retry
originally fell back to manual installation; the final opening helper resumes the
missing dependencies in the same cell, without replacing an already exact engine.

The stock Matplotlib 3.10.0 backend registry rejected the installed ipympl backend
in a restarted kernel. The opening helper now pins Matplotlib 3.11.2, which passed
both the local widget tests and the hosted opening/reconnect checks. These changes
are confined to frontend setup. The failed attempts and successful receipts remain
in the owner's local project evidence; they are not represented as earlier passes.

## Preserved boundaries

No scientific source, numerical limits, desktop installation, release assets,
release notebook pins or public README links changed. Public admission stays at
256 MiB per result, 2.5 GiB working plus resident results, 512 MiB reserve, a
60-second cooperative PR budget, and a 128 MiB ZIP ceiling. Full-transverse PR,
private large-result recovery helpers and expanded GPU support are excluded.
This record does not authorize propagation, merging or publication.

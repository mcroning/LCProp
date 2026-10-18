# Stage 3H — Restart-free Colab installation audit

2026-10-17. **NO-GO for removing the required restart from the qualified hosted workflow.** A restart-free sequence is conditional on backend discovery state as well as unchanged dependencies. That condition has not been established in a fresh hosted Colab runtime. No notebook, controller, package, dependency or artifact was changed; no installation, upgrade, propagation, commit or publication occurred.

## Evidence and causal limits

The original user traceback occurred during `%matplotlib widget`, before controller construction: Matplotlib rejected `module://ipympl.backend_nbagg`. A session restart followed by verification and initialization succeeded. The user subsequently accepted two-beam editing and the corrected non-accordion presentation. This establishes an actual pre-controller backend-selection failure, not merely a conservative instruction that was never tested.

The post-restart hosted record showed Python 3.13.6, NumPy 2.1.3, SciPy 1.16.3, Matplotlib 3.10.0, ipywidgets 7.7.1, IPython 7.34.0 and ipympl 0.10.0. No pre-install module/version inventory, pip transaction report or backend-cache state was retained from the failing session. Therefore actual dependency replacement cannot be established or excluded from the screenshot. The exact historical cause is not proven. The earlier notebook wording that restart “clears Colab backend registration state” should be understood as a supported hypothesis, not a measured diagnosis of that session.

An upstream Colab report shows the same exception and successful restart workaround. It supports the symptom match, not proof that every current Colab failure has the same cause: https://github.com/googlecolab/colabtools/issues/5155 .

## Installation and import audit

The current notebook imports only standard-library modules before its installation command: os, sys, json, hashlib, tempfile, subprocess, zipfile, urllib.request/error, pathlib and importlib.metadata. Metadata queries do not import the corresponding dependency packages. IPython is necessarily already active as the kernel shell; other Colab-preloaded modules cannot be enumerated retrospectively from this repository.

Before pip, the cell samples installed distribution versions for six package names present in sys.modules: numpy, scipy, matplotlib, ipywidgets, IPython and ipympl. After pip, a changed sampled version triggers a restart instruction. This is a reasonable safety guard but incomplete evidence: it compares disk metadata, not the loaded module's own version; it omits transitive packages and does not detect a newly installed backend missing from an existing registry. A successful setup cell does not prove that no imported dependency changed. A prior install in the same session could also have already made disk and loaded versions inconsistent.

After verified installation, the cell imports lcprop and its deployment verifier. Initialization imports InteractiveLC, enables Colab's custom widget manager, selects the widget backend, then constructs the controller. The InteractiveLC module imports numerical/request helpers; it does not explicitly import Matplotlib or ipympl at module scope. Changing controller import order cannot undo earlier Colab backend discovery.

The current pip command has no --upgrade or --force-reinstall. It uses the exact downloaded, hash-verified wheel with [colab], selects production PyPI, removes inherited PIP_* variables and disables pip config. Compatible installed dependencies are normally retained; incompatible requirements can still force replacement. For a future controlled experiment, exact constraints for already installed packages and a pip dry-run/report could reject any proposed replacement before installation. Constraints would preserve compatible versions or fail resolution, not override requirements. No such resolver experiment was executed here. `--no-deps` alone is not a safe substitute for dependency validation. See https://pip.pypa.io/en/stable/cli/pip_install/ .

## Backend discovery mechanism

Matplotlib 3.10.0 maps the explicit ipympl module name to `widget`, reads backend entry points once and retains that registry. If discovery happens before ipympl is installed, the same registry can reject the backend afterward. Import-cache invalidation does not refresh this registry. Its reset method is explicitly private/testing-only, not a supported notebook fix. Source: https://raw.githubusercontent.com/matplotlib/matplotlib/v3.10.0/lib/matplotlib/backends/registry.py .

Installed ipympl 0.10.0's __init__.py calls matplotlib.use with that backend when Matplotlib is already imported. Simply importing ipympl earlier after installation can therefore encounter the same rejection. Changing `%matplotlib widget` to `%matplotlib ipympl` does not reliably bypass discovery. Colab's custom widget manager enables frontend widget support; it is not a Matplotlib backend-cache refresh API. Official initialization and compatibility guidance: https://matplotlib.org/ipympl/installing.html .

No private registry mutation, module reload, alias trick, custom JavaScript or automatic restart is recommended.

## Bounded local validation

Using the existing qualified disposable environment (Matplotlib 3.11.2, ipympl 0.10.0), a test-only isolated registry with mocked metadata discovery reproduced the mechanism without installing or replacing anything:

| State | Backend accepted |
| --- | --- |
| Entry points discovered while ipympl metadata is absent | False |
| Actual installed entry points subsequently available, same registry | False |
| After importlib.invalidate_caches(), same registry | False |
| Fresh registry with installed entry points | True |

The fixture mocks public metadata reads; it does not change a live frontend's private backend state. This is a deterministic model of installation timing, not an actual Colab install. The relevant mechanism also exists in the reviewed 3.10.0 source. A separate fresh Python process with existing dependencies and IPython likewise accepted the backend. This proves only preinstalled backend discovery, not fresh-hosted install-and-display success.

Retained local evidence: `.codex-work/beam-stage3h/probe.py` and `probe.json`. No browser was available to this audit; no fresh hosted runtime was launched. The full controller/frontend was not exercised by these registry probes.

## Minimal conditional restart-free experiment

This is a proposed qualification procedure, not a replacement for accepted setup:

1. In a genuinely fresh hosted CPU runtime, capture sys.modules membership plus loaded and installed versions before setup. Avoid invoking backend-list discovery merely to diagnose it before installation; that can itself prime the cache.
2. Retain exact TestPyPI release inventory, both advertised hashes, independent wheel hash and expected version. Preserve compatible preloaded dependencies; stop if dependency resolution requires replacing imported packages. Retain a pip transaction report in a future authorized trial.
3. Install the verified wheel with its optional dependencies from production PyPI, before notebook-initiated Matplotlib/backend discovery. Colab may already have performed discovery during kernel startup; notebook ordering cannot guarantee otherwise.
4. Retain all 269 installed runtime file checks, import ownership, distribution verifier and pip check. Compare captured loaded/disk dependency identities.
5. Enable the custom widget manager, select `%matplotlib widget`, then initialize the existing controller and accepted presentation. On backend rejection, stop the restart-free trial; retain the established manual-restart/static fallback rather than altering registry internals.
6. Confirm actual browser painting, first Beam 2 activation, repeated toggles, Move/Tilt, numeric edits and repeated initialization without propagation. Backend acceptance alone is insufficient.

The smallest additional evidence is one fresh hosted attempt with pre/post dependency identities, installation transaction and exact initialization outcome. If Colab preloads a registry lacking the newly installed backend, no supported refresh operation was identified in the reviewed API. A compatible image with ipympl already installed before kernel startup could avoid that condition, but this notebook does not control Colab's base image.

## Decision

**NO-GO for a production notebook correction that promises restart-free installation.** Dependency preservation is desirable and already partly provided by ordinary pip behavior, but does not eliminate cached backend discovery. The exact original dependency transaction is unknown, and a safe restart-free hosted frontend path has not been demonstrated. Keep the accepted initial-install → manual session restart → verification rerun → initialization procedure. Preserve all artifact/scientific integrity gates. No additional diagnostic system or speculative patch was introduced.

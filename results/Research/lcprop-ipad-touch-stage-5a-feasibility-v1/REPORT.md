# Stage 5A — iPad touch-event feasibility preparation

**GO for the bounded diagnostic; NO-GO for touch-support implementation pending hosted evidence.** 2026-10-18.

Notebook: `notebooks/lcprop_ipad_touch_stage5a_diagnostic.ipynb`.
Original prepared SHA-256: `f535be5137b4748c0d8b7e1fb4aee76d9e16545adb3a198c7170646465a1a4d2`.
Publication notebook SHA-256 (setup instructions clarified; diagnostic code unchanged): `08dfb1e30e870fb18bd736695a9ea1409f5ef2528cd26e540bcb283f7a8a6452`.

## Scope and setup

Separate, output-free notebook with a disposable synthetic Matplotlib/ipympl canvas. It never imports LCProp, invokes science, installs dependencies, upgrades packages or changes a working controller's backend. Use a hosted CPU session already configured with working ipympl using the separately qualified setup. Upload this notebook manually through Colab's Open notebook → Upload, or copy its cells into a separate section of that compatible session. An existing runtime must already have the required backend; the notebook deliberately stops if it is not active. It has not been published.

It records Python and installed/loaded package versions, active backend, canvas class and requested widget module/version constraints. These do not prove the frontend bundle actually loaded in the browser. Browser/module evidence must be supplied separately using supported inspection tools; unknown versions remain marked UNVERIFIED. iPad-only sessions without inspection cannot close this identity gate by inferring versions from Python.

## Experiment and framing boundary

The widget is assigned a random unique class through supported DOMWidget.add_class. A sibling ipywidgets.HTML holds a style rule targeting only canvas descendants of that unique widget: touch-action:none plus a magenta outline. Baseline removes the rule. There are no global scroll rules, custom JavaScript, frame traversal, querySelector calls, private ipympl selectors, DOM reparenting or frontend patches. Styling is an experiment: Colab may sanitize the style element or isolate its document. We do not assume success merely because the Python HTML value is set.

If the outline fails to appear, stop the CSS experiment and report possible sanitization/framing limitation. Do not substitute DOM manipulation. If present, the outline shows scoped rule delivery but not the computed touch-action value; verify that separately where supported browser inspection is available. Require notebook scrolling outside this canvas to remain normal. No claim that CSS alone creates touch-to-mouse motion events is made.

## Logging and test matrix

Start/End trial controls label independent gestures. Log monotonic arrival order, press/motion/release/figure-enter/figure-leave names, display coordinates, data coordinates, in-axes flag and button state. Mode cannot change during an active trial. Missing coordinates remain null rather than fabricated. Logs distinguish callbacks from browser observations: no Python pointercancel/lostpointercapture callback exists in this protocol. Missing release is an incomplete sequence, not proof of a particular browser cancellation event.

Test baseline then scoped CSS, then baseline again. Three gestures per input/mode are suggested, with finger, Pencil and mouse in Safari and Chrome. Include leaving axes/canvas, release outside, second contact, interruption, subsequent recovery, and notebook scrolling outside. Inspect exact sequence and duplicate press/release pairs, not only totals. Delayed compatibility events may require browser evidence to distinguish. No adapter is installed. Retain evidence.json printed/saved by the last evidence cell through Colab Files → Download; fill observation fields before saving.

| Observation | Result |
| --- | --- |
| iPad finger baseline event sequence | Pending user-hosted run |
| iPad Pencil baseline event sequence | Pending |
| Scoped CSS delivery/computed touch-action | Pending |
| CSS finger/Pencil ordered press-motion-release | Pending |
| Outside scroll unaffected | Pending |
| Safari/Chrome actual versions and loaded frontend | Pending |
| Leaving/interruption/cancellation/recovery | Pending |
| Real mouse/duplicate-event comparison | Pending |

No hosted connection or iPad event trace was available to this preparation. Stage 5 user's reported scrolling behavior is background evidence, not a recorded Stage 5A event sequence.

## Local validation

All code cells compile; notebook has empty outputs and no execution counts. Existing Stage 4F environment used without installing anything: Python 3.12.15, macOS arm64, Matplotlib 3.11.2, ipympl 0.10.0, ipywidgets 8.1.9, IPython 9.17.1. Mocked only Colab's widget-manager setup and executed setup/display cells against a real local ipympl canvas. Injected synthetic Matplotlib left press, motion and release: exactly one of each logged in order, all with valid in-axes coordinates. Unique CSS rule/toggle and baseline removal passed; listener cleanup passed. This validates logging and Python setup, not frontend CSS application, browser mouse events or iPad input. No scientific execution.

## Decision and smallest next implementation

Do not implement or claim touch support until hosted evidence establishes the transport path. If scoped CSS yields reliable events with the correct frontend identity, assess a narrowly scoped style-only correction, including terminal events, scroll/accessibility and desktop regression gates. If style cannot reach the interaction surface through this supported widget presentation, stop and record that limitation.

If CSS arrives but drags remain incomplete, the smallest next candidate is a separately reviewed pointer-event input adapter at the frontend boundary: primary finger/Pencil coordinates mapped to the existing left-button event protocol, canvas-local gesture policy, capture and cancellation, correct pixel scaling, throttled movement with guaranteed terminal delivery, and compatibility-mouse deduplication. Preserve Active Beam/Move-Tilt semantics, existing controller validation and desktop mouse path. No engine/request change is needed. Such adapter work is not part of this stage and no implementation has been introduced.

Only the separate diagnostic notebook and this report were created in the repository. Existing Product files, published distribution, qualification notebooks, unrelated modifications and protected evidence remain untouched. No commits, pushes or publication.

## Publication preparation addendum — 2026-10-19

Added Markdown-only instructions for opening this notebook in its own Colab runtime, linking the immutable production 0.4.0 guide and spelling out the manual install/restart/backend initialization prerequisites. All diagnostic code cells remain unchanged. No local installation or backend change to an LCProp controller was performed. Earlier unpublished/manual-upload statements describe the original preparation; publication status is recorded separately in Stage 5A publication report. Compact local validation is retained in `local-validation.json`.

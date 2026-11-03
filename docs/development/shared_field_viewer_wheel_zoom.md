# Shared field viewer wheel zoom

The shared `ImageView` normalizes Qt wheel input before Matplotlib dispatch.
Previously its scroll callback used only event direction, making small deltas
as large as a full notch. The inspected Matplotlib Qt backend also chooses
pixel deltas on some platforms and angle deltas on others; its `step` is
therefore not a portable wheel-notch unit without this boundary normalization.

The contract is:

- `steps = QWheelEvent.angleDelta().y() / 120.0`.
- Visible x/y spans multiply by `0.9 ** steps` about the cursor.
- +120 reduces each visible span by 10%; -120 multiplies it by 1/0.9.
- Fractional steps are not rounded or accumulated to a notch. Four +30 events
  compose to one +120 event. Opposite deltas are reciprocal, apart from normal
  floating-point error and the existing full-extent boundary clamp.
- Qt's supplied sign is honored; `inverted()` is not applied a second time.
- Horizontal-only and zero-angle events do not zoom, including zero-delta
  gesture begin/end events. Events outside the image axes are ignored.
- When both angle and pixel deltas exist, angle wins. Screen pixels have no
  portable conversion to wheel notches; pixel-only events are ignored rather
  than assigned an invented conversion constant. High-resolution **angular**
  deltas, including 1/120 notch, remain supported immediately.

This follows [Qt's QWheelEvent angle units and fractional-delta contract](https://doc.qt.io/qt-6/qwheelevent.html).
There are no OS-specific constants or platform checks. Existing cursor
anchoring, extent clamp, pan, reset, scientific arrays and display coordinates
are unchanged. Synthetic Qt events exercise the actual QWidget event path,
including angle+pixel input, both signs, fractional composition, inversion
metadata, phase-only events and events outside the axes.

## Hardware commissioning still required

Offscreen synthetic tests validate LCProp's mapping, not OS input drivers.
On real Windows and Linux (both Wayland and X11 where supported), check a
conventional wheel and a high-resolution wheel/touchpad: delivered angular
increments, natural-scroll direction, momentum/gesture phases, event routing,
DPI/multi-monitor cursor anchoring, and interaction with surrounding scroll
areas. Also check Qt touchpad input on macOS. Record Qt/platform/device versions
and raw angle/pixel deltas if an issue occurs. A device delivering only pixel
deltas would need a separately justified non-angular interaction contract;
this change does not claim commissioning of that path. Physical feel, driver
acceleration/coalescing and platform event delivery cannot be established by
synthetic tests alone.

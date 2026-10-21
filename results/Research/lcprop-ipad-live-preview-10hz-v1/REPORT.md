# iPad 10 Hz live-preview trial

Experimental continuation of published 5 Hz trial `93eae01e96c4dcac7244b28ac6d531a28fd7ba04`.

User reports beam intensity follows for a few seconds then decouples, while interaction remains laggy. The retained editor's 20-update cap explains a plausible approximately four-second cutoff at 5 Hz; no browser trace was supplied, so the cause is not proven. The unchanged 100 ms slow-computation fallback can also stop provisional updates.

This notebook changes only the disposable editor instance to `LIVE_INTERVAL=0.1` and `MAX_LIVE_UPDATES=100`. Thus the requested maximum preview rate is 10 Hz, with a bounded roughly ten-second-or-longer allowance per gesture. Raising rate without raising the cap would shorten the previous allowance. Pointer motion remains at 25 Hz, hit tolerance at 30 pixels. No timers or request serialization were added. Setup, exact development wheel, synthetic adapter and safety gates are identical to the prior trial. Run remains disconnected and disabled. The 100 ms slow-computation fallback and exact canonical release refresh remain unchanged.

Wheel SHA-256: `51520c31974ecfd6c0a911a4000ba4c2c6786456e14cd4168a737afac81e66a1`, immutable artifact commit `9aee474c300e3900ebca3e908965a026b929450b`. No rebuild or dependency changes.

Four local synthetic tests (Move/Tilt, one/two beams) used 150 events over a six-second synthetic 25 Hz schedule. Each continued past 20 live updates, retained frozen numeric requests during motion, committed exact final coordinates and canonical intensity on release, and restored the committed request/image on cancellation. Runtime identities were verified against all 269 accepted wheel entries. Full measurements retained separately. No propagation. Controlled scheduling and local PNG/raster checks do not measure Colab lag or browser corruption. Prior 63 regression passes remain retained in the 5 Hz report; they were not rerun for these two diagnostic instance settings.

GO for one bounded hosted experiment only; not a performance guarantee or production recommendation. Run Setup/restart/verify, synthetic validation, then try short finger/Pencil Move/Tilt gestures. Observe whether live intensity continues beyond four seconds and whether lag is acceptable. Stop if lag worsens or images corrupt; retain accepted release-only behavior in that case. No uploads, diagnostic downloads or event-log transfers.

The numeric rate is an upper bound: at 25 Hz pointer delivery, a 0.1-second deadline may sample every third event (about 8.3 Hz), and computation/transport can lower it further. This is deliberately no faster than 10 Hz. The cap still intentionally stops provisional updates on very long gestures; release refresh remains exact.

# LCProp Architecture Overview

The authoritative LCProp architecture and package-boundary contract is
[`LCProp_Architecture_Blueprint_v1.1.md`](LCProp_Architecture_Blueprint_v1.1.md).
Read that document for the core data model, runtime and workflow boundaries,
LaunchPane integration contract, validation strategy, and planned extension
constraints.

LCProp is organized around one stable flow:

```text
user interface or script
    -> immutable LCProp request
    -> runner / workflow
    -> physics and numerical engines
    -> LCProp result
    -> products, diagnostics, and display

planned extension: request/result persistence
```

Document roles:

- [`LCProp_Architecture_Blueprint_v1.1.md`](LCProp_Architecture_Blueprint_v1.1.md)
  is the architecture source of truth.
- [`STATUS.md`](STATUS.md) is the concise current implementation checkpoint.
- [`development_plan.md`](development_plan.md) contains forward-looking
  implementation order and milestones.

This overview intentionally does not duplicate the detailed architecture.

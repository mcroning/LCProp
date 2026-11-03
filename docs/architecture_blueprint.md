# LCProp Architecture Overview

The canonical architecture and package-boundary contract is
[`architecture/LCProp_Target_Architecture.md`](architecture/LCProp_Target_Architecture.md).

LCProp is organized around material-owned physics composed with shared
optical, execution, persistence-dispatch, and presentation infrastructure:

```text
material-owned request and source construction
    -> material-owned state solve and prepared optical response
    -> shared optical propagation
    -> material-owned result and product adapter
    -> shared execution and presentation services
```

Liquid crystal and photorefractive implementations are peer material packages.
They share explicit interfaces, not a universal material base class or dynamic
plugin framework.

Document roles:

- [`architecture/LCProp_Target_Architecture.md`](architecture/LCProp_Target_Architecture.md)
  is the architecture source of truth.
- [`STATUS.md`](STATUS.md) is the concise current implementation checkpoint.
- [`development_plan.md`](development_plan.md) contains forward-looking public
  package work.
- [`LCProp_Architecture_Blueprint_v1.1.md`](LCProp_Architecture_Blueprint_v1.1.md)
  is a historical July 2026 LC-centric baseline and is not current guidance.

This overview intentionally does not duplicate the complete architecture
decision record.

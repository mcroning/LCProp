# LCProp Documentation

This index separates current package documentation from historical design
records. Public documentation describes the software and reusable development
procedures. Private experiment logs, raw scheduler output, unpublished
references, and event-specific prompts are maintained outside this repository.

## Start here

- [`../README.md`](../README.md) — package overview, installation, examples,
  testing, and contribution guidance.
- [`STATUS.md`](STATUS.md) — current implementation status and supported
  boundaries.
- [`architecture/LCProp_Target_Architecture.md`](architecture/LCProp_Target_Architecture.md)
  — canonical architecture decision record.
- [`development_plan.md`](development_plan.md) — forward-looking public package
  work.

## Architecture

- [`architecture/LCProp_Target_Architecture.md`](architecture/LCProp_Target_Architecture.md)
  — source of truth for peer material ownership and dependency direction.
- [`architecture_blueprint.md`](architecture_blueprint.md) — concise
  architecture index.
- [`architecture/material_plugin_architectural_rationale.md`](architecture/material_plugin_architectural_rationale.md)
  — rationale for the in-tree material-composition design.
- [`architecture/material_plugin_review.md`](architecture/material_plugin_review.md)
  — historical review that preceded the completed ownership migration.
- [`architecture/pr_transverse_reference_model.md`](architecture/pr_transverse_reference_model.md)
  — bounded PR transverse-reference architecture.

The older [`LCProp_Architecture_Blueprint_v1.1.md`](LCProp_Architecture_Blueprint_v1.1.md)
records the July 2026 LC-centric baseline and is retained only as historical
context.

## Material documentation

- [`../src/lcprop/pr/README.md`](../src/lcprop/pr/README.md) — PR state,
  normalization, boundary conditions, numerical methods, and workflows.
- [`architecture/pr_gui_vertical_slice_design.md`](architecture/pr_gui_vertical_slice_design.md)
  — PR GUI design record.
- [`architecture/pr_second_order_numerics_design.md`](architecture/pr_second_order_numerics_design.md)
  — second-order PR integration design.
- [`architecture/pr_partition_independent_scattering_design.md`](architecture/pr_partition_independent_scattering_design.md)
  — canonical partition-independent scattering design.
- [`pr_image_amplification_validation.md`](pr_image_amplification_validation.md)
  — public validation contract and bounded benchmark summary.

## Developer operations

- [`codex/README.md`](codex/README.md) — reusable prompt-library procedures,
  lifecycle gates, and reproducibility policy.
- [`codex/TEMPLATE.md`](codex/TEMPLATE.md) — authoritative prompt template.

Historical implementation briefs and research notebooks are not canonical
package documentation. When such a record is retained publicly, its role and
status are stated explicitly in the document or this index.

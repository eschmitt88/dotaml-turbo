---
name: data-rules
description: Applies when handling data and large artifacts.
applies_to: ["raw/**", "data/**", "**/*.parquet", "**/*.arrow", "**/*.ckpt"]
---

# Rules for data

- `raw/` is **immutable**. Never modify a file under `raw/` after ingest.
  If the source changed, re-ingest as a new dated file.
- Large artifacts (>10 MB) are tracked by DVC, not Git. They live on the
  SN850X under `~/projects/`.
- The HuggingFace cache and any model checkpoints belong under
  `~/projects/` (SN850X), never in `~/`.
- Processed data derived from `raw/` is reproducible from a DVC stage.
  Document the stage in `dvc.yaml`.
- Do not check in credentials, API keys, or personal data with the raw
  material. Redact before ingest.

# Architecture

The package has one dependency direction:

```text
config + contracts
        ↓
data → models → objectives
        ↓         ↓
      training ← tuning
        ↓
tracking → inference → exploration
        ↓
       CLI
```

## Boundaries

- `config` owns immutable, strict Pydantic schemas and YAML loading.
- `data` owns raw schema validation, temporal windows, fitted transformations,
  typed Parquet shards, and manifests.
- `models` only map `FeatureBatch` to `ModelOutput`.
- `objectives` own all supervision, including aggregate, auxiliary, diversity,
  and recursive ensemble terms.
- `training` is the only optimization loop. Tuning invokes it directly.
- `tracking` owns run lineage, metrics, artifacts, and promotion records.
- `inference` only loads validated bundles.
- `exploration` emits JSON and self-contained HTML from public contracts.

No module reads a global configuration. The CLI composes these boundaries and
does not contain feature or model logic.

## Model contract

`FeatureBatch` has separate `int64` categorical and `float32` numerical lanes.
`ModelOutput.aggregate_logits` is always the deployed prediction. Multihead and
ensemble structure is explicit through auxiliary and child outputs.

The primary objective is BCE on `aggregate_logits`. Auxiliary head BCE and
positive residual-correlation penalties cannot replace aggregate supervision.

## Artifact contract

A processed dataset is valid only with its manifest and checksums. An inference
model is valid only as a `safetensors` file plus `bundle.json` and its fitted
preprocessor state. Artifacts must match their declared schemas.

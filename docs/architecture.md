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
- `data` owns raw schema validation, temporal windows, typed feature recipes,
  fitted transformations, typed Parquet shards, and manifests.
- `models` only map `FeatureBatch` to `ModelOutput`.
- `objectives` own all supervision, including aggregate, auxiliary, diversity,
  and recursive ensemble terms.
- `training` owns the only batch-level optimization loop. Candidate training
  adds validation and early stopping; production refit supplies a fixed epoch
  budget and no validation loader.
- `tracking` owns run lineage, metrics, typed selection evidence, selection
  decisions, deployment records, and atomic replacement.
- `inference` owns validated production bundles and device-specific prediction
  execution.
- `exploration` emits JSON and self-contained HTML from public contracts.

No module reads a global configuration. The CLI composes these boundaries and
does not contain feature or model logic.

## Model contract

`FeatureBatch` has separate `int64` categorical and `float32` numerical lanes.
Labels and row IDs are optional; training omits row IDs, while evaluation and
inference request them explicitly. The batch owns its tensor pinning contract,
so a CUDA `DataLoader` can transfer every tensor lane asynchronously.
`ModelOutput.aggregate_logits` is always the deployed prediction. Multihead and
ensemble structure is explicit through auxiliary and child outputs.
CUDA inference compiles the complete model-to-probability graph, runs model
operations under float16 autocast, and returns float32 probabilities. CPU
inference remains eager float32.

The primary objective is BCE on `aggregate_logits`. Auxiliary head BCE and
positive residual-correlation penalties cannot replace aggregate supervision.

### Model families

- `dcn` uses a shared field encoder and SENet backbone with an explicitly
  selected cross network. `dcnv2` is the standard cumulative recurrence.
  `delta_routed_dcnv2` lets each cross layer from the third onward attend over
  the distinct interaction-order increments produced by earlier layers. Its
  softmax weights are scaled to conserve total residual mass and centered
  around the standard recurrence, so zero-initialized routing queries make the
  routed network exactly equal to DCNv2 at initialization. New increments still
  enter the ordinary cumulative state; routing changes only the context used to
  generate the next increment. One or more prediction heads operate on the
  shared backbone; gated or mean aggregation applies only when more than one
  prediction head is configured.
- `stec` computes the full per-dimension query/key Hadamard interaction before
  averaging it into scaled dot-product attention. Every STEC layer exposes that
  grouped bilinear tensor, a final standalone bilinear layer exposes the last
  hidden state, and the resulting `N + 1` tensors are batch-normalized,
  concatenated, and supervised through the deployed logit. BatchNorm keeps
  cumulative training-batch estimates so deployed predictions are independent
  of inference batch partitioning and are not dominated by the last shard tail.
- `ngpt` treats the ordered field embeddings plus a learned classification token
  as a non-causal sequence. It uses normalized query/key attention, learned
  scaling vectors, SwiGLU, and hyperspherical LERP updates without LayerNorm or
  RMSNorm. Its deployed sigmoid logit is the difference between two scaled,
  normalized class-prototype logits. All embedding and matrix vectors are
  normalized in-place at initialization and after every successful optimizer
  step. Its optimizer contract requires Adam, zero weight decay, and zero
  warmup.

STEC and nGPT attention heads partition an embedding space; they are unrelated
to the independently supervised prediction heads in `dcn`.

## Selection and deployment

Selection and deployment are separate state transitions:

```text
screening → walk-forward confirmation → final-holdout evidence
                                              ↓
                                      active selection
                                              ↓
all-labelled preprocessing → fixed-budget refit → atomic deployment
```

Promotion selects configuration and evidence. It never promotes a validation
checkpoint. The final holdout's best zero-based epoch becomes the production
budget `best_epoch + 1`; the production scheduler is rebuilt for the resulting
all-data step count.

## Artifact contract

A processed dataset is valid only with its schema-v6 role-specific manifest and
checksums. Evaluation manifests require validation and forbid test. Production
manifests require test and forbid validation.

The manifest embeds the ordered feature definitions, inductive or explicit
competition-transductive mode, fitted-table label and split provenance, and OOV
diagnostics. Vocabularies and label-dependent tables can only declare the
training split; unknown vocabulary IDs have a fixed zero embedding.

Fitted covariate state is key-centric: every join key has one sorted Parquet
lookup containing all frequency and distinct-count outputs for that key.
Compatible lazy fitting queries execute in bounded groups, and Polars writes
the transformed population directly into deterministic, row-bounded Parquet
parts. The runtime loader reads only model lanes, labels, and explicitly
requested IDs, then coalesces rows across shard boundaries. Consequently a
single-process epoch has exactly `ceil(rows / batch_size)` batches; each
additional data worker contributes at most one independent tail batch.

An inference model is valid only as a production `safetensors` file plus
`bundle.json` and its fitted all-data preprocessor state. The bundle records the
selection evidence checksum, refit run, exact epoch/step plan, exact production
manifest, and state checksums.

# Avazu CTR

A correctness-first PyTorch pipeline for temporal click-through-rate modelling on
the Avazu dataset.

The repository covers schema validation, leakage-safe feature fitting, sharded
preprocessing, model training, staged tuning, experiment tracking, TensorBoard,
champion promotion, and deterministic submission generation.

## Historical result

The original repository reported a Kaggle private logloss of **0.38484** and a
public logloss of **0.38671** using a SENet + DCNv2 multihead model. Those values
are preserved as provenance, not regression targets: the historical tuning and
final-training paths used different effective architectures, and the old
validation ordering was not temporal.

Version 2 starts from that model family but retrains it under the corrected
protocol.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA driver compatible with the CUDA 13.0 PyTorch build for GPU training

Install one—and only one—PyTorch backend:

```powershell
# CPU development and CI
uv sync --extra cpu --group dev

# Windows/Linux NVIDIA training
uv sync --extra cu130 --group dev
```

Verify that a CUDA environment really resolved a GPU build:

```powershell
uv run --extra cu130 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

All retained dependencies are resolved to current stable releases in
[`uv.lock`](uv.lock). Weekly dependency and GitHub Actions updates are enabled.

## Workflow

Place Kaggle files at `data/raw/train.gz` and `data/raw/test.gz`. Data and all
generated artifacts are ignored by Git.

```powershell
# Build the final holdout dataset
uv run --extra cu130 avazu-ctr preprocess configs/champion.yaml

# Build all walk-forward folds plus the final holdout
uv run --extra cu130 avazu-ctr preprocess configs/champion.yaml --all-windows

# Train; the first valid run becomes the initial champion
uv run --extra cu130 avazu-ctr train configs/champion.yaml artifacts/datasets/senet-dcnv2-multihead/final_holdout/manifest.json

# Watch live curves
uv run --extra cu130 avazu-ctr tensorboard configs/champion.yaml

# Run staged tuning
uv run --extra cu130 avazu-ctr tune configs/tuning.yaml artifacts/datasets/senet-dcnv2-multihead/walk_forward_0/manifest.json `
  --confirm-manifest artifacts/datasets/senet-dcnv2-multihead/walk_forward_0/manifest.json `
  --confirm-manifest artifacts/datasets/senet-dcnv2-multihead/walk_forward_1/manifest.json `
  --confirm-manifest artifacts/datasets/senet-dcnv2-multihead/walk_forward_2/manifest.json

# Produce the competition submission
uv run --extra cu130 avazu-ctr predict artifacts/champion artifacts/datasets/senet-dcnv2-multihead/final_holdout/manifest.json --device cuda --output submission.csv
```

After an initial champion exists, `train --export-candidate` retains that run's
inference bundle. Evaluate the candidate and incumbent on the same final rows,
then use `avazu-ctr promote` with their `row_losses.npz` files and three
walk-forward losses each. Promotion verifies run IDs, row IDs, and labels before
the paired bootstrap and fold guard. The temporary candidate is deleted after
either acceptance or rejection.

`configs/baseline.yaml` is the clean DCNv2 benchmark.
`configs/champion.yaml` is the corrected SENet + DCNv2 multihead candidate.
`configs/tuning.yaml` defines the staged search.

## Correctness contracts

- Temporal windows are chosen before fitting any data-dependent transform.
- Validation/test covariates cannot affect vocabularies, counts, numerical
  statistics, priors, or target encodings.
- Training target rates use labels only from earlier temporal blocks.
- Categorical values stay `int64`; numerical values stay `float32`.
- Models return the exact aggregate logit deployed by inference.
- The aggregate logit receives direct BCE supervision.
- Hash coefficients and masks are serialized model state.
- Every architecture is tested for complete expected gradient coverage.
- Inference only accepts strict, checksummed `safetensors` bundles.

See [data lineage](docs/data-lineage.md) and
[architecture](docs/architecture.md) for the full contracts.

## Tracking and storage

SQLite is authoritative for run lineage and metrics. TensorBoard mirrors scalar
histories under the same run ID for live curves:

```powershell
uv run --extra cu130 avazu-ctr tensorboard configs/champion.yaml --port 6006
```

TensorBoard stores no weights, graphs, datasets, or embeddings. Optuna trials
store no checkpoints. Production resume state is opt-in, overwritten in place,
and deleted after successful completion. Only the promoted champion's
inference-only weights are retained, with a hard 512 MiB cap.

See [experiment tracking](docs/tracking.md).

## Development

```powershell
uv run --extra cpu ruff format --check .
uv run --extra cpu ruff check .
uv run --extra cpu ty check
uv run --extra cpu pyrefly check --min-severity warn
uv run --extra cpu pytest
uv build
```

Linux CI runs the CPU suite. Before GPU-facing changes are merged, run the same
checks on Windows plus a CUDA forward/backward and AMP smoke run.

## License

MIT. See [LICENSE](LICENSE).

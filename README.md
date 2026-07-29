# Avazu CTR

A PyTorch pipeline for temporal click-through-rate modelling on the Avazu
dataset.

The repository covers schema validation, provenance-tracked feature fitting,
sharded preprocessing, candidate training, staged tuning, experiment tracking,
TensorBoard, full-data production refitting, and deterministic submission
generation.

## Best recorded result

The selected one-epoch SENet + DCNv2 multihead model achieved a Kaggle private
logloss of **0.38476** and a public logloss of **0.38689** in submission
`55049425`.

This was a late submission after the competition deadline, so it is comparable
to the final leaderboard but is not officially ranked. `configs/champion.yaml`
defines the current leakage-safe recipe, while
[`benchmarks/champion.json`](benchmarks/champion.json) keeps its contract
separate from the immutable selection and submission evidence.

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- An NVIDIA driver compatible with the CUDA 13.2 PyTorch build for GPU training

uv automatically downloads and manages the Python 3.12 runtime requested by
`.python-version`; a system Python installation is not required.

Install one—and only one—PyTorch backend:

```powershell
# CPU development and CI
uv sync --extra cpu --group dev

# Windows/Linux NVIDIA training, including the platform Triton compiler
uv sync --extra cu132 --group dev
```

Verify that a CUDA environment really resolved a GPU build:

```powershell
uv run --extra cu132 python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

All retained dependencies are resolved to current stable releases in
[`uv.lock`](uv.lock). Weekly dependency and GitHub Actions updates are enabled.

## Kaggle CLI

Install the locked official Kaggle CLI alongside the selected PyTorch backend:

```powershell
uv sync --extra cu132 --group dev --group kaggle
uv run --locked --extra cu132 --group kaggle kaggle --version
```

Authenticate through Kaggle's browser-based OAuth flow. Credentials are stored
outside the repository:

```powershell
uv run --locked --extra cu132 --group kaggle kaggle auth login
```

Download the competition data, inspect submission history, or submit a generated
file:

```powershell
uv run --locked --extra cu132 --group kaggle kaggle competitions download avazu-ctr-prediction -p data/raw
uv run --locked --extra cu132 --group kaggle kaggle competitions submissions avazu-ctr-prediction
uv run --locked --extra cu132 --group kaggle kaggle competitions submit avazu-ctr-prediction -f submission.csv -m "reproduction"
```

Use `--extra cpu` instead of `--extra cu132` on CPU-only systems. Never commit
Kaggle tokens or credential files; repository-local credential fallbacks are
ignored by Git.

## Workflow

Place Kaggle files at `data/raw/train.gz` and `data/raw/test.gz`. Data and all
generated artifacts are ignored by Git.

```powershell
# Build three walk-forward folds and the final holdout.
# Evaluation preprocessing never reads test.gz.
uv run --extra cu132 avazu-ctr preprocess configs/champion.yaml --all-windows

# Watch live curves
uv run --extra cu132 avazu-ctr tensorboard configs/champion.yaml

# Confirm a fixed configuration on every walk-forward fold
uv run --extra cu132 avazu-ctr confirm configs/champion.yaml `
  --fold-manifest artifacts/datasets/senet-dcnv2-multihead/walk_forward_0/manifest.json `
  --fold-manifest artifacts/datasets/senet-dcnv2-multihead/walk_forward_1/manifest.json `
  --fold-manifest artifacts/datasets/senet-dcnv2-multihead/walk_forward_2/manifest.json `
  --output artifacts/tuning/confirmation.json

# Train once on the final-holdout protocol and retain evidence, not weights
uv run --extra cu132 avazu-ctr candidate artifacts/tuning/confirmation.json `
  artifacts/datasets/senet-dcnv2-multihead/final_holdout/manifest.json `
  --output artifacts/selection-candidates/champion

# Select the configuration through the paired statistical gate
uv run --extra cu132 avazu-ctr promote configs/champion.yaml `
  artifacts/selection-candidates/champion

# Fit features on every labelled row, refit for best_epoch + 1, and deploy
uv run --extra cu132 avazu-ctr prepare-production configs/champion.yaml
uv run --extra cu132 avazu-ctr refit configs/champion.yaml `
  artifacts/datasets/senet-dcnv2-multihead/production/manifest.json

# Produce the competition submission from the production-only bundle
uv run --extra cu132 avazu-ctr predict artifacts/champion `
  artifacts/datasets/senet-dcnv2-multihead/production/manifest.json `
  --device cuda --output submission.csv
```

CUDA prediction automatically compiles the complete model-to-probability graph
with Inductor and executes it under float16 autocast. Input batches use pinned
memory and nonblocking device transfers. CPU prediction remains eager float32.

Staged tuning produces the same typed confirmation artifact:

```powershell
uv run --extra cu132 avazu-ctr preprocess configs/tuning.yaml --all-windows
uv run --extra cu132 avazu-ctr tune configs/tuning.yaml artifacts/datasets/senet-dcnv2-staged-tuning/walk_forward_0/manifest.json `
  --confirm-manifest artifacts/datasets/senet-dcnv2-staged-tuning/walk_forward_0/manifest.json `
  --confirm-manifest artifacts/datasets/senet-dcnv2-staged-tuning/walk_forward_1/manifest.json `
  --confirm-manifest artifacts/datasets/senet-dcnv2-staged-tuning/walk_forward_2/manifest.json `
  --output artifacts/tuning/confirmation.json
```

`promote` accepts only complete, checksummed evidence generated from recorded
confirmation and final-holdout runs. It verifies ordered populations, run
configurations, manifests, and SQLite metrics before applying the paired
bootstrap and fold guard. Candidate evidence is deleted after rejection or
atomically becomes the active selection after acceptance.

`configs/baseline.yaml` is the clean DCNv2 benchmark.
`configs/champion.yaml` is the SENet + DCNv2 multihead candidate.
`configs/full_features.yaml` keeps that architecture fixed while expanding the
leakage-safe information set for the feature-only champion experiment. It
enables full-graph CUDA Inductor compilation; compilation is strict and never
silently falls back to eager execution.
`configs/stec.yaml` is the paper-faithful STEC candidate.
`configs/ngpt.yaml` is the paper-faithful nGPT candidate adapted to field tokens.
`configs/tuning.yaml` defines the staged search.

The champion compiles 32 categorical and 31 numerical fields; the expanded
feature candidate compiles 53 categorical and 57 numerical fields. Both use
raw context, time, bounded crosses, covariate aggregates, causal impression
history, and temporal target evidence. They explicitly use
`competition_transductive` frequency and distinct-count statistics to describe
the fixed Kaggle scoring batch. Learned categorical vocabularies and target
statistics remain training-only. Use `inductive` mode for an unseen online
stream. See [feature system](docs/features.md).

## Correctness contracts

- Temporal windows are chosen before fitting any data-dependent transform.
- Evaluation datasets contain train/validation only; test data is forbidden.
- Production datasets contain all labelled rows plus test; validation is
  forbidden.
- Learned categorical vocabularies always fit training rows only. Unseen values
  map to ID zero, whose embedding is fixed to the zero vector.
- Inductive covariate aggregates fit only training rows; competition
  transduction of frequency and distinct-count tables is explicit, label-free,
  recorded per table, and never opens test during evaluation.
- Training target evidence uses labels only from earlier temporal blocks.
- The first target block and unseen categories have zero lift and zero evidence.
- Impression history is causal, deterministically ordered, and label-free.
- Click history uses completed labelled hours only; scoring labels are ignored.
- Manifests record feature definitions, fitted sources, label use, and OOV rates.
- Categorical values stay `int64`; numerical values stay `float32`.
- Models return the exact aggregate logit deployed by inference.
- The aggregate logit receives direct BCE supervision.
- STEC exposes the unpooled bilinear interaction that its attention calculation
  averages, and fuses one interaction per encoder level plus the final state.
- nGPT reprojects every matrix and embedding vector to the unit hypersphere
  after each successful optimizer step.
- Hash coefficients and masks are serialized model state.
- Every architecture is tested for complete expected gradient coverage.
- Final-holdout weights can never become an inference bundle.
- Inference accepts only strict, checksummed production-refit bundles and the
  exact production manifest embedded by that deployment.

See [data lineage](docs/data-lineage.md) and
[architecture](docs/architecture.md) for the full contracts.

## Tracking and storage

SQLite is authoritative for run lineage and metrics. TensorBoard mirrors scalar
histories under the same run ID for live curves:

```powershell
uv run --extra cu132 avazu-ctr tensorboard configs/champion.yaml --port 6006
```

TensorBoard stores no weights, graphs, datasets, or embeddings. Tuning,
confirmation, and final-holdout runs retain no weights. Selection retains only
configuration, lineage, metrics, the epoch budget, and row-level holdout losses.
Production refit trains without validation for exactly `best_epoch + 1` epochs.
Only one deployed production bundle is retained, with a hard 512 MiB weight cap.

See [experiment tracking](docs/tracking.md).

## Development

```powershell
uv run --extra cpu ruff format --check .
uv run --extra cpu ruff check .
uv run --extra cpu ty check
uv run --extra cpu pytest
uv build
```

Linux CI runs the CPU suite. Before GPU-facing changes are merged, run the same
checks on Windows plus a CUDA forward/backward and AMP smoke run.

## License

MIT. See [LICENSE](LICENSE).

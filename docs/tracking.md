# Experiment tracking and retention

## Authoritative metadata

`artifacts/experiments.sqlite3` links:

- run and parent IDs;
- resolved configuration and hash;
- dataset manifest;
- Git commit, dirty flag, and diff fingerprint;
- Python, PyTorch, CUDA, platform, and device details;
- train/validation metrics;
- Optuna trial identity;
- artifacts and promotion decisions;
- terminal status and failure traceback.

Failed trials stay failed. Only explicit Optuna pruning is recorded as pruned.

## Curves

TensorBoard event files live under `artifacts/tensorboard/<run-id>`. The run ID
matches SQLite exactly.

```powershell
uv run --extra cu130 avazu-ctr tensorboard configs/champion.yaml
```

Only scalar histories are mirrored. SQLite remains authoritative.

## Weight retention

- Optuna trials never write weights.
- Production resume state is disabled by default.
- When enabled, one `resume.pt` is atomically overwritten and deleted after a
  successful run.
- Candidate inference artifacts are temporary.
- Later runs retain no weights unless `train --export-candidate` is explicit.
- Evaluation comparison files carry the originating run ID and ordered rows.
- `promote` requires paired final-holdout rows plus every walk-forward loss.
- Uncertainty uses a paired contiguous-block bootstrap, avoiding billions of
  row resamples while retaining local temporal dependence.
- Promotion verifies strict loading and checksums before replacement.
- Superseded champion weights are deleted only after the new champion loads.
- Promoted model weights cannot exceed 512 MiB.

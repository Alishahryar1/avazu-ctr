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
- immutable training plans and terminal summaries;
- selection decisions and evidence checksums;
- production deployments and replacement lineage;
- terminal status and failure traceback.

Failed trials stay failed. Only explicit Optuna pruning is recorded as pruned.

## Curves

TensorBoard event files live under `artifacts/tensorboard/<run-id>`. The run ID
matches SQLite exactly.

```powershell
uv run --extra cu132 avazu-ctr tensorboard configs/champion.yaml
```

Only scalar histories are mirrored. SQLite remains authoritative.

## Weight retention

- Optuna trials never write weights.
- Candidate resume state is opt-in. One `resume.pt` is atomically overwritten
  and deleted after a successful run.
- Production refits never write resume checkpoints.
- Tuning trials, confirmations, and final-holdout candidates retain no weights.
- Confirmation evidence contains exact fold run IDs, manifests, populations,
  row counts, and logloss values.
- Selection evidence adds the final-holdout run, best epoch, metrics, and
  checksummed row losses.
- Evaluation uses a single ordered validation reader so paired row losses stay
  aligned even when training uses multiple workers.
- `promote` cross-checks evidence against completed SQLite runs before applying
  the paired gate.
- Uncertainty uses a paired contiguous-block bootstrap, avoiding billions of
  row resamples while retaining local temporal dependence.
- The active selection stores no model weights.
- Production refit runs a recorded fixed epoch/step plan without validation.
- Deployment verifies that the selection did not change during refit.
- Superseded champion weights are deleted only after the staged replacement
  strictly loads and its deployment record commits.
- Exactly one production bundle is retained, with a 512 MiB weight cap.

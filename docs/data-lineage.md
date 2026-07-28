# Data lineage

## Temporal protocol

Avazu's parsed hourly timestamp is the sole split key. Rows are never sorted by
categorical columns.

- The final 24 hours are the primary holdout.
- The preceding three 24-hour windows are expanding walk-forward folds.
- Each fold trains on every earlier hour and validates on its own contiguous
  window.
- The final holdout remains untouched during screening and walk-forward
  confirmation.
- A confirmed configuration is trained once on the final-holdout protocol.
  Its row losses determine selection and its best epoch determines the refit
  budget. Its weights are discarded.
- After selection, production preprocessing fits on every labelled row,
  including the former final holdout. No validation split exists in this role.

Missing hours and invalid window geometry fail preprocessing.

## Fitted state

In `inductive` mode, every vocabulary, frequency table, distinct-count table,
target statistic, and prior is fitted from the active training window only.
Validation and test covariates do not participate.

In explicit `competition_transductive` mode, label-free tables use all
covariates available for the fixed scoring batch: train plus validation during
evaluation and labelled train plus competition test during production.
Evaluation preprocessing still never opens the competition test source.

Target state always uses training labels only. Temporal target features expose a
smoothed category log-odds lift and its evidence count. A training row can see
category labels only from earlier blocks; the first block receives zero lift
and zero evidence. Validation and test rows use the complete preceding training
state, with unseen categories also represented as zero lift and zero evidence.

Frequency and distinct-count features are covariate statistics and do not use
labels. Causal history features count prior impressions and time since the
previous impression in event order; they never read clicks.

## Manifest

Each manifest declares one role:

- `evaluation`: train and validation shards are required; test is forbidden;
- `production`: all-labelled train and test shards are required; validation is
  forbidden.

Manifests record raw-source and shard checksums, population identities, split
boundaries, ordered feature definitions, dtypes, cardinalities, embedding kinds,
feature mode, fitted-table sources and label use, per-split OOV diagnostics,
feature-configuration and resolved-configuration hashes, and the package-lock
hash.

Training and inference consume ordered manifest fields rather than rediscovering
features from files.

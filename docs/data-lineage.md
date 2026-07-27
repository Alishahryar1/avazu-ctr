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

For evaluation, every vocabulary, count table, category threshold, target
statistic, and prior is fitted from the active training window only. Validation
and test covariates do not participate; evaluation preprocessing does not open
the test source.

For production, the same transforms are refitted from all labelled rows. Test
rows use that fitted state and never influence it.

Target encoding uses temporal blocks. A training row can see category labels
only from earlier blocks; the first block receives a neutral `0.5` prior.
Validation and test rows use the complete preceding training state, smoothed by
that training window's label rate.

Count features are named `training_impressions`, not clicks. They are training
window frequency lookups and do not use labels.

## Manifest

Each manifest declares one role:

- `evaluation`: train and validation shards are required; test is forbidden;
- `production`: all-labelled train and test shards are required; validation is
  forbidden.

Manifests record raw-source and shard checksums, population identities, split
boundaries, ordered feature names, dtypes, cardinalities, embedding kinds,
fitted-table checksums, feature-configuration and resolved-configuration hashes,
and the package-lock hash.

Training and inference consume ordered manifest fields rather than rediscovering
features from files.

# Data lineage

## Temporal protocol

Avazu's parsed hourly timestamp is the sole split key. Rows are never sorted by
categorical columns.

- The final 24 hours are the primary holdout.
- The preceding three 24-hour windows are expanding walk-forward folds.
- Each fold trains on every earlier hour and validates on its own contiguous
  window.
- The final holdout remains untouched for model selection and promotion; its
  rows never enter the fitted state of the promoted selection bundle.

Missing hours and invalid window geometry fail preprocessing.

## Fitted state

Every vocabulary, count table, category threshold, target statistic, and prior
is fitted from the active training window only. Validation/test covariates do
not participate.

Target encoding uses temporal blocks. A training row can see category labels
only from earlier blocks; the first block receives a neutral `0.5` prior.
Validation and test rows use the complete preceding training state, smoothed by
that training window's label rate.

Count features are named `training_impressions`, not clicks. They are training
window frequency lookups and do not use labels.

## Manifest

Each fold records raw and shard checksums, split boundaries, ordered feature
names, dtypes, cardinalities, embedding kinds, fitted-table checksums, resolved
configuration hash, and package-lock hash.

Training and inference consume ordered manifest fields rather than rediscovering
features from files.

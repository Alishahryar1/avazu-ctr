# Data lineage

## Profile FFM competition population

Profile FFM preparation validates and checksums the labelled training source
and scoring source before reading them. Label-free covariate frequencies and
publisher profiles are fitted over both fixed competition populations. Click
labels are consumed only from the labelled source.

Rows stay in source order while they are partitioned into app and site
inventories. Each inventory has one training population and one scoring
population. Aligned selector sidecars preserve scoring IDs for final
composition.

Publisher profiles summarize each eligible user's publisher-ID and
publisher-domain distribution. The site selector derives its seen-publisher
set from site training rows and marks scoring rows whose publisher is absent
from that set.

App proxy-user click history advances through training rows and then scoring
rows. Labels from the current hour remain buffered until a later hour begins.
Scoring rows query the completed training history without adding labels. The
app selector marks scoring rows whose completed-hour history is nonempty.

The preparation manifest embeds the resolved configuration, source checksums,
population partition, selector counts, profile coverage, and checksums for all
published sparse artifacts.

## PyTorch temporal protocol

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

## PyTorch fitted state

Learned categorical vocabularies always fit the active training window only.
Validation- or prediction-only values therefore map to the reserved unknown ID
zero rather than receiving parameters that no training row can update. The
corresponding embedding row is fixed to the zero vector.

In `inductive` mode, frequency and distinct-count tables also fit the active
training window only. In explicit `competition_transductive` mode, those
label-free aggregate tables use all covariates available for the fixed scoring
batch: train plus validation during evaluation and labelled train plus
competition test during production. Evaluation preprocessing still never opens
the competition test source. Frequency and distinct-count outputs with the same
join key share one fitted lookup, whose manifest entry declares every output
column and its common provenance.

Target state always uses training labels only. Temporal target features expose a
smoothed category log-odds lift and its evidence count. A training row can see
category labels only from earlier blocks; the first block receives zero lift
and zero evidence. Validation and test rows use the complete preceding training
state, with unseen categories also represented as zero lift and zero evidence.

Frequency and distinct-count features are covariate statistics and do not use
labels. Causal impression history counts prior events in event order. Configured
click-history features consume only completed training hours: the current hour
is buffered, and validation or prediction labels are never read.

## PyTorch manifest

Each manifest declares one role:

- `evaluation`: train and validation shards are required; test is forbidden;
- `production`: all-labelled train and test shards are required; validation is
  forbidden.

Manifests record raw-source and shard checksums, population identities, split
boundaries, ordered feature definitions, dtypes, cardinalities, embedding kinds,
feature mode, the categorical unknown-value contract, fitted-table sources and
label use, fitted join keys and output columns, per-split OOV diagnostics,
feature-configuration and
resolved-configuration hashes, and the package-lock hash.

Training and inference consume ordered manifest fields rather than rediscovering
features from files.

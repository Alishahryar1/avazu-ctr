# Feature system

The feature system compiles strict configuration recipes into one ordered model
contract. The compiled definitions, fitted-table provenance, categorical
coverage, and resolved configuration are stored in every dataset manifest.
Training and inference consume that contract; neither rediscovers columns.

## Selected and expanded feature sets

The selected champion exposes 63 fields:

| Family | Categorical | Numerical |
| --- | ---: | ---: |
| Raw Avazu fields | 21 | 0 |
| Calendar and cyclic time | 4 | 2 |
| User and ad crosses | 7 | 0 |
| Covariate frequency | 0 | 8 |
| Covariate distinct counts | 0 | 4 |
| Causal impression history | 0 | 5 |
| Temporal target evidence | 0 | 12 |
| **Total** | **32** | **31** |

`configs/full_features.yaml` keeps every selected-champion field and adds the
remaining information families as one feature-only experiment:

| Family | Categorical | Numerical |
| --- | ---: | ---: |
| Raw Avazu fields | 21 | 0 |
| Calendar and cyclic time | 4 | 2 |
| Unified inventory and identity context | 6 | 0 |
| User, publisher, ad, and temporal crosses | 11 | 0 |
| Bounded numerical buckets | 10 | 0 |
| Covariate frequency | 0 | 12 |
| Covariate distinct counts | 0 | 8 |
| Causal impression and completed-hour click history | 1 | 13 |
| Hierarchical temporal target evidence | 0 | 22 |
| **Total** | **53** | **57** |

The model architecture and one-epoch optimization recipe are unchanged. All
new families can be removed through configuration for subsequent ablations.

Crosses are built in declared order, so a recipe may consume an earlier derived
field such as `user_proxy`. Every cross must use a bounded hash embedding.
High-drift raw identifiers (`site_id`, `app_id`, `device_id`, `device_ip`,
`C14`, `C17`, and `C21`) are also hashed in the shipped configurations.

## Covariate modes

`data.features.mode` controls only label-free aggregate state. Learned
categorical vocabularies always fit training rows:

- `inductive` fits frequency and distinct-count tables from the training
  partition. Validation or prediction covariates cannot change training
  features.
- `competition_transductive` fits frequency and distinct-count tables from
  every covariate batch available at prediction time. Evaluation uses train
  plus validation covariates; production uses labelled train plus
  competition-test covariates.

Target tables and priors always use training labels only. Evaluation never opens
the competition test source. The explicit mode exists because batch
transduction is useful for a fixed Kaggle test set but is not an honest model of
an unseen online stream. Metrics from different modes must not be compared as
if their information sets were identical.

Every fitted table records:

- the transform kind, exact join keys, and ordered output columns;
- whether labels were used;
- its exact source splits;
- row count, relative path, and SHA-256 checksum.

The manifest rejects a vocabulary or label-dependent table sourced from
validation or prediction data, an undeclared transductive aggregate table, or a
table whose sources do not match the configured mode.

## Row-local categorical features

Calendar fields are deterministic functions of Avazu's parsed timestamp:

- `hour_of_day`;
- `day_of_week`;
- `day_of_month`;
- `hour_of_week`.

The default crosses are:

- `user_proxy = device_ip × device_model`;
- `device_id_x_app_id`;
- `device_ip_x_C14`;
- `user_proxy_x_app_id`;
- `user_proxy_x_site_id`;
- `site_id_x_C14`;
- `app_id_x_C14`.

Cross inputs are joined with a non-data separator before stable full-width
hashing. Model-side hash coefficients are serialized buffers, so preprocessing,
training, and deployed inference use the same deterministic mapping.

The expanded recipe also creates one inventory namespace over app and site
traffic (`publisher_id`, `publisher_domain`, and `publisher_category`) and one
identity namespace. A non-placeholder `device_id` is authoritative; otherwise
`device_ip × device_model` is used as the proxy identity. Explicit
`inventory_type` and `identity_kind` fields preserve which branch supplied each
value.

## Covariate aggregates

Frequency features are `log1p` counts looked up from the configured covariate
population. Distinct-count recipes measure how many apps or sites were observed
for `device_ip` and `user_proxy`, also transformed with `log1p`.

All aggregates sharing a join key are fitted into one sorted wide lookup and
joined once during transformation. The final numerical fields remain separate
ordered model features; the physical consolidation removes duplicate keys and
repeated joins without changing their values.

These transforms use no labels. In transductive mode they deliberately describe
the complete fixed scoring batch; their fitted-table sources make that choice
auditable.

## Causal impression history

History consumes canonical rows in `(_timestamp_hour, source partition,
_row_index)` order. Polars 1.43.1 streams the canonical scan into compact
open-addressed `uint64`/`uint32` identity state, then consumes that ordered Arrow
stream as the source of the remaining lazy feature plan. Fitted-table joins,
hashing, projection, and output collection therefore stay inside the native
streaming engine. No global group sort or full-population history tensor is
materialized. A training row sees only earlier training events; a scoring row
may see training context and earlier events from its own scoring stream.

The features are:

- prior impressions for `user_proxy` and `device_ip`;
- hours since the previous impression for both identities;
- prior impressions in the current hour for `user_proxy`.

All are `log1p` transformed. They are intentionally named *impressions*: no
click label is read. A zero prior count distinguishes the first observation
from a repeated impression occurring in the same hour.

The expanded recipe additionally tracks completed-hour click state for the
unified identity. It emits prior clicks and non-clicks, smoothed CTR logit lift,
hours and impressions since the last click, and a four-hour click pattern.
Labels from the current hour remain buffered until the next hour begins, even
when an hour spans multiple input batches. Validation and prediction labels are
never queued, so scoring rows see labelled training history only.

## Temporal target evidence

For `app_id`, `site_id`, `site_domain`, `app_domain`, `C14`, and `C17`, training
hours are divided into contiguous blocks. A row can use category labels only
from earlier blocks.

Each category produces two fields:

- `*_target_logit_lift`: the smoothed category posterior log-odds minus the
  preceding global-prior log-odds;
- `*_target_evidence_log1p`: `log1p` of the preceding category count.

The first block has zero lift and zero evidence. An unseen scoring category also
has zero lift and zero evidence. This makes “no history” semantically identical
across training, validation, and production instead of inventing a `0.5`
click-rate feature. Probabilities are clipped only for finite log-odds.

The expanded target recipe includes the unified publisher hierarchy, `C21`,
and `device_model`. A missing child statistic has zero lift and evidence while
its domain/category and related ad-taxonomy fields remain available as natural
backoff evidence.

## Bounded numerical buckets

Configured numerical features may be projected into additional categorical
buckets without replacing their continuous values. Boundary values are part of
the feature contract, and sources stored as `log1p` can declare boundaries in
their original count scale. Bucket outputs use small bounded hash tables, so
they require neither learned preprocessing vocabularies nor unbounded
checkpoint growth.

## Coverage diagnostics

Vocabulary-encoded fields reserve ID zero for values absent from the fitted
training vocabulary or below `minimum_frequency`. ID zero is a fixed zero-vector
embedding: scoring-only categories cannot introduce randomly initialized,
untrained model state. Every manifest records the row count, unknown count, and
exact OOV rate for every vocabulary field and split.

Hashed fields have no OOV state and therefore do not appear in the OOV map.
Their bucket counts and embedding kinds remain part of the feature contract and
model-size estimate.

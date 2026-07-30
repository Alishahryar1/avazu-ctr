# Reproducibility

Python is fixed to 3.12 and `uv.lock` fixes the complete dependency graph.
CPU and CUDA 13.2 PyTorch installs are explicit, mutually exclusive extras.

## Profile FFM

The checked profile FFM configuration fixes raw source hashes, expected
populations, feature thresholds, hash space, publisher-mask rate, cold token,
rank, learning rate, regularization, epoch count, source-order training, and a
single OpenMP thread.

Sparse hashing, profile-token hashing, and the SplitMix publisher mask are
deterministic. Publisher-profile edges execute as lazy Polars plans and are
sorted before sparse serialization. Completed-hour history buffers current-hour
labels and ignores scoring labels.

Preparation manifests embed the resolved recipe and checksum every source,
sparse file, and selector. Run manifests record the compiler version, native
source checksum, compiled binary checksum, executed commands, prediction and
log checksums, preparation checksum, composition counts, and final submission
checksum.

## PyTorch

Splits, feature recipes, preprocessing, hashing, causal event order, masks,
initialization, data order, workers, and Optuna samplers are seeded. Hash
coefficients and feature masks are checkpoint buffers.

Dataset manifests distinguish inductive and competition-transductive covariate
state and checksum every fitted table with its declared source splits. Runs
therefore cannot silently compare or deploy a different information set.
Categorical vocabularies are always training-only, and the reserved unknown ID
has a fixed zero embedding, so scoring-only values cannot create untrained
parameters.

Processed shard names and row limits are deterministic. Data workers receive
fixed strided shard partitions before epoch-local shuffling, and each worker
coalesces its partition across file boundaries. Recorded step budgets therefore
match the batches the iterable dataset actually emits.

Production CUDA permits seeded nondeterministic kernels when they improve
throughput. Tests enable `torch.use_deterministic_algorithms(True)`. Exact
bitwise equivalence is therefore required for data and CPU state round trips,
while normal GPU comparisons use recorded seeds and metric tolerances.
Production CUDA inference additionally uses float16 autocast and returns
float32 probabilities; GPU contract tests compare it with eager float32
predictions under explicit numerical tolerances.

Every run stores the package-lock hash, data manifest, resolved configuration,
source commit/diff fingerprint, environment snapshot, immutable training plan,
and terminal execution summary.

The selected final-holdout epoch is evidence, not a checkpoint. Production
refit restarts from the recorded seed on the all-labelled manifest, rebuilds
the optimizer and scheduler for exactly `best_epoch + 1` epochs, and records
the resulting epoch and step counts in the deployed bundle.

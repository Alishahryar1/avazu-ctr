# Reproducibility

Python is fixed to 3.12 and `uv.lock` fixes the complete dependency graph.
CPU and CUDA 13.0 PyTorch installs are explicit, mutually exclusive extras.

Splits, feature recipes, preprocessing, hashing, causal event order, masks,
initialization, data order, workers, and Optuna samplers are seeded. Hash
coefficients and feature masks are checkpoint buffers.

Dataset manifests distinguish inductive and competition-transductive covariate
state and checksum every fitted table with its declared source splits. Runs
therefore cannot silently compare or deploy a different information set.

Production CUDA permits seeded nondeterministic kernels when they improve
throughput. Tests enable `torch.use_deterministic_algorithms(True)`. Exact
bitwise equivalence is therefore required for data and CPU state round trips,
while normal GPU comparisons use recorded seeds and metric tolerances.

Every run stores the package-lock hash, data manifest, resolved configuration,
source commit/diff fingerprint, environment snapshot, immutable training plan,
and terminal execution summary.

The selected final-holdout epoch is evidence, not a checkpoint. Production
refit restarts from the recorded seed on the all-labelled manifest, rebuilds
the optimizer and scheduler for exactly `best_epoch + 1` epochs, and records
the resulting epoch and step counts in the deployed bundle.

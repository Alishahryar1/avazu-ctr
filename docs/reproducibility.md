# Reproducibility

Python is fixed to 3.12 and `uv.lock` fixes the complete dependency graph.
CPU and CUDA 13.0 PyTorch installs are explicit, mutually exclusive extras.

Splits, preprocessing, hashing, masks, initialization, data order, workers, and
Optuna samplers are seeded. Hash coefficients and feature masks are checkpoint
buffers.

Production CUDA permits seeded nondeterministic kernels when they improve
throughput. Tests enable `torch.use_deterministic_algorithms(True)`. Exact
bitwise equivalence is therefore required for data and CPU state round trips,
while normal GPU comparisons use recorded seeds and metric tolerances.

Every run stores the package-lock hash, data manifest, resolved configuration,
source commit/diff fingerprint, and environment snapshot.

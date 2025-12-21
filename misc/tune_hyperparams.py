"""
Hyperparameter tuning script using Optuna for MultiHeadDiversityModel.

This script performs Bayesian optimization over ~34 hyperparameters including:
- Training parameters (learning rates, weight decay, grad clip)
- Regularization (dropout)
- Model architecture (diversity weight, feature bagging, aggregation)
- DCN configuration (layers, low rank, layer norm)
- MLP configuration (depth, width, activation, skip connections)
- Per-head configurations (4 heads with individual hidden dims, activations, etc.)

Usage:
    python misc/tune_hyperparams.py --n-trials 50 --timeout 14400
    python misc/tune_hyperparams.py --resume  # Resume existing study
"""

import argparse
import math
import os
import sys
from copy import deepcopy
from typing import Any, cast

import optuna
from optuna.trial import Trial
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CONFIG, seed_everything
from src.processing.data_processor import load_metadata, get_parquet_path
from src.processing.dataset import ParquetFullDataset
from src.models.architectures import create_model
from src.training.evaluator import evaluate
from src.training.schedulers import LRSchedulerWithWarmup

# Constants for search space
ACTIVATIONS = ["relu", "gelu", "silu", "mish", "tanh"]
HEAD_HIDDEN_DIMS = [16, 32, 64, 128, 256, 512]
NUM_HEADS = 4

# Database path for study persistence
STUDY_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "optuna_study.db"
)


def create_config_from_trial(
    trial: Trial, base_config: dict[str, Any]
) -> dict[str, Any]:
    """
    Sample hyperparameters from Optuna trial and return modified config.

    Args:
        trial: Optuna trial object for suggesting hyperparameters
        base_config: Base configuration to modify

    Returns:
        Modified configuration dictionary with sampled hyperparameters
    """
    config = deepcopy(base_config)

    # === Training params ===
    config["dense_optimizer"]["lr"] = trial.suggest_float(
        "dense_lr", 1e-5, 1e-1, log=True
    )
    config["embedding_optimizer"]["lr"] = trial.suggest_float(
        "embed_lr", 1e-3, 1.0, log=True
    )
    config["dense_optimizer"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )

    # === Dense optimizer scheduler ===
    config["dense_optimizer"]["scheduler"] = {
        "decay_type": trial.suggest_categorical(
            "dense_decay_type", ["none", "cosine", "linear"]
        ),
        "warmup_epoch_ratio": trial.suggest_float("dense_warmup_ratio", 0.0, 0.8),
        "min_lr": trial.suggest_float("dense_min_lr", 1e-8, 1e-4, log=True),
    }

    # === Embedding optimizer scheduler ===
    config["embedding_optimizer"]["scheduler"] = {
        "decay_type": trial.suggest_categorical(
            "embed_decay_type", ["none", "cosine", "linear"]
        ),
        "warmup_epoch_ratio": trial.suggest_float("embed_warmup_ratio", 0.0, 0.8),
        "min_lr": trial.suggest_float("embed_min_lr", 1e-8, 1e-4, log=True),
    }

    # === Regularization ===
    config["model"]["backbone_config"]["mlp_dropout"] = trial.suggest_float(
        "mlp_dropout", 0.0, 0.5
    )
    config["grad_clip"] = trial.suggest_float("grad_clip", 0.1, 5.0)

    # === Model architecture ===
    config["model"]["diversity_weight"] = trial.suggest_float(
        "diversity_weight", 0.001, 1.0, log=True
    )
    config["model"]["feature_bagging_ratio"] = trial.suggest_float(
        "feature_bagging_ratio", 0.5, 1.0
    )
    config["model"]["aggregation_method"] = trial.suggest_categorical(
        "aggregation_method", ["mean", "gated"]
    )
    if config["model"]["aggregation_method"] == "gated":
        config["model"]["gating_hidden_dim"] = trial.suggest_categorical(
            "gating_hidden_dim", [None, 16, 32, 64, 128]
        )
    else:
        config["model"]["gating_hidden_dim"] = None

    # === DCN ===
    config["model"]["backbone_config"]["dcn_num_layers"] = trial.suggest_int(
        "dcn_num_layers", 2, 16
    )
    config["model"]["backbone_config"]["dcn_low_rank"] = trial.suggest_int(
        "dcn_low_rank", 16, 256
    )
    config["model"]["backbone_config"]["dcn_use_layernorm"] = trial.suggest_categorical(
        "dcn_use_layernorm", [True, False]
    )

    # === MLP architecture ===
    n_layers = trial.suggest_int("n_mlp_layers", 1, 6)
    width = trial.suggest_int("mlp_width", 128, 4096, step=128)
    config["model"]["backbone_config"]["mlp_hidden_dims"] = [width] * n_layers
    config["model"]["backbone_config"]["mlp_activation"] = trial.suggest_categorical(
        "mlp_activation", ["relu", "gelu", "silu", "mish"]
    )
    config["model"]["backbone_config"]["mlp_use_skip_connections"] = (
        trial.suggest_categorical("mlp_use_skip_connections", [True, False])
    )

    # === Head parameters (all 4 heads tuned) ===
    heads = []
    for i in range(NUM_HEADS):
        head_config = {
            "hidden_dims": [
                trial.suggest_categorical(f"head_{i}_hidden_dim", HEAD_HIDDEN_DIMS)
            ],
            "activation": trial.suggest_categorical(
                f"head_{i}_activation", ACTIVATIONS
            ),
            "dropout": trial.suggest_float(f"head_{i}_dropout", 0.0, 0.5),
            "use_layer_norm": trial.suggest_categorical(
                f"head_{i}_use_layer_norm", [True, False]
            ),
            "use_skip_connections": trial.suggest_categorical(
                f"head_{i}_use_skip_connections", [True, False]
            ),
        }
        heads.append(head_config)
    config["model"]["heads"] = heads

    return config


def train_single_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    config: dict[str, Any],
    trial: Trial,
) -> float:
    """
    Train model for a single epoch with pruning support.

    Args:
        model: The model to train
        train_loader: Training data loader
        config: Configuration dictionary
        trial: Optuna trial for pruning

    Returns:
        Average training loss
    """
    device = config["device"]
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config["epochs"]

    # Separate parameters for embeddings vs other layers
    embedding_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            embedding_params.append(param)
        else:
            other_params.append(param)

    # Create optimizers
    embedding_optimizer = optim.Adagrad(
        embedding_params,
        lr=config["embedding_optimizer"]["lr"],
        weight_decay=config["embedding_optimizer"].get("weight_decay", 0.0),
    )
    other_optimizer = optim.AdamW(
        other_params,
        lr=config["dense_optimizer"]["lr"],
        weight_decay=config["dense_optimizer"].get("weight_decay", 1e-4),
    )

    # LR scheduler with warmup (matching main trainer)
    dense_opt_cfg = config["dense_optimizer"]
    warmup_ratio = float(dense_opt_cfg.get("warmup_epoch_ratio", 0.0))
    warmup_steps = int(steps_per_epoch * warmup_ratio)
    scheduler = LRSchedulerWithWarmup(
        other_optimizer, warmup_steps=warmup_steps, total_steps=total_steps
    )

    # AMP setup
    use_amp = config.get("auto_amp", True) and device == "cuda"
    amp_dtype = torch.float16

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Trial {trial.number}", leave=False)
    for batch_idx, batch_data in enumerate(pbar):
        X_batch, y_batch = batch_data
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        embedding_optimizer.zero_grad()
        other_optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            output = model(X_batch)
            loss = model.compute_loss(output, y_batch)  # pyrefly: ignore

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if config["grad_clip"] > 0:
                scaler.unscale_(embedding_optimizer)
                scaler.unscale_(other_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(embedding_optimizer)
            scaler.step(other_optimizer)
            scaler.update()
        else:
            loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            embedding_optimizer.step()
            other_optimizer.step()
            scheduler.step()

        loss_val = loss.item()

        # Check for NaN loss and prune if detected
        if math.isnan(loss_val) or math.isinf(loss_val):
            print(f"\n[Trial {trial.number}] Loss became NaN/Inf, pruning trial.")
            raise optuna.TrialPruned()

        total_loss += loss_val
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # Report intermediate value for pruning (every 10% of epoch)
        if batch_idx > 0 and batch_idx % max(1, num_batches // 10) == 0:
            intermediate_loss = total_loss / (batch_idx + 1)
            trial.report(intermediate_loss, batch_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return total_loss / num_batches


def objective(trial: Trial) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object

    Returns:
        Validation AUC (to maximize)
    """
    # Create config with sampled hyperparameters
    config = create_config_from_trial(trial, cast(dict[str, Any], CONFIG))

    print(f"\n[Trial {trial.number}] Hyperparameters:")
    for name, value in trial.params.items():
        print(f"  {name}: {value}")

    # Force settings for tuning
    config["validation_split"] = 0.1
    config["epochs"] = 1
    config["use_tensorboard"] = False
    config["compile_model"] = False

    seed_everything(config["seed"])

    # Load data
    try:
        vocab_sizes, feature_names = load_metadata()
    except FileNotFoundError:
        print("Processed data not found. Run 'python data_processor.py' first.")
        raise optuna.TrialPruned()

    train_parquet = get_parquet_path("train")
    full_dataset = ParquetFullDataset(
        parquet_path=train_parquet, feature_cols=feature_names, label_col="click"
    )

    # Sequential split for time-sorted data
    split_idx = int(len(full_dataset) * (1 - config["validation_split"]))
    train_indices = list(range(split_idx))
    val_indices = list(range(split_idx, len(full_dataset)))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = None
    try:
        model = create_model(config, vocab_sizes, feature_names)  # pyrefly: ignore
        model.to(config["device"])

        # Train for one epoch
        train_loss = train_single_epoch(model, train_loader, config, trial)
        print(f"[Trial {trial.number}] Train loss: {train_loss:.5f}")
    except optuna.TrialPruned:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        raise
    except Exception as e:
        print(f"Model creation or training failed: {e}")
        if model is not None:
            del model
            torch.cuda.empty_cache()
        raise optuna.TrialPruned()

    # Evaluate
    use_amp = config.get("auto_amp", True) and config["device"] == "cuda"
    val_loss, val_auc, val_logloss = evaluate(
        model, val_loader, config["device"], use_amp=use_amp, amp_dtype=torch.float16
    )

    # Clean up GPU memory after evaluation
    del model
    torch.cuda.empty_cache()

    # Log additional metrics
    trial.set_user_attr("val_loss", val_loss)
    trial.set_user_attr("val_auc", val_auc)

    return val_logloss


def print_best_params(study: optuna.Study) -> None:
    """Print best parameters from completed study."""
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation LogLoss: {study.best_value:.5f}")
    if "val_auc" in study.best_trial.user_attrs:
        print(f"Associated AUC: {study.best_trial.user_attrs['val_auc']:.5f}")

    print("\nBest hyperparameters:")
    print("-" * 40)
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Export as config dict
    print("\n" + "-" * 40)
    print("Copy-paste config update:")
    print("-" * 40)
    print("CONFIG updates = {")
    for key, value in study.best_params.items():
        if isinstance(value, str):
            print(f'    "{key}": "{value}",')
        else:
            print(f'    "{key}": {value},')
    print("}")


def main() -> None:
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials to run"
    )
    parser.add_argument(
        "--timeout", type=int, default=14400, help="Timeout in seconds (default: 4h)"
    )
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument(
        "--study-name", type=str, default="avazu_ctr_tuning", help="Study name"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("AVAZU CTR - HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Study name: {args.study_name}")
    print(f"Database: {STUDY_DB_PATH}")
    print(f"Max trials: {args.n_trials}")
    print(f"Timeout: {args.timeout}s ({args.timeout / 3600:.1f}h)")
    print("=" * 80)

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )

    if args.resume and len(study.trials) > 0:
        print(f"Resuming study with {len(study.trials)} existing trials")
        print(f"Current best LogLoss: {study.best_value:.5f}")

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    # Print results
    print_best_params(study)


if __name__ == "__main__":
    main()

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Polars-Data%20Processing-CD792C?style=for-the-badge&logo=polars&logoColor=white" alt="Polars">
  <img src="https://img.shields.io/badge/TensorBoard-Logging-FF6F00?style=for-the-badge&logo=tensorboard&logoColor=white" alt="TensorBoard">
</p>

<h1 align="center">🎯 Avazu Click-Through Rate Prediction</h1>

<p align="center">
  <strong>A state-of-the-art deep learning solution combining DCNv2 and FiBiNET architectures for CTR prediction</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-results">Results</a>
</p>

---

## 📋 Overview

This project implements a cutting-edge Click-Through Rate (CTR) prediction model for the [Avazu CTR Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) Kaggle competition. The model combines multiple advanced deep learning techniques to effectively capture both implicit and explicit feature interactions in advertising data.

### Key Highlights

- 🚀 **Modern Architecture**: Combines DCNv2 (Deep Cross Network V2) with SE-Net attention from FiBiNET
- ⚡ **Efficient Training**: Low-rank decomposition, mixed optimizers, and TensorBoard monitoring
- 🔧 **Highly Configurable**: Easy-to-modify configuration for rapid experimentation
- 📊 **Production Ready**: Clean codebase with comprehensive test suite

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **DCNv2 Cross Network** | Explicit high-order feature interactions with optional low-rank decomposition |
| **SE-Net (FiBiNET)** | Squeeze-and-Excitation for dynamic feature importance weighting |
| **Focal Loss** | Handles severe class imbalance (CTR datasets are typically 80%+ negative) |
| **Dual Optimizer** | Adagrad for embeddings + AdamW for other parameters |
| **LR Warmup** | Linear warmup with cosine annealing for stable training |
| **Early Stopping** | Prevents overfitting with patience-based stopping |
| **TensorBoard** | Real-time training visualization |
| **Graceful Interrupt** | Ctrl+C safely stops training without losing progress |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Features                           │
│              (Categorical: site_id, app_id, device_id, ...)     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding Layer (64-dim)                     │
│                     + Batch Normalization                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SE-Net Layer                            │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  Squeeze: Mean + Max Pooling → [Batch, 2*Fields]     │     │
│    │  Excitation: MLP → Field Weights                     │     │
│    │  Re-weight: Scale embeddings by importance           │     │
│    └──────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DCNv2 Cross Network                          │
│    ┌──────────────────────────────────────────────────────┐     │
│    │  Layer i: x_{i+1} = x_0 ⊙ (W_i · x_i + b_i) + x_i   │     │
│    │  Low-rank: W = U @ V reduces O(d²) → O(2dr)          │     │
│    └──────────────────────────────────────────────────────┘     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Deep MLP Network                           │
│                 [1024 → 512 → 512 → 1]                          │
│              + BatchNorm + GELU + Dropout                       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Sigmoid → CTR Prediction                    │
└─────────────────────────────────────────────────────────────────┘
```

### Model Components

#### 1. SE-Net Layer (Squeeze-and-Excitation)
From the **FiBiNET** paper, this layer learns the importance of each feature field dynamically:
- **Squeeze**: Aggregates each field's embedding into a scalar (supports mean, max, or both)
- **Excitation**: 2-layer MLP with bottleneck to compute field importance weights
- **Re-weight**: Scales each field embedding by its importance

#### 2. DCNv2 (Deep Cross Network V2)
Captures explicit feature interactions up to arbitrary order:
- **Full-rank mode**: Standard cross layers with O(d²) parameters
- **Low-rank mode**: Matrix decomposition W = U @ V, reducing to O(2dr) parameters

#### 3. Deep Network
Standard MLP with modern best practices:
- Batch Normalization for training stability
- GELU activation (shown to outperform ReLU in many tasks)
- Dropout for regularization

---

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision polars numpy scikit-learn tqdm tensorboard
```

### Data Preparation

1. Download the Avazu dataset from [Kaggle](https://www.kaggle.com/c/avazu-ctr-prediction/data)
2. Place `train.gz` and `test.gz` in the `./data/` directory

```bash
# Process the raw data (builds vocabulary and encodes features)
python data_processor.py
```

### Training

```bash
# Start training
python train.py

# Monitor training with TensorBoard
tensorboard --logdir=runs
```

### Inference

```bash
# Generate predictions on test set
python inference.py

# Output: submission.csv
```

### Running Tests

```bash
# Run all tests
python tests.py

# Run specific test
python tests.py TestModelStructure.test_dcn_layers

# List available tests
python tests.py --list
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`. Here are the key settings:

### Model Architecture

```python
{
    "embedding_dim": 64,          # Dimension of feature embeddings
    
    # DCNv2 Settings
    "use_dcn": True,              # Enable/disable DCNv2
    "dcn_num_layers": 4,          # Number of cross layers
    "dcn_low_rank": 32,           # Low-rank dimension (None = full-rank)
    
    # SE-Net Settings
    "use_senet": True,            # Enable/disable SE-Net
    "senet_squeeze_funcs": ["mean", "max"],  # Squeeze functions
    "senet_reduction_ratio": 3,   # Bottleneck reduction
    "senet_activation": "tanh",   # Output activation
    
    # MLP Settings
    "mlp_hidden_dims": [1024, 512, 512],
    "mlp_activation": "gelu",
}
```

### Training Settings

```python
{
    "batch_size": 2048,
    "epochs": 50,
    "lr": 1e-4,                   # Learning rate for MLP/DCN
    "embedding_lr": 1.0,          # Learning rate for embeddings
    "lr_warmup_epoch_ratio": 0.2, # 20% of first epoch for warmup
    "early_stopping_patience": 50,
    
    # Regularization
    "mlp_dropout": 0.1,
    "weight_decay": 1e-5,
    "focal_loss_gamma": 2.0,      # Focal loss (0 = BCE)
    "grad_clip": 10.0,
}
```

---

## 📁 Project Structure

```
avazu-ctr/
├── 📄 config.py           # Centralized configuration
├── 📄 model.py            # Model architecture (DCNv2, SE-Net, MLP)
├── 📄 train.py            # Training loop with TensorBoard logging
├── 📄 inference.py        # Generate test predictions
├── 📄 data_processor.py   # Polars-based data preprocessing
├── 📄 dataset.py          # PyTorch Dataset wrapper
├── 📄 tests.py            # Comprehensive test suite
├── 📁 data/               # Raw and processed data
│   ├── train.gz
│   ├── test.gz
│   ├── X_train.npy
│   └── ...
├── 📁 models/             # Saved model checkpoints
│   └── best_model.pth
├── 📁 runs/               # TensorBoard logs
└── 📄 submission.csv      # Final predictions
```

---

## 📊 Training Features

### Dual Optimizer Strategy
Following recommendations from CTR literature, embeddings and other parameters use different optimizers:
- **Embeddings**: Adagrad with lr=1.0 (works well for sparse categorical features)
- **Other layers**: AdamW with lr=1e-4 and weight decay

### Learning Rate Schedule
```
    LR
    │
1e-4├────────────╲
    │   warmup    ╲  cosine decay
    │  ╱           ╲
    │╱              ╲
1e-6├────────────────────────────
    └────────────────────────────→ Steps
         20%              100%
```

### Focal Loss for Imbalance
CTR datasets are extremely imbalanced (2-17% positive rate). Focal loss down-weights easy negatives:

```
FL(p) = -(1-p)^γ * log(p)
```

With γ=2, well-classified examples contribute less to the loss.

---

## 🧪 Experimentation Tips

### Quick Ablations

```python
# Test without DCNv2
CONFIG["use_dcn"] = False

# Test without SE-Net
CONFIG["use_senet"] = False

# Try different MLP architectures
CONFIG["mlp_hidden_dims"] = [512, 256]
CONFIG["mlp_activation"] = "silu"

# Full-rank DCN (more expressive but slower)
CONFIG["dcn_low_rank"] = None
```

### Memory Optimization

If running out of GPU memory:
```python
CONFIG["batch_size"] = 1024
CONFIG["embedding_dim"] = 32
CONFIG["mlp_hidden_dims"] = [512, 256]
```

---

## 📚 References

- **DCN V2**: [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)
- **FiBiNET**: [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction](https://arxiv.org/abs/1905.09433)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 📝 License

This project is for educational and competition purposes.

---

<p align="center">
  Built with ❤️ using PyTorch
</p>

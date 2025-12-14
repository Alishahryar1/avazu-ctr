
import torch
import torch.nn as nn
from config import CONFIG
from model import GatedDCNModel

def verify_model_structure():
    print("Testing Model Configuration...")
    print(f"Config: DCN Layers={CONFIG['dcn_num_layers']}, MLP Dims={CONFIG['mlp_hidden_dims']}, Dropout={CONFIG['mlp_dropout']}")
    
    # Mock data
    vocab_sizes = {'f1': 100, 'f2': 100}
    feature_names = ['f1', 'f2']
    
    model = GatedDCNModel(
        vocab_sizes, 
        CONFIG['embedding_dim'], 
        feature_names,
        dcn_num_layers=CONFIG['dcn_num_layers'],
        mlp_hidden_dims=CONFIG['mlp_hidden_dims'],
        mlp_dropout=CONFIG['mlp_dropout']
    )
    
    print("\nModel Architecture:")
    print(model)
    
    # Check DCN layers
    assert len(model.dcn.W) == CONFIG['dcn_num_layers'], f"Expected {CONFIG['dcn_num_layers']} DCN layers, got {len(model.dcn.W)}"
    
    # Check MLP structure
    # MLP sequence: [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
    # For [256, 128]:
    # 0: Linear(input -> 256)
    # 1: ReLU
    # 2: Dropout
    # 3: Linear(256 -> 128)
    # 4: ReLU
    # 5: Dropout
    # 6: Linear(128 -> 1)
    # Total layers = len(dims) * 3 + 1
    expected_mlp_len = len(CONFIG['mlp_hidden_dims']) * 3 + 1
    assert len(model.mlp) == expected_mlp_len, f"Expected MLP length {expected_mlp_len}, got {len(model.mlp)}"
    
    print("\nVerification Successful: Model matches configuration.")

if __name__ == "__main__":
    verify_model_structure()

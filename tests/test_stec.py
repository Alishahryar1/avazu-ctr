"""Tests for STEC model architecture and layers."""
import unittest
import torch
import torch.nn as nn
from src.config.config import CONFIG, ConfigType
from src.models.layers.stec_block import STECBlock
from src.models.layers.multi_head_stec import MultiHeadSTEC
from src.models.layers.position_wise_ffn import PositionWiseFFN
from src.models.layers.stec_encoder import STECEncoderLayer
from src.models.architectures.stec import STECModel, BilinearInteractionLayer
from tests.test_models import make_test_config


class TestSTECLayers(unittest.TestCase):
    """Test individual STEC layer components."""
    
    def test_stec_block_forward(self):
        """Test single head STEC block output shapes."""
        embed_dim = 16
        block = STECBlock(embed_dim)
        
        batch = 4
        fields = 10
        x = torch.randn(batch, fields, embed_dim)
        
        attn_out, bilinear = block(x)
        
        # Check attention output: [B, F, D]
        self.assertEqual(attn_out.shape, (batch, fields, embed_dim))
        
        # Check bilinear output: [B, F, F, D]
        self.assertEqual(bilinear.shape, (batch, fields, fields, embed_dim))
    
    def test_multi_head_stec_forward(self):
        """Test multi-head STEC block output shapes."""
        embed_dim = 32
        num_heads = 4
        block = MultiHeadSTEC(embed_dim, num_heads)
        
        batch = 4
        fields = 10
        x = torch.randn(batch, fields, embed_dim)
        
        attn_out, bilinear = block(x)
        
        # Check attention output: [B, F, D]
        self.assertEqual(attn_out.shape, (batch, fields, embed_dim))
        
        # Check bilinear output: [B, H*F*F, head_dim]
        head_dim = embed_dim // num_heads
        expected_bilinear_shape = (batch, num_heads * fields * fields, head_dim)
        self.assertEqual(bilinear.shape, expected_bilinear_shape)
    
    def test_stec_encoder_layer(self):
        """Test full encoder layer with FFN."""
        embed_dim = 32
        num_heads = 4
        layer = STECEncoderLayer(embed_dim, num_heads, use_ffn=True)
        
        batch = 4
        fields = 10
        x = torch.randn(batch, fields, embed_dim)
        
        out, bilinear = layer(x)
        
        self.assertEqual(out.shape, (batch, fields, embed_dim))
        expected_bilinear_dim = embed_dim // num_heads
        self.assertEqual(bilinear.shape[2], expected_bilinear_dim)
        
    def test_bilinear_interaction_layer(self):
        """Test standalone bilinear interaction layer."""
        embed_dim = 32
        num_heads = 4
        layer = BilinearInteractionLayer(embed_dim, num_heads)
        
        batch = 4
        fields = 5
        x = torch.randn(batch, fields, embed_dim)
        
        bilinear = layer(x)
        
        # Expected: [B, H*F*F, head_dim]
        head_dim = embed_dim // num_heads
        expected_shape = (batch, num_heads * fields * fields, head_dim)
        self.assertEqual(bilinear.shape, expected_shape)


class TestSTECModel(unittest.TestCase):
    """Test full STEC model architecture."""
    
    @classmethod
    def setUpClass(cls):
        cls.vocab_sizes = {'f1': 100, 'f2': 100}
        cls.feature_names = ['f1', 'f2']
        
        # Create config tailored for STEC
        cls.config = make_test_config(
            embedding_dim=16,
            use_stec=True,
            stec_num_layers=2,
            stec_num_heads=4,
            stec_hidden_dim=32,
            stec_dropout=0.1,
            stec_use_ffn=True,
            stec_mlp_hidden_dims=[32, 16],
            embedding_projection_dim=16,  # Must be divisible by num_heads (4)
            feature_embedding_overrides={}
        )
    
    def test_model_instantiation(self):
        """Verify model can be created."""
        model = STECModel(self.vocab_sizes, self.feature_names, self.config)
        self.assertIsInstance(model, STECModel)
        self.assertEqual(len(model.stec_layers), 2)
    
    def test_model_forward(self):
        """Verify forward pass works."""
        model = STECModel(self.vocab_sizes, self.feature_names, self.config)
        
        batch_size = 4
        x = torch.randint(0, 100, (batch_size, 2))
        
        out = model(x)
        self.assertEqual(out['logits'].shape, (batch_size, 1))
    
    def test_variable_embeddings_adaptation(self):
        """Test model handles variable embeddings by projecting."""
        config = make_test_config(
            use_variable_embeddings=True,
            embedding_dim=16,
            stec_num_heads=4,
            embedding_projection_dim=None  # Should force auto-adaptation
        )
        
        vocab_sizes = {'small': 10, 'large': 1000}
        feature_names = ['small', 'large']
        
        model = STECModel(vocab_sizes, feature_names, config)
        
        # Check if projection or adjustment was added
        self.assertTrue(getattr(model, 'use_projection', False) or model.dim_adjust is not None)
        
        # Forward pass
        x = torch.randint(0, 10, (4, 2))
        out = model(x)
        self.assertEqual(out['logits'].shape, (4, 1))
    
    def test_dimension_constraints(self):
        """Verify model adjusts or errors on incompatible dimensions."""
        # e.g. embed_dim not divisible by num_heads
        config = make_test_config(
            embedding_dim=17, # Prime number, hard to divide
            stec_num_heads=4,
            embedding_projection_dim=None
        )
        
        vocab_sizes = {'f1': 10}
        model = STECModel(vocab_sizes, ['f1'], config)
        
        # Should have added a projection to fix dimensions
        self.assertTrue(model.projection is not None or model.dim_adjust is not None)
        self.assertEqual(model.embed_per_field % 4, 0)


if __name__ == '__main__':
    unittest.main()

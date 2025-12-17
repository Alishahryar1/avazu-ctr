"""
Test suite for data processing functions.

This module tests all data processing functions including:
- Time feature extraction
- Vocabulary building and mapping
- User proxy features
- Interaction features
- Count features and binning
- Cumulative count features
- Hourly aggregated features
- Time-delta features
- Previous click count features
"""

import unittest

class TestDataProcessorTimeFeatures(unittest.TestCase):
    """Tests for data_processor time feature extraction."""
    
    def test_time_feature_expressions_output(self):
        """Test that time feature expressions produce correct output."""
        import polars as pl
        from src.processing.data_processor import get_time_feature_expressions
        
        # Create test data with known hour values (YYMMDDHH format)
        test_data = pl.DataFrame({
            'hour': ['14102100', '14102223', '14110105']  # 2014-10-21 00:00, 2014-10-22 23:00, 2014-11-01 05:00
        })
        
        time_exprs = get_time_feature_expressions()
        result = test_data.lazy().with_columns(time_exprs).collect()
        
        # Verify extracted values (year removed as it has zero variance)
        self.assertEqual(result['month'].to_list(), [10, 10, 11])
        self.assertEqual(result['day_of_month'].to_list(), [21, 22, 1])
        self.assertEqual(result['hour_of_day'].to_list(), [0, 23, 5])
    
    def test_time_feature_day_of_week(self):
        """Test day_of_week calculation."""
        import polars as pl
        from src.processing.data_processor import get_time_feature_expressions
        
        # 2014-10-21 was a Tuesday (weekday = 1 in Polars, 0=Monday)
        test_data = pl.DataFrame({
            'hour': ['14102100']
        })
        
        time_exprs = get_time_feature_expressions()
        result = test_data.lazy().with_columns(time_exprs).collect()
        
        # Tuesday = 2 (1-indexed in Polars dt.weekday())
        self.assertEqual(result['day_of_week'].to_list()[0], 2)
    
    def test_time_features_types(self):
        """Verify time features have correct types."""
        import polars as pl
        from src.processing.data_processor import get_time_feature_expressions
        
        test_data = pl.DataFrame({
            'hour': ['14102100']
        })
        
        time_exprs = get_time_feature_expressions()
        result = test_data.lazy().with_columns(time_exprs).collect()
        
        # year removed as it has zero variance in the dataset
        self.assertEqual(result['month'].dtype, pl.UInt8)
        self.assertEqual(result['day_of_month'].dtype, pl.UInt8)
        self.assertEqual(result['hour_of_day'].dtype, pl.UInt8)
        self.assertEqual(result['day_of_week'].dtype, pl.UInt8)


class TestDataProcessorVocabulary(unittest.TestCase):
    """Tests for vocabulary building functions."""
    
    def test_build_vocabularies_basic(self):
        """Test basic vocabulary building."""
        import polars as pl
        from src.processing.data_processor import build_vocabularies
        
        # Create test data
        test_data = pl.DataFrame({
            'cat1': ['a', 'b', 'a', 'c', 'a', 'b', 'a'],  # a=4, b=2, c=1
            'cat2': ['x', 'x', 'y', 'y', 'z', 'z', 'z'],  # x=2, y=2, z=3
        })
        
        vocab_sizes, feat_maps = build_vocabularies(
            test_data.lazy(), ['cat1', 'cat2'], min_freq=2
        )
        
        # cat1: a and b pass min_freq=2, c doesn't -> size = 2 + 1 (UNK) = 3
        self.assertEqual(vocab_sizes['cat1'], 3)
        # cat2: all pass min_freq=2 -> size = 3 + 1 (UNK) = 4
        self.assertEqual(vocab_sizes['cat2'], 4)
        
        # Verify mappings exist
        self.assertIn('a', feat_maps['cat1'])
        self.assertIn('b', feat_maps['cat1'])
        self.assertNotIn('c', feat_maps['cat1'])  # Filtered by min_freq
    
    def test_build_vocabularies_min_freq_filtering(self):
        """Test that min_freq correctly filters low-frequency values."""
        import polars as pl
        from src.processing.data_processor import build_vocabularies
        
        test_data = pl.DataFrame({
            'cat': ['a'] * 10 + ['b'] * 5 + ['c'] * 2 + ['d'] * 1
        })
        
        vocab_sizes, feat_maps = build_vocabularies(
            test_data.lazy(), ['cat'], min_freq=5
        )
        
        # Only 'a' (10) and 'b' (5) should pass min_freq=5
        self.assertEqual(vocab_sizes['cat'], 3)  # a, b + UNK
        self.assertIn('a', feat_maps['cat'])
        self.assertIn('b', feat_maps['cat'])
        self.assertNotIn('c', feat_maps['cat'])
        self.assertNotIn('d', feat_maps['cat'])
    
    def test_build_vocabularies_mapping_starts_at_one(self):
        """Verify vocabulary indices start at 1 (0 reserved for UNK)."""
        import polars as pl
        from src.processing.data_processor import build_vocabularies
        
        test_data = pl.DataFrame({
            'cat': ['x', 'y', 'z'] * 5
        })
        
        _, feat_maps = build_vocabularies(
            test_data.lazy(), ['cat'], min_freq=1
        )
        
        # All indices should be >= 1
        for val, idx in feat_maps['cat'].items():
            self.assertGreaterEqual(idx, 1)
        
        # Index 0 should not be used (reserved for UNK)
        self.assertNotIn(0, feat_maps['cat'].values())


class TestLazyVocabularyMapping(unittest.TestCase):
    """Tests for lazy vocabulary mapping functions."""
    
    def test_get_lazy_vocab_map_basic(self):
        """Test that lazy vocab map creates correct mappings."""
        import polars as pl
        from src.processing.data_processor import get_lazy_vocab_map
        
        # Create test data with known frequencies
        test_data = pl.DataFrame({
            'cat1': ['a', 'a', 'a', 'b', 'b', 'c']  # a=3, b=2, c=1
        }).lazy()
        
        vocab_lf = get_lazy_vocab_map(test_data, 'cat1', min_freq=2)
        vocab_df = vocab_lf.collect()
        
        # Only 'a' and 'b' should pass min_freq=2
        self.assertEqual(len(vocab_df), 2)
        self.assertIn('cat1', vocab_df.columns)
        self.assertIn('cat1_id', vocab_df.columns)
        
        # IDs should start at 1 (sorted: 'a' -> 1, 'b' -> 2)
        vocab_dict = dict(zip(vocab_df['cat1'].to_list(), vocab_df['cat1_id'].to_list()))
        self.assertEqual(vocab_dict['a'], 1)
        self.assertEqual(vocab_dict['b'], 2)
        self.assertNotIn('c', vocab_dict)  # Filtered by min_freq
    
    def test_apply_lazy_vocab_transforms_values(self):
        """Test that apply_lazy_vocab correctly transforms values to IDs."""
        import polars as pl
        from src.processing.data_processor import get_lazy_vocab_map, apply_lazy_vocab
        
        # Create training data
        train_data = pl.DataFrame({
            'cat1': ['a', 'a', 'a', 'b', 'b', 'c']
        }).lazy()
        
        # Build vocab from train
        vocab_lf = get_lazy_vocab_map(train_data, 'cat1', min_freq=1)
        
        # Apply to test data (including unknown value)
        test_data = pl.DataFrame({
            'cat1': ['a', 'b', 'c', 'unknown']
        }).lazy()
        
        result_lf = apply_lazy_vocab(test_data, vocab_lf, 'cat1')
        result = result_lf.collect()
        
        # 'a', 'b', 'c' should have IDs 1, 2, 3 (sorted)
        # 'unknown' should be 0 (UNK)
        self.assertEqual(result['cat1'].to_list(), [1, 2, 3, 0])
    
    def test_apply_lazy_vocab_fills_null_with_unk(self):
        """Test that unknown values get mapped to 0 (UNK)."""
        import polars as pl
        from src.processing.data_processor import get_lazy_vocab_map, apply_lazy_vocab
        
        # Train only has 'a' and 'b'
        train_data = pl.DataFrame({
            'cat1': ['a', 'a', 'b']
        }).lazy()
        
        vocab_lf = get_lazy_vocab_map(train_data, 'cat1', min_freq=1)
        
        # Test has values not in vocab
        test_data = pl.DataFrame({
            'cat1': ['x', 'y', 'z']
        }).lazy()
        
        result = apply_lazy_vocab(test_data, vocab_lf, 'cat1').collect()
        
        # All should be 0 (UNK)
        self.assertEqual(result['cat1'].to_list(), [0, 0, 0])
    
    def test_lazy_vocab_preserves_other_columns(self):
        """Test that apply_lazy_vocab preserves other columns."""
        import polars as pl
        from src.processing.data_processor import get_lazy_vocab_map, apply_lazy_vocab
        
        train_data = pl.DataFrame({
            'cat1': ['a', 'a', 'b'],
            'other': [1, 2, 3]
        }).lazy()
        
        vocab_lf = get_lazy_vocab_map(train_data, 'cat1', min_freq=1)
        
        test_data = pl.DataFrame({
            'cat1': ['a', 'b'],
            'other': [10, 20]
        }).lazy()
        
        result = apply_lazy_vocab(test_data, vocab_lf, 'cat1').collect()
        
        # 'other' column should be preserved
        self.assertIn('other', result.columns)
        self.assertEqual(result['other'].to_list(), [10, 20])
        # 'cat1' should now be integer IDs
        self.assertEqual(result['cat1'].dtype, pl.Int32)


# =============================================================================
# Tests for data_processor.py - User Proxy Feature
# =============================================================================
class TestUserProxyFeature(unittest.TestCase):
    """Tests for user proxy feature (device_ip + device_model)."""
    
    def test_user_proxy_expression_creates_combined_id(self):
        """Test that user proxy correctly combines device_ip and device_model."""
        import polars as pl
        from src.processing.data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '10.0.0.1', '192.168.1.1'],
            'device_model': ['iPhone12', 'Galaxy_S21', 'iPhone12']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        expected = ['192.168.1.1_iPhone12', '10.0.0.1_Galaxy_S21', '192.168.1.1_iPhone12']
        self.assertEqual(result['user_proxy'].to_list(), expected)
    
    def test_user_proxy_same_ip_different_model(self):
        """Test that same IP with different models creates different user proxies."""
        import polars as pl
        from src.processing.data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '192.168.1.1'],
            'device_model': ['iPhone12', 'iPhone13']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        proxies = result['user_proxy'].to_list()
        self.assertNotEqual(proxies[0], proxies[1])
        self.assertEqual(proxies[0], '192.168.1.1_iPhone12')
        self.assertEqual(proxies[1], '192.168.1.1_iPhone13')
    
    def test_user_proxy_different_ip_same_model(self):
        """Test that different IPs with same model creates different user proxies."""
        import polars as pl
        from src.processing.data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '10.0.0.1'],
            'device_model': ['iPhone12', 'iPhone12']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        proxies = result['user_proxy'].to_list()
        self.assertNotEqual(proxies[0], proxies[1])
    
    def test_user_proxy_with_empty_values(self):
        """Test user proxy handles empty/null values gracefully."""
        import polars as pl
        from src.processing.data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1', '', 'null'],
            'device_model': ['iPhone12', 'Galaxy', '']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        # Should still produce valid strings (even if containing empty parts)
        proxies = result['user_proxy'].to_list()
        self.assertEqual(len(proxies), 3)
        self.assertEqual(proxies[0], '192.168.1.1_iPhone12')
        self.assertEqual(proxies[1], '_Galaxy')
        self.assertEqual(proxies[2], 'null_')
    
    def test_user_proxy_with_special_characters(self):
        """Test user proxy handles special characters in values."""
        import polars as pl
        from src.processing.data_processor import get_user_proxy_expression
        
        test_data = pl.DataFrame({
            'device_ip': ['192.168.1.1'],
            'device_model': ['iPhone-12_Pro Max']
        })
        
        user_proxy_expr = get_user_proxy_expression()
        result = test_data.lazy().with_columns(user_proxy_expr).collect()
        
        proxies = result['user_proxy'].to_list()
        self.assertEqual(proxies[0], '192.168.1.1_iPhone-12_Pro Max')


# =============================================================================
# Tests for data_processor.py - Interaction Features
# =============================================================================
class TestInteractionFeatures(unittest.TestCase):
    """Tests for interaction feature creation (device_id_x_app_id, device_ip_x_C14)."""
    
    def test_interaction_expressions_create_correct_columns(self):
        """Test that interaction expressions create the expected columns."""
        import polars as pl
        from src.processing.data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_002'],
            'app_id': ['app_A', 'app_B'],
            'device_ip': ['192.168.1.1', '10.0.0.1'],
            'C14': ['14001', '14002']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        # Check columns exist
        self.assertIn('device_id_x_app_id', result.columns)
        self.assertIn('device_ip_x_C14', result.columns)
    
    def test_interaction_device_id_app_id(self):
        """Test device_id x app_id interaction feature values."""
        import polars as pl
        from src.processing.data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_002', 'dev_001'],
            'app_id': ['app_A', 'app_B', 'app_B'],
            'device_ip': ['ip1', 'ip2', 'ip3'],
            'C14': ['c1', 'c2', 'c3']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        expected = ['dev_001_app_A', 'dev_002_app_B', 'dev_001_app_B']
        self.assertEqual(result['device_id_x_app_id'].to_list(), expected)
    
    def test_interaction_device_ip_c14(self):
        """Test device_ip x C14 interaction feature values."""
        import polars as pl
        from src.processing.data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_002'],
            'app_id': ['app_A', 'app_B'],
            'device_ip': ['192.168.1.1', '10.0.0.1'],
            'C14': ['14001', '14002']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        expected = ['192.168.1.1_14001', '10.0.0.1_14002']
        self.assertEqual(result['device_ip_x_C14'].to_list(), expected)
    
    def test_interaction_uniqueness(self):
        """Test that different input combinations produce different interaction values."""
        import polars as pl
        from src.processing.data_processor import get_interaction_feature_expressions
        
        test_data = pl.DataFrame({
            'device_id': ['dev_001', 'dev_001', 'dev_002', 'dev_002'],
            'app_id': ['app_A', 'app_B', 'app_A', 'app_B'],
            'device_ip': ['ip1', 'ip1', 'ip1', 'ip1'],
            'C14': ['c1', 'c1', 'c1', 'c1']
        })
        
        interaction_exprs = get_interaction_feature_expressions()
        result = test_data.lazy().with_columns(interaction_exprs).collect()
        
        interactions = result['device_id_x_app_id'].to_list()
        # All 4 combinations should be unique
        self.assertEqual(len(set(interactions)), 4)


# =============================================================================
# Tests for data_processor.py - Count Features
# =============================================================================
class TestCountFeatures(unittest.TestCase):
    """Tests for count/frequency feature computation."""
    
    def test_compute_count_features_basic(self):
        """Test basic count feature computation."""
        import polars as pl
        from src.processing.data_processor import compute_count_features_from_train
        
        # Create train data with known frequencies
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip1', 'ip2', 'ip2', 'ip3']  # ip1=3, ip2=2, ip3=1
        }).lazy()
        
        # Test data with same and new values
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip2', 'ip4']  # ip4 is new (count=0)
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip']
        )
        
        train_result = lf_train.collect()
        test_result = lf_test.collect()
        
        # Verify train counts
        self.assertIn('device_ip_count', train_result.columns)
        train_counts = train_result['device_ip_count'].to_list()
        self.assertEqual(train_counts, [3, 3, 3, 2, 2, 1])
        
        # Verify test counts (based on train frequencies)
        test_counts = test_result['device_ip_count'].to_list()
        self.assertEqual(test_counts, [3, 2, 0])  # ip4 has count 0 (not in train)
    
    def test_compute_count_features_multiple_columns(self):
        """Test count features for multiple columns."""
        import polars as pl
        from src.processing.data_processor import compute_count_features_from_train
        
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip2'],
            'C14': ['c1', 'c2', 'c1']  # c1=2, c2=1
        }).lazy()
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip3'],
            'C14': ['c1', 'c2']
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip', 'C14']
        )
        
        test_result = lf_test.collect()
        
        self.assertIn('device_ip_count', test_result.columns)
        self.assertIn('C14_count', test_result.columns)
        
        self.assertEqual(test_result['device_ip_count'].to_list(), [2, 0])
        self.assertEqual(test_result['C14_count'].to_list(), [2, 1])
    
    def test_compute_count_features_no_data_leakage(self):
        """Test that count features don't leak test data into training stats."""
        import polars as pl
        from src.processing.data_processor import compute_count_features_from_train
        
        # Train has ip1 only
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1']
        }).lazy()
        
        # Test has ip2 only (not in train)
        test_data = pl.DataFrame({
            'device_ip': ['ip2', 'ip2', 'ip2']
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip']
        )
        
        test_result = lf_test.collect()
        
        # ip2 should have count 0 (not in train), not 3 (from test)
        test_counts = test_result['device_ip_count'].to_list()
        self.assertEqual(test_counts, [0, 0, 0])
    
    def test_count_features_dtype(self):
        """Verify count features have correct data type (UInt32)."""
        import polars as pl
        from src.processing.data_processor import compute_count_features_from_train
        
        train_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip2']
        }).lazy()
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1']
        }).lazy()
        
        lf_train, lf_test = compute_count_features_from_train(
            train_data, test_data, ['device_ip']
        )
        
        train_result = lf_train.collect()
        self.assertEqual(train_result['device_ip_count'].dtype, pl.UInt32)


# =============================================================================
# Tests for data_processor.py - Count Binning
# =============================================================================
class TestCountBinning(unittest.TestCase):
    """Tests for count feature binning."""
    
    def test_bin_count_features_basic(self):
        """Test basic count binning."""
        import polars as pl
        from src.processing.data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [0, 1, 3, 7, 25, 75, 250, 750, 2000]
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        expected_bins = ['0', '1', '2-5', '6-10', '11-50', '51-100', '101-500', '501-1000', '1000+']
        self.assertEqual(result['device_ip_count_bin'].to_list(), expected_bins)
    
    def test_bin_count_features_boundary_values(self):
        """Test binning at exact boundary values."""
        import polars as pl
        from src.processing.data_processor import bin_count_features
        
        # Test exact boundary values
        test_data = pl.DataFrame({
            'device_ip_count': [0, 1, 5, 6, 10, 11, 50, 51, 100, 101, 500, 501, 1000, 1001]
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        bins = result['device_ip_count_bin'].to_list()
        
        # Verify boundaries
        self.assertEqual(bins[0], '0')       # 0
        self.assertEqual(bins[1], '1')       # 1
        self.assertEqual(bins[2], '2-5')     # 5 (upper bound of 2-5)
        self.assertEqual(bins[3], '6-10')    # 6 (lower edge of 6-10)
        self.assertEqual(bins[4], '6-10')    # 10 (upper bound of 6-10)
        self.assertEqual(bins[5], '11-50')   # 11 (lower edge)
        self.assertEqual(bins[6], '11-50')   # 50 (upper bound)
        self.assertEqual(bins[7], '51-100')  # 51
        self.assertEqual(bins[8], '51-100')  # 100
        self.assertEqual(bins[9], '101-500') # 101
        self.assertEqual(bins[10], '101-500') # 500
        self.assertEqual(bins[11], '501-1000') # 501
        self.assertEqual(bins[12], '501-1000') # 1000
        self.assertEqual(bins[13], '1000+')    # 1001
    
    def test_bin_count_features_multiple_columns(self):
        """Test binning for multiple columns."""
        import polars as pl
        from src.processing.data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [0, 100, 5000],
            'C14_count': [1, 50, 1500]
        })
        
        bin_exprs = bin_count_features(['device_ip', 'C14'])
        result = test_data.with_columns(bin_exprs)
        
        self.assertIn('device_ip_count_bin', result.columns)
        self.assertIn('C14_count_bin', result.columns)
        
        self.assertEqual(result['device_ip_count_bin'].to_list(), ['0', '51-100', '1000+'])
        self.assertEqual(result['C14_count_bin'].to_list(), ['1', '11-50', '1000+'])
    
    def test_bin_count_features_all_same_bin(self):
        """Test when all values fall in the same bin."""
        import polars as pl
        from src.processing.data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [3, 4, 5, 2, 3]  # All in '2-5' bin
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        bins = result['device_ip_count_bin'].to_list()
        self.assertTrue(all(b == '2-5' for b in bins))
    
    def test_bin_count_features_output_type(self):
        """Verify binned features are string type (for categorical encoding)."""
        import polars as pl
        from src.processing.data_processor import bin_count_features
        
        test_data = pl.DataFrame({
            'device_ip_count': [0, 100, 5000]
        })
        
        bin_exprs = bin_count_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        # Binned columns should be strings (for later label encoding)
        self.assertEqual(result['device_ip_count_bin'].dtype, pl.String)


# =============================================================================
# Tests for data_processor.py - Cumulative Count Features
# =============================================================================
class TestCumulativeCountFeatures(unittest.TestCase):
    """Tests for cumulative count feature computation."""
    
    def test_cumulative_count_basic(self):
        """Test basic cumulative count computation."""
        import polars as pl
        from src.processing.data_processor import get_cumulative_count_expressions
        
        # Create test data with repeated values
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip2', 'ip1', 'ip1', 'ip2']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip'])
        result = test_data.with_columns(cumcount_exprs)
        
        # ip1 appears at positions 0, 2, 3 -> cumcount should be 1, 2, 3
        # ip2 appears at positions 1, 4 -> cumcount should be 1, 2
        expected = [1, 1, 2, 3, 2]
        self.assertEqual(result['device_ip_cumcount'].to_list(), expected)
    
    def test_cumulative_count_multiple_columns(self):
        """Test cumulative count for multiple columns."""
        import polars as pl
        from src.processing.data_processor import get_cumulative_count_expressions
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip2'],
            'user_proxy': ['u1', 'u2', 'u1']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip', 'user_proxy'])
        result = test_data.with_columns(cumcount_exprs)
        
        self.assertEqual(result['device_ip_cumcount'].to_list(), [1, 2, 1])
        self.assertEqual(result['user_proxy_cumcount'].to_list(), [1, 1, 2])
    
    def test_cumulative_count_all_unique(self):
        """Test cumulative count when all values are unique."""
        import polars as pl
        from src.processing.data_processor import get_cumulative_count_expressions
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip2', 'ip3', 'ip4']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip'])
        result = test_data.with_columns(cumcount_exprs)
        
        # All unique values should have cumcount of 1
        self.assertEqual(result['device_ip_cumcount'].to_list(), [1, 1, 1, 1])
    
    def test_cumulative_count_all_same(self):
        """Test cumulative count when all values are the same."""
        import polars as pl
        from src.processing.data_processor import get_cumulative_count_expressions
        
        test_data = pl.DataFrame({
            'device_ip': ['ip1', 'ip1', 'ip1', 'ip1']
        })
        
        cumcount_exprs = get_cumulative_count_expressions(['device_ip'])
        result = test_data.with_columns(cumcount_exprs)
        
        # All same values should have increasing cumcount
        self.assertEqual(result['device_ip_cumcount'].to_list(), [1, 2, 3, 4])


class TestCumulativeCountBinning(unittest.TestCase):
    """Tests for cumulative count binning."""
    
    def test_bin_cumcount_basic(self):
        """Test basic cumulative count binning."""
        import polars as pl
        from src.processing.data_processor import bin_cumcount_features
        
        test_data = pl.DataFrame({
            'device_ip_cumcount': [1, 2, 5, 15, 75, 150]
        })
        
        bin_exprs = bin_cumcount_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        expected = ['first', '2-3', '4-10', '11-50', '51-100', '100+']
        self.assertEqual(result['device_ip_cumcount_bin'].to_list(), expected)
    
    def test_bin_cumcount_boundary_values(self):
        """Test binning at exact boundaries."""
        import polars as pl
        from src.processing.data_processor import bin_cumcount_features
        
        test_data = pl.DataFrame({
            'device_ip_cumcount': [1, 3, 4, 10, 11, 50, 51, 100, 101]
        })
        
        bin_exprs = bin_cumcount_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        bins = result['device_ip_cumcount_bin'].to_list()
        self.assertEqual(bins[0], 'first')   # 1
        self.assertEqual(bins[1], '2-3')     # 3
        self.assertEqual(bins[2], '4-10')    # 4
        self.assertEqual(bins[3], '4-10')    # 10
        self.assertEqual(bins[4], '11-50')   # 11
        self.assertEqual(bins[5], '11-50')   # 50
        self.assertEqual(bins[6], '51-100')  # 51
        self.assertEqual(bins[7], '51-100')  # 100
        self.assertEqual(bins[8], '100+')    # 101
    
    def test_bin_cumcount_output_type(self):
        """Verify binned features are string type."""
        import polars as pl
        from src.processing.data_processor import bin_cumcount_features
        
        test_data = pl.DataFrame({
            'device_ip_cumcount': [1, 50, 200]
        })
        
        bin_exprs = bin_cumcount_features(['device_ip'])
        result = test_data.with_columns(bin_exprs)
        
        self.assertEqual(result['device_ip_cumcount_bin'].dtype, pl.String)


# =============================================================================
# Tests for data_processor.py - Hourly Aggregated Features
# =============================================================================
class TestHourlyAggregatedFeatures(unittest.TestCase):
    """Tests for hourly aggregated feature computation."""
    
    def test_compute_hourly_features_basic(self):
        """Test basic hourly aggregated feature computation."""
        import polars as pl
        from src.processing.data_processor import compute_hourly_aggregated_features
        
        # Create train data with known user-hour patterns
        train_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1', 'u1', 'u2', 'u2'],
            'hour': ['14102100', '14102100', '14102101', '14102100', '14102100']
        }).lazy()
        
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u3'],
            'hour': ['14102100', '14102100', '14102100']
        }).lazy()
        
        lf_train, lf_test = compute_hourly_aggregated_features(train_data, test_data)
        
        train_result = lf_train.collect()
        test_result = lf_test.collect()
        
        # u1 in hour 14102100 appears 2 times in train
        # u2 in hour 14102100 appears 2 times in train
        # u1 in hour 14102101 appears 1 time in train
        self.assertIn('user_hourly_impressions', train_result.columns)
        
        # Test data: u1@14102100 -> 2, u2@14102100 -> 2, u3@14102100 -> 1 (unknown)
        test_impressions = test_result['user_hourly_impressions'].to_list()
        self.assertEqual(test_impressions[0], 2)  # u1@14102100
        self.assertEqual(test_impressions[1], 2)  # u2@14102100
        self.assertEqual(test_impressions[2], 1)  # u3@14102100 (not in train, defaults to 1)
    
    def test_hourly_features_no_leakage(self):
        """Test that hourly features don't leak test data."""
        import polars as pl
        from src.processing.data_processor import compute_hourly_aggregated_features
        
        train_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1'],
            'hour': ['14102100', '14102100']
        }).lazy()
        
        # Test has u2 who's not in train
        test_data = pl.DataFrame({
            'user_proxy': ['u2', 'u2', 'u2'],
            'hour': ['14102100', '14102100', '14102100']
        }).lazy()
        
        _, lf_test = compute_hourly_aggregated_features(train_data, test_data)
        test_result = lf_test.collect()
        
        # u2 should have count 1 (default), not 3 (from test)
        self.assertTrue(all(c == 1 for c in test_result['user_hourly_impressions'].to_list()))


class TestHourlyImpressionsBinning(unittest.TestCase):
    """Tests for hourly impressions binning."""
    
    def test_bin_hourly_impressions_basic(self):
        """Test basic hourly impressions binning.
        
        EDA-optimized bins:
        - 'single' (1): Most common
        - '2' (2): Returning within hour
        - '3-4' (3-4): Up to P90
        - '5+': High-frequency users
        """
        import polars as pl
        from src.processing.data_processor import bin_hourly_impressions
        
        test_data = pl.DataFrame({
            'user_hourly_impressions': [1, 2, 3, 4, 5]
        })
        
        bin_expr = bin_hourly_impressions()
        result = test_data.with_columns(bin_expr)
        
        expected = ['single', '2', '3-4', '3-4', '5+']
        self.assertEqual(result['user_hourly_impressions_bin'].to_list(), expected)
    
    def test_bin_hourly_impressions_boundaries(self):
        """Test binning at exact EDA-optimized boundaries."""
        import polars as pl
        from src.processing.data_processor import bin_hourly_impressions
        
        test_data = pl.DataFrame({
            'user_hourly_impressions': [1, 2, 3, 4, 5, 10]
        })
        
        bin_expr = bin_hourly_impressions()
        result = test_data.with_columns(bin_expr)
        
        bins = result['user_hourly_impressions_bin'].to_list()
        self.assertEqual(bins[0], 'single')  # 1
        self.assertEqual(bins[1], '2')       # 2
        self.assertEqual(bins[2], '3-4')     # 3
        self.assertEqual(bins[3], '3-4')     # 4
        self.assertEqual(bins[4], '5+')      # 5
        self.assertEqual(bins[5], '5+')      # 10
    
    def test_bin_hourly_impressions_output_type(self):
        """Verify binned feature is string type."""
        import polars as pl
        from src.processing.data_processor import bin_hourly_impressions
        
        test_data = pl.DataFrame({
            'user_hourly_impressions': [1, 10, 100]
        })
        
        bin_expr = bin_hourly_impressions()
        result = test_data.with_columns(bin_expr)
        
        self.assertEqual(result['user_hourly_impressions_bin'].dtype, pl.String)


# =============================================================================
# Tests for data_processor.py - Time-Delta Features
# =============================================================================
class TestTimeDeltaFeatures(unittest.TestCase):
    """Tests for time-delta feature computation (hours since last click)."""
    
    def test_compute_time_delta_basic(self):
        """Test basic time delta computation."""
        import polars as pl
        from src.processing.data_processor import compute_time_delta_features
        
        # Create test data with known time sequence for same user
        # user u1 clicks at hours 00, 01, 05 -> deltas should be 0, 1, 4
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1', 'u1'],
            'hour': ['14102100', '14102101', '14102105']
        }).lazy()
        
        result = compute_time_delta_features(test_data, group_col='user_proxy').collect()
        
        self.assertIn('hours_since_last_click', result.columns)
        deltas = result['hours_since_last_click'].to_list()
        self.assertEqual(deltas[0], 0)  # First click, no previous
        self.assertEqual(deltas[1], 1)  # 1 hour after first
        self.assertEqual(deltas[2], 4)  # 4 hours after second
    
    def test_compute_time_delta_multiple_users(self):
        """Test time delta computation with multiple users."""
        import polars as pl
        from src.processing.data_processor import compute_time_delta_features
        
        # Two users with different click patterns
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u1', 'u2'],
            'hour': ['14102100', '14102100', '14102102', '14102110']
        }).lazy()
        
        result = compute_time_delta_features(test_data, group_col='user_proxy').collect()
        deltas = result['hours_since_last_click'].to_list()
        
        # u1: first click (0), then 2 hours later
        # u2: first click (0), then 10 hours later
        self.assertEqual(deltas[0], 0)   # u1 first
        self.assertEqual(deltas[1], 0)   # u2 first
        self.assertEqual(deltas[2], 2)   # u1 second, 2 hours after first
        self.assertEqual(deltas[3], 10)  # u2 second, 10 hours after first
    
    def test_compute_time_delta_across_days(self):
        """Test time delta computation across different days."""
        import polars as pl
        from src.processing.data_processor import compute_time_delta_features
        
        # User clicks on different days
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1'],
            'hour': ['14102100', '14102200']  # Oct 21 00:00 -> Oct 22 00:00 = 24 hours
        }).lazy()
        
        result = compute_time_delta_features(test_data, group_col='user_proxy').collect()
        deltas = result['hours_since_last_click'].to_list()
        
        self.assertEqual(deltas[0], 0)   # First click
        self.assertEqual(deltas[1], 24)  # 24 hours later


class TestTimeDeltaBinning(unittest.TestCase):
    """Tests for time delta binning."""
    
    def test_bin_time_delta_basic(self):
        """Test basic time delta binning.
        
        EDA-optimized bins:
        - 'first': First click (0 hours)
        - '1-4h': Short interval (<=4)
        - '5-19h': Medium interval (<=19)
        - '20-53h': Long interval (<=53)
        - '>53h': Re-engagement
        """
        import polars as pl
        from src.processing.data_processor import bin_time_delta_features
        
        test_data = pl.DataFrame({
            'hours_since_last_click': [0, 3, 15, 50, 100]
        })
        
        bin_expr = bin_time_delta_features()
        result = test_data.with_columns(bin_expr)
        
        bins = result['hours_since_last_click_bin'].to_list()
        self.assertEqual(bins[0], 'first')   # 0
        self.assertEqual(bins[1], '1-4h')    # 3 (<=4)
        self.assertEqual(bins[2], '5-19h')   # 15 (<=19)
        self.assertEqual(bins[3], '20-53h')  # 50 (<=53)
        self.assertEqual(bins[4], '>53h')    # 100 (>53)
    
    def test_bin_time_delta_boundaries(self):
        """Test binning at exact EDA-optimized boundaries."""
        import polars as pl
        from src.processing.data_processor import bin_time_delta_features
        
        test_data = pl.DataFrame({
            'hours_since_last_click': [0, 4, 5, 19, 20, 53, 54]
        })
        
        bin_expr = bin_time_delta_features()
        result = test_data.with_columns(bin_expr)
        
        bins = result['hours_since_last_click_bin'].to_list()
        self.assertEqual(bins[0], 'first')   # 0
        self.assertEqual(bins[1], '1-4h')    # 4 (boundary)
        self.assertEqual(bins[2], '5-19h')   # 5 (boundary)
        self.assertEqual(bins[3], '5-19h')   # 19 (boundary)
        self.assertEqual(bins[4], '20-53h')  # 20 (boundary)
        self.assertEqual(bins[5], '20-53h')  # 53 (boundary)
        self.assertEqual(bins[6], '>53h')    # 54


# =============================================================================
# Tests for data_processor.py - Previous Click Count Features
# =============================================================================
class TestPreviousClickCount(unittest.TestCase):
    """Tests for previous click count feature computation."""
    
    def test_compute_prev_click_count_basic(self):
        """Test basic previous click count computation."""
        import polars as pl
        from src.processing.data_processor import compute_previous_click_count
        
        # User makes 4 clicks -> prev counts should be 0, 1, 2, 3
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u1', 'u1', 'u1']
        }).lazy()
        
        result = compute_previous_click_count(test_data, group_col='user_proxy').collect()
        
        self.assertIn('user_proxy_prev_clicks', result.columns)
        prev_clicks = result['user_proxy_prev_clicks'].to_list()
        self.assertEqual(prev_clicks, [0, 1, 2, 3])
    
    def test_compute_prev_click_count_multiple_users(self):
        """Test previous click count with multiple users."""
        import polars as pl
        from src.processing.data_processor import compute_previous_click_count
        
        # Two users with different click counts
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u1', 'u2', 'u1']
        }).lazy()
        
        result = compute_previous_click_count(test_data, group_col='user_proxy').collect()
        prev_clicks = result['user_proxy_prev_clicks'].to_list()
        
        # u1: 0, 1, 2 (positions 0, 2, 4)
        # u2: 0, 1 (positions 1, 3)
        self.assertEqual(prev_clicks[0], 0)  # u1 first
        self.assertEqual(prev_clicks[1], 0)  # u2 first
        self.assertEqual(prev_clicks[2], 1)  # u1 second
        self.assertEqual(prev_clicks[3], 1)  # u2 second
        self.assertEqual(prev_clicks[4], 2)  # u1 third
    
    def test_prev_click_count_all_unique(self):
        """Test previous click count when all users are unique."""
        import polars as pl
        from src.processing.data_processor import compute_previous_click_count
        
        test_data = pl.DataFrame({
            'user_proxy': ['u1', 'u2', 'u3', 'u4']
        }).lazy()
        
        result = compute_previous_click_count(test_data, group_col='user_proxy').collect()
        prev_clicks = result['user_proxy_prev_clicks'].to_list()
        
        # All first-time users should have 0 previous clicks
        self.assertEqual(prev_clicks, [0, 0, 0, 0])


class TestPreviousClicksBinning(unittest.TestCase):
    """Tests for previous clicks binning."""
    
    def test_bin_prev_clicks_basic(self):
        """Test basic previous clicks binning.
        
        EDA-optimized bins:
        - 'new' (0): First-time user
        - 'returning' (1-7): Up to P50
        - 'regular' (8-32): P50 to P75
        - 'heavy' (33-224): P75 to P90
        - 'power' (>224): Top 10% most active
        """
        import polars as pl
        from src.processing.data_processor import bin_prev_clicks
        
        test_data = pl.DataFrame({
            'user_proxy_prev_clicks': [0, 3, 15, 75, 250]
        })
        
        bin_expr = bin_prev_clicks('user_proxy')
        result = test_data.with_columns(bin_expr)
        
        bins = result['user_proxy_prev_clicks_bin'].to_list()
        self.assertEqual(bins[0], 'new')       # 0
        self.assertEqual(bins[1], 'returning') # 3 (<=7)
        self.assertEqual(bins[2], 'regular')   # 15 (<=32)
        self.assertEqual(bins[3], 'heavy')     # 75 (<=224)
        self.assertEqual(bins[4], 'power')     # 250 (>224)
    
    def test_bin_prev_clicks_boundaries(self):
        """Test binning at exact EDA-optimized boundaries."""
        import polars as pl
        from src.processing.data_processor import bin_prev_clicks
        
        test_data = pl.DataFrame({
            'user_proxy_prev_clicks': [0, 7, 8, 32, 33, 224, 225]
        })
        
        bin_expr = bin_prev_clicks('user_proxy')
        result = test_data.with_columns(bin_expr)
        
        bins = result['user_proxy_prev_clicks_bin'].to_list()
        self.assertEqual(bins[0], 'new')       # 0
        self.assertEqual(bins[1], 'returning') # 7 (boundary)
        self.assertEqual(bins[2], 'regular')   # 8 (boundary)
        self.assertEqual(bins[3], 'regular')   # 32 (boundary)
        self.assertEqual(bins[4], 'heavy')     # 33 (boundary)
        self.assertEqual(bins[5], 'heavy')     # 224 (boundary)
        self.assertEqual(bins[6], 'power')     # 225
    
    def test_bin_prev_clicks_output_type(self):
        """Verify binned feature is string type."""
        import polars as pl
        from src.processing.data_processor import bin_prev_clicks
        
        test_data = pl.DataFrame({
            'user_proxy_prev_clicks': [0, 50, 200]
        })
        
        bin_expr = bin_prev_clicks('user_proxy')
        result = test_data.with_columns(bin_expr)
        
        self.assertEqual(result['user_proxy_prev_clicks_bin'].dtype, pl.String)


if __name__ == "__main__":
    unittest.main(verbosity=2)

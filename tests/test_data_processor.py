"""
Test suite for data processor module.

Tests feature engineering expressions, binning functions, vocabulary mapping,
and pipeline helpers.
"""

import unittest
import polars as pl


class TestTimeFeatures(unittest.TestCase):
    """Tests for time feature extraction."""

    def test_time_feature_expressions_output(self):
        """Test time features produce correct output from YYMMDDHH format."""
        from src.processing.data_processor import get_time_feature_expressions

        test_data = pl.DataFrame({"hour": ["14102100", "14102223", "14110105"]})

        result = test_data.lazy().with_columns(get_time_feature_expressions()).collect()

        self.assertEqual(result["month"].to_list(), [10, 10, 11])
        self.assertEqual(result["day_of_month"].to_list(), [21, 22, 1])
        self.assertEqual(result["hour_of_day"].to_list(), [0, 23, 5])

    def test_time_feature_day_of_week(self):
        """Test day_of_week calculation (2014-10-21 was Tuesday = 2)."""
        from src.processing.data_processor import get_time_feature_expressions

        test_data = pl.DataFrame({"hour": ["14102100"]})
        result = test_data.lazy().with_columns(get_time_feature_expressions()).collect()

        self.assertEqual(result["day_of_week"].to_list()[0], 2)

    def test_time_features_types(self):
        """Verify time features are UInt8."""
        from src.processing.data_processor import get_time_feature_expressions

        test_data = pl.DataFrame({"hour": ["14102100"]})
        result = test_data.lazy().with_columns(get_time_feature_expressions()).collect()

        for col in ["month", "day_of_month", "hour_of_day", "day_of_week"]:
            self.assertEqual(result[col].dtype, pl.UInt8)

    def test_time_features_edge_cases(self):
        """Test edge case times: midnight, month boundaries."""
        from src.processing.data_processor import get_time_feature_expressions

        test_data = pl.DataFrame(
            {"hour": ["14010100", "14123123"]}  # Jan 1 00:00, Dec 31 23:00
        )

        result = test_data.lazy().with_columns(get_time_feature_expressions()).collect()

        self.assertEqual(result["month"].to_list(), [1, 12])
        self.assertEqual(result["day_of_month"].to_list(), [1, 31])
        self.assertEqual(result["hour_of_day"].to_list(), [0, 23])


class TestUserProxyFeature(unittest.TestCase):
    """Tests for user proxy feature (device_ip + device_model)."""

    def test_user_proxy_creates_combined_id(self):
        """Test user proxy correctly combines device_ip and device_model."""
        from src.processing.data_processor import get_user_proxy_expression

        test_data = pl.DataFrame(
            {
                "device_ip": ["192.168.1.1", "10.0.0.1", "192.168.1.1"],
                "device_model": ["iPhone12", "Galaxy_S21", "iPhone12"],
            }
        )

        result = test_data.lazy().with_columns(get_user_proxy_expression()).collect()

        expected = [
            "192.168.1.1_iPhone12",
            "10.0.0.1_Galaxy_S21",
            "192.168.1.1_iPhone12",
        ]
        self.assertEqual(result["user_proxy"].to_list(), expected)

    def test_user_proxy_uniqueness(self):
        """Test same IP + different model creates different proxies."""
        from src.processing.data_processor import get_user_proxy_expression

        test_data = pl.DataFrame(
            {
                "device_ip": ["192.168.1.1", "192.168.1.1"],
                "device_model": ["iPhone12", "iPhone13"],
            }
        )

        result = test_data.lazy().with_columns(get_user_proxy_expression()).collect()
        proxies = result["user_proxy"].to_list()

        self.assertNotEqual(proxies[0], proxies[1])

    def test_user_proxy_with_empty_values(self):
        """Test user proxy handles empty values gracefully."""
        from src.processing.data_processor import get_user_proxy_expression

        test_data = pl.DataFrame(
            {"device_ip": ["192.168.1.1", ""], "device_model": ["iPhone12", "Galaxy"]}
        )

        result = test_data.lazy().with_columns(get_user_proxy_expression()).collect()
        proxies = result["user_proxy"].to_list()

        self.assertEqual(proxies[0], "192.168.1.1_iPhone12")
        self.assertEqual(proxies[1], "_Galaxy")


class TestInteractionFeatures(unittest.TestCase):
    """Tests for interaction features."""

    def test_interaction_creates_correct_columns(self):
        """Test interaction expressions create expected columns."""
        from src.processing.data_processor import get_interaction_feature_expressions

        test_data = pl.DataFrame(
            {
                "device_id": ["dev_001", "dev_002"],
                "app_id": ["app_A", "app_B"],
                "device_ip": ["192.168.1.1", "10.0.0.1"],
                "C14": ["14001", "14002"],
            }
        )

        result = (
            test_data.lazy()
            .with_columns(get_interaction_feature_expressions())
            .collect()
        )

        self.assertIn("device_id_x_app_id", result.columns)
        self.assertIn("device_ip_x_C14", result.columns)

    def test_interaction_values(self):
        """Test interaction feature values are correctly formed."""
        from src.processing.data_processor import get_interaction_feature_expressions

        test_data = pl.DataFrame(
            {
                "device_id": ["dev_001", "dev_002"],
                "app_id": ["app_A", "app_B"],
                "device_ip": ["192.168.1.1", "10.0.0.1"],
                "C14": ["14001", "14002"],
            }
        )

        result = (
            test_data.lazy()
            .with_columns(get_interaction_feature_expressions())
            .collect()
        )

        self.assertEqual(
            result["device_id_x_app_id"].to_list(), ["dev_001_app_A", "dev_002_app_B"]
        )
        self.assertEqual(
            result["device_ip_x_C14"].to_list(), ["192.168.1.1_14001", "10.0.0.1_14002"]
        )


class TestCumulativeCountFeatures(unittest.TestCase):
    """Tests for cumulative count features."""

    def test_cumulative_count_basic(self):
        """Test basic cumulative count computation."""
        from src.processing.data_processor import get_cumulative_count_expressions

        test_data = pl.DataFrame({"device_ip": ["ip1", "ip2", "ip1", "ip1", "ip2"]})
        result = test_data.with_columns(get_cumulative_count_expressions(["device_ip"]))

        expected = [1, 1, 2, 3, 2]
        self.assertEqual(result["device_ip_cumcount"].to_list(), expected)

    def test_cumulative_count_all_unique(self):
        """Test cumulative count when all values are unique."""
        from src.processing.data_processor import get_cumulative_count_expressions

        test_data = pl.DataFrame({"device_ip": ["ip1", "ip2", "ip3", "ip4"]})
        result = test_data.with_columns(get_cumulative_count_expressions(["device_ip"]))

        self.assertEqual(result["device_ip_cumcount"].to_list(), [1, 1, 1, 1])

    def test_cumulative_count_all_same(self):
        """Test cumulative count when all values are the same."""
        from src.processing.data_processor import get_cumulative_count_expressions

        test_data = pl.DataFrame({"device_ip": ["ip1", "ip1", "ip1", "ip1"]})
        result = test_data.with_columns(get_cumulative_count_expressions(["device_ip"]))

        self.assertEqual(result["device_ip_cumcount"].to_list(), [1, 2, 3, 4])


class TestCumulativeCountBinning(unittest.TestCase):
    """Tests for cumulative count binning."""

    def test_bin_cumcount_basic(self):
        """Test basic cumulative count binning."""
        from src.processing.data_processor import bin_cumcount_features

        test_data = pl.DataFrame({"device_ip_cumcount": [1, 2, 5, 15, 75, 150]})
        result = test_data.with_columns(bin_cumcount_features(["device_ip"]))

        expected = ["first", "2-3", "4-10", "11-50", "51-100", "100+"]
        self.assertEqual(result["device_ip_cumcount_bin"].to_list(), expected)

    def test_bin_cumcount_boundaries(self):
        """Test binning at exact boundaries."""
        from src.processing.data_processor import bin_cumcount_features

        test_data = pl.DataFrame(
            {"device_ip_cumcount": [1, 3, 4, 10, 11, 50, 51, 100, 101]}
        )
        result = test_data.with_columns(bin_cumcount_features(["device_ip"]))

        bins = result["device_ip_cumcount_bin"].to_list()
        self.assertEqual(
            bins,
            [
                "first",
                "2-3",
                "4-10",
                "4-10",
                "11-50",
                "11-50",
                "51-100",
                "51-100",
                "100+",
            ],
        )

    def test_bin_cumcount_output_type(self):
        """Verify binned features are string type."""
        from src.processing.data_processor import bin_cumcount_features

        test_data = pl.DataFrame({"device_ip_cumcount": [1, 50, 200]})
        result = test_data.with_columns(bin_cumcount_features(["device_ip"]))

        self.assertEqual(result["device_ip_cumcount_bin"].dtype, pl.String)


class TestCountBinning(unittest.TestCase):
    """Tests for count feature binning."""

    def test_bin_count_features_basic(self):
        """Test basic count binning."""
        from src.processing.data_processor import bin_count_features

        test_data = pl.DataFrame(
            {"device_ip_count": [0, 1, 3, 7, 25, 75, 250, 750, 2000]}
        )
        result = test_data.with_columns(bin_count_features(["device_ip"]))

        expected = [
            "0",
            "1",
            "2-5",
            "6-10",
            "11-50",
            "51-100",
            "101-500",
            "501-1000",
            "1000+",
        ]
        self.assertEqual(result["device_ip_count_bin"].to_list(), expected)

    def test_bin_count_features_boundaries(self):
        """Test binning at exact boundaries."""
        from src.processing.data_processor import bin_count_features

        test_data = pl.DataFrame(
            {
                "device_ip_count": [
                    0,
                    1,
                    5,
                    6,
                    10,
                    11,
                    50,
                    51,
                    100,
                    101,
                    500,
                    501,
                    1000,
                    1001,
                ]
            }
        )
        result = test_data.with_columns(bin_count_features(["device_ip"]))

        bins = result["device_ip_count_bin"].to_list()
        expected = [
            "0",
            "1",
            "2-5",
            "6-10",
            "6-10",
            "11-50",
            "11-50",
            "51-100",
            "51-100",
            "101-500",
            "101-500",
            "501-1000",
            "501-1000",
            "1000+",
        ]
        self.assertEqual(bins, expected)

    def test_bin_count_features_multiple_columns(self):
        """Test binning for multiple columns."""
        from src.processing.data_processor import bin_count_features

        test_data = pl.DataFrame(
            {"device_ip_count": [0, 100, 5000], "C14_count": [1, 50, 1500]}
        )
        result = test_data.with_columns(bin_count_features(["device_ip", "C14"]))

        self.assertEqual(
            result["device_ip_count_bin"].to_list(), ["0", "51-100", "1000+"]
        )
        self.assertEqual(result["C14_count_bin"].to_list(), ["1", "11-50", "1000+"])

    def test_bin_count_features_output_type(self):
        """Verify binned features are string type."""
        from src.processing.data_processor import bin_count_features

        test_data = pl.DataFrame({"device_ip_count": [0, 100, 5000]})
        result = test_data.with_columns(bin_count_features(["device_ip"]))

        self.assertEqual(result["device_ip_count_bin"].dtype, pl.String)


class TestHourlyImpressionsBinning(unittest.TestCase):
    """Tests for hourly impressions binning."""

    def test_bin_hourly_impressions_basic(self):
        """Test basic hourly impressions binning."""
        from src.processing.data_processor import bin_hourly_impressions

        test_data = pl.DataFrame({"user_hourly_impressions": [1, 2, 3, 4, 5]})
        result = test_data.with_columns(bin_hourly_impressions())

        expected = ["single", "2", "3-4", "3-4", "5+"]
        self.assertEqual(result["user_hourly_impressions_bin"].to_list(), expected)

    def test_bin_hourly_impressions_boundaries(self):
        """Test binning at exact boundaries."""
        from src.processing.data_processor import bin_hourly_impressions

        test_data = pl.DataFrame({"user_hourly_impressions": [1, 2, 3, 4, 5, 10]})
        result = test_data.with_columns(bin_hourly_impressions())

        bins = result["user_hourly_impressions_bin"].to_list()
        self.assertEqual(bins, ["single", "2", "3-4", "3-4", "5+", "5+"])

    def test_bin_hourly_impressions_output_type(self):
        """Verify binned feature is string type."""
        from src.processing.data_processor import bin_hourly_impressions

        test_data = pl.DataFrame({"user_hourly_impressions": [1, 10, 100]})
        result = test_data.with_columns(bin_hourly_impressions())

        self.assertEqual(result["user_hourly_impressions_bin"].dtype, pl.String)


class TestTimeDeltaBinning(unittest.TestCase):
    """Tests for time delta binning."""

    def test_bin_time_delta_basic(self):
        """Test basic time delta binning."""
        from src.processing.data_processor import bin_time_delta_features

        test_data = pl.DataFrame({"hours_since_last_click": [0, 3, 15, 50, 100]})
        result = test_data.with_columns(bin_time_delta_features())

        bins = result["hours_since_last_click_bin"].to_list()
        self.assertEqual(bins, ["first", "1-4h", "5-19h", "20-53h", ">53h"])

    def test_bin_time_delta_boundaries(self):
        """Test binning at exact boundaries."""
        from src.processing.data_processor import bin_time_delta_features

        test_data = pl.DataFrame({"hours_since_last_click": [0, 4, 5, 19, 20, 53, 54]})
        result = test_data.with_columns(bin_time_delta_features())

        bins = result["hours_since_last_click_bin"].to_list()
        self.assertEqual(
            bins, ["first", "1-4h", "5-19h", "5-19h", "20-53h", "20-53h", ">53h"]
        )


class TestPreviousClicksBinning(unittest.TestCase):
    """Tests for previous clicks binning."""

    def test_bin_prev_clicks_basic(self):
        """Test basic previous clicks binning."""
        from src.processing.data_processor import bin_prev_clicks

        test_data = pl.DataFrame({"user_proxy_prev_clicks": [0, 3, 15, 75, 250]})
        result = test_data.with_columns(bin_prev_clicks("user_proxy"))

        bins = result["user_proxy_prev_clicks_bin"].to_list()
        self.assertEqual(bins, ["new", "returning", "regular", "heavy", "power"])

    def test_bin_prev_clicks_boundaries(self):
        """Test binning at exact boundaries."""
        from src.processing.data_processor import bin_prev_clicks

        test_data = pl.DataFrame(
            {"user_proxy_prev_clicks": [0, 7, 8, 32, 33, 224, 225]}
        )
        result = test_data.with_columns(bin_prev_clicks("user_proxy"))

        bins = result["user_proxy_prev_clicks_bin"].to_list()
        self.assertEqual(
            bins, ["new", "returning", "regular", "regular", "heavy", "heavy", "power"]
        )

    def test_bin_prev_clicks_output_type(self):
        """Verify binned feature is string type."""
        from src.processing.data_processor import bin_prev_clicks

        test_data = pl.DataFrame({"user_proxy_prev_clicks": [0, 50, 200]})
        result = test_data.with_columns(bin_prev_clicks("user_proxy"))

        self.assertEqual(result["user_proxy_prev_clicks_bin"].dtype, pl.String)


class TestPrevClicksExpression(unittest.TestCase):
    """Tests for previous clicks expression."""

    def test_prev_clicks_basic(self):
        """Test basic previous click count computation."""
        from src.processing.data_processor import get_prev_clicks_expression

        test_data = pl.DataFrame({"user_proxy": ["u1", "u1", "u1", "u1"]})
        result = test_data.with_columns(get_prev_clicks_expression("user_proxy"))

        self.assertEqual(result["user_proxy_prev_clicks"].to_list(), [0, 1, 2, 3])

    def test_prev_clicks_multiple_users(self):
        """Test previous click count with multiple users."""
        from src.processing.data_processor import get_prev_clicks_expression

        test_data = pl.DataFrame({"user_proxy": ["u1", "u2", "u1", "u2", "u1"]})
        result = test_data.with_columns(get_prev_clicks_expression("user_proxy"))

        self.assertEqual(result["user_proxy_prev_clicks"].to_list(), [0, 0, 1, 1, 2])

    def test_prev_clicks_all_unique(self):
        """Test previous click count when all users are unique."""
        from src.processing.data_processor import get_prev_clicks_expression

        test_data = pl.DataFrame({"user_proxy": ["u1", "u2", "u3", "u4"]})
        result = test_data.with_columns(get_prev_clicks_expression("user_proxy"))

        self.assertEqual(result["user_proxy_prev_clicks"].to_list(), [0, 0, 0, 0])

    def test_prev_clicks_dtype(self):
        """Verify previous clicks are UInt32."""
        from src.processing.data_processor import get_prev_clicks_expression

        test_data = pl.DataFrame({"user_proxy": ["u1", "u1"]})
        result = test_data.with_columns(get_prev_clicks_expression("user_proxy"))

        self.assertEqual(result["user_proxy_prev_clicks"].dtype, pl.UInt32)


class TestLazyVocabularyMapping(unittest.TestCase):
    """Tests for lazy vocabulary mapping functions."""

    def test_get_lazy_vocab_map_basic(self):
        """Test lazy vocab map creates correct mappings."""
        from src.processing.data_processor import get_lazy_vocab_map

        test_data = pl.DataFrame({"cat1": ["a", "a", "a", "b", "b", "c"]}).lazy()
        vocab_lf = get_lazy_vocab_map(test_data, "cat1", min_freq=2)
        vocab_df = vocab_lf.collect()

        self.assertEqual(len(vocab_df), 2)  # Only 'a' and 'b' pass min_freq=2
        self.assertIn("cat1", vocab_df.columns)
        self.assertIn("cat1_id", vocab_df.columns)

    def test_get_lazy_vocab_map_ids_start_at_one(self):
        """Test vocabulary IDs start at 1 (0 reserved for UNK)."""
        from src.processing.data_processor import get_lazy_vocab_map

        test_data = pl.DataFrame({"cat1": ["a", "a", "b", "b"]}).lazy()
        vocab_lf = get_lazy_vocab_map(test_data, "cat1", min_freq=1)
        vocab_df = vocab_lf.collect()

        vocab_dict = dict(
            zip(vocab_df["cat1"].to_list(), vocab_df["cat1_id"].to_list())
        )
        self.assertEqual(vocab_dict["a"], 1)
        self.assertEqual(vocab_dict["b"], 2)

    def test_apply_lazy_vocab_transforms(self):
        """Test apply_lazy_vocab transforms values to IDs."""
        from src.processing.data_processor import get_lazy_vocab_map, apply_lazy_vocab

        train_data = pl.DataFrame({"cat1": ["a", "a", "a", "b", "b", "c"]}).lazy()
        vocab_lf = get_lazy_vocab_map(train_data, "cat1", min_freq=1)

        test_data = pl.DataFrame({"cat1": ["a", "b", "c", "unknown"]}).lazy()
        result = apply_lazy_vocab(test_data, vocab_lf, "cat1").collect()

        self.assertEqual(result["cat1"].to_list(), [1, 2, 3, 0])

    def test_apply_lazy_vocab_fills_unknown_with_zero(self):
        """Test unknown values get mapped to 0 (UNK)."""
        from src.processing.data_processor import get_lazy_vocab_map, apply_lazy_vocab

        train_data = pl.DataFrame({"cat1": ["a", "a", "b"]}).lazy()
        vocab_lf = get_lazy_vocab_map(train_data, "cat1", min_freq=1)

        test_data = pl.DataFrame({"cat1": ["x", "y", "z"]}).lazy()
        result = apply_lazy_vocab(test_data, vocab_lf, "cat1").collect()

        self.assertEqual(result["cat1"].to_list(), [0, 0, 0])

    def test_apply_lazy_vocab_preserves_other_columns(self):
        """Test apply_lazy_vocab preserves other columns."""
        from src.processing.data_processor import get_lazy_vocab_map, apply_lazy_vocab

        train_data = pl.DataFrame({"cat1": ["a", "a", "b"], "other": [1, 2, 3]}).lazy()
        vocab_lf = get_lazy_vocab_map(train_data, "cat1", min_freq=1)

        test_data = pl.DataFrame({"cat1": ["a", "b"], "other": [10, 20]}).lazy()
        result = apply_lazy_vocab(test_data, vocab_lf, "cat1").collect()

        self.assertIn("other", result.columns)
        self.assertEqual(result["other"].to_list(), [10, 20])
        self.assertEqual(result["cat1"].dtype, pl.Int32)

    def test_get_lazy_vocab_map_empty_after_filter(self):
        """Test vocab map handles case where all values below min_freq."""
        from src.processing.data_processor import get_lazy_vocab_map

        test_data = pl.DataFrame({"cat1": ["a", "b", "c"]}).lazy()  # All freq=1
        vocab_lf = get_lazy_vocab_map(test_data, "cat1", min_freq=5)
        vocab_df = vocab_lf.collect()

        self.assertEqual(len(vocab_df), 0)


class TestCollectStatsPass(unittest.TestCase):
    """Tests for statistics collection pipeline stage."""

    def test_collect_stats_pass_returns_correct_structure(self):
        """Test collect_stats_pass returns expected tuple structure."""
        from src.processing.data_processor import collect_stats_pass

        test_data = pl.DataFrame(
            {
                "user_proxy": ["u1", "u1", "u2"],
                "hour": ["14102100", "14102100", "14102101"],
                "device_ip": ["ip1", "ip1", "ip2"],
            }
        ).lazy()

        hourly_lf, count_lfs, vocab_lfs = collect_stats_pass(
            test_data, ["device_ip"], [], 1
        )

        self.assertIsInstance(hourly_lf, pl.LazyFrame)
        self.assertIsInstance(count_lfs, dict)
        self.assertIn("device_ip", count_lfs)

    def test_collect_stats_pass_hourly_aggregation(self):
        """Test hourly aggregation produces correct counts."""
        from src.processing.data_processor import collect_stats_pass

        test_data = pl.DataFrame(
            {
                "user_proxy": ["u1", "u1", "u1", "u2"],
                "hour": ["14102100", "14102100", "14102101", "14102100"],
                "device_ip": ["ip1"] * 4,
            }
        ).lazy()

        hourly_lf, _, _ = collect_stats_pass(test_data, [], [], 1)
        hourly_df = hourly_lf.collect()

        # u1@14102100 = 2, u1@14102101 = 1, u2@14102100 = 1
        self.assertEqual(len(hourly_df), 3)

    def test_collect_stats_pass_count_features(self):
        """Test count features are computed correctly."""
        from src.processing.data_processor import collect_stats_pass

        test_data = pl.DataFrame(
            {
                "user_proxy": ["u1"] * 5,
                "hour": ["14102100"] * 5,
                "device_ip": ["ip1", "ip1", "ip1", "ip2", "ip2"],  # ip1=3, ip2=2
            }
        ).lazy()

        _, count_lfs, _ = collect_stats_pass(test_data, ["device_ip"], [], 1)
        count_df = count_lfs["device_ip"].collect()

        count_dict = dict(
            zip(count_df["device_ip"].to_list(), count_df["device_ip_count"].to_list())
        )
        self.assertEqual(count_dict["ip1"], 3)
        self.assertEqual(count_dict["ip2"], 2)


class TestStringCastExpressions(unittest.TestCase):
    """Tests for string cast expressions."""

    def test_get_string_cast_expressions(self):
        """Test string cast expressions work correctly."""
        from src.processing.data_processor import get_string_cast_expressions

        test_data = pl.DataFrame({"col1": [1, 2, 3], "col2": [4.0, 5.0, 6.0]})
        result = test_data.with_columns(get_string_cast_expressions(["col1", "col2"]))

        self.assertEqual(result["col1"].dtype, pl.String)
        self.assertEqual(result["col2"].dtype, pl.String)
        self.assertEqual(result["col1"].to_list(), ["1", "2", "3"])


class TestTimeDeltaExpressions(unittest.TestCase):
    """Tests for time delta expressions."""

    def test_time_delta_expressions_creates_timestamp(self):
        """Test time delta expressions create _timestamp column."""
        from src.processing.data_processor import get_time_delta_expressions

        test_data = pl.DataFrame({"hour": ["14102100", "14102101"]})
        result = test_data.with_columns(get_time_delta_expressions())

        self.assertIn("_timestamp", result.columns)
        self.assertEqual(result["_timestamp"].dtype, pl.Datetime("us"))

    def test_time_delta_window_expressions(self):
        """Test time delta window expressions compute deltas."""
        from src.processing.data_processor import (
            get_time_delta_expressions,
            get_time_delta_window_expressions,
            get_user_proxy_expression,
        )

        test_data = pl.DataFrame(
            {
                "device_ip": ["ip1", "ip1", "ip1"],
                "device_model": ["m1", "m1", "m1"],
                "hour": ["14102100", "14102101", "14102105"],
            }
        )

        result = (
            test_data.lazy()
            .with_columns(get_user_proxy_expression())
            .with_columns(get_time_delta_expressions())
            .with_columns(get_time_delta_window_expressions())
            .collect()
        )

        deltas = result["hours_since_last_click"].to_list()
        self.assertEqual(deltas[0], 0)  # First click
        self.assertEqual(deltas[1], 1)  # 1 hour later
        self.assertEqual(deltas[2], 4)  # 4 hours later


if __name__ == "__main__":
    unittest.main(verbosity=2)

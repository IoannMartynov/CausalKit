"""
Tests for the traffic_splitter module.
"""

import unittest
import pandas as pd
import numpy as np
from causalkit.design.traffic_splitter import split_traffic


class TestTrafficSplitter(unittest.TestCase):
    """Test cases for the traffic_splitter module."""

    def setUp(self):
        """Set up test data."""
        # Create a sample DataFrame
        np.random.seed(42)
        self.df = pd.DataFrame({
            'user_id': range(100),
            'group': ['A', 'B'] * 50,
            'value': np.random.randn(100)
        })

    def test_simple_split(self):
        """Test a simple 50/50 split."""
        train_df, test_df = split_traffic(self.df, random_state=42)

        # Check that the split sizes are correct
        self.assertEqual(len(train_df), 50)
        self.assertEqual(len(test_df), 50)

        # Check that all rows are preserved
        self.assertEqual(len(train_df) + len(test_df), len(self.df))

        # Check that there's no overlap between the splits
        train_ids = set(train_df['user_id'])
        test_ids = set(test_df['user_id'])
        self.assertEqual(len(train_ids.intersection(test_ids)), 0)

    def test_custom_ratio_split(self):
        """Test a custom ratio split (80/20)."""
        train_df, test_df = split_traffic(self.df, split_ratio=0.8, random_state=42)

        # Check that the split sizes are correct
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)

        # Check that all rows are preserved
        self.assertEqual(len(train_df) + len(test_df), len(self.df))

    def test_multiple_splits(self):
        """Test multiple splits (70/20/10)."""
        train_df, val_df, test_df = split_traffic(
            self.df, split_ratio=[0.7, 0.2], random_state=42
        )

        # Check that the split sizes are correct
        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 20)
        self.assertEqual(len(test_df), 10)

        # Check that all rows are preserved
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.df))

    def test_stratified_split(self):
        """Test stratified split by group."""
        train_df, test_df = split_traffic(
            self.df, split_ratio=0.8, stratify_column='group', random_state=42
        )

        # Check that the split sizes are correct
        self.assertEqual(len(train_df), 80)
        self.assertEqual(len(test_df), 20)

        # Check that the stratification is preserved
        train_group_counts = train_df['group'].value_counts(normalize=True)
        test_group_counts = test_df['group'].value_counts(normalize=True)

        # The proportions should be approximately equal
        self.assertAlmostEqual(train_group_counts['A'], 0.5, places=1)
        self.assertAlmostEqual(train_group_counts['B'], 0.5, places=1)
        self.assertAlmostEqual(test_group_counts['A'], 0.5, places=1)
        self.assertAlmostEqual(test_group_counts['B'], 0.5, places=1)

    def test_invalid_ratio(self):
        """Test that an invalid ratio raises an error."""
        with self.assertRaises(ValueError):
            split_traffic(self.df, split_ratio=[0.7, 0.5], random_state=42)


if __name__ == '__main__':
    unittest.main()

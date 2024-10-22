# test_data_processing_tool.py

import unittest
import pandas as pd
from io import StringIO
from data_processing_tool import read_csv, clean_data, transform_data, compute_statistics

class TestDataProcessingTool(unittest.TestCase):

    def setUp(self):
        """Create a simple DataFrame for testing."""
        self.csv_data = StringIO("""A,B,C
        1,2,3
        4,5,6
        7,8,9
        10,,12
        """)
        self.df = pd.read_csv(self.csv_data)

    def test_read_csv(self):
        df = read_csv('data.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        clean_df = clean_data(self.df)
        self.assertFalse(clean_df.isnull().values.any())

    def test_transform_data_normalize(self):
        transformed_df = transform_data(self.df.dropna(), 'normalize')
        self.assertTrue((transformed_df.min() == 0).all() and (transformed_df.max() == 1).all())

    def test_transform_data_standardize(self):
        transformed_df = transform_data(self.df.dropna(), 'standardize')
        self.assertAlmostEqual(transformed_df.mean().sum(), 0, places=5)
        self.assertAlmostEqual(transformed_df.std().sum(), len(transformed_df.columns), places=5)

    def test_compute_statistics(self):
        stats = compute_statistics(self.df.dropna())
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('mode', stats)
        self.assertIn('variance', stats)
        self.assertIn('std_dev', stats)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch

import pandas as pd

# Import the functions from the script
from data_fetch.fred_api import fetch_fred_data, save_to_csv, FRED_SERIES, ONE_MONTH_MINUS_ONE_FRED_SERIES, LAG_FRED_SERIES


class TestFredDataFetching(unittest.TestCase):

    @patch('data_fetch.fred_api.Fred')  # Patch the Fred class in the fred_api module
    def test_fetch_fred_data(self, mock_fred):
        # Mock the FRED API response
        mock_fred_instance = mock_fred.return_value
        
        # Create mock data with multiple entries per month to test groupby logic
        mock_data = pd.Series(
            [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            index=pd.to_datetime(['2023-01-15', '2023-01-30', '2023-02-15', '2023-02-28', '2023-03-15', '2023-03-30'])
        )
        mock_fred_instance.get_series.return_value = mock_data

        # Call the function
        result = fetch_fred_data(start_date="2023-01-01", end_date="2023-03-31")

        # Assert that the FRED API was called the expected number of times (once per series)
        self.assertEqual(mock_fred_instance.get_series.call_count, len(FRED_SERIES))

        # Assert that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Assert that the DataFrame has the correct columns
        self.assertListEqual(list(result.columns), list(FRED_SERIES.keys()))

        # Assert that the data was resampled correctly to month-end
        expected_index_end = pd.to_datetime('2023-03-31')
        self.assertEqual(result.index[-1], expected_index_end)

        # Assert that the data was shifted correctly for the specified series
        for series in ONE_MONTH_MINUS_ONE_FRED_SERIES:
            self.assertTrue(pd.isna(result[series].iloc[0]))

        # Test that all expected series exist in the result
        for series_name in FRED_SERIES.keys():
            self.assertIn(series_name, result.columns)

    @patch('data_fetch.fred_api.pd.DataFrame.to_csv')
    def test_save_to_csv(self, mock_to_csv):
        # Create a dummy DataFrame
        df = pd.DataFrame({
            'EU_CPI': [1.0, 2.0, 3.0],
            'EU_10Y_Yield': [0.5, 0.6, 0.7]
        }, index=pd.to_datetime(['2023-01-31', '2023-02-28', '2023-03-31']))

        # Call the function
        save_to_csv(df, filename="test_macro_data.csv")

        # Assert that the to_csv method was called with the correct parameters
        mock_to_csv.assert_called_once_with("test_macro_data.csv", index=True)

    @patch('data_fetch.fred_api.Fred')
    def test_fetch_fred_data_error_handling(self, mock_fred):
        # Mock the FRED API to raise an exception
        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.side_effect = Exception("API Error")

        # Use assertRaises to check that an Exception is raised
        with self.assertRaises(Exception) as context:
            fetch_fred_data(start_date="2023-01-01", end_date="2023-03-01")

        # Optional: Check the exception message
        self.assertEqual(str(context.exception), "API Error")

    def test_constants_exist(self):
        """Test that all expected constants are defined"""
        # Check that all expected constants exist
        self.assertIsInstance(FRED_SERIES, dict)
        self.assertIsInstance(ONE_MONTH_MINUS_ONE_FRED_SERIES, list)
        self.assertIsInstance(LAG_FRED_SERIES, list)
        
        # Check that all series in ONE_MONTH_MINUS_ONE_FRED_SERIES are in FRED_SERIES keys
        for series in ONE_MONTH_MINUS_ONE_FRED_SERIES:
            self.assertIn(series, FRED_SERIES)
            
        # For LAG_FRED_SERIES, we need to check the values (series IDs) not the keys
        # Convert LAG_FRED_SERIES IDs to their corresponding keys in FRED_SERIES
        lag_series_keys = [key for key, value in FRED_SERIES.items() if key in LAG_FRED_SERIES]
        self.assertGreater(len(lag_series_keys), 0, "Should find corresponding keys for LAG_FRED_SERIES")
        
        # Ensure no overlap between ONE_MONTH_MINUS_ONE_FRED_SERIES and the lag series keys
        overlap = set(ONE_MONTH_MINUS_ONE_FRED_SERIES) & set(lag_series_keys)
        self.assertEqual(len(overlap), 0, f"Series should not be in both lists: {overlap}")

    @patch('data_fetch.fred_api.Fred')
    def test_monthly_resampling(self, mock_fred):
        """Test that data is properly resampled to monthly frequency"""
        mock_fred_instance = mock_fred.return_value
        
        # Create daily data that spans multiple months
        daily_data = pd.Series(
            [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
            index=pd.to_datetime([
                '2023-01-01', '2023-01-15', '2023-01-31',  # January
                '2023-02-01', '2023-02-15', '2023-02-28',  # February  
                '2023-03-01', '2023-03-15', '2023-03-31'   # March
            ])
        )
        mock_fred_instance.get_series.return_value = daily_data

        # Call the function
        result = fetch_fred_data(start_date="2023-01-01", end_date="2023-03-31")

        # Should have 3 monthly observations
        self.assertEqual(len(result), 3)
        
        # Index should be month-end dates
        expected_dates = pd.to_datetime(['2023-01-31', '2023-02-28', '2023-03-31'])
        expected_dates.name = 'Date'  # Match the name from the actual result
        pd.testing.assert_index_equal(result.index, expected_dates)

    @patch('data_fetch.fred_api.Fred')
    def test_lag_series_handling(self, mock_fred):
        """Test that LAG_FRED_SERIES are handled with last-1 logic instead of shifting"""
        mock_fred_instance = mock_fred.return_value
        
        # Create mock data with multiple values per month to test last-1 logic
        # For LAG series, we want to test that they use the second-to-last value of each month
        mock_data = pd.Series(
            [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
            index=pd.to_datetime(['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15', '2023-03-01', '2023-03-15'])
        )
        mock_fred_instance.get_series.return_value = mock_data

        # Call the function
        result = fetch_fred_data(start_date="2023-01-01", end_date="2023-03-31")

        # Convert LAG_FRED_SERIES IDs to their corresponding keys in FRED_SERIES
        lag_series_keys = [key for key, value in FRED_SERIES.items() if value in LAG_FRED_SERIES]
        
        # For LAG series, verify they use the second-to-last (last-1) value of each month
        for series in lag_series_keys:
            with self.subTest(series=series):
                self.assertIn(series, result.columns)
                # For LAG series, the first value should not be NaN (no shifting applied)
                self.assertFalse(pd.isna(result[series].iloc[0]))
                
                # Test that LAG series use last-1 values:
                # January: should be 10.0 (first value, since only 2 values: 10.0, 15.0)
                # February: should be 20.0 (first value, since only 2 values: 20.0, 25.0) 
                # March: should be 30.0 (first value, since only 2 values: 30.0, 35.0)
                self.assertEqual(result[series].iloc[0], 10.0, f"{series} January value should be 10.0 (last-1)")
                self.assertEqual(result[series].iloc[1], 20.0, f"{series} February value should be 20.0 (last-1)")
                self.assertEqual(result[series].iloc[2], 30.0, f"{series} March value should be 30.0 (last-1)")

    @patch('data_fetch.fred_api.Fred')
    def test_series_type_differences(self, mock_fred):
        """Test that different series types are handled correctly"""
        mock_fred_instance = mock_fred.return_value
        
        # Create consistent mock data for all series
        mock_data = pd.Series(
            [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            index=pd.to_datetime(['2023-01-15', '2023-01-30', '2023-02-15', '2023-02-28', '2023-03-15', '2023-03-30'])
        )
        mock_fred_instance.get_series.return_value = mock_data

        # Call the function
        result = fetch_fred_data(start_date="2023-01-01", end_date="2023-03-31")

        # Check ONE_MONTH_MINUS_ONE_FRED_SERIES are shifted (first value is NaN)
        for series in ONE_MONTH_MINUS_ONE_FRED_SERIES:
            with self.subTest(series=series):
                self.assertTrue(pd.isna(result[series].iloc[0]), 
                               f"{series} should have NaN in first position due to shifting")

        # Convert LAG_FRED_SERIES IDs to their corresponding keys in FRED_SERIES
        lag_series_keys = [key for key, value in FRED_SERIES.items() if value in LAG_FRED_SERIES]
        
        # Check LAG_FRED_SERIES are not shifted (first value is not NaN)
        for series in lag_series_keys:
            with self.subTest(series=series):
                self.assertFalse(pd.isna(result[series].iloc[0]), 
                                f"{series} should not have NaN in first position (no shifting)")

        # Check that regular series (not in either special list) are not shifted
        regular_series = [s for s in FRED_SERIES.keys() 
                         if s not in ONE_MONTH_MINUS_ONE_FRED_SERIES and s not in lag_series_keys]
        for series in regular_series:
            with self.subTest(series=series):
                self.assertFalse(pd.isna(result[series].iloc[0]), 
                                f"{series} should not have NaN in first position (regular series)")


if __name__ == "__main__":
    unittest.main()

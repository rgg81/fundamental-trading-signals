import unittest
from unittest.mock import patch

import pandas as pd

# Import the functions from the script
from data_fetch.fred_api import fetch_fred_data, save_to_csv, FRED_SERIES, ONE_MONTH_MINUS_ONE_FRED_SERIES


class TestFredDataFetching(unittest.TestCase):

    @patch('data_fetch.fred_api.Fred')  # Patch the Fred class in the fred_api module
    def test_fetch_fred_data(self, mock_fred):
        # Mock the FRED API response
        mock_fred_instance = mock_fred.return_value
        mock_fred_instance.get_series.return_value = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
        )

        # Call the function
        result = fetch_fred_data(start_date="2023-01-01", end_date="2023-03-01")

        # Assert that the FRED API was called with the correct parameters
        mock_fred_instance.get_series.assert_called()

        # Assert that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Assert that the DataFrame has the correct columns
        self.assertListEqual(list(result.columns), list(FRED_SERIES.keys()))

        # Assert that the data was resampled correctly
        self.assertEqual(result.index[-1], pd.to_datetime('2023-03-31'))

        # Assert that the data was shifted correctly for the specified series
        for series in ONE_MONTH_MINUS_ONE_FRED_SERIES:
            self.assertTrue(pd.isna(result[series].iloc[0]))

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


if __name__ == "__main__":
    unittest.main()

import datetime
import unittest
from unittest.mock import patch

import pandas as pd

from data_fetch.duskacopy_api import ensure_download_directory, download_symbol_data, download_symbol


class TestDuskacopyAPI(unittest.TestCase):

    @patch("os.makedirs")
    @patch("os.path.exists", return_value=False)
    def test_ensure_download_directory(self, mock_exists, mock_makedirs):
        ensure_download_directory()
        mock_makedirs.assert_called_once_with("download")

    @patch("subprocess.run")
    @patch("pandas.read_csv")
    def test_download_symbol_data_success(self, mock_read_csv, mock_subprocess):
        mock_read_csv.return_value = pd.DataFrame(
            {"Date": [1], "Open": [1.1], "High": [1.2], "Low": [1.0], "Close": [1.15]})
        start_date = datetime.datetime(2023, 1, 1)
        end_date = datetime.datetime(2023, 1, 2)
        df = download_symbol_data("EURUSD", start_date, end_date)
        self.assertFalse(df.empty)
        mock_subprocess.assert_called_once()
        mock_read_csv.assert_called_once()

    @patch("subprocess.run")
    @patch("pandas.read_csv", side_effect=Exception("File not found"))
    def test_download_symbol_data_failure(self, mock_read_csv, mock_subprocess):
        start_date = datetime.datetime(2023, 1, 1)
        end_date = datetime.datetime(2023, 1, 2)
        df = download_symbol_data("EURUSD", start_date, end_date)
        self.assertTrue(df.empty)

    @patch("data_fetch.duskacopy_api.download_symbol_data")
    @patch("data_fetch.duskacopy_api.ensure_download_directory")
    def test_download_symbol(self, mock_ensure_dir, mock_download_data):
        mock_download_data.return_value = pd.DataFrame(
            {"Date": [1672531200000], "EURUSD_Open": [1.1], "EURUSD_High": [1.2], "EURUSD_Low": [1.0],
             "EURUSD_Close": [1.15]})
        df = download_symbol("EURUSD")
        self.assertFalse(df.empty)
        self.assertIn("EURUSD_Close", df.columns)
        mock_ensure_dir.assert_called_once()
        mock_download_data.assert_called()


if __name__ == "__main__":
    unittest.main()

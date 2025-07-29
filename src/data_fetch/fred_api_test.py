import os

import pandas as pd
from fredapi import Fred

# Load API Key from environment variable or config file
FRED_API_KEY = os.getenv("FRED_API_KEY")
if FRED_API_KEY is None:
    FRED_API_KEY = "NO_API_KEY"

# List of FRED series IDs for macroeconomic indicators
FRED_SERIES = {
    "EU_CPI": "CP0000EZ19M086NEST",  # Euro Area Harmonized CPI (-1 month)
    "EU_10Y_Yield": "IRLTLT01EZM156N",  # Euro Area 10-Year Bond Yield (-1 month)
    "ECB_Deposit_Rate": "ECBDFR",  # ECB Deposit Rate
    "US_CPI": "CPIAUCSL",  # US Consumer Price Index (-1 month)
    "US_Core_CPI": "CPILFESL",  # US Core CPI (-1 month)
    "Fed_Funds_Rate": "FEDFUNDS",  # US Effective Federal Funds Rate (-1 month)
    "US_10Y_Yield": "DGS10",  # US 10-Year Treasury Yield
    "VIX": "VIXCLS",  # Volatility Index
}

ONE_MONTH_MINUS_ONE_FRED_SERIES = ["EU_CPI", "EU_10Y_Yield", "US_CPI", "US_Core_CPI", "Fed_Funds_Rate"]


def fetch_fred_data(start_date="2000-01-01", end_date=None):
    fred = Fred(api_key=FRED_API_KEY)
    """Fetches macroeconomic data from FRED and resamples it to the last day of each month."""
    data = {}

    for name, series_id in FRED_SERIES.items():
        try:
            df = fred.get_series(series_id, start_date, end_date)
            df = df.to_frame(name)
            df.index = pd.to_datetime(df.index)

            print(df.iloc[-1])
            data[name] = df
        except Exception as e:
            print(f"Error fetching {name} ({series_id}): {e}")
            raise





if __name__ == "__main__":
    fetch_fred_data()
    

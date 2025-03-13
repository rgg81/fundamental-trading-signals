import os
import pandas as pd
from fredapi import Fred
import requests
from pyjstat import pyjstat


# Load API Key from environment variable or config file
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# List of FRED series IDs for macroeconomic indicators
FRED_SERIES = {
    "US_CPI": "CPIAUCSL",  # US Consumer Price Index
    "US_Core_CPI": "CPILFESL",  # US Core CPI
    "Fed_Funds_Rate": "FEDFUNDS",  # US Effective Federal Funds Rate
    "ECB_Deposit_Rate": "ECBDFR",  # ECB Deposit Rate
    "US_10Y_Yield": "DGS10",  # US 10-Year Treasury Yield
    "VIX": "VIXCLS",  # Volatility Index
}


def fetch_fred_data(start_date="2000-01-01", end_date=None):
    """Fetches macroeconomic data from FRED and resamples it to the last day of each month."""
    data = {}

    for name, series_id in FRED_SERIES.items():
        try:
            df = fred.get_series(series_id, start_date, end_date)
            df = df.to_frame(name)
            df.index = pd.to_datetime(df.index)

            # Resample to get the last available value of each month
            df = df.resample("ME").last()
            data[name] = df
        except Exception as e:
            print(f"Error fetching {name} ({series_id}): {e}")

    # Merge all series into a single DataFrame
    macro_data = pd.concat(data.values(), axis=1)
    macro_data.index.name = "Date"
    return macro_data


def save_to_csv(df, filename="macro_data.csv"):
    """Saves the DataFrame to a CSV file without duplicated headers."""
    df.to_csv(filename, index=True)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    df_fred = fetch_fred_data()
    print(df_fred.tail())
    save_to_csv(df_fred)

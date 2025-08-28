import os

import pandas as pd
from fredapi import Fred
from sympy import series

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
LAG_FRED_SERIES = ["US_10Y_Yield", "VIX"]


def fetch_fred_data(start_date="2000-01-01", end_date=None):
    fred = Fred(api_key=FRED_API_KEY)
    """Fetches macroeconomic data from FRED and resamples it to the last day of each month."""
    data = {}

    for name, series_id in FRED_SERIES.items():
        try:
            df = fred.get_series(series_id, start_date, end_date)
            df = df.to_frame(name)
            df.index = pd.to_datetime(df.index)

            # Resample to get the last available value of each month
            df = df.resample("ME")
            #df = df.last()
            if name in LAG_FRED_SERIES:
                def second_last_or_last(series):
                    non_nan_values = series.dropna()
                    return non_nan_values.iloc[-2]
                df = df.agg(second_last_or_last)
            else:
                df = df.last()
            data[name] = df
        except Exception as e:
            print(f"Error fetching {name} ({series_id}): {e}")
            raise


    # Merge all series into a single DataFrame
    macro_data = pd.concat(data.values(), axis=1)
    macro_data.index.name = "Date"
    # Shift data by two months to account for data delay
    macro_data[ONE_MONTH_MINUS_ONE_FRED_SERIES] = macro_data[ONE_MONTH_MINUS_ONE_FRED_SERIES].shift(1)
    return macro_data


def save_to_csv(df, filename="macro_data.csv"):
    """Saves the DataFrame to a CSV file without duplicated headers."""
    df.to_csv(filename, index=True)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    df_fred = fetch_fred_data()
    print(df_fred.tail())
    save_to_csv(df_fred)

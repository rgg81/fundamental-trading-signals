import requests
import pandas as pd
from pyjstat import pyjstat
import json

# Eurostat API URL for HICP data (Consumer Price Index for Europe)
EUROSTAT_HICP_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_midx?format=JSON&geo=EU"


def fetch_eurostat_hicp():
    """Fetches the Harmonized Index of Consumer Prices (HICP) from Eurostat."""
    try:
        response = requests.get(EUROSTAT_HICP_URL)
        response.raise_for_status()  # Ensure we catch HTTP errors
        data_json = json.dumps(response.json())  # Convert JSON to string
        dataset = pyjstat.Dataset.read(data_json)  # Read JSON string
        df = dataset.write('dataframe')
        df.index = pd.to_datetime(df['Time'])
        df = df[['value']].rename(columns={'value': 'HICP'})
        df = df.resample("ME").last()
        df.index.name = "Date"
        return df
    except Exception as e:
        print(f"Error fetching HICP data from Eurostat: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    df_hicp = fetch_eurostat_hicp()
    print(df_hicp.tail())
    df_hicp.to_csv("eu_cpi.csv", index=True)

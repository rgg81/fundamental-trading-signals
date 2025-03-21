from duskacopy_api import get_all_symbols_data
from fred_api import fetch_fred_data, save_to_csv


def merge_dataframes(symbol_data, macro_data):
    """
    Merges symbol data and macroeconomic data on the Date index.
    """
    final_df = macro_data.copy()
    for df in symbol_data:
        if df is not None and not df.empty:
            final_df = final_df.join(df, how="outer")
    return final_df


def main():
    print("Downloading financial market data from Dukascopy...")
    symbol_data = get_all_symbols_data()

    print("Fetching macroeconomic data from FRED...")
    macro_data = fetch_fred_data()

    print("Merging all data...")
    final_df = merge_dataframes(symbol_data, macro_data)

    print("Saving final dataset...")
    save_to_csv(final_df, "final_dataset.csv")
    print("Final dataset saved as 'final_dataset.csv'")


if __name__ == "__main__":
    main()

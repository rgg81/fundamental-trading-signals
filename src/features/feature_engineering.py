import pandas as pd

# Configuration
LAG_FEATURES = ["EU_CPI", "US_CPI", "US_Core_CPI", "EU_10Y_Yield", "US_10Y_Yield", "Fed_Funds_Rate", "ECB_Deposit_Rate",
                "VIX"]
RATE_OF_CHANGE_FEATURES = LAG_FEATURES
MOVING_AVERAGE_FEATURES = LAG_FEATURES
VOLATILITY_FEATURES = ["VIX", "US_10Y_Yield", "EU_10Y_Yield"]

# Moving average windows
MOVING_AVERAGE_WINDOWS = [3, 6, 12]
VOLATILITY_WINDOWS = [3, 6, 12]


def add_lag_features(df, features, lags=3):
    """Generate lag features for the given columns."""
    for feature in features:
        for lag in range(1, lags + 1):
            df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
    return df


def add_rate_of_change_features(df, features, periods=3):
    """Generate rate of change features."""
    for feature in features:
        for period in range(1, periods + 1):
            df[f"{feature}_roc_{period}"] = df[feature].pct_change(periods=period)
    return df


def add_moving_average_features(df, features, windows):
    """Generate moving average features."""
    for feature in features:
        for window in windows:
            df[f"{feature}_ma_{window}"] = df[feature].rolling(window=window).mean()
    return df


def add_volatility_features(df, features, windows):
    """Generate rolling volatility features."""
    for feature in features:
        for window in windows:
            df[f"{feature}_vol_{window}"] = df[feature].rolling(window=window).std()
    return df


def generate_features(df):
    """Pipeline to generate all features."""
    df = add_lag_features(df, LAG_FEATURES)
    df = add_rate_of_change_features(df, RATE_OF_CHANGE_FEATURES)
    df = add_moving_average_features(df, MOVING_AVERAGE_FEATURES, MOVING_AVERAGE_WINDOWS)
    df = add_volatility_features(df, VOLATILITY_FEATURES, VOLATILITY_WINDOWS)

    # Drop rows with NaNs introduced by feature generation
    # Analyze NaN values before removal
    print("\n=== NaN Analysis Before Removal ===")
    print(df[df.isna().any(axis=1)])
    
    #df = df.dropna()
    print(f"\nAfter dropping NaNs: {len(df)} rows remaining")
    if len(df) > 0:
        print(f"New date range: {df.index.min()} to {df.index.max()}")
        print(f"New first row (index 0): {df.index[0]}")
        print(f"New last row (index {len(df)-1}): {df.index[-1]}")

    return df


if __name__ == "__main__":
    # Example: Load macroeconomic data
    df = pd.read_csv("macro_data.csv", parse_dates=["Date"], index_col="Date")

    # Generate features
    df_features = generate_features(df)

    # Save to CSV
    df_features.to_csv("macro_features.csv")
    print("Feature engineering complete. Output saved to macro_features.csv")

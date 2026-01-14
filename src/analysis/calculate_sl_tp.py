import pandas as pd
import numpy as np

def calculate_optimal_sl_tp(symbol: str = "GBPUSD", file_path: str = None):
    """
    Calculate optimal stop loss and take profit percentages based on historical price movements.
    Analyzes both long and short trade scenarios.
    
    Parameters:
    -----------
    symbol : str
        Currency pair symbol (e.g., 'GBPUSD', 'EURUSD', 'EURCHF')
    file_path : str, optional
        Path to the CSV file containing OHLC data. If None, uses '{symbol}.csv'
        
    Returns:
    --------
    dict
        Dictionary containing statistics for long and short trades
    """
    # Set default file path if not provided
    if file_path is None:
        file_path = f"{symbol}.csv"
    
    # Construct column name based on symbol
    close_col = f"{symbol}_Close"
    
    # Load the data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Verify the close column exists
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in {file_path}. Available columns: {df.columns.tolist()}")
    
    print(f"Loaded {len(df)} rows for {symbol}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Calculate price changes
    df['Next_Close'] = df[close_col].shift(-1)
    df['Price_Change'] = df['Next_Close'] - df[close_col]
    df['Price_Change_Pct'] = (df['Price_Change'] / df[close_col]) * 100
    
    # Remove the last row (no next close available)
    df = df[:-1].copy()
    
    # Separate long and short trades
    long_trades = df[df['Price_Change'] > 0].copy()
    short_trades = df[df['Price_Change'] < 0].copy()
    
    print("\n" + "="*80)
    print("=== LONG TRADES ANALYSIS ===")
    print("="*80)
    print(f"Total periods with price increase: {len(long_trades)}")
    print(f"Percentage of total periods: {len(long_trades)/len(df)*100:.2f}%")
    
    if len(long_trades) > 0:
        avg_increase_pct = long_trades['Price_Change_Pct'].mean()
        median_increase_pct = long_trades['Price_Change_Pct'].median()
        std_increase_pct = long_trades['Price_Change_Pct'].std()
        min_increase_pct = long_trades['Price_Change_Pct'].min()
        max_increase_pct = long_trades['Price_Change_Pct'].max()
        
        print(f"\nPrice increase statistics:")
        print(f"  Average increase: {avg_increase_pct:.4f}%")
        print(f"  Median increase: {median_increase_pct:.4f}%")
        print(f"  Std deviation: {std_increase_pct:.4f}%")
        print(f"  Min increase: {min_increase_pct:.4f}%")
        print(f"  Max increase: {max_increase_pct:.4f}%")
        
        # Calculate percentiles
        percentiles = [25, 50, 75, 90, 95]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = np.percentile(long_trades['Price_Change_Pct'], p)
            print(f"  {p}th percentile: {value:.4f}%")
        
        print(f"\n>>> RECOMMENDED TAKE PROFIT for LONG trades: {avg_increase_pct:.4f}%")
    
    print("\n" + "="*80)
    print("=== SHORT TRADES ANALYSIS ===")
    print("="*80)
    print(f"Total periods with price decrease: {len(short_trades)}")
    print(f"Percentage of total periods: {len(short_trades)/len(df)*100:.2f}%")
    
    if len(short_trades) > 0:
        # For short trades, we want the absolute value of the decrease
        avg_decrease_pct = abs(short_trades['Price_Change_Pct'].mean())
        median_decrease_pct = abs(short_trades['Price_Change_Pct'].median())
        std_decrease_pct = short_trades['Price_Change_Pct'].std()
        min_decrease_pct = abs(short_trades['Price_Change_Pct'].max())  # Max because it's negative
        max_decrease_pct = abs(short_trades['Price_Change_Pct'].min())  # Min because it's negative
        
        print(f"\nPrice decrease statistics:")
        print(f"  Average decrease: {avg_decrease_pct:.4f}%")
        print(f"  Median decrease: {median_decrease_pct:.4f}%")
        print(f"  Std deviation: {std_decrease_pct:.4f}%")
        print(f"  Min decrease: {min_decrease_pct:.4f}%")
        print(f"  Max decrease: {max_decrease_pct:.4f}%")
        
        # Calculate percentiles (using absolute values)
        percentiles = [25, 50, 75, 90, 95]
        print(f"\nPercentiles:")
        for p in percentiles:
            value = abs(np.percentile(short_trades['Price_Change_Pct'], 100-p))
            print(f"  {p}th percentile: {value:.4f}%")
        
        print(f"\n>>> RECOMMENDED TAKE PROFIT for SHORT trades: {avg_decrease_pct:.4f}%")
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("=== OVERALL STATISTICS ===")
    print("="*80)
    
    if len(long_trades) > 0 and len(short_trades) > 0:
        # Average of both directions
        avg_movement = (avg_increase_pct + avg_decrease_pct) / 2
        
        print(f"\nAverage absolute price movement: {avg_movement:.4f}%")
        print(f"\n{'='*80}")
        print(f">>> RECOMMENDED STOP LOSS / TAKE PROFIT (same for both): {avg_movement:.4f}%")
        print(f"{'='*80}")
        
        # Calculate win rate if using this SL/TP
        print(f"\nIf using SL/TP of {avg_movement:.4f}%:")
        print(f"  Long trade win rate: {len(long_trades[long_trades['Price_Change_Pct'] >= avg_movement])/len(df)*100:.2f}%")
        print(f"  Short trade win rate: {len(short_trades[short_trades['Price_Change_Pct'] <= -avg_movement])/len(df)*100:.2f}%")
    
    # Show distribution of price changes
    print("\n" + "="*80)
    print("=== PRICE CHANGE DISTRIBUTION ===")
    print("="*80)
    
    ranges = [
        (0, 0.5, "0-0.5%"),
        (0.5, 1.0, "0.5-1.0%"),
        (1.0, 1.5, "1.0-1.5%"),
        (1.5, 2.0, "1.5-2.0%"),
        (2.0, 3.0, "2.0-3.0%"),
        (3.0, 5.0, "3.0-5.0%"),
        (5.0, 100.0, ">5.0%")
    ]
    
    print("\nLong trades (price increases):")
    for min_val, max_val, label in ranges:
        count = len(long_trades[(long_trades['Price_Change_Pct'] >= min_val) & 
                                 (long_trades['Price_Change_Pct'] < max_val)])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count} trades ({pct:.2f}% of all periods)")
    
    print("\nShort trades (price decreases):")
    for min_val, max_val, label in ranges:
        count = len(short_trades[(abs(short_trades['Price_Change_Pct']) >= min_val) & 
                                  (abs(short_trades['Price_Change_Pct']) < max_val)])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {label}: {count} trades ({pct:.2f}% of all periods)")
    
    # Return results as dictionary
    results = {
        'long_trades': {
            'count': len(long_trades),
            'avg_increase_pct': avg_increase_pct if len(long_trades) > 0 else 0,
            'median_increase_pct': median_increase_pct if len(long_trades) > 0 else 0,
            'std_increase_pct': std_increase_pct if len(long_trades) > 0 else 0
        },
        'short_trades': {
            'count': len(short_trades),
            'avg_decrease_pct': avg_decrease_pct if len(short_trades) > 0 else 0,
            'median_decrease_pct': median_decrease_pct if len(short_trades) > 0 else 0,
            'std_decrease_pct': std_decrease_pct if len(short_trades) > 0 else 0
        },
        'recommended_sl_tp_pct': avg_movement if len(long_trades) > 0 and len(short_trades) > 0 else 0
    }
    
    return results


def main():
    """Main function to run the analysis."""
    
    import sys
    
    # Get symbol from command line argument or use default
    symbol = 'AUDUSD'
    
    print(f"Analyzing {symbol}...")
    print("="*80)
    
    # Calculate for the specified symbol
    results = calculate_optimal_sl_tp(symbol)
    
    print("\n" + "="*80)
    print(f"Analysis complete for {symbol}!")
    print("="*80)


if __name__ == "__main__":
    main()

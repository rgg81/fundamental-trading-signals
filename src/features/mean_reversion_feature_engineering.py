import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

class MeanReversionFeatureEngineering:
    """
    Feature engineering class that creates mean reversion indicators from FX price data.
    Supports multiple currency pairs (EURUSD, USDJPY, etc.).
    Focuses on sideways market behaviors, support/resistance levels, and reversal signals.
    Optimized for monthly data analysis in ranging/consolidating markets.
    """
    
    def __init__(self, lookback_periods: List[int] = [3, 6, 12], currency_pairs: List[str] = ['EURUSD']):
        """
        Initialize the mean reversion feature engineering class.
        
        Parameters:
        -----------
        lookback_periods : List[int]
            Different lookback periods for indicators (default: [3, 6, 12] months)
        currency_pairs : List[str]
            List of currency pairs to process (default: ['EURUSD'])
            Supported pairs: 'EURUSD', 'USDJPY', etc.
        """
        self.lookback_periods = lookback_periods
        self.currency_pairs = currency_pairs
        
    def load_eurusd_data(self, file_path: str = "EURUSD.csv", currency_pair: str = "EURUSD") -> pd.DataFrame:
        """
        Load FX data from CSV file and convert to monthly data.
        
        Parameters:
        -----------
        file_path : str
            Path to the FX CSV file
        currency_pair : str
            Currency pair symbol (e.g., 'EURUSD', 'USDJPY')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with monthly OHLC data
        """
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Construct column names based on currency pair
            open_col = f'{currency_pair}_Open'
            high_col = f'{currency_pair}_High'
            low_col = f'{currency_pair}_Low'
            close_col = f'{currency_pair}_Close'
            
            # Check if columns exist
            if open_col not in df.columns:
                raise ValueError(f"Column {open_col} not found in {file_path}. Available columns: {df.columns.tolist()}")
            
            # Convert to monthly data (OHLC)
            monthly_data = pd.DataFrame()
            monthly_data['Open'] = df[open_col].resample('ME').first()
            monthly_data['High'] = df[high_col].resample('ME').max()
            monthly_data['Low'] = df[low_col].resample('ME').min()
            monthly_data['Close'] = df[close_col].resample('ME').last()
            
            return monthly_data.dropna()
            
        except Exception as e:
            raise ValueError(f"Error loading {currency_pair} data from {file_path}: {e}")
    
    def generate_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion indicators optimized for sideways markets.
        Keep only the most stationary and predictive features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with mean reversion indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating mean reversion indicators...")
        
        for period in self.lookback_periods:
            # 1. Price Z-Score - Most stationary mean reversion indicator
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            features[f'mr_Price_ZScore_{period}'] = (df['Close'] - sma) / std
            
            # 2. RSI - Bounded oscillator (0-100), highly stationary
            rsi = ta.momentum.rsi(df['Close'], window=period)
            features[f'mr_RSI_{period}'] = rsi
        
        return features
    
    def generate_support_resistance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate support and resistance level indicators.
        Keep only normalized distance measures (stationary).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with support/resistance indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating support/resistance indicators...")
        
        for period in self.lookback_periods:
            # Normalized distance to range extremes - stationary and predictive
            rolling_high = df['High'].rolling(period).max()
            rolling_low = df['Low'].rolling(period).min()
            range_size = rolling_high - rolling_low
            
            # Position within range (0 = at low, 1 = at high) - bounded and stationary
            features[f'support_Range_Position_{period}'] = (df['Close'] - rolling_low) / range_size
        
        return features
    
    def generate_volatility_contraction_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility contraction indicators (low volatility often precedes reversals).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility contraction indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating volatility indicators...")
        
        for period in self.lookback_periods:
            # Normalized ATR - stationary volatility measure
            atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
            features[f'volatility_ATR_Ratio_{period}'] = atr / df['Close']
        
        return features
    
    def generate_momentum_exhaustion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum exhaustion indicators (weakening trends often reverse).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum exhaustion indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating momentum indicators...")
        
        for period in self.lookback_periods:
            # ROC - percentage change (stationary)
            roc = ta.momentum.roc(df['Close'], window=period)
            features[f'momentum_ROC_{period}'] = roc
        
        return features
    
    def generate_custom_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate custom mean reversion indicators specifically for sideways markets.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with custom mean reversion indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating custom indicators...")
        
        for period in self.lookback_periods:
            # Normalized return (stationary)
            features[f'custom_Return_{period}'] = df['Close'].pct_change(period)
        
        return features
    
    def calculate_correlation_matrix(self, features: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Calculate correlation matrix and identify highly correlated features.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with all features
        threshold : float
            Correlation threshold for flagging highly correlated features
            
        Returns:
        --------
        pd.DataFrame
            Correlation matrix
        """
        corr_matrix = features.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs (>{threshold}):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print(f"\nNo highly correlated features found (threshold: {threshold})")
        
        return corr_matrix
    
    def remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Remove highly correlated features while prioritizing shorter timeframes.
        Keep one representative feature from each highly correlated group.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with all features
        threshold : float
            Correlation threshold for removing features (default: 0.7)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with uncorrelated features
        """
        print(f"\n=== Removing Correlated Features (threshold: {threshold}) ===")
        
        # Create correlation matrix
        corr_matrix = features.corr().abs()
        
        # Create a priority map for features (shorter periods get higher priority)
        def get_feature_priority(feature_name):
            """Assign priority based on timeframe and feature type."""
            # Extract period from feature name
            if '_3' in feature_name:
                period_priority = 40  # Highest priority for 3-month
            elif '_6' in feature_name:
                period_priority = 30  # Highest priority for 6-month
            elif '_12' in feature_name:
                period_priority = 20  # Medium priority for 12-month  
            elif '_18' in feature_name:
                period_priority = 15  # Medium-low priority for 18-month
            elif '_24' in feature_name:
                period_priority = 10  # Lowest priority for 24-month
            else:
                period_priority = 15  # Default medium priority
            
            # Feature type priority (based on prefix)
            feature_type_priority = 0
            if feature_name.startswith('mr_'):
                feature_type_priority = 15  # Highest - mean reversion indicators
            elif feature_name.startswith('support_'):
                feature_type_priority = 10  # Important - position in range
            elif feature_name.startswith('volatility_'):
                feature_type_priority = 8   # Volatility measure
            elif feature_name.startswith('momentum_'):
                feature_type_priority = 6   # Momentum
            elif feature_name.startswith('custom_'):
                feature_type_priority = 4   # Custom features
            
            return period_priority + feature_type_priority
        
        # Get feature priorities
        feature_priorities = {col: get_feature_priority(col) for col in features.columns}
        
        # Group features into correlation clusters
        processed_features = set()
        features_to_remove = set()
        correlation_groups = []
        
        # First, create a graph of all correlations above threshold
        correlation_graph = {}
        for feat in features.columns:
            correlation_graph[feat] = []
        
        for i, feat1 in enumerate(features.columns):
            for j, feat2 in enumerate(features.columns):
                if i != j:
                    correlation = corr_matrix.loc[feat1, feat2]
                    if correlation > threshold:
                        correlation_graph[feat1].append(feat2)
        
        # Find connected components (correlation clusters)
        for feat in features.columns:
            if feat in processed_features:
                continue
                
            # Find all features in this connected component using DFS
            stack = [feat]
            cluster = []
            while stack:
                current = stack.pop()
                if current not in processed_features:
                    cluster.append(current)
                    processed_features.add(current)
                    # Add all correlated features to the stack
                    for neighbor in correlation_graph[current]:
                        if neighbor not in processed_features:
                            stack.append(neighbor)
            
            if len(cluster) > 1:
                # Sort by priority (highest first) and keep only the best one
                cluster.sort(key=lambda x: feature_priorities[x], reverse=True)
                keep_feature = cluster[0]
                remove_features = cluster[1:]
                
                # Calculate max correlation within the cluster
                max_corr = 0
                for f1 in cluster:
                    for f2 in cluster:
                        if f1 != f2:
                            max_corr = max(max_corr, corr_matrix.loc[f1, f2])
                
                correlation_groups.append({
                    'group': cluster,
                    'kept': keep_feature,
                    'removed': remove_features,
                    'max_correlation': max_corr
                })
                
                features_to_remove.update(remove_features)
        
        # Print correlation analysis
        print(f"Found {len(correlation_groups)} correlation groups:")
        for i, group in enumerate(correlation_groups[:15]):  # Show first 15 groups
            print(f"  Group {i+1}: Kept {group['kept']} (max corr: {group['max_correlation']:.3f})")
            print(f"    Removed: {', '.join(group['removed'][:3])}" + 
                  (f" + {len(group['removed'])-3} more" if len(group['removed']) > 3 else ""))
        
        if len(correlation_groups) > 15:
            print(f"  ... and {len(correlation_groups) - 15} more groups")
        
        # Remove correlated features
        selected_features = features.drop(columns=list(features_to_remove))
        
        print(f"\nFeature selection summary:")
        print(f"  Original features: {len(features.columns)}")
        print(f"  Correlation groups: {len(correlation_groups)}")
        print(f"  Removed features: {len(features_to_remove)}")
        print(f"  Final features: {len(selected_features.columns)}")
        
        # Show distribution of kept features by timeframe
        timeframe_count = {'3': 0, '6': 0, '12': 0, '18': 0, '24': 0, 'other': 0}
        for col in selected_features.columns:
            if '_3' in col:
                timeframe_count['3'] += 1
            elif '_6' in col:
                timeframe_count['6'] += 1
            elif '_12' in col:
                timeframe_count['12'] += 1
            elif '_18' in col:
                timeframe_count['18'] += 1
            elif '_24' in col:
                timeframe_count['24'] += 1
            else:
                timeframe_count['other'] += 1
        
        print(f"\nFinal features by timeframe:")
        print(f"  6-month features: {timeframe_count['6']}")
        print(f"  12-month features: {timeframe_count['12']}")
        print(f"  18-month features: {timeframe_count['18']}")
        print(f"  3-month features: {timeframe_count['3']}")
        print(f"  Other features: {timeframe_count['other']}")
        
        return selected_features
    
    def run_feature_engineering(self, 
                               fx_file_path: str = None,
                               currency_pair: str = None,
                               save_features: bool = True,
                               output_file: str = None,
                               correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        Run the complete mean reversion feature engineering pipeline for a specific currency pair.
        
        Parameters:
        -----------
        fx_file_path : str
            Path to FX CSV file (default: constructs from currency_pair)
        currency_pair : str
            Currency pair to process (e.g., 'EURUSD', 'USDJPY')
            If None, uses first pair from self.currency_pairs
        save_features : bool
            Whether to save features to CSV
        output_file : str
            Output CSV filename (default: auto-generated based on currency_pair)
        correlation_threshold : float
            Threshold for correlation analysis
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with mean reversion features
        """
        # Set default currency pair
        if currency_pair is None:
            currency_pair = self.currency_pairs[0]
        
        # Set default file paths
        if fx_file_path is None:
            fx_file_path = f"{currency_pair}.csv"
        
        if output_file is None:
            output_file = f"mean_reversion_features_{currency_pair}.csv"
        
        print(f"=== Mean Reversion Feature Engineering Pipeline for {currency_pair} ===")
        
        # Step 1: Load and prepare data
        print(f"1. Loading {currency_pair} data from {fx_file_path}...")
        monthly_data = self.load_eurusd_data(fx_file_path, currency_pair)
        print(f"   Loaded {len(monthly_data)} monthly observations")
        print(f"   Date range: {monthly_data.index[0].strftime('%Y-%m')} to {monthly_data.index[-1].strftime('%Y-%m')}")
        
        # Step 2: Generate all mean reversion indicator categories
        print("2. Generating mean reversion indicators...")
        
        # Generate all indicator types focused on mean reversion
        mean_reversion_features = self.generate_mean_reversion_indicators(monthly_data)
        support_resistance_features = self.generate_support_resistance_indicators(monthly_data)
        volatility_contraction_features = self.generate_volatility_contraction_indicators(monthly_data)
        momentum_exhaustion_features = self.generate_momentum_exhaustion_indicators(monthly_data)
        custom_features = self.generate_custom_mean_reversion_indicators(monthly_data)
        
        # Combine all features
        all_features = pd.concat([
            mean_reversion_features,
            support_resistance_features, 
            volatility_contraction_features,
            momentum_exhaustion_features,
            custom_features
        ], axis=1)
        
        print(f"   Generated {all_features.shape[1]} mean reversion indicators")
        print(f"   Features available for {len(all_features)} time periods")
        
        # Step 3: Clean features
        print("3. Cleaning features...")
        # Remove infinite values
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        
        # Remove features with too many NaN values (>50%)
        nan_threshold = len(all_features) * 0.5
        valid_features = all_features.dropna(thresh=nan_threshold, axis=1)
        
        print(f"   Removed {all_features.shape[1] - valid_features.shape[1]} features with >50% NaN values")
        print(f"   Final feature count: {valid_features.shape[1]}")
        
        # Step 4: Remove highly correlated features
        print("4. Removing highly correlated features...")
        selected_features = self.remove_correlated_features(valid_features, threshold=correlation_threshold)
        
        # Step 5: Final correlation analysis on selected features
        print("5. Analyzing final feature correlations...")
        final_correlation_matrix = self.calculate_correlation_matrix(selected_features, correlation_threshold)
        
        # Step 6: Save features
        if save_features:
            print(f"6. Saving features to {output_file}...")
            selected_features.to_csv(output_file)
            
            # Save correlation matrix
            corr_file = output_file.replace('.csv', '_correlations.csv')
            final_correlation_matrix.to_csv(corr_file)
            print(f"   Features saved to {output_file}")
            print(f"   Correlations saved to {corr_file}")
        
        print(f"\n=== Feature Engineering Complete for {currency_pair} ===")
        print(f"Final dataset shape: {selected_features.shape}")
        
        # Print feature categories summary
        categories = {
            'Mean Reversion (mr_)': [col for col in selected_features.columns if col.startswith('mr_')],
            'Support/Resistance (support_)': [col for col in selected_features.columns if col.startswith('support_')],
            'Volatility (volatility_)': [col for col in selected_features.columns if col.startswith('volatility_')],
            'Momentum (momentum_)': [col for col in selected_features.columns if col.startswith('momentum_')],
            'Custom (custom_)': [col for col in selected_features.columns if col.startswith('custom_')]
        }
        
        print("\nFeature categories:")
        for category, features in categories.items():
            print(f"  {category}: {len(features)} features")
        
        return selected_features
    
    def run_feature_engineering_all_pairs(self,
                                         save_features: bool = True,
                                         correlation_threshold: float = 0.7) -> Dict[str, pd.DataFrame]:
        """
        Run feature engineering for all configured currency pairs.
        
        Parameters:
        -----------
        save_features : bool
            Whether to save features to CSV
        correlation_threshold : float
            Threshold for correlation analysis
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping currency pair to its features DataFrame
        """
        print("="*80)
        print("=== Mean Reversion Feature Engineering - All Currency Pairs ===")
        print(f"Processing {len(self.currency_pairs)} currency pair(s): {', '.join(self.currency_pairs)}")
        print("="*80)
        
        all_pair_features = {}
        
        for pair in self.currency_pairs:
            print(f"\n{'='*80}")
            try:
                features = self.run_feature_engineering(
                    fx_file_path=f"{pair}.csv",
                    currency_pair=pair,
                    save_features=save_features,
                    output_file=f"mean_reversion_features_{pair}.csv",
                    correlation_threshold=correlation_threshold
                )
                all_pair_features[pair] = features
                print(f"✅ Successfully processed {pair}")
            except Exception as e:
                print(f"❌ Error processing {pair}: {e}")
                continue
        
        print(f"\n{'='*80}")
        print(f"=== Processing Complete ===")
        print(f"Successfully processed {len(all_pair_features)}/{len(self.currency_pairs)} currency pairs")
        print("="*80)
        
        return all_pair_features


def main():
    """Main function to run the mean reversion feature engineering pipeline."""
    
    # Initialize feature engineering class with different lookback periods and currency pairs
    feature_engineer = MeanReversionFeatureEngineering(
        lookback_periods=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # 6, 12, and 24 months
        currency_pairs=["EURUSD", "USDJPY", "EURJPY", "AUDUSD", "XAUUSD", "GBPUSD"]  # Multiple currency pairs
    )
    
    # Run the complete pipeline for all currency pairs
    try:
        # Process all currency pairs
        all_features = feature_engineer.run_feature_engineering_all_pairs(
            save_features=True,
            correlation_threshold=1.0  # Threshold for final analysis
        )
        
        # Display summary for each currency pair
        for pair, features in all_features.items():
            print(f"\n{'='*80}")
            print(f"=== Summary for {pair} ===")
            print(f"{'='*80}")
            
            print("\n=== Summary Statistics ===")
            print(features.describe())
            
            print(f"\n=== Sample Features (Last 5 rows) ===")
            print(features.tail())
            
            print(f"\n=== Feature List ({len(features.columns)} features) ===")
            for i, col in enumerate(features.columns, 1):
                print(f"{i:2d}. {col}")
            
            # Final validation - check correlation matrix one more time
            print(f"\n=== Final Correlation Validation ===")
            final_corr = features.corr().abs()
            max_corr = 0
            max_pair_corr = None
            
            for i in range(len(final_corr.columns)):
                for j in range(i+1, len(final_corr.columns)):
                    corr_val = final_corr.iloc[i, j]
                    if corr_val > max_corr:
                        max_corr = corr_val
                        max_pair_corr = (final_corr.columns[i], final_corr.columns[j])
            
            print(f"Maximum correlation between final features: {max_corr:.3f}")
            if max_pair_corr:
                print(f"  Between: {max_pair_corr[0]} and {max_pair_corr[1]}")
            
            if max_corr < 0.5:
                print(f"✅ All {pair} features have correlation < 0.5")
            else:
                print(f"⚠️ Some {pair} features still have high correlation")
        
        print(f"\n{'='*80}")
        print(f"=== All Currency Pairs Processed Successfully ===")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error in mean reversion pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
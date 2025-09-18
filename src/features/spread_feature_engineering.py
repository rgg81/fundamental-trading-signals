import pandas as pd
import numpy as np
from typing import List
import warnings

warnings.filterwarnings('ignore')

class SpreadFeatureEngineering:
    """
    Feature engineering class that creates EU-US spread indicators from macroeconomic data.
    Focuses on the differences between European and US economic indicators.
    Optimized for EUR/USD trading with correlation-based feature selection.
    """
    
    def __init__(self, windows: List[int] = [3, 6]):
        """
        Initialize the spread feature engineering class.
        
        Parameters:
        -----------
        windows : List[int]
            Different windows for moving averages and volatility (default: [3, 6] months)
        """
        self.windows = windows
        
        # Define spread pairs (EU indicator - US indicator)
        self.spread_pairs = {
            'CPI_Spread': ('EU_CPI', 'US_CPI'),
            'Core_CPI_Spread': ('EU_CPI', 'US_Core_CPI'),  # EU doesn't have core CPI, use regular
            'Yield_Spread': ('EU_10Y_Yield', 'US_10Y_Yield'),
            'Rate_Spread': ('ECB_Deposit_Rate', 'Fed_Funds_Rate')
        }
        
        # VIX is included as standalone (no EU equivalent)
        self.standalone_features = ['VIX']
        
    def load_macro_data(self, file_path: str = "macro_data.csv") -> pd.DataFrame:
        """
        Load macroeconomic data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the macro data CSV file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with macroeconomic data
        """
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            return df.dropna()
            
        except Exception as e:
            raise ValueError(f"Error loading macro data: {e}")
    
    def generate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate spread features between EU and US indicators.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with macro data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with spread features
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating spread features...")
        
        # 1. Basic spreads
        for spread_name, (eu_col, us_col) in self.spread_pairs.items():
            if eu_col in df.columns and us_col in df.columns:
                features[spread_name] = df[eu_col] / df[us_col]
                print(f"   Created {spread_name}: {eu_col} / {us_col}")
            else:
                print(f"   Warning: Missing columns for {spread_name}: {eu_col}, {us_col}")
        
        # 2. Add VIX as standalone feature
        if 'VIX' in df.columns:
            features['VIX'] = df['VIX']
            print("   Added VIX as standalone feature")
        
        return features
    
    def add_lag_features(self, df: pd.DataFrame, features: List[str], lags: int = 3) -> pd.DataFrame:
        """
        Generate lag features for the given columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create lags for
        lags : int
            Number of lag periods
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with lag features added
        """
        print(f"Generating lag features (up to {lags} periods)...")
        
        for feature in features:
            if feature in df.columns:
                for lag in range(1, lags + 1):
                    df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
        
        return df
    
    def add_rate_of_change_features(self, df: pd.DataFrame, features: List[str], periods: int = 3) -> pd.DataFrame:
        """
        Generate rate of change features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create rate of change for
        periods : int
            Number of periods for rate of change calculation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with rate of change features added
        """
        print(f"Generating rate of change features (up to {periods} periods)...")
        
        for feature in features:
            if feature in df.columns:
                for period in range(1, periods + 1):
                    df[f"{feature}_roc_{period}"] = df[feature].pct_change(periods=period)
        
        return df
    
    def add_moving_average_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate moving average features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create moving averages for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with moving average features added
        """
        print(f"Generating moving average features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    df[f"{feature}_ma_{window}"] = df[feature].rolling(window=window).mean()
                    
                    # Add moving average spread (current value - MA)
                    df[f"{feature}_ma_spread_{window}"] = df[feature] - df[f"{feature}_ma_{window}"]
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate rolling volatility features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create volatility for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility features added
        """
        print(f"Generating volatility features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    df[f"{feature}_vol_{window}"] = df[feature].rolling(window=window).std()
                    
                    # Add volatility-adjusted spread (current spread / volatility)
                    vol_col = f"{feature}_vol_{window}"
                    df[f"{feature}_vol_adj_{window}"] = df[feature] / (df[vol_col] + 1e-8)  # Avoid division by zero
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate momentum and trend features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create momentum features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum features added
        """
        print("Generating momentum and trend features...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # Trend direction (positive/negative momentum)
                    df[f"{feature}_trend_{window}"] = (df[feature] > df[feature].shift(window)).astype(int)
                    
                    # Acceleration (second derivative)
                    df[f"{feature}_accel_{window}"] = df[feature].diff().diff(periods=window)
                    
                    # Z-score normalization
                    rolling_mean = df[feature].rolling(window=window).mean()
                    rolling_std = df[feature].rolling(window=window).std()
                    df[f"{feature}_zscore_{window}"] = (df[feature] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
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
        Remove highly correlated features while prioritizing shorter timeframes and core spreads.
        
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
        
        # Create a priority map for features
        def get_feature_priority(feature_name):
            """Assign priority based on feature type and timeframe."""
            priority = 0
            
            # Core spread features get highest priority
            if any(x in feature_name for x in ['CPI_Spread', 'Yield_Spread', 'Rate_Spread']):
                priority += 50
            
            # VIX gets high priority (unique risk indicator)
            if 'VIX' in feature_name:
                priority += 45
            
            # Shorter timeframes get higher priority
            if '_3' in feature_name:
                priority += 20  # 3-month window
            elif '_6' in feature_name:
                priority += 15  # 6-month window
            
            # Feature type priorities
            if any(x in feature_name for x in ['_zscore_', '_vol_adj_']):
                priority += 10  # Normalized features
            elif any(x in feature_name for x in ['_ma_spread_', '_trend_']):
                priority += 8   # Trend and MA spread features
            elif any(x in feature_name for x in ['_roc_', '_accel_']):
                priority += 6   # Momentum features
            elif any(x in feature_name for x in ['_vol_', '_ma_']):
                priority += 4   # Volatility and MA features
            elif any(x in feature_name for x in ['_lag_']):
                priority += 2   # Lag features
            
            return priority
        
        # Get feature priorities
        feature_priorities = {col: get_feature_priority(col) for col in features.columns}
        
        # Group features into correlation clusters using DFS
        processed_features = set()
        features_to_remove = set()
        correlation_groups = []
        
        # Create correlation graph
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
        for i, group in enumerate(correlation_groups[:10]):  # Show first 10 groups
            print(f"  Group {i+1}: Kept {group['kept']} (max corr: {group['max_correlation']:.3f})")
            print(f"    Removed: {', '.join(group['removed'][:3])}" + 
                  (f" + {len(group['removed'])-3} more" if len(group['removed']) > 3 else ""))
        
        if len(correlation_groups) > 10:
            print(f"  ... and {len(correlation_groups) - 10} more groups")
        
        # Remove correlated features
        selected_features = features.drop(columns=list(features_to_remove))
        
        print(f"\nFeature selection summary:")
        print(f"  Original features: {len(features.columns)}")
        print(f"  Correlation groups: {len(correlation_groups)}")
        print(f"  Removed features: {len(features_to_remove)}")
        print(f"  Final features: {len(selected_features.columns)}")
        
        # Show distribution of kept features by type
        feature_types = {
            'Core Spreads': [col for col in selected_features.columns if any(x in col for x in ['CPI_Spread', 'Yield_Spread', 'Rate_Spread']) and '_' not in col.split('_')[-1]],
            'VIX Features': [col for col in selected_features.columns if 'VIX' in col],
            'Momentum': [col for col in selected_features.columns if any(x in col for x in ['_roc_', '_accel_', '_trend_'])],
            'Volatility': [col for col in selected_features.columns if any(x in col for x in ['_vol_', '_vol_adj_'])],
            'Moving Average': [col for col in selected_features.columns if any(x in col for x in ['_ma_', '_ma_spread_'])],
            'Normalized': [col for col in selected_features.columns if '_zscore_' in col],
            'Lag Features': [col for col in selected_features.columns if '_lag_' in col]
        }
        
        print(f"\nFinal features by type:")
        for feature_type, feature_list in feature_types.items():
            print(f"  {feature_type}: {len(feature_list)} features")
        
        return selected_features
    
    def run_feature_engineering(self, 
                               macro_file_path: str = "macro_data.csv",
                               save_features: bool = True,
                               output_file: str = "spread_features.csv",
                               correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        Run the complete spread feature engineering pipeline.
        
        Parameters:
        -----------
        macro_file_path : str
            Path to macro data CSV file
        save_features : bool
            Whether to save features to CSV
        output_file : str
            Output CSV filename
        correlation_threshold : float
            Threshold for correlation analysis
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with spread features
        """
        print("=== Spread Feature Engineering Pipeline ===")
        
        # Step 1: Load and prepare data
        print("1. Loading macroeconomic data...")
        macro_data = self.load_macro_data(macro_file_path)
        print(f"   Loaded {len(macro_data)} monthly observations")
        print(f"   Date range: {macro_data.index[0].strftime('%Y-%m')} to {macro_data.index[-1].strftime('%Y-%m')}")
        print(f"   Available columns: {list(macro_data.columns)}")
        
        # Step 2: Generate basic spread features
        print("2. Generating spread features...")
        spread_features = self.generate_spread_features(macro_data)
        print(f"   Generated {len(spread_features.columns)} basic spread features")
        
        # Step 3: Generate all feature types
        print("3. Generating advanced features...")
        all_base_features = list(spread_features.columns)
        
        # Add lag features
        spread_features = self.add_lag_features(spread_features, all_base_features, lags=3)
        
        # Add rate of change features
        spread_features = self.add_rate_of_change_features(spread_features, all_base_features, periods=3)
        
        # Add moving average features
        spread_features = self.add_moving_average_features(spread_features, all_base_features)
        
        # Add volatility features
        spread_features = self.add_volatility_features(spread_features, all_base_features)
        
        # Add momentum features
        spread_features = self.add_momentum_features(spread_features, all_base_features)
        
        print(f"   Generated {spread_features.shape[1]} total features")
        print(f"   Features available for {len(spread_features)} time periods")
        
        # Step 4: Clean features
        print("4. Cleaning features...")
        # Remove infinite values
        spread_features = spread_features.replace([np.inf, -np.inf], np.nan)
        
        # Remove features with too many NaN values (>50%)
        nan_threshold = len(spread_features) * 0.5
        valid_features = spread_features.dropna(thresh=nan_threshold, axis=1)
        
        print(f"   Removed {spread_features.shape[1] - valid_features.shape[1]} features with >50% NaN values")
        print(f"   Final feature count before correlation removal: {valid_features.shape[1]}")
        
        # Step 5: Remove highly correlated features
        print("5. Removing highly correlated features...")
        selected_features = self.remove_correlated_features(valid_features, threshold=correlation_threshold)
        
        # Step 6: Final correlation analysis on selected features
        print("6. Analyzing final feature correlations...")
        final_correlation_matrix = self.calculate_correlation_matrix(selected_features, correlation_threshold)
        
        # Step 7: Save features
        if save_features:
            print(f"7. Saving features to {output_file}...")
            selected_features.to_csv(output_file)
            
            # Save correlation matrix
            corr_file = output_file.replace('.csv', '_correlations.csv')
            final_correlation_matrix.to_csv(corr_file)
            print(f"   Features saved to {output_file}")
            print(f"   Correlations saved to {corr_file}")
        
        print("\n=== Feature Engineering Complete ===")
        print(f"Final dataset shape: {selected_features.shape}")
        
        return selected_features


def main():
    """Main function to run the spread feature engineering pipeline."""
    
    # Initialize feature engineering class with 3 and 6 month windows
    feature_engineer = SpreadFeatureEngineering(
        windows=[3, 6]
    )
    
    # Run the complete pipeline
    try:
        features = feature_engineer.run_feature_engineering(
            macro_file_path="macro_data.csv",
            save_features=True,
            output_file="spread_features.csv",
            correlation_threshold=0.5  # Threshold for correlation analysis
        )
        
        print("\n=== Summary Statistics ===")
        print(features.describe())
        
        print(f"\n=== Sample Features (Last 5 rows) ===")
        print(features.tail())
        
        print(f"\n=== Feature List ===")
        for i, col in enumerate(features.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Final validation - check correlation matrix one more time
        print(f"\n=== Final Correlation Validation ===")
        final_corr = features.corr().abs()
        max_corr = 0
        max_pair = None
        
        for i in range(len(final_corr.columns)):
            for j in range(i+1, len(final_corr.columns)):
                corr_val = final_corr.iloc[i, j]
                if corr_val > max_corr:
                    max_corr = corr_val
                    max_pair = (final_corr.columns[i], final_corr.columns[j])
        
        print(f"Maximum correlation between final features: {max_corr:.3f}")
        if max_pair:
            print(f"  Between: {max_pair[0]} and {max_pair[1]}")
        
        if max_corr < 0.7:
            print("✅ All features have correlation < 0.7")
        else:
            print("⚠️ Some features still have high correlation")
        
    except Exception as e:
        print(f"Error in spread feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
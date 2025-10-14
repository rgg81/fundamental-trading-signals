import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

class TechnicalIndicatorsFeatureEngineering:
    """
    Feature engineering class that creates technical indicators from EURUSD close prices.
    Focuses on uncorrelated indicators across different categories: trend, momentum, volatility.
    Optimized for monthly data analysis without volume indicators.
    """
    
    def __init__(self, lookback_periods: List[int] = [6, 12, 24]):
        """
        Initialize the technical indicators feature engineering class.
        
        Parameters:
        -----------
        lookback_periods : List[int]
            Different lookback periods for indicators (default: [6, 12, 24] months)
        """
        self.lookback_periods = lookback_periods
        
    def load_eurusd_data(self, file_path: str = "EURUSD.csv") -> pd.DataFrame:
        """
        Load EURUSD data from CSV file and convert to monthly data.
        
        Parameters:
        -----------
        file_path : str
            Path to the EURUSD CSV file
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with monthly OHLC data
        """
        try:
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Convert to monthly data (OHLC)
            monthly_data = pd.DataFrame()
            monthly_data['Open'] = df['EURUSD_Open'].resample('ME').first()
            monthly_data['High'] = df['EURUSD_High'].resample('ME').max()
            monthly_data['Low'] = df['EURUSD_Low'].resample('ME').min()
            monthly_data['Close'] = df['EURUSD_Close'].resample('ME').last()
            
            return monthly_data.dropna()
            
        except Exception as e:
            raise ValueError(f"Error loading EURUSD data: {e}")
    
    def generate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend-following indicators optimized for monthly data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trend indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating trend indicators...")
        
        for period in self.lookback_periods:
            # 1. Average Directional Index (ADX) - Trend strength
            features[f'ADX_{period}'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=period)
            
            # 2. Aroon Oscillator - Trend direction and strength
            features[f'Aroon_Osc_{period}'] = ta.trend.aroon_up(df['High'], df['Low'], window=period) - ta.trend.aroon_down(df['High'], df['Low'], window=period)
            
            # 3. Commodity Channel Index (CCI) - Trend strength and reversals
            features[f'CCI_{period}'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)
            
            # 4. Vortex Indicator - Trend direction
            features[f'VI_Diff_{period}'] = (ta.trend.vortex_indicator_pos(df['High'], df['Low'], df['Close'], window=period) - 
                                            ta.trend.vortex_indicator_neg(df['High'], df['Low'], df['Close'], window=period))
            
            # 5. MACD Line - Trend momentum
            features[f'MACD_{period}'] = ta.trend.macd(df['Close'], window_slow=period, window_fast=period//2)
        
        return features
    
    def generate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum indicators for monthly data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with momentum indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating momentum indicators...")
        
        for period in self.lookback_periods:
            # 1. Rate of Change (ROC) - Price momentum
            features[f'ROC_{period}'] = ta.momentum.roc(df['Close'], window=period)
            
            # 2. Williams %R - Overbought/oversold momentum
            features[f'Williams_R_{period}'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=period)
            
            # 3. Ultimate Oscillator - Multi-timeframe momentum
            features[f'UO_{period}'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'], 
                                                                        window1=period//3, window2=period//2, window3=period)
            
            # 4. Stochastic Oscillator %K - Momentum position within range
            features[f'Stoch_K_{period}'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], 
                                                              window=period, smooth_window=3)
            
            # 5. RSI - Relative Strength Index
            features[f'RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)
        
        return features
    
    def generate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility indicators for monthly data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with volatility indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating volatility indicators...")
        
        for period in self.lookback_periods:
            # 1. Average True Range (ATR) - Volatility measure
            features[f'ATR_{period}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
            
            # 2. Bollinger Band Width - Volatility squeeze/expansion
            bb_high = ta.volatility.bollinger_hband(df['Close'], window=period)
            bb_low = ta.volatility.bollinger_lband(df['Close'], window=period)
            features[f'BB_Width_{period}'] = (bb_high - bb_low) / df['Close']
            
            # 3. Bollinger Band Position - Price position within bands
            bb_mavg = ta.volatility.bollinger_mavg(df['Close'], window=period)
            features[f'BB_Position_{period}'] = (df['Close'] - bb_low) / (bb_high - bb_low)
            
            # 4. Keltner Channel Width - Alternative volatility measure
            kc_high = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'], window=period)
            kc_low = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'], window=period)
            features[f'KC_Width_{period}'] = (kc_high - kc_low) / df['Close']
            
            # 5. Donchian Channel Width - Range-based volatility
            dc_high = ta.volatility.donchian_channel_hband(df['High'], df['Low'], df['Close'], window=period)
            dc_low = ta.volatility.donchian_channel_lband(df['High'], df['Low'], df['Close'], window=period)
            features[f'DC_Width_{period}'] = (dc_high - dc_low) / df['Close']
        
        return features
    
    def generate_oscillator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate oscillator indicators for mean reversion signals.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with oscillator indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating oscillator indicators...")
        
        for period in self.lookback_periods:
            # 1. Awesome Oscillator - Momentum oscillator
            features[f'AO_{period}'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], 
                                                                      window1=period//2, window2=period)
            
            # 2. KAMA - Adaptive moving average for trend following
            features[f'KAMA_{period}'] = ta.momentum.kama(df['Close'], window=period)
            features[f'KAMA_Signal_{period}'] = (df['Close'] > features[f'KAMA_{period}']).astype(int)
            
            # 3. Percent Price Oscillator (PPO) - MACD variant
            features[f'PPO_{period}'] = ta.momentum.ppo(df['Close'], window_slow=period, window_fast=period//2)
            
            # 4. Stochastic RSI - Combined stochastic and RSI
            features[f'StochRSI_{period}'] = ta.momentum.stochrsi(df['Close'], window=period)
        
        return features
    
    def generate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate custom indicators specifically useful for monthly FX data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with custom indicators
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating custom indicators...")
        
        for period in self.lookback_periods:
            # 1. Monthly Return Volatility
            monthly_returns = df['Close'].pct_change()
            features[f'Return_Vol_{period}'] = monthly_returns.rolling(period).std()
            
            # 2. High-Low Range Percentage
            features[f'HL_Range_Pct_{period}'] = ((df['High'] - df['Low']) / df['Close']).rolling(period).mean()
            
            # 3. Close Position in Range (where close is relative to high-low range)
            features[f'Close_Position_{period}'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).rolling(period).mean()
            
            # 4. Trend Consistency (percentage of up months)
            up_months = (df['Close'] > df['Close'].shift(1)).rolling(period).mean()
            features[f'Trend_Consistency_{period}'] = up_months
            
            # 5. Monthly Gap Analysis (gap between current open and previous close)
            gaps = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            features[f'Avg_Gap_{period}'] = gaps.rolling(period).mean()
            
            # 6. Volatility Adjusted Return
            features[f'Vol_Adj_Return_{period}'] = (monthly_returns.rolling(period).mean() / 
                                                   monthly_returns.rolling(period).std())
            
            # 7. Price Momentum (simple price change)
            features[f'Price_Momentum_{period}'] = (df['Close'] / df['Close'].shift(period) - 1)
            
            # 8. High-Low Momentum
            features[f'HL_Momentum_{period}'] = ((df['High'] + df['Low']) / 2).pct_change(periods=period)
        
        return features
    
    def calculate_correlation_matrix(self, features: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
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
    
    def remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Remove highly correlated features while prioritizing shorter timeframes.
        Keep one representative feature from each highly correlated group.
        
        Parameters:
        -----------
        features : pd.DataFrame
            DataFrame with all features
        threshold : float
            Correlation threshold for removing features (default: 0.5)
            
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
            if '_6' in feature_name:
                period_priority = 30  # Highest priority for 6-month
            elif '_12' in feature_name:
                period_priority = 20  # Medium priority for 12-month  
            elif '_24' in feature_name:
                period_priority = 10  # Lowest priority for 24-month
            else:
                period_priority = 15  # Default medium priority
            
            # Feature type priority (prefer certain indicator types)
            feature_type_priority = 0
            if any(x in feature_name for x in ['ADX', 'RSI', 'MACD', 'ATR']):
                feature_type_priority = 5  # High priority for key indicators
            elif any(x in feature_name for x in ['ROC', 'Williams_R', 'CCI']):
                feature_type_priority = 3  # Medium priority
            elif any(x in feature_name for x in ['BB_Width', 'Return_Vol', 'Stoch']):
                feature_type_priority = 2  # Lower priority
            
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
        timeframe_count = {'6': 0, '12': 0, '24': 0, 'other': 0}
        for col in selected_features.columns:
            if '_6' in col:
                timeframe_count['6'] += 1
            elif '_12' in col:
                timeframe_count['12'] += 1
            elif '_24' in col:
                timeframe_count['24'] += 1
            else:
                timeframe_count['other'] += 1
        
        print(f"\nFinal features by timeframe:")
        print(f"  6-month features: {timeframe_count['6']}")
        print(f"  12-month features: {timeframe_count['12']}")
        print(f"  24-month features: {timeframe_count['24']}")
        print(f"  Other features: {timeframe_count['other']}")
        
        return selected_features
    
    def run_feature_engineering(self, 
                               eurusd_file_path: str = "EURUSD.csv",
                               save_features: bool = True,
                               output_file: str = "technical_indicators_features.csv",
                               correlation_threshold: float = 0.8) -> pd.DataFrame:
        """
        Run the complete technical indicators feature engineering pipeline.
        
        Parameters:
        -----------
        eurusd_file_path : str
            Path to EURUSD CSV file
        save_features : bool
            Whether to save features to CSV
        output_file : str
            Output CSV filename
        correlation_threshold : float
            Threshold for correlation analysis
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with technical indicator features
        """
        print("=== Technical Indicators Feature Engineering Pipeline ===")
        
        # Step 1: Load and prepare data
        print("1. Loading EURUSD data...")
        monthly_data = self.load_eurusd_data(eurusd_file_path)
        print(f"   Loaded {len(monthly_data)} monthly observations")
        print(f"   Date range: {monthly_data.index[0].strftime('%Y-%m')} to {monthly_data.index[-1].strftime('%Y-%m')}")
        
        # Step 2: Generate all indicator categories
        print("2. Generating technical indicators...")
        
        # Generate all indicator types (no volume indicators)
        trend_features = self.generate_trend_indicators(monthly_data)
        momentum_features = self.generate_momentum_indicators(monthly_data)
        volatility_features = self.generate_volatility_indicators(monthly_data)
        oscillator_features = self.generate_oscillator_indicators(monthly_data)
        custom_features = self.generate_custom_indicators(monthly_data)
        
        # Combine all features
        all_features = pd.concat([
          # trend_features,
          # momentum_features, 
          # volatility_features,
            oscillator_features,
         #   custom_features
        ], axis=1)
        
        print(f"   Generated {all_features.shape[1]} technical indicators")
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
        
        print("\n=== Feature Engineering Complete ===")
        print(f"Final dataset shape: {selected_features.shape}")
        
        # Print feature categories summary
        categories = {
            'Trend': [col for col in selected_features.columns if any(x in col for x in ['ADX', 'Aroon', 'CCI', 'VI', 'MACD'])],
            'Momentum': [col for col in selected_features.columns if any(x in col for x in ['ROC', 'Williams', 'UO', 'Stoch', 'RSI'])],
            'Volatility': [col for col in selected_features.columns if any(x in col for x in ['ATR', 'BB', 'KC', 'DC'])],
            'Oscillators': [col for col in selected_features.columns if any(x in col for x in ['AO', 'KAMA', 'PPO', 'StochRSI'])],
            'Custom': [col for col in selected_features.columns if any(x in col for x in ['Return_Vol', 'HL_Range', 'Close_Position', 'Trend_Consistency', 'Gap', 'Vol_Adj', 'Price_Momentum', 'HL_Momentum'])]
        }
        
        print("\nFeature categories:")
        for category, features in categories.items():
            print(f"  {category}: {len(features)} features")
        
        return selected_features


def main():
    """Main function to run the technical indicators feature engineering pipeline."""
    
    # Initialize feature engineering class with different lookback periods
    feature_engineer = TechnicalIndicatorsFeatureEngineering(
        lookback_periods=[18]  # 3, 6, 12, and 24 months
    )
    
    # Run the complete pipeline
    try:
        features = feature_engineer.run_feature_engineering(
            eurusd_file_path="EURUSD.csv",
            save_features=True,
            output_file="technical_indicators_features.csv",
            correlation_threshold=1.0  # Threshold for final analysis
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
        print(f"Error in technical indicators pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

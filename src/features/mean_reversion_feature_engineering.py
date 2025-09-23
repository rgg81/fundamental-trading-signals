import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

class MeanReversionFeatureEngineering:
    """
    Feature engineering class that creates mean reversion indicators from EURUSD close prices.
    Focuses on sideways market behaviors, support/resistance levels, and reversal signals.
    Optimized for monthly data analysis in ranging/consolidating markets.
    """
    
    def __init__(self, lookback_periods: List[int] = [3, 6, 12]):
        """
        Initialize the mean reversion feature engineering class.
        
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
    
    def generate_mean_reversion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion indicators optimized for sideways markets.
        
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
            # 1. Price Distance from Moving Average (Z-Score)
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            features[f'Price_ZScore_{period}'] = (df['Close'] - sma) / std
            
            # 2. Bollinger Band Position (0-1 scale)
            bb_high = ta.volatility.bollinger_hband(df['Close'], window=period)
            bb_low = ta.volatility.bollinger_lband(df['Close'], window=period)
            features[f'BB_Position_{period}'] = (df['Close'] - bb_low) / (bb_high - bb_low)
            
            # 3. RSI Divergence from 50 (mean reversion signal)
            rsi = ta.momentum.rsi(df['Close'], window=period)
            features[f'RSI_Divergence_{period}'] = abs(rsi - 50)
            features[f'RSI_Oversold_{period}'] = (rsi < 30).astype(int)
            features[f'RSI_Overbought_{period}'] = (rsi > 70).astype(int)
            
            # 4. Williams %R Extremes
            williams_r = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=period)
            features[f'Williams_Extreme_{period}'] = ((williams_r < -80) | (williams_r > -20)).astype(int)
            
            # 5. Stochastic Oscillator Extremes
            stoch_k = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=period)
            features[f'Stoch_Oversold_{period}'] = (stoch_k < 20).astype(int)
            features[f'Stoch_Overbought_{period}'] = (stoch_k > 80).astype(int)
        
        return features
    
    def generate_support_resistance_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate support and resistance level indicators.
        
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
            # 1. Distance to Rolling High/Low (support/resistance levels)
            rolling_high = df['High'].rolling(period).max()
            rolling_low = df['Low'].rolling(period).min()
            
            features[f'Distance_To_High_{period}'] = (rolling_high - df['Close']) / df['Close']
            features[f'Distance_To_Low_{period}'] = (df['Close'] - rolling_low) / df['Close']
            
            # 2. Price near Support/Resistance (within 2% of extremes)
            threshold = 0.02  # 2% threshold
            features[f'Near_Resistance_{period}'] = (features[f'Distance_To_High_{period}'] < threshold).astype(int)
            features[f'Near_Support_{period}'] = (features[f'Distance_To_Low_{period}'] < threshold).astype(int)
            
            # 3. Donchian Channel Position
            dc_high = ta.volatility.donchian_channel_hband(df['High'], df['Low'], df['Close'], window=period)
            dc_low = ta.volatility.donchian_channel_lband(df['High'], df['Low'], df['Close'], window=period)
            features[f'Donchian_Position_{period}'] = (df['Close'] - dc_low) / (dc_high - dc_low)
            
            # 4. Price Reversal Signals (price bouncing off extremes)
            prev_close = df['Close'].shift(1)
            features[f'Reversal_From_High_{period}'] = ((prev_close >= rolling_high.shift(1) * 0.98) & 
                                                       (df['Close'] < prev_close)).astype(int)
            features[f'Reversal_From_Low_{period}'] = ((prev_close <= rolling_low.shift(1) * 1.02) & 
                                                      (df['Close'] > prev_close)).astype(int)
            
            # 5. Range-bound Market Detection
            range_size = (rolling_high - rolling_low) / df['Close'].rolling(period).mean()
            features[f'Range_Size_{period}'] = range_size
            features[f'Tight_Range_{period}'] = (range_size < range_size.rolling(period).quantile(0.3)).astype(int)
        
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
        
        print("Generating volatility contraction indicators...")
        
        for period in self.lookback_periods:
            # 1. Bollinger Band Squeeze
            bb_high = ta.volatility.bollinger_hband(df['Close'], window=period)
            bb_low = ta.volatility.bollinger_lband(df['Close'], window=period)
            kc_high = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'], window=period)
            kc_low = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'], window=period)
            
            features[f'BB_Squeeze_{period}'] = ((bb_high < kc_high) & (bb_low > kc_low)).astype(int)
            
            # 2. Average True Range Compression
            atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
            atr_ma = atr.rolling(period).mean()
            features[f'ATR_Compression_{period}'] = atr / atr_ma
            features[f'Low_Volatility_{period}'] = (features[f'ATR_Compression_{period}'] < 0.8).astype(int)
            
            # 3. High-Low Range Contraction
            hl_range = (df['High'] - df['Low']) / df['Close']
            avg_range = hl_range.rolling(period).mean()
            features[f'Range_Contraction_{period}'] = hl_range / avg_range
            features[f'Narrow_Range_{period}'] = (features[f'Range_Contraction_{period}'] < 0.7).astype(int)
            
            # 4. Price Stability (low standard deviation)
            price_std = df['Close'].rolling(period).std()
            price_mean = df['Close'].rolling(period).mean()
            features[f'Price_Stability_{period}'] = price_std / price_mean
            features[f'Stable_Price_{period}'] = (features[f'Price_Stability_{period}'] < 
                                                features[f'Price_Stability_{period}'].rolling(period).quantile(0.3)).astype(int)
        
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
        
        print("Generating momentum exhaustion indicators...")
        
        for period in self.lookback_periods:
            # 1. MACD Signal Line Crossover
            macd_line = ta.trend.macd(df['Close'], window_slow=period, window_fast=period//2)
            macd_signal = ta.trend.macd_signal(df['Close'], window_slow=period, window_fast=period//2)
            features[f'MACD_Bullish_Cross_{period}'] = ((macd_line > macd_signal) & 
                                                       (macd_line.shift(1) <= macd_signal.shift(1))).astype(int)
            features[f'MACD_Bearish_Cross_{period}'] = ((macd_line < macd_signal) & 
                                                       (macd_line.shift(1) >= macd_signal.shift(1))).astype(int)
            
            # 2. ROC (Rate of Change) Deceleration
            roc = ta.momentum.roc(df['Close'], window=period)
            roc_change = roc - roc.shift(1)
            features[f'ROC_Deceleration_{period}'] = roc_change
            features[f'Momentum_Exhaustion_{period}'] = ((roc > 0) & (roc_change < 0)).astype(int) | \
                                                       ((roc < 0) & (roc_change > 0)).astype(int)
            
            # 3. Commodity Channel Index (CCI) Extremes and Reversals
            cci = ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)
            features[f'CCI_Extreme_{period}'] = ((cci > 100) | (cci < -100)).astype(int)
            features[f'CCI_Reversal_{period}'] = ((cci.shift(1) > 100) & (cci < 100)).astype(int) | \
                                                 ((cci.shift(1) < -100) & (cci > -100)).astype(int)
            
            # 4. Awesome Oscillator Zero Line Cross
            ao = ta.momentum.awesome_oscillator(df['High'], df['Low'], window1=period//2, window2=period)
            features[f'AO_Zero_Cross_{period}'] = ((ao > 0) & (ao.shift(1) <= 0)).astype(int) | \
                                                  ((ao < 0) & (ao.shift(1) >= 0)).astype(int)
        
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
        
        print("Generating custom mean reversion indicators...")
        
        for period in self.lookback_periods:
            # 1. Percentage of Time in Upper/Lower Half of Range
            rolling_high = df['High'].rolling(period).max()
            rolling_low = df['Low'].rolling(period).min()
            range_midpoint = (rolling_high + rolling_low) / 2
            
            above_mid = (df['Close'] > range_midpoint).rolling(period).mean()
            features[f'Time_Above_Mid_{period}'] = above_mid
            features[f'Range_Extreme_{period}'] = ((above_mid > 0.8) | (above_mid < 0.2)).astype(int)
            
            # 2. Consecutive Periods in Same Direction (trend exhaustion)
            price_direction = (df['Close'] > df['Close'].shift(1)).astype(int)
            consecutive_up = (price_direction.rolling(period).sum() == period).astype(int)
            consecutive_down = (price_direction.rolling(period).sum() == 0).astype(int)
            features[f'Trend_Exhaustion_{period}'] = consecutive_up | consecutive_down
            
            # 3. Mean Reversion Speed (how fast price returns to mean)
            sma = df['Close'].rolling(period).mean()
            distance_from_mean = abs(df['Close'] - sma)
            features[f'Distance_From_Mean_{period}'] = distance_from_mean / sma
            
            # Price returning to mean signal
            was_far_from_mean = distance_from_mean.shift(1) > distance_from_mean.shift(1).rolling(period).quantile(0.7)
            is_close_to_mean = distance_from_mean < distance_from_mean.rolling(period).quantile(0.3)
            features[f'Return_To_Mean_{period}'] = (was_far_from_mean & is_close_to_mean).astype(int)
            
            # 4. False Breakout Detection
            breakout_up = df['Close'] > rolling_high.shift(1)
            breakout_down = df['Close'] < rolling_low.shift(1)
            
            # False breakout if price breaks but then reverses quickly
            false_breakout_up = (breakout_up.shift(1) & (df['Close'] < rolling_high.shift(2))).astype(int)
            false_breakout_down = (breakout_down.shift(1) & (df['Close'] > rolling_low.shift(2))).astype(int)
            features[f'False_Breakout_{period}'] = false_breakout_up | false_breakout_down
            
            # 5. Oscillator Divergence (price makes new high/low but oscillator doesn't)
            rsi = ta.momentum.rsi(df['Close'], window=period)
            
            # Price making new highs but RSI not confirming
            price_new_high = df['Close'] == df['Close'].rolling(period//2).max()
            rsi_lower_high = rsi < rsi.shift(period//4)
            bearish_divergence = (price_new_high & rsi_lower_high).astype(int)
            
            # Price making new lows but RSI not confirming  
            price_new_low = df['Close'] == df['Close'].rolling(period//2).min()
            rsi_higher_low = rsi > rsi.shift(period//4)
            bullish_divergence = (price_new_low & rsi_higher_low).astype(int)
            
            features[f'Bullish_Divergence_{period}'] = bullish_divergence
            features[f'Bearish_Divergence_{period}'] = bearish_divergence
        
            features[f'Channel_Top_{period}'] = ((df['Close'] - sma) / sma > 0.01).astype(int)
            features[f'Channel_Bottom_{period}'] = ((sma - df['Close']) / sma > 0.01).astype(int)
        
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
            
            # Feature type priority (prefer certain indicator types for mean reversion)
            feature_type_priority = 0
            if any(x in feature_name for x in ['RSI_', 'BB_Position', 'Price_ZScore', 'Distance_To_']):
                feature_type_priority = 5  # High priority for key mean reversion indicators
            elif any(x in feature_name for x in ['Williams_', 'Stoch_', 'MACD_', 'Reversal_']):
                feature_type_priority = 3  # Medium priority
            elif any(x in feature_name for x in ['Range_', 'Volatility', 'Squeeze']):
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
                               eurusd_file_path: str = "EURUSD.csv",
                               save_features: bool = True,
                               output_file: str = "mean_reversion_features.csv",
                               correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        Run the complete mean reversion feature engineering pipeline.
        
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
            DataFrame with mean reversion features
        """
        print("=== Mean Reversion Feature Engineering Pipeline ===")
        
        # Step 1: Load and prepare data
        print("1. Loading EURUSD data...")
        monthly_data = self.load_eurusd_data(eurusd_file_path)
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
        selected_features = self.remove_correlated_features(valid_features, threshold=0.3)
        
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
            'Mean Reversion': [col for col in selected_features.columns if any(x in col for x in ['Price_ZScore', 'BB_Position', 'RSI_', 'Williams_', 'Stoch_'])],
            'Support/Resistance': [col for col in selected_features.columns if any(x in col for x in ['Distance_To_', 'Near_', 'Donchian_', 'Reversal_', 'Range_'])],
            'Volatility Contraction': [col for col in selected_features.columns if any(x in col for x in ['BB_Squeeze', 'ATR_Compression', 'Range_Contraction', 'Price_Stability', 'Low_Volatility', 'Narrow_Range', 'Stable_Price'])],
            'Momentum Exhaustion': [col for col in selected_features.columns if any(x in col for x in ['MACD_', 'ROC_', 'CCI_', 'AO_', 'Momentum_Exhaustion'])],
            'Custom Mean Reversion': [col for col in selected_features.columns if any(x in col for x in ['Time_Above_Mid', 'Trend_Exhaustion', 'Return_To_Mean', 'False_Breakout', 'Divergence', 'Channel_'])]
        }
        
        print("\nFeature categories:")
        for category, features in categories.items():
            print(f"  {category}: {len(features)} features")
        
        return selected_features


def main():
    """Main function to run the mean reversion feature engineering pipeline."""
    
    # Initialize feature engineering class with different lookback periods
    feature_engineer = MeanReversionFeatureEngineering(
        lookback_periods=[3, 6]  # 3, 6, 12, and 18 months
    )
    
    # Run the complete pipeline
    try:
        features = feature_engineer.run_feature_engineering(
            eurusd_file_path="EURUSD.csv",
            save_features=True,
            output_file="mean_reversion_features.csv",
            correlation_threshold=0.5  # Threshold for final analysis
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
        
        if max_corr < 0.5:
            print("✅ All features have correlation < 0.5")
        else:
            print("⚠️ Some features still have high correlation")
        
    except Exception as e:
        print(f"Error in mean reversion pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
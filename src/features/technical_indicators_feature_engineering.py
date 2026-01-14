import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

class TechnicalIndicatorsFeatureEngineering:
    """
    Feature engineering class that creates technical indicators from FX price data.
    Supports multiple currency pairs (EURUSD, USDJPY, etc.).
    Focuses on uncorrelated indicators across different categories: trend, momentum, volatility.
    Optimized for monthly data analysis without volume indicators.
    """
    
    def __init__(self, lookback_periods: List[int] = [6, 12, 24], currency_pairs: List[str] = ['EURUSD']):
        """
        Initialize the technical indicators feature engineering class.
        
        Parameters:
        -----------
        lookback_periods : List[int]
            Different lookback periods for indicators (default: [6, 12, 24] months)
        currency_pairs : List[str]
            List of currency pairs to process (default: ['EURUSD'])
            Supported pairs: 'EURUSD', 'USDJPY'
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
            features[f'trend_ADX_{period}'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=period)
            
            # 2. Aroon Oscillator - Trend direction and strength
            features[f'trend_Aroon_Osc_{period}'] = ta.trend.aroon_up(df['High'], df['Low'], window=period) - ta.trend.aroon_down(df['High'], df['Low'], window=period)
            
            # 3. Commodity Channel Index (CCI) - Trend strength and reversals
            features[f'trend_CCI_{period}'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)
            
            # 4. Vortex Indicator - Trend direction
            features[f'trend_VI_Diff_{period}'] = (ta.trend.vortex_indicator_pos(df['High'], df['Low'], df['Close'], window=period) - 
                                            ta.trend.vortex_indicator_neg(df['High'], df['Low'], df['Close'], window=period))
            
            # 5. MACD Line - Trend momentum
            features[f'trend_MACD_{period}'] = ta.trend.macd(df['Close'], window_slow=period, window_fast=period//2)
        
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
            features[f'momentum_ROC_{period}'] = ta.momentum.roc(df['Close'], window=period)
            
            # 2. Williams %R - Overbought/oversold momentum
            features[f'momentum_Williams_R_{period}'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=period)
            
            # 3. Ultimate Oscillator - Multi-timeframe momentum
            features[f'momentum_UO_{period}'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'], 
                                                                        window1=period//3, window2=period//2, window3=period)
            
            # 4. Stochastic Oscillator %K - Momentum position within range
            features[f'momentum_Stoch_K_{period}'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], 
                                                              window=period, smooth_window=3)
            
            # 5. RSI - Relative Strength Index
            features[f'momentum_RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)
        
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
            features[f'volatility_ATR_{period}'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=period)
            
            # 2. Bollinger Band Width - Volatility squeeze/expansion
            bb_high = ta.volatility.bollinger_hband(df['Close'], window=period)
            bb_low = ta.volatility.bollinger_lband(df['Close'], window=period)
            features[f'volatility_BB_Width_{period}'] = (bb_high - bb_low) / df['Close']
            
            # 3. Bollinger Band Position - Price position within bands
            bb_mavg = ta.volatility.bollinger_mavg(df['Close'], window=period)
            features[f'volatility_BB_Position_{period}'] = (df['Close'] - bb_low) / (bb_high - bb_low)
            
            # 4. Keltner Channel Width - Alternative volatility measure
            kc_high = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'], window=period)
            kc_low = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'], window=period)
            features[f'volatility_KC_Width_{period}'] = (kc_high - kc_low) / df['Close']
            
            # 5. Donchian Channel Width - Range-based volatility
            dc_high = ta.volatility.donchian_channel_hband(df['High'], df['Low'], df['Close'], window=period)
            dc_low = ta.volatility.donchian_channel_lband(df['High'], df['Low'], df['Close'], window=period)
            features[f'volatility_DC_Width_{period}'] = (dc_high - dc_low) / df['Close']
        
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
            features[f'oscillator_AO_{period}'] = ta.momentum.awesome_oscillator(df['High'], df['Low'], 
                                                                      window1=period//2, window2=period)
            
            # 2. KAMA - Adaptive moving average for trend following
            features[f'oscillator_KAMA_{period}'] = ta.momentum.kama(df['Close'], window=period)
            features[f'oscillator_KAMA_Signal_{period}'] = (df['Close'] > features[f'oscillator_KAMA_{period}']).astype(int)
            
            # 3. Percent Price Oscillator (PPO) - MACD variant
            features[f'oscillator_PPO_{period}'] = ta.momentum.ppo(df['Close'], window_slow=period, window_fast=period//2)
            
            # 4. Stochastic RSI - Combined stochastic and RSI
            features[f'oscillator_StochRSI_{period}'] = ta.momentum.stochrsi(df['Close'], window=period)
        
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
            features[f'custom_Return_Vol_{period}'] = monthly_returns.rolling(period).std()
            
            # 2. High-Low Range Percentage
            features[f'custom_HL_Range_Pct_{period}'] = ((df['High'] - df['Low']) / df['Close']).rolling(period).mean()
            
            # 3. Close Position in Range (where close is relative to high-low range)
            features[f'custom_Close_Position_{period}'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).rolling(period).mean()
            
            # 4. Trend Consistency (percentage of up months)
            up_months = (df['Close'] > df['Close'].shift(1)).rolling(period).mean()
            features[f'custom_Trend_Consistency_{period}'] = up_months
            
            # 5. Monthly Gap Analysis (gap between current open and previous close)
            gaps = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            features[f'custom_Avg_Gap_{period}'] = gaps.rolling(period).mean()
            
            # 6. Volatility Adjusted Return
            features[f'custom_Vol_Adj_Return_{period}'] = (monthly_returns.rolling(period).mean() / 
                                                   monthly_returns.rolling(period).std())
            
            # 7. Price Momentum (simple price change)
            features[f'custom_Price_Momentum_{period}'] = (df['Close'] / df['Close'].shift(period) - 1)
            
            # 8. High-Low Momentum
            features[f'custom_HL_Momentum_{period}'] = ((df['High'] + df['Low']) / 2).pct_change(periods=period)
        
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
        Optimized for many frequencies (e.g., 6-24 months).
        
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
            # Extract period from feature name dynamically
            import re
            period_match = re.search(r'_(\d+)(?:$|[^0-9])', feature_name)
            
            if period_match:
                period = int(period_match.group(1))
                # Shorter periods get higher priority (inverse relationship)
                # Period 6 gets ~95 points, period 24 gets ~76 points
                period_priority = 100 - period
            else:
                period_priority = 50  # Default for features without clear period
            
            # Feature type priority (prefer certain indicator types based on prefix)
            feature_type_priority = 0
            if feature_name.startswith('trend_'):
                feature_type_priority = 10  # High priority for trend indicators
            elif feature_name.startswith('momentum_'):
                feature_type_priority = 8  # Medium-high priority for momentum
            elif feature_name.startswith('volatility_'):
                feature_type_priority = 7  # Medium priority for volatility
            elif feature_name.startswith('oscillator_'):
                feature_type_priority = 5  # Lower priority for oscillators
            elif feature_name.startswith('custom_'):
                feature_type_priority = 4  # Lowest priority for custom
            
            # Variance-based priority: features with more variance are more informative
            try:
                variance_priority = features[feature_name].var()
                # Normalize to 0-10 range
                variance_priority = min(10, variance_priority * 100)
            except:
                variance_priority = 0
            
            return period_priority + feature_type_priority + variance_priority
        
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
                # Enhanced selection strategy for large clusters
                # Strategy: Keep diverse representatives across the period range
                cluster.sort(key=lambda x: feature_priorities[x], reverse=True)
                
                # Extract base indicator name (without period)
                import re
                def get_base_indicator(feat_name):
                    return re.sub(r'_\d+$', '', feat_name)
                
                def extract_period(feat_name):
                    match = re.search(r'_(\d+)$', feat_name)
                    return int(match.group(1)) if match else None
                
                # Group by base indicator
                indicator_groups = {}
                for feat in cluster:
                    base = get_base_indicator(feat)
                    if base not in indicator_groups:
                        indicator_groups[base] = []
                    indicator_groups[base].append(feat)
                
                # For each base indicator, keep strategic representatives
                keep_features = []
                for base, feat_list in indicator_groups.items():
                    if len(feat_list) <= 3:
                        # Keep all if small group
                        keep_features.extend(feat_list[:1])  # Keep best one
                    else:
                        # Keep representatives from short, medium, and long periods
                        periods = [(extract_period(f), f) for f in feat_list]
                        periods = [p for p in periods if p[0] is not None]
                        periods.sort(key=lambda x: x[0])  # Sort by period
                        
                        if len(periods) > 0:
                            # Keep shortest period (most reactive)
                            keep_features.append(periods[0][1])
                            
                            # If many periods, also keep medium
                            if len(periods) >= 7:
                                mid_idx = len(periods) // 2
                                keep_features.append(periods[mid_idx][1])
                            
                            # If very many periods, also keep longest
                            if len(periods) >= 13:
                                keep_features.append(periods[-1][1])
                
                # Sort kept features by priority
                keep_features = sorted(set(keep_features), key=lambda x: feature_priorities[x], reverse=True)
                
                # Remove duplicates and ensure we keep at least one
                if not keep_features:
                    keep_features = [cluster[0]]
                
                remove_features = [f for f in cluster if f not in keep_features]
                
                # Calculate max correlation within the cluster
                max_corr = 0
                for f1 in cluster:
                    for f2 in cluster:
                        if f1 != f2:
                            max_corr = max(max_corr, corr_matrix.loc[f1, f2])
                
                correlation_groups.append({
                    'group': cluster,
                    'kept': keep_features,
                    'removed': remove_features,
                    'max_correlation': max_corr,
                    'cluster_size': len(cluster)
                })
                
                features_to_remove.update(remove_features)
        
        # Print correlation analysis with enhanced details
        print(f"Found {len(correlation_groups)} correlation groups:")
        
        # Sort groups by size (largest first) for more informative output
        sorted_groups = sorted(correlation_groups, key=lambda x: x['cluster_size'], reverse=True)
        
        for i, group in enumerate(sorted_groups[:20]):  # Show first 20 groups
            kept_str = ', '.join(group['kept']) if isinstance(group['kept'], list) else group['kept']
            print(f"  Group {i+1} (size: {group['cluster_size']}): Kept {kept_str}")
            print(f"    Max correlation: {group['max_correlation']:.3f}")
            if len(group['removed']) > 0:
                print(f"    Removed: {', '.join(group['removed'][:5])}" + 
                      (f" + {len(group['removed'])-5} more" if len(group['removed']) > 5 else ""))
        
        if len(correlation_groups) > 20:
            print(f"  ... and {len(correlation_groups) - 20} more groups")
        
        # Print statistics about cluster sizes
        cluster_sizes = [g['cluster_size'] for g in correlation_groups]
        if cluster_sizes:
            print(f"\nCluster size statistics:")
            print(f"  Mean cluster size: {np.mean(cluster_sizes):.1f}")
            print(f"  Max cluster size: {max(cluster_sizes)}")
            print(f"  Clusters with >10 features: {sum(1 for s in cluster_sizes if s > 10)}")
        
        # Remove correlated features
        selected_features = features.drop(columns=list(features_to_remove))
        
        print(f"\nFeature selection summary:")
        print(f"  Original features: {len(features.columns)}")
        print(f"  Correlation groups: {len(correlation_groups)}")
        print(f"  Removed features: {len(features_to_remove)}")
        print(f"  Final features: {len(selected_features.columns)}")
        
        # Show distribution of kept features by timeframe (dynamic for all periods)
        import re
        timeframe_count = {}
        
        for col in selected_features.columns:
            period_match = re.search(r'_(\d+)(?:$|[^0-9])', col)
            if period_match:
                period = period_match.group(1)
                timeframe_count[period] = timeframe_count.get(period, 0) + 1
            else:
                timeframe_count['other'] = timeframe_count.get('other', 0) + 1
        
        print(f"\nFinal features by timeframe:")
        # Sort by period number
        sorted_periods = sorted([k for k in timeframe_count.keys() if k != 'other'], 
                               key=lambda x: int(x))
        
        # Group into ranges for cleaner output
        short_term = sum(timeframe_count.get(str(p), 0) for p in range(6, 11))
        medium_term = sum(timeframe_count.get(str(p), 0) for p in range(11, 18))
        long_term = sum(timeframe_count.get(str(p), 0) for p in range(18, 25))
        
        print(f"  Short-term (6-10 months): {short_term} features")
        print(f"  Medium-term (11-17 months): {medium_term} features")
        print(f"  Long-term (18-24 months): {long_term} features")
        print(f"  Other/No period: {timeframe_count.get('other', 0)} features")
        
        # Detailed breakdown if requested
        if sorted_periods:
            print(f"\n  Detailed breakdown:")
            for period in sorted_periods:
                count = timeframe_count[period]
                if count > 0:
                    print(f"    {period}-month: {count} features")
        
        return selected_features
    
    def run_feature_engineering(self, 
                               fx_file_path: str = None,
                               currency_pair: str = None,
                               save_features: bool = True,
                               output_file: str = None,
                               correlation_threshold: float = 0.8) -> pd.DataFrame:
        """
        Run the complete technical indicators feature engineering pipeline for a specific currency pair.
        
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
            DataFrame with technical indicator features
        """
        # Set default currency pair
        if currency_pair is None:
            currency_pair = self.currency_pairs[0]
        
        # Set default file paths
        if fx_file_path is None:
            fx_file_path = f"{currency_pair}.csv"
        
        if output_file is None:
            output_file = f"technical_indicators_features_{currency_pair}.csv"
        
        print(f"=== Technical Indicators Feature Engineering Pipeline for {currency_pair} ===")
        
        # Step 1: Load and prepare data
        print(f"1. Loading {currency_pair} data from {fx_file_path}...")
        monthly_data = self.load_eurusd_data(fx_file_path, currency_pair)
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
           trend_features,
           momentum_features, 
           volatility_features,
           oscillator_features,
           custom_features
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
        
        print(f"\n=== Feature Engineering Complete for {currency_pair} ===")
        print(f"Final dataset shape: {selected_features.shape}")
        
        # Print feature categories summary
        categories = {
            'Trend': [col for col in selected_features.columns if col.startswith('trend_')],
            'Momentum': [col for col in selected_features.columns if col.startswith('momentum_')],
            'Volatility': [col for col in selected_features.columns if col.startswith('volatility_')],
            'Oscillators': [col for col in selected_features.columns if col.startswith('oscillator_')],
            'Custom': [col for col in selected_features.columns if col.startswith('custom_')]
        }
        
        print("\nFeature categories:")
        for category, features in categories.items():
            print(f"  {category}: {len(features)} features")
        
        return selected_features
    
    def run_feature_engineering_all_pairs(self,
                                         save_features: bool = True,
                                         correlation_threshold: float = 0.8) -> Dict[str, pd.DataFrame]:
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
        print("=== Technical Indicators Feature Engineering - All Currency Pairs ===")
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
                    output_file=f"technical_indicators_features_{pair}.csv",
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
    """Main function to run the technical indicators feature engineering pipeline."""
    
    # Initialize feature engineering class with different lookback periods and currency pairs
    feature_engineer = TechnicalIndicatorsFeatureEngineering(
        lookback_periods=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 9 and 18 months
        #currency_pairs=['EURUSD', 'USDJPY', 'EURJPY', 'GBPUSD', 'USDCHF']  # Multiple currency pairs
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
            
            if max_corr < 0.7:
                print(f"✅ All {pair} features have correlation < 0.7")
            else:
                print(f"⚠️ Some {pair} features still have high correlation")
        
        print(f"\n{'='*80}")
        print(f"=== All Currency Pairs Processed Successfully ===")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"Error in technical indicators pipeline: {e}")
        raise


if __name__ == "__main__":
    main()

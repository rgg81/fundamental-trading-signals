import pandas as pd
import numpy as np
from typing import List
import warnings
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

class AdvancedSpreadFeatureEngineering:
    """
    Advanced feature engineering class that creates EU-US spread indicators from macroeconomic data.
    Focuses on advanced statistical and signal processing features derived from the same base spreads
    as spread_feature_engineering.py, but using different transformation methods to create
    uncorrelated features.
    
    Feature Types:
    - Rank-based features (percentile ranks)
    - Fractal and chaos theory features (Hurst exponent, fractal dimension)
    - Statistical distribution features (skewness, kurtosis)
    - Regime detection features (change points, breakouts)
    - Cross-sectional features (spread relationships)
    - Time series decomposition features (seasonal, residual components)
    """
    
    def __init__(self, windows: List[int] = [3, 6]):
        """
        Initialize the advanced spread feature engineering class.
        
        Parameters:
        -----------
        windows : List[int]
            Different windows for calculations (default: [3, 6] months)
        """
        self.windows = windows
        
        # Define spread pairs (EU indicator - US indicator) - SAME AS ORIGINAL
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
            
            # Instead of dropping all NaN rows, forward fill missing values for recent data
            # This helps maintain recent observations with some data rather than losing them entirely
            #df = df.fillna(method='ffill')
            
            # Only drop rows where ALL values are missing
            #df = df.dropna(how='all')
            
            print(f"   Loaded data shape: {df.shape}")
            print(f"   Missing values per column: {df.isnull().sum().to_dict()}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error loading macro data: {e}")
    
    def generate_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate spread features between EU and US indicators.
        EXACT SAME METHOD AS ORIGINAL SCRIPT.
        
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
    
    def add_rank_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate sophisticated rank-based features for trend analysis, mean reversion,
        and momentum detection using percentile ranks across different market regimes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create rank features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with advanced rank features added
        """
        print(f"Generating advanced rank-based features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    lookback_window = window * 4  # Extended lookback for better rank context
                    
                    # 1. Basic Percentile Rank (foundational)
                    #rolling_rank = df[feature].rolling(window=window).rank(pct=True)
                    #df[f"{feature}_pct_rank_{window}"] = rolling_rank
                    
                    ## 2. Trend-Based Ranks
                    ## Velocity rank: rank of rate of change
                    velocity = df[feature].pct_change(periods=window//2)
                    df[f"{feature}_velocity_rank_{window}"] = velocity.rolling(window=lookback_window, min_periods=window).rank(pct=True)
                    
                    ## Acceleration rank: rank of change in velocity  
                    acceleration = velocity.diff(periods=window//3)
                    df[f"{feature}_accel_rank_{window}"] = acceleration.rolling(window=lookback_window, min_periods=window).rank(pct=True)
                    
                    ## Trend strength rank: rank based on how consistently trending
                    trend_consistency = df[feature].rolling(window=window).apply(
                       lambda x: len([i for i in range(1, len(x)) if (x.iloc[i] - x.iloc[i-1]) * (x.iloc[-1] - x.iloc[0]) > 0]) / max(1, len(x)-1)
                    )
                    df[f"{feature}_trend_strength_rank_{window}"] = trend_consistency.rolling(window=lookback_window, min_periods=window).rank(pct=True)
                    
                    # 3. Mean Reversion Ranks
                    # Deviation from mean rank: how far from historical average
                    #rolling_mean_long = df[feature].rolling(window=3*window).mean()
                    #rolling_mean = df[feature].rolling(window=window).mean()
                    #mean_deviation = (rolling_mean / rolling_mean_long)
                    #df[f"{feature}_mean_dev_rank_{window}"] = mean_deviation
                    
                    # Reversion probability rank: likelihood of mean reversion
                    #distance_from_mean = np.abs(df[feature] - rolling_mean)
                    #df[f"{feature}_reversion_prob_rank_{window}"] = distance_from_mean.rolling(window=window, min_periods=window).rank(pct=True)

                    # Volatility-adjusted rank: current position relative to volatility-adjusted historical range
                    #rolling_std = df[feature].rolling(window=lookback_window, min_periods=window).std()
                    #vol_adj_position = (df[feature] - rolling_mean) / (rolling_std + 1e-8)
                    #df[f"{feature}_vol_adj_rank_{window}"] = vol_adj_position.rolling(window=lookback_window, min_periods=window).rank(pct=True)
                    
                    # 4. Momentum Persistence Ranks
                    # Momentum streak rank: rank based on consecutive directional moves
                    price_changes = df[feature].diff()
                    momentum_streaks = []
                    current_streak = 0
                    last_direction = 0

                    for change in price_changes:
                        if pd.isna(change):
                            momentum_streaks.append(np.nan)
                            continue
                        
                        direction = 1 if change > 0 else -1 if change < 0 else 0
                        if direction == last_direction and direction != 0:
                            current_streak += 1
                        else:
                            current_streak = 1 if direction != 0 else 0
                        
                        momentum_streaks.append(current_streak * direction)
                        last_direction = direction
                    
                    df[f"{feature}_momentum_streak_{window}"] = pd.Series(momentum_streaks, index=df.index)
                    df[f"{feature}_momentum_rank_{window}"] = df[f"{feature}_momentum_streak_{window}"].rolling(window=window).rank(pct=True)
                    
                    # # 5. Regime-Based Ranks
                    # # High/Low regime rank: position within recent high/low range
                    #rolling_high = df[feature].rolling(window=lookback_window, min_periods=window).max()
                    #rolling_low = df[feature].rolling(window=lookback_window, min_periods=window).min()
                    #regime_position = (df[feature] - rolling_low) / (rolling_high - rolling_low + 1e-8)
                    #df[f"{feature}_regime_rank_{window}"] = regime_position
                    
                    # # Volatility regime rank: current volatility vs historical volatility distribution
                    #vol_window = max(2, window//2)  # Ensure minimum window of 2
                    #current_vol = df[feature].rolling(window=vol_window, min_periods=2).std()
                    #df[f"{feature}_vol_regime_rank_{window}"] = current_vol.rolling(window=lookback_window, min_periods=window).rank(pct=True)
                    
                    # # 6. Cross-Temporal Ranks
                    # # Multi-timeframe rank: average rank across different windows
                    # short_window = max(2, window//2)  # Ensure minimum window of 2
                    # short_rank = df[feature].rolling(window=short_window, min_periods=max(1, short_window//2)).rank(pct=True)
                    # medium_rank = df[feature].rolling(window=window, min_periods=max(1, window//2)).rank(pct=True)
                    # long_rank = df[feature].rolling(window=window*2, min_periods=window).rank(pct=True)
                    
                    # # # Weighted average of ranks (shorter timeframes get higher weight)
                    # multi_timeframe_rank = (0.5 * short_rank + 0.3 * medium_rank + 0.2 * long_rank)
                    # df[f"{feature}_multi_tf_rank_{window}"] = multi_timeframe_rank
                    
                    # # # Rank divergence: difference between short and long-term ranks
                    # rank_divergence = short_rank - long_rank
                    # df[f"{feature}_rank_divergence_{window}"] = rank_divergence
                    
                    #                     # 7. Adaptive Ranks
                    # # Volatility-adjusted window rank: expand window during volatile periods
                    # base_vol = df[feature].rolling(window=window, min_periods=max(1, window//2)).std()
                    # vol_multiplier = (base_vol / base_vol.rolling(window=lookback_window, min_periods=window).median()).fillna(1)
                    # adaptive_window = np.clip(window * vol_multiplier, window, window*3).astype(int)
                    
                    # # Dynamic rank calculation with adaptive window
                    # adaptive_ranks = []
                    # for i in range(len(df)):
                    #     if i < window:
                    #         adaptive_ranks.append(np.nan)
                    #     else:
                    #         win_size = min(adaptive_window.iloc[i], i+1)
                    #         win_size = max(2, int(win_size))  # Ensure minimum window of 2
                    #         subset = df[feature].iloc[max(0, i-win_size+1):i+1]
                    #         if len(subset) >= 2:
                    #             rank_val = subset.rank(pct=True).iloc[-1]
                    #             adaptive_ranks.append(rank_val)
                    #         else:
                    #             adaptive_ranks.append(np.nan)
                    # df[f"{feature}_adaptive_rank_{window}"] = adaptive_ranks
                    
                    # 8. Extremes and Outlier Ranks
                    # Extreme value rank: how extreme current value is
                    # rolling_mean = df[feature].rolling(window=window).mean()
                    # rolling_std = df[feature].rolling(window=window).std()
                    # rolling_rank = df[feature].rolling(window=window).rank(pct=True)
                    # z_scores = (df[feature] - rolling_mean) / (rolling_std + 1e-8)
                    # extreme_rank = np.abs(z_scores).rolling(window=lookback_window, min_periods=window).rank(pct=True)
                    # df[f"{feature}_extreme_rank_{window}"] = extreme_rank
                    
                    # # Tail rank: position in distribution tails
                    # tail_indicator = np.where(rolling_rank > 0.9, 1,  # Upper tail
                    #                         np.where(rolling_rank < 0.1, -1,  # Lower tail  
                    #                                0))  # Middle
                    # df[f"{feature}_tail_position_{window}"] = tail_indicator
                    
                    # # Clean up temporary columns
                    # df.drop(columns=[f"{feature}_momentum_streak_{window}"], inplace=True, errors='ignore')
        for spread_name, (eu_col, us_col) in self.spread_pairs.items():
            df.drop(spread_name, axis=1, inplace=True, errors='ignore')
        df.drop("VIX", axis=1, inplace=True, errors='ignore')
        
        return df
    
    def add_fractal_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate fractal and chaos theory based features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create fractal features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with fractal features added
        """
        print(f"Generating fractal features (windows: {self.windows})...")
        
        def hurst_exponent(series, max_lag=None):
            """Calculate Hurst exponent using R/S analysis."""
            if len(series) < 10:
                return np.nan
            
            if max_lag is None:
                max_lag = min(len(series) // 4, 20)
            
            lags = range(2, max_lag)
            rs_values = []
            
            for lag in lags:
                if lag >= len(series):
                    break
                    
                # Calculate mean and cumulative deviation
                mean_val = np.mean(series[:lag])
                deviations = np.cumsum(series[:lag] - mean_val)
                
                # Calculate range and standard deviation
                R = np.max(deviations) - np.min(deviations)
                S = np.std(series[:lag])
                
                if S > 0:
                    rs_values.append(R / S)
                else:
                    rs_values.append(np.nan)
            
            if len(rs_values) < 2:
                return np.nan
            
            # Remove NaN values
            rs_values = [rs for rs in rs_values if not np.isnan(rs)]
            if len(rs_values) < 2:
                return np.nan
            
            # Linear regression of log(R/S) vs log(lag)
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            try:
                slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
                return slope
            except:
                return np.nan
        
        def fractal_dimension(series):
            """Calculate fractal dimension using box counting method."""
            if len(series) < 5:
                return np.nan
            
            # Normalize series to [0, 1]
            normalized = (series - np.min(series)) / (np.max(series) - np.min(series) + 1e-8)
            
            # Simple fractal dimension approximation
            changes = np.abs(np.diff(normalized))
            total_variation = np.sum(changes)
            
            if total_variation == 0:
                return 1.0
            
            # Approximate fractal dimension
            n = len(series)
            dimension = 1 + np.log(total_variation) / np.log(n)
            return max(1.0, min(2.0, dimension))  # Bound between 1 and 2
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # Rolling Hurst exponent
                    hurst_values = []
                    fractal_values = []
                    
                    for i in range(len(df)):
                        start_idx = max(0, i - window + 1)
                        end_idx = i + 1
                        
                        if end_idx - start_idx >= window:
                            series = df[feature].iloc[start_idx:end_idx].values
                            
                            # Calculate Hurst exponent
                            hurst = hurst_exponent(series)
                            hurst_values.append(hurst)
                            
                            # Calculate fractal dimension
                            fractal = fractal_dimension(series)
                            fractal_values.append(fractal)
                        else:
                            hurst_values.append(np.nan)
                            fractal_values.append(np.nan)
                    
                    df[f"{feature}_hurst_{window}"] = hurst_values
                    df[f"{feature}_fractal_{window}"] = fractal_values
                    
                    # Hurst-based regime detection (trending vs mean-reverting)
                    df[f"{feature}_regime_{window}"] = (pd.Series(hurst_values) > 0.5).astype(int)
        
        return df
    
    def add_distribution_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate statistical distribution features (skewness, kurtosis, etc.).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create distribution features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with distribution features added
        """
        print(f"Generating distribution features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # Rolling skewness
                    df[f"{feature}_skew_{window}"] = df[feature].rolling(window=window).skew()
                    
                    # Rolling kurtosis
                    df[f"{feature}_kurt_{window}"] = df[feature].rolling(window=window).kurt()
                    
                    # Rolling median absolute deviation (robust measure of spread)
                    rolling_median = df[feature].rolling(window=window).median()
                    df[f"{feature}_mad_{window}"] = (df[feature] - rolling_median).abs().rolling(window=window).median()
                    
                    # Jarque-Bera normality test statistic (rolling)
                    jb_stats = []
                    for i in range(len(df)):
                        start_idx = max(0, i - window + 1)
                        end_idx = i + 1
                        
                        if end_idx - start_idx >= window:
                            series = df[feature].iloc[start_idx:end_idx].dropna()
                            if len(series) >= 8:  # Minimum for JB test
                                try:
                                    jb_stat, _ = stats.jarque_bera(series)
                                    jb_stats.append(jb_stat)
                                except:
                                    jb_stats.append(np.nan)
                            else:
                                jb_stats.append(np.nan)
                        else:
                            jb_stats.append(np.nan)
                    
                    df[f"{feature}_normality_{window}"] = jb_stats
        
        return df
    
    def add_regime_detection_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate regime detection and change point features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create regime features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime features added
        """
        print(f"Generating regime detection features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # CUSUM (Cumulative Sum) for change point detection
                    mean_val = df[feature].rolling(window=window*2).mean()
                    cusum = (df[feature] - mean_val).cumsum()
                    df[f"{feature}_cusum_{window}"] = cusum
                    
                    # Change point indicator (significant CUSUM deviation)
                    cusum_std = cusum.rolling(window=window).std()
                    df[f"{feature}_changepoint_{window}"] = (np.abs(cusum) > 2 * cusum_std).astype(int)
                    
                    # Level shift detection (mean comparison)
                    past_mean = df[feature].shift(window).rolling(window=window).mean()
                    current_mean = df[feature].rolling(window=window).mean()
                    df[f"{feature}_levelshift_{window}"] = current_mean - past_mean
                    
                    # Breakout intensity (distance from recent range)
                    rolling_max = df[feature].rolling(window=window).max()
                    rolling_min = df[feature].rolling(window=window).min()
                    rolling_range = rolling_max - rolling_min
                    breakout_up = (df[feature] - rolling_max) / (rolling_range + 1e-8)
                    breakout_down = (rolling_min - df[feature]) / (rolling_range + 1e-8)
                    df[f"{feature}_breakout_{window}"] = np.maximum(breakout_up, breakout_down)
                    
                    # Regime persistence (how long in current regime)
                    trend_signal = (df[feature] > df[feature].rolling(window=window).mean()).astype(int)
                    regime_changes = trend_signal.diff() != 0
                    regime_duration = regime_changes.cumsum()
                    df[f"{feature}_persistence_{window}"] = regime_duration
        
        return df
    
    def add_cross_sectional_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate cross-sectional features (relationships between different spreads).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create cross-sectional features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with cross-sectional features added
        """
        print(f"Generating cross-sectional features (windows: {self.windows})...")
        
        # Create pairs of features for cross-sectional analysis
        feature_pairs = []
        base_spreads = [f for f in features if f in ['CPI_Spread', 'Core_CPI_Spread', 'Yield_Spread', 'Rate_Spread']]
        
        for i, feat1 in enumerate(base_spreads):
            for feat2 in base_spreads[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    feature_pairs.append((feat1, feat2))
        
        for feat1, feat2 in feature_pairs:
            for window in self.windows:
                # Correlation between spreads (with min_periods)
                corr = df[feat1].rolling(window=window, min_periods=max(1, window//2)).corr(df[feat2])
                df[f"{feat1}_{feat2}_corr_{window}"] = corr
                
                # Relative strength (ratio of z-scores) with NaN handling
                zscore1 = (df[feat1] - df[feat1].rolling(window=window, min_periods=max(1, window//2)).mean()) / (df[feat1].rolling(window=window, min_periods=max(1, window//2)).std() + 1e-8)
                zscore2 = (df[feat2] - df[feat2].rolling(window=window, min_periods=max(1, window//2)).mean()) / (df[feat2].rolling(window=window, min_periods=max(1, window//2)).std() + 1e-8)
                
                # Only calculate relative strength when both z-scores are valid
                relstrength = np.where(
                    ~np.isnan(zscore1) & ~np.isnan(zscore2) & (np.abs(zscore2) > 1e-8),
                    zscore1 / (zscore2 + 1e-8),
                    np.nan
                )
                df[f"{feat1}_{feat2}_relstrength_{window}"] = relstrength
                
                # Spread divergence (difference in normalized values)
                norm1 = (df[feat1] - df[feat1].rolling(window=window*2, min_periods=window).min()) / (df[feat1].rolling(window=window*2, min_periods=window).max() - df[feat1].rolling(window=window*2, min_periods=window).min() + 1e-8)
                norm2 = (df[feat2] - df[feat2].rolling(window=window*2, min_periods=window).min()) / (df[feat2].rolling(window=window*2, min_periods=window).max() - df[feat2].rolling(window=window*2, min_periods=window).min() + 1e-8)
                df[f"{feat1}_{feat2}_divergence_{window}"] = norm1 - norm2
        
        return df
    
    def add_decomposition_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Generate time series decomposition features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features
        features : List[str]
            List of feature names to create decomposition features for
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with decomposition features added
        """
        print(f"Generating decomposition features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # Trend component (use regular rolling mean to avoid center=True issues)
                    trend = df[feature].rolling(window=window*2, min_periods=window).mean()
                    df[f"{feature}_trend_comp_{window}"] = trend
                    
                    # Detrended series (residual after removing trend)
                    detrended = df[feature] - trend
                    df[f"{feature}_detrended_{window}"] = detrended
                    
                    # Cyclical component (using Hodrick-Prescott filter approximation)
                    # Simple HP filter approximation using second differences
                    lambda_hp = 1600  # Standard for monthly data
                    diff2 = df[feature].diff().diff()
                    cycle_approx = df[feature] - lambda_hp * diff2.rolling(window=window, min_periods=max(1, window//2)).mean()
                    df[f"{feature}_cycle_{window}"] = cycle_approx
                    
                    # Residual variance (unexplained component)
                    residual_var = detrended.rolling(window=window, min_periods=max(1, window//2)).var()
                    df[f"{feature}_residual_var_{window}"] = residual_var
                    
                    # Signal-to-noise ratio (with better NaN handling)
                    signal_var = trend.rolling(window=window, min_periods=max(1, window//2)).var()
                    # Only calculate SNR when both variances are valid and signal_var > 0
                    snr = np.where(
                        (signal_var > 1e-10) & (residual_var > 1e-10) & ~np.isnan(signal_var) & ~np.isnan(residual_var),
                        signal_var / (residual_var + 1e-8),
                        np.nan
                    )
                    df[f"{feature}_snr_{window}"] = snr
        
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
        Remove highly correlated features while prioritizing advanced feature types.
        
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
        
        # Create a priority map for advanced features
        def get_feature_priority(feature_name):
            """Assign priority based on advanced feature type and timeframe."""
            priority = 0
            
            # Base spread features get highest priority
            if any(x in feature_name for x in ['CPI_Spread', 'Yield_Spread', 'Rate_Spread']) and '_' not in feature_name.split('_')[2:]:
                priority += 50
            
            # VIX gets high priority
            if 'VIX' in feature_name and len(feature_name.split('_')) <= 2:
                priority += 45
            
            # Advanced feature type priorities (higher than basic features)
            if any(x in feature_name for x in ['_hurst_', '_fractal_', '_regime_']):
                priority += 40  # Fractal features
            elif any(x in feature_name for x in ['_rank_', '_rank_mom_', '_rank_dev_']):
                priority += 35  # Rank features
            elif any(x in feature_name for x in ['_cusum_', '_changepoint_', '_levelshift_', '_breakout_']):
                priority += 30  # Regime detection
            elif any(x in feature_name for x in ['_corr_', '_relstrength_', '_divergence_']):
                priority += 25  # Cross-sectional
            elif any(x in feature_name for x in ['_skew_', '_kurt_', '_mad_', '_normality_']):
                priority += 20  # Distribution features
            elif any(x in feature_name for x in ['_cycle_', '_snr_', '_detrended_']):
                priority += 15  # Decomposition features
            
            # Shorter timeframes get higher priority
            if '_3' in feature_name:
                priority += 10  # 3-month window
            elif '_6' in feature_name:
                priority += 5   # 6-month window
            
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
            'Base Spreads': [col for col in selected_features.columns if any(x in col for x in ['CPI_Spread', 'Yield_Spread', 'Rate_Spread']) and '_' not in col.split('_')[2:]],
            'VIX': [col for col in selected_features.columns if 'VIX' in col and len(col.split('_')) <= 2],
            'Fractal/Chaos': [col for col in selected_features.columns if any(x in col for x in ['_hurst_', '_fractal_', '_regime_'])],
            'Rank Features': [col for col in selected_features.columns if any(x in col for x in ['_rank_', '_rank_mom_', '_rank_dev_', '_rank_extreme_'])],
            'Regime Detection': [col for col in selected_features.columns if any(x in col for x in ['_cusum_', '_changepoint_', '_levelshift_', '_breakout_', '_persistence_'])],
            'Cross-Sectional': [col for col in selected_features.columns if any(x in col for x in ['_corr_', '_relstrength_', '_divergence_'])],
            'Distribution': [col for col in selected_features.columns if any(x in col for x in ['_skew_', '_kurt_', '_mad_', '_normality_'])],
            'Decomposition': [col for col in selected_features.columns if any(x in col for x in ['_trend_comp_', '_cycle_', '_snr_', '_detrended_', '_residual_var_'])]
        }
        
        print(f"\nFinal features by type:")
        for feature_type, feature_list in feature_types.items():
            print(f"  {feature_type}: {len(feature_list)} features")
        
        return selected_features
    
    def run_feature_engineering(self, 
                               macro_file_path: str = "macro_data.csv",
                               save_features: bool = True,
                               output_file: str = "spread_features_advanced.csv",
                               correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        Run the complete advanced spread feature engineering pipeline.
        
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
            DataFrame with advanced spread features
        """
        print("=== Advanced Spread Feature Engineering Pipeline ===")
        
        # Step 1: Load and prepare data
        print("1. Loading macroeconomic data...")
        macro_data = self.load_macro_data(macro_file_path)
        print(f"   Loaded {len(macro_data)} monthly observations")
        print(f"   Date range: {macro_data.index[0].strftime('%Y-%m')} to {macro_data.index[-1].strftime('%Y-%m')}")
        print(f"   Available columns: {list(macro_data.columns)}")
        
        # Step 2: Generate basic spread features (SAME AS ORIGINAL)
        print("2. Generating spread features...")
        spread_features = self.generate_spread_features(macro_data)
        print(f"   Generated {len(spread_features.columns)} basic spread features")
        
        # Step 3: Generate all ADVANCED feature types
        print("3. Generating advanced features...")
        all_base_features = list(spread_features.columns)
        
        # Add rank-based features
        #spread_features = self.add_rank_features(spread_features, all_base_features)
        # Add fractal and chaos theory features
        #spread_features = self.add_fractal_features(spread_features, all_base_features)
        # Add statistical distribution features
        spread_features = self.add_distribution_features(spread_features, all_base_features)
        # Add regime detection features
        #spread_features = self.add_regime_detection_features(spread_features, all_base_features)
        # Add cross-sectional features
        #spread_features = self.add_cross_sectional_features(spread_features, all_base_features)
        # Add time series decomposition features
        #spread_features = self.add_decomposition_features(spread_features, all_base_features)

        print(f"   Generated {spread_features.shape[1]} total advanced features")
        print(f"   Features available for {len(spread_features)} time periods")
        
        # Step 4: Clean features
        print("4. Cleaning features...")
        # Remove infinite values
        spread_features = spread_features.replace([np.inf, -np.inf], np.nan)
        
        # Remove features with too many NaN values (>60% to be more lenient for recent data)
        nan_threshold = len(spread_features) * 0.1  # Keep features with at least 90% valid data
        valid_features = spread_features.dropna(thresh=nan_threshold, axis=1)

        print(f"   Removed {spread_features.shape[1] - valid_features.shape[1]} features with >90% NaN values")
        print(f"   Final feature count before correlation removal: {valid_features.shape[1]}")
        
        # Final check - count remaining NaN values in recent data
        recent_nan_count = valid_features.tail(12).isnull().sum().sum()
        total_recent_values = valid_features.tail(12).shape[0] * valid_features.tail(12).shape[1]
        recent_nan_pct = (recent_nan_count / total_recent_values) * 100 if total_recent_values > 0 else 0
        print(f"   Recent 12 months NaN percentage: {recent_nan_pct:.1f}% ({recent_nan_count}/{total_recent_values})")
        
        # Step 5: Remove highly correlated features
        print("5. Removing highly correlated features...")
        valid_features = self.remove_correlated_features(valid_features, threshold=correlation_threshold)
        
        # Step 6: Final correlation analysis on selected features
        print("6. Analyzing final feature correlations...")
        final_correlation_matrix = self.calculate_correlation_matrix(valid_features, correlation_threshold)
        
        # Step 7: Save features
        if save_features:
            print(f"7. Saving features to {output_file}...")
            valid_features.to_csv(output_file)
            
            # Save correlation matrix
            corr_file = output_file.replace('.csv', '_correlations.csv')
            final_correlation_matrix.to_csv(corr_file)
            print(f"   Features saved to {output_file}")
            print(f"   Correlations saved to {corr_file}")
        
        print("\n=== Advanced Feature Engineering Complete ===")
        print(f"Final dataset shape: {valid_features.shape}")
        
        return valid_features


def main():
    """Main function to run the advanced spread feature engineering pipeline."""
    
    # Initialize advanced feature engineering class with 6 month windows
    feature_engineer = AdvancedSpreadFeatureEngineering(
        windows=[12, 24]
    )
    
    # Run the complete pipeline
    try:
        features = feature_engineer.run_feature_engineering(
            macro_file_path="macro_data.csv",
            save_features=True,
            output_file="spread_features_advanced.csv",
            correlation_threshold=0.9  # Threshold for correlation analysis
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
        print(f"Error in advanced spread feature engineering pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
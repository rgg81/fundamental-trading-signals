import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import warnings
from scipy import stats
from scipy.signal import find_peaks, argrelextrema
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

class RegimeDetectionFeatureEngineering:
    """
    Specialized feature engineering class focused on regime detection, change points, 
    cycles, and market state identification from EU-US spread indicators.
    
    This class creates sophisticated features to identify:
    - Market regimes (trending, sideways, volatile)
    - Change points and structural breaks
    - Cyclical patterns and seasonality
    - Mean reversion vs momentum regimes
    - Volatility clusters and regimes
    - Correlation regimes between spreads
    """
    
    def __init__(self, windows: List[int] = [3, 6, 12], 
                 currency_pairs: List[str] = ['EURUSD']):
        """
        Initialize the regime detection feature engineering class.
        
        Parameters:
        -----------
        windows : List[int]
            Different windows for calculations (default: [3, 6, 12] months)
        currency_pairs : List[str]
            List of currency pairs to process (default: ['EURUSD'])
        """
        self.windows = windows
        self.currency_pairs = currency_pairs
        
    def load_eurusd_data(self, file_path: str = "EURUSD.csv", 
                         currency_pair: str = "EURUSD") -> pd.DataFrame:
        """
        Load FX price data from CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing FX data
        currency_pair : str
            Currency pair symbol (e.g., 'EURUSD', 'USDJPY')
        """
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            
            # Construct dynamic column names
            close_col = f'{currency_pair}_Close'
            
            # Validate required columns exist
            if close_col not in df.columns:
                raise ValueError(f"Column {close_col} not found in {file_path}. "
                               f"Available columns: {list(df.columns)}")
            
            print(f"   Loaded data shape: {df.shape}")
            print(f"   Missing values per column: {df.isnull().sum().to_dict()}")
            print(f"   Available columns: {list(df.columns)}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading {currency_pair} data from {file_path}: {e}")
    
    def generate_price_features(self, df: pd.DataFrame, 
                               currency_pair: str = "EURUSD") -> pd.DataFrame:
        """
        Generate price-based features from FX data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing FX price data
        currency_pair : str
            Currency pair symbol (e.g., 'EURUSD', 'USDJPY')
        """
        features = pd.DataFrame(index=df.index)
        
        print("Generating price features...")
        
        # Construct dynamic column name
        close_col = f'{currency_pair}_Close'
        
        # 1. Use FX Close price as base feature
        if close_col in df.columns:
            features[f'{currency_pair}_Close'] = df[close_col]
            print(f"   Added {close_col} as base feature")
            
            # 2. Generate additional price-based features for regime detection
            # Price returns (different periods)
            #features[f'{currency_pair}_Return_1M'] = df[close_col].pct_change(periods=1)
            #features[f'{currency_pair}_Return_3M'] = df[close_col].pct_change(periods=3)
            #features[f'{currency_pair}_Return_6M'] = df[close_col].pct_change(periods=6)
            print(f"   Added return features for 1M, 3M, 6M periods")
            
            # Price momentum (log returns)
            #features[f'{currency_pair}_LogReturn'] = np.log(df[close_col] / df[close_col].shift(1))
            #print(f"   Added log return feature")
            
            # Price relative to moving averages (trend indicators)
            for window in [12, 24, 36]:  # 1-year, 2-year, 3-year MAs
                ma = df[close_col].rolling(window=window, min_periods=max(1, window//2)).mean()
                features[f'{currency_pair}_vs_MA{window}'] = df[close_col] / ma - 1
                print(f"   Added price vs MA{window} feature")
            
        else:
           raise ValueError(f"{close_col} column not found in data")
        
        return features
    
    def add_structural_break_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Detect structural breaks and regime changes using multiple statistical tests.
        """
        print(f"Generating structural break features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # 1. CUSUM (Cumulative Sum) for change point detection
                    rolling_mean = df[feature].rolling(window=window*2, min_periods=window).mean()
                    cusum_pos = np.maximum(0, (df[feature] - rolling_mean).cumsum())
                    cusum_neg = np.minimum(0, (df[feature] - rolling_mean).cumsum()) 
                    df[f"{feature}_cusum_pos_{window}"] = cusum_pos
                    df[f"{feature}_cusum_neg_{window}"] = cusum_neg
                    
                    # CUSUM change point intensity
                    cusum_range = cusum_pos - cusum_neg
                    cusum_std = cusum_range.rolling(window=window, min_periods=max(1, window//2)).std()
                    df[f"{feature}_cusum_intensity_{window}"] = cusum_range / (cusum_std + 1e-8)
                    
                    # 2. Level shift detection (mean comparison across periods)
                    past_mean = df[feature].shift(window).rolling(window=window, min_periods=max(1, window//2)).mean()
                    current_mean = df[feature].rolling(window=window, min_periods=max(1, window//2)).mean()
                    # Remove future data leakage - use only past data
                    very_past_mean = df[feature].shift(window*2).rolling(window=window, min_periods=max(1, window//2)).mean()
                    
                    df[f"{feature}_level_shift_{window}"] = current_mean - past_mean
                    # Use past acceleration instead of future: (current - past) - (past - very_past)
                    df[f"{feature}_level_acceleration_{window}"] = (current_mean - past_mean) - (past_mean - very_past_mean)
                    
                    # 3. Variance shift detection (volatility regime changes)
                    past_var = df[feature].shift(window).rolling(window=window, min_periods=max(1, window//2)).var()
                    current_var = df[feature].rolling(window=window, min_periods=max(1, window//2)).var()
                    df[f"{feature}_var_shift_{window}"] = np.log(current_var + 1e-8) - np.log(past_var + 1e-8)
                    
                    # 4. Regime persistence (how long in current state)
                    # Define regime based on deviation from long-term mean
                    long_mean = df[feature].rolling(window=window*4, min_periods=window).mean()
                    long_std = df[feature].rolling(window=window*4, min_periods=window).std()
                    z_score = (df[feature] - long_mean) / (long_std + 1e-8)
                    
                    # Regime classification: High (>1), Normal (-1 to 1), Low (<-1)
                    regime = np.where(z_score > 1, 2,  # High regime
                                    np.where(z_score < -1, 0, 1))  # Low, Normal regimes
                    
                    # Calculate regime persistence (consecutive periods in same regime)
                    regime_changes = pd.Series(regime, index=df.index).diff() != 0
                    regime_duration = []
                    current_duration = 0
                    
                    for is_change in regime_changes:
                        if is_change:
                            current_duration = 1
                        else:
                            current_duration += 1
                        regime_duration.append(current_duration)
                    
                    df[f"{feature}_regime_type_{window}"] = regime
                    df[f"{feature}_regime_duration_{window}"] = regime_duration
                    
                    # 5. Breakout detection (price breaking out of recent range)
                    rolling_max = df[feature].rolling(window=window, min_periods=max(1, window//2)).max()
                    rolling_min = df[feature].rolling(window=window, min_periods=max(1, window//2)).min()
                    rolling_range = rolling_max - rolling_min
                    
                    # Upward and downward breakouts
                    breakout_up = np.maximum(0, (df[feature] - rolling_max) / (rolling_range + 1e-8))
                    breakout_down = np.maximum(0, (rolling_min - df[feature]) / (rolling_range + 1e-8))
                    df[f"{feature}_breakout_up_{window}"] = breakout_up
                    df[f"{feature}_breakout_down_{window}"] = breakout_down
                    
                    # Combined breakout intensity
                    df[f"{feature}_breakout_intensity_{window}"] = breakout_up + breakout_down
        
        return df
    
    def add_cycle_detection_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Detect cyclical patterns, seasonality, and periodic behavior.
        """
        print(f"Generating cycle detection features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # 1. Causal peak and trough detection (no future data)
                    peak_indicator = np.zeros(len(df))
                    trough_indicator = np.zeros(len(df))
                    
                    # Rolling peak/trough detection using only historical data
                    for i in range(window, len(df)):
                        # Use only past data up to current point
                        lookback_window = min(window, i)
                        if lookback_window >= 3:  # Need minimum data for peak detection
                            historical_data = df[feature].iloc[i-lookback_window:i+1].dropna()
                            
                            if len(historical_data) >= 3:
                                values = historical_data.values
                                
                                # Simple causal peak detection: current point is peak if higher than neighbors
                                # and higher than recent local maximum
                                current_val = values[-1]  # Current value
                                
                                # Check if current point is local maximum
                                if len(values) >= 3:
                                    # Current is peak if higher than previous and next isn't available (causal)
                                    # Use rolling maximum of past window to determine if this is significant peak
                                    recent_max = np.max(values[:-1])  # Max excluding current
                                    recent_min = np.min(values[:-1])  # Min excluding current
                                    recent_range = recent_max - recent_min
                                    
                                    # Peak: current value is higher than 90% of recent values
                                    if recent_range > 0:
                                        peak_threshold = recent_min + 0.9 * recent_range
                                        if current_val >= peak_threshold and current_val > values[-2]:
                                            peak_indicator[i] = 1
                                        
                                        # Trough: current value is lower than 10% of recent values  
                                        trough_threshold = recent_min + 0.1 * recent_range
                                        if current_val <= trough_threshold and current_val < values[-2]:
                                            trough_indicator[i] = 1
                    
                    df[f"{feature}_peak_indicator_{window}"] = peak_indicator
                    df[f"{feature}_trough_indicator_{window}"] = trough_indicator
                        
                    # 2. Causal cycle length estimation (time between peaks) - updated for new peak detection
                    cycle_lengths = []
                    peak_positions = np.where(peak_indicator == 1)[0]
                    
                    for i in range(len(df)):
                        # Find average cycle length based on past peaks only
                        past_peaks = peak_positions[peak_positions < i]  # Only use past peaks (exclude current)
                        if len(past_peaks) >= 2:
                            past_peaks = past_peaks[-min(4, len(past_peaks)):]  # Last 4 past peaks
                            cycle_diffs = np.diff(past_peaks)
                            avg_cycle = np.mean(cycle_diffs) if len(cycle_diffs) > 0 else np.nan
                            cycle_lengths.append(avg_cycle)
                        else:
                            cycle_lengths.append(np.nan)
                    
                    df[f"{feature}_cycle_length_{window}"] = cycle_lengths
                        
                    # 3. Causal position in cycle (where are we in the current cycle?)
                    cycle_position = []
                    for i in range(len(df)):
                        if i > 0 and not np.isnan(cycle_lengths[i]):
                            # Only use past peaks to determine cycle position
                            past_peaks = peak_positions[peak_positions < i]
                            if len(past_peaks) > 0:
                                last_peak = past_peaks[-1]
                                time_since_peak = i - last_peak
                                expected_cycle = cycle_lengths[i]
                                if expected_cycle > 0:
                                    position = (time_since_peak / expected_cycle) % 1.0
                                    cycle_position.append(position)
                                else:
                                    cycle_position.append(np.nan)
                            else:
                                cycle_position.append(np.nan)
                        else:
                            cycle_position.append(np.nan)
                    
                    df[f"{feature}_cycle_position_{window}"] = cycle_position
                    
                    # 4. Seasonal decomposition approximation
                    # Simple seasonal pattern detection using autocorrelation
                    if len(df) > window * 3:
                        # Monthly seasonality (12-month pattern) - causal correlation
                        if window >= 12:
                            seasonal_lag = 12
                            # Use only past data for both sides of correlation
                            past_current = df[feature].shift(1)  # Previous period values
                            past_seasonal = df[feature].shift(seasonal_lag + 1)  # Past seasonal values
                            seasonal_corr = past_current.rolling(window=window*2, min_periods=window).corr(
                                past_seasonal
                            )
                            df[f"{feature}_seasonal_corr_{window}"] = seasonal_corr
                        
                        # Quarterly pattern (3-month pattern) - causal correlation
                        quarterly_lag = 3
                        past_current = df[feature].shift(1)  # Previous period values
                        past_quarterly = df[feature].shift(quarterly_lag + 1)  # Past quarterly values
                        quarterly_corr = past_current.rolling(window=window*2, min_periods=window).corr(
                            past_quarterly
                        )
                        df[f"{feature}_quarterly_corr_{window}"] = quarterly_corr
                    
                    # 5. Causal dominant frequency estimation using historical data only
                    dominant_periods = []
                    for i in range(len(df)):
                        if i < window:
                            dominant_periods.append(np.nan)
                        else:
                            # Use only past data (excluding current period)
                            past_subset = df[feature].iloc[max(0, i-window):i].dropna()
                            if len(past_subset) >= window//2:
                                # Simple periodogram using autocorrelation on past data only
                                autocorrs = []
                                for lag in range(1, min(len(past_subset)//2, 6)):
                                    if len(past_subset) > lag:
                                        autocorr = np.corrcoef(past_subset[:-lag], past_subset[lag:])[0, 1]
                                        if not np.isnan(autocorr):
                                            autocorrs.append((lag, abs(autocorr)))
                                
                                if autocorrs:
                                    # Find lag with highest autocorrelation
                                    dominant_lag = max(autocorrs, key=lambda x: x[1])[0]
                                    dominant_periods.append(dominant_lag)
                                else:
                                    dominant_periods.append(np.nan)
                            else:
                                dominant_periods.append(np.nan)
                    
                    df[f"{feature}_dominant_period_{window}"] = dominant_periods
        
        return df
    
    def add_market_state_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Identify market states: trending, sideways, volatile, calm, etc.
        """
        print(f"Generating market state features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # 1. Trend strength measurement
                    # Linear regression slope significance
                    trend_slopes = []
                    trend_r2 = []
                    
                    for i in range(window, len(df)):
                        y_vals = df[feature].iloc[i-window:i].values
                        x_vals = np.arange(len(y_vals))
                        
                        if len(y_vals) >= 3 and not np.all(np.isnan(y_vals)):
                            valid_mask = ~np.isnan(y_vals)
                            if np.sum(valid_mask) >= 3:
                                try:
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                                        x_vals[valid_mask], y_vals[valid_mask]
                                    )
                                    trend_slopes.append(slope)
                                    trend_r2.append(r_value**2)
                                except:
                                    trend_slopes.append(np.nan)
                                    trend_r2.append(np.nan)
                            else:
                                trend_slopes.append(np.nan)
                                trend_r2.append(np.nan)
                        else:
                            trend_slopes.append(np.nan)
                            trend_r2.append(np.nan)
                    
                    # Pad with NaN for initial periods
                    trend_slopes = [np.nan] * window + trend_slopes
                    trend_r2 = [np.nan] * window + trend_r2
                    
                    df[f"{feature}_trend_slope_{window}"] = trend_slopes[:len(df)]
                    df[f"{feature}_trend_r2_{window}"] = trend_r2[:len(df)]
                    
                    # 2. Sideways market detection
                    # High volatility but low trend strength indicates sideways movement
                    rolling_vol = df[feature].rolling(window=window, min_periods=max(1, window//2)).std()
                    rolling_range = (df[feature].rolling(window=window, min_periods=max(1, window//2)).max() - 
                                   df[feature].rolling(window=window, min_periods=max(1, window//2)).min())
                    
                    # Normalize trend strength
                    trend_strength = pd.Series(trend_r2[:len(df)], index=df.index)
                    
                    # Sideways indicator: high volatility, low trend strength
                    vol_percentile = rolling_vol.rolling(window=window*2, min_periods=window).rank(pct=True)
                    trend_percentile = trend_strength.rolling(window=window*2, min_periods=window).rank(pct=True)
                    
                    sideways_score = vol_percentile - trend_percentile  # High vol, low trend = positive score
                    df[f"{feature}_sideways_score_{window}"] = sideways_score
                    
                    # 3. Volatility clustering (GARCH-like behavior)
                    # High volatility followed by high volatility, low by low
                    vol_persistence = rolling_vol.rolling(window=max(2, window//2), min_periods=2).corr(
                        rolling_vol.shift(1)
                    )
                    df[f"{feature}_vol_clustering_{window}"] = vol_persistence
                    
                    # 4. Mean reversion tendency
                    # How often does the series return to its mean after deviations?
                    rolling_mean = df[feature].rolling(window=window*2, min_periods=window).mean()
                    rolling_std = df[feature].rolling(window=window*2, min_periods=window).std()
                    z_score = (df[feature] - rolling_mean) / (rolling_std + 1e-8)
                    
                    # Mean reversion signal: use past relationship instead of future data
                    # Correlation between lagged z-score and subsequent return (using only past data)
                    past_return = df[feature].pct_change()
                    lagged_z_score = z_score.shift(1)  # Previous period's z-score
                    reversion_corr = lagged_z_score.rolling(window=window, min_periods=max(1, window//2)).corr(past_return)
                    df[f"{feature}_mean_reversion_{window}"] = -reversion_corr  # Negative correlation indicates reversion
                    
                    # 5. Momentum persistence
                    # How often do directional moves continue?
                    price_changes = df[feature].pct_change()
                    momentum_signal = price_changes * price_changes.shift(1)  # Same direction = positive
                    momentum_persistence = momentum_signal.rolling(window=window, min_periods=max(1, window//2)).mean()
                    df[f"{feature}_momentum_persistence_{window}"] = momentum_persistence
                    
                    # 6. Causal market regime classification using expanding window approach
                    # Combine trend, volatility, and level information
                    if i % len(self.windows) == 0:  # Only do this once per feature to avoid redundancy
                        regime_features = pd.DataFrame({
                            'level': (df[feature] - df[feature].rolling(window=window*4, min_periods=window).mean()) / 
                                   (df[feature].rolling(window=window*4, min_periods=window).std() + 1e-8),
                            'trend': pd.Series(trend_slopes[:len(df)], index=df.index).fillna(0),
                            'volatility': rolling_vol / (rolling_vol.rolling(window=window*2, min_periods=window).mean() + 1e-8)
                        }).fillna(0)
                        
                        # Use causal expanding window regime classification
                        regime_labels = []
                        fitted_kmeans = None
                        
                        for i in range(len(df)):
                            if i < window*2:  # Need minimum historical data
                                regime_labels.append(np.nan)
                            else:
                                # Use only past data to train clustering model (excluding current period)
                                historical_data = regime_features.iloc[max(0, i-window*3):i].fillna(0)  # Past data only
                                current_features = regime_features.iloc[i:i+1].fillna(0)  # Current period features
                                
                                if len(historical_data) >= 6 and not historical_data.isnull().all().all():
                                    try:
                                        # Train clustering on historical data only
                                        if len(historical_data) >= 9:  # Enough data for 3 clusters
                                            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                                            kmeans.fit(historical_data)
                                            fitted_kmeans = kmeans
                                        elif fitted_kmeans is not None:
                                            # Use previously fitted model if not enough new data
                                            kmeans = fitted_kmeans
                                        else:
                                            # Fall back to simple quantile-based classification
                                            level_val = current_features.iloc[0]['level']
                                            hist_levels = historical_data['level']
                                            if level_val > hist_levels.quantile(0.67):
                                                regime_labels.append(2)  # High regime
                                            elif level_val < hist_levels.quantile(0.33):
                                                regime_labels.append(0)  # Low regime  
                                            else:
                                                regime_labels.append(1)  # Medium regime
                                            continue
                                        
                                        # Predict current regime using historical model
                                        current_regime = kmeans.predict(current_features)[0]
                                        regime_labels.append(current_regime)
                                        
                                    except:
                                        # Fallback to simple classification
                                        level_val = current_features.iloc[0]['level'] if len(current_features) > 0 else 0
                                        hist_levels = historical_data['level'] if len(historical_data) > 0 else pd.Series([0])
                                        if level_val > hist_levels.quantile(0.67):
                                            regime_labels.append(2)
                                        elif level_val < hist_levels.quantile(0.33):  
                                            regime_labels.append(0)
                                        else:
                                            regime_labels.append(1)
                                else:
                                    regime_labels.append(np.nan)
                        
                        df[f"{feature}_market_regime_{window}"] = regime_labels
        
        return df
    
    def add_volatility_regime_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Specialized volatility regime detection and clustering analysis.
        """
        print(f"Generating volatility regime features (windows: {self.windows})...")
        
        for feature in features:
            if feature in df.columns:
                for window in self.windows:
                    # 1. Multi-scale volatility analysis
                    vol_short = df[feature].rolling(window=max(2, window//2), min_periods=2).std()
                    vol_medium = df[feature].rolling(window=window, min_periods=max(1, window//2)).std()
                    vol_long = df[feature].rolling(window=window*2, min_periods=window).std()
                    
                    # Volatility ratios (regime identification)
                    df[f"{feature}_vol_ratio_sm_{window}"] = vol_short / (vol_medium + 1e-8)
                    df[f"{feature}_vol_ratio_ml_{window}"] = vol_medium / (vol_long + 1e-8)
                    
                    # 2. Volatility regime transitions
                    # Detect when volatility moves from one regime to another
                    vol_percentile_short = vol_short.rolling(window=window*2, min_periods=window).rank(pct=True)
                    vol_percentile_long = vol_long.rolling(window=window*2, min_periods=window).rank(pct=True)
                    
                    # High/Low volatility regime (above/below median)
                    vol_regime_short = (vol_percentile_short > 0.5).astype(int)
                    vol_regime_long = (vol_percentile_long > 0.5).astype(int)
                    
                    # Regime transition indicators
                    vol_transition_short = vol_regime_short.diff() != 0
                    vol_transition_long = vol_regime_long.diff() != 0
                    
                    df[f"{feature}_vol_regime_short_{window}"] = vol_regime_short
                    df[f"{feature}_vol_regime_long_{window}"] = vol_regime_long
                    df[f"{feature}_vol_transition_{window}"] = (vol_transition_short | vol_transition_long).astype(int)
                    
                    # 3. Volatility clustering strength
                    # Measure how much current volatility predicts future volatility
                    vol_autocorr_1 = vol_medium.rolling(window=window, min_periods=max(1, window//2)).corr(vol_medium.shift(1))
                    vol_autocorr_3 = vol_medium.rolling(window=window, min_periods=max(1, window//2)).corr(vol_medium.shift(3))
                    
                    df[f"{feature}_vol_autocorr_1_{window}"] = vol_autocorr_1
                    df[f"{feature}_vol_autocorr_3_{window}"] = vol_autocorr_3
                    
                    # 4. Volatility surprise (unexpected volatility changes)
                    # Compare realized volatility to recent average
                    expected_vol = vol_medium.rolling(window=window, min_periods=max(1, window//2)).mean()
                    vol_surprise = (vol_medium - expected_vol) / (expected_vol + 1e-8)
                    df[f"{feature}_vol_surprise_{window}"] = vol_surprise
                    
                    # 5. Extreme volatility events
                    # Identify periods of unusually high or low volatility
                    vol_z_score = (vol_medium - vol_medium.rolling(window=window*2, min_periods=window).mean()) / (
                        vol_medium.rolling(window=window*2, min_periods=window).std() + 1e-8
                    )
                    
                    extreme_vol_high = (vol_z_score > 2).astype(int)  # More than 2 std above mean
                    extreme_vol_low = (vol_z_score < -2).astype(int)  # More than 2 std below mean
                    
                    df[f"{feature}_extreme_vol_high_{window}"] = extreme_vol_high
                    df[f"{feature}_extreme_vol_low_{window}"] = extreme_vol_low
        
        return df
    
    def add_cross_asset_regime_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Detect regimes based on relationships between different spread features.
        """
        print(f"Generating cross-asset regime features (windows: {self.windows})...")
        
        # Create pairs of base price features for cross-timeframe analysis
        # Match any currency pair pattern
        base_features = [f for f in features if '_Close' in f or '_Return_' in f or '_vs_MA' in f]
        
        for i, feature1 in enumerate(base_features):
            for feature2 in base_features[i+1:]:
                if feature1 in df.columns and feature2 in df.columns:
                    for window in self.windows:
                        # 1. Rolling correlation regime
                        rolling_corr = df[feature1].rolling(window=window, min_periods=max(1, window//2)).corr(df[feature2])
                        df[f"{feature1}_{feature2}_corr_{window}"] = rolling_corr
                        
                        # Correlation regime classification
                        corr_percentile = rolling_corr.rolling(window=window*2, min_periods=window).rank(pct=True)
                        corr_regime = np.where(corr_percentile > 0.75, 2,  # High correlation
                                             np.where(corr_percentile < 0.25, 0, 1))  # Low, Medium correlation
                        df[f"{feature1}_{feature2}_corr_regime_{window}"] = corr_regime
                        
                        # 2. Relative performance regime
                        # Which feature is outperforming?
                        norm1 = (df[feature1] - df[feature1].rolling(window=window*2, min_periods=window).mean()) / (
                            df[feature1].rolling(window=window*2, min_periods=window).std() + 1e-8
                        )
                        norm2 = (df[feature2] - df[feature2].rolling(window=window*2, min_periods=window).mean()) / (
                            df[feature2].rolling(window=window*2, min_periods=window).std() + 1e-8
                        )
                        
                        relative_performance = norm1 - norm2
                        df[f"{feature1}_{feature2}_rel_perf_{window}"] = relative_performance
                        
                        # Performance regime (which is leading)
                        perf_percentile = relative_performance.rolling(window=window*2, min_periods=window).rank(pct=True)
                        perf_regime = np.where(perf_percentile > 0.6, 1,  # Feature1 leading
                                             np.where(perf_percentile < 0.4, -1, 0))  # Feature2 leading, neutral
                        df[f"{feature1}_{feature2}_leader_regime_{window}"] = perf_regime
                        
                        # 3. Divergence and convergence patterns
                        # Are the features diverging or converging?
                        feature_diff = df[feature1] - df[feature2]
                        divergence_trend = feature_diff.rolling(window=window, min_periods=max(1, window//2)).apply(
                            lambda x: stats.linregress(np.arange(len(x)), x)[0] if len(x) >= 3 else np.nan
                        )
                        df[f"{feature1}_{feature2}_divergence_{window}"] = divergence_trend
                        
                        # 4. Volatility spillover regime
                        # Does high volatility in one feature predict high volatility in another?
                        vol1 = df[feature1].rolling(window=max(2, window//2), min_periods=2).std()
                        vol2 = df[feature2].rolling(window=max(2, window//2), min_periods=2).std()
                        
                        vol_spillover = vol1.shift(1).rolling(window=window, min_periods=max(1, window//2)).corr(vol2)
                        df[f"{feature1}_{feature2}_vol_spillover_{window}"] = vol_spillover
        
        return df
    
    
    def calculate_correlation_matrix(self, features: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """Calculate correlation matrix and identify highly correlated features."""
        corr_matrix = features.corr().abs()
        
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs (>{threshold}):")
            for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
                print(f"  {feat1} <-> {feat2}: {corr:.3f}")
            if len(high_corr_pairs) > 10:
                print(f"  ... and {len(high_corr_pairs) - 10} more pairs")
        else:
            print(f"\nNo highly correlated features found (threshold: {threshold})")
        
        return corr_matrix
    
    def remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """Remove highly correlated features with priority for regime features."""
        print(f"\n=== Removing Correlated Features (threshold: {threshold}) ===")
        
        corr_matrix = features.corr().abs()
        
        def get_feature_priority(feature_name):
            """Assign priority based on regime detection feature type."""
            priority = 0
            
            # Base price features get highest priority (any currency pair)
            if '_Close' in feature_name or '_Return_' in feature_name or '_LogReturn' in feature_name or '_vs_MA' in feature_name:
                priority += 100
            
            # Regime detection feature priorities
            if any(x in feature_name for x in ['_market_regime_', '_regime_type_', '_corr_regime_']):
                priority += 90  # Market regime classification
            elif any(x in feature_name for x in ['_cusum_', '_level_shift_', '_var_shift_']):
                priority += 85  # Structural breaks
            elif any(x in feature_name for x in ['_breakout_', '_regime_duration_']):
                priority += 80  # Breakout and persistence
            elif any(x in feature_name for x in ['_cycle_', '_peak_', '_trough_', '_seasonal_']):
                priority += 75  # Cycle detection
            elif any(x in feature_name for x in ['_sideways_', '_mean_reversion_', '_momentum_persistence_']):
                priority += 70  # Market state
            elif any(x in feature_name for x in ['_vol_regime_', '_vol_clustering_', '_vol_surprise_']):
                priority += 65  # Volatility regimes
            elif any(x in feature_name for x in ['_rel_perf_', '_divergence_', '_vol_spillover_']):
                priority += 60  # Cross-asset
            
            # Shorter timeframes get slight priority
            if '_3' in feature_name:
                priority += 3
            elif '_6' in feature_name:
                priority += 2
            elif '_12' in feature_name:
                priority += 1
            
            return priority
        
        # Group features into correlation clusters
        feature_priorities = {col: get_feature_priority(col) for col in features.columns}
        
        processed_features = set()
        features_to_remove = set()
        correlation_groups = []
        
        # Create correlation graph
        correlation_graph = {feat: [] for feat in features.columns}
        for i, feat1 in enumerate(features.columns):
            for j, feat2 in enumerate(features.columns):
                if i != j and corr_matrix.loc[feat1, feat2] > threshold:
                    correlation_graph[feat1].append(feat2)
        
        # Find connected components using DFS
        for feat in features.columns:
            if feat in processed_features:
                continue
            
            stack = [feat]
            cluster = []
            while stack:
                current = stack.pop()
                if current not in processed_features:
                    cluster.append(current)
                    processed_features.add(current)
                    for neighbor in correlation_graph[current]:
                        if neighbor not in processed_features:
                            stack.append(neighbor)
            
            if len(cluster) > 1:
                cluster.sort(key=lambda x: feature_priorities[x], reverse=True)
                keep_feature = cluster[0]
                remove_features = cluster[1:]
                
                max_corr = max(corr_matrix.loc[f1, f2] for f1 in cluster for f2 in cluster if f1 != f2)
                
                correlation_groups.append({
                    'group': cluster,
                    'kept': keep_feature,
                    'removed': remove_features,
                    'max_correlation': max_corr
                })
                
                features_to_remove.update(remove_features)
        
        # Print results
        print(f"Found {len(correlation_groups)} correlation groups:")
        for i, group in enumerate(correlation_groups[:8]):
            print(f"  Group {i+1}: Kept {group['kept']} (max corr: {group['max_correlation']:.3f})")
            print(f"    Removed: {', '.join(group['removed'][:2])}" + 
                  (f" + {len(group['removed'])-2} more" if len(group['removed']) > 2 else ""))
        
        if len(correlation_groups) > 8:
            print(f"  ... and {len(correlation_groups) - 8} more groups")
        
        selected_features = features.drop(columns=list(features_to_remove))
        
        print(f"\nFeature selection summary:")
        print(f"  Original features: {len(features.columns)}")
        print(f"  Correlation groups: {len(correlation_groups)}")
        print(f"  Removed features: {len(features_to_remove)}")
        print(f"  Final features: {len(selected_features.columns)}")
        
        return selected_features
    
    def run_regime_detection_pipeline(self, 
                                    fx_file_path: str = None,
                                    currency_pair: str = None,
                                    save_features: bool = True,
                                    output_file: str = None,
                                    correlation_threshold: float = 0.7) -> pd.DataFrame:
        """
        Run the complete regime detection feature engineering pipeline.
        
        Parameters:
        -----------
        fx_file_path : str, optional
            Path to the FX data CSV file. If None, defaults to "{currency_pair}.csv"
        currency_pair : str, optional
            Currency pair to process. If None, uses first pair in currency_pairs list
        save_features : bool
            Whether to save features to CSV
        output_file : str, optional
            Output filename. If None, auto-generates based on currency_pair
        correlation_threshold : float
            Threshold for removing correlated features
        """
        # Set defaults
        if currency_pair is None:
            currency_pair = self.currency_pairs[0]
        
        if fx_file_path is None:
            fx_file_path = f"{currency_pair}.csv"
        
        if output_file is None:
            output_file = f"regime_detection_features_{currency_pair}.csv"
        
        print(f"=== Regime Detection Feature Engineering Pipeline for {currency_pair} ===")
        
        # Step 1: Load and prepare data
        print(f"1. Loading {currency_pair} price data from {fx_file_path}...")
        fx_data = self.load_eurusd_data(fx_file_path, currency_pair)
        print(f"   Loaded {len(fx_data)} monthly observations")
        print(f"   Date range: {fx_data.index[0].strftime('%Y-%m')} to {fx_data.index[-1].strftime('%Y-%m')}")
        
        # Step 2: Generate basic price features
        print("2. Generating price features...")
        price_features = self.generate_price_features(fx_data, currency_pair)
        print(f"   Generated {len(price_features.columns)} basic price features")
        
        # Step 3: Generate all regime detection features
        print("3. Generating regime detection features...")
        all_base_features = list(price_features.columns)
        
        # Add all regime detection feature types
        price_features = self.add_structural_break_features(price_features, all_base_features)
        price_features = self.add_cycle_detection_features(price_features, all_base_features)
        price_features = self.add_market_state_features(price_features, all_base_features)
        price_features = self.add_volatility_regime_features(price_features, all_base_features)
        price_features = self.add_cross_asset_regime_features(price_features, all_base_features)

        print(f"  Generating Labels one month ahead")
        close_col = f'{currency_pair}_Close'
        price = price_features[close_col]
        price_returns = (price.shift(-1) > price).astype(int)  # 1 if next month's price is higher, else 0
        price_features['Label'] = price_returns

        print(f"   Generated {price_features.shape[1]} total regime features")
        print(f"   Features available for {len(price_features)} time periods")
        
        # Step 4: Clean features
        #print("4. Cleaning features...")
        price_features = price_features.replace([np.inf, -np.inf], np.nan)
        
        # Remove features with too many NaN values
        nan_threshold = len(price_features) * 0.15  # Keep features with at least 85% valid data
        valid_features = price_features.dropna(thresh=nan_threshold, axis=1)
        
        print(f"   Removed {price_features.shape[1] - valid_features.shape[1]} features with >85% NaN values")
        print(f"   Final feature count before correlation removal: {valid_features.shape[1]}")
        
        # Check recent data quality
        recent_nan_count = valid_features.tail(12).isnull().sum().sum()
        total_recent_values = valid_features.tail(12).shape[0] * valid_features.tail(12).shape[1]
        recent_nan_pct = (recent_nan_count / total_recent_values) * 100 if total_recent_values > 0 else 0
        print(f"   Recent 12 months NaN percentage: {recent_nan_pct:.1f}% ({recent_nan_count}/{total_recent_values})")
        
        # Step 5: Remove highly correlated features
        #print("5. Removing highly correlated features...")
        valid_features = self.remove_correlated_features(valid_features, threshold=correlation_threshold)
        
        # Step 6: Final correlation analysis
        print("6. Analyzing final feature correlations...")
        final_correlation_matrix = self.calculate_correlation_matrix(valid_features, correlation_threshold)
        
        # Step 7: Save features
        if save_features:
            print(f"7. Saving features to {output_file}...")
            close_col = f'{currency_pair}_Close'
            if close_col in valid_features.columns:
                valid_features = valid_features.drop(close_col, axis=1)
            valid_features.to_csv(output_file)
            
            corr_file = output_file.replace('.csv', '_correlations.csv')
            final_correlation_matrix.to_csv(corr_file)
            print(f"   Features saved to {output_file}")
            print(f"   Correlations saved to {corr_file}")
        
        print(f"\n=== Regime Detection Feature Engineering Complete for {currency_pair} ===")
        print(f"Final dataset shape: {valid_features.shape}")
        
        return valid_features
    
    def run_regime_detection_all_pairs(self,
                                      save_features: bool = True,
                                      correlation_threshold: float = 0.7) -> Dict[str, pd.DataFrame]:
        """
        Run regime detection feature engineering for all configured currency pairs.
        
        Parameters:
        -----------
        save_features : bool
            Whether to save features to CSV files
        correlation_threshold : float
            Threshold for removing correlated features
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping currency pair to its features DataFrame
        """
        print("=" * 80)
        print(f"=== Regime Detection Feature Engineering - All Currency Pairs ===")
        print(f"Processing {len(self.currency_pairs)} currency pair(s): {', '.join(self.currency_pairs)}")
        print("=" * 80)
        
        all_pair_features = {}
        
        for pair in self.currency_pairs:
            print(f"\n{'=' * 80}")
            try:
                features = self.run_regime_detection_pipeline(
                    fx_file_path=f"{pair}.csv",
                    currency_pair=pair,
                    save_features=save_features,
                    output_file=f"regime_detection_features_{pair}.csv",
                    correlation_threshold=correlation_threshold
                )
                all_pair_features[pair] = features
                print(f"✅ Successfully processed {pair}")
            except Exception as e:
                print(f"❌ Error processing {pair}: {e}")
                continue
        
        print(f"\n{'=' * 80}")
        print(f"=== Processing Complete ===")
        print(f"Successfully processed {len(all_pair_features)}/{len(self.currency_pairs)} currency pairs")
        print("=" * 80)
        
        return all_pair_features


def main():
    """Main function to run the regime detection feature engineering pipeline."""
    
    # Initialize regime detection feature engineering class with multiple currency pairs
    feature_engineer = RegimeDetectionFeatureEngineering(
        windows=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],  # 6, 12, and 24 month windows
        currency_pairs=["EURUSD", "USDJPY", "EURJPY", "AUDUSD", "XAUUSD", "GBPUSD"]  # Multiple currency pairs
    )
    
    # Run the complete pipeline for all currency pairs
    try:
        all_features = feature_engineer.run_regime_detection_all_pairs(
            save_features=True,
            correlation_threshold=1.0
        )

        # Display summary for each pair
        for pair, features in all_features.items():
            print(f"\n=== Summary for {pair} ===")
            print("\n=== Summary Statistics ===")
            print(features.describe())
            
            print(f"\n=== Sample Features (Last 5 rows) ===")
            print(features.tail())
            
            print(f"\n=== Feature Categories ===")
            feature_categories = {
                'Base Price Features': [col for col in features.columns if '_Close' in col or '_Return_' in col or '_vs_MA' in col],
                'Structural Breaks': [col for col in features.columns if any(x in col for x in ['cusum', 'level_shift', 'var_shift', 'regime_type', 'regime_duration', 'breakout'])],
                'Cycle Detection': [col for col in features.columns if any(x in col for x in ['peak', 'trough', 'cycle_length', 'cycle_position', 'seasonal', 'quarterly', 'dominant_period'])],
                'Market States': [col for col in features.columns if any(x in col for x in ['trend_slope', 'trend_r2', 'sideways', 'mean_reversion', 'momentum_persistence', 'market_regime'])],
                'Volatility Regimes': [col for col in features.columns if any(x in col for x in ['vol_ratio', 'vol_regime', 'vol_transition', 'vol_autocorr', 'vol_surprise', 'extreme_vol'])],
                'Cross-Timeframe': [col for col in features.columns if any(x in col for x in ['_corr_', '_rel_perf_', '_leader_regime_', '_divergence_', '_vol_spillover_'])]
            }
            
            for category, feature_list in feature_categories.items():
                print(f"  {category}: {len(feature_list)} features")
            
            print(f"\n=== Feature List ===")
            for i, col in enumerate(features.columns, 1):
                print(f"{i:2d}. {col}")
            
            # Final validation
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
        print(f"Error in regime detection pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
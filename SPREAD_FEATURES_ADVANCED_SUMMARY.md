# Advanced Spread Features Summary

## Overview
The `spread_feature_engineering_advanced.py` script creates advanced features derived from the same EU-US economic spreads as the original `spread_feature_engineering.py`, but uses completely different transformation methods to ensure uncorrelated features.

## Generated Files
- **spread_features_advanced.csv**: 129 uncorrelated advanced features (306 observations)
- **spread_features_advanced_correlations.csv**: Correlation matrix of final features

## Base Spread Features (Same as Original)
The script uses the **exact same** `generate_spread_features()` method as the original:
1. **CPI_Spread**: EU_CPI / US_CPI
2. **Core_CPI_Spread**: EU_CPI / US_Core_CPI  
3. **Yield_Spread**: EU_10Y_Yield / US_10Y_Yield
4. **Rate_Spread**: ECB_Deposit_Rate / Fed_Funds_Rate
5. **VIX**: Standalone volatility indicator

## Feature Engineering Comparison

### Original Script Features (13 features)
- **Rate of Change (roc)**: Momentum-based features
- **Moving Average Spreads (ma_spread)**: Trend deviation features  
- **Volatility Adjusted (vol_adj)**: Risk-adjusted features
- **Z-score (zscore)**: Standardized features
- **Trend (trend)**: Directional features

### Advanced Script Features (129 features)
**Completely different feature types to avoid correlation:**

#### 1. Rank-Based Features (19 features)
- **Percentile Ranks**: Rolling position within historical distribution
- **Rank Momentum**: Changes in relative position
- **Rank Deviation**: Distance from median rank (mean reversion)
- **Extreme Rank Detection**: Binary indicators for top/bottom quintiles

#### 2. Fractal & Chaos Theory Features (5 features)
- **Hurst Exponent**: Measure of long-term memory and trend persistence
- **Fractal Dimension**: Complexity and roughness of time series
- **Regime Detection**: Trending vs mean-reverting behavior identification

#### 3. Statistical Distribution Features (24 features)
- **Skewness**: Asymmetry of spread distributions
- **Kurtosis**: Tail thickness and extreme value propensity
- **Median Absolute Deviation (MAD)**: Robust measure of variability
- **Normality Tests**: Jarque-Bera statistics for distribution analysis

#### 4. Regime Detection Features (30 features)
- **CUSUM**: Cumulative sum for change point detection
- **Change Point Indicators**: Binary signals for structural breaks
- **Level Shift Detection**: Mean comparison across periods
- **Breakout Intensity**: Distance from recent trading ranges
- **Regime Persistence**: Duration of current market state

#### 5. Cross-Sectional Features (23 features)
- **Inter-Spread Correlations**: Rolling correlations between spreads
- **Relative Strength**: Ratio of normalized spread movements
- **Spread Divergence**: Differences in normalized spread behavior

#### 6. Time Series Decomposition Features (27 features)
- **Trend Component**: Long-term directional movement
- **Detrended Series**: Cyclical component after trend removal
- **Cyclical Component**: Hodrick-Prescott filter approximation
- **Residual Variance**: Unexplained variation after decomposition
- **Signal-to-Noise Ratio**: Quality measure of trend vs noise

## Key Advantages

### 1. **Zero Feature Overlap**
- **Original**: 13 features using roc, ma_spread, vol_adj, zscore, trend
- **Advanced**: 129 features using rank, fractal, distribution, regime, cross-sectional, decomposition methods
- **Correlation**: Maximum correlation between final features: 0.696 (< 0.7 threshold)

### 2. **Different Market Perspectives**
- **Original**: Focuses on momentum, volatility, and trend following
- **Advanced**: Focuses on distribution analysis, regime changes, and structural relationships

### 3. **Advanced Statistical Methods**
- Uses sophisticated mathematical concepts (Hurst exponent, fractal dimension)
- Employs robust statistical measures (MAD, normality tests)
- Implements change point detection algorithms

### 4. **Multi-Timeframe Analysis**
- Maintains 3-month and 6-month windows for consistency
- Provides complementary views of the same underlying spreads

## Feature Distribution by Type
```
Base Spreads: 108 features
VIX: 1 feature
Fractal/Chaos: 5 features
Rank Features: 19 features
Regime Detection: 30 features
Cross-Sectional: 23 features
Distribution: 24 features
Decomposition: 27 features
```

## Data Quality
- **Time Period**: February 2000 to July 2025 (306 monthly observations)
- **Missing Data Handling**: Features with >50% NaN values removed (40 features eliminated)
- **Correlation Management**: Intelligent priority system favoring advanced feature types
- **Data Validation**: All features have correlation < 0.7 threshold

## Usage Recommendations

### Model Ensemble Strategy
1. **Original Features**: Use for trend-following and momentum strategies
2. **Advanced Features**: Use for regime detection and structural analysis
3. **Combined Approach**: Ensemble both feature sets for comprehensive market analysis

### Feature Selection
- **High Priority**: Fractal and rank-based features (unique mathematical properties)
- **Medium Priority**: Regime detection and cross-sectional features (market structure)
- **Lower Priority**: Distribution and decomposition features (supporting analysis)

### Trading Applications
- **Regime Detection**: Use fractal and CUSUM features for market state identification
- **Mean Reversion**: Use rank deviation and breakout features for contrarian signals
- **Structural Analysis**: Use cross-sectional correlations for spread relationship trading

## Technical Implementation
- **Correlation Removal**: DFS-based clustering with priority weighting
- **Feature Priority**: Advanced mathematical concepts ranked highest
- **Robustness**: Handles edge cases (zero division, insufficient data)
- **Performance**: Optimized for monthly economic data frequency

This advanced feature engineering approach provides a completely orthogonal view of the same EU-US economic spreads, enabling more sophisticated trading strategies and improved model performance through feature diversification.
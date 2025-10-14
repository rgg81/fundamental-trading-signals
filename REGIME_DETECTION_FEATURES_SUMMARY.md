# Regime Detection Features Summary

## Overview
The `regime_detection_features.py` script generates **300 sophisticated features** from EURUSD price data, specifically designed to detect market regimes, cycles, structural breaks, and trading patterns. This is a pure price action approach focused on identifying different market states in EUR/USD currency movements.

## Generated Features by Category

### 1. Base Price Features (109 features)
**Core Price Data**:
- `EURUSD_Close`: Monthly EUR/USD closing price
- `EURUSD_Return_1M/3M/6M`: Multi-period return analysis
- `EURUSD_LogReturn`: Logarithmic returns for continuous compounding
- `EURUSD_vs_MA12/24/36`: Price relative to moving averages (trend indicators)

**Enhanced Price Features**: All base features generate regime detection variants including breakout analysis, regime duration, variance shifts, and CUSUM change point detection.

### 2. Structural Break Features (83 features)
**Purpose**: Detect regime changes and structural breaks in the economic relationships

**Creative Features**:
- **CUSUM Indicators**: Cumulative sum analysis for change point detection
  - `_cusum_pos_` / `_cusum_neg_`: Positive/negative cumulative deviations
  - `_cusum_intensity_`: Strength of change point signals

- **Level Shift Detection**: Mean change analysis
  - `_level_shift_`: Change in mean levels between periods
  - `_level_acceleration_`: Rate of change in level shifts (second derivative)

- **Variance Shifts**: Volatility regime changes
  - `_var_shift_`: Log changes in variance between periods

- **Regime Classification**: State identification
  - `_regime_type_`: High/Normal/Low regime classification based on z-scores
  - `_regime_duration_`: How long in current regime (persistence measure)

- **Breakout Detection**: Range breakout analysis
  - `_breakout_up_` / `_breakout_down_`: Upward/downward breakouts from recent range
  - `_breakout_intensity_`: Combined breakout strength

### 3. Cycle Detection Features (39 features)
**Purpose**: Identify cyclical patterns, seasonality, and periodic behavior

**Creative Features**:
- **Peak/Trough Analysis**: Cycle turning points
  - `_peak_indicator_` / `_trough_indicator_`: Local maxima/minima detection
  - `_cycle_length_`: Average time between peaks
  - `_cycle_position_`: Current position within the cycle (0-1 scale)

- **Seasonal Patterns**: Periodic correlation analysis
  - `_seasonal_corr_`: 12-month seasonal correlation
  - `_quarterly_corr_`: 3-month quarterly correlation

- **Dominant Frequency**: Periodogram analysis
  - `_dominant_period_`: Most significant cycle length in recent data

### 4. Market State Features (44 features)
**Purpose**: Identify trending, sideways, mean-reverting, and momentum regimes

**Creative Features**:
- **Trend Analysis**: Linear regression based
  - `_trend_slope_`: Rate of trend (regression slope)
  - `_trend_r2_`: Trend strength (R-squared)

- **Sideways Market Detection**: Range-bound behavior
  - `_sideways_score_`: High volatility + low trend strength indicator

- **Mean Reversion Analysis**: Return-to-mean tendency
  - `_mean_reversion_`: Negative correlation between z-score and future returns
  - `_vol_clustering_`: Volatility persistence (GARCH-like)

- **Momentum Analysis**: Directional persistence
  - `_momentum_persistence_`: Continuation of directional moves

### 5. Volatility Regime Features (88 features)
**Purpose**: Sophisticated volatility clustering and regime analysis

**Creative Features**:
- **Multi-Scale Volatility**: Different timeframe analysis
  - `_vol_ratio_sm_`: Short/medium volatility ratio
  - `_vol_ratio_ml_`: Medium/long volatility ratio

- **Volatility Regimes**: State classification
  - `_vol_regime_short_` / `_vol_regime_long_`: High/low volatility regimes
  - `_vol_transition_`: Volatility regime change indicator

- **Volatility Clustering**: Persistence analysis
  - `_vol_autocorr_1_` / `_vol_autocorr_3_`: 1-period and 3-period volatility autocorrelation

- **Volatility Surprises**: Unexpected changes
  - `_vol_surprise_`: Realized vs expected volatility deviation

- **Extreme Volatility**: Outlier detection
  - `_extreme_vol_high_` / `_extreme_vol_low_`: Unusually high/low volatility periods

### 6. Cross-Timeframe Regime Features (38 features)
**Purpose**: Relationships between different timeframe price features

**Creative Features**:
- **Multi-Timeframe Correlation**: Dynamic correlation analysis
  - `_corr_`: Rolling correlation between different return periods
  - `_corr_regime_`: High/medium/low correlation classification

- **Relative Performance**: Which timeframe is leading
  - `_rel_perf_`: Relative z-score performance between timeframes
  - `_leader_regime_`: Which return period is outperforming

- **Price-Return Relationships**: Price level vs momentum interactions
  - `_divergence_`: Trend differences between price and returns
  - `_vol_spillover_`: Volatility contagion between timeframes

## Key Creative Elements

### 1. Multi-Timeframe Analysis
- Uses 3 and 6-month windows for focused regime detection
- Captures short-term vs medium-term regime changes
- Multiple return periods (1M, 3M, 6M) for comprehensive momentum analysis

### 2. Advanced Statistical Methods
- **CUSUM Analysis**: Classic change point detection
- **Linear Regression**: Trend strength and direction
- **Autocorrelation**: Pattern persistence analysis
- **Peak/Trough Detection**: Cycle identification using scipy.signal
- **K-means Clustering**: Market regime classification

### 3. Trading-Focused Features
- **Breakout Detection**: Identifies potential trend changes
- **Mean Reversion Signals**: Contrarian opportunities
- **Momentum Persistence**: Trend continuation signals
- **Sideways Market Detection**: Range-bound trading opportunities
- **Volatility Clustering**: Risk regime identification

### 4. Cross-Timeframe Intelligence
- **Correlation Regimes**: When different timeframes move together vs independently
- **Leadership Analysis**: Which timeframe is driving price moves
- **Spillover Effects**: How volatility/trends propagate between timeframes
- **Price-Momentum Dynamics**: Interaction between price levels and return momentum

## Data Quality
- **Final Dataset**: 300 features from 268 time periods (2003-05 to 2025-08)
- **Recent Data Quality**: 0.0% NaN values in last 12 months (perfect recent data)
- **Correlation Management**: All features have correlation < 0.7
- **Data Source**: Pure EURUSD monthly price data with no missing values

## Use Cases for Trading
1. **Regime Detection**: Identify when market conditions change
2. **Cycle Trading**: Time entries based on cyclical patterns
3. **Breakout Trading**: Detect when spreads break from ranges
4. **Mean Reversion**: Identify overbought/oversold conditions
5. **Volatility Trading**: Trade volatility regime changes
6. **Pairs Trading**: Identify relative performance opportunities
7. **Risk Management**: Detect high-risk volatility regimes

## File Outputs
- `regime_detection_features.csv`: Main feature dataset
- `regime_detection_features_correlations.csv`: Correlation matrix for analysis

This regime detection approach provides a comprehensive framework for understanding market microstructure and identifying trading opportunities across different market conditions and timeframes.
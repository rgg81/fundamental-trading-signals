# Feature Engineering Comparison: Trend Following vs Mean Reversion

## Overview

This document compares two feature engineering scripts designed for different market conditions:

1. **Technical Indicators Feature Engineering** - Optimized for trending markets
2. **Mean Reversion Feature Engineering** - Optimized for sideways/ranging markets

## EUR/USD 2023 Market Analysis

Based on the provided EUR/USD monthly chart for 2023, the market exhibited classic sideways behavior:
- Range-bound movement between ~1.055 and ~1.113
- Multiple reversals at key levels
- Lack of sustained directional trends
- High frequency of mean reversion behaviors

This type of market is ideal for mean reversion strategies rather than trend-following approaches.

## Feature Engineering Approaches

### Technical Indicators (Trend Following)
**File:** `technical_indicators_feature_engineering.py`
**Final Features:** 4 uncorrelated features
**Market Condition:** Trending markets

**Key Features:**
- ADX_6 (trend strength)
- ATR_6 (volatility)
- BB_Width_24 (volatility expansion)
- StochRSI_6 (momentum oscillator)

**Focus Areas:**
- Trend detection and strength
- Momentum confirmation
- Volatility breakouts
- Directional signals

### Mean Reversion (Sideways Markets)
**File:** `mean_reversion_feature_engineering.py`
**Final Features:** 65 uncorrelated features
**Market Condition:** Ranging/sideways markets

**Key Feature Categories:**

#### 1. Mean Reversion Signals (9 features)
- Price Z-Score relative to moving averages
- RSI extremes and divergences
- Williams %R extreme conditions
- Bollinger Band position indicators

#### 2. Support/Resistance Levels (14 features)
- Distance to rolling highs/lows
- Price near support/resistance zones
- Donchian channel positions
- Reversal signals from extremes
- Range size and tight range detection

#### 3. Volatility Contraction (11 features)
- Bollinger Band squeeze conditions
- ATR compression signals
- Range contraction indicators
- Price stability measures
- Low volatility environments

#### 4. Momentum Exhaustion (20 features)
- MACD signal line crossovers
- Rate of Change deceleration
- CCI extreme readings and reversals
- Awesome Oscillator zero line crosses
- Momentum exhaustion signals

#### 5. Custom Mean Reversion (16 features)
- Time spent in upper/lower range halves
- Trend exhaustion detection
- Return to mean signals
- False breakout identification
- Bullish/bearish divergences
- Channel trading signals

## When to Use Each Approach

### Use Technical Indicators (Trend Following) When:
- Market shows sustained directional movement
- Clear trend establishment with momentum
- Breakout patterns are common
- Volatility expansion indicates new trends
- Examples: Strong bull/bear markets, breakout scenarios

### Use Mean Reversion Features When:
- Market is range-bound or sideways
- Frequent reversals at support/resistance
- Low trending behavior
- High mean reversion characteristics
- Examples: Consolidation periods, sideways markets like EUR/USD 2023

## Implementation Notes

### Technical Indicators Script
- Fewer but highly selective features
- Focus on trend confirmation
- Suitable for lower-frequency trading
- Better for capturing large moves

### Mean Reversion Script
- Comprehensive feature set for sideways markets
- Multiple signal confirmation approaches
- Suitable for higher-frequency mean reversion
- Better for capturing range-bound profits

## Feature Selection Philosophy

Both scripts use intelligent correlation removal with priority given to:
1. Shorter timeframes (6-month > 12-month > 24-month)
2. Key indicator types for each market condition
3. Maximum of 0.7 correlation threshold
4. Representative features from each correlation cluster

## Usage Recommendations

For EUR/USD analysis (based on 2023 behavior):
1. **Primary:** Use mean reversion features for main strategy
2. **Secondary:** Use technical indicators for trend confirmation
3. **Combined:** Create ensemble models using both feature sets
4. **Market Regime:** Switch between approaches based on market conditions

## Output Files

### Technical Indicators
- `technical_indicators_features.csv` - 4 features
- `technical_indicators_features_correlations.csv` - Correlation matrix

### Mean Reversion
- `mean_reversion_features.csv` - 65 features  
- `mean_reversion_features_correlations.csv` - Correlation matrix

Both feature sets are ready for machine learning model training and backtesting.
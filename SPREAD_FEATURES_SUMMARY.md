# Spread Feature Engineering Summary

## Overview

The **Spread Feature Engineering** script creates EU-US economic spread indicators specifically designed for EUR/USD trading. It focuses on the fundamental differences between European and US economic conditions.

## Key Spreads Generated

### 1. **Core Economic Spreads**
- **CPI_Spread**: EU_CPI - US_CPI (inflation differential)
- **Core_CPI_Spread**: EU_CPI - US_Core_CPI (inflation vs core inflation)
- **Yield_Spread**: EU_10Y_Yield - US_10Y_Yield (yield curve differential)
- **Rate_Spread**: ECB_Deposit_Rate - Fed_Funds_Rate (monetary policy differential)

### 2. **Risk Indicator**
- **VIX**: Market volatility indicator (standalone, no EU equivalent)

## Feature Categories (38 Total Features)

### **Rate of Change Features (6 features)**
- 1-3 period rate of change for yield and rate spreads
- Captures momentum in policy divergence

### **Moving Average Features (1 feature)**
- 3-month moving average for core CPI spread
- Smoothed inflation differential trend

### **Volatility Features (15 features)**
- Rolling standard deviation (3 & 6 month windows)
- Volatility-adjusted spreads (spread / volatility)
- Measures uncertainty and stability of spreads

### **Momentum & Trend Features (11 features)**
- Z-score normalization (spread relative to historical mean/std)
- Trend direction indicators (binary up/down)
- Acceleration (second derivative of spreads)

### **VIX Features (6 features)**
- Volatility measures and normalized indicators
- Risk-adjusted metrics for market stress

## Time Windows Used

- **Short-term**: 3 months (more responsive to recent changes)
- **Medium-term**: 6 months (captures sustained trends)

## Correlation Removal Results

- **Original features**: 105
- **After correlation removal**: 38 features
- **Removed**: 67 highly correlated features (>0.7 threshold)
- **Maximum remaining correlation**: 0.667

## Priority System for Feature Selection

1. **Core Spreads** (highest priority): CPI, Yield, Rate spreads
2. **VIX Features** (high priority): Unique risk indicators
3. **Shorter Timeframes** (3-month > 6-month)
4. **Normalized Features** (Z-scores, volatility-adjusted)
5. **Trend/Momentum Features**
6. **Basic Volatility/MA Features**

## Why These Features Matter for EUR/USD

### **Economic Fundamentals**
- **CPI Spreads**: Higher EU inflation relative to US suggests EUR strength
- **Yield Spreads**: Higher EU yields attract capital flows to EUR
- **Rate Spreads**: ECB vs Fed policy divergence drives currency flows

### **Market Dynamics**
- **VIX**: High volatility often favors USD as safe haven
- **Volatility-Adjusted Spreads**: Normalize for market stress conditions
- **Momentum Features**: Capture acceleration/deceleration in divergence

### **Trading Applications**
- **Mean Reversion**: When spreads deviate from historical norms
- **Trend Following**: When policy divergence accelerates
- **Risk Management**: VIX features for position sizing

## Output Files

- **spread_features.csv**: 38 uncorrelated features (306 monthly observations)
- **spread_features_correlations.csv**: Correlation matrix for analysis

## Data Range
- **Period**: February 2000 to July 2025 (306 months)
- **Frequency**: Monthly
- **Missing Data**: Minimal (robust feature engineering with NaN handling)

## Usage in Trading Models

These features are ideal for:
1. **Fundamental Analysis Models**: Capture economic divergence
2. **Mean Reversion Strategies**: Spread normalization features
3. **Trend Following**: Momentum and acceleration features
4. **Risk Adjustment**: VIX-based position sizing
5. **Multi-Asset Models**: Economic regime identification
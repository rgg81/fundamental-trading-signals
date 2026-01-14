# Fundamental Trading Signals - AI Agent Guide

## Project Overview
A quantitative trading system generating buy/sell signals for FX pairs and gold (XAUUSD) using macroeconomic fundamentals and technical indicators. Employs ensemble ML strategies with Optuna hyperparameter optimization.

## Architecture

### Data Pipeline
- **Sources**: FX price data (EURUSD, USDJPY, etc.) + macroeconomic indicators from FRED
- **Structure**: Monthly OHLC data in `{PAIR}.csv` (e.g., `EURUSD.csv`) with columns: `Date`, `{PAIR}_Open`, `{PAIR}_High`, `{PAIR}_Low`, `{PAIR}_Close`
- **Integration**: `src/data_fetch/data_loader.py` merges FX and macro data

### Feature Engineering (`src/features/`)
Four specialized feature classes, each supporting **multi-currency pipelines**:

1. **Spread Features** (`spread_feature_engineering.py`): EU-US macro spreads (CPI, yields, rates)
2. **Technical Indicators** (`technical_indicators_feature_engineering.py`): Trend, momentum, volatility oscillators
3. **Mean Reversion** (`mean_reversion_feature_engineering.py`): Bollinger Bands, support/resistance, momentum exhaustion
4. **Regime Detection** (`regime_detection_features.py`): Market state, structural breaks, cycle detection

**Critical Convention**: All classes have `run_feature_engineering_all_pairs()` method processing multiple currency pairs in batch. Individual pair processing via `run_feature_engineering()` or `run_regime_detection_pipeline()`.

**Output Format**: `{feature_type}_features_{PAIR}.csv` and `{feature_type}_features_{PAIR}_correlations.csv`

### Strategy Pattern (`src/signals/`)
- **Base**: `Strategy` abstract class requires `generate_signal(past_data, current_data) -> (signals, amounts)`
- **Implementation**: Each strategy inherits from `Strategy`, implements Optuna hyperparameter tuning in `fit()` method
- **Naming**: `{Model}OptunaStrategy` (e.g., `LGBMOptunaStrategy`, `RandomForestOptunaStrategy`)
- **Return Values**: 
  - `signals`: List of 1 (buy) or 0 (sell)
  - `amounts`: List of trade sizes (1 to `max_amount`, typically 10)

### Ensemble Architecture (`ensemble_strategy.py`)
**Group-based weighted voting** across feature sets:
```python
# Groups strategies by (feature_set, feature_frequency)
'LGBM_regime_3_0': LGBMOptunaStrategy(feature_set="regime_", feature_frequency="_3")
'LGBM_tech_3_0': LGBMOptunaStrategy(feature_set="tech_", feature_frequency="_3")
'LGBM_mr_3_0': LGBMOptunaStrategy(feature_set="mr_", feature_frequency="_3")
```
- **Aggregation**: Weighted voting by amounts within each group
- **Tracking**: `accumulated_returns` dict tracks performance per (feature_set, frequency) group

### Backtesting (`src/signals/backtest.py`)
- **Execution**: Walks forward monthly with `step_size=6` (6-month steps)
- **Stop-loss**: Default 1.5% (`stop_loss=0.015`)
- **Returns**: Percentage-based with stop-loss capping: `max(min(profit_loss, stop_loss), -stop_loss)`
- **Metrics**: Uses `quantstats` for cumulative returns, Sharpe ratio, drawdown

## Critical Data Leakage Prevention

**Temporal Integrity Rules** (see `DATA_LEAKAGE_FIXES_COMPREHENSIVE.md`):
1. **Peak/Trough Detection**: Rolling window using only historical data (`df[i-lookback:i+1]`)
2. **K-Means Clustering**: Train on `historical_data.iloc[:i]`, predict `current_features.iloc[i:i+1]`
3. **Cycle Analysis**: Use `peak_positions[peak_positions < i]` (exclude current)
4. **Labels**: Always `Label = (price.shift(-1) > price).astype(int)` (next month prediction)

**Never** use `.fit_predict()` or full-series operations when generating features.

## Feature Selection Optimization

### Two-Step Bitmap Approach (see `TWO_STEP_FEATURE_SELECTION.md`, `BITMAP_FEATURE_OPTIMIZATION.md`)
```python
# Step 1: SelectKBest reduces 200+ features → ~50
k_features = trial.suggest_int('k_features', 3, 50)
selector = SelectKBest(score_func=f_classif, k=k_features)
kbest_features = X.columns[selector.get_support()].tolist()

# Step 2: Bitmap explores combinations within reduced set
feature_bitmap = trial.suggest_int('feature_bitmap', 1, 2**len(kbest_features)-1, log=True)
selected_features = [kbest_features[i] for i in range(len(kbest_features)) if feature_bitmap & (1 << i)]
```
**Why**: Avoids 2^200 search space overflow while maintaining intelligent feature combination discovery.

## Development Workflows

### Running Feature Engineering
```bash
cd src/features
# Single currency
python technical_indicators_feature_engineering.py

# Multi-currency (edit main() to specify pairs)
# Example: currency_pairs=['EURUSD', 'USDJPY', 'GBPUSD', 'XAUUSD']
```

### Running Backtests
```bash
cd src/signals
# Ensemble backtest with multiple strategies
python run_ensemble_backtest.py

# Individual strategy
python lgbm_strategy.py  # Most strategies have __main__ blocks
```

### Testing
```bash
# Run all tests
python -m unittest discover -s tests

# Specific test
python -m unittest tests.test_backtest
```

### Common Data Cleaning Pattern
```python
def _clean_data(self, df):
    """Remove NaN and infinity values - standard across all strategies"""
    df_clean = df.copy()
    df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
    df_clean = df_clean.dropna()
    return df_clean
```

## File Conventions

- **Checkpoints**: Model states saved to `checkpoints/`
- **Logs**: Lightning logs in `lightning_logs/`, CatBoost in `catboost_info/`
- **Results**: Random strategy experiments as `random_strategy_results_{N}.csv`
- **Documentation**: Feature analysis in `{FEATURE_TYPE}_MULTI_CURRENCY.md` files

## Key Dependencies

- **ML**: `lightgbm`, `xgboost`, `catboost`, `ngboost`, `pytorch-tabnet`, `scikit-learn`
- **Optimization**: `optuna` (TPE sampler for hyperparameter tuning)
- **Backtesting**: `quantstats` (performance metrics)
- **Data**: `pandas==2.2.3`, `numpy==1.26.4`

## Common Pitfalls

1. **Column Naming**: Always use `{CURRENCY_PAIR}_Close` format (e.g., `EURUSD_Close`, not `Close`)
2. **Feature Frequency**: When using feature subsets, match `feature_frequency` (e.g., `_3`, `_6`) to data generation
3. **Optuna Trials**: Keep `n_trials` low (3-10) for fast iteration; production uses 50-100
4. **NaN Handling**: Feature engineering creates NaNs from rolling windows - always `.dropna()` or use `min_periods` parameter
5. **Date Filtering**: Backtest starts from `start_year=2016` by default, skips first `min_history=154` rows

## When Adding New Strategies

1. Inherit from `Strategy` base class
2. Implement `fit(X, y)` with Optuna optimization
3. Implement `generate_signal(past_data, current_data)` returning `(signals, amounts)`
4. Add `_clean_data()` method for consistency
5. Use `TimeSeriesSplit` for cross-validation (not random splits)
6. Add to `ensemble_strategy.py` if appropriate

## Debugging Features

Print flags in strategies:
- `features_optimization=True`: Verbose output with `***` markers
- `features_optimization=False`: Standard output with `---` markers

Use correlation CSVs to verify feature independence before training.

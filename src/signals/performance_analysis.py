import pandas as pd
import numpy as np
import quantstats as qs
import matplotlib.pyplot as plt
from adaboost_strategy import AdaBoostOptunaStrategy
from knn_strategy import KNNOptunaStrategy
from lgbm_strategy import LGBMOptunaStrategy
from backtest import Backtest
from logistic_strategy import LogisticRegressionOptunaStrategy
from mlp_strategy import MLPOptunaStrategy
from nb_strategy import GaussianNBOptunaStrategy
from rf_strategy import RandomForestOptunaStrategy
from svc_strategy import SVCOptunaStrategy
from histgb_strategy import HistGBOptunaStrategy
from xgboost_strategy import XGBoostOptunaStrategy
from catboost_strategy import CatBoostOptunaStrategy
from tff_strategy import TemporalFusionTransformerOptunaStrategy
from tabnet_strategy import TabNetOptunaStrategy
from ngboost_strategy import NGBoostOptunaStrategy
from gp_strategy import GaussianProcessOptunaStrategy
from pytorch_nn_strategy import PyTorchNeuralNetOptunaStrategy
from ensemble_strategy import EnsembleOptunaStrategy

# Set the output format for QuantStats
qs.extend_pandas()

def analyze_strategy_performance(strategy_results, benchmark_data=None, strategy_name="Trading Strategy", symbol=None):
    """
    Analyze the performance of a trading strategy using QuantStats.
    
    Parameters:
    -----------
    strategy_results : pd.DataFrame
        DataFrame with columns ['Date', 'Signal', 'Amount', 'Return']
    benchmark_data : pd.Series, optional
        Benchmark returns series with DatetimeIndex
    strategy_name : str, optional
        Name of the strategy for display purposes
    
    Returns:
    --------
    None (displays performance metrics and plots)
    """
    # Convert strategy returns to a pandas Series with DatetimeIndex
    result_array_series = strategy_results.set_index("Date")["Return"]
    
    # Calculate cumulative returns
    cum_returns = qs.stats.compsum(result_array_series)
    
    # Generate full report
    output = 'stats.html' if symbol is None else f'stats_{symbol}.html'
    qs.reports.html(result_array_series, output=output, title=f'{strategy_name} Performance Report')
    
    # Print some key metrics including cumulative returns
    print(f"\n----- {strategy_name} Performance Metrics -----")
    print(f"Cumulative Return: {cum_returns[-1]:.2%}")
    print(f"CAGR: {qs.stats.cagr(result_array_series):.2%}")
    print(f"Sharpe Ratio: {qs.stats.sharpe(result_array_series):.2f}")
    print(f"Max Drawdown: {qs.stats.max_drawdown(result_array_series):.2%}")

    return cum_returns[-1]
    

def load_features_data(symbol):
    """
    Load and merge all feature sets for a given currency pair.
    
    Parameters:
    -----------
    symbol : str
        Currency pair symbol (e.g., 'EURUSD', 'USDCHF', 'XAUUSD')
    
    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with price data and all feature sets (technical, mean reversion, regime detection)
        with appropriate prefixes (tech_, mr_, regime_)
    """
    # Load price data
    data = pd.read_csv(f"{symbol}.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index("Date", inplace=True)
    
    # Load technical indicators features
    features_technical = pd.read_csv(f"technical_indicators_features_{symbol}.csv", parse_dates=["Date"])
    features_technical.set_index("Date", inplace=True)
    features_technical.rename(columns=lambda x: f"tech_{x}" if x != "Date" and x != "Label" and x != f"{symbol}_Close"
                                else x, inplace=True)
    data = data.join(features_technical, how="inner")

    # Load mean reversion features
    features_mean_reversion = pd.read_csv(f"mean_reversion_features_{symbol}.csv", parse_dates=["Date"])
    features_mean_reversion.set_index("Date", inplace=True)
    features_mean_reversion.rename(columns=lambda x: f"mr_{x}" if x != "Date" and x != "Label" and x != f"{symbol}_Close"
                                else x, inplace=True)
    data = data.join(features_mean_reversion, how="inner")

    # Load regime detection features
    features_regime = pd.read_csv(f"regime_detection_features_{symbol}.csv", parse_dates=["Date"])
    features_regime.set_index("Date", inplace=True)
    features_regime.rename(columns=lambda x: f"regime_{x}" if x != "Date" and x != "Label" and x != f"{symbol}_Close"
                                else x, inplace=True)
    data = data.join(features_regime, how="inner")

    data.reset_index(inplace=True)
    
    return data


def calculate_adaptive_portfolio_returns(pair_results_dict, weights=None):
    """
    Calculate portfolio returns by dynamically including only pairs with positive accumulated returns.
    
    At each time step, only pairs that have positive cumulative returns are included in the
    average calculation. This adaptively excludes underperforming pairs.
    
    Parameters:
    -----------
    pair_results_dict : dict
        Dictionary mapping pair names to their result DataFrames
        Each DataFrame must have 'Date' and 'Return' columns
    weights : dict, optional
        Dictionary mapping pair names to their weights for weighted average calculation
        If None, equal weights are used (simple average)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with 'Date' and 'Return' columns representing the adaptive portfolio returns
    """
    # Set default weights if not provided
    if weights is None:
        weights = {pair: 1.0 for pair in pair_results_dict.keys()}
    
    # Combine all returns into a single DataFrame
    combined_df = None
    for pair_name, result_df in pair_results_dict.items():
        if combined_df is None:
            combined_df = pd.DataFrame({'Date': result_df['Date']})
        combined_df[pair_name] = result_df['Return'].values
    
    # Calculate cumulative returns for each pair using compsum
    pair_cumulative_returns = {}
    for pair in pair_results_dict.keys():
        pair_series = pd.Series(combined_df[pair].values)
        pair_cumulative_returns[pair] = qs.stats.compsum(pair_series)
    
    adaptive_returns = []
    
    for idx in range(len(combined_df)):
        # For current month, check accumulated returns UP TO (but not including) current month
        if idx == 0:
            # First month: include all pairs (no history yet)
            profitable_pairs = list(pair_results_dict.keys())
        else:
            # Get accumulated returns up to previous month (idx-1)
            accumulated_returns = {pair: pair_cumulative_returns[pair].iloc[idx-1] 
                                  for pair in pair_results_dict.keys()}
            
            # Identify pairs with positive accumulated returns
            profitable_pairs = [pair for pair, acc_return in accumulated_returns.items() if acc_return > 0]
            
            # If no pairs are profitable, use all pairs (fallback)
        if len(profitable_pairs) == 0:
            current_return = 0.0
            adaptive_returns.append(current_return)
            print(f"Month {idx}: No profitable pairs, return set to 0.0  \n combined return: {combined_df.iloc[idx]}\n\n", flush=True)
            continue
        
        # Calculate weighted average return from profitable pairs for current month
        returns_all_pairs_profitable = [combined_df[pair].iloc[idx] * weights[pair] for pair in profitable_pairs]
        weighted_sum = sum(returns_all_pairs_profitable)
        total_weight = sum(weights[pair] for pair in profitable_pairs)
        current_return = weighted_sum / total_weight
        
        adaptive_returns.append(current_return)
        
        # Print debug info for first few iterations
        if idx == 0:
            print(f"Month {idx}: All pairs included (first month) \n combined return: {combined_df.iloc[idx]}\n\n", flush=True)
        else:
            accumulated_returns = {pair: pair_cumulative_returns[pair].iloc[idx-1] 
                                    for pair in pair_results_dict.keys()}
            print(f"Month {idx}: Profitable pairs: {profitable_pairs} weighted returns_all_pairs_profitable: {returns_all_pairs_profitable} not weigheted: {[combined_df[pair].iloc[idx] for pair in profitable_pairs]} \n combined return: {combined_df.iloc[idx]}", flush=True)
            print(f"  Accumulated returns (up to month {idx-1}): {accumulated_returns}\n\n", flush=True)
        print(f"  Current return: {current_return:.4f}", flush=True)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'Date': combined_df['Date'],
        'Return': adaptive_returns
    })
    
    return result_df


if __name__ == "__main__":
    # Example usage
    #symbol = "USDCHF"
    
    # Load all features data for the symbol
    #data = load_features_data(symbol)

    #data.drop("Label_delete", axis=1, inplace=True)

    
    # Create and run backtest with random strategy
    #strategy = RandomStrategy()
    #strategy = LGBMOptunaStrategy(feature_set="regime_", symbol=symbol, n_trials=10, features_optimization=False, feature_frequency="_18")
    #strategy = MLPOptunaStrategy() NO GO
    #strategy = LogisticRegressionOptunaStrategy() NO GO
    #strategy = GaussianNBOptunaStrategy() NO GO
    #strategy = KNNOptunaStrategy() NO GO
    #strategy = RandomForestOptunaStrategy()
    #strategy = SVCOptunaStrategy() NO GO
    #strategy = AdaBoostOptunaStrategy()
    #strategy = HistGBOptunaStrategy()
    #strategy = XGBoostOptunaStrategy()
    #strategy = CatBoostOptunaStrategy()
    #strategy = TemporalFusionTransformerOptunaStrategy() NO GO
    #strategy = TabNetOptunaStrategy() NO GO
    #strategy = NGBoostOptunaStrategy()
    #strategy = GaussianProcessOptunaStrategy()
    #strategy = PyTorchNeuralNetOptunaStrategy() NO GO
    #strategy = EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(6, 9))
    #strategy = EBMOptunaStrategy()
    # strategy = VotingEnsembleStrategy(
    #    voting_method='majority',
    #    n_splits=5,
    #    threshold=0.5,
    #    random_state=42
    # )

    # run 100 times and get statistics about the result of analyzing the strategy performance
    all_cum_returns = []
    for i in range(1):
        #if hasattr(strategy, 'clear_group_returns'):
        #    strategy.clear_group_returns()
        print(f"Running backtest iteration {i+1}/400")
        #backtest = Backtest(strategy, close_col=f'{symbol}_Close', stop_loss=0.019, start_year=2014, min_history=100)
        #backtest = Backtest(strategy, close_col=f'{symbol}_Close', stop_loss=0.025, start_year=2014, min_history=100)
        #backtest = Backtest(strategy, close_col=f'{symbol}_Close', stop_loss=0.025, start_year=2014, min_history=100)
        
        # EURUSD - Model 1
        symbol = "EURUSD"
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3), close_col=f'{symbol}_Close', stop_loss=0.015, start_year='2014-06-01', min_history=100)
        result_eur_usd_model1 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_eur_usd_model1, strategy_name=f"EURUSD Model1 Strategy Iteration {i+1}", symbol=f"{symbol}_model1")

        # EURUSD - Model 2
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1), close_col=f'{symbol}_Close', stop_loss=0.015, start_year='2014-06-01', min_history=100)
        result_eur_usd_model2 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_eur_usd_model2, strategy_name=f"EURUSD Model2 Strategy Iteration {i+1}", symbol=f"{symbol}_model2")
        
        # Average EURUSD models
        result_eur_usd = pd.DataFrame({
            'Date': result_eur_usd_model1['Date'],
            'Signal': result_eur_usd_model1['Signal'],
            'Amount': result_eur_usd_model1['Amount'],
            'Return': (result_eur_usd_model1['Return'] + result_eur_usd_model2['Return']) / 2
        })
        analyze_strategy_performance(result_eur_usd, strategy_name=f"EURUSD Combined Strategy Iteration {i+1}", symbol=f"{symbol}_combined")

        # USDJPY - Model 1
        symbol = "USDJPY"
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_jpy_usd_model1 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_jpy_usd_model1, strategy_name=f"USDJPY Model1 Strategy Iteration {i+1}", symbol=f"{symbol}_model1")

        # USDJPY - Model 2
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_jpy_usd_model2 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_jpy_usd_model2, strategy_name=f"USDJPY Model2 Strategy Iteration {i+1}", symbol=f"{symbol}_model2")
        
        # Average USDJPY models
        result_jpy_usd = pd.DataFrame({
            'Date': result_jpy_usd_model1['Date'],
            'Signal': result_jpy_usd_model1['Signal'],
            'Amount': result_jpy_usd_model1['Amount'],
            'Return': (result_jpy_usd_model1['Return'] + result_jpy_usd_model2['Return']) / 2
        })
        analyze_strategy_performance(result_jpy_usd, strategy_name=f"USDJPY Combined Strategy Iteration {i+1}", symbol=f"{symbol}_combined")

        # EURJPY - Model 1
        symbol = "EURJPY"
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_eur_jpy_model1 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_eur_jpy_model1, strategy_name=f"EURJPY Model1 Strategy Iteration {i+1}", symbol=f"{symbol}_model1")

        # EURJPY - Model 2
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_eur_jpy_model2 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_eur_jpy_model2, strategy_name=f"EURJPY Model2 Strategy Iteration {i+1}", symbol=f"{symbol}_model2")
        
        # Average EURJPY models
        result_eur_jpy = pd.DataFrame({
            'Date': result_eur_jpy_model1['Date'],
            'Signal': result_eur_jpy_model1['Signal'],
            'Amount': result_eur_jpy_model1['Amount'],
            'Return': (result_eur_jpy_model1['Return'] + result_eur_jpy_model2['Return']) / 2
        })
        analyze_strategy_performance(result_eur_jpy, strategy_name=f"EURJPY Combined Strategy Iteration {i+1}", symbol=f"{symbol}_combined")

        # GBPUSD - Model 1
        symbol = "GBPUSD"
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3), close_col=f'{symbol}_Close', stop_loss=0.020, start_year='2014-06-01', min_history=100)
        result_gbp_usd_model1 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_gbp_usd_model1, strategy_name=f"GBPUSD Model1 Strategy Iteration {i+1}", symbol=f"{symbol}_model1")

        # GBPUSD - Model 2
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1), close_col=f'{symbol}_Close', stop_loss=0.020, start_year='2014-06-01', min_history=100)
        result_gbp_usd_model2 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_gbp_usd_model2, strategy_name=f"GBPUSD Model2 Strategy Iteration {i+1}", symbol=f"{symbol}_model2")
        
        # Average GBPUSD models
        result_gbp_usd = pd.DataFrame({
            'Date': result_gbp_usd_model1['Date'],
            'Signal': result_gbp_usd_model1['Signal'],
            'Amount': result_gbp_usd_model1['Amount'],
            'Return': (result_gbp_usd_model1['Return'] + result_gbp_usd_model2['Return']) / 2
        })
        analyze_strategy_performance(result_gbp_usd, strategy_name=f"GBPUSD Combined Strategy Iteration {i+1}", symbol=f"{symbol}_combined")

        # AUDUSD - Model 1
        symbol = "AUDUSD"
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_aud_usd_model1 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_aud_usd_model1, strategy_name=f"AUDUSD Model1 Strategy Iteration {i+1}", symbol=f"{symbol}_model1")

        # AUDUSD - Model 2
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_aud_usd_model2 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_aud_usd_model2, strategy_name=f"AUDUSD Model2 Strategy Iteration {i+1}", symbol=f"{symbol}_model2")
        
        # Average AUDUSD models
        result_aud_usd = pd.DataFrame({
            'Date': result_aud_usd_model1['Date'],
            'Signal': result_aud_usd_model1['Signal'],
            'Amount': result_aud_usd_model1['Amount'],
            'Return': (result_aud_usd_model1['Return'] + result_aud_usd_model2['Return']) / 2
        })
        analyze_strategy_performance(result_aud_usd, strategy_name=f"AUDUSD Combined Strategy Iteration {i+1}", symbol=f"{symbol}_combined")

        # XAUUSD - Model 1
        symbol = "XAUUSD"
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 18), frequency_range_step=3), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_xau_usd_model1 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_xau_usd_model1, strategy_name=f"XAUUSD Model1 Strategy Iteration {i+1}", symbol=f"{symbol}_model1")

        # XAUUSD - Model 2
        backtest = Backtest(EnsembleOptunaStrategy(feature_set=None, symbol=symbol, frequency_range=(3, 4), frequency_range_step=1), close_col=f'{symbol}_Close', stop_loss=0.025, start_year='2014-06-01', min_history=100)
        result_xau_usd_model2 = backtest.run(load_features_data(symbol))
        analyze_strategy_performance(result_xau_usd_model2, strategy_name=f"XAUUSD Model2 Strategy Iteration {i+1}", symbol=f"{symbol}_model2")
        
        # Average XAUUSD models
        result_xau_usd = pd.DataFrame({
            'Date': result_xau_usd_model1['Date'],
            'Signal': result_xau_usd_model1['Signal'],
            'Amount': result_xau_usd_model1['Amount'],
            'Return': (result_xau_usd_model1['Return'] + result_xau_usd_model2['Return']) / 2
        })
        analyze_strategy_performance(result_xau_usd, strategy_name=f"XAUUSD Combined Strategy Iteration {i+1}", symbol=f"{symbol}_combined")
        # Build adaptive portfolio using only profitable pairs month by month
        pair_results = {
            'GBPUSD': result_gbp_usd,
            'AUDUSD': result_aud_usd,
            'XAUUSD': result_xau_usd,
            'EURUSD': result_eur_usd,
            'USDJPY': result_jpy_usd,
            'EURJPY': result_eur_jpy
        }
        
        # Define weights for each pair
        pair_weights = {
            'GBPUSD': 2.0,
            'AUDUSD': 1.5,
            'XAUUSD': 1.5,
            'EURUSD': 2.5,
            'USDJPY': 1.5,
            'EURJPY': 1.5
        }
        
        random_results = calculate_adaptive_portfolio_returns(pair_results, weights=pair_weights)

        # calculate simple average using results and pair_weights
        average_results = pd.DataFrame()
        average_results['Date'] = result_eur_usd['Date']
        average_results['Return'] = (
           result_gbp_usd['Return'] * pair_weights['GBPUSD'] +
           result_aud_usd['Return'] * pair_weights['AUDUSD'] +
           result_xau_usd['Return'] * pair_weights['XAUUSD'] +
           result_eur_usd['Return'] * pair_weights['EURUSD'] +
           result_jpy_usd['Return'] * pair_weights['USDJPY'] +
           result_eur_jpy['Return'] * pair_weights['EURJPY']
        ) / sum(pair_weights.values())
        analyze_strategy_performance(average_results, strategy_name=f"Weighted Average Strategy Iteration {i+1}", symbol="simple_average")   

        # In my projections I apply a conservative 5% haircut to monthly returns to account for spreads, commissions, slippage and execution effects.
        # If it's negative return, apply 5% penalty. Make the 5% a variable
        penalty_rate = 0.05
        average_results['Return'] = average_results['Return'].apply(lambda x: x * (1 - penalty_rate) if x > 0 else x * (1 + penalty_rate))
        analyze_strategy_performance(average_results, strategy_name=f"Adaptive Portfolio Strategy Iteration {i+1}", symbol="accounting_for_costs")
        
        # Remove last row which might have incomplete data
        # random_results = random_results[:-1]
        
        # Save results to CSV
        #random_results.to_csv(f"random_strategy_results_{i+1}.csv", index=False)
        
        # Analyze performance
        all_cum_returns.append(analyze_strategy_performance(random_results, strategy_name=f"Mean Cumulative Return: {np.mean(all_cum_returns):.2%} Strategy Iteration {i+1}"))
        # Print overall statistics
        print(f"\n----- Overall Performance Metrics {i + 1} / 400 -----")
        print(f"Mean Cumulative Return: {np.mean(all_cum_returns):.2%}")
        print(f"Standard Deviation of Cumulative Returns: {np.std(all_cum_returns):.2%}")
        print(f"Max Cumulative Return: {np.max(all_cum_returns):.2%}")
        print(f"Min Cumulative Return: {np.min(all_cum_returns):.2%}")

    # Print overall statistics
    print("\n----- Overall Performance Metrics -----")
    print(f"Mean Cumulative Return: {np.mean(all_cum_returns):.2%}")
    print(f"Standard Deviation of Cumulative Returns: {np.std(all_cum_returns):.2%}")
    print(f"Max Cumulative Return: {np.max(all_cum_returns):.2%}")
    print(f"Min Cumulative Return: {np.min(all_cum_returns):.2%}")

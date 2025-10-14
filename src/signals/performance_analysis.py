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

def analyze_strategy_performance(strategy_results, benchmark_data=None, strategy_name="Trading Strategy"):
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
    qs.reports.html(result_array_series, output='stats.html', title=f'{strategy_name} Performance Report')
    
    # Print some key metrics including cumulative returns
    print(f"\n----- {strategy_name} Performance Metrics -----")
    print(f"Cumulative Return: {cum_returns[-1]:.2%}")
    print(f"CAGR: {qs.stats.cagr(result_array_series):.2%}")
    print(f"Sharpe Ratio: {qs.stats.sharpe(result_array_series):.2f}")
    print(f"Max Drawdown: {qs.stats.max_drawdown(result_array_series):.2%}")

    return cum_returns[-1]
    


if __name__ == "__main__":
    # Example usage
        # Load price data
    features_macro = pd.read_csv("macro_features.csv", parse_dates=["Date"])
    #features = pd.read_csv("technical_indicators_features.csv", parse_dates=["Date"])
    features_macro.set_index("Date", inplace=True)
    
    data = pd.read_csv("EURUSD.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    data.set_index("Date", inplace=True)
    price = data["EURUSD_Close"]
    # percentage returns of next period
    price_returns = (price.shift(-1) - price) / price
    labels = price_returns.apply(lambda x: 1 if x > 0 else 0)
    # merge price and labels by date index
    features_macro = features_macro.join(price.rename("EURUSD_Close"), how="inner")
    data = features_macro
    # make index Date as a column Date
    
    # rename all the features columns to have prefix macro_ except ['Label', 'Date', 'EURUSD_Close']
    data.rename(columns=lambda x: f"macro_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close" else x, inplace=True)

    # read economic indicators features
    features_economic = pd.read_csv("economic_indicators_features.csv", parse_dates=["Date"])
    features_economic.set_index("Date", inplace=True)
    # rename all the features columns to have prefix econ_ except ['Label', 'Date', 'EURUSD_Close']
    features_economic.rename(columns=lambda x: f"econ_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close"
                                else x, inplace=True)
    # outer join economic indicators features with data by Date
    data = data.join(features_economic, how="inner")

    # read technical indicators features
    features_technical = pd.read_csv("technical_indicators_features.csv", parse_dates=["Date"])
    features_technical.set_index("Date", inplace=True)
    # rename all the features columns to have prefix tech_ except ['Label', 'Date', 'EURUSD_Close']
    features_technical.rename(columns=lambda x: f"tech_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close"
                                else x, inplace=True)
    # outer join technical indicators features with data by Date
    data = data.join(features_technical, how="inner")

    # read mean_reversion features
    features_mean_reversion = pd.read_csv("mean_reversion_features.csv", parse_dates=["Date"])
    features_mean_reversion.set_index("Date", inplace=True)
    # rename all the features columns to have prefix mr_ except ['Label', 'Date', 'EURUSD_Close']
    features_mean_reversion.rename(columns=lambda x: f"mr_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close"
                                else x, inplace=True)
    # outer join mean_reversion features with data by Date
    data = data.join(features_mean_reversion, how="inner")

    # read spread features
    features_spread = pd.read_csv("spread_features.csv", parse_dates=["Date"])
    features_spread.set_index("Date", inplace=True)
    # rename all the features columns to have prefix spread_ except ['Label', 'Date', 'EURUSD_Close']
    features_spread.rename(columns=lambda x: f"spread_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close"
                                else x, inplace=True)
    # outer join spread features with data by Date
    data = data.join(features_spread, how="inner")

    # read spread advanced features
    features_spread_advanced = pd.read_csv("spread_features_advanced.csv", parse_dates=["Date"])
    features_spread_advanced.set_index("Date", inplace=True)
    # rename all the features columns to have prefix spread_ except ['Label', 'Date', 'EURUSD_Close']
    features_spread_advanced.rename(columns=lambda x: f"spreadadv_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close"
                                else x, inplace=True)
    # outer join spread advanced features with data by Date
    data = data.join(features_spread_advanced, how="inner")

    # read regime detection features
    features_regime = pd.read_csv("regime_detection_features.csv", parse_dates=["Date"])
    features_regime.set_index("Date", inplace=True)
    # rename all the features columns to have prefix regime_ except ['Label', 'Date', 'EURUSD_Close']
    features_regime.rename(columns=lambda x: f"regime_{x}" if x != "Date" and x != "Label" and x != "EURUSD_Close"
                                else x, inplace=True)
    # outer join regime detection features with data by Date
    data = data.join(features_regime, how="inner")

    data.reset_index(inplace=True)

    #data.drop("Label_delete", axis=1, inplace=True)

    
    # Create and run backtest with random strategy
    #strategy = RandomStrategy()
    strategy = LGBMOptunaStrategy(feature_set="macro_")
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
    #strategy = EnsembleOptunaStrategy(feature_set=None)
    #strategy = EBMOptunaStrategy()
    # strategy = VotingEnsembleStrategy(
    #    voting_method='majority',
    #    n_splits=5,
    #    threshold=0.5,
    #    random_state=42
    # )

    # run 100 times and get statistics about the result of analyzing the strategy performance
    all_cum_returns = []
    for i in range(400):
        print(f"Running backtest iteration {i+1}/400")
        backtest = Backtest(strategy, close_col='EURUSD_Close')
        random_results = backtest.run(data)
        
        # Remove last row which might have incomplete data
        random_results = random_results[:-1]
        
        # Save results to CSV
        random_results.to_csv(f"random_strategy_results_{i+1}.csv", index=False)
        
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

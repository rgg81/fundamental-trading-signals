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
    result_array_series = strategy_results.set_index("Date")["Return"]
    qs.reports.html(result_array_series, output='stats.html', title='My Backtest')


if __name__ == "__main__":
    # Example usage
        # Load price data
    features = pd.read_csv("src/features/macro_features.csv", parse_dates=["Date"])
    features.set_index("Date", inplace=True)
    
    data = pd.read_csv("src/data_fetch/EURUSD.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    data.set_index("Date", inplace=True)
    price = data["EURUSD_Close"]
    price_returns = price.pct_change().dropna()
    labels = price_returns.apply(lambda x: 1 if x > 0 else 0)
    # merge price and labels by date index
    features = features.join(price.rename("EURUSD_Close"), how="inner")
    data = features.join(labels.rename("Label"), how="inner")
    # make index Date as a column Date
    data.reset_index(inplace=True)
    # merge 


    
    # Create and run backtest with random strategy
    #strategy = RandomStrategy()
    #strategy = LGBMOptunaStrategy()
    #strategy = MLPOptunaStrategy()
    #strategy = LogisticRegressionOptunaStrategy()
    #strategy = GaussianNBOptunaStrategy()
    #strategy = KNNOptunaStrategy()
    #strategy = RandomForestOptunaStrategy()
    # strategy = SVCOptunaStrategy()
    # strategy = AdaBoostOptunaStrategy()
    # strategy = HistGBOptunaStrategy()
    # strategy = XGBoostOptunaStrategy()
    strategy = CatBoostOptunaStrategy()
    #strategy = EBMOptunaStrategy()
    #strategy = VotingEnsembleStrategy(
    #    voting_method='majority',
    #    n_splits=5,
    #    threshold=0.5,
    #    random_state=42
    #)

    backtest = Backtest(strategy, close_col='EURUSD_Close')
    random_results = backtest.run(data)
    
    # Remove last row which might have incomplete data
    random_results = random_results[:-1]
        
    random_results.to_csv("random_strategy_results.csv", index=False)
    
    # Analyze performance
    analyze_strategy_performance(random_results, strategy_name="Random Strategy")
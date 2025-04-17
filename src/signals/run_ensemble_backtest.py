import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ensemble_strategy import VotingEnsembleStrategy
from backtest import Backtest
from performance_analysis import analyze_strategy_performance

def run_ensemble_backtest():
    """
    Run a backtest using the VotingEnsembleStrategy with 10 models
    and compare results between majority and weighted voting methods.
    """
    # Load price and feature data
    print("Loading data...")
    data = pd.read_csv("src/data_fetch/final_dataset.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Run backtest with majority voting
    print("\n===== Running Majority Voting Ensemble Backtest =====")
    majority_strategy = VotingEnsembleStrategy(
        voting_method='majority',
        n_splits=5,
        threshold=0.5,
        random_state=42
    )
    
    majority_backtest = Backtest(
        strategy=majority_strategy,
        max_amount=10,
        stop_loss=0.02,
        close_col='EURUSD_Close'
    )
    
    majority_results = majority_backtest.run(data)
    majority_results.to_csv("majority_voting_results.csv", index=False)
    
    # Run backtest with weighted voting
    print("\n===== Running Weighted Voting Ensemble Backtest =====")
    weighted_strategy = VotingEnsembleStrategy(
        voting_method='weighted',
        n_splits=5,
        threshold=0.5,
        random_state=42
    )
    
    weighted_backtest = Backtest(
        strategy=weighted_strategy,
        max_amount=10,
        stop_loss=0.02,
        close_col='EURUSD_Close'
    )
    
    weighted_results = weighted_backtest.run(data)
    weighted_results.to_csv("weighted_voting_results.csv", index=False)
    
    # Analyze and compare performance
    print("\n===== Analyzing Majority Voting Performance =====")
    analyze_strategy_performance(majority_results, strategy_name="Majority Voting Ensemble")
    
    print("\n===== Analyzing Weighted Voting Performance =====")
    analyze_strategy_performance(weighted_results, strategy_name="Weighted Voting Ensemble")
    

if __name__ == "__main__":
    run_ensemble_backtest()
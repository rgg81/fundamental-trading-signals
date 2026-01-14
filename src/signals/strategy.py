import pandas as pd
import numpy as np
import random
import quantstats as qs
from abc import ABC, abstractmethod

class Strategy(ABC):
    """Base strategy class with common functionality for all trading strategies"""
    
    def __init__(self, symbol="EURUSD", step_size=6, feature_set=None, feature_frequency=None):
        """
        Initialize base strategy parameters
        
        Args:
            symbol: Trading pair symbol (e.g., "EURUSD", "XAUUSD")
            step_size: Number of periods to step forward in backtesting
            feature_set: Feature prefix filter (e.g., "macro_", "tech_", "regime_")
            feature_frequency: Feature frequency suffix (e.g., "_18", "_12", "_6")
        """
        self.symbol = symbol
        self.step_size = step_size
        self.feature_set = feature_set
        self.feature_frequency = feature_frequency
        self.fitted = False
    
    def _filter_features(self, data):
        """
        Filter features based on feature_set and feature_frequency
        
        Args:
            data: DataFrame with all features
            
        Returns:
            DataFrame with filtered features plus required columns (Label, Date, {symbol}_Close)
        """
        required_columns = ['Label', 'Date', f'{self.symbol}_Close']
        
        if self.feature_set is not None:
            # Start with features matching the prefix
            feature_columns = [col for col in data.columns if col.startswith(self.feature_set)]
            # Add required columns
            feature_columns.extend(required_columns)
        else:
            # Use all columns
            feature_columns = data.columns.tolist()
        
        # Apply frequency filter if specified
        if self.feature_frequency is not None:
            feature_columns = [
                col for col in feature_columns 
                if col.endswith(self.feature_frequency) or col in required_columns
            ]
        
        print(f"Feature columns for prediction: {feature_columns}")
        return data[feature_columns]
    
    def analyze_strategy_performance(self, strategy_results, iteration=1, strategy_name="Trading Strategy"):
        """
        Analyze the performance of a trading strategy using QuantStats
        
        Args:
            strategy_results: DataFrame with columns ['Date', 'Signal', 'Amount', 'Return']
            iteration: Iteration number for file naming
            strategy_name: Name of the strategy for display purposes
            
        Returns:
            Sharpe ratio
        """
        # Convert strategy returns to a pandas Series with DatetimeIndex
        result_array_series = strategy_results.set_index("Date")["Return"]

        # Calculate cumulative returns
        cum_returns = qs.stats.compsum(result_array_series)

        # Generate full report
        qs.reports.html(
            result_array_series, 
            output=f'stats_{iteration}.html', 
            title=f'{strategy_name} Performance Report'
        )

        # Print key metrics
        print(f"\n----- {strategy_name} Performance Metrics -----")
        print(f"Cumulative Return: {cum_returns[-1]:.2%}")
        print(f"CAGR: {qs.stats.cagr(result_array_series):.2%}")
        print(f"Sharpe Ratio: {qs.stats.sharpe(result_array_series):.2f}")
        print(f"Max Drawdown: {qs.stats.max_drawdown(result_array_series):.2%}")

        return qs.stats.sharpe(result_array_series)
    
    @abstractmethod
    def fit(self, X, y):
        """
        Train the strategy model
        
        Args:
            X: Feature DataFrame
            y: Target labels
        """
        pass
    
    @abstractmethod
    def generate_signal(self, past_data, current_data):
        """
        Generate trading signals
        
        Args:
            past_data: Historical data for training
            current_data: Current data for prediction
            
        Returns:
            Tuple of (signals, amounts) where:
                signals: List of 1 (buy) or 0 (sell)
                amounts: List of trade sizes (1 to 10)
        """
        pass


class RandomStrategy(Strategy):
    """Random strategy for baseline comparison"""
    
    def __init__(self, symbol="EURUSD", step_size=6):
        super().__init__(symbol=symbol, step_size=step_size)
    
    def fit(self, X, y):
        """Random strategy doesn't need training"""
        self.fitted = True
    
    def generate_signal(self, past_data, current_data):
        """
        Generate random signals alternating between buy and sell
        
        Returns:
            Tuple of (signal, amount) where signal is 0 or 1 and amount is 10
        """
        signal = random.randint(0, 1)
        amount = 10
        return signal, amount




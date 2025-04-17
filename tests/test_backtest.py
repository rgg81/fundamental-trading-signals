import unittest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

from signals.backtest import Backtest
from signals.strategy import Strategy, RandomStrategy

class MockStrategy(Strategy):
    def __init__(self, signal_value=1):
        self.signal_value = signal_value
    
    def generate_signal(self, past_data, current_data):
        # Always return the same signal for testing purposes
        return self.signal_value, 10


class TestBacktest(unittest.TestCase):
    
    def setUp(self):
        """Set up test data that can be used across test methods"""
        # Create a DataFrame with price data
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        prices = np.linspace(100, 200, 100)  # Linear price increase from 100 to 200
        self.data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Feature1': range(100),
            'Feature2': range(100, 200)
        })
        
        # Create test strategies
        self.buy_strategy = MockStrategy(signal_value=1)  # Always buy
        self.sell_strategy = MockStrategy(signal_value=0)  # Always sell
        self.random_strategy = RandomStrategy()
    
    def test_backtest_initialization(self):
        """Test that the Backtest class initializes correctly with different parameters"""
        # Default parameters
        backtest = Backtest(self.buy_strategy)
        self.assertEqual(backtest.strategy, self.buy_strategy)
        self.assertEqual(backtest.max_amount, 10)
        self.assertEqual(backtest.stop_loss, 0.02)
        self.assertEqual(backtest.close_col, 'Close')
        
        # Custom parameters
        backtest = Backtest(
            strategy=self.sell_strategy,
            max_amount=20,
            stop_loss=0.05,
            close_col='CustomClose'
        )
        self.assertEqual(backtest.strategy, self.sell_strategy)
        self.assertEqual(backtest.max_amount, 20)
        self.assertEqual(backtest.stop_loss, 0.05)
        self.assertEqual(backtest.close_col, 'CustomClose')
    
    def test_backtest_run_with_buy_strategy(self):
        """Test backtesting with a strategy that always generates buy signals"""
        backtest = Backtest(self.buy_strategy)
        results = backtest.run(self.data)
        
        # Should skip first 60 rows due to the minimum past data requirement
        self.assertEqual(len(results), len(self.data) - 60)
        
        # All signals should be buy (1)
        self.assertTrue(all(results['Signal'] == 1))
        
        # All amounts should be 10
        self.assertTrue(all(results['Amount'] == 10))
        
        # Test return values (should be positive in an uptrend)
        self.assertTrue(sum(results['Return'] > 0) > sum(results['Return'] <= 0))
    
    def test_backtest_run_with_sell_strategy(self):
        """Test backtesting with a strategy that always generates sell signals"""
        backtest = Backtest(self.sell_strategy)
        results = backtest.run(self.data)
        
        # Should skip first 60 rows due to the minimum past data requirement
        self.assertEqual(len(results), len(self.data) - 60)
        
        # All signals should be sell (0)
        self.assertTrue(all(results['Signal'] == 0))
        
        # Test return values (should be negative in an uptrend)
        self.assertTrue(sum(results['Return'] < 0) > sum(results['Return'] >= 0))
    
    def test_backtest_with_stop_loss(self):
        """Test that the backtest respects the stop loss parameter"""
        backtest = Backtest(self.buy_strategy, stop_loss=0.01)
        results = backtest.run(self.data)
        
        # All returns should be limited by the stop loss
        self.assertTrue(all(results['Return'] <= 0.01))
        self.assertTrue(all(results['Return'] >= -0.01))
    
    def test_backtest_with_different_close_column(self):
        """Test that the backtest works with a different close price column name"""
        # Create data with a different column name for close prices
        data_copy = self.data.copy()
        data_copy['CustomClose'] = data_copy['Close']
        
        backtest = Backtest(self.buy_strategy, close_col='CustomClose')
        results = backtest.run(data_copy)
        
        # Should skip first 60 rows due to the minimum past data requirement
        self.assertEqual(len(results), len(data_copy) - 60)
    
    def test_backtest_with_empty_data(self):
        """Test that the backtest handles an empty DataFrame gracefully"""
        empty_data = pd.DataFrame(columns=['Date', 'Close', 'Feature1', 'Feature2'])
        backtest = Backtest(self.buy_strategy)
        results = backtest.run(empty_data)
        
        # Should return an empty DataFrame
        self.assertTrue(results.empty)
    
    def test_backtest_with_random_strategy(self):
        """Test that the backtest works with a random strategy"""
        backtest = Backtest(self.random_strategy)
        results = backtest.run(self.data)
        
        # Should skip first 60 rows due to the minimum past data requirement
        self.assertEqual(len(results), len(self.data) - 60)
        
        # Signals should be a mix of 0s and 1s
        self.assertTrue(0 in results['Signal'].values and 1 in results['Signal'].values)
    
    def test_backtest_with_volatility_data(self):
        """Test backtesting with volatile price data (up and down movements)"""
        # Create volatile price data (sine wave)
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        t = np.linspace(0, 4*np.pi, 100)
        prices = 150 + 50 * np.sin(t)  # Oscillate between 100 and 200
        
        volatile_data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Feature1': range(100),
            'Feature2': range(100, 200)
        })
        
        # Test with buy strategy
        backtest = Backtest(self.buy_strategy)
        results = backtest.run(volatile_data)
        
        # Should have a mix of positive and negative returns
        self.assertTrue(sum(results['Return'] > 0) > 0)
        self.assertTrue(sum(results['Return'] < 0) > 0)


if __name__ == '__main__':
    unittest.main()
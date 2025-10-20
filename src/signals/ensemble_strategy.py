import pandas as pd
import numpy as np
from strategy import Strategy
from lgbm_strategy import LGBMOptunaStrategy
from rf_strategy import RandomForestOptunaStrategy
from adaboost_strategy import AdaBoostOptunaStrategy
from histgb_strategy import HistGBOptunaStrategy
from xgboost_strategy import XGBoostOptunaStrategy
from catboost_strategy import CatBoostOptunaStrategy
from ngboost_strategy import NGBoostOptunaStrategy
from gp_strategy import GaussianProcessOptunaStrategy
import warnings

warnings.filterwarnings('ignore')

class EnsembleOptunaStrategy(Strategy):
    """
    An ensemble strategy that combines multiple Optuna-optimized strategies.
    Uses LGBMOptunaStrategy, RandomForestOptunaStrategy, AdaBoostOptunaStrategy,
    HistGBOptunaStrategy, XGBoostOptunaStrategy, CatBoostOptunaStrategy, 
    NGBoostOptunaStrategy, and GaussianProcessOptunaStrategy.
    """
    
    def __init__(self, max_amount=10, feature_set=None):
        self.max_amount = max_amount
        self.strategies = {}
        self.fitted = False
        self.feature_set = feature_set

        # Initialize all strategies
        #for i in range(1, 50):
        #    self.strategies[f'LGBM_{i}'] = LGBMOptunaStrategy(feature_set="macro_")
        for i in range(0, 10):
            self.strategies[f'LGBM_{i}'] = LGBMOptunaStrategy(feature_set="tech_")
        for i in range(10, 20):
            self.strategies[f'LGBM_{i}'] = LGBMOptunaStrategy(feature_set="econ_")
        for i in range(20, 30):
            self.strategies[f'LGBM_{i}'] = LGBMOptunaStrategy(feature_set="mr_")
        for i in range(30, 40):
            self.strategies[f'LGBM_{i}'] = LGBMOptunaStrategy(feature_set="spread_")
        for i in range(40, 50):
            self.strategies[f'LGBM_{i}'] = LGBMOptunaStrategy(feature_set="spreadadv_")

        print(f"Ensemble Strategy initialized with {len(self.strategies)} strategies")

    def _aggregate_predictions(self, predictions):
        """Aggregate predictions using weighted voting based on amounts"""
        if not predictions:
            print("No valid predictions from strategies, returning default")
            return 0, self.max_amount
        
        total_weight = 0
        weighted_signal_sum = 0
        
        print(f"\nAggregating {len(predictions)} strategy predictions:")
        
        for name, pred in predictions.items():
            signal = pred['signal']
            amount = pred['amount']
            
            if signal == 1:
                # Add weight for buy signal
                total_weight += self.max_amount
                weighted_signal_sum += amount
            else:  # signal == 0
                # Subtract weight for sell signal
                total_weight += self.max_amount
                weighted_signal_sum -= amount
            
            print(f"  {name}: Signal={signal}, Amount={amount}, Weight={'+'if signal==1 else '-'}{amount}")
        
        # Calculate ensemble signal and amount
        if total_weight == 0:
            ensemble_signal = 0
            ensemble_amount = self.max_amount
        else:
            # Normalize the weighted signal to [-1, 1] range
            normalized_signal = weighted_signal_sum / total_weight
            
            # Convert to final signal and amount
            if normalized_signal > 0:
                ensemble_signal = 1
                # Scale amount proportionally to confidence (0 to max_amount)
                ensemble_amount = min(self.max_amount, abs(normalized_signal) * self.max_amount)
            else:
                ensemble_signal = 0
                # Scale amount proportionally to confidence (0 to max_amount)
                ensemble_amount = min(self.max_amount, abs(normalized_signal) * self.max_amount)
            
            # Ensure minimum amount of 1 and maximum of max_amount
            ensemble_amount = max(1, min(self.max_amount, int(ensemble_amount)))
        
        print(f"\nEnsemble Result:")
        print(f"  Weighted Sum: {weighted_signal_sum}")
        print(f"  Total Weight: {total_weight}")
        print(f"  Normalized Signal: {weighted_signal_sum/total_weight if total_weight > 0 else 0:.3f}")
        print(f"  Final Signal: {ensemble_signal}")
        print(f"  Final Amount: {ensemble_amount}")
        
        return ensemble_signal, ensemble_amount

    def generate_signal(self, past_data, current_data):
        """Generate ensemble signal from all strategies"""
        # Prepare data for fitting if not already fitted
        print(f"\n--- Ensemble Prediction ---")
        # include ['Label', 'Date', 'EURUSD_Close'] in features_columns
        if self.feature_set is None:
            feature_columns = past_data.columns
        else:
            feature_columns = [col for col in past_data.columns if col.startswith(self.feature_set)]
            feature_columns.extend(['Label', 'Date', 'EURUSD_Close'])
        # print feature_columns with some explanation
        print(f"Feature columns for prediction: {feature_columns}")

        # Get predictions from all strategies
        strategy_predictions = {}
        
        for name, strategy in self.strategies.items():
            try:
                # Each strategy will call its own fit method in generate_signal
                signal, amount = strategy.generate_signal(past_data[feature_columns], current_data[feature_columns])
                strategy_predictions[name] = {'signal': signal, 'amount': amount}
                print(f"{name}: Signal={signal}, Amount={amount}")
                
            except Exception as e:
                print(f"Failed to get prediction from {name}: {e}")
                continue
        
        # Aggregate all predictions
        ensemble_signal, ensemble_amount = self._aggregate_predictions(strategy_predictions)
        
        return int(ensemble_signal), int(ensemble_amount)
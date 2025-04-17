import pandas as pd
import numpy as np
import random
from abc import ABC, abstractmethod
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, past_data, current_data):
        """
        Implement your strategy logic here.
        This method should return a tuple: (signal, amount)
        signal: 1 for buy, 0 for hold/sell
        amount: the amount to trade
        """
        pass

class RandomStrategy(Strategy):
    def generate_signal(self, past_data, current_data):
        """
        Generate a random signal alternating between buy and sell.
        Returns: (signal, amount)
        signal: 1 for buy, 0 for hold/sell
        amount: the amount to trade (always 1)
        """
        # Generate a random signal (0 or 1)
        signal = random.randint(0, 1)
        
        # Amount is always 1
        amount = 10
        
        return signal, amount

class LGBMOptunaStrategy(Strategy):
    def __init__(self, n_trials=30, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False

    def fit(self, X, y):
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                # can I get different booting types?
                'boosting_type': 'gbdt',
                'num_iterations': trial.suggest_int('num_iterations', 5, 50),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.3, 1.0),
                'is_unbalance': True
            }
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                gbm = lgb.LGBMClassifier(**params)
                gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                preds = gbm.predict(X_val)
                scores.append(accuracy_score(y_val, preds))
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'boosting_type': 'gbdt', 'seed': self.random_state})
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X, y)
        self.fitted = True
        preds = self.model.predict(X)
        print("Classification Report:\n", classification_report(y, preds))

    def generate_signal(self, past_data, current_data):
        # X all data frame less Label, Date and Close
        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']
        # Fit model
        # I want to reset the model every time I call generate_signal
        self.model = None
        self.fit(X, y)
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        # current_data should be a pandas Series or DataFrame with the same features as X
        pred = self.model.predict(X_pred)[0]
        return int(pred), 10  # signal, amount
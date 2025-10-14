import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy

class LGBMOptunaStrategy(Strategy):
    def __init__(self, n_trials=50, n_splits=36, feature_set="macro_"):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.models = []
        self.best_params = None
        self.fitted = False
        self.feature_set = feature_set

    def fit(self, X, y):
        seed_random = random.randint(1, 10000)
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                # can I get different booting types?
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                'num_iterations': trial.suggest_int('num_iterations', 10, 50),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
                'is_unbalance': True,
                # seed parameter for light gbm
                'seed': seed_random,             # main random seed
                'bagging_seed': seed_random,     # specifically for bagging
                'feature_fraction_seed': seed_random,  # for feature subsampling
                'data_random_seed': seed_random, # affects data-related randomness
                'drop_seed': seed_random, 
            }
            # I want to optimize the max_train_size and gap
            max_train_size = trial.suggest_int('max_train_size', 30, 120)
            gap = trial.suggest_int('gap', 0, 10)
             

            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=1, gap=gap)
            scores = []
            splits = list(tscv.split(X))
            for train_idx, val_idx in splits[:-1]:  # only the last 3 splits to save time
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                gbm = lgb.LGBMClassifier(**params)
                gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                
                preds = gbm.predict(X_val)
                scores.append(accuracy_score(y_val, preds))
            for train_idx, val_idx in splits[-1:]:  # only the last 3 splits to save time
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                gbm = lgb.LGBMClassifier(**params)
                gbm.fit(X_train, y_train)
                preds = gbm.predict(X_val)
                # print the classification report of this preds
                print(f"\n Trial number: {trial.number} Validation classification report:\n", classification_report(y_val, preds), flush=True)
            print(f"\n Trial number: {trial.number} Scores:{scores}\n", flush=True)
            mean_score = np.mean(scores)
            
            return 1-mean_score  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        # instead of returning the best trial, I want to get the best n trials and fit a model for each of them
        # return all trials, I want to sort them by value
        top_n_best_trials = 5
        self.best_trials = sorted(study.trials, key=lambda x: x.value)[:top_n_best_trials]
        print("Best trials: ", [(t.number, t.value) for t in self.best_trials])
        for trial in self.best_trials:
            print(f"Trial {trial.number} params: {trial.params}")
            self.best_params = trial.params
            # run the folds again with the best params and save the model for each fold
            print("Best params:", self.best_params)
            # update with the seeds
            self.best_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                'is_unbalance': True,
                'seed': seed_random,             # main random seed
                'bagging_seed': seed_random,     # specifically for bagging
                'feature_fraction_seed': seed_random,  # for feature subsampling
                'data_random_seed': seed_random, # affects data-related randomness
                'drop_seed': seed_random, 
            })
            max_train_size = self.best_params.pop('max_train_size')
            gap = self.best_params.pop('gap')

            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=1, gap=gap)
            splits = list(tscv.split(X))

            last_predictions = []

            for train_idx, val_idx in splits[-1:]:  # only the last 3 splits to save time
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                gbm = lgb.LGBMClassifier(**self.best_params)
                gbm.fit(X_train, y_train)
                preds = gbm.predict(X_val)
                self.models.append(gbm)
                last_predictions.append([preds[-1], y_val.iloc[-1]])

            print(f"Trial number: {trial.number} Accuracy: {trial.value} Last best params predictions report:\n", classification_report([x[1] for x in last_predictions], [x[0] for x in last_predictions]), flush=True)
        self.fitted = True

    def generate_signal(self, past_data, current_data):
        feature_columns = [col for col in past_data.columns if col.startswith(self.feature_set)]
        feature_columns.extend(['Label', 'Date', 'EURUSD_Close'])
        # print feature_columns with some explanation
        print(f"Feature columns for prediction: {feature_columns}")
        # X all data frame less Label, Date and Close
        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        past_data = past_data[feature_columns]
        current_data = current_data[feature_columns]

        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']
        # Fit model
        # I want to reset the model every time I call generate_signal
        self.models = []
        self.fit(X, y)
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        # iterate over all models and get the predictions, each prediction will have the same weight, it should be maximum 10
        votes = [model.predict(X_pred)[0] for model in self.models]
        print(f"Individual model votes: {votes}")
        
        total_weight = 0
        weighted_signal_sum = 0
        for vote in votes:
            if vote == 1:
                total_weight += 10
                weighted_signal_sum += 10
            else:
                total_weight += 10
                weighted_signal_sum -= 10

        normalized_signal = weighted_signal_sum / total_weight
        # Calculate the final prediction and amount
        if normalized_signal > 0:
            pred = 1
        else:
            pred = 0
        return int(pred), min(abs(normalized_signal) * 10, 10)  # signal, amount
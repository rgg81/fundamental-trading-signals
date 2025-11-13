import random
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy
from functools import partial

class LGBMOptunaStrategy(Strategy):
    def __init__(self, n_trials=50, n_splits=6, feature_set="macro_"):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.models = []
        self.features_per_model = []
        self.best_params = None
        self.fitted = False
        self.feature_set = feature_set
        self.step_size = 6  # Added step_size attribute

    def fit(self, X, y):

        def objective(trial, InputFeatures, y_label, seed_random):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'verbosity': -1,
                # can I get different booting types?
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
                'num_iterations': trial.suggest_int('num_iterations', 50, 100),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
                #'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                #'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
                #'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
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
             

            tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=self.step_size, gap=gap)
            scores = []
            splits = list(tscv.split(InputFeatures))
            for train_idx, val_idx in splits[:-1]:  # only the last 3 splits to save time
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                gbm = lgb.LGBMClassifier(**params)
                gbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                
                preds = gbm.predict(X_val)
                scores.append(accuracy_score(y_val, preds))
            for train_idx, val_idx in splits[-1:]:  # only the last 3 splits to save time
                X_train, X_val = InputFeatures.iloc[train_idx], InputFeatures.iloc[val_idx]
                y_train, y_val = y_label.iloc[train_idx], y_label.iloc[val_idx]
                gbm = lgb.LGBMClassifier(**params)
                gbm.fit(X_train, y_train)
                preds = gbm.predict(X_val)
                # print the classification report of this preds
                print(f"\n Trial number: {trial.number} Validation classification report:\n", classification_report(y_val, preds), flush=True)
            print(f"\n Trial number: {trial.number} Scores:{scores}\n", flush=True)
            mean_score = np.mean(scores)
            
            return 1-mean_score
        
        def objective_feature_selection(trial):
            seed_random = trial.suggest_int('seed_random', 1, 100000)
            random.seed(seed_random)
            np.random.seed(seed_random)
            selected_features = []
            for col in X.columns:
                if trial.suggest_categorical(f'use_{col}', [True, False]):
                    selected_features.append(col)
            
            # Ensure at least one feature (or minimum 3 for stability)
            if len(selected_features) == 0:  # Adjust minimum as needed
                return float('inf')  # Worst score - Optuna will avoid this
            
            X_selected = X[selected_features]
            print(f"Features used in this trial: {X_selected.columns.tolist()}", flush=True)
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.step_size)
            all_preds = []
            all_targets = []
            splits = list(tscv.split(X_selected))
            for train_idx, val_idx in splits[:-1]:  # only the last 3 splits to save time
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                objective_func = partial(objective, InputFeatures=X_train, y_label=y_train, seed_random=seed_random)
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_random))
                study.optimize(objective_func, n_trials=self.n_trials)
                best_trial = study.best_trial
                best_params = best_trial.params
                best_params.update({
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
                gbm = lgb.LGBMClassifier(**best_params)
                gbm.fit(X_train, y_train)
                preds = gbm.predict(X_val)
                all_preds.extend(list(preds))
                all_targets.extend(list(y_val))
            for train_idx, val_idx in splits[-1:]:
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                objective_func = partial(objective, InputFeatures=X_train, y_label=y_train, seed_random=seed_random)
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_random))
                study.optimize(objective_func, n_trials=self.n_trials)
                best_trial = study.best_trial
                best_params = best_trial.params
                best_params.update({
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
                max_train_size = best_params.pop('max_train_size')
                gap = best_params.pop('gap')

                tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=self.step_size, gap=gap)
                splits = list(tscv.split(X_selected))

                for train_idx, val_idx in splits[-1:]:  # only the last 3 splits to save time
                    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    gbm = lgb.LGBMClassifier(**best_params)
                    gbm.fit(X_train, y_train)
                    preds = gbm.predict(X_val)
                    print(f"\n Feature selection Trial number: {trial.number} Classification report last fold:\n", classification_report(y_val, preds), flush=True)

            
            accuracy = accuracy_score(all_targets, all_preds)
            # classification report
            print(f"\n Feature selection Trial number: {trial.number} Overall classification report:\n", classification_report(all_targets, all_preds), flush=True)
            return 1 - accuracy
            
        study_feature = optuna.create_study(direction='minimize')
        study_feature.optimize(objective_feature_selection, n_trials=30)
        top_n_best_trials_feature = 10
        best_trials_feature = sorted(study_feature.trials, key=lambda x: x.value)[:top_n_best_trials_feature]    
        print("Best trials Features: ", [(t.number, t.value) for t in best_trials_feature], flush=True)
        for trial_feature in best_trials_feature:
            selected_features = []
            for col in X.columns:
                if trial_feature.params.get(f'use_{col}', False):
                    selected_features.append(col)
            print(f"Trial {trial_feature.number} selected features: {selected_features}", flush=True)
            X_selected = X[selected_features]
            seed_random = trial_feature.params.get('seed_random')
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.step_size)
            splits = list(tscv.split(X_selected))
            for train_idx, val_idx in splits[-1:]:
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                objective_func = partial(objective, InputFeatures=X_train, y_label=y_train, seed_random=seed_random)
                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed_random))
                study.optimize(objective_func, n_trials=self.n_trials)
                
                # instead of returning the best trial, I want to get the best n trials and fit a model for each of them
                # return all trials, I want to sort them by value
                top_n_best_trials = 5
                best_trials = sorted(study.trials, key=lambda x: x.value)[:top_n_best_trials]
                print("Best trials: ", [(t.number, t.value) for t in best_trials])
                for trial in best_trials:
                    print(f"Trial {trial.number} params: {trial.params}")
                    best_params = trial.params
                    # run the folds again with the best params and save the model for each fold
                    print("Best params:", best_params)
                    # update with the seeds
                    best_params.update({
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
                    max_train_size = best_params.pop('max_train_size')
                    gap = best_params.pop('gap')

                    tscv = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=max_train_size, test_size=self.step_size, gap=gap)
                    splits = list(tscv.split(X_selected))

                    for train_idx, val_idx in splits[-1:]:  # only the last 3 splits to save time
                        X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        gbm = lgb.LGBMClassifier(**best_params)
                        gbm.fit(X_train, y_train)
                        preds = gbm.predict(X_val)
                        self.models.append(gbm)
                        self.features_per_model.append(X_train.columns.tolist())
                        print(f"Trial number: {trial.number} Accuracy: {trial.value} Last best params predictions report:\n", classification_report(y_val, preds), flush=True)
        self.fitted = True

    def generate_signal(self, past_data, current_data):
        if self.feature_set is not None:
            feature_columns = [col for col in past_data.columns if col.startswith(self.feature_set)]
            feature_columns.extend(['Label', 'Date', 'EURUSD_Close'])
        else:
            feature_columns = past_data.columns.tolist()
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
        self.features_per_model = []
        self.fit(X, y)
        # X_pred should be the same features as X
        X_preds = current_data.drop(columns=columns_to_drop)
        preds = []
        amounts = []
        # iterate over all models and get the predictions, each prediction will have the same weight, it should be maximum 10
        for _, X_pred in X_preds.iterrows():
            votes = []
        
            # Each model gets only its relevant features
            for i, model in enumerate(self.models):
                # Get the features this specific model was trained on
                model_features = self.features_per_model[i]
                
                # Select only those features from X_pred
                X_pred_filtered = X_pred[model_features]
                
                # Make prediction with the filtered features
                vote = model.predict([X_pred_filtered])[0]
                votes.append(vote)
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
            preds.append(pred)
            amounts.append(int(round(min(abs(normalized_signal) * 10, 10))))
        return preds, amounts
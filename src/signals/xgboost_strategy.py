import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy

class XGBoostOptunaStrategy(Strategy):
    def __init__(self, n_trials=15, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False


    def fit(self, X, y):
        # Clean data to handle infinities
        
        def objective(trial):
            # Calculate class balance ratio
            negative_count = (y == 0).sum()
            positive_count = (y == 1).sum()
            scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
            
            # Define the hyperparameter search space for XGBoost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                # different boosting types for xgboost
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                # Handle class imbalance
                'scale_pos_weight': scale_pos_weight,
               # 'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
               # 'subsample': trial.suggest_float('subsample', 0.9, 1.0),
               # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
              #  'gamma': trial.suggest_float('gamma', 0.1, 10, log=True),
               # 'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),
               # 'reg_lambda': trial.suggest_float('reg_lambda', 1, 10, log=True),
               # 'random_state': self.random_state,
             #   'use_label_encoder': False,  # Prevents warning about label encoder
              #  'eval_metric': 'logloss'  # Required for newer versions of XGBoost
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    xgb = XGBClassifier(**params)
                    xgb.fit(X_train, y_train)
                    preds = xgb.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
                except Exception as e:
                    print(f"Error in XGBoost training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best XGBoost parameters: {self.best_params}")
        
        # Train the final model with best parameters
        self.model = XGBClassifier(**self.best_params)
        self.model.fit(X, y)
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X)
        print("Classification Report:\n", classification_report(y, preds))

        # Feature importance for XGBoost
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        

    def generate_signal(self, past_data, current_data):
        # Remove inf data
        past_data = past_data.replace([np.inf, -np.inf], np.nan).dropna()
        current_data = current_data.replace([np.inf, -np.inf], np.nan).dropna()

        if current_data.empty:
            return None, 10
        
        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']
        
        # Fit model
        # Reset the model every time we call generate_signal
        self.model = None
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        
        # Predict
        pred = self.model.predict(X_pred)[0]
        
        # Get prediction probabilities
        pred_probs = self.model.predict_proba(X_pred)[0]
        print(f"Prediction probabilities: Class 0: {pred_probs[0]:.4f}, Class 1: {pred_probs[1]:.4f}")
        
        return int(pred), 10  # signal, amount

import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy

class HistGBOptunaStrategy(Strategy):
    def __init__(self, n_trials=10, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.fitted = False

    def _clean_data(self, df):
        """Clean data by removing rows with NaN or infinity values"""
        # Make a copy to avoid modifying the original data
        df_clean = df.copy()
        
        # Remove rows with infinity values
        df_clean = df_clean[~df_clean.isin([np.inf, -np.inf]).any(axis=1)]
        
        # Remove rows with NaN values
        df_clean = df_clean.dropna()
        
        return df_clean

    def fit(self, X, y):
        # Clean data to handle infinities
        X_clean = self._clean_data(X)
        
        def objective(trial):
            # Define the hyperparameter search space for HistGradientBoostingClassifier
            params = {
                'max_iter': trial.suggest_int('max_iter', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0)
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    hist_gb = HistGradientBoostingClassifier(**params)
                    hist_gb.fit(X_train, y_train)
                    preds = hist_gb.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
                except Exception as e:
                    print(f"Error in HistGradientBoosting training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best HistGradientBoosting parameters: {self.best_params}")
        
        # Train the final model with best parameters
        self.model = HistGradientBoostingClassifier(**self.best_params)
        self.model.fit(X_clean, y)
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X_clean)
        print("Classification Report:\n", classification_report(y, preds))

        # Feature importance for HistGradientBoostingClassifier
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': X_clean.columns,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 most important features:")
            print(feature_importance.head(10))

    def generate_signal(self, past_data, current_data):
        # Clean input data
        past_data = self._clean_data(past_data)
        current_data = self._clean_data(current_data)

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
        
        # Clean prediction data
        X_pred_clean = self._clean_data(X_pred)
        
        # Predict
        pred = self.model.predict(X_pred_clean)[0]
        
        # Get prediction probabilities
        pred_probs = self.model.predict_proba(X_pred_clean)[0]
        print(f"Prediction probabilities: Class 0: {pred_probs[0]:.4f}, Class 1: {pred_probs[1]:.4f}")
        
        # Decision function or margins can be useful for confidence assessment
        if hasattr(self.model, 'decision_function'):
            decision_values = self.model.decision_function(X_pred_clean)[0]
            print(f"Decision function value: {decision_values:.4f}")
        
        return int(pred), 10  # signal, amount

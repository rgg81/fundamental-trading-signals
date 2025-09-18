import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from strategy import Strategy

class RandomForestOptunaStrategy(Strategy):
    def __init__(self, n_trials=3, n_splits=5, random_state=42):
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
        X_clean = X
        def objective(trial):
            # Define the hyperparameter search space for RandomForest
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 201),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                #'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                #'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', [None]),
                'bootstrap': True,
                #'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
                #'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            for train_idx, val_idx in tscv.split(X_clean):
                X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                try:
                    rf = RandomForestClassifier(**params)
                    rf.fit(X_train, y_train)
                    preds = rf.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))
                except Exception as e:
                    print(f"Error in Random Forest training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best Random Forest parameters: {self.best_params}")
        
        # Train the final model with best parameters
        self.model = RandomForestClassifier(**self.best_params)
        self.model.fit(X_clean, y)
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X_clean)
        print("Classification Report:\n", classification_report(y, preds))
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_clean.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

    def generate_signal(self, past_data, current_data):
        # X all data frame less Label, Date and Close
        past_data = self._clean_data(past_data)
        current_data = self._clean_data(current_data)
        if current_data.empty:
            return None, 10

        columns_to_drop = ['Label', 'Date', 'EURUSD_Close']
        X = past_data.drop(columns=columns_to_drop)
        # y only Label
        y = past_data['Label']

        X_pred = current_data.drop(columns=columns_to_drop)
        # Clean prediction data
        X_pred_clean = X_pred
        
        # Fit model
        # Reset the model every time we call generate_signal
        predictions = []
        self.model = None
        self.fit(X, y)
        predictions.append(self.model.predict(X_pred_clean)[0])
        
        # Majority vote
        print(f"Predictions: {predictions}", flush=True)
        pred = max(set(predictions), key=predictions.count)
        
        return int(pred), 10  # signal, amount
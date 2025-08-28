import pandas as pd
import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from strategy import Strategy

class GaussianProcessOptunaStrategy(Strategy):
    def __init__(self, n_trials=5, n_splits=3):  # Reduced trials and splits
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.best_params = None
        self.fitted = False
        self.is_fallback = False

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
        
        # Much more aggressive sample limiting for speed
        max_samples = 60  # Reduced from 200 to 50
        if len(X_clean) > max_samples:
            print(f"Limiting dataset to {max_samples} samples for Gaussian Process efficiency")
            X_clean = X_clean.sample(n=max_samples)
            y = y.loc[X_clean.index]
        
        
        def objective(trial):
            # Simplified hyperparameter space for speed
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern'])  # Removed complex kernels
            
            # Reduced parameter ranges
            length_scale = trial.suggest_float('length_scale', 0.5, 5.0)  # Narrower range
            
            if kernel_type == 'rbf':
                kernel = RBF(length_scale=length_scale)  # Removed ConstantKernel
            else:  # matern
                nu = trial.suggest_categorical('nu', [1.5, 2.5])  # Removed 0.5 (slowest)
                kernel = Matern(length_scale=length_scale, nu=nu)
            
            # Reduced iterations
            max_iter_predict = trial.suggest_int('max_iter_predict', 10, 20)  # Much lower
            
            # Always use feature selection to reduce dimensionality
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                try:
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Skip if validation set is too small
                    if len(y_val) < 2:
                        continue
                        
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    X_train_selected = X_train_scaled
                    X_val_selected = X_val_scaled
                    
                    
                    # Train Gaussian Process classifier with timeout
                    gp = GaussianProcessClassifier(
                        kernel=kernel,
                        max_iter_predict=max_iter_predict,
                        optimizer=None  # Disable hyperparameter optimization for speed
                    )
                    gp.fit(X_train_selected, y_train)
                    
                    # Predict
                    preds = gp.predict(X_val_selected)
                    scores.append(accuracy_score(y_val, preds))
                    
                except Exception as e:
                    print(f"Error in GP fold: {e}")
                    continue  # Skip failed folds instead of returning 0.0
                    
            if len(scores) == 0:
                return 1.0  # Return worst score if all folds failed
                
            return 1.0 - np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best Gaussian Process parameters: {self.best_params}")
        
        # Build the best kernel
        kernel_type = self.best_params['kernel_type']
        length_scale = self.best_params['length_scale']
        
        if kernel_type == 'rbf':
            best_kernel = RBF(length_scale=length_scale)
        else:  # matern
            nu = self.best_params['nu']
            best_kernel = Matern(length_scale=length_scale, nu=nu)
        
        # Train the final model with best parameters
        self.model = GaussianProcessClassifier(
            kernel=best_kernel,
            max_iter_predict=self.best_params['max_iter_predict'],
            optimizer=None  # Disable optimization for speed
        )
        
        # Always use feature selection in final model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Force feature selection for speed
        X_selected = X_scaled
        
        # Fit the model
        self.model.fit(X_selected, y)
        self.fitted = True
        self.is_fallback = False
        
        # Evaluate on training data
        preds = self.model.predict(X_selected)
        print("Classification Report:\n", classification_report(y, preds))

        # Feature importance analysis
        try:
            if self.feature_selector is not None:
                feature_names = X_clean.columns[self.feature_selector.get_support()]
                print(f"\nUsing {len(feature_names)} selected features for Gaussian Process")
                print("Selected features:", list(feature_names))
            else:
                feature_names = X_clean.columns
                print(f"\nUsing all {len(feature_names)} features for Gaussian Process")
            
            if not self.is_fallback:
                print("Gaussian Process provides prediction uncertainties instead of feature importance")
            
        except Exception as e:
            print(f"Could not analyze features: {e}")

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
        self.scaler = None
        self.feature_selector = None
        self.fit(X, y)
        
        # X_pred should be the same features as X
        X_pred = current_data.drop(columns=columns_to_drop)
        
        # Clean prediction data
        X_pred_clean = self._clean_data(X_pred)
        
        # Scale prediction data
        X_pred_scaled = self.scaler.transform(X_pred_clean)
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            X_pred_selected = self.feature_selector.transform(X_pred_scaled)
        else:
            X_pred_selected = X_pred_scaled
        
        # Predict
        pred = self.model.predict(X_pred_selected)[0]
        
        # Try to get prediction uncertainty if using GP (not fallback)
        try:
            if not self.is_fallback and hasattr(self.model, 'predict_proba'):
                pred_proba = self.model.predict_proba(X_pred_selected)[0]
                confidence = max(pred_proba)
                print(f"Prediction confidence: {confidence:.3f}")
        except Exception as e:
            pass  # Ignore uncertainty calculation errors
        
        return int(pred), 10  # signal, amount

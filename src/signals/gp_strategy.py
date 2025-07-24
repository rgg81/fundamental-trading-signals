import pandas as pd
import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from strategy import Strategy

class GaussianProcessOptunaStrategy(Strategy):
    def __init__(self, n_trials=10, n_splits=5, random_state=42):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_selector = None
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
        
        # Check if we have enough data - GP is particularly sensitive to small datasets
        if len(X_clean) < 20:
            print(f"Warning: Only {len(X_clean)} samples available. Gaussian Process may be very slow or fail.")
        
        # Limit the dataset size for GP (it's O(n³) complexity)
        max_samples = 200
        if len(X_clean) > max_samples:
            print(f"Limiting dataset to {max_samples} samples for Gaussian Process efficiency")
            X_clean = X_clean.sample(n=max_samples, random_state=self.random_state)
            y = y.loc[X_clean.index]
        
        def objective(trial):
            # Define the hyperparameter search space for Gaussian Process Classifier
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern', 'rbf_white', 'matern_white'])
            
            # Kernel parameters
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
            
            if kernel_type == 'rbf':
                kernel = C(1.0) * RBF(length_scale=length_scale)
            elif kernel_type == 'matern':
                nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
                kernel = C(1.0) * Matern(length_scale=length_scale, nu=nu)
            elif kernel_type == 'rbf_white':
                noise_level = trial.suggest_float('noise_level', 1e-5, 1e-1, log=True)
                kernel = C(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
            else:  # matern_white
                nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
                noise_level = trial.suggest_float('noise_level', 1e-5, 1e-1, log=True)
                kernel = C(1.0) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level)
            
            # Other GP parameters
            max_iter_predict = trial.suggest_int('max_iter_predict', 50, 200)
            
            # Feature selection parameters
            use_feature_selection = trial.suggest_categorical('use_feature_selection', [True, False])
            if use_feature_selection:
                k_features = trial.suggest_int('k_features', min(3, X_clean.shape[1]), 
                                             min(X_clean.shape[1], max(3, X_clean.shape[1] // 3)))
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X_clean):
                try:
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Feature selection if enabled
                    if use_feature_selection:
                        selector = SelectKBest(f_classif, k=k_features)
                        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                        X_val_selected = selector.transform(X_val_scaled)
                    else:
                        X_train_selected = X_train_scaled
                        X_val_selected = X_val_scaled
                    
                    # Train Gaussian Process classifier
                    gp = GaussianProcessClassifier(
                        kernel=kernel,
                        max_iter_predict=max_iter_predict
                    )
                    gp.fit(X_train_selected, y_train)
                    
                    # Predict
                    preds = gp.predict(X_val_selected)
                    scores.append(accuracy_score(y_val, preds))
                    
                except Exception as e:
                    print(f"Error in Gaussian Process training: {e}")
                    return 0.0  # Return worst score for failed trials
                    
            return 1.0 - np.mean(scores)  # minimize error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        print(f"Best Gaussian Process parameters: {self.best_params}")
        
        # Build the best kernel
        kernel_type = self.best_params['kernel_type']
        length_scale = self.best_params['length_scale']
        
        if kernel_type == 'rbf':
            best_kernel = C(1.0) * RBF(length_scale=length_scale)
        elif kernel_type == 'matern':
            nu = self.best_params['nu']
            best_kernel = C(1.0) * Matern(length_scale=length_scale, nu=nu)
        elif kernel_type == 'rbf_white':
            noise_level = self.best_params['noise_level']
            best_kernel = C(1.0) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
        else:  # matern_white
            nu = self.best_params['nu']
            noise_level = self.best_params['noise_level']
            best_kernel = C(1.0) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(noise_level=noise_level)
        
        # Train the final model with best parameters
        self.model = GaussianProcessClassifier(
            kernel=best_kernel,
            max_iter_predict=self.best_params['max_iter_predict']
        )
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Feature selection if enabled
        if self.best_params.get('use_feature_selection', False):
            self.feature_selector = SelectKBest(f_classif, k=self.best_params['k_features'])
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
        else:
            self.feature_selector = None
            X_selected = X_scaled
        
        # Fit the model
        self.model.fit(X_selected, y)
        self.fitted = True
        
        # Evaluate on training data
        preds = self.model.predict(X_selected)
        print("Classification Report:\n", classification_report(y, preds))

        # Feature importance for GP is not directly available, but we can use 
        # the variance of predictions when features are permuted
        try:
            if self.feature_selector is not None:
                feature_names = X_clean.columns[self.feature_selector.get_support()]
            else:
                feature_names = X_clean.columns
            
            print(f"\nUsing {len(feature_names)} features for Gaussian Process")
            print("Feature importance cannot be directly computed for Gaussian Process")
            print("but model provides prediction uncertainties")
            
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
        
        return int(pred), 10  # signal, amount
